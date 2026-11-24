"""Deterministic Round-Turn Demo Trade Reconstructor & Lifecycle Synchronizer for ATLAS AI Phase 8.1."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from atlas_ai.execution.engine import MT5DemoExecutionEngine

from atlas_ai.analytics.config import PostTradeAnalyticsConfig
from atlas_ai.analytics.exceptions import TradeReconstructionError
from atlas_ai.analytics.journal import (
    PersistentTradeJournal,
    compute_initial_risk_amount,
)
from atlas_ai.analytics.models import (
    CompletedTradeRecord,
    ContextAttributionStatus,
    EntryContextSnapshot,
    PartialExitProgressRecord,
    TradeReconstructionReport,
    TradeSource,
)
from atlas_ai.backtest.models import BacktestTradeRecord, ExitReason
from atlas_ai.core.logger import get_logger
from atlas_ai.domain.market import InstrumentSpec
from atlas_ai.domain.signals import SignalDirection
from atlas_ai.execution.account_state import deduplicate_deals
from atlas_ai.execution.connection import MT5ConnectionAdapter
from atlas_ai.execution.exceptions import AccountStateReconstructionError
from atlas_ai.execution.ledger import (
    EPOCH_UTC,
    PersistentExecutionLedger,
    extract_comment_identifiers,
)
from atlas_ai.execution.models import (
    ExecutionLedgerEntry,
    MT5DealEntry,
    MT5DealRecord,
    MT5DealType,
    MT5PositionRecord,
    OrderLedgerStatus,
    ReconciliationStatus,
)
from atlas_ai.strategies.models import SignalLifecycleState
from atlas_ai.strategies.signal_lifecycle import SignalLifecycleManager

logger = get_logger("analytics.trade_reconstructor")

QUANTIZE_MONEY = Decimal("0.01")
QUANTIZE_RATIO = Decimal("0.0001")
QUANTIZE_PRICE = Decimal("0.00001")


class DemoTradeReconstructor:
    """Reconstructs closed and partially closed MT5 DEMO positions from broker deals,
    enforces strict anti-double-counting commission/fee attribution, fails closed on
    `DEAL_ENTRY_INOUT` / `DEAL_ENTRY_OUT_BY` or volume mismatches, and synchronizes
    `PersistentExecutionLedger`, `SignalLifecycleManager`, and `PersistentTradeJournal`.

    Read-Only Broker Guarantee:
    - Queries only `positions_get()` and `history_deals_get()`.
    - Never invokes `order_send()` or modifies open positions on the broker.
    """

    def __init__(
        self,
        *,
        connection: MT5ConnectionAdapter,
        ledger: PersistentExecutionLedger,
        journal: PersistentTradeJournal,
        lifecycle_manager: SignalLifecycleManager | None = None,
        config: PostTradeAnalyticsConfig | None = None,
        execution_engine: "MT5DemoExecutionEngine | None" = None,
        require_staged_context: bool = False,
    ) -> None:
        self._connection = connection
        self._ledger = ledger
        self._journal = journal
        self._lifecycle_manager = lifecycle_manager
        self._config = config or journal.config
        self._execution_engine = execution_engine
        if execution_engine is not None:
            self._journal.bind_execution_engine(
                execution_engine,
                require_staged_context=require_staged_context,
            )

    @property
    def connection(self) -> MT5ConnectionAdapter:
        return self._connection

    @property
    def ledger(self) -> PersistentExecutionLedger:
        return self._ledger

    @property
    def journal(self) -> PersistentTradeJournal:
        return self._journal

    @property
    def lifecycle_manager(self) -> SignalLifecycleManager | None:
        return self._lifecycle_manager

    @property
    def config(self) -> PostTradeAnalyticsConfig:
        return self._config

    @staticmethod
    def convert_backtest_trade(
        bt_trade: BacktestTradeRecord,
        *,
        symbol_point: Decimal = Decimal("0.01"),
        entry_context: EntryContextSnapshot | None = None,
    ) -> CompletedTradeRecord:
        """Convert a Phase 4 `BacktestTradeRecord` into a canonical Phase 8 `CompletedTradeRecord`."""
        pt = symbol_point if symbol_point > Decimal("0") else Decimal("0.01")
        if bt_trade.direction == SignalDirection.BUY:
            entry_slip = (
                (bt_trade.entry_fill_price - bt_trade.requested_entry_price) / pt
            ).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)
            exit_slip = (
                (bt_trade.requested_exit_price - bt_trade.exit_fill_price) / pt
            ).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)
        else:
            entry_slip = (
                (bt_trade.requested_entry_price - bt_trade.entry_fill_price) / pt
            ).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)
            exit_slip = (
                (bt_trade.exit_fill_price - bt_trade.requested_exit_price) / pt
            ).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)

        entry_ts = (
            bt_trade.entry_timestamp.astimezone(timezone.utc)
            if bt_trade.entry_timestamp.tzinfo is not None
            else bt_trade.entry_timestamp.replace(tzinfo=timezone.utc)
        )
        exit_ts = (
            bt_trade.exit_timestamp.astimezone(timezone.utc)
            if bt_trade.exit_timestamp.tzinfo is not None
            else bt_trade.exit_timestamp.replace(tzinfo=timezone.utc)
        )

        return CompletedTradeRecord(
            trade_id=bt_trade.trade_id,
            source=TradeSource.BACKTEST,
            client_order_id=bt_trade.position_id,
            signal_id=bt_trade.signal_id,
            position_ticket=None,
            symbol=bt_trade.symbol,
            timeframe=bt_trade.timeframe,
            direction=bt_trade.direction,
            strategy_id=bt_trade.strategy_id,
            entry_timestamp_utc=entry_ts,
            exit_timestamp_utc=exit_ts,
            requested_entry_price=bt_trade.requested_entry_price,
            entry_fill_price=bt_trade.entry_fill_price,
            requested_exit_price=bt_trade.requested_exit_price,
            exit_fill_price=bt_trade.exit_fill_price,
            stop_loss=bt_trade.stop_loss,
            take_profit=bt_trade.take_profit,
            volume=bt_trade.lots,
            entry_slippage_points=entry_slip,
            exit_slippage_points=exit_slip,
            initial_risk_amount=bt_trade.initial_risk_amount.quantize(
                QUANTIZE_MONEY, rounding=ROUND_HALF_UP
            ),
            gross_pnl=bt_trade.gross_pnl.quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP),
            commission=bt_trade.commission.quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP),
            fee=Decimal("0.00"),
            swap=Decimal("0.00"),
            net_pnl=bt_trade.net_pnl.quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP),
            realized_r_multiple=bt_trade.realized_r_multiple.quantize(
                QUANTIZE_RATIO, rounding=ROUND_HALF_UP
            ),
            mfe_usd=bt_trade.mfe_usd.quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP),
            mae_usd=bt_trade.mae_usd.quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP),
            exit_reason=bt_trade.exit_reason,
            entry_deal_tickets=(),
            exit_deal_tickets=(),
            commission_deal_tickets=(),
            partial_exit_count=1,
            context_attribution_status=(
                ContextAttributionStatus.VERIFIED_ENTRY_CONTEXT
                if entry_context is not None
                else ContextAttributionStatus.UNATTRIBUTED_MISSING_CONTEXT
            ),
            entry_context=entry_context,
        )

    def sync_closed_demo_trades(
        self,
        *,
        reference_now_utc: datetime | None = None,
    ) -> TradeReconstructionReport:
        """Inspect active, partially filled, and filled execution ledger entries against
        live broker positions (`positions_get(None)`) and historical deals (`history_deals_get()`).
        """
        account = self._connection.ensure_connected()
        now_utc = (
            reference_now_utc.astimezone(timezone.utc)
            if (reference_now_utc is not None and reference_now_utc.tzinfo is not None)
            else (
                reference_now_utc.replace(tzinfo=timezone.utc)
                if reference_now_utc is not None
                else datetime.now(timezone.utc)
            )
        )

        self._journal.load_or_initialize(
            account_login=account.login,
            server=account.server,
        )

        open_positions = self._connection.backend.positions_get(None)
        future_horizon_utc = max(
            now_utc + timedelta(days=3650),
            datetime(2100, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        )
        raw_deals = self._connection.backend.history_deals_get(EPOCH_UTC, future_horizon_utc)

        if open_positions is None or raw_deals is None:
            raise TradeReconstructionError(
                "Post-trade reconstruction failed closed: positions_get(None) or "
                "history_deals_get() returned None."
            )

        for d in raw_deals:
            if d.time_utc > now_utc:
                raise TradeReconstructionError(
                    f"Future-dated deal detected in broker history during post-trade reconstruction "
                    f"(ticket={d.ticket}, time_utc={d.time_utc.isoformat()} > now_utc={now_utc.isoformat()})."
                )

        try:
            deduped_deals, _ = deduplicate_deals(raw_deals)
        except AccountStateReconstructionError as exc:
            raise TradeReconstructionError(str(exc)) from exc

        # 1. Fail-closed check for DEAL_ENTRY_INOUT (2) and DEAL_ENTRY_OUT_BY (3) across tracked positions or ATLAS magic
        tracked_entries = [
            e
            for e in self._ledger.list_entries()
            if e.status
            in (
                OrderLedgerStatus.ACTIVE,
                OrderLedgerStatus.PARTIALLY_FILLED,
                OrderLedgerStatus.FILLED,
            )
        ]
        tracked_pos_ids = {
            e.position_ticket
            for e in tracked_entries
            if e.position_ticket is not None and e.position_ticket > 0
        }
        atlas_magic = self._ledger.safety_config.magic_number

        for d in deduped_deals:
            if d.deal_type in (int(MT5DealType.DEAL_TYPE_BUY), int(MT5DealType.DEAL_TYPE_SELL)):
                if d.entry in (
                    int(MT5DealEntry.DEAL_ENTRY_INOUT),
                    int(MT5DealEntry.DEAL_ENTRY_OUT_BY),
                ) or d.entry not in (
                    int(MT5DealEntry.DEAL_ENTRY_IN),
                    int(MT5DealEntry.DEAL_ENTRY_OUT),
                ):
                    if d.magic == atlas_magic or (d.position_id > 0 and d.position_id in tracked_pos_ids):
                        entry_name = (
                            MT5DealEntry(d.entry).name
                            if d.entry in (0, 1, 2, 3)
                            else f"UNKNOWN_ENTRY_{d.entry}"
                        )
                        raise TradeReconstructionError(
                            f"Forbidden deal entry type {entry_name} (entry={d.entry}) detected on deal "
                            f"ticket={d.ticket} (order={d.order}, position_id={d.position_id}, magic={d.magic}). "
                            "Position reversals (DEAL_ENTRY_INOUT) and close-by (DEAL_ENTRY_OUT_BY) are strictly "
                            "blocked; failing closed."
                        )

        open_pos_by_ticket: dict[int, MT5PositionRecord] = {
            p.ticket: p for p in open_positions
        }

        newly_completed: list[CompletedTradeRecord] = []
        partial_progress_list: list[PartialExitProgressRecord] = []
        still_open_ids: list[str] = []
        already_journaled_ids: list[str] = []

        for entry in sorted(tracked_entries, key=lambda item: (item.created_at_utc, item.client_order_id)):
            cid = entry.client_order_id

            # If already finalized in PersistentTradeJournal, record as already_journaled (idempotent)
            existing_completed = self._journal.get_completed_trade_by_client_order_id(cid)
            if existing_completed is not None:
                already_journaled_ids.append(cid)
                continue

            pos_id = self._resolve_position_id_for_entry(entry, deduped_deals, open_positions)
            if pos_id is None or pos_id <= 0:
                raise TradeReconstructionError(
                    f"Cannot resolve broker position_id for executed ledger entry '{cid}' "
                    f"(status={entry.status.value})."
                )

            in_deals, out_deals, comm_deals = self._collect_and_verify_position_deals(
                entry=entry,
                position_id=pos_id,
                all_deals=deduped_deals,
            )

            if not in_deals:
                raise TradeReconstructionError(
                    f"No DEAL_ENTRY_IN execution deals found in broker history for position_id={pos_id} "
                    f"(client_order_id='{cid}'); failing closed."
                )

            v_in = sum((d.volume for d in in_deals), Decimal("0"))
            v_out = sum((d.volume for d in out_deals), Decimal("0"))
            open_pos = open_pos_by_ticket.get(pos_id)

            if v_in <= Decimal("0"):
                raise TradeReconstructionError(
                    f"Total entry deal volume for position_id={pos_id} ('{cid}') is {v_in} <= 0."
                )

            if v_out > v_in:
                raise TradeReconstructionError(
                    f"Exit deal volume ({v_out}) exceeds entry deal volume ({v_in}) for "
                    f"position_id={pos_id} ('{cid}'); failing closed."
                )

            # Case 1: Position has zero exit deals so far (v_out == 0)
            if v_out == Decimal("0"):
                if open_pos is None:
                    raise TradeReconstructionError(
                        f"Position position_id={pos_id} ('{cid}') is absent from positions_get(None), "
                        "but zero DEAL_ENTRY_OUT deals exist in broker history; failing closed."
                    )
                if open_pos.volume != v_in:
                    raise TradeReconstructionError(
                        f"Open position volume ({open_pos.volume}) on broker does not match entry deal "
                        f"volume ({v_in}) for position_id={pos_id} ('{cid}') with zero exit deals."
                    )
                still_open_ids.append(cid)
                continue

            # Case 2: Partial position closure (0 < v_out < v_in)
            if v_out < v_in:
                expected_open_vol = v_in - v_out
                if open_pos is None:
                    raise TradeReconstructionError(
                        f"Partially exited position_id={pos_id} ('{cid}', v_in={v_in}, v_out={v_out}) "
                        f"is missing from positions_get(None) while remaining volume ({expected_open_vol}) > 0."
                    )
                if open_pos.volume != expected_open_vol:
                    raise TradeReconstructionError(
                        f"Partially exited position_id={pos_id} ('{cid}') has broker open volume "
                        f"{open_pos.volume} != expected remaining volume {expected_open_vol} "
                        f"(v_in={v_in}, v_out={v_out})."
                    )

                partial_rec = self._build_partial_exit_record(
                    entry=entry,
                    position_id=pos_id,
                    in_deals=in_deals,
                    out_deals=out_deals,
                    comm_deals=comm_deals,
                    v_in=v_in,
                    v_out=v_out,
                    v_open=expected_open_vol,
                )
                self._journal.record_partial_exit_progress(
                    partial_rec,
                    account_login=account.login,
                    server=account.server,
                    now_utc=now_utc,
                )
                partial_progress_list.append(partial_rec)
                still_open_ids.append(cid)
                continue

            # Case 3: Full position closure (v_out == v_in)
            if open_pos is not None:
                raise TradeReconstructionError(
                    f"Position position_id={pos_id} ('{cid}') has v_out ({v_out}) == v_in ({v_in}), "
                    f"but still appears as an open position (volume={open_pos.volume}) in positions_get(None)."
                )

            completed_trade = self._build_completed_trade_record(
                entry=entry,
                position_id=pos_id,
                in_deals=in_deals,
                out_deals=out_deals,
                comm_deals=comm_deals,
                v_in=v_in,
            )

            persisted_trade = self._journal.record_completed_trade(
                completed_trade,
                account_login=account.login,
                server=account.server,
                now_utc=now_utc,
            )

            # Transition execution ledger entry from ACTIVE / PARTIALLY_FILLED -> FILLED
            if entry.status in (OrderLedgerStatus.ACTIVE, OrderLedgerStatus.PARTIALLY_FILLED):
                self._ledger.transition_entry(
                    cid,
                    OrderLedgerStatus.FILLED,
                    account_login=account.login,
                    server=account.server,
                    now_utc=now_utc,
                    executed_volume=v_in,
                    remaining_volume=Decimal("0"),
                    fill_price=persisted_trade.entry_fill_price,
                    position_ticket=pos_id,
                    reconciliation_status=ReconciliationStatus.RECONCILED_EXECUTED,
                    reason=(
                        f"Position position_id={pos_id} fully closed on broker "
                        f"({persisted_trade.exit_reason.value}): net_pnl={persisted_trade.net_pnl}, "
                        f"R={persisted_trade.realized_r_multiple}."
                    ),
                )

            # Transition SignalLifecycleManager from ACTIVE -> TARGET_HIT / STOP_LOSS_HIT / CANCELLED
            self._transition_signal_on_full_close(
                signal_id=entry.signal_id,
                completed_trade=persisted_trade,
                now_utc=persisted_trade.exit_timestamp_utc,
            )

            newly_completed.append(persisted_trade)

        return TradeReconstructionReport(
            newly_completed_trades=tuple(newly_completed),
            partial_exit_records=tuple(partial_progress_list),
            still_open_client_order_ids=tuple(still_open_ids),
            already_journaled_client_order_ids=tuple(already_journaled_ids),
            journal_generation=self._journal.last_generation,
        )

    @staticmethod
    def _resolve_position_id_for_entry(
        entry: ExecutionLedgerEntry,
        deals: Sequence[MT5DealRecord],
        open_positions: Sequence[MT5PositionRecord],
    ) -> int | None:
        if entry.position_ticket is not None and entry.position_ticket > 0:
            return entry.position_ticket

        for d in deals:
            if (
                d.magic == entry.magic
                and d.symbol.strip().upper() == entry.symbol
                and d.entry == int(MT5DealEntry.DEAL_ENTRY_IN)
                and d.position_id > 0
            ):
                tokens = extract_comment_identifiers(d.comment)
                if (
                    (entry.deal_ticket is not None and d.ticket == entry.deal_ticket)
                    or (entry.order_ticket is not None and d.order == entry.order_ticket)
                    or entry.client_order_id in tokens
                    or entry.signal_id in tokens
                ):
                    return d.position_id

        for p in open_positions:
            if p.magic == entry.magic and p.symbol.strip().upper() == entry.symbol:
                tokens = extract_comment_identifiers(p.comment)
                if entry.client_order_id in tokens or entry.signal_id in tokens:
                    return p.ticket

        return None

    def _collect_and_verify_position_deals(
        self,
        *,
        entry: ExecutionLedgerEntry,
        position_id: int,
        all_deals: Sequence[MT5DealRecord],
    ) -> tuple[list[MT5DealRecord], list[MT5DealRecord], list[MT5DealRecord]]:
        """Collect and strictly validate `DEAL_ENTRY_IN`, `DEAL_ENTRY_OUT`, and separate
        `DEAL_TYPE_COMMISSION` deals for `position_id`.
        """
        expected_in_type = (
            int(MT5DealType.DEAL_TYPE_BUY)
            if entry.direction == SignalDirection.BUY
            else int(MT5DealType.DEAL_TYPE_SELL)
        )
        expected_out_type = (
            int(MT5DealType.DEAL_TYPE_SELL)
            if entry.direction == SignalDirection.BUY
            else int(MT5DealType.DEAL_TYPE_BUY)
        )

        in_deals: list[MT5DealRecord] = []
        out_deals: list[MT5DealRecord] = []
        order_tickets: set[int] = set()

        for d in all_deals:
            if d.position_id != position_id:
                continue

            if self._journal.is_deal_ticket_consumed(d.ticket):
                raise TradeReconstructionError(
                    f"Deal ticket {d.ticket} (position_id={position_id}) is already consumed by another "
                    "completed trade in PersistentTradeJournal."
                )

            if d.deal_type in (int(MT5DealType.DEAL_TYPE_BUY), int(MT5DealType.DEAL_TYPE_SELL)):
                if d.symbol.strip().upper() != entry.symbol:
                    raise TradeReconstructionError(
                        f"Deal ticket={d.ticket} on position_id={position_id} has symbol '{d.symbol}' "
                        f"!= expected '{entry.symbol}'."
                    )
                if d.volume <= Decimal("0") or d.price <= Decimal("0"):
                    raise TradeReconstructionError(
                        f"Execution deal ticket={d.ticket} on position_id={position_id} has invalid "
                        f"volume={d.volume} or price={d.price}."
                    )
                if d.commission > Decimal("0") or d.fee > Decimal("0"):
                    raise TradeReconstructionError(
                        f"Execution deal ticket={d.ticket} has positive commission ({d.commission}) or "
                        f"positive fee ({d.fee}); failing closed."
                    )

                if d.entry == int(MT5DealEntry.DEAL_ENTRY_IN):
                    if d.deal_type != expected_in_type:
                        raise TradeReconstructionError(
                            f"Entry deal ticket={d.ticket} on position_id={position_id} has deal_type={d.deal_type}, "
                            f"which opposes entry direction {entry.direction.value}."
                        )
                    in_deals.append(d)
                    if d.order > 0:
                        order_tickets.add(d.order)
                elif d.entry == int(MT5DealEntry.DEAL_ENTRY_OUT):
                    if d.deal_type != expected_out_type:
                        raise TradeReconstructionError(
                            f"Exit deal ticket={d.ticket} on position_id={position_id} has deal_type={d.deal_type}, "
                            f"which does not close a {entry.direction.value} position."
                        )
                    out_deals.append(d)
                    if d.order > 0:
                        order_tickets.add(d.order)
                else:
                    raise TradeReconstructionError(
                        f"Unsupported deal entry={d.entry} on deal ticket={d.ticket} (position_id={position_id})."
                    )

        # Track inline charges by order and across the position
        orders_with_inline_charge: set[int] = set()
        position_has_inline_charge = False
        position_has_orderless_inline_charge = False

        for exec_d in (*in_deals, *out_deals):
            if exec_d.commission != Decimal("0") or exec_d.fee != Decimal("0"):
                position_has_inline_charge = True
                if exec_d.order > 0:
                    orders_with_inline_charge.add(exec_d.order)
                else:
                    position_has_orderless_inline_charge = True

        # Collect and validate separate DEAL_TYPE_COMMISSION (7) deals linked by position_id or order_tickets
        comm_deals: list[MT5DealRecord] = []
        orders_with_separate_comm: set[int] = set()
        has_orderless_separate_comm = False

        for d in all_deals:
            if d.deal_type != int(MT5DealType.DEAL_TYPE_COMMISSION):
                continue

            linked_by_pos = d.position_id == position_id
            linked_by_order = d.order > 0 and d.order in order_tickets
            if not (linked_by_pos or linked_by_order):
                continue

            if self._journal.is_deal_ticket_consumed(d.ticket):
                raise TradeReconstructionError(
                    f"Separate commission deal ticket {d.ticket} is already consumed by another trade."
                )

            if d.position_id > 0 and d.position_id != position_id:
                raise TradeReconstructionError(
                    f"Separate DEAL_TYPE_COMMISSION (ticket={d.ticket}) has inconsistent "
                    f"order={d.order} (linked to position_id={position_id}) and position_id={d.position_id}."
                )
            if d.symbol.strip() != "" and d.symbol.strip().upper() != entry.symbol:
                raise TradeReconstructionError(
                    f"Separate DEAL_TYPE_COMMISSION (ticket={d.ticket}, symbol='{d.symbol}') "
                    f"mismatches position symbol '{entry.symbol}'."
                )
            if d.net_cash_impact > Decimal("0"):
                raise TradeReconstructionError(
                    f"Positive separate DEAL_TYPE_COMMISSION (ticket={d.ticket}, net={d.net_cash_impact}) "
                    "is forbidden; failing closed."
                )

            # Anti-double-counting checks against inline commission or inline fee
            if d.order > 0:
                if d.order in orders_with_inline_charge or position_has_orderless_inline_charge:
                    raise TradeReconstructionError(
                        f"Commission double-counting detected on position_id={position_id}: separate "
                        f"DEAL_TYPE_COMMISSION (ticket={d.ticket}, order={d.order}) conflicts with "
                        "non-zero inline commission or fee."
                    )
                if d.order in orders_with_separate_comm:
                    raise TradeReconstructionError(
                        f"Duplicate separate DEAL_TYPE_COMMISSION (ticket={d.ticket}) for order={d.order} "
                        f"on position_id={position_id}."
                    )
                if has_orderless_separate_comm:
                    raise TradeReconstructionError(
                        f"Mixed order-linked and position-only separate DEAL_TYPE_COMMISSION "
                        f"(ticket={d.ticket}) on position_id={position_id}."
                    )
                orders_with_separate_comm.add(d.order)
            else:
                if position_has_inline_charge:
                    raise TradeReconstructionError(
                        f"Commission double-counting detected on position_id={position_id}: orderless "
                        f"DEAL_TYPE_COMMISSION (ticket={d.ticket}) conflicts with non-zero inline commission or fee."
                    )
                if has_orderless_separate_comm or bool(orders_with_separate_comm):
                    raise TradeReconstructionError(
                        f"Duplicate or mixed order/position separate DEAL_TYPE_COMMISSION "
                        f"(ticket={d.ticket}) on position_id={position_id}."
                    )
                has_orderless_separate_comm = True

            comm_deals.append(d)

        return in_deals, out_deals, comm_deals

    @staticmethod
    def _aggregate_cash_flows(
        in_deals: Sequence[MT5DealRecord],
        out_deals: Sequence[MT5DealRecord],
        comm_deals: Sequence[MT5DealRecord],
    ) -> tuple[Decimal, Decimal, Decimal, Decimal, Decimal]:
        """Return `(gross_pnl, commission_cost, fee_cost, swap_signed, net_pnl)` quantized to 0.01.

        Guarantees `net_pnl == gross_pnl - commission_cost - fee_cost + swap_signed`
        without ever subtracting commissions or fees twice.
        """
        exec_deals = (*in_deals, *out_deals)

        raw_gross = sum((d.profit for d in exec_deals), Decimal("0"))
        raw_inline_comm = sum((d.commission for d in exec_deals), Decimal("0"))
        raw_sep_comm = sum((c.net_cash_impact for c in comm_deals), Decimal("0"))
        raw_fee = sum((d.fee for d in exec_deals), Decimal("0"))
        raw_swap = sum((d.swap for d in exec_deals), Decimal("0"))

        gross_pnl = raw_gross.quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
        commission_cost = (-(raw_inline_comm + raw_sep_comm)).quantize(
            QUANTIZE_MONEY, rounding=ROUND_HALF_UP
        )
        fee_cost = (-raw_fee).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
        swap_signed = raw_swap.quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
        net_pnl = gross_pnl - commission_cost - fee_cost + swap_signed

        return gross_pnl, commission_cost, fee_cost, swap_signed, net_pnl

    def _build_partial_exit_record(
        self,
        *,
        entry: ExecutionLedgerEntry,
        position_id: int,
        in_deals: Sequence[MT5DealRecord],
        out_deals: Sequence[MT5DealRecord],
        comm_deals: Sequence[MT5DealRecord],
        v_in: Decimal,
        v_out: Decimal,
        v_open: Decimal,
    ) -> PartialExitProgressRecord:
        avg_in_px = (
            sum((d.price * d.volume for d in in_deals), Decimal("0")) / v_in
        ).quantize(QUANTIZE_PRICE, rounding=ROUND_HALF_UP)
        avg_out_px = (
            sum((d.price * d.volume for d in out_deals), Decimal("0")) / v_out
        ).quantize(QUANTIZE_PRICE, rounding=ROUND_HALF_UP)

        gross_pnl, comm_cost, fee_cost, swap_signed, net_pnl = self._aggregate_cash_flows(
            in_deals,
            out_deals,
            comm_deals,
        )

        return PartialExitProgressRecord(
            client_order_id=entry.client_order_id,
            signal_id=entry.signal_id,
            position_ticket=position_id,
            symbol=entry.symbol,
            direction=entry.direction,
            initial_entry_volume=v_in,
            closed_volume_so_far=v_out,
            remaining_open_volume=v_open,
            average_entry_price=avg_in_px,
            average_exit_price_so_far=avg_out_px,
            realized_gross_pnl_so_far=gross_pnl,
            realized_commission_so_far=comm_cost,
            realized_fee_so_far=fee_cost,
            realized_swap_so_far=swap_signed,
            realized_net_pnl_so_far=net_pnl,
            entry_deal_tickets=tuple(d.ticket for d in in_deals),
            exit_deal_tickets_so_far=tuple(d.ticket for d in out_deals),
            commission_deal_tickets_so_far=tuple(c.ticket for c in comm_deals),
            last_exit_time_utc=out_deals[-1].time_utc,
        )

    def _build_completed_trade_record(
        self,
        *,
        entry: ExecutionLedgerEntry,
        position_id: int,
        in_deals: Sequence[MT5DealRecord],
        out_deals: Sequence[MT5DealRecord],
        comm_deals: Sequence[MT5DealRecord],
        v_in: Decimal,
    ) -> CompletedTradeRecord:
        avg_in_px = (
            sum((d.price * d.volume for d in in_deals), Decimal("0")) / v_in
        ).quantize(QUANTIZE_PRICE, rounding=ROUND_HALF_UP)
        avg_out_px = (
            sum((d.price * d.volume for d in out_deals), Decimal("0")) / v_in
        ).quantize(QUANTIZE_PRICE, rounding=ROUND_HALF_UP)

        gross_pnl, comm_cost, fee_cost, swap_signed, net_pnl = self._aggregate_cash_flows(
            in_deals,
            out_deals,
            comm_deals,
        )

        # Look up symbol point and InstrumentSpec if available on connection
        sym_point = Decimal("0.01")
        inst_spec: InstrumentSpec | None = None
        try:
            sym_info = self._connection.get_symbol_info(entry.symbol)
            sym_point = sym_info.point
            inst_spec = self._connection.get_instrument_spec(entry.symbol)
        except Exception:
            pass

        exit_reason, requested_exit_px = self._determine_exit_reason_and_target(
            entry=entry,
            out_deals=out_deals,
            avg_exit_price=avg_out_px,
            net_pnl=net_pnl,
            symbol_point=sym_point,
        )

        if entry.direction == SignalDirection.BUY:
            entry_slip_pts = (
                (avg_in_px - entry.requested_price) / sym_point
            ).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)
            exit_slip_pts = (
                (requested_exit_px - avg_out_px) / sym_point
            ).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)
        else:
            entry_slip_pts = (
                (entry.requested_price - avg_in_px) / sym_point
            ).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)
            exit_slip_pts = (
                (avg_out_px - requested_exit_px) / sym_point
            ).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)

        stored_ctx = self._journal.get_entry_context(
            client_order_id=entry.client_order_id,
            signal_id=entry.signal_id,
        )
        if stored_ctx is not None:
            init_risk = stored_ctx.initial_risk_amount.quantize(
                QUANTIZE_MONEY, rounding=ROUND_HALF_UP
            )
            strategy_id = stored_ctx.strategy_id
            attr_status = ContextAttributionStatus.VERIFIED_ENTRY_CONTEXT
        else:
            init_risk = compute_initial_risk_amount(
                entry_price=avg_in_px,
                stop_loss=entry.stop_loss,
                volume=v_in,
                instrument_spec=inst_spec,
            )
            strategy_id = "mt5_demo_unattributed"
            attr_status = ContextAttributionStatus.UNATTRIBUTED_MISSING_CONTEXT

        r_mult = (net_pnl / init_risk).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)

        entry_ts = in_deals[0].time_utc
        exit_ts = max(out_deals[-1].time_utc, entry_ts)

        return CompletedTradeRecord(
            trade_id=f"demo-tr-{position_id}",
            source=TradeSource.MT5_DEMO,
            client_order_id=entry.client_order_id,
            signal_id=entry.signal_id,
            position_ticket=position_id,
            symbol=entry.symbol,
            timeframe=entry.timeframe,
            direction=entry.direction,
            strategy_id=strategy_id,
            entry_timestamp_utc=entry_ts,
            exit_timestamp_utc=exit_ts,
            requested_entry_price=entry.requested_price,
            entry_fill_price=avg_in_px,
            requested_exit_price=requested_exit_px,
            exit_fill_price=avg_out_px,
            stop_loss=entry.stop_loss,
            take_profit=entry.take_profit,
            volume=v_in,
            entry_slippage_points=entry_slip_pts,
            exit_slippage_points=exit_slip_pts,
            initial_risk_amount=init_risk,
            gross_pnl=gross_pnl,
            commission=comm_cost,
            fee=fee_cost,
            swap=swap_signed,
            net_pnl=net_pnl,
            realized_r_multiple=r_mult,
            mfe_usd=max(Decimal("0.00"), gross_pnl),
            mae_usd=max(Decimal("0.00"), -gross_pnl),
            exit_reason=exit_reason,
            entry_deal_tickets=tuple(d.ticket for d in in_deals),
            exit_deal_tickets=tuple(d.ticket for d in out_deals),
            commission_deal_tickets=tuple(c.ticket for c in comm_deals),
            partial_exit_count=len(out_deals),
            context_attribution_status=attr_status,
            entry_context=stored_ctx,
        )

    def _determine_exit_reason_and_target(
        self,
        *,
        entry: ExecutionLedgerEntry,
        out_deals: Sequence[MT5DealRecord],
        avg_exit_price: Decimal,
        net_pnl: Decimal,
        symbol_point: Decimal,
    ) -> tuple[ExitReason, Decimal]:
        combined_comments = " ".join(d.comment.lower() for d in out_deals)
        if "[sl" in combined_comments or "stop loss" in combined_comments:
            return ExitReason.STOP_LOSS, entry.stop_loss
        if "[tp" in combined_comments or "take profit" in combined_comments:
            return ExitReason.TAKE_PROFIT, entry.take_profit

        tol_price = self._config.exit_level_tolerance_points * symbol_point
        dist_to_sl = abs(avg_exit_price - entry.stop_loss)
        dist_to_tp = abs(avg_exit_price - entry.take_profit)

        if dist_to_sl <= tol_price and dist_to_sl <= dist_to_tp:
            return ExitReason.STOP_LOSS, entry.stop_loss
        if dist_to_tp <= tol_price and dist_to_tp < dist_to_sl:
            return ExitReason.TAKE_PROFIT, entry.take_profit

        if entry.direction == SignalDirection.BUY:
            if avg_exit_price <= entry.stop_loss:
                return ExitReason.STOP_LOSS, entry.stop_loss
            if avg_exit_price >= entry.take_profit:
                return ExitReason.TAKE_PROFIT, entry.take_profit
        else:
            if avg_exit_price >= entry.stop_loss:
                return ExitReason.STOP_LOSS, entry.stop_loss
            if avg_exit_price <= entry.take_profit:
                return ExitReason.TAKE_PROFIT, entry.take_profit

        return ExitReason.MANUAL_CANCEL, avg_exit_price

    def _transition_signal_on_full_close(
        self,
        *,
        signal_id: str,
        completed_trade: CompletedTradeRecord,
        now_utc: datetime,
    ) -> None:
        if self._lifecycle_manager is None:
            return
        try:
            rec = self._lifecycle_manager.get_record(signal_id)
        except Exception:
            return
        if rec.state != SignalLifecycleState.ACTIVE:
            return

        if completed_trade.exit_reason == ExitReason.TAKE_PROFIT:
            target_state = SignalLifecycleState.TARGET_HIT
        elif completed_trade.exit_reason == ExitReason.STOP_LOSS:
            target_state = SignalLifecycleState.STOP_LOSS_HIT
        else:
            target_state = SignalLifecycleState.CANCELLED

        self._lifecycle_manager.transition(
            signal_id,
            target_state,
            now_utc,
            (
                f"Position ticket={completed_trade.position_ticket} fully closed on broker "
                f"({completed_trade.exit_reason.value}): exit_price={completed_trade.exit_fill_price}, "
                f"net_pnl={completed_trade.net_pnl}, R={completed_trade.realized_r_multiple}."
            ),
        )
