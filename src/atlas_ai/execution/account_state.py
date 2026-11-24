"""Persistent Account Risk Tracker, Trusted Balance Anchor verification, 5 PM America/New_York
DST-aware rollover, and complete official MQL5 (0..17) deal & commission accounting.
"""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Sequence
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from atlas_ai.core.logger import get_logger
from atlas_ai.domain.risk import AccountState
from atlas_ai.execution.config import ExecutionSafetyConfig, TrustedBalanceAnchorConfig
from atlas_ai.execution.connection import MT5ConnectionAdapter
from atlas_ai.execution.exceptions import (
    AccountStateReconstructionError,
    PersistenceAccountMismatchError,
    PersistenceCorruptionError,
)
from atlas_ai.execution.models import (
    AccountRiskPersistenceSnapshot,
    MT5DealRecord,
    MT5DealType,
)
from atlas_ai.execution.persistence import CrashSafeJsonPersistence
from atlas_ai.strategies.sessions import compute_trading_day_label

logger = get_logger("execution.account_state")

EPOCH_UTC = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


class TrustedAnchorSource(StrEnum):
    """Origin of the verified TrustedBalanceAnchor used to prove deal history completeness."""

    PERSISTED_ROLLOVER_ANCHOR = "PERSISTED_ROLLOVER_ANCHOR"
    CONFIGURED_BALANCE_ANCHOR = "CONFIGURED_BALANCE_ANCHOR"


class DealAccountingCategory(StrEnum):
    """Conservative risk accounting bucket for a classified MT5 deal record."""

    TRADING_PNL = "TRADING_PNL"
    NON_TRADING_CASH_FLOW = "NON_TRADING_CASH_FLOW"


class ClassifiedDealRecord(BaseModel):
    """Immutable audit record for a single classified and deduplicated MT5 deal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticket: int = Field(..., gt=0)
    order: int = Field(default=0, ge=0)
    position_id: int = Field(default=0, ge=0)
    time_utc: datetime
    deal_type: MT5DealType
    deal_type_name: str = Field(..., min_length=1)
    symbol: str = Field(default="")
    magic: int = Field(default=0, ge=0)
    net_cash_impact: Decimal
    category: DealAccountingCategory
    linked_to_trade: bool = Field(default=False)
    linked_order_ticket: int | None = Field(default=None)
    linked_position_id: int | None = Field(default=None)
    classification_note: str = Field(..., min_length=1)


class DealClassificationSummary(BaseModel):
    """Aggregated result of classifying and deduplicating a sequence of MT5 deals."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    realized_trading_pnl: Decimal = Field(default=Decimal("0"))
    net_non_trading_cash_flows: Decimal = Field(default=Decimal("0"))
    total_net_cash_impact: Decimal = Field(default=Decimal("0"))
    classified_deals: tuple[ClassifiedDealRecord, ...] = Field(default=())
    deduplicated_deal_count: int = Field(default=0, ge=0)
    duplicates_skipped_count: int = Field(default=0, ge=0)


class AccountRiskReconstructionResult(BaseModel):
    """Detailed audit report and reconstructed Phase 1 AccountState."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    account_state: AccountState
    snapshot: AccountRiskPersistenceSnapshot
    anchor_source: TrustedAnchorSource
    anchor_timestamp_utc: datetime
    anchor_balance: Decimal
    trading_day_label: str
    day_start_utc: datetime
    day_end_utc: datetime
    pre_rollover_net_change: Decimal
    daily_starting_balance: Decimal
    realized_trading_pnl_today: Decimal
    net_non_trading_cash_flows_today: Decimal
    unrealized_floating_pnl: Decimal
    current_daily_pnl: Decimal
    reconstructed_balance: Decimal
    broker_balance: Decimal
    balance_discrepancy: Decimal
    reconciliation_tolerance: Decimal
    peak_balance: Decimal
    rolled_over: bool
    envelope_generation: int = Field(..., ge=1)
    classified_deals_today: tuple[ClassifiedDealRecord, ...] = Field(default=())
    classified_deals_pre_rollover: tuple[ClassifiedDealRecord, ...] = Field(default=())


def compute_trading_day_utc_bounds(
    trading_day: date,
    tz: ZoneInfo,
    rollover_hour: int = 17,
) -> tuple[datetime, datetime]:
    """Compute exact timezone-aware UTC `[day_start_utc, day_end_utc)` boundaries for `trading_day`.

    When `tz = ZoneInfo('America/New_York')` and `rollover_hour = 17` (5 PM NY close):
    - Logical trading day `D` starts at 17:00:00 `America/New_York` on calendar date `D - 1 day`.
    - Logical trading day `D` ends at 17:00:00 `America/New_York` on calendar date `D`.
    - Automatically accounts for EST (UTC-5, 22:00 UTC) and EDT (UTC-4, 21:00 UTC) DST transitions.
    """
    if rollover_hour == 0:
        start_local = datetime(trading_day.year, trading_day.month, trading_day.day, 0, 0, 0, tzinfo=tz)
        next_day = trading_day + timedelta(days=1)
        end_local = datetime(next_day.year, next_day.month, next_day.day, 0, 0, 0, tzinfo=tz)
    else:
        prev_day = trading_day - timedelta(days=1)
        start_local = datetime(
            prev_day.year,
            prev_day.month,
            prev_day.day,
            rollover_hour,
            0,
            0,
            tzinfo=tz,
        )
        end_local = datetime(
            trading_day.year,
            trading_day.month,
            trading_day.day,
            rollover_hour,
            0,
            0,
            tzinfo=tz,
        )
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def resolve_trading_day_window(
    reference_now_utc: datetime,
    *,
    day_boundary_timezone: str = "America/New_York",
    day_rollover_hour: int = 17,
) -> tuple[str, datetime, datetime]:
    """Resolve `(trading_day_label, day_start_utc, day_end_utc)` for `reference_now_utc`."""
    now_utc = (
        reference_now_utc.astimezone(timezone.utc)
        if reference_now_utc.tzinfo is not None
        else reference_now_utc.replace(tzinfo=timezone.utc)
    )
    tz = ZoneInfo(day_boundary_timezone)
    day_date = compute_trading_day_label(now_utc, tz, day_rollover_hour)
    day_start_utc, day_end_utc = compute_trading_day_utc_bounds(day_date, tz, day_rollover_hour)
    return day_date.isoformat(), day_start_utc, day_end_utc


def _deals_have_identical_fields(a: MT5DealRecord, b: MT5DealRecord) -> bool:
    return (
        a.ticket == b.ticket
        and a.order == b.order
        and a.time_utc == b.time_utc
        and a.deal_type == b.deal_type
        and a.entry == b.entry
        and a.magic == b.magic
        and a.position_id == b.position_id
        and a.symbol.strip().upper() == b.symbol.strip().upper()
        and a.volume == b.volume
        and a.price == b.price
        and a.profit == b.profit
        and a.commission == b.commission
        and a.swap == b.swap
        and a.fee == b.fee
    )


def deduplicate_deals(deals: Sequence[MT5DealRecord]) -> tuple[tuple[MT5DealRecord, ...], int]:
    """Deduplicate deals chronologically by unique `ticket`, rejecting conflicting duplicate tickets."""
    deduped_by_ticket: dict[int, MT5DealRecord] = {}
    duplicates_skipped = 0

    for d in sorted(deals, key=lambda item: (item.time_utc, item.ticket)):
        if d.ticket in deduped_by_ticket:
            existing = deduped_by_ticket[d.ticket]
            if not _deals_have_identical_fields(existing, d):
                raise AccountStateReconstructionError(
                    f"Conflicting duplicate deal ticket {d.ticket} encountered in deal history."
                )
            duplicates_skipped += 1
            continue
        deduped_by_ticket[d.ticket] = d

    return tuple(deduped_by_ticket.values()), duplicates_skipped


def classify_and_aggregate_deals(
    deals: Sequence[MT5DealRecord],
    *,
    reference_execution_deals: Sequence[MT5DealRecord] | None = None,
    reference_deals: Sequence[MT5DealRecord] | None = None,
) -> DealClassificationSummary:
    """Classify and aggregate MT5 deals across all 18 official MQL5 ENUM_DEAL_TYPE values (0..17).

    Enforces:
    1. Deterministic ticket deduplication (rejecting conflicting duplicate tickets).
    2. Complete classification of official MQL5 deal types 0..17 (rejecting unknown codes).
    3. Strict separation of trading PnL vs. non-trading cash flows.
    4. Unified order-level and position-level commission linkage validation (never treating
       `symbol != ''` or `magic > 0` alone as proof of trade linkage).
    5. Anti-double-counting checks preventing inline (`commission` or `fee`) + separate commission
       duplication, mixed `(order=0, position_id=P)` + `(order=O, position_id=P)` separate
       commission duplication, and cross-rollover duplicate separate commissions.
    6. Dividend `position_id`, `order`, and `symbol` consistency validation against known execution deals.
    """
    deduped_deals, duplicates_skipped = deduplicate_deals(deals)

    combined_reference: list[MT5DealRecord] = []
    if reference_execution_deals:
        combined_reference.extend(reference_execution_deals)
    if reference_deals:
        combined_reference.extend(reference_deals)
    deduped_reference, _ = deduplicate_deals(combined_reference) if combined_reference else ((), 0)

    ref_tickets = {d.ticket: d for d in deduped_reference}
    for d in deduped_deals:
        if d.ticket in ref_tickets:
            raise AccountStateReconstructionError(
                f"Duplicate deal ticket {d.ticket} spans both reference/pre-rollover and active deal windows."
            )

    # Index all known execution deals (BUY=0, SELL=1) across reference history + current window
    exec_by_order: dict[int, list[MT5DealRecord]] = {}
    exec_by_position: dict[int, list[MT5DealRecord]] = {}
    order_to_positions: dict[int, set[int]] = {}
    orders_with_inline_charge: set[int] = set()
    positions_with_inline_charge: set[int] = set()
    positions_with_orderless_inline_charge: set[int] = set()

    for exec_deal in (*deduped_reference, *deduped_deals):
        if exec_deal.deal_type not in (
            int(MT5DealType.DEAL_TYPE_BUY),
            int(MT5DealType.DEAL_TYPE_SELL),
        ):
            continue
        sym_clean = exec_deal.symbol.strip().upper()
        if not sym_clean:
            raise AccountStateReconstructionError(
                f"Execution deal (ticket={exec_deal.ticket}, type={exec_deal.deal_type}) is missing a symbol."
            )
        if exec_deal.order <= 0 and exec_deal.position_id <= 0:
            raise AccountStateReconstructionError(
                f"Execution deal (ticket={exec_deal.ticket}, symbol='{sym_clean}') lacks both order and position_id."
            )

        has_inline_charge = (
            exec_deal.commission != Decimal("0") or exec_deal.fee != Decimal("0")
        )
        if exec_deal.order > 0:
            exec_by_order.setdefault(exec_deal.order, []).append(exec_deal)
            if exec_deal.position_id > 0:
                order_to_positions.setdefault(exec_deal.order, set()).add(exec_deal.position_id)
            if has_inline_charge:
                orders_with_inline_charge.add(exec_deal.order)
        if exec_deal.position_id > 0:
            exec_by_position.setdefault(exec_deal.position_id, []).append(exec_deal)
            if has_inline_charge:
                positions_with_inline_charge.add(exec_deal.position_id)
                if exec_deal.order == 0:
                    positions_with_orderless_inline_charge.add(exec_deal.position_id)

    # Track separate commissions across reference deals (pre-rollover) and current window (H-1 fix)
    orders_with_separate_commission: set[int] = set()
    positions_with_orderless_separate_commission: set[int] = set()
    positions_with_ordered_separate_commission: set[int] = set()

    for ref_d in deduped_reference:
        if ref_d.deal_type == int(MT5DealType.DEAL_TYPE_COMMISSION):
            if ref_d.order > 0:
                orders_with_separate_commission.add(ref_d.order)
                target_pids = (
                    {ref_d.position_id}
                    if ref_d.position_id > 0
                    else order_to_positions.get(ref_d.order, set())
                )
                for pid in target_pids:
                    positions_with_ordered_separate_commission.add(pid)
            elif ref_d.position_id > 0:
                positions_with_orderless_separate_commission.add(ref_d.position_id)

    realized_trading_pnl = Decimal("0")
    net_non_trading_cash_flows = Decimal("0")
    classified_records: list[ClassifiedDealRecord] = []

    for deal in deduped_deals:
        try:
            dtype = MT5DealType(deal.deal_type)
        except ValueError as exc:
            raise AccountStateReconstructionError(
                f"Unknown or unsupported MT5 deal_type={deal.deal_type} on deal ticket={deal.ticket}; failing closed."
            ) from exc

        net = deal.net_cash_impact
        sym = deal.symbol.strip().upper()
        has_order = deal.order > 0
        has_pos = deal.position_id > 0

        if dtype in (MT5DealType.DEAL_TYPE_BUY, MT5DealType.DEAL_TYPE_SELL):
            realized_trading_pnl += net
            classified_records.append(
                ClassifiedDealRecord(
                    ticket=deal.ticket,
                    order=deal.order,
                    position_id=deal.position_id,
                    time_utc=deal.time_utc,
                    deal_type=dtype,
                    deal_type_name=dtype.name,
                    symbol=sym,
                    magic=deal.magic,
                    net_cash_impact=net,
                    category=DealAccountingCategory.TRADING_PNL,
                    linked_to_trade=True,
                    linked_order_ticket=deal.order if has_order else None,
                    linked_position_id=deal.position_id if has_pos else None,
                    classification_note="Direct trading execution deal (profit + commission + fee + swap).",
                )
            )

        elif dtype in (
            MT5DealType.DEAL_TYPE_BALANCE,
            MT5DealType.DEAL_TYPE_CREDIT,
            MT5DealType.DEAL_TYPE_BONUS,
        ):
            if has_pos or has_order or sym != "":
                raise AccountStateReconstructionError(
                    f"Non-trading {dtype.name} deal (ticket={deal.ticket}) has unexpected trade/symbol linkage "
                    f"(order={deal.order}, position_id={deal.position_id}, symbol='{sym}'); failing closed."
                )
            net_non_trading_cash_flows += net
            classified_records.append(
                ClassifiedDealRecord(
                    ticket=deal.ticket,
                    order=deal.order,
                    position_id=deal.position_id,
                    time_utc=deal.time_utc,
                    deal_type=dtype,
                    deal_type_name=dtype.name,
                    symbol=sym,
                    magic=deal.magic,
                    net_cash_impact=net,
                    category=DealAccountingCategory.NON_TRADING_CASH_FLOW,
                    linked_to_trade=False,
                    classification_note=f"Verified account-level {dtype.name} cash flow.",
                )
            )

        elif dtype == MT5DealType.DEAL_TYPE_CHARGE:
            if net > Decimal("0"):
                raise AccountStateReconstructionError(
                    f"Positive DEAL_TYPE_CHARGE (ticket={deal.ticket}, net={net}) is ambiguous; failing closed."
                )
            realized_trading_pnl += net
            classified_records.append(
                ClassifiedDealRecord(
                    ticket=deal.ticket,
                    order=deal.order,
                    position_id=deal.position_id,
                    time_utc=deal.time_utc,
                    deal_type=dtype,
                    deal_type_name=dtype.name,
                    symbol=sym,
                    magic=deal.magic,
                    net_cash_impact=net,
                    category=DealAccountingCategory.TRADING_PNL,
                    linked_to_trade=bool(has_order or has_pos),
                    linked_order_ticket=deal.order if has_order else None,
                    linked_position_id=deal.position_id if has_pos else None,
                    classification_note="Conservative charge policy: negative fee deducted from trading PnL.",
                )
            )

        elif dtype == MT5DealType.DEAL_TYPE_CORRECTION:
            raise AccountStateReconstructionError(
                f"DEAL_TYPE_CORRECTION (ticket={deal.ticket}, net={net}) has ambiguous historical vs. intraday "
                "risk attribution; failing closed."
            )

        elif dtype == MT5DealType.DEAL_TYPE_COMMISSION:
            if net > Decimal("0"):
                raise AccountStateReconstructionError(
                    f"Positive DEAL_TYPE_COMMISSION rebate (ticket={deal.ticket}, net={net}) is forbidden; failing closed."
                )

            if has_order or has_pos:
                if has_order and deal.order not in exec_by_order:
                    raise AccountStateReconstructionError(
                        f"Separate DEAL_TYPE_COMMISSION (ticket={deal.ticket}) references order={deal.order} "
                        "which does not exist in verified execution deal history."
                    )
                if has_pos and deal.position_id not in exec_by_position:
                    raise AccountStateReconstructionError(
                        f"Separate DEAL_TYPE_COMMISSION (ticket={deal.ticket}) references position_id={deal.position_id} "
                        "which does not exist in verified execution deal history."
                    )

                linked_execs = exec_by_order[deal.order] if has_order else exec_by_position[deal.position_id]
                if sym != "" and any(d.symbol.strip().upper() != sym for d in linked_execs):
                    raise AccountStateReconstructionError(
                        f"Separate DEAL_TYPE_COMMISSION (ticket={deal.ticket}, symbol='{sym}') mismatches "
                        "the symbol of its linked execution deal."
                    )
                if has_order and has_pos:
                    if all(d.position_id != deal.position_id for d in exec_by_order[deal.order] if d.position_id > 0):
                        raise AccountStateReconstructionError(
                            f"Separate DEAL_TYPE_COMMISSION (ticket={deal.ticket}) has inconsistent "
                            f"order={deal.order} and position_id={deal.position_id}."
                        )

                target_pos_ids: set[int] = (
                    {deal.position_id}
                    if has_pos
                    else set(order_to_positions.get(deal.order, set()))
                )

                # Unified anti-double-counting check against inline commission AND inline fee (H-1 fix)
                if has_order and deal.order in orders_with_inline_charge:
                    raise AccountStateReconstructionError(
                        f"Commission double-counting detected: DEAL_TYPE_COMMISSION (ticket={deal.ticket}) "
                        f"references order={deal.order} which already has non-zero inline commission or fee."
                    )
                if not has_order and any(pid in positions_with_inline_charge for pid in target_pos_ids):
                    raise AccountStateReconstructionError(
                        f"Commission double-counting detected: DEAL_TYPE_COMMISSION (ticket={deal.ticket}) "
                        f"references position_id={deal.position_id} which already has non-zero inline commission or fee."
                    )
                if has_order and any(pid in positions_with_orderless_inline_charge for pid in target_pos_ids):
                    raise AccountStateReconstructionError(
                        f"Commission double-counting detected: DEAL_TYPE_COMMISSION (ticket={deal.ticket}) "
                        f"references order={deal.order} on position_id={target_pos_ids} which already has "
                        "an orderless inline commission or fee."
                    )

                # Unified anti-double-counting across order-linked and position-linked separate commissions (H-1 fix)
                if has_order:
                    if deal.order in orders_with_separate_commission:
                        raise AccountStateReconstructionError(
                            f"Duplicate separate DEAL_TYPE_COMMISSION (ticket={deal.ticket}) for order={deal.order}."
                        )
                    if any(pid in positions_with_orderless_separate_commission for pid in target_pos_ids):
                        raise AccountStateReconstructionError(
                            f"Mixed order/position duplicate separate DEAL_TYPE_COMMISSION (ticket={deal.ticket}) "
                            f"for order={deal.order} on position_id={target_pos_ids}."
                        )
                    orders_with_separate_commission.add(deal.order)
                    for pid in target_pos_ids:
                        positions_with_ordered_separate_commission.add(pid)
                else:
                    if (
                        deal.position_id in positions_with_orderless_separate_commission
                        or deal.position_id in positions_with_ordered_separate_commission
                    ):
                        raise AccountStateReconstructionError(
                            f"Duplicate or mixed order/position separate DEAL_TYPE_COMMISSION (ticket={deal.ticket}) "
                            f"for position_id={deal.position_id}."
                        )
                    positions_with_orderless_separate_commission.add(deal.position_id)

                realized_trading_pnl += net
                classified_records.append(
                    ClassifiedDealRecord(
                        ticket=deal.ticket,
                        order=deal.order,
                        position_id=deal.position_id,
                        time_utc=deal.time_utc,
                        deal_type=dtype,
                        deal_type_name=dtype.name,
                        symbol=sym,
                        magic=deal.magic,
                        net_cash_impact=net,
                        category=DealAccountingCategory.TRADING_PNL,
                        linked_to_trade=True,
                        linked_order_ticket=deal.order if has_order else None,
                        linked_position_id=deal.position_id if has_pos else None,
                        classification_note="Verified per-trade separate commission linked by order/position_id.",
                    )
                )
            else:
                # order == 0 and position_id == 0: symbol != '' or magic > 0 alone is NEVER treated as trade linkage
                if net >= Decimal("0"):
                    raise AccountStateReconstructionError(
                        f"Unlinked DEAL_TYPE_COMMISSION (ticket={deal.ticket}, net={net}) without order or "
                        "position_id must be a strictly negative charge; failing closed."
                    )
                realized_trading_pnl += net
                classified_records.append(
                    ClassifiedDealRecord(
                        ticket=deal.ticket,
                        order=0,
                        position_id=0,
                        time_utc=deal.time_utc,
                        deal_type=dtype,
                        deal_type_name=dtype.name,
                        symbol=sym,
                        magic=deal.magic,
                        net_cash_impact=net,
                        category=DealAccountingCategory.TRADING_PNL,
                        linked_to_trade=False,
                        linked_order_ticket=None,
                        linked_position_id=None,
                        classification_note=(
                            "Unlinked commission charge (order=0, position_id=0; symbol/magic alone not sufficient "
                            "for trade linkage): conservatively deducted from trading PnL."
                        ),
                    )
                )

        elif dtype in (
            MT5DealType.DEAL_TYPE_COMMISSION_DAILY,
            MT5DealType.DEAL_TYPE_COMMISSION_MONTHLY,
            MT5DealType.DEAL_TYPE_COMMISSION_AGENT_DAILY,
            MT5DealType.DEAL_TYPE_COMMISSION_AGENT_MONTHLY,
        ):
            if net > Decimal("0"):
                raise AccountStateReconstructionError(
                    f"Positive account-level {dtype.name} (ticket={deal.ticket}, net={net}) is invalid; failing closed."
                )
            realized_trading_pnl += net
            classified_records.append(
                ClassifiedDealRecord(
                    ticket=deal.ticket,
                    order=deal.order,
                    position_id=deal.position_id,
                    time_utc=deal.time_utc,
                    deal_type=dtype,
                    deal_type_name=dtype.name,
                    symbol=sym,
                    magic=deal.magic,
                    net_cash_impact=net,
                    category=DealAccountingCategory.TRADING_PNL,
                    linked_to_trade=False,
                    classification_note=f"Account-level {dtype.name} charge included in trading PnL.",
                )
            )

        elif dtype == MT5DealType.DEAL_TYPE_INTEREST:
            if net < Decimal("0"):
                realized_trading_pnl += net
                classified_records.append(
                    ClassifiedDealRecord(
                        ticket=deal.ticket,
                        order=deal.order,
                        position_id=deal.position_id,
                        time_utc=deal.time_utc,
                        deal_type=dtype,
                        deal_type_name=dtype.name,
                        symbol=sym,
                        magic=deal.magic,
                        net_cash_impact=net,
                        category=DealAccountingCategory.TRADING_PNL,
                        linked_to_trade=False,
                        classification_note="Negative DEAL_TYPE_INTEREST charge deducted from trading PnL.",
                    )
                )
            else:
                if has_pos or has_order or sym != "":
                    raise AccountStateReconstructionError(
                        f"Positive DEAL_TYPE_INTEREST (ticket={deal.ticket}, net={net}) linked to "
                        f"order/position/symbol is ambiguous; failing closed."
                    )
                net_non_trading_cash_flows += net
                classified_records.append(
                    ClassifiedDealRecord(
                        ticket=deal.ticket,
                        order=0,
                        position_id=0,
                        time_utc=deal.time_utc,
                        deal_type=dtype,
                        deal_type_name=dtype.name,
                        symbol="",
                        magic=deal.magic,
                        net_cash_impact=net,
                        category=DealAccountingCategory.NON_TRADING_CASH_FLOW,
                        linked_to_trade=False,
                        classification_note=(
                            "Positive free-margin interest credit classified as non-trading cash flow "
                            "so interest income never masks trading losses."
                        ),
                    )
                )

        elif dtype in (
            MT5DealType.DEAL_TYPE_BUY_CANCELED,
            MT5DealType.DEAL_TYPE_SELL_CANCELED,
        ):
            raise AccountStateReconstructionError(
                f"Canceled execution deal {dtype.name} (ticket={deal.ticket}) retroactively alters trade history; "
                "failing closed."
            )

        elif dtype in (
            MT5DealType.DEAL_DIVIDEND,
            MT5DealType.DEAL_DIVIDEND_FRANKED,
        ):
            # M-1 remediation: strictly require position_id > 0, non-empty symbol, and verified trade consistency
            if not has_pos or sym == "":
                raise AccountStateReconstructionError(
                    f"Unattributed {dtype.name} (ticket={deal.ticket}) requires position_id > 0 and non-empty symbol; "
                    "failing closed."
                )
            if deal.position_id not in exec_by_position:
                raise AccountStateReconstructionError(
                    f"{dtype.name} (ticket={deal.ticket}) references position_id={deal.position_id} "
                    "which does not exist in verified execution deal history; failing closed."
                )
            if has_order:
                if deal.order not in exec_by_order or (
                    deal.position_id not in order_to_positions.get(deal.order, set())
                ):
                    raise AccountStateReconstructionError(
                        f"{dtype.name} (ticket={deal.ticket}) has invalid or inconsistent "
                        f"order={deal.order} for position_id={deal.position_id}; failing closed."
                    )
            if any(d.symbol.strip().upper() != sym for d in exec_by_position[deal.position_id]):
                raise AccountStateReconstructionError(
                    f"{dtype.name} (ticket={deal.ticket}, symbol='{sym}') mismatches the symbol "
                    f"of its linked execution position (position_id={deal.position_id}); failing closed."
                )

            realized_trading_pnl += net
            classified_records.append(
                ClassifiedDealRecord(
                    ticket=deal.ticket,
                    order=deal.order,
                    position_id=deal.position_id,
                    time_utc=deal.time_utc,
                    deal_type=dtype,
                    deal_type_name=dtype.name,
                    symbol=sym,
                    magic=deal.magic,
                    net_cash_impact=net,
                    category=DealAccountingCategory.TRADING_PNL,
                    linked_to_trade=True,
                    linked_order_ticket=deal.order if has_order else None,
                    linked_position_id=deal.position_id,
                    classification_note=f"Verified position-linked {dtype.name} adjustment included in trading PnL.",
                )
            )

        elif dtype == MT5DealType.DEAL_TAX:
            if net > Decimal("0"):
                raise AccountStateReconstructionError(
                    f"Positive DEAL_TAX refund (ticket={deal.ticket}, net={net}) is ambiguous; failing closed."
                )
            realized_trading_pnl += net
            classified_records.append(
                ClassifiedDealRecord(
                    ticket=deal.ticket,
                    order=deal.order,
                    position_id=deal.position_id,
                    time_utc=deal.time_utc,
                    deal_type=dtype,
                    deal_type_name=dtype.name,
                    symbol=sym,
                    magic=deal.magic,
                    net_cash_impact=net,
                    category=DealAccountingCategory.TRADING_PNL,
                    linked_to_trade=bool(has_order or has_pos),
                    linked_order_ticket=deal.order if has_order else None,
                    linked_position_id=deal.position_id if has_pos else None,
                    classification_note="Conservative tax deduction included in trading PnL.",
                )
            )

    total_net = realized_trading_pnl + net_non_trading_cash_flows
    return DealClassificationSummary(
        realized_trading_pnl=realized_trading_pnl,
        net_non_trading_cash_flows=net_non_trading_cash_flows,
        total_net_cash_impact=total_net,
        classified_deals=tuple(classified_records),
        deduplicated_deal_count=len(deduped_deals),
        duplicates_skipped_count=duplicates_skipped,
    )


class PersistentAccountRiskTracker:
    """Reconstructs and persists verified daily_starting_balance, current_daily_pnl, and peak_balance
    using a mandatory TrustedBalanceAnchor and complete official MQL5 deal classification.

    Mandatory Safety Invariants:
    1. Requires a valid TrustedBalanceAnchor (`PERSISTED_ROLLOVER_ANCHOR` or `CONFIGURED_BALANCE_ANCHOR`)
       bound to the exact `(account_login, server)`.
    2. Never assumes `daily_starting_balance == current_balance` without a verified anchor.
    3. Never treats an empty `history_deals_get()` result as proof of complete history.
    4. Normalizes all timestamps to timezone-aware UTC and computes trading-day boundaries at
       5 PM `America/New_York` with IANA `ZoneInfo` DST handling.
    5. Verifies reconstructed balance against `account_info().balance` within `currency_digits`-aware tolerance.
    6. Preserves `peak_balance` across 5 PM NY daily rollovers and persists verified rollover snapshots.
    """

    def __init__(
        self,
        connection: MT5ConnectionAdapter,
        *,
        safety_config: ExecutionSafetyConfig | None = None,
        persistence: CrashSafeJsonPersistence | None = None,
        trusted_balance_anchor: TrustedBalanceAnchorConfig | None = None,
    ) -> None:
        self._connection = connection
        self._safety_config = safety_config or ExecutionSafetyConfig()
        self._persistence = persistence or CrashSafeJsonPersistence(
            self._safety_config.account_risk_state_path,
            lock_timeout_seconds=self._safety_config.lock_timeout_seconds,
        )
        self._trusted_anchor_override = trusted_balance_anchor
        self._last_result: AccountRiskReconstructionResult | None = None

    @property
    def connection(self) -> MT5ConnectionAdapter:
        return self._connection

    @property
    def safety_config(self) -> ExecutionSafetyConfig:
        return self._safety_config

    @property
    def persistence(self) -> CrashSafeJsonPersistence:
        return self._persistence

    @property
    def last_reconstruction_result(self) -> AccountRiskReconstructionResult | None:
        return self._last_result

    def reconstruct_report(
        self,
        *,
        reference_now_utc: datetime | None = None,
        symbol: str | None = None,
    ) -> AccountRiskReconstructionResult:
        """Reconstruct and verify account risk state from the trusted anchor and broker deal history."""
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

        # 1. Load persisted rollover snapshot (if present). Never silence corruption or rollback errors!
        persisted_env = self._persistence.load(
            expected_login=account.login,
            expected_server=account.server,
        )

        persisted_snapshot: AccountRiskPersistenceSnapshot | None = None
        if persisted_env is not None:
            try:
                persisted_snapshot = AccountRiskPersistenceSnapshot.model_validate(persisted_env.payload)
            except ValidationError as exc:
                raise PersistenceCorruptionError(
                    f"Persisted AccountRiskPersistenceSnapshot payload is invalid: {exc}"
                ) from exc

            if (
                persisted_snapshot.account_login != account.login
                or persisted_snapshot.server.strip().lower() != account.server.strip().lower()
            ):
                raise PersistenceAccountMismatchError(
                    f"Persisted snapshot identity ({persisted_snapshot.account_login}@{persisted_snapshot.server}) "
                    f"does not match active MT5 account ({account.login}@{account.server})."
                )

        # 2. Resolve trading day window at 5 PM America/New_York (with IANA ZoneInfo DST handling)
        current_day_label, day_start_utc, day_end_utc = resolve_trading_day_window(
            now_utc,
            day_boundary_timezone=self._safety_config.day_boundary_timezone,
            day_rollover_hour=self._safety_config.day_rollover_hour,
        )

        # 3. Determine active TrustedBalanceAnchor
        configured_anchor_ticket: int | None = None
        persisted_init_deposit_ticket: int | None = None

        if persisted_snapshot is not None:
            if current_day_label < persisted_snapshot.trading_day_label:
                raise AccountStateReconstructionError(
                    f"Trading-day rollback detected: current trading day ({current_day_label}) is earlier "
                    f"than persisted snapshot trading day ({persisted_snapshot.trading_day_label})."
                )
            if now_utc < persisted_snapshot.last_updated_utc:
                raise AccountStateReconstructionError(
                    f"Clock rollback detected: reference_now_utc ({now_utc.isoformat()}) is earlier "
                    f"than persisted snapshot last_updated_utc ({persisted_snapshot.last_updated_utc.isoformat()})."
                )
            anchor_source = TrustedAnchorSource.PERSISTED_ROLLOVER_ANCHOR
            anchor_balance = persisted_snapshot.anchor_balance
            anchor_timestamp_utc = persisted_snapshot.anchor_timestamp_utc
            require_initial_deposit = False
            prior_peak_balance = persisted_snapshot.peak_balance
            persisted_init_deposit_ticket = persisted_snapshot.initial_deposit_ticket
        else:
            configured_anchor = (
                self._trusted_anchor_override or self._safety_config.trusted_balance_anchor
            )
            if configured_anchor is None:
                raise AccountStateReconstructionError(
                    f"No TrustedBalanceAnchor available for account {account.login}@{account.server}: "
                    "no persisted rollover snapshot exists and trusted_balance_anchor is not configured. "
                    "Failing closed (never assuming daily_starting_balance == current_balance)."
                )
            anchor_source = TrustedAnchorSource.CONFIGURED_BALANCE_ANCHOR
            anchor_balance = configured_anchor.anchor_balance
            anchor_timestamp_utc = configured_anchor.anchor_timestamp_utc
            require_initial_deposit = (
                configured_anchor.require_initial_deposit_deal or anchor_balance == Decimal("0")
            )
            configured_anchor_ticket = configured_anchor.anchor_deal_ticket
            prior_peak_balance = anchor_balance if anchor_balance > Decimal("0") else account.balance

        if anchor_timestamp_utc > now_utc:
            raise AccountStateReconstructionError(
                f"TrustedBalanceAnchor timestamp ({anchor_timestamp_utc.isoformat()}) is in the future "
                f"relative to reference_now_utc ({now_utc.isoformat()})."
            )

        if anchor_timestamp_utc > day_start_utc:
            raise AccountStateReconstructionError(
                f"TrustedBalanceAnchor timestamp ({anchor_timestamp_utc.isoformat()}) is after current "
                f"trading-day start ({day_start_utc.isoformat()}); cannot verify full trading-day history."
            )

        # 4. Query deal history and never treat an empty history_deals_get() result as proof of completeness.
        # Query with a forward horizon beyond now_utc so any broker deal with time_utc > now_utc is detected
        # rather than silently excluded by the backend query window (H-4 remediation).
        future_horizon_utc = max(
            now_utc + timedelta(days=3650),
            datetime(2100, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        )
        lifetime_deals = self._connection.backend.history_deals_get(EPOCH_UTC, future_horizon_utc)
        if lifetime_deals is None:
            raise AccountStateReconstructionError(
                "MT5 history_deals_get(EPOCH_UTC, future_horizon_utc) returned None; deal history is unavailable."
            )
        if len(lifetime_deals) == 0:
            raise AccountStateReconstructionError(
                "MT5 history_deals_get() returned an empty deal history (); an empty result is never "
                "accepted as proof of complete history."
            )

        raw_window_deals = self._connection.backend.history_deals_get(
            anchor_timestamp_utc,
            future_horizon_utc,
        )
        if raw_window_deals is None:
            raise AccountStateReconstructionError(
                f"MT5 history_deals_get({anchor_timestamp_utc.isoformat()}, {future_horizon_utc.isoformat()}) returned None."
            )

        # H-4 remediation: Reject any deal with timestamp later than reference_now_utc
        for candidate_deal in (*lifetime_deals, *raw_window_deals):
            if candidate_deal.time_utc > now_utc:
                raise AccountStateReconstructionError(
                    f"Future-dated deal detected in broker history (ticket={candidate_deal.ticket}, "
                    f"time_utc={candidate_deal.time_utc.isoformat()} > reference_now_utc={now_utc.isoformat()}). "
                    "Never silently filtering future deals; failing closed."
                )

        deduped_window_deals, _ = deduplicate_deals(
            [d for d in raw_window_deals if d.time_utc >= anchor_timestamp_utc]
        )
        window_deals = list(deduped_window_deals)

        if require_initial_deposit:
            has_initial_deposit = any(
                d.deal_type == int(MT5DealType.DEAL_TYPE_BALANCE)
                and d.net_cash_impact > Decimal("0")
                and d.position_id == 0
                and d.order == 0
                and d.symbol.strip() == ""
                for d in window_deals
            )
            if not has_initial_deposit:
                raise AccountStateReconstructionError(
                    "Initial deposit verification failed: require_initial_deposit_deal=True (or anchor_balance=0), "
                    "but no positive account-level DEAL_TYPE_BALANCE deposit deal was found in deal history."
                )

        # H-2 remediation: Use explicit ticket provenance (anchor_deal_ticket) rather than amount heuristics
        if configured_anchor_ticket is not None:
            matched_anchor_deals = [
                d
                for d in window_deals
                if d.ticket == configured_anchor_ticket and d.time_utc == anchor_timestamp_utc
            ]
            if not matched_anchor_deals:
                raise AccountStateReconstructionError(
                    f"Configured anchor_deal_ticket={configured_anchor_ticket} was not found at "
                    f"anchor_timestamp_utc={anchor_timestamp_utc.isoformat()}."
                )
            window_deals = [d for d in window_deals if d.ticket != configured_anchor_ticket]

        # 5. Partition deals into pre-rollover [anchor_timestamp_utc, day_start_utc) and current-day [day_start_utc, now_utc]
        pre_rollover_raw = [d for d in window_deals if d.time_utc < day_start_utc]
        current_day_raw = [d for d in window_deals if d.time_utc >= day_start_utc]

        pre_summary = classify_and_aggregate_deals(pre_rollover_raw)
        today_summary = classify_and_aggregate_deals(
            current_day_raw,
            reference_deals=pre_rollover_raw,
        )

        pre_rollover_net_change = pre_summary.total_net_cash_impact
        realized_trading_pnl_today = today_summary.realized_trading_pnl
        net_non_trading_cash_flows_today = today_summary.net_non_trading_cash_flows
        active_initial_deposit_ticket: int | None = None

        rolled_over = (
            persisted_snapshot is not None
            and persisted_snapshot.trading_day_label != current_day_label
        )

        # C-2 & H-2 remediation: Handle same-day account-opening initial deposit via explicit ticket provenance
        # while always keeping snapshot.anchor_timestamp_utc == day_start_utc.
        if persisted_init_deposit_ticket is not None:
            if not rolled_over:
                # Same trading day as when the account was opened from 0.00
                matched_init = next(
                    (
                        d
                        for d in current_day_raw
                        if d.ticket == persisted_init_deposit_ticket
                        and d.deal_type == int(MT5DealType.DEAL_TYPE_BALANCE)
                    ),
                    None,
                )
                if matched_init is None or matched_init.net_cash_impact != anchor_balance:
                    raise AccountStateReconstructionError(
                        f"Persisted same-day initial_deposit_ticket={persisted_init_deposit_ticket} "
                        "is missing or modified in current-day deal history; failing closed."
                    )
                net_non_trading_cash_flows_today -= matched_init.net_cash_impact
                active_initial_deposit_ticket = persisted_init_deposit_ticket
            else:
                # Rolled over to a new trading day after the opening day: the opening deposit is in pre_rollover_raw
                matched_init_pre = next(
                    (
                        d
                        for d in pre_rollover_raw
                        if d.ticket == persisted_init_deposit_ticket
                        and d.deal_type == int(MT5DealType.DEAL_TYPE_BALANCE)
                    ),
                    None,
                )
                if matched_init_pre is None or matched_init_pre.net_cash_impact != anchor_balance:
                    raise AccountStateReconstructionError(
                        f"Persisted opening-day initial_deposit_ticket={persisted_init_deposit_ticket} "
                        "is missing or modified in pre-rollover deal history; failing closed."
                    )
                pre_rollover_net_change -= matched_init_pre.net_cash_impact
                active_initial_deposit_ticket = None

        daily_starting_balance = anchor_balance + pre_rollover_net_change

        if daily_starting_balance == Decimal("0") and require_initial_deposit:
            first_deposit = next(
                (
                    d
                    for d in sorted(current_day_raw, key=lambda item: (item.time_utc, item.ticket))
                    if d.deal_type == int(MT5DealType.DEAL_TYPE_BALANCE)
                    and d.net_cash_impact > Decimal("0")
                    and d.position_id == 0
                    and d.order == 0
                    and d.symbol.strip() == ""
                ),
                None,
            )
            if first_deposit is not None:
                daily_starting_balance = first_deposit.net_cash_impact
                net_non_trading_cash_flows_today -= first_deposit.net_cash_impact
                active_initial_deposit_ticket = first_deposit.ticket

        if daily_starting_balance <= Decimal("0"):
            raise AccountStateReconstructionError(
                f"Reconstructed daily_starting_balance ({daily_starting_balance}) must be > 0 "
                f"(anchor_balance={anchor_balance}, pre_rollover_net_change={pre_rollover_net_change})."
            )

        # 6. Verify reconstructed balance against account_info().balance within currency_digits-aware tolerance
        reconstructed_balance = (
            daily_starting_balance + realized_trading_pnl_today + net_non_trading_cash_flows_today
        )
        tolerance = account.reconciliation_tolerance(
            self._safety_config.reconciliation_tolerance_units
        )
        discrepancy = abs(account.balance - reconstructed_balance)

        if discrepancy > tolerance:
            raise AccountStateReconstructionError(
                f"Balance completeness verification failed for {account.login}@{account.server}: "
                f"reconstructed_balance={reconstructed_balance} {account.currency} "
                f"(anchor={anchor_balance}, pre_rollover={pre_rollover_net_change}, "
                f"trading_pnl_today={realized_trading_pnl_today}, "
                f"non_trading_today={net_non_trading_cash_flows_today}) != "
                f"broker_balance={account.balance} {account.currency} "
                f"(discrepancy={discrepancy} > tolerance={tolerance}, currency_digits={account.currency_digits})."
            )

        if (
            persisted_snapshot is not None
            and persisted_snapshot.trading_day_label == current_day_label
            and abs(daily_starting_balance - persisted_snapshot.daily_starting_balance) > tolerance
        ):
            raise AccountStateReconstructionError(
                f"Intraday daily_starting_balance drift detected for {current_day_label}: "
                f"reconstructed={daily_starting_balance} != persisted={persisted_snapshot.daily_starting_balance}."
            )

        # 7. Query all account-wide open positions (H-3 remediation: never filter open_positions_count by symbol)
        all_positions = self._connection.backend.positions_get(None)
        if all_positions is None:
            raise AccountStateReconstructionError(
                "MT5 positions_get(None) returned None while building AccountState."
            )
        if symbol is not None and self._connection.backend.positions_get(symbol) is None:
            raise AccountStateReconstructionError(
                f"MT5 positions_get('{symbol}') returned None while building AccountState."
            )
        open_positions_count = len(all_positions)

        # 8. H-3 remediation: Broker credit & floating PnL accounting
        positions_floating_pnl = sum(
            (pos.profit + pos.swap for pos in all_positions),
            Decimal("0"),
        )
        if account.credit < Decimal("0"):
            raise AccountStateReconstructionError(
                f"Negative account.credit ({account.credit}) is invalid; failing closed."
            )
        if account.credit > Decimal("0"):
            # In MT5, equity = balance + credit + floating_pnl. Verify against open positions floating PnL.
            credit_adjusted_floating = account.equity - account.balance - account.credit
            if abs(credit_adjusted_floating - positions_floating_pnl) > tolerance:
                raise AccountStateReconstructionError(
                    f"Ambiguous broker credit accounting for {account.login}@{account.server}: "
                    f"balance={account.balance}, credit={account.credit}, equity={account.equity}, "
                    f"positions_floating_pnl={positions_floating_pnl} "
                    f"(credit_adjusted_floating={credit_adjusted_floating}, tolerance={tolerance}); "
                    "failing closed."
                )
            unrealized_floating_pnl = credit_adjusted_floating
            effective_equity = account.equity - account.credit
            if effective_equity <= Decimal("0"):
                raise AccountStateReconstructionError(
                    f"Account true cash equity excluding credit ({effective_equity}) must be > 0; failing closed."
                )
        else:
            unrealized_floating_pnl = account.equity - account.balance
            effective_equity = account.equity

        current_daily_pnl = realized_trading_pnl_today + unrealized_floating_pnl

        # 9. Track running balance high-water mark to preserve peak_balance across rollovers
        running_bal = (
            Decimal("0")
            if (persisted_init_deposit_ticket is not None and not rolled_over)
            else anchor_balance
        )
        max_running_bal = max(anchor_balance, daily_starting_balance)
        for cd in (*pre_summary.classified_deals, *today_summary.classified_deals):
            if (
                persisted_init_deposit_ticket is not None
                and rolled_over
                and cd.ticket == persisted_init_deposit_ticket
            ):
                continue
            running_bal += cd.net_cash_impact
            if running_bal > max_running_bal:
                max_running_bal = running_bal

        peak_balance = max(
            prior_peak_balance,
            max_running_bal,
            daily_starting_balance,
            account.balance,
            effective_equity,
        )

        # Clamp free_margin safely to <= effective_equity for AccountState validation
        effective_free_margin = min(max(Decimal("0"), account.margin_free), effective_equity)

        account_state = AccountState(
            balance=account.balance,
            equity=effective_equity,
            free_margin=effective_free_margin,
            currency=account.currency,
            open_positions_count=open_positions_count,
            current_daily_pnl=current_daily_pnl,
            daily_starting_balance=daily_starting_balance,
            peak_balance=peak_balance,
        )

        # 10. M-3 remediation: Avoid rewriting snapshot or incrementing generation when verified state is unchanged
        state_unchanged = (
            persisted_snapshot is not None
            and persisted_env is not None
            and not rolled_over
            and persisted_snapshot.account_login == account.login
            and persisted_snapshot.server.strip().lower() == account.server.strip().lower()
            and persisted_snapshot.currency == account.currency
            and persisted_snapshot.currency_digits == account.currency_digits
            and persisted_snapshot.trading_day_label == current_day_label
            and persisted_snapshot.anchor_timestamp_utc == day_start_utc
            and persisted_snapshot.anchor_balance == daily_starting_balance
            and persisted_snapshot.daily_starting_balance == daily_starting_balance
            and persisted_snapshot.peak_balance == peak_balance
            and persisted_snapshot.last_verified_balance == account.balance
            and persisted_snapshot.initial_deposit_ticket == active_initial_deposit_ticket
        )

        if state_unchanged:
            assert persisted_snapshot is not None and persisted_env is not None
            active_snapshot = persisted_snapshot
            active_generation = persisted_env.generation
        else:
            active_snapshot = AccountRiskPersistenceSnapshot(
                account_login=account.login,
                server=account.server,
                currency=account.currency,
                currency_digits=account.currency_digits,
                trading_day_label=current_day_label,
                anchor_timestamp_utc=day_start_utc,
                anchor_balance=daily_starting_balance,
                daily_starting_balance=daily_starting_balance,
                peak_balance=peak_balance,
                last_verified_balance=account.balance,
                last_updated_utc=now_utc,
                initial_deposit_ticket=active_initial_deposit_ticket,
            )
            saved_envelope = self._persistence.save(
                active_snapshot.model_dump(mode="json"),
                account_login=account.login,
                server=account.server,
                now_utc=now_utc,
            )
            active_generation = saved_envelope.generation

        result = AccountRiskReconstructionResult(
            account_state=account_state,
            snapshot=active_snapshot,
            anchor_source=anchor_source,
            anchor_timestamp_utc=anchor_timestamp_utc,
            anchor_balance=anchor_balance,
            trading_day_label=current_day_label,
            day_start_utc=day_start_utc,
            day_end_utc=day_end_utc,
            pre_rollover_net_change=pre_rollover_net_change,
            daily_starting_balance=daily_starting_balance,
            realized_trading_pnl_today=realized_trading_pnl_today,
            net_non_trading_cash_flows_today=net_non_trading_cash_flows_today,
            unrealized_floating_pnl=unrealized_floating_pnl,
            current_daily_pnl=current_daily_pnl,
            reconstructed_balance=reconstructed_balance,
            broker_balance=account.balance,
            balance_discrepancy=discrepancy,
            reconciliation_tolerance=tolerance,
            peak_balance=peak_balance,
            rolled_over=rolled_over,
            envelope_generation=active_generation,
            classified_deals_today=today_summary.classified_deals,
            classified_deals_pre_rollover=pre_summary.classified_deals,
        )
        self._last_result = result
        return result

    def reconstruct_account_state(
        self,
        *,
        reference_now_utc: datetime | None = None,
        symbol: str | None = None,
    ) -> AccountState:
        """Reconstruct and return a validated Phase 1 `AccountState` for `RiskEngine.evaluate()`."""
        report = self.reconstruct_report(
            reference_now_utc=reference_now_utc,
            symbol=symbol,
        )
        return report.account_state
