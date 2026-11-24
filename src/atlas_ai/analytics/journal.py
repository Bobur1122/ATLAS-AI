"""Crash-Safe Persistent Post-Trade Journal for ATLAS AI Phase 8.1."""

from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal
import threading
from typing import TYPE_CHECKING, Any, Sequence

from pydantic import ValidationError

from atlas_ai.analytics.config import PostTradeAnalyticsConfig
from atlas_ai.analytics.exceptions import TradeReconstructionError
from atlas_ai.analytics.models import (
    CompletedTradeRecord,
    ContextAttributionStatus,
    EntryContextSnapshot,
    KillZoneSession,
    PartialExitProgressRecord,
)
from atlas_ai.core.logger import get_logger
from atlas_ai.domain.market import InstrumentSpec
from atlas_ai.domain.signals import SignalDirection
from atlas_ai.execution.exceptions import (
    LedgerRecoveryLockError,
    PersistenceAccountMismatchError,
    PersistenceCorruptionError,
    PersistenceError,
    PersistenceRollbackDetectedError,
)
from atlas_ai.execution.models import (
    ExecutionLedgerEntry,
    OrderLedgerStatus,
    ReconciliationStatus,
)
from atlas_ai.execution.persistence import CrashSafeJsonPersistence
from atlas_ai.strategies.models import (
    MarketRegime,
    SignalLifecycleState,
    StrategyAnalysisReport,
)

if TYPE_CHECKING:
    from atlas_ai.execution.engine import MT5DemoExecutionEngine

logger = get_logger("analytics.journal")


def build_entry_context_snapshot(
    report: StrategyAnalysisReport,
    *,
    client_order_id: str,
    signal_id: str,
    approved_volume: Decimal,
    initial_risk_amount: Decimal,
    expected_spread_points: Decimal = Decimal("0"),
    captured_at_utc: datetime | None = None,
) -> EntryContextSnapshot:
    """Extract an immutable `EntryContextSnapshot` from a Phase 3 `StrategyAnalysisReport`
    and validated order sizing parameters.
    """
    sig = report.signal
    if (
        sig.direction not in (SignalDirection.BUY, SignalDirection.SELL)
        or sig.entry_price is None
        or sig.stop_loss is None
        or sig.take_profit is None
    ):
        raise TradeReconstructionError(
            f"Cannot build EntryContextSnapshot for non-executable signal '{signal_id}'."
        )

    ts = (
        captured_at_utc.astimezone(timezone.utc)
        if (captured_at_utc is not None and captured_at_utc.tzinfo is not None)
        else (
            captured_at_utc.replace(tzinfo=timezone.utc)
            if captured_at_utc is not None
            else (
                sig.timestamp.astimezone(timezone.utc)
                if sig.timestamp.tzinfo is not None
                else sig.timestamp.replace(tzinfo=timezone.utc)
            )
        )
    )

    killzone = KillZoneSession.OFF_SESSION
    contributing_strats: list[str] = []
    contributing_rules: list[str] = []
    ml_model_id: str | None = None
    ml_pred_dir: SignalDirection | None = None
    ml_buy_prob: Decimal | None = None
    ml_sell_prob: Decimal | None = None
    ml_wait_prob: Decimal | None = None

    for strat_id, comp_eval in report.confluence.component_evaluations.items():
        if comp_eval.direction == sig.direction and comp_eval.confidence > Decimal("0"):
            contributing_strats.append(strat_id)
        comp_ev = comp_eval.supporting_evidence
        raw_session = comp_ev.get("active_killzone") or comp_ev.get("killzone_session")
        if isinstance(raw_session, str) and raw_session.upper() in KillZoneSession.__members__:
            killzone = KillZoneSession(raw_session.upper())
        raw_rules = comp_ev.get("matched_rule_ids")
        if isinstance(raw_rules, (list, tuple)):
            contributing_rules.extend(str(r) for r in raw_rules if str(r).strip())
        raw_ml_id = comp_ev.get("model_id") or comp_ev.get("ml_model_id")
        if isinstance(raw_ml_id, str) and raw_ml_id.strip():
            ml_model_id = raw_ml_id.strip()

    # Extract optional session, knowledge rules, or ML probabilities if attached in lifecycle_record supporting_evidence
    if report.lifecycle_record is not None:
        ev = report.lifecycle_record.supporting_evidence
        raw_session = ev.get("killzone_session")
        if isinstance(raw_session, str) and raw_session.upper() in KillZoneSession.__members__:
            killzone = KillZoneSession(raw_session.upper())
        raw_rules = ev.get("matched_rule_ids")
        if isinstance(raw_rules, (list, tuple)):
            contributing_rules.extend(str(r) for r in raw_rules if str(r).strip())
        raw_ml_id = ev.get("ml_model_id")
        if isinstance(raw_ml_id, str) and raw_ml_id.strip():
            ml_model_id = raw_ml_id.strip()

    mtf_aligned = bool(
        report.mtf_evaluation is not None
        and not report.mtf_evaluation.has_conflict
        and report.mtf_evaluation.aligned_direction == sig.direction
    )

    return EntryContextSnapshot(
        client_order_id=client_order_id.strip(),
        signal_id=signal_id.strip(),
        symbol=report.symbol,
        timeframe=report.timeframe,
        direction=sig.direction,
        strategy_id=sig.strategy_id,
        captured_at_utc=ts,
        regime=report.regime if isinstance(report.regime, MarketRegime) else MarketRegime.RANGING,
        killzone_session=killzone,
        mtf_aligned=mtf_aligned,
        buy_confluence_score=report.confluence.buy_score,
        sell_confluence_score=report.confluence.sell_score,
        signal_confidence=sig.confidence,
        contributing_strategy_ids=tuple(sorted(set(contributing_strats))),
        contributing_knowledge_rule_ids=tuple(sorted(set(contributing_rules))),
        ml_model_id=ml_model_id,
        ml_predicted_direction=ml_pred_dir,
        ml_buy_probability=ml_buy_prob,
        ml_sell_probability=ml_sell_prob,
        ml_wait_probability=ml_wait_prob,
        requested_entry_price=sig.entry_price,
        stop_loss=sig.stop_loss,
        take_profit=sig.take_profit,
        expected_spread_points=expected_spread_points,
        approved_volume=approved_volume,
        initial_risk_amount=initial_risk_amount,
    )


def compute_initial_risk_amount(
    *,
    entry_price: Decimal,
    stop_loss: Decimal,
    volume: Decimal,
    instrument_spec: InstrumentSpec | None = None,
) -> Decimal:
    """Compute deterministic 1R initial monetary risk for `(entry_price, stop_loss, volume)`."""
    stop_distance = abs(entry_price - stop_loss)
    if stop_distance <= Decimal("0") or volume <= Decimal("0"):
        raise TradeReconstructionError(
            f"Invalid stop_distance ({stop_distance}) or volume ({volume}) when computing initial_risk_amount."
        )
    if instrument_spec is not None:
        ticks_at_risk = stop_distance / instrument_spec.tick_size
        loss_per_lot = ticks_at_risk * instrument_spec.tick_value
        risk_val = volume * loss_per_lot
    else:
        # Fallback 100-unit contract specification if instrument_spec is omitted
        risk_val = volume * stop_distance * Decimal("100")

    quantized = risk_val.quantize(Decimal("0.01"))
    return quantized if quantized > Decimal("0.00") else Decimal("0.01")


class PersistentTradeJournal:
    """Crash-safe, OS-locked (`CrashSafeJsonPersistence`) post-trade journal storing:
    1. `EntryContextSnapshot` indexed by `client_order_id` and `signal_id`.
    2. `PartialExitProgressRecord` indexed by `client_order_id` for active partially closed positions.
    3. `CompletedTradeRecord` indexed by `trade_id` and `client_order_id`, enforcing cross-restart
       idempotency and deal-ticket uniqueness so no broker deal or closed position is double-counted.
    """

    def __init__(
        self,
        *,
        config: PostTradeAnalyticsConfig | None = None,
        persistence: CrashSafeJsonPersistence | None = None,
    ) -> None:
        self._config = config or PostTradeAnalyticsConfig()
        self._persistence = persistence or CrashSafeJsonPersistence(
            self._config.journal_path,
            lock_timeout_seconds=self._config.lock_timeout_seconds,
        )
        self._mutex = threading.RLock()
        self._loaded_account: tuple[int, str] | None = None
        self._entry_contexts: dict[str, EntryContextSnapshot] = {}
        self._signal_to_client_id: dict[str, str] = {}
        self._partial_exits: dict[str, PartialExitProgressRecord] = {}
        self._completed_trades: dict[str, CompletedTradeRecord] = {}
        self._client_id_to_trade_id: dict[str, str] = {}
        self._position_ticket_to_trade_id: dict[int, str] = {}
        self._consumed_deal_tickets: dict[int, str] = {}
        self._staged_reports: dict[str, StrategyAnalysisReport] = {}
        self._staged_contexts: dict[str, EntryContextSnapshot] = {}
        self._bound_engine: "MT5DemoExecutionEngine | None" = None
        self._require_staged_context: bool = False
        self._chained_before_send_hook: Callable[[ExecutionLedgerEntry], None] | None = None
        self._last_generation: int = 0

    @property
    def config(self) -> PostTradeAnalyticsConfig:
        return self._config

    @property
    def persistence(self) -> CrashSafeJsonPersistence:
        return self._persistence

    @property
    def last_generation(self) -> int:
        return self._last_generation

    def get_entry_context(
        self,
        client_order_id: str | None = None,
        *,
        signal_id: str | None = None,
    ) -> EntryContextSnapshot | None:
        """Look up a persisted `EntryContextSnapshot` by `client_order_id` or `signal_id`."""
        with self._mutex:
            if client_order_id is not None:
                cid = client_order_id.strip()
                if cid in self._entry_contexts:
                    return self._entry_contexts[cid]
            if signal_id is not None:
                sid = signal_id.strip()
                mapped_cid = self._signal_to_client_id.get(sid)
                if mapped_cid and mapped_cid in self._entry_contexts:
                    return self._entry_contexts[mapped_cid]
            return None

    def get_partial_exit_progress(
        self,
        client_order_id: str,
    ) -> PartialExitProgressRecord | None:
        with self._mutex:
            return self._partial_exits.get(client_order_id.strip())

    def list_partial_exit_progress(self) -> tuple[PartialExitProgressRecord, ...]:
        with self._mutex:
            return tuple(
                sorted(
                    self._partial_exits.values(),
                    key=lambda item: (item.last_exit_time_utc, item.client_order_id),
                )
            )

    def get_completed_trade(self, trade_id: str) -> CompletedTradeRecord | None:
        with self._mutex:
            return self._completed_trades.get(trade_id.strip())

    def get_completed_trade_by_client_order_id(
        self,
        client_order_id: str,
    ) -> CompletedTradeRecord | None:
        with self._mutex:
            tid = self._client_id_to_trade_id.get(client_order_id.strip())
            return self._completed_trades.get(tid) if tid is not None else None

    def get_completed_trade_by_position_ticket(
        self,
        position_ticket: int,
    ) -> CompletedTradeRecord | None:
        with self._mutex:
            tid = self._position_ticket_to_trade_id.get(position_ticket)
            return self._completed_trades.get(tid) if tid is not None else None

    def list_completed_trades(self) -> tuple[CompletedTradeRecord, ...]:
        with self._mutex:
            return tuple(
                sorted(
                    self._completed_trades.values(),
                    key=lambda item: (item.exit_timestamp_utc, item.trade_id),
                )
            )

    def _serialize_payload(self) -> dict[str, Any]:
        return {
            "entry_contexts": [
                ctx.model_dump(mode="json")
                for ctx in sorted(
                    self._entry_contexts.values(),
                    key=lambda item: (item.captured_at_utc, item.client_order_id),
                )
            ],
            "partial_exits": [
                pe.model_dump(mode="json")
                for pe in sorted(
                    self._partial_exits.values(),
                    key=lambda item: (item.last_exit_time_utc, item.client_order_id),
                )
            ],
            "completed_trades": [
                tr.model_dump(mode="json")
                for tr in sorted(
                    self._completed_trades.values(),
                    key=lambda item: (item.exit_timestamp_utc, item.trade_id),
                )
            ],
        }

    def _rebuild_indices(
        self,
        *,
        contexts: Sequence[EntryContextSnapshot],
        partials: Sequence[PartialExitProgressRecord],
        trades: Sequence[CompletedTradeRecord],
    ) -> None:
        ctx_map: dict[str, EntryContextSnapshot] = {}
        sig_to_cid: dict[str, str] = {}
        for ctx in contexts:
            if ctx.client_order_id in ctx_map:
                raise PersistenceCorruptionError(
                    f"Duplicate client_order_id '{ctx.client_order_id}' in persisted entry_contexts."
                )
            ctx_map[ctx.client_order_id] = ctx
            sig_to_cid[ctx.signal_id] = ctx.client_order_id

        partial_map: dict[str, PartialExitProgressRecord] = {}
        for pe in partials:
            if pe.client_order_id in partial_map:
                raise PersistenceCorruptionError(
                    f"Duplicate client_order_id '{pe.client_order_id}' in persisted partial_exits."
                )
            partial_map[pe.client_order_id] = pe

        trade_map: dict[str, CompletedTradeRecord] = {}
        cid_to_tid: dict[str, str] = {}
        pos_to_tid: dict[int, str] = {}
        deal_to_tid: dict[int, str] = {}

        for tr in trades:
            if tr.trade_id in trade_map:
                raise PersistenceCorruptionError(
                    f"Duplicate trade_id '{tr.trade_id}' in persisted completed_trades."
                )
            if tr.client_order_id in cid_to_tid:
                raise PersistenceCorruptionError(
                    f"Duplicate client_order_id '{tr.client_order_id}' in persisted completed_trades."
                )
            if tr.position_ticket is not None:
                if tr.position_ticket in pos_to_tid:
                    raise PersistenceCorruptionError(
                        f"Duplicate position_ticket {tr.position_ticket} in persisted completed_trades."
                    )
                pos_to_tid[tr.position_ticket] = tr.trade_id

            all_tickets = (
                *tr.entry_deal_tickets,
                *tr.exit_deal_tickets,
                *tr.commission_deal_tickets,
            )
            if len(set(all_tickets)) != len(all_tickets):
                raise PersistenceCorruptionError(
                    f"CompletedTradeRecord '{tr.trade_id}' contains duplicate deal tickets within itself."
                )
            for d_tkt in all_tickets:
                if d_tkt in deal_to_tid:
                    raise PersistenceCorruptionError(
                        f"Deal ticket {d_tkt} is claimed by multiple completed trades "
                        f"('{deal_to_tid[d_tkt]}' and '{tr.trade_id}')."
                    )
                deal_to_tid[d_tkt] = tr.trade_id

            trade_map[tr.trade_id] = tr
            cid_to_tid[tr.client_order_id] = tr.trade_id

        self._entry_contexts = ctx_map
        self._signal_to_client_id = sig_to_cid
        self._partial_exits = partial_map
        self._completed_trades = trade_map
        self._client_id_to_trade_id = cid_to_tid
        self._position_ticket_to_trade_id = pos_to_tid
        self._consumed_deal_tickets = deal_to_tid

    def load_or_initialize(
        self,
        *,
        account_login: int,
        server: str,
    ) -> int:
        """Load and verify the persisted trade journal from disk under an exclusive OS file lock.

        Fails closed (`LedgerRecoveryLockError`, `PersistenceCorruptionError`,
        `PersistenceAccountMismatchError`, `PersistenceRollbackDetectedError`) if the journal
        file is corrupted, rolled back, or has interrupted `.tmp.*` writes.
        """
        cleaned_server = server.strip()
        with self._mutex:
            env = self._persistence.load(
                expected_login=account_login,
                expected_server=cleaned_server,
            )
            self._loaded_account = (account_login, cleaned_server)
            if env is None:
                self._rebuild_indices(contexts=(), partials=(), trades=())
                self._last_generation = 0
                return 0

            payload = env.payload
            raw_contexts = payload.get("entry_contexts", [])
            raw_partials = payload.get("partial_exits", [])
            raw_trades = payload.get("completed_trades", [])

            if (
                not isinstance(raw_contexts, list)
                or not isinstance(raw_partials, list)
                or not isinstance(raw_trades, list)
            ):
                raise PersistenceCorruptionError(
                    "Trade journal payload must contain 'entry_contexts', 'partial_exits', and 'completed_trades' lists."
                )

            try:
                contexts = [EntryContextSnapshot.model_validate(item) for item in raw_contexts]
                partials = [PartialExitProgressRecord.model_validate(item) for item in raw_partials]
                trades = [CompletedTradeRecord.model_validate(item) for item in raw_trades]
            except ValidationError as exc:
                raise PersistenceCorruptionError(
                    f"Trade journal schema validation failed: {exc}"
                ) from exc

            self._rebuild_indices(contexts=contexts, partials=partials, trades=trades)
            self._last_generation = env.generation
            return self._last_generation

    def _ensure_loaded(self, *, account_login: int, server: str) -> None:
        cleaned_server = server.strip()
        if self._loaded_account != (account_login, cleaned_server):
            self.load_or_initialize(account_login=account_login, server=cleaned_server)

    def record_entry_context(
        self,
        snapshot: EntryContextSnapshot,
        *,
        account_login: int,
        server: str,
        now_utc: datetime | None = None,
        allow_sizing_rebind: bool = False,
    ) -> EntryContextSnapshot:
        """Durably persist an `EntryContextSnapshot` bound to `(client_order_id, signal_id)`.

        Idempotent if an identical `EntryContextSnapshot` is already recorded for `client_order_id`;
        raises `TradeReconstructionError` if a conflicting snapshot is provided unless
        `allow_sizing_rebind=True` (used by the pre-send execution hook to enforce actual
        Phase 7 approved volume, live price, and initial risk amount).
        """
        with self._mutex:
            self._ensure_loaded(account_login=account_login, server=server)
            cid = snapshot.client_order_id
            if cid in self._entry_contexts:
                existing = self._entry_contexts[cid]
                if existing == snapshot:
                    return existing
                if not allow_sizing_rebind:
                    raise TradeReconstructionError(
                        f"Conflicting EntryContextSnapshot already exists for client_order_id='{cid}'."
                    )

            # Also check if a prior snapshot was keyed under a placeholder client_order_id for the same signal_id
            prior_cid = self._signal_to_client_id.get(snapshot.signal_id)
            if prior_cid is not None and prior_cid != cid and prior_cid in self._entry_contexts:
                if not allow_sizing_rebind:
                    raise TradeReconstructionError(
                        f"Conflicting EntryContextSnapshot already exists for signal_id='{snapshot.signal_id}' "
                        f"under client_order_id='{prior_cid}'."
                    )
                self._entry_contexts.pop(prior_cid, None)

            self._entry_contexts[cid] = snapshot
            self._signal_to_client_id[snapshot.signal_id] = cid

            # If a CompletedTradeRecord was already recorded without context, attach it deterministically
            if cid in self._client_id_to_trade_id:
                tid = self._client_id_to_trade_id[cid]
                existing_trade = self._completed_trades[tid]
                if existing_trade.entry_context is None:
                    self._completed_trades[tid] = existing_trade.model_copy(
                        update={
                            "entry_context": snapshot,
                            "context_attribution_status": ContextAttributionStatus.VERIFIED_ENTRY_CONTEXT,
                        }
                    )

            saved_env = self._persistence.save(
                self._serialize_payload(),
                account_login=account_login,
                server=server.strip(),
                now_utc=now_utc or snapshot.captured_at_utc,
            )
            self._last_generation = saved_env.generation
            return snapshot

    def stage_analysis_report(
        self,
        report_or_signal_id: StrategyAnalysisReport | str,
        report: StrategyAnalysisReport | None = None,
        *,
        signal_id: str | None = None,
        killzone_session: KillZoneSession | None = None,
        contributing_knowledge_rule_ids: Sequence[str] | None = None,
        ml_model_id: str | None = None,
        ml_predicted_direction: SignalDirection | None = None,
        ml_buy_probability: Decimal | None = None,
        ml_sell_probability: Decimal | None = None,
        ml_wait_probability: Decimal | None = None,
    ) -> None:
        """Stage a causal `StrategyAnalysisReport` prior to calling `engine.execute_signal()`.

        Can be called either as `stage_analysis_report(report, ...)` (matching by `(symbol, timeframe, direction)`
        or `report.lifecycle_record.signal_id`) or `stage_analysis_report(signal_id, report, ...)`.
        When `engine.execute_signal()` runs, the pre-send hook combines this staged report with the
        exact Phase 7 approved volume, live execution price, `client_order_id`, and 1R `initial_risk_amount`,
        and persists the verified `EntryContextSnapshot` to disk before `backend.order_send()` is called.
        """
        if isinstance(report_or_signal_id, StrategyAnalysisReport):
            actual_report = report_or_signal_id
            cleaned_sid = (
                signal_id.strip()
                if signal_id
                else (
                    actual_report.lifecycle_record.signal_id
                    if actual_report.lifecycle_record is not None
                    else f"__setup__:{actual_report.symbol}:{actual_report.timeframe.value}:{actual_report.signal.direction.value if actual_report.signal else 'NONE'}"
                )
            )
        else:
            if report is None:
                raise TradeReconstructionError("report must be provided when signal_id is passed as first argument.")
            actual_report = report
            cleaned_sid = report_or_signal_id.strip()

        if not cleaned_sid:
            raise TradeReconstructionError("signal_id must be a non-empty string when staging analysis report.")
        setup_key = (
            f"__setup__:{actual_report.symbol}:{actual_report.timeframe.value}:"
            f"{actual_report.signal.direction.value if actual_report.signal else 'NONE'}"
        )
        with self._mutex:
            self._staged_reports[cleaned_sid] = actual_report
            self._staged_reports[setup_key] = actual_report
            # Also store any explicit metadata overrides as a staged prototype snapshot
            proto = build_entry_context_snapshot(
                actual_report,
                client_order_id=f"staged-{cleaned_sid}",
                signal_id=cleaned_sid,
                approved_volume=Decimal("0.01"),
                initial_risk_amount=Decimal("0.01"),
            )
            updates: dict[str, Any] = {}
            if killzone_session is not None:
                updates["killzone_session"] = killzone_session
            if contributing_knowledge_rule_ids is not None:
                updates["contributing_knowledge_rule_ids"] = tuple(contributing_knowledge_rule_ids)
            if ml_model_id is not None:
                updates["ml_model_id"] = ml_model_id
            if ml_predicted_direction is not None:
                updates["ml_predicted_direction"] = ml_predicted_direction
            if ml_buy_probability is not None:
                updates["ml_buy_probability"] = ml_buy_probability
            if ml_sell_probability is not None:
                updates["ml_sell_probability"] = ml_sell_probability
            if ml_wait_probability is not None:
                updates["ml_wait_probability"] = ml_wait_probability
            if updates:
                proto = proto.model_copy(update=updates)
            self._staged_contexts[cleaned_sid] = proto
            self._staged_contexts[setup_key] = proto

    def stage_entry_context(self, snapshot: EntryContextSnapshot) -> None:
        """Stage a prototype `EntryContextSnapshot` prior to `engine.execute_signal()`.

        The pre-send hook will bind it to the actual `client_order_id`, `approved_volume`,
        `requested_entry_price`, and `initial_risk_amount` approved by `PreTradeSafetyGate`
        and persist it to disk before `backend.order_send()` executes.
        """
        with self._mutex:
            self._staged_contexts[snapshot.signal_id] = snapshot
            self._staged_contexts[snapshot.client_order_id] = snapshot

    def bind_execution_engine(
        self,
        engine: "MT5DemoExecutionEngine",
        *,
        require_staged_context: bool = False,
        chained_hook: Callable[[ExecutionLedgerEntry], None] | None = None,
    ) -> None:
        """Install a mandatory pre-send persistence guard hook on `MT5DemoExecutionEngine`
        guaranteeing that `EntryContextSnapshot` is durably persisted to disk with the exact
        approved volume and initial monetary risk before `backend.order_send()` can execute.
        """
        with self._mutex:
            self._bound_engine = engine
            self._require_staged_context = require_staged_context
            self._chained_before_send_hook = chained_hook
            engine.set_after_submitting_before_send_hook(self._pre_send_execution_hook)

    def unbind_execution_engine(self, engine: "MT5DemoExecutionEngine | None" = None) -> None:
        """Remove the pre-send persistence hook from `engine`."""
        with self._mutex:
            target = engine or self._bound_engine
            if target is not None:
                target.set_after_submitting_before_send_hook(self._chained_before_send_hook)
            self._bound_engine = None
            self._require_staged_context = False

    def _pre_send_execution_hook(self, submitting_entry: ExecutionLedgerEntry) -> None:
        """Mandatory pre-send hook invoked by `MT5DemoExecutionEngine.execute_signal()` after
        `PreTradeSafetyGate` validation and `SUBMITTING` persistence, and before `backend.order_send()`.
        """
        engine = self._bound_engine
        if engine is None:
            return

        cid = submitting_entry.client_order_id
        sig_id = submitting_entry.signal_id
        approved_vol = submitting_entry.requested_volume
        live_entry_px = submitting_entry.requested_price
        sl_px = submitting_entry.stop_loss
        tp_px = submitting_entry.take_profit
        now_utc = submitting_entry.updated_at_utc

        acct_info = engine.connection.backend.account_info()
        acct_login = acct_info.login if acct_info is not None else 0
        acct_server = acct_info.server if acct_info is not None else ""

        try:
            if acct_info is None or acct_login <= 0 or not acct_server.strip():
                raise TradeReconstructionError(
                    "Cannot persist pre-send EntryContextSnapshot: active MT5 account_info() is unavailable."
                )

            # Obtain the exact approved 1R initial_risk_amount from SignalLifecycleManager's RiskDecision
            # (or compute directly from InstrumentSpec and approved_vol)
            inst_spec: InstrumentSpec | None = None
            sym_point = Decimal("0.01")
            try:
                sym_info = engine.connection.get_symbol_info(submitting_entry.symbol)
                sym_point = sym_info.point
                inst_spec = engine.connection.get_instrument_spec(submitting_entry.symbol)
            except Exception:
                pass

            expected_spread_pts = Decimal("0")
            tick = engine.connection.backend.symbol_info_tick(submitting_entry.symbol)
            if tick is not None and sym_point > Decimal("0"):
                expected_spread_pts = tick.spread_points(sym_point)

            strategy_id = "mt5_demo_direct"
            signal_conf = Decimal("0.50")
            approved_risk_amount: Decimal | None = None
            ev_dict: dict[str, object] = {}

            if engine.lifecycle_manager is not None:
                try:
                    managed_rec = engine.lifecycle_manager.get_record(sig_id)
                    strategy_id = managed_rec.signal.strategy_id
                    signal_conf = managed_rec.signal.confidence
                    ev_dict = dict(managed_rec.supporting_evidence)
                    if (
                        managed_rec.risk_decision is not None
                        and managed_rec.risk_decision.approved
                        and managed_rec.risk_decision.calculated_position_size == approved_vol
                        and managed_rec.risk_decision.calculated_risk_amount > Decimal("0")
                    ):
                        approved_risk_amount = managed_rec.risk_decision.calculated_risk_amount
                except Exception:
                    pass

            if approved_risk_amount is None:
                approved_risk_amount = compute_initial_risk_amount(
                    entry_price=live_entry_px,
                    stop_loss=sl_px,
                    volume=approved_vol,
                    instrument_spec=inst_spec,
                )

            with self._mutex:
                setup_key = (
                    f"__setup__:{submitting_entry.symbol}:{submitting_entry.timeframe.value}:"
                    f"{submitting_entry.direction.value}"
                )
                staged_ctx = (
                    self._staged_contexts.get(cid)
                    or self._staged_contexts.get(sig_id)
                    or self._staged_contexts.get(setup_key)
                    or self.get_entry_context(client_order_id=cid, signal_id=sig_id)
                )
                staged_rep = (
                    self._staged_reports.get(sig_id)
                    or self._staged_reports.get(cid)
                    or self._staged_reports.get(setup_key)
                )

                if self._require_staged_context and staged_ctx is None and staged_rep is None:
                    raise TradeReconstructionError(
                        f"Direct order execution blocked before order_send() (require_staged_context=True): "
                        f"mandatory EntryContextSnapshot or StrategyAnalysisReport was not staged for "
                        f"signal_id='{sig_id}' (client_order_id='{cid}')."
                    )

                if staged_ctx is not None:
                    final_snapshot = staged_ctx.model_copy(
                        update={
                            "client_order_id": cid,
                            "signal_id": sig_id,
                            "symbol": submitting_entry.symbol,
                            "timeframe": submitting_entry.timeframe,
                            "direction": submitting_entry.direction,
                            "captured_at_utc": now_utc,
                            "requested_entry_price": live_entry_px,
                            "stop_loss": sl_px,
                            "take_profit": tp_px,
                            "expected_spread_points": (
                                staged_ctx.expected_spread_points
                                if staged_ctx.expected_spread_points > Decimal("0")
                                else expected_spread_pts
                            ),
                            "approved_volume": approved_vol,
                            "initial_risk_amount": approved_risk_amount,
                        }
                    )
                elif staged_rep is not None:
                    final_snapshot = build_entry_context_snapshot(
                        staged_rep,
                        client_order_id=cid,
                        signal_id=sig_id,
                        approved_volume=approved_vol,
                        initial_risk_amount=approved_risk_amount,
                        expected_spread_points=expected_spread_pts,
                        captured_at_utc=now_utc,
                    ).model_copy(
                        update={
                            "requested_entry_price": live_entry_px,
                            "stop_loss": sl_px,
                            "take_profit": tp_px,
                        }
                    )
                else:
                    raw_regime = ev_dict.get("regime")
                    regime = (
                        MarketRegime(raw_regime)
                        if isinstance(raw_regime, str) and raw_regime in MarketRegime.__members__
                        else MarketRegime.RANGING
                    )
                    raw_kz = ev_dict.get("killzone_session")
                    kz = (
                        KillZoneSession(raw_kz)
                        if isinstance(raw_kz, str) and raw_kz in KillZoneSession.__members__
                        else KillZoneSession.OFF_SESSION
                    )
                    final_snapshot = EntryContextSnapshot(
                        client_order_id=cid,
                        signal_id=sig_id,
                        symbol=submitting_entry.symbol,
                        timeframe=submitting_entry.timeframe,
                        direction=submitting_entry.direction,
                        strategy_id=strategy_id,
                        captured_at_utc=now_utc,
                        regime=regime,
                        killzone_session=kz,
                        mtf_aligned=bool(ev_dict.get("mtf_aligned", False)),
                        buy_confluence_score=Decimal(str(ev_dict.get("buy_score", "0"))),
                        sell_confluence_score=Decimal(str(ev_dict.get("sell_score", "0"))),
                        signal_confidence=signal_conf,
                        requested_entry_price=live_entry_px,
                        stop_loss=sl_px,
                        take_profit=tp_px,
                        expected_spread_points=expected_spread_pts,
                        approved_volume=approved_vol,
                        initial_risk_amount=approved_risk_amount,
                    )

                self.record_entry_context(
                    final_snapshot,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                    allow_sizing_rebind=True,
                )
                self._staged_contexts.pop(sig_id, None)
                self._staged_contexts.pop(cid, None)
                self._staged_contexts.pop(setup_key, None)
                self._staged_reports.pop(sig_id, None)
                self._staged_reports.pop(cid, None)
                self._staged_reports.pop(setup_key, None)

        except Exception as exc:
            # Safely abort before order_send(): transition SUBMITTING -> FAILED and cancel signal lifecycle
            if acct_login > 0 and acct_server.strip():
                try:
                    engine.ledger.transition_entry(
                        cid,
                        OrderLedgerStatus.FAILED,
                        account_login=acct_login,
                        server=acct_server,
                        now_utc=now_utc,
                        executed_volume=Decimal("0"),
                        remaining_volume=approved_vol,
                        reconciliation_status=ReconciliationStatus.NOT_REQUIRED,
                        synchronous_broker_rejection=True,
                        reason=(
                            f"Aborted before order_send() due to mandatory EntryContextSnapshot "
                            f"persistence failure: {type(exc).__name__}: {exc}"
                        ),
                    )
                except Exception:
                    pass
            if engine.lifecycle_manager is not None:
                try:
                    rec = engine.lifecycle_manager.get_record(sig_id)
                    if rec.state in (
                        SignalLifecycleState.GENERATED,
                        SignalLifecycleState.PENDING_CONFIRMATION,
                        SignalLifecycleState.CONFIRMED,
                        SignalLifecycleState.RISK_APPROVED,
                    ):
                        engine.lifecycle_manager.transition(
                            sig_id,
                            SignalLifecycleState.CANCELLED,
                            now_utc,
                            f"Aborted before order_send() due to EntryContextSnapshot persistence failure: {exc}",
                        )
                except Exception:
                    pass

            if isinstance(exc, (TradeReconstructionError, PersistenceError)):
                raise
            raise TradeReconstructionError(
                f"Mandatory EntryContextSnapshot persistence failed before order_send(): {exc}"
            ) from exc

        if self._chained_before_send_hook is not None:
            self._chained_before_send_hook(submitting_entry)

    def record_partial_exit_progress(
        self,
        progress: PartialExitProgressRecord,
        *,
        account_login: int,
        server: str,
        now_utc: datetime | None = None,
    ) -> PartialExitProgressRecord:
        """Durably persist or update `PartialExitProgressRecord` for an open partially closed position."""
        with self._mutex:
            self._ensure_loaded(account_login=account_login, server=server)
            cid = progress.client_order_id
            if cid in self._client_id_to_trade_id:
                raise TradeReconstructionError(
                    f"Cannot record partial exit for '{cid}' because it is already finalized as a CompletedTradeRecord."
                )
            existing = self._partial_exits.get(cid)
            if existing == progress:
                return existing

            self._partial_exits[cid] = progress
            saved_env = self._persistence.save(
                self._serialize_payload(),
                account_login=account_login,
                server=server.strip(),
                now_utc=now_utc or progress.last_exit_time_utc,
            )
            self._last_generation = saved_env.generation
            return progress

    def record_completed_trade(
        self,
        trade: CompletedTradeRecord,
        *,
        account_login: int,
        server: str,
        now_utc: datetime | None = None,
    ) -> CompletedTradeRecord:
        """Durably persist a verified `CompletedTradeRecord`, enforcing cross-restart idempotency
        and deal-ticket uniqueness across all journaled trades.
        """
        with self._mutex:
            self._ensure_loaded(account_login=account_login, server=server)

            # Attach stored EntryContextSnapshot if available and not yet populated on `trade`
            stored_ctx = self.get_entry_context(
                client_order_id=trade.client_order_id,
                signal_id=trade.signal_id,
            )
            if stored_ctx is not None and trade.entry_context is None:
                trade = trade.model_copy(
                    update={
                        "entry_context": stored_ctx,
                        "context_attribution_status": ContextAttributionStatus.VERIFIED_ENTRY_CONTEXT,
                    }
                )

            # Idempotency check if already recorded
            if trade.trade_id in self._completed_trades:
                existing = self._completed_trades[trade.trade_id]
                if existing == trade:
                    return existing
                raise TradeReconstructionError(
                    f"Conflicting CompletedTradeRecord already exists for trade_id='{trade.trade_id}'."
                )

            if trade.client_order_id in self._client_id_to_trade_id:
                existing_tid = self._client_id_to_trade_id[trade.client_order_id]
                existing_tr = self._completed_trades[existing_tid]
                if existing_tr == trade:
                    return existing_tr
                raise TradeReconstructionError(
                    f"client_order_id '{trade.client_order_id}' is already finalized under trade_id='{existing_tid}'."
                )

            if (
                trade.position_ticket is not None
                and trade.position_ticket in self._position_ticket_to_trade_id
            ):
                existing_tid = self._position_ticket_to_trade_id[trade.position_ticket]
                raise TradeReconstructionError(
                    f"position_ticket {trade.position_ticket} is already finalized under trade_id='{existing_tid}'."
                )

            all_tickets = (
                *trade.entry_deal_tickets,
                *trade.exit_deal_tickets,
                *trade.commission_deal_tickets,
            )
            if len(set(all_tickets)) != len(all_tickets):
                raise TradeReconstructionError(
                    f"CompletedTradeRecord '{trade.trade_id}' contains duplicate deal tickets within itself."
                )
            for d_tkt in all_tickets:
                if d_tkt in self._consumed_deal_tickets:
                    other_tid = self._consumed_deal_tickets[d_tkt]
                    raise TradeReconstructionError(
                        f"Broker deal ticket {d_tkt} is already claimed by completed trade '{other_tid}'; "
                        f"cannot double-count in '{trade.trade_id}'."
                    )

            self._completed_trades[trade.trade_id] = trade
            self._client_id_to_trade_id[trade.client_order_id] = trade.trade_id
            if trade.position_ticket is not None:
                self._position_ticket_to_trade_id[trade.position_ticket] = trade.trade_id
            for d_tkt in all_tickets:
                self._consumed_deal_tickets[d_tkt] = trade.trade_id

            # Clear any in-flight partial exit progress record now that the position is 100% closed
            self._partial_exits.pop(trade.client_order_id, None)

            saved_env = self._persistence.save(
                self._serialize_payload(),
                account_login=account_login,
                server=server.strip(),
                now_utc=now_utc or trade.exit_timestamp_utc,
            )
            self._last_generation = saved_env.generation
            return trade

    def is_deal_ticket_consumed(self, deal_ticket: int, *, exclude_trade_id: str | None = None) -> bool:
        """Return True if `deal_ticket` has already been consumed by a finalized `CompletedTradeRecord`."""
        with self._mutex:
            owner = self._consumed_deal_tickets.get(deal_ticket)
            if owner is None:
                return False
            return owner != exclude_trade_id
