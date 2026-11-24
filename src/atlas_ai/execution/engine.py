"""Safety-Gated MT5 Demo Execution Engine with Write-Ahead Ledger and Ambiguous Recovery for ATLAS AI Phase 7.4."""

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import threading
import time

from pydantic import BaseModel, ConfigDict, Field

from atlas_ai.core.logger import get_logger
from atlas_ai.domain.risk import RiskParameters
from atlas_ai.domain.signals import SignalDirection, TradeSignal
from atlas_ai.execution.account_state import PersistentAccountRiskTracker
from atlas_ai.execution.config import ExecutionSafetyConfig
from atlas_ai.execution.connection import MT5ConnectionAdapter
from atlas_ai.execution.exceptions import (
    LiveTradingBlockedError,
    MT5AuthenticationError,
    OrderValidationError,
    StaleMarketDataError,
)
from atlas_ai.execution.ledger import (
    EPOCH_UTC,
    LedgerReconciliationReport,
    PersistentExecutionLedger,
)
from atlas_ai.execution.market_data import MT5MarketDataAdapter
from atlas_ai.execution.models import (
    AMBIGUOUS_SUBMISSION_RETCODES,
    DETERMINISTIC_REJECTION_RETCODES,
    ExecutionLedgerEntry,
    MT5DealEntry,
    MT5DealRecord,
    MT5DealType,
    MT5OrderRecord,
    MT5OrderSendResult,
    MT5PositionRecord,
    MT5Retcode,
    OrderLedgerStatus,
    ReconciliationStatus,
)
from atlas_ai.execution.safety import (
    PreTradeSafetyGate,
    PreTradeSafetyValidationResult,
)
from atlas_ai.risk.engine import RiskEngine
from atlas_ai.strategies.models import SignalLifecycleState
from atlas_ai.strategies.signal_lifecycle import SignalLifecycleManager

logger = get_logger("execution.engine")


class DemoExecutionResult(BaseModel):
    """Immutable outcome of a safety-gated MT5 Demo order execution attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    client_order_id: str = Field(..., min_length=1)
    signal_id: str = Field(..., min_length=1)
    executed: bool
    partially_filled: bool = Field(default=False)
    ambiguous_requires_reconciliation: bool = Field(default=False)
    ledger_entry: ExecutionLedgerEntry
    safety_validation: PreTradeSafetyValidationResult
    order_send_result: MT5OrderSendResult | None = Field(default=None)


class MT5DemoExecutionEngine:
    """Orchestrates safety-gated DEMO order execution with durable write-ahead persistence,
    complete MQL5 retcode handling, post-send broker verification, and ambiguous recovery.

    Core Invariants:
    1. Strictly DEMO-only (`trade_mode == 0`, `execution_mode == DEMO`, `live_trading_enabled == False`).
       Immediately shuts down the terminal connection if account mode or broker state becomes unsafe.
    2. Persists `CREATED` -> `VALIDATED` -> `SUBMITTING` to the write-ahead ledger before calling `order_send()`.
    3. Never blindly retries `order_send()` after a timeout (`10012`), connection loss (`10031`),
       `None` response, or exception; marks the intent `UNKNOWN` -> `RECONCILIATION_REQUIRED` and blocks
       all new submissions until `reconcile_unresolved_orders()` completes.
    4. Verifies actual broker positions and/or deals before marking an intent `FILLED` / `ACTIVE`
       or transitioning `SignalLifecycleManager` to `ACTIVE`.
    """

    def __init__(
        self,
        *,
        connection: MT5ConnectionAdapter,
        market_data: MT5MarketDataAdapter,
        account_tracker: PersistentAccountRiskTracker,
        ledger: PersistentExecutionLedger,
        risk_engine: RiskEngine,
        safety_config: ExecutionSafetyConfig | None = None,
        lifecycle_manager: SignalLifecycleManager | None = None,
        safety_gate: PreTradeSafetyGate | None = None,
    ) -> None:
        self._connection = connection
        self._market_data = market_data
        self._account_tracker = account_tracker
        self._ledger = ledger
        self._risk_engine = risk_engine
        self._safety_config = safety_config or ExecutionSafetyConfig()
        self._lifecycle_manager = lifecycle_manager
        self._safety_gate = safety_gate or PreTradeSafetyGate(
            connection=connection,
            market_data=market_data,
            account_tracker=account_tracker,
            ledger=ledger,
            risk_engine=risk_engine,
            safety_config=self._safety_config,
            lifecycle_manager=lifecycle_manager,
        )
        self._engine_mutex = threading.RLock()
        self._after_submitting_before_send_hook: Callable[[ExecutionLedgerEntry], None] | None = None

    @property
    def connection(self) -> MT5ConnectionAdapter:
        return self._connection

    @property
    def market_data(self) -> MT5MarketDataAdapter:
        return self._market_data

    @property
    def account_tracker(self) -> PersistentAccountRiskTracker:
        return self._account_tracker

    @property
    def ledger(self) -> PersistentExecutionLedger:
        return self._ledger

    @property
    def risk_engine(self) -> RiskEngine:
        return self._risk_engine

    @property
    def safety_config(self) -> ExecutionSafetyConfig:
        return self._safety_config

    @property
    def lifecycle_manager(self) -> SignalLifecycleManager | None:
        return self._lifecycle_manager

    @property
    def safety_gate(self) -> PreTradeSafetyGate:
        return self._safety_gate

    def set_after_submitting_before_send_hook(
        self,
        hook: Callable[[ExecutionLedgerEntry], None] | None,
    ) -> None:
        """Install an optional offline test hook executed immediately after `SUBMITTING` is persisted
        to disk and before `backend.order_send()` is invoked (to simulate crash between write-ahead and send).
        """
        self._after_submitting_before_send_hook = hook

    def execute_signal(
        self,
        signal: TradeSignal,
        *,
        signal_id: str,
        client_order_id: str | None = None,
        requested_volume: Decimal | None = None,
        risk_parameters: RiskParameters | None = None,
        reference_now_utc: datetime | None = None,
    ) -> DemoExecutionResult:
        """Validate, write-ahead-persist, submit, and broker-verify a DEMO order for `signal`."""
        start_monotonic = time.monotonic()
        now_utc = (
            reference_now_utc.astimezone(timezone.utc)
            if (reference_now_utc is not None and reference_now_utc.tzinfo is not None)
            else (
                reference_now_utc.replace(tzinfo=timezone.utc)
                if reference_now_utc is not None
                else datetime.now(timezone.utc)
            )
        )

        with self._engine_mutex:
            # Step 1: Run all 14 deterministic pre-trade safety checks
            validation = self._safety_gate.validate_and_build_order(
                signal,
                signal_id=signal_id,
                client_order_id=client_order_id,
                requested_volume=requested_volume,
                risk_parameters=risk_parameters,
                reference_now_utc=now_utc,
            )
            cid = validation.client_order_id
            sig_id = validation.signal_id
            acct_login = validation.account_info.login
            acct_server = validation.account_info.server

            # Step 2: Persist order intent in CREATED, then transition to VALIDATED, then write-ahead SUBMITTING
            try:
                self._ledger.create_intent(
                    account_login=acct_login,
                    server=acct_server,
                    client_order_id=cid,
                    signal_id=sig_id,
                    symbol=validation.signal.symbol,
                    timeframe=validation.signal.timeframe,
                    direction=validation.signal.direction,
                    magic=validation.order_request.magic,
                    requested_volume=validation.approved_volume,
                    requested_price=validation.order_request.price,
                    stop_loss=validation.order_request.sl,
                    take_profit=validation.order_request.tp,
                    filling_mode=validation.filling_mode,
                    now_utc=now_utc,
                    reason="Order intent created prior to safety validation commit.",
                )
                self._ledger.transition_entry(
                    cid,
                    OrderLedgerStatus.VALIDATED,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                    reason=(
                        f"PreTradeSafetyGate passed all 14 checks: volume={validation.approved_volume}, "
                        f"spread={validation.spread_points}pts, filling={validation.filling_mode.name}."
                    ),
                )

                # Step 3: Persist durable write-ahead SUBMITTING state BEFORE calling order_send()
                submitting_entry = self._ledger.transition_entry(
                    cid,
                    OrderLedgerStatus.SUBMITTING,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                    reconciliation_status=ReconciliationStatus.PENDING,
                    reason="Durable write-ahead SUBMITTING intent persisted before order_send().",
                )
            except Exception as persist_exc:
                self._cancel_signal_if_approved(
                    sig_id,
                    now_utc,
                    f"Aborted before SUBMITTING persistence completed: {persist_exc}",
                )
                raise

            if self._after_submitting_before_send_hook is not None:
                self._after_submitting_before_send_hook(submitting_entry)

            # Re-verify DEMO account and TOCTOU fresh tick/spread/drift immediately before order_send()
            pre_send_now_utc = (
                now_utc + timedelta(seconds=max(0.0, time.monotonic() - start_monotonic))
                if reference_now_utc is not None
                else datetime.now(timezone.utc)
            )
            try:
                self._connection.verify_demo_account_only()
                self._verify_pre_send_market_conditions(
                    validation,
                    reference_now_utc=pre_send_now_utc,
                )
            except Exception as pre_send_exc:
                if isinstance(pre_send_exc, (LiveTradingBlockedError, MT5AuthenticationError)):
                    self._connection.disconnect()
                abort_status = (
                    OrderLedgerStatus.CANCELLED
                    if isinstance(pre_send_exc, (OrderValidationError, StaleMarketDataError))
                    else OrderLedgerStatus.FAILED
                )
                self._ledger.transition_entry(
                    cid,
                    abort_status,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=pre_send_now_utc,
                    executed_volume=Decimal("0"),
                    remaining_volume=validation.approved_volume,
                    reconciliation_status=ReconciliationStatus.NOT_REQUIRED,
                    synchronous_broker_rejection=True,
                    reason=f"Aborted before order_send(): {type(pre_send_exc).__name__}: {pre_send_exc}",
                )
                self._cancel_signal_if_approved(
                    sig_id,
                    pre_send_now_utc,
                    f"Aborted before order_send(): {pre_send_exc}",
                )
                raise

            # Step 4: Invoke backend.order_send() WITHOUT ever blindly retrying
            try:
                send_result = self._connection.backend.order_send(validation.order_request)
            except LiveTradingBlockedError as live_exc:
                self._connection.disconnect()
                self._ledger.transition_entry(
                    cid,
                    OrderLedgerStatus.FAILED,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                    reconciliation_status=ReconciliationStatus.NOT_REQUIRED,
                    synchronous_broker_rejection=True,
                    reason=f"Hardware-level demo guard blocked order_send: {live_exc}",
                )
                self._cancel_signal_if_approved(
                    sig_id,
                    now_utc,
                    f"Hardware-level demo guard blocked order_send: {live_exc}",
                )
                raise
            except Exception as send_exc:
                # Ambiguous exception during order_send -> mark UNKNOWN -> RECONCILIATION_REQUIRED
                self._ledger.transition_entry(
                    cid,
                    OrderLedgerStatus.UNKNOWN,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                    reconciliation_status=ReconciliationStatus.PENDING,
                    reason=f"Exception during order_send(): {type(send_exc).__name__}: {send_exc}",
                )
                recon_entry = self._ledger.transition_entry(
                    cid,
                    OrderLedgerStatus.RECONCILIATION_REQUIRED,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                    reconciliation_status=ReconciliationStatus.PENDING,
                    reason=(
                        f"Ambiguous order_send() exception ({type(send_exc).__name__}); "
                        "blocking new orders until broker reconciliation."
                    ),
                )
                return DemoExecutionResult(
                    client_order_id=cid,
                    signal_id=sig_id,
                    executed=False,
                    partially_filled=False,
                    ambiguous_requires_reconciliation=True,
                    ledger_entry=recon_entry,
                    safety_validation=validation,
                    order_send_result=None,
                )

            if send_result is None:
                self._ledger.transition_entry(
                    cid,
                    OrderLedgerStatus.UNKNOWN,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                    reconciliation_status=ReconciliationStatus.PENDING,
                    reason="MT5 order_send() returned None; execution outcome is unknown.",
                )
                recon_entry = self._ledger.transition_entry(
                    cid,
                    OrderLedgerStatus.RECONCILIATION_REQUIRED,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                    reconciliation_status=ReconciliationStatus.PENDING,
                    reason="MT5 order_send() returned None; broker reconciliation required.",
                )
                return DemoExecutionResult(
                    client_order_id=cid,
                    signal_id=sig_id,
                    executed=False,
                    partially_filled=False,
                    ambiguous_requires_reconciliation=True,
                    ledger_entry=recon_entry,
                    safety_validation=validation,
                    order_send_result=None,
                )

            retcode = send_result.retcode

            # Step 5: Handle live-only rejection retcode with immediate controlled shutdown
            if retcode == int(MT5Retcode.TRADE_RETCODE_ONLY_REAL):
                self._connection.disconnect()
                self._ledger.transition_entry(
                    cid,
                    OrderLedgerStatus.REJECTED,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                    retcode=retcode,
                    reconciliation_status=ReconciliationStatus.NOT_REQUIRED,
                    synchronous_broker_rejection=True,
                    reason="Broker returned TRADE_RETCODE_ONLY_REAL (10032); terminal disconnected.",
                )
                self._cancel_signal_if_approved(
                    sig_id,
                    now_utc,
                    "Broker returned TRADE_RETCODE_ONLY_REAL (10032).",
                )
                raise LiveTradingBlockedError(
                    "SAFETY VIOLATION: Broker returned TRADE_RETCODE_ONLY_REAL (10032); terminal shut down."
                )

            # Step 6: Handle ambiguous retcodes (10012 TIMEOUT, 10031 CONNECTION)
            if retcode in AMBIGUOUS_SUBMISSION_RETCODES:
                self._ledger.transition_entry(
                    cid,
                    OrderLedgerStatus.UNKNOWN,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                    retcode=retcode,
                    order_ticket=send_result.order if send_result.order > 0 else None,
                    deal_ticket=send_result.deal if send_result.deal > 0 else None,
                    reconciliation_status=ReconciliationStatus.PENDING,
                    reason=(
                        f"Ambiguous MT5 retcode={retcode} ({send_result.comment}); "
                        "order outcome on server is uncertain."
                    ),
                )
                recon_entry = self._ledger.transition_entry(
                    cid,
                    OrderLedgerStatus.RECONCILIATION_REQUIRED,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                    retcode=retcode,
                    reconciliation_status=ReconciliationStatus.PENDING,
                    reason=(
                        f"Broker reconciliation required after ambiguous retcode={retcode}; "
                        "new submissions remain blocked."
                    ),
                )
                return DemoExecutionResult(
                    client_order_id=cid,
                    signal_id=sig_id,
                    executed=False,
                    partially_filled=False,
                    ambiguous_requires_reconciliation=True,
                    ledger_entry=recon_entry,
                    safety_validation=validation,
                    order_send_result=send_result,
                )

            # Step 7: Handle full or partial execution retcodes (10009 DONE, 10010 DONE_PARTIAL, 10008 PLACED)
            if retcode in (
                int(MT5Retcode.TRADE_RETCODE_DONE),
                int(MT5Retcode.TRADE_RETCODE_DONE_PARTIAL),
                int(MT5Retcode.TRADE_RETCODE_PLACED),
            ):
                submitted_entry = self._ledger.transition_entry(
                    cid,
                    OrderLedgerStatus.SUBMITTED,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                    retcode=retcode,
                    order_ticket=send_result.order if send_result.order > 0 else None,
                    deal_ticket=send_result.deal if send_result.deal > 0 else None,
                    fill_price=send_result.price if send_result.price > Decimal("0") else None,
                    reconciliation_status=ReconciliationStatus.PENDING,
                    reason=f"Broker accepted request with retcode={retcode}; verifying position and deal.",
                )

                verified_entry = self._verify_broker_execution_after_accept(
                    submitted_entry,
                    send_result=send_result,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                )

                is_exec = verified_entry.status in (
                    OrderLedgerStatus.ACTIVE,
                    OrderLedgerStatus.FILLED,
                    OrderLedgerStatus.PARTIALLY_FILLED,
                )
                is_partial = verified_entry.status == OrderLedgerStatus.PARTIALLY_FILLED
                is_ambig = verified_entry.status in (
                    OrderLedgerStatus.RECONCILIATION_REQUIRED,
                    OrderLedgerStatus.UNKNOWN,
                    OrderLedgerStatus.UNKNOWN_PENDING_RECONCILIATION,
                )

                if is_exec and verified_entry.executed_volume > Decimal("0"):
                    self._activate_signal_if_approved(
                        sig_id,
                        now_utc,
                        (
                            f"MT5 DEMO execution verified ({verified_entry.status.value}): "
                            f"volume={verified_entry.executed_volume}/{verified_entry.requested_volume} lots "
                            f"at price={verified_entry.fill_price}, position={verified_entry.position_ticket}."
                        ),
                    )

                return DemoExecutionResult(
                    client_order_id=cid,
                    signal_id=sig_id,
                    executed=is_exec,
                    partially_filled=is_partial,
                    ambiguous_requires_reconciliation=is_ambig,
                    ledger_entry=verified_entry,
                    safety_validation=validation,
                    order_send_result=send_result,
                )

            # Step 8: Explicit synchronous broker rejection / cancellation retcode allowlist
            if retcode in DETERMINISTIC_REJECTION_RETCODES:
                target_status = (
                    OrderLedgerStatus.CANCELLED
                    if retcode == int(MT5Retcode.TRADE_RETCODE_CANCEL)
                    else OrderLedgerStatus.REJECTED
                )
                rejected_entry = self._ledger.transition_entry(
                    cid,
                    target_status,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                    retcode=retcode,
                    executed_volume=Decimal("0"),
                    remaining_volume=validation.approved_volume,
                    reconciliation_status=ReconciliationStatus.NOT_REQUIRED,
                    synchronous_broker_rejection=True,
                    reason=f"Broker rejected order request: retcode={retcode}, comment='{send_result.comment}'.",
                )
                self._cancel_signal_if_approved(
                    sig_id,
                    now_utc,
                    f"Broker rejected order (retcode={retcode}, comment='{send_result.comment}').",
                )
                return DemoExecutionResult(
                    client_order_id=cid,
                    signal_id=sig_id,
                    executed=False,
                    partially_filled=False,
                    ambiguous_requires_reconciliation=False,
                    ledger_entry=rejected_entry,
                    safety_validation=validation,
                    order_send_result=send_result,
                )

            # Step 9: Unknown, zero, negative, or undocumented retcode -> fail closed as UNKNOWN -> RECONCILIATION_REQUIRED
            self._ledger.transition_entry(
                cid,
                OrderLedgerStatus.UNKNOWN,
                account_login=acct_login,
                server=acct_server,
                now_utc=now_utc,
                retcode=retcode,
                order_ticket=send_result.order if send_result.order > 0 else None,
                deal_ticket=send_result.deal if send_result.deal > 0 else None,
                reconciliation_status=ReconciliationStatus.PENDING,
                reason=(
                    f"Unrecognized or non-deterministic MT5 retcode={retcode} ('{send_result.comment}'); "
                    "failing closed as UNKNOWN."
                ),
            )
            unknown_recon_entry = self._ledger.transition_entry(
                cid,
                OrderLedgerStatus.RECONCILIATION_REQUIRED,
                account_login=acct_login,
                server=acct_server,
                now_utc=now_utc,
                retcode=retcode,
                reconciliation_status=ReconciliationStatus.PENDING,
                reason=(
                    f"Broker reconciliation required for unrecognized retcode={retcode}; "
                    "new submissions remain blocked."
                ),
            )
            return DemoExecutionResult(
                client_order_id=cid,
                signal_id=sig_id,
                executed=False,
                partially_filled=False,
                ambiguous_requires_reconciliation=True,
                ledger_entry=unknown_recon_entry,
                safety_validation=validation,
                order_send_result=send_result,
            )

    def _verify_pre_send_market_conditions(
        self,
        validation: PreTradeSafetyValidationResult,
        *,
        reference_now_utc: datetime,
    ) -> None:
        """Re-check tick freshness, spread (in symbol_info.point), price drift, and stop/freeze levels
        after write-ahead persistence fsyncs and immediately prior to calling backend.order_send().
        """
        sig = validation.signal
        sym_info = validation.symbol_info
        assert sig.entry_price is not None and sig.stop_loss is not None and sig.take_profit is not None

        fresh_tick = self._market_data.fetch_latest_tick(
            sig.symbol,
            reference_now_utc=reference_now_utc,
        )
        fresh_spread_points = fresh_tick.spread_points(sym_info.point)
        if fresh_spread_points > self._safety_config.max_allowed_spread_points:
            raise OrderValidationError(
                f"Pre-send TOCTOU check failed: live Bid-Ask spread for '{sig.symbol}' "
                f"({fresh_spread_points} points) exceeds max_allowed_spread_points "
                f"({self._safety_config.max_allowed_spread_points})."
            )

        fresh_live_price = (
            fresh_tick.ask if sig.direction == SignalDirection.BUY else fresh_tick.bid
        )
        fresh_drift_points = abs(fresh_live_price - sig.entry_price) / sym_info.point
        if fresh_drift_points > Decimal(self._safety_config.max_entry_deviation_points):
            raise OrderValidationError(
                f"Pre-send TOCTOU check failed: live execution price ({fresh_live_price}) for "
                f"'{sig.symbol}' drifted {fresh_drift_points:.1f} points from signal.entry_price "
                f"({sig.entry_price}), exceeding max_entry_deviation_points "
                f"({self._safety_config.max_entry_deviation_points})."
            )

        if sig.direction == SignalDirection.BUY:
            if not (sig.stop_loss < fresh_live_price < sig.take_profit):
                raise OrderValidationError(
                    f"Pre-send TOCTOU check failed: BUY SL/TP invalid at fresh Ask ({fresh_live_price})."
                )
        else:
            if not (sig.take_profit < fresh_live_price < sig.stop_loss):
                raise OrderValidationError(
                    f"Pre-send TOCTOU check failed: SELL SL/TP invalid at fresh Bid ({fresh_live_price})."
                )

        min_level_points = max(sym_info.trade_stops_level, sym_info.trade_freeze_level)
        min_stop_distance = Decimal(min_level_points) * sym_info.point
        if (
            abs(fresh_live_price - sig.stop_loss) < min_stop_distance
            or abs(sig.take_profit - fresh_live_price) < min_stop_distance
        ):
            raise OrderValidationError(
                f"Pre-send TOCTOU check failed: stop/take-profit distance at fresh price ({fresh_live_price}) "
                f"violates broker stop/freeze level minimum distance ({min_stop_distance})."
            )

    def _verify_broker_execution_after_accept(
        self,
        entry: ExecutionLedgerEntry,
        *,
        send_result: MT5OrderSendResult,
        account_login: int,
        server: str,
        now_utc: datetime,
    ) -> ExecutionLedgerEntry:
        """Verify open position, active working order, and entry deal on the broker before marking
        an accepted order `FILLED` / `ACTIVE` or `PARTIALLY_FILLED`.
        """
        try:
            positions = self._connection.backend.positions_get(entry.symbol)
            orders = self._connection.backend.orders_get(entry.symbol)
            future_horizon_utc = max(
                now_utc + timedelta(days=3650),
                datetime(2100, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            )
            deals = self._connection.backend.history_deals_get(
                EPOCH_UTC,
                future_horizon_utc,
                symbol=entry.symbol,
            )
        except Exception as verify_exc:
            return self._ledger.transition_entry(
                entry.client_order_id,
                OrderLedgerStatus.RECONCILIATION_REQUIRED,
                account_login=account_login,
                server=server,
                now_utc=now_utc,
                reconciliation_status=ReconciliationStatus.PENDING,
                reason=(
                    f"Post-send broker verification raised {type(verify_exc).__name__} ({verify_exc}); "
                    "promoted to RECONCILIATION_REQUIRED."
                ),
            )

        if positions is None or orders is None or deals is None:
            return self._ledger.transition_entry(
                entry.client_order_id,
                OrderLedgerStatus.RECONCILIATION_REQUIRED,
                account_login=account_login,
                server=server,
                now_utc=now_utc,
                reconciliation_status=ReconciliationStatus.PENDING,
                reason=(
                    "Post-send broker verification could not read positions_get(), orders_get(), or "
                    "history_deals_get(); promoted to RECONCILIATION_REQUIRED."
                ),
            )

        # Collect tickets and identifiers already claimed by other ledger entries
        claimed_deal_tickets: set[int] = set()
        claimed_order_tickets: set[int] = set()
        claimed_pos_tickets: set[int] = set()
        other_entry_ids: set[str] = set()
        for existing in self._ledger.list_entries():
            if existing.client_order_id == entry.client_order_id:
                continue
            other_entry_ids.add(existing.client_order_id)
            other_entry_ids.add(existing.signal_id)
            if existing.deal_ticket:
                claimed_deal_tickets.add(existing.deal_ticket)
            if existing.order_ticket:
                claimed_order_tickets.add(existing.order_ticket)
            if existing.position_ticket:
                claimed_pos_tickets.add(existing.position_ticket)

        other_ids_frozen = frozenset(other_entry_ids)

        matched_deals: list[MT5DealRecord] = [
            d
            for d in deals
            if d.ticket not in claimed_deal_tickets
            and (d.order == 0 or d.order not in claimed_order_tickets)
            and d.deal_type in (int(MT5DealType.DEAL_TYPE_BUY), int(MT5DealType.DEAL_TYPE_SELL))
            and d.entry == int(MT5DealEntry.DEAL_ENTRY_IN)
            and PersistentExecutionLedger._matches_entry_on_broker(
                entry,
                comment=d.comment,
                magic=d.magic,
                symbol=d.symbol,
                direction=(
                    SignalDirection.BUY
                    if d.deal_type == int(MT5DealType.DEAL_TYPE_BUY)
                    else SignalDirection.SELL
                ),
                order_ticket=d.order,
                deal_ticket=d.ticket,
                position_ticket=d.position_id,
                other_entry_ids=other_ids_frozen,
            )
        ]
        matched_deal_pos_ids = {d.position_id for d in matched_deals if d.position_id > 0}
        matched_positions: list[MT5PositionRecord] = [
            p
            for p in positions
            if p.ticket not in claimed_pos_tickets
            and (
                (
                    p.magic == entry.magic
                    and p.symbol.strip().upper() == entry.symbol
                    and (SignalDirection.BUY if p.position_type == 0 else SignalDirection.SELL)
                    == entry.direction
                    and p.ticket in matched_deal_pos_ids
                )
                or PersistentExecutionLedger._matches_entry_on_broker(
                    entry,
                    comment=p.comment,
                    magic=p.magic,
                    symbol=p.symbol,
                    direction=(
                        SignalDirection.BUY if p.position_type == 0 else SignalDirection.SELL
                    ),
                    position_ticket=p.ticket,
                    other_entry_ids=other_ids_frozen,
                )
            )
        ]
        matched_orders: list[MT5OrderRecord] = [
            o
            for o in orders
            if o.ticket not in claimed_order_tickets
            and PersistentExecutionLedger._matches_entry_on_broker(
                entry,
                comment=o.comment,
                magic=o.magic,
                symbol=o.symbol,
                direction=(
                    SignalDirection.BUY if (o.order_type % 2 == 0) else SignalDirection.SELL
                ),
                order_ticket=o.ticket,
                position_ticket=o.position_id,
                other_entry_ids=other_ids_frozen,
            )
        ]

        if not matched_deals and not matched_positions:
            if matched_orders:
                work_ord = matched_orders[-1]
                return self._ledger.transition_entry(
                    entry.client_order_id,
                    OrderLedgerStatus.RECONCILIATION_REQUIRED,
                    account_login=account_login,
                    server=server,
                    now_utc=now_utc,
                    order_ticket=work_ord.ticket,
                    executed_volume=Decimal("0"),
                    remaining_volume=entry.requested_volume,
                    reconciliation_status=ReconciliationStatus.PENDING,
                    reason=(
                        f"Order '{entry.client_order_id}' placed and still working on broker "
                        f"(retcode={send_result.retcode}, order_ticket={work_ord.ticket}, "
                        f"volume_current={work_ord.volume_current}); awaiting fill or cancellation reconciliation."
                    ),
                )
            return self._ledger.transition_entry(
                entry.client_order_id,
                OrderLedgerStatus.RECONCILIATION_REQUIRED,
                account_login=account_login,
                server=server,
                now_utc=now_utc,
                reconciliation_status=ReconciliationStatus.PENDING,
                reason=(
                    f"Broker returned retcode={send_result.retcode}, but no matching position or deal "
                    f"was found on server for '{entry.client_order_id}'; marked RECONCILIATION_REQUIRED."
                ),
            )

        deal_vol = sum((d.volume for d in matched_deals), Decimal("0"))
        pos_vol = sum((p.volume for p in matched_positions), Decimal("0"))
        verified_vol = deal_vol if deal_vol > Decimal("0") else pos_vol

        if verified_vol <= Decimal("0"):
            return self._ledger.transition_entry(
                entry.client_order_id,
                OrderLedgerStatus.RECONCILIATION_REQUIRED,
                account_login=account_login,
                server=server,
                now_utc=now_utc,
                reconciliation_status=ReconciliationStatus.PENDING,
                reason="Verified broker volume is 0 after accept retcode; marked RECONCILIATION_REQUIRED.",
            )

        rem_vol = max(Decimal("0"), entry.requested_volume - verified_vol)
        fill_px = (
            matched_deals[-1].price
            if matched_deals
            else (
                matched_positions[-1].price_open
                if matched_positions
                else (send_result.price if send_result.price > Decimal("0") else entry.requested_price)
            )
        )
        ord_tkt = (
            send_result.order
            if send_result.order > 0
            else (
                matched_deals[-1].order
                if (matched_deals and matched_deals[-1].order > 0)
                else (matched_orders[-1].ticket if matched_orders else entry.order_ticket)
            )
        )
        dl_tkt = (
            send_result.deal
            if send_result.deal > 0
            else (matched_deals[-1].ticket if matched_deals else entry.deal_ticket)
        )
        pos_tkt = (
            matched_positions[-1].ticket
            if matched_positions
            else (
                matched_deals[-1].position_id
                if (matched_deals and matched_deals[-1].position_id > 0)
                else entry.position_ticket
            )
        )

        if (
            send_result.retcode == int(MT5Retcode.TRADE_RETCODE_DONE_PARTIAL)
            or verified_vol < entry.requested_volume
            or bool(matched_orders)
        ):
            return self._ledger.transition_entry(
                entry.client_order_id,
                OrderLedgerStatus.PARTIALLY_FILLED,
                account_login=account_login,
                server=server,
                now_utc=now_utc,
                executed_volume=verified_vol,
                remaining_volume=rem_vol,
                fill_price=fill_px,
                order_ticket=ord_tkt,
                deal_ticket=dl_tkt,
                position_ticket=pos_tkt,
                retcode=send_result.retcode,
                reconciliation_status=(
                    ReconciliationStatus.PENDING
                    if matched_orders
                    else ReconciliationStatus.RECONCILED_EXECUTED
                ),
                reason=(
                    f"Partial fill verified on broker: executed_volume={verified_vol}, "
                    f"remaining_volume={rem_vol}, fill_price={fill_px}."
                ),
            )

        if matched_positions:
            return self._ledger.transition_filled_to_active_atomic(
                entry.client_order_id,
                account_login=account_login,
                server=server,
                now_utc=now_utc,
                executed_volume=verified_vol,
                remaining_volume=Decimal("0"),
                fill_price=fill_px,
                order_ticket=ord_tkt,
                deal_ticket=dl_tkt,
                position_ticket=pos_tkt,
                retcode=send_result.retcode,
                filled_reason=(
                    f"Full fill verified on broker: executed_volume={verified_vol}, "
                    f"fill_price={fill_px}, deal_ticket={dl_tkt}, position_ticket={pos_tkt}."
                ),
                active_reason=f"Position ticket={pos_tkt} is ACTIVE on DEMO account.",
            )

        return self._ledger.transition_entry(
            entry.client_order_id,
            OrderLedgerStatus.FILLED,
            account_login=account_login,
            server=server,
            now_utc=now_utc,
            executed_volume=verified_vol,
            remaining_volume=Decimal("0"),
            fill_price=fill_px,
            order_ticket=ord_tkt,
            deal_ticket=dl_tkt,
            position_ticket=pos_tkt,
            retcode=send_result.retcode,
            reconciliation_status=ReconciliationStatus.RECONCILED_EXECUTED,
            reason=(
                f"Full fill verified on broker (position already closed): executed_volume={verified_vol}, "
                f"fill_price={fill_px}, deal_ticket={dl_tkt}, position_ticket={pos_tkt}."
            ),
        )

    def reconcile_unresolved_orders(
        self,
        *,
        reference_now_utc: datetime | None = None,
    ) -> LedgerReconciliationReport:
        """Reconcile any unresolved or recovery-locked ledger intents against live broker state
        and synchronize `SignalLifecycleManager` transitions (`ACTIVE` or `CANCELLED`).
        """
        now_utc = (
            reference_now_utc.astimezone(timezone.utc)
            if (reference_now_utc is not None and reference_now_utc.tzinfo is not None)
            else (
                reference_now_utc.replace(tzinfo=timezone.utc)
                if reference_now_utc is not None
                else datetime.now(timezone.utc)
            )
        )
        with self._engine_mutex:
            report = self._ledger.reconcile_with_broker(
                self._connection,
                reference_now_utc=now_utc,
            )

            if self._lifecycle_manager is not None:
                for cid in report.reconciled_executed_ids:
                    entry = self._ledger.get_entry(cid)
                    if entry is not None and entry.executed_volume > Decimal("0"):
                        self._activate_signal_if_approved(
                            entry.signal_id,
                            now_utc,
                            (
                                f"Broker reconciliation confirmed execution for '{cid}': "
                                f"volume={entry.executed_volume} at {entry.fill_price}."
                            ),
                        )
                for cid in report.reconciled_not_executed_ids:
                    entry = self._ledger.get_entry(cid)
                    if entry is not None:
                        self._cancel_signal_if_approved(
                            entry.signal_id,
                            now_utc,
                            f"Broker reconciliation confirmed order '{cid}' was not executed.",
                        )

            return report

    def _activate_signal_if_approved(
        self,
        signal_id: str,
        now_utc: datetime,
        reason: str,
    ) -> None:
        if self._lifecycle_manager is None:
            return
        try:
            rec = self._lifecycle_manager.get_record(signal_id)
        except Exception:
            return
        if rec.state == SignalLifecycleState.RISK_APPROVED:
            self._lifecycle_manager.transition(
                signal_id,
                SignalLifecycleState.ACTIVE,
                now_utc,
                reason,
            )

    def _cancel_signal_if_approved(
        self,
        signal_id: str,
        now_utc: datetime,
        reason: str,
    ) -> None:
        if self._lifecycle_manager is None:
            return
        try:
            rec = self._lifecycle_manager.get_record(signal_id)
        except Exception:
            return
        if rec.state in (
            SignalLifecycleState.GENERATED,
            SignalLifecycleState.PENDING_CONFIRMATION,
            SignalLifecycleState.CONFIRMED,
            SignalLifecycleState.RISK_APPROVED,
        ):
            self._lifecycle_manager.transition(
                signal_id,
                SignalLifecycleState.CANCELLED,
                now_utc,
                reason,
            )
