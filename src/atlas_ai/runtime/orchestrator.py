"""Closed-Loop Safety-Gated Demo Trading Orchestrator (`DemoTradingOrchestrator`) for ATLAS AI Phase 9.2."""

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
import threading

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from atlas_ai.analytics.config import PostTradeAnalyticsConfig
from atlas_ai.analytics.drift_and_governance import DriftDetectionEngine
from atlas_ai.analytics.journal import PersistentTradeJournal
from atlas_ai.analytics.models import DriftAndHealthReport
from atlas_ai.analytics.trade_reconstructor import DemoTradeReconstructor, TradeReconstructionReport
from atlas_ai.core.logger import get_logger
from atlas_ai.data.features import FeaturePipeline
from atlas_ai.domain.market import Timeframe
from atlas_ai.domain.risk import AccountState, RiskDecision, RiskParameters
from atlas_ai.domain.signals import SignalDirection, TradeSignal
from atlas_ai.execution.account_state import PersistentAccountRiskTracker
from atlas_ai.execution.connection import MT5ConnectionAdapter
from atlas_ai.execution.engine import DemoExecutionResult, MT5DemoExecutionEngine
from atlas_ai.execution.exceptions import (
    ExecutionError,
    LedgerRecoveryLockError,
    LiveTradingBlockedError,
    MT5AuthenticationError,
    MT5ConnectionError,
    OrderValidationError,
    PersistenceAccountMismatchError,
    PersistenceCorruptionError,
    PersistenceError,
    PersistenceRollbackDetectedError,
    StaleMarketDataError,
)
from atlas_ai.execution.ledger import LedgerReconciliationReport, PersistentExecutionLedger
from atlas_ai.execution.market_data import MT5MarketDataAdapter
from atlas_ai.execution.models import LedgerHealthStatus, OrderLedgerStatus
from atlas_ai.execution.persistence import CrashSafeJsonPersistence
from atlas_ai.risk.engine import RiskEngine
from atlas_ai.runtime.circuit_breaker import EmergencyStopController
from atlas_ai.runtime.config import DemoRuntimeConfig
from atlas_ai.runtime.exceptions import (
    CircuitBreakerLatchedError,
    EmergencyStopError,
    RuntimeOrchestrationError,
)
from atlas_ai.runtime.models import EmergencyStopReason, EmergencyStopState
from atlas_ai.strategies.engine import StrategyEngine
from atlas_ai.strategies.models import (
    EconomicEvent,
    RiskEvaluationStatus,
    SignalLifecycleState,
    StrategyAnalysisReport,
)
from atlas_ai.strategies.signal_lifecycle import SignalLifecycleManager

logger = get_logger("runtime.orchestrator")


def _normalize_utc(ts: datetime | None = None) -> datetime:
    if ts is None:
        return datetime.now(timezone.utc)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _validate_account_identity(account_login: int, server: str) -> tuple[int, str]:
    if isinstance(account_login, bool) or not isinstance(account_login, int) or account_login <= 0:
        raise RuntimeOrchestrationError(
            f"account_login must be a positive integer (> 0), got {account_login!r}."
        )
    if not isinstance(server, str) or not server.strip():
        raise RuntimeOrchestrationError("server must be a non-empty string.")
    return account_login, server.strip()


def _format_bar_key(symbol: str, timeframe: Timeframe) -> str:
    cleaned_symbol = symbol.strip().upper()
    if not cleaned_symbol or not cleaned_symbol.isalnum():
        raise RuntimeOrchestrationError(f"Invalid symbol for bar key: {symbol!r}.")
    return f"{cleaned_symbol}:{timeframe.value}"


class RuntimeCycleStep(StrEnum):
    """Canonical 14-step closed-loop cycle sequence enforced by `DemoTradingOrchestrator`."""

    STEP_01_VERIFY_CONNECTION = "01_VERIFY_CONNECTION"
    STEP_02_RECONCILE_UNRESOLVED_ORDERS = "02_RECONCILE_UNRESOLVED_ORDERS"
    STEP_03_SYNC_CLOSED_TRADES = "03_SYNC_CLOSED_TRADES"
    STEP_04_RECONSTRUCT_ACCOUNT_STATE = "04_RECONSTRUCT_ACCOUNT_STATE"
    STEP_05_EVALUATE_DRIFT_AND_LEDGER_HEALTH = "05_EVALUATE_DRIFT_AND_LEDGER_HEALTH"
    STEP_06_EVALUATE_EMERGENCY_STOP_STATE = "06_EVALUATE_EMERGENCY_STOP_STATE"
    STEP_07_ASSERT_TRADING_AUTHORIZED = "07_ASSERT_TRADING_AUTHORIZED"
    STEP_08_CHECK_BAR_CLOSE_DEDUPLICATION = "08_CHECK_BAR_CLOSE_DEDUPLICATION"
    STEP_09_FETCH_SYNCHRONIZED_MARKET_DATA = "09_FETCH_SYNCHRONIZED_MARKET_DATA"
    STEP_10_RUN_FEATURE_PIPELINE = "10_RUN_FEATURE_PIPELINE"
    STEP_11_RUN_STRATEGY_ENGINE = "11_RUN_STRATEGY_ENGINE"
    STEP_12_EVALUATE_RISK = "12_EVALUATE_RISK"
    STEP_13_STAGE_JOURNAL_CONTEXT = "13_STAGE_JOURNAL_CONTEXT"
    STEP_14_EXECUTE_OR_DRY_RUN_AND_COMMIT_BAR = "14_EXECUTE_OR_DRY_RUN_AND_COMMIT_BAR"


class RuntimeCycleStatus(StrEnum):
    """Deterministic terminal classification of a single `DemoTradingOrchestrator` cycle."""

    EXECUTED_ORDER = "EXECUTED_ORDER"
    DRY_RUN_EVALUATED = "DRY_RUN_EVALUATED"
    SKIPPED_DUPLICATE_BAR = "SKIPPED_DUPLICATE_BAR"
    NO_EXECUTABLE_SIGNAL = "NO_EXECUTABLE_SIGNAL"
    RISK_REJECTED = "RISK_REJECTED"
    BROKER_REJECTED = "BROKER_REJECTED"
    AMBIGUOUS_EXECUTION_REQUIRES_RECONCILIATION = "AMBIGUOUS_EXECUTION_REQUIRES_RECONCILIATION"
    HALTED_BY_EMERGENCY_STOP = "HALTED_BY_EMERGENCY_STOP"


class BarEvaluationState(BaseModel):
    """Immutable persisted bar-close evaluation tracking payload bound to `(account_login, server)`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = Field(default="atlas_ai.bar_evaluation_state.v1", min_length=1)
    updated_at_utc: datetime
    last_evaluated_bar_timestamps_utc: dict[str, datetime] = Field(
        default_factory=dict,
        description="Mapping of '<SYMBOL>:<TIMEFRAME>' to the most recently evaluated closed bar open UTC timestamp.",
    )

    @field_validator("updated_at_utc", mode="before")
    @classmethod
    def validate_updated_at_utc(cls, value: datetime | str) -> datetime:
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        if not isinstance(value, datetime):
            raise ValueError(f"Expected datetime, got {type(value).__name__}.")
        return _normalize_utc(value)

    @field_validator("last_evaluated_bar_timestamps_utc", mode="before")
    @classmethod
    def validate_bar_timestamps(cls, value: object) -> dict[str, datetime]:
        if not isinstance(value, dict):
            raise ValueError("last_evaluated_bar_timestamps_utc must be a dictionary.")
        cleaned: dict[str, datetime] = {}
        for k, raw_ts in value.items():
            if not isinstance(k, str) or ":" not in k:
                raise ValueError(f"Bar state key must be '<SYMBOL>:<TIMEFRAME>', got {k!r}.")
            sym_part, tf_part = k.split(":", 1)
            sym_clean = sym_part.strip().upper()
            if not sym_clean or not sym_clean.isalnum():
                raise ValueError(f"Invalid symbol in bar state key {k!r}.")
            Timeframe(tf_part.strip())
            if isinstance(raw_ts, str):
                parsed_ts = datetime.fromisoformat(raw_ts)
            elif isinstance(raw_ts, datetime):
                parsed_ts = raw_ts
            else:
                raise ValueError(f"Invalid timestamp for bar state key {k!r}: {raw_ts!r}.")
            cleaned[f"{sym_clean}:{tf_part.strip()}"] = _normalize_utc(parsed_ts)
        return cleaned


class RuntimeCycleReport(BaseModel):
    """Structured, immutable audit report produced by `DemoTradingOrchestrator.run_single_cycle()`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cycle_timestamp_utc: datetime
    account_login: int = Field(..., gt=0)
    server: str = Field(..., min_length=1)
    symbol: str = Field(..., min_length=3, max_length=16)
    primary_timeframe: Timeframe
    higher_timeframes: tuple[Timeframe, ...]
    dry_run_only: bool
    status: RuntimeCycleStatus
    executed_steps: tuple[RuntimeCycleStep, ...]
    latest_closed_bar_timestamp_utc: datetime | None = Field(default=None)
    previous_evaluated_bar_timestamp_utc: datetime | None = Field(default=None)
    committed_evaluated_bar_timestamp_utc: datetime | None = Field(default=None)
    reconciliation_report: LedgerReconciliationReport
    trade_reconstruction_report: TradeReconstructionReport
    account_state: AccountState
    drift_report: DriftAndHealthReport
    emergency_stop_state: EmergencyStopState
    analysis_report: StrategyAnalysisReport | None = Field(default=None)
    signal_id: str | None = Field(default=None)
    signal: TradeSignal | None = Field(default=None)
    risk_status: RiskEvaluationStatus = Field(default=RiskEvaluationStatus.NOT_EVALUATED)
    risk_decision: RiskDecision | None = Field(default=None)
    execution_result: DemoExecutionResult | None = Field(default=None)
    notes: tuple[str, ...] = Field(default_factory=tuple)


class RuntimeSessionSummary(BaseModel):
    """Structured summary of a multi-cycle `DemoTradingOrchestrator.run_session()` run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    started_at_utc: datetime
    completed_at_utc: datetime
    total_cycles_run: int = Field(..., ge=0)
    executed_order_cycles: int = Field(default=0, ge=0)
    dry_run_evaluated_cycles: int = Field(default=0, ge=0)
    skipped_duplicate_bar_cycles: int = Field(default=0, ge=0)
    no_executable_signal_cycles: int = Field(default=0, ge=0)
    risk_rejected_cycles: int = Field(default=0, ge=0)
    broker_rejected_cycles: int = Field(default=0, ge=0)
    ambiguous_execution_cycles: int = Field(default=0, ge=0)
    halted_by_emergency_stop: bool = Field(default=False)
    graceful_shutdown_requested: bool = Field(default=False)
    cycle_reports: tuple[RuntimeCycleReport, ...] = Field(default_factory=tuple)


class PersistentBarStateTracker:
    """Crash-safe persistent tracker for `last_evaluated_bar_timestamp_utc` per `(symbol, timeframe)`.

    Safety Invariants:
    1. Reuses Phase 7 `CrashSafeJsonPersistence` with SHA-256 checksums, monotonic generations,
       hash chaining, atomic `.tmp` + `fsync` + `os.replace`, and OS-level `FileLock`.
    2. Bound strictly to `(account_login, server)` — rejects account/server mismatch, corrupted JSON,
       tampered payloads, or lingering `.tmp.*` recovery locks.
    3. Enforces monotonic bar progression per `(symbol, timeframe)` so an older closed bar cannot
       overwrite a newer already-evaluated bar timestamp.
    """

    def __init__(
        self,
        state_path: Path,
        *,
        lock_timeout_seconds: float = 5.0,
    ) -> None:
        self._state_path = Path(state_path)
        self._persistence = CrashSafeJsonPersistence(
            self._state_path,
            lock_timeout_seconds=lock_timeout_seconds,
        )
        self._mutex = threading.RLock()
        self._loaded_account: tuple[int, str] | None = None
        self._state: BarEvaluationState | None = None
        self._last_generation: int = 0

    @property
    def state_path(self) -> Path:
        return self._state_path

    @property
    def persistence(self) -> CrashSafeJsonPersistence:
        return self._persistence

    @property
    def last_generation(self) -> int:
        with self._mutex:
            return self._last_generation

    def load_or_initialize(
        self,
        *,
        account_login: int,
        server: str,
        now_utc: datetime | None = None,
    ) -> BarEvaluationState:
        """Load and verify the persisted bar-close evaluation state from disk under an exclusive lock."""
        login, cleaned_server = _validate_account_identity(account_login, server)
        ts = _normalize_utc(now_utc)

        with self._mutex:
            env = self._persistence.load(
                expected_login=login,
                expected_server=cleaned_server,
            )
            self._loaded_account = (login, cleaned_server)
            if env is None:
                self._state = BarEvaluationState(
                    updated_at_utc=ts,
                    last_evaluated_bar_timestamps_utc={},
                )
                self._last_generation = 0
                return self._state

            try:
                loaded = BarEvaluationState.model_validate(env.payload)
            except ValidationError as exc:
                raise PersistenceCorruptionError(
                    f"BarEvaluationState schema validation failed for '{self._state_path}': {exc}"
                ) from exc

            self._state = loaded
            self._last_generation = env.generation
            return self._state

    def get_last_evaluated_bar_timestamp(
        self,
        symbol: str,
        timeframe: Timeframe,
        *,
        account_login: int,
        server: str,
        now_utc: datetime | None = None,
    ) -> datetime | None:
        """Return the last durably committed closed-bar open UTC timestamp for `(symbol, timeframe)`."""
        state = self.load_or_initialize(
            account_login=account_login,
            server=server,
            now_utc=now_utc,
        )
        key = _format_bar_key(symbol, timeframe)
        return state.last_evaluated_bar_timestamps_utc.get(key)

    def is_bar_already_evaluated(
        self,
        symbol: str,
        timeframe: Timeframe,
        bar_open_timestamp_utc: datetime,
        *,
        account_login: int,
        server: str,
        now_utc: datetime | None = None,
    ) -> tuple[bool, datetime | None]:
        """Return `(True, last_ts)` if `bar_open_timestamp_utc <= last_ts` for `(symbol, timeframe)`."""
        bar_ts = _normalize_utc(bar_open_timestamp_utc)
        last_ts = self.get_last_evaluated_bar_timestamp(
            symbol,
            timeframe,
            account_login=account_login,
            server=server,
            now_utc=now_utc,
        )
        if last_ts is not None and bar_ts <= last_ts:
            return True, last_ts
        return False, last_ts

    def commit_evaluated_bar_timestamp(
        self,
        symbol: str,
        timeframe: Timeframe,
        bar_open_timestamp_utc: datetime,
        *,
        account_login: int,
        server: str,
        now_utc: datetime | None = None,
    ) -> BarEvaluationState:
        """Durably persist `bar_open_timestamp_utc` as evaluated for `(symbol, timeframe)`."""
        login, cleaned_server = _validate_account_identity(account_login, server)
        bar_ts = _normalize_utc(bar_open_timestamp_utc)
        ts = _normalize_utc(now_utc)
        key = _format_bar_key(symbol, timeframe)

        with self._mutex:
            current = self.load_or_initialize(
                account_login=login,
                server=cleaned_server,
                now_utc=ts,
            )
            existing_ts = current.last_evaluated_bar_timestamps_utc.get(key)
            if existing_ts is not None and bar_ts < existing_ts:
                raise PersistenceRollbackDetectedError(
                    f"Cannot regress evaluated bar timestamp for '{key}' from "
                    f"{existing_ts.isoformat()} backwards to {bar_ts.isoformat()}."
                )

            updated_map = dict(current.last_evaluated_bar_timestamps_utc)
            updated_map[key] = bar_ts
            new_state = BarEvaluationState(
                updated_at_utc=ts,
                last_evaluated_bar_timestamps_utc=updated_map,
            )
            saved_env = self._persistence.save(
                new_state.model_dump(mode="json"),
                account_login=login,
                server=cleaned_server,
                now_utc=ts,
            )
            self._state = new_state
            self._loaded_account = (login, cleaned_server)
            self._last_generation = saved_env.generation
            return new_state


class DemoTradingOrchestrator:
    """Deterministic, safety-gated closed-loop MT5 DEMO trading orchestrator for ATLAS AI Phase 9.2.

    Enforces the mandatory 14-step cycle order on every invocation of `run_single_cycle()`:
    1. Verify MT5 connection and DEMO account (`connection.ensure_connected()`).
    2. Reconcile unresolved execution orders (`execution_engine.reconcile_unresolved_orders()`).
    3. Synchronize closed trades with `PersistentTradeJournal` (`trade_reconstructor.sync_closed_demo_trades()`).
    4. Reconstruct account risk state (`account_tracker.reconstruct_account_state()`).
    5. Evaluate drift (`drift_engine.evaluate_journal()`) and ledger health (`reconciliation_report.health_status`).
    6. Evaluate and persist `EmergencyStopController` state (`evaluate_ledger_health()`, `evaluate_drift_report()`).
    7. Call `emergency_stop.assert_trading_authorized(...)`.
    8. Check closed-bar timestamps and bar-close deduplication (`PersistentBarStateTracker`).
       (Steps 1–7 always execute even when the latest closed bar has already been evaluated!)
    9. Fetch synchronized closed market data across `primary_timeframe` and `higher_timeframes`.
    10. Run `FeaturePipeline.transform()` on primary and higher timeframe DataFrames.
    11. Run `StrategyEngine.analyze()` with `mtf_data`, `economic_events`, `account_state`,
        `risk_parameters`, and `instrument_spec`.
    12. Evaluate risk decision and signal executability (`WAIT` vs `RISK_REJECTED` vs `RISK_APPROVED`).
    13. Stage causal analysis context in `PersistentTradeJournal` (`stage_analysis_report()`).
    14. Execute via `execution_engine.execute_signal()` only if all safety conditions pass and
        `config.dry_run_only == False` (or emit `DRY_RUN_EVALUATED` with zero `order_send` calls when
        `config.dry_run_only == True`), and durably commit `last_evaluated_bar_timestamp_utc` only
        after the cycle's evaluation/execution outcome is durably recorded.
    """

    def __init__(
        self,
        *,
        config: DemoRuntimeConfig,
        connection: MT5ConnectionAdapter,
        market_data: MT5MarketDataAdapter,
        account_tracker: PersistentAccountRiskTracker,
        ledger: PersistentExecutionLedger,
        execution_engine: MT5DemoExecutionEngine,
        journal: PersistentTradeJournal,
        trade_reconstructor: DemoTradeReconstructor,
        drift_engine: DriftDetectionEngine | None = None,
        emergency_stop: EmergencyStopController | None = None,
        feature_pipeline: FeaturePipeline | None = None,
        strategy_engine: StrategyEngine | None = None,
        risk_engine: RiskEngine | None = None,
        lifecycle_manager: SignalLifecycleManager | None = None,
        bar_state_tracker: PersistentBarStateTracker | None = None,
        bar_state_path: Path | None = None,
        require_staged_context: bool = True,
    ) -> None:
        self._config = config
        self._connection = connection
        self._market_data = market_data
        self._account_tracker = account_tracker
        self._ledger = ledger
        self._execution_engine = execution_engine
        self._journal = journal
        self._trade_reconstructor = trade_reconstructor
        self._drift_engine = drift_engine or DriftDetectionEngine(
            config=journal.config
            if isinstance(getattr(journal, "config", None), PostTradeAnalyticsConfig)
            else PostTradeAnalyticsConfig(),
            journal=journal,
        )
        if self._drift_engine.journal is None:
            self._drift_engine._journal = journal
        self._emergency_stop = emergency_stop or EmergencyStopController(config=config)
        self._feature_pipeline = feature_pipeline or FeaturePipeline()
        self._strategy_engine = strategy_engine or StrategyEngine(
            feature_pipeline=self._feature_pipeline,
        )
        self._lifecycle_manager = (
            lifecycle_manager
            or self._strategy_engine.lifecycle_manager
        )
        # Ensure StrategyEngine, MT5DemoExecutionEngine, PreTradeSafetyGate, and DemoTradeReconstructor
        # share the exact same SignalLifecycleManager instance.
        if self._strategy_engine._lifecycle_manager is not self._lifecycle_manager:
            self._strategy_engine._lifecycle_manager = self._lifecycle_manager
        if self._execution_engine._lifecycle_manager is not self._lifecycle_manager:
            self._execution_engine._lifecycle_manager = self._lifecycle_manager
        if self._execution_engine.safety_gate._lifecycle_manager is not self._lifecycle_manager:
            self._execution_engine.safety_gate._lifecycle_manager = self._lifecycle_manager
        if getattr(self._trade_reconstructor, "_lifecycle_manager", None) is not self._lifecycle_manager:
            self._trade_reconstructor._lifecycle_manager = self._lifecycle_manager

        self._risk_engine = risk_engine or execution_engine.risk_engine

        resolved_bar_path = bar_state_path or config.emergency_state_path.with_name(
            f"{config.emergency_state_path.stem}_bar_state.json"
        )
        self._bar_state_tracker = bar_state_tracker or PersistentBarStateTracker(
            resolved_bar_path,
            lock_timeout_seconds=config.lock_timeout_seconds,
        )

        # Ensure the journal's pre-send EntryContextSnapshot hook is bound to execution_engine
        self._journal.bind_execution_engine(
            self._execution_engine,
            require_staged_context=require_staged_context,
        )

        self._mutex = threading.RLock()
        self._shutdown_event = threading.Event()
        self._step_observer: Callable[[RuntimeCycleStep], None] | None = None

    @property
    def config(self) -> DemoRuntimeConfig:
        return self._config

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
    def execution_engine(self) -> MT5DemoExecutionEngine:
        return self._execution_engine

    @property
    def journal(self) -> PersistentTradeJournal:
        return self._journal

    @property
    def trade_reconstructor(self) -> DemoTradeReconstructor:
        return self._trade_reconstructor

    @property
    def drift_engine(self) -> DriftDetectionEngine:
        return self._drift_engine

    @property
    def emergency_stop(self) -> EmergencyStopController:
        return self._emergency_stop

    @property
    def feature_pipeline(self) -> FeaturePipeline:
        return self._feature_pipeline

    @property
    def strategy_engine(self) -> StrategyEngine:
        return self._strategy_engine

    @property
    def risk_engine(self) -> RiskEngine:
        return self._risk_engine

    @property
    def lifecycle_manager(self) -> SignalLifecycleManager:
        return self._lifecycle_manager

    @property
    def bar_state_tracker(self) -> PersistentBarStateTracker:
        return self._bar_state_tracker

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_event.is_set()

    def request_shutdown(self) -> None:
        """Signal the orchestrator to stop cleanly at the current or next cycle boundary."""
        self._shutdown_event.set()

    def clear_shutdown_request(self) -> None:
        """Clear a prior shutdown request flag."""
        self._shutdown_event.clear()

    def set_step_observer(
        self,
        observer: Callable[[RuntimeCycleStep], None] | None,
    ) -> None:
        """Register an optional callback invoked as each `RuntimeCycleStep` begins."""
        self._step_observer = observer

    def _record_step(
        self,
        steps: list[RuntimeCycleStep],
        step: RuntimeCycleStep,
    ) -> None:
        steps.append(step)
        if self._step_observer is not None:
            self._step_observer(step)

    def _resolve_fallback_account_identity(self) -> tuple[int, str]:
        """Resolve `(account_login, server)` from connection state or config when a connection check fails."""
        try:
            acct = self._connection.backend.account_info()
            if acct is not None and acct.login > 0 and acct.server.strip():
                return acct.login, acct.server.strip()
        except Exception:
            pass
        cfg = self._connection.config
        login = cfg.expected_login or cfg.login or 0
        server = (cfg.expected_server or cfg.server or "").strip()
        return login, server

    def _handle_persistence_exception(
        self,
        exc: Exception,
        *,
        account_login: int,
        server: str,
        now_utc: datetime,
    ) -> None:
        """Record a persistence or recovery-lock failure in `EmergencyStopController` without masking the root error."""
        if account_login <= 0 or not server.strip():
            return
        try:
            if isinstance(exc, LedgerRecoveryLockError) or (
                isinstance(exc, PersistenceError)
                and self._ledger.health_status == LedgerHealthStatus.RECOVERY_REQUIRED_LOCKED
            ):
                self._emergency_stop.evaluate_ledger_health(
                    LedgerHealthStatus.RECOVERY_REQUIRED_LOCKED,
                    account_login=account_login,
                    server=server,
                    detail=f"{type(exc).__name__}: {exc}",
                    now_utc=now_utc,
                )
            self._emergency_stop.record_persistence_failure(
                f"{type(exc).__name__}: {exc}",
                account_login=account_login,
                server=server,
                now_utc=now_utc,
            )
        except Exception:
            # Never mask the original PersistenceError / LedgerRecoveryLockError if emergency_state_path itself is locked/corrupt
            pass

    def _sync_signal_lifecycle_before_execution(
        self,
        signal_id: str,
        risk_decision: RiskDecision,
        now_utc: datetime,
    ) -> None:
        """Temporarily synchronize a `RISK_APPROVED` signal back to `CONFIRMED` in `SignalLifecycleManager`
        immediately before invoking `MT5DemoExecutionEngine.execute_signal()`, because `PreTradeSafetyGate`
        re-evaluates `RiskEngine` against live Bid/Ask and calls `lifecycle_manager.apply_risk_decision()`.
        """
        try:
            rec = self._lifecycle_manager.get_record(signal_id)
        except Exception:
            return
        if rec.state == SignalLifecycleState.RISK_APPROVED:
            reverted = rec.model_copy(
                update={
                    "state": SignalLifecycleState.CONFIRMED,
                    "risk_status": RiskEvaluationStatus.NOT_EVALUATED,
                    "risk_decision": risk_decision,
                }
            )
            self._lifecycle_manager._records[signal_id] = reverted

    def _restore_signal_lifecycle_for_cancelled_or_rejected_order(
        self,
        signal_id: str,
        now_utc: datetime,
        reason: str,
    ) -> None:
        """Ensure any signal whose execution did not complete is cleanly moved out of non-terminal states."""
        try:
            rec = self._lifecycle_manager.get_record(signal_id)
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
        except Exception:
            pass

    def run_single_cycle(
        self,
        *,
        reference_now_utc: datetime | None = None,
        economic_events: Sequence[EconomicEvent] = (),
        risk_parameters: RiskParameters | None = None,
        client_order_id: str | None = None,
    ) -> RuntimeCycleReport:
        """Execute one complete, deterministic 14-step closed-loop demo trading cycle."""
        now_utc = _normalize_utc(reference_now_utc)
        executed_steps: list[RuntimeCycleStep] = []
        notes: list[str] = []
        symbol = self._config.symbol
        primary_tf = self._config.primary_timeframe
        higher_tfs = self._config.higher_timeframes

        with self._mutex:
            # =========================================================================
            # STEP 1: Verify MT5 connection and DEMO account
            # =========================================================================
            self._record_step(executed_steps, RuntimeCycleStep.STEP_01_VERIFY_CONNECTION)
            try:
                account_info = self._connection.ensure_connected()
            except (LiveTradingBlockedError, MT5AuthenticationError) as fatal_conn_exc:
                fallback_login, fallback_server = self._resolve_fallback_account_identity()
                if fallback_login > 0 and fallback_server:
                    try:
                        self._emergency_stop.record_connection_failure(
                            f"{type(fatal_conn_exc).__name__}: {fatal_conn_exc}",
                            account_login=fallback_login,
                            server=fallback_server,
                            now_utc=now_utc,
                        )
                    except Exception:
                        pass
                raise
            except MT5ConnectionError as conn_exc:
                fallback_login, fallback_server = self._resolve_fallback_account_identity()
                if fallback_login > 0 and fallback_server:
                    try:
                        self._emergency_stop.record_connection_failure(
                            f"{type(conn_exc).__name__}: {conn_exc}",
                            account_login=fallback_login,
                            server=fallback_server,
                            now_utc=now_utc,
                        )
                    except Exception:
                        pass
                raise

            acct_login = account_info.login
            acct_server = account_info.server.strip()
            try:
                self._emergency_stop.record_connection_success(
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                )
            except (PersistenceError, LedgerRecoveryLockError) as persist_exc:
                self._handle_persistence_exception(
                    persist_exc,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                )
                raise

            # =========================================================================
            # STEP 2: Reconcile unresolved execution orders
            # =========================================================================
            self._record_step(executed_steps, RuntimeCycleStep.STEP_02_RECONCILE_UNRESOLVED_ORDERS)
            try:
                reconciliation_report = self._execution_engine.reconcile_unresolved_orders(
                    reference_now_utc=now_utc,
                )
            except (PersistenceError, LedgerRecoveryLockError) as persist_exc:
                self._handle_persistence_exception(
                    persist_exc,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                )
                raise

            # =========================================================================
            # STEP 3: Synchronize closed trades with PersistentTradeJournal
            # =========================================================================
            self._record_step(executed_steps, RuntimeCycleStep.STEP_03_SYNC_CLOSED_TRADES)
            try:
                trade_reconstruction_report = self._trade_reconstructor.sync_closed_demo_trades(
                    reference_now_utc=now_utc,
                )
            except (PersistenceError, LedgerRecoveryLockError) as persist_exc:
                self._handle_persistence_exception(
                    persist_exc,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                )
                raise

            # =========================================================================
            # STEP 4: Reconstruct account risk state
            # =========================================================================
            self._record_step(executed_steps, RuntimeCycleStep.STEP_04_RECONSTRUCT_ACCOUNT_STATE)
            try:
                account_state = self._account_tracker.reconstruct_account_state(
                    reference_now_utc=now_utc,
                    symbol=symbol,
                )
            except (PersistenceError, LedgerRecoveryLockError) as persist_exc:
                self._handle_persistence_exception(
                    persist_exc,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                )
                raise

            # =========================================================================
            # STEP 5: Evaluate drift and ledger health
            # =========================================================================
            self._record_step(
                executed_steps,
                RuntimeCycleStep.STEP_05_EVALUATE_DRIFT_AND_LEDGER_HEALTH,
            )
            try:
                self._journal.load_or_initialize(
                    account_login=acct_login,
                    server=acct_server,
                )
                drift_report = self._drift_engine.evaluate_trades(
                    self._journal.list_completed_trades(),
                    initial_balance=account_state.daily_starting_balance,
                )
                ledger_health = reconciliation_report.health_status
            except (PersistenceError, LedgerRecoveryLockError) as persist_exc:
                self._handle_persistence_exception(
                    persist_exc,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                )
                raise

            # =========================================================================
            # STEP 6: Evaluate and persist EmergencyStopController state
            # =========================================================================
            self._record_step(
                executed_steps,
                RuntimeCycleStep.STEP_06_EVALUATE_EMERGENCY_STOP_STATE,
            )
            try:
                self._emergency_stop.evaluate_ledger_health(
                    ledger_health,
                    account_login=acct_login,
                    server=acct_server,
                    detail=(
                        f"Ledger reconciliation health status={ledger_health.value} "
                        f"(still_working={len(reconciliation_report.still_working_ids)}, "
                        f"orphans={len(reconciliation_report.discovered_orphan_ids)})."
                    ),
                    now_utc=now_utc,
                )
                emergency_state = self._emergency_stop.evaluate_drift_report(
                    drift_report,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                )
            except (PersistenceError, LedgerRecoveryLockError) as persist_exc:
                self._handle_persistence_exception(
                    persist_exc,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                )
                raise

            # =========================================================================
            # STEP 7: Call emergency_stop.assert_trading_authorized()
            # =========================================================================
            self._record_step(
                executed_steps,
                RuntimeCycleStep.STEP_07_ASSERT_TRADING_AUTHORIZED,
            )
            try:
                emergency_state = self._emergency_stop.assert_trading_authorized(
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                )
            except (PersistenceError, LedgerRecoveryLockError) as persist_exc:
                self._handle_persistence_exception(
                    persist_exc,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                )
                raise

            # =========================================================================
            # STEP 8: Check closed-bar timestamps and bar-close deduplication
            # =========================================================================
            self._record_step(
                executed_steps,
                RuntimeCycleStep.STEP_08_CHECK_BAR_CLOSE_DEDUPLICATION,
            )
            primary_candles = self._market_data.fetch_closed_candles(
                symbol=symbol,
                timeframe=primary_tf,
                count=self._config.candle_lookback_bars,
                reference_now_utc=now_utc,
            )
            latest_closed_bar_ts = primary_candles[-1].timestamp

            try:
                already_evaluated, prev_bar_ts = self._bar_state_tracker.is_bar_already_evaluated(
                    symbol,
                    primary_tf,
                    latest_closed_bar_ts,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                )
            except (PersistenceError, LedgerRecoveryLockError) as persist_exc:
                self._handle_persistence_exception(
                    persist_exc,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                )
                raise

            if already_evaluated:
                notes.append(
                    f"Skipped duplicate closed bar {latest_closed_bar_ts.isoformat()} "
                    f"for {symbol} {primary_tf.value} (last_evaluated={prev_bar_ts.isoformat() if prev_bar_ts else 'none'})."
                )
                return RuntimeCycleReport(
                    cycle_timestamp_utc=now_utc,
                    account_login=acct_login,
                    server=acct_server,
                    symbol=symbol,
                    primary_timeframe=primary_tf,
                    higher_timeframes=higher_tfs,
                    dry_run_only=self._config.dry_run_only,
                    status=RuntimeCycleStatus.SKIPPED_DUPLICATE_BAR,
                    executed_steps=tuple(executed_steps),
                    latest_closed_bar_timestamp_utc=latest_closed_bar_ts,
                    previous_evaluated_bar_timestamp_utc=prev_bar_ts,
                    committed_evaluated_bar_timestamp_utc=prev_bar_ts,
                    reconciliation_report=reconciliation_report,
                    trade_reconstruction_report=trade_reconstruction_report,
                    account_state=account_state,
                    drift_report=drift_report,
                    emergency_stop_state=emergency_state,
                    notes=tuple(notes),
                )

            # =========================================================================
            # STEP 9: Fetch synchronized market data (primary + higher timeframes)
            # =========================================================================
            self._record_step(
                executed_steps,
                RuntimeCycleStep.STEP_09_FETCH_SYNCHRONIZED_MARKET_DATA,
            )
            primary_pipeline_res = self._market_data.fetch_market_data_pipeline_result(
                symbol=symbol,
                timeframe=primary_tf,
                count=self._config.candle_lookback_bars,
                reference_now_utc=now_utc,
            )
            raw_htf_dfs: dict[Timeframe, pd.DataFrame] = {}
            for htf in higher_tfs:
                htf_res = self._market_data.fetch_market_data_pipeline_result(
                    symbol=symbol,
                    timeframe=htf,
                    count=self._config.candle_lookback_bars,
                    reference_now_utc=now_utc,
                )
                raw_htf_dfs[htf] = htf_res.validated_dataframe

            # =========================================================================
            # STEP 10: Run FeaturePipeline
            # =========================================================================
            self._record_step(executed_steps, RuntimeCycleStep.STEP_10_RUN_FEATURE_PIPELINE)
            featured_primary_df = self._feature_pipeline.transform(
                primary_pipeline_res.validated_dataframe
            )
            featured_htf_dfs: dict[Timeframe, pd.DataFrame] = {
                htf: self._feature_pipeline.transform(htf_df)
                for htf, htf_df in raw_htf_dfs.items()
            }

            # =========================================================================
            # STEP 11: Run StrategyEngine.analyze()
            # =========================================================================
            self._record_step(executed_steps, RuntimeCycleStep.STEP_11_RUN_STRATEGY_ENGINE)
            active_risk_params = risk_parameters or self._risk_engine.risk_parameters
            instrument_spec = self._connection.get_instrument_spec(symbol)

            analysis_report = self._strategy_engine.analyze(
                featured_primary_df,
                symbol=symbol,
                timeframe=primary_tf,
                mtf_data=featured_htf_dfs,
                economic_events=economic_events,
                account_state=account_state,
                risk_parameters=active_risk_params,
                instrument_spec=instrument_spec,
                register_in_lifecycle=True,
            )

            # =========================================================================
            # STEP 12: Evaluate risk decision & signal executability
            # =========================================================================
            self._record_step(executed_steps, RuntimeCycleStep.STEP_12_EVALUATE_RISK)
            signal = analysis_report.signal
            risk_status = analysis_report.risk_status
            risk_decision = analysis_report.risk_decision
            lifecycle_rec = analysis_report.lifecycle_record
            signal_id = lifecycle_rec.signal_id if lifecycle_rec is not None else None

            if not signal.is_executable or signal.direction == SignalDirection.WAIT:
                try:
                    self._bar_state_tracker.commit_evaluated_bar_timestamp(
                        symbol,
                        primary_tf,
                        latest_closed_bar_ts,
                        account_login=acct_login,
                        server=acct_server,
                        now_utc=now_utc,
                    )
                except (PersistenceError, LedgerRecoveryLockError) as persist_exc:
                    self._handle_persistence_exception(
                        persist_exc,
                        account_login=acct_login,
                        server=acct_server,
                        now_utc=now_utc,
                    )
                    raise

                notes.append(f"Non-executable WAIT signal on bar {latest_closed_bar_ts.isoformat()}: {signal.rationale}")
                return RuntimeCycleReport(
                    cycle_timestamp_utc=now_utc,
                    account_login=acct_login,
                    server=acct_server,
                    symbol=symbol,
                    primary_timeframe=primary_tf,
                    higher_timeframes=higher_tfs,
                    dry_run_only=self._config.dry_run_only,
                    status=RuntimeCycleStatus.NO_EXECUTABLE_SIGNAL,
                    executed_steps=tuple(executed_steps),
                    latest_closed_bar_timestamp_utc=latest_closed_bar_ts,
                    previous_evaluated_bar_timestamp_utc=prev_bar_ts,
                    committed_evaluated_bar_timestamp_utc=latest_closed_bar_ts,
                    reconciliation_report=reconciliation_report,
                    trade_reconstruction_report=trade_reconstruction_report,
                    account_state=account_state,
                    drift_report=drift_report,
                    emergency_stop_state=emergency_state,
                    analysis_report=analysis_report,
                    signal_id=signal_id,
                    signal=signal,
                    risk_status=risk_status,
                    risk_decision=risk_decision,
                    notes=tuple(notes),
                )

            if (
                risk_status != RiskEvaluationStatus.RISK_APPROVED
                or risk_decision is None
                or not risk_decision.approved
            ):
                try:
                    self._bar_state_tracker.commit_evaluated_bar_timestamp(
                        symbol,
                        primary_tf,
                        latest_closed_bar_ts,
                        account_login=acct_login,
                        server=acct_server,
                        now_utc=now_utc,
                    )
                except (PersistenceError, LedgerRecoveryLockError) as persist_exc:
                    self._handle_persistence_exception(
                        persist_exc,
                        account_login=acct_login,
                        server=acct_server,
                        now_utc=now_utc,
                    )
                    raise

                rej_reason = (
                    risk_decision.rejection_reason
                    if risk_decision is not None
                    else "Risk decision missing or not approved."
                )
                notes.append(f"Signal '{signal_id}' rejected by RiskEngine: {rej_reason}")
                return RuntimeCycleReport(
                    cycle_timestamp_utc=now_utc,
                    account_login=acct_login,
                    server=acct_server,
                    symbol=symbol,
                    primary_timeframe=primary_tf,
                    higher_timeframes=higher_tfs,
                    dry_run_only=self._config.dry_run_only,
                    status=RuntimeCycleStatus.RISK_REJECTED,
                    executed_steps=tuple(executed_steps),
                    latest_closed_bar_timestamp_utc=latest_closed_bar_ts,
                    previous_evaluated_bar_timestamp_utc=prev_bar_ts,
                    committed_evaluated_bar_timestamp_utc=latest_closed_bar_ts,
                    reconciliation_report=reconciliation_report,
                    trade_reconstruction_report=trade_reconstruction_report,
                    account_state=account_state,
                    drift_report=drift_report,
                    emergency_stop_state=emergency_state,
                    analysis_report=analysis_report,
                    signal_id=signal_id,
                    signal=signal,
                    risk_status=RiskEvaluationStatus.RISK_REJECTED,
                    risk_decision=risk_decision,
                    notes=tuple(notes),
                )

            if signal_id is None:
                raise RuntimeOrchestrationError(
                    "Executable RISK_APPROVED signal is missing a SignalLifecycleManager signal_id."
                )

            # =========================================================================
            # STEP 13: Stage analysis context in PersistentTradeJournal
            # =========================================================================
            self._record_step(executed_steps, RuntimeCycleStep.STEP_13_STAGE_JOURNAL_CONTEXT)
            self._journal.stage_analysis_report(
                signal_id,
                analysis_report,
            )

            # =========================================================================
            # STEP 14: Execute via execution_engine.execute_signal() (or dry-run)
            #          and commit evaluated bar timestamp
            # =========================================================================
            self._record_step(
                executed_steps,
                RuntimeCycleStep.STEP_14_EXECUTE_OR_DRY_RUN_AND_COMMIT_BAR,
            )

            # Dry-Run / Shadow Mode structural guard: NEVER call execute_signal() or order_send()
            if self._config.dry_run_only:
                self._restore_signal_lifecycle_for_cancelled_or_rejected_order(
                    signal_id,
                    now_utc,
                    "Dry-run shadow evaluation completed without broker execution.",
                )
                try:
                    self._bar_state_tracker.commit_evaluated_bar_timestamp(
                        symbol,
                        primary_tf,
                        latest_closed_bar_ts,
                        account_login=acct_login,
                        server=acct_server,
                        now_utc=now_utc,
                    )
                except (PersistenceError, LedgerRecoveryLockError) as persist_exc:
                    self._handle_persistence_exception(
                        persist_exc,
                        account_login=acct_login,
                        server=acct_server,
                        now_utc=now_utc,
                    )
                    raise

                notes.append(
                    f"Dry-run shadow evaluation completed for signal '{signal_id}' "
                    f"({signal.direction.value} {symbol} {primary_tf.value}); order submission skipped."
                )
                return RuntimeCycleReport(
                    cycle_timestamp_utc=now_utc,
                    account_login=acct_login,
                    server=acct_server,
                    symbol=symbol,
                    primary_timeframe=primary_tf,
                    higher_timeframes=higher_tfs,
                    dry_run_only=True,
                    status=RuntimeCycleStatus.DRY_RUN_EVALUATED,
                    executed_steps=tuple(executed_steps),
                    latest_closed_bar_timestamp_utc=latest_closed_bar_ts,
                    previous_evaluated_bar_timestamp_utc=prev_bar_ts,
                    committed_evaluated_bar_timestamp_utc=latest_closed_bar_ts,
                    reconciliation_report=reconciliation_report,
                    trade_reconstruction_report=trade_reconstruction_report,
                    account_state=account_state,
                    drift_report=drift_report,
                    emergency_stop_state=emergency_state,
                    analysis_report=analysis_report,
                    signal_id=signal_id,
                    signal=signal,
                    risk_status=risk_status,
                    risk_decision=risk_decision,
                    execution_result=None,
                    notes=tuple(notes),
                )

            # Prepare SignalLifecycleManager so PreTradeSafetyGate can perform its live-tick RiskEngine check
            self._sync_signal_lifecycle_before_execution(
                signal_id,
                risk_decision,
                now_utc,
            )

            try:
                execution_result = self._execution_engine.execute_signal(
                    signal,
                    signal_id=signal_id,
                    client_order_id=client_order_id,
                    requested_volume=None,
                    risk_parameters=active_risk_params,
                    reference_now_utc=now_utc,
                )
            except (PersistenceError, LedgerRecoveryLockError) as persist_exc:
                self._restore_signal_lifecycle_for_cancelled_or_rejected_order(
                    signal_id,
                    now_utc,
                    f"Execution aborted due to persistence failure: {persist_exc}",
                )
                self._handle_persistence_exception(
                    persist_exc,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                )
                raise
            except Exception as exec_exc:
                self._restore_signal_lifecycle_for_cancelled_or_rejected_order(
                    signal_id,
                    now_utc,
                    f"Execution aborted before completion: {exec_exc}",
                )
                raise

            # Durably commit the evaluated bar timestamp AFTER execution outcome is recorded in the ledger
            try:
                self._bar_state_tracker.commit_evaluated_bar_timestamp(
                    symbol,
                    primary_tf,
                    latest_closed_bar_ts,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                )
            except (PersistenceError, LedgerRecoveryLockError) as persist_exc:
                self._handle_persistence_exception(
                    persist_exc,
                    account_login=acct_login,
                    server=acct_server,
                    now_utc=now_utc,
                )
                raise

            if execution_result.ambiguous_requires_reconciliation:
                cycle_status = RuntimeCycleStatus.AMBIGUOUS_EXECUTION_REQUIRES_RECONCILIATION
                notes.append(
                    f"Order '{execution_result.client_order_id}' outcome is ambiguous "
                    f"(status={execution_result.ledger_entry.status.value}); new submissions blocked until reconciliation."
                )
            elif execution_result.executed:
                cycle_status = RuntimeCycleStatus.EXECUTED_ORDER
                notes.append(
                    f"Executed DEMO order '{execution_result.client_order_id}' for signal '{signal_id}' "
                    f"(status={execution_result.ledger_entry.status.value}, "
                    f"volume={execution_result.ledger_entry.executed_volume})."
                )
            else:
                cycle_status = RuntimeCycleStatus.BROKER_REJECTED
                notes.append(
                    f"Broker rejected DEMO order '{execution_result.client_order_id}' "
                    f"(status={execution_result.ledger_entry.status.value}, "
                    f"retcode={execution_result.ledger_entry.retcode})."
                )

            return RuntimeCycleReport(
                cycle_timestamp_utc=now_utc,
                account_login=acct_login,
                server=acct_server,
                symbol=symbol,
                primary_timeframe=primary_tf,
                higher_timeframes=higher_tfs,
                dry_run_only=False,
                status=cycle_status,
                executed_steps=tuple(executed_steps),
                latest_closed_bar_timestamp_utc=latest_closed_bar_ts,
                previous_evaluated_bar_timestamp_utc=prev_bar_ts,
                committed_evaluated_bar_timestamp_utc=latest_closed_bar_ts,
                reconciliation_report=reconciliation_report,
                trade_reconstruction_report=trade_reconstruction_report,
                account_state=account_state,
                drift_report=drift_report,
                emergency_stop_state=emergency_state,
                analysis_report=analysis_report,
                signal_id=signal_id,
                signal=signal,
                risk_status=risk_status,
                risk_decision=risk_decision,
                execution_result=execution_result,
                notes=tuple(notes),
            )

    def run_session(
        self,
        *,
        max_cycles: int = 1,
        cycle_timestamps_utc: Sequence[datetime] | None = None,
        economic_events: Sequence[EconomicEvent] = (),
        risk_parameters: RiskParameters | None = None,
        stop_on_emergency_halt: bool = True,
    ) -> RuntimeSessionSummary:
        """Run a bounded sequence of closed-loop demo trading cycles with graceful shutdown support.

        Stops cleanly between cycles when `request_shutdown()` is set or when the emergency circuit
        breaker latches (`CircuitBreakerLatchedError`) and `stop_on_emergency_halt=True`.
        """
        if max_cycles < 1:
            raise RuntimeOrchestrationError(f"max_cycles must be >= 1, got {max_cycles}.")
        if cycle_timestamps_utc is not None and len(cycle_timestamps_utc) != max_cycles:
            raise RuntimeOrchestrationError(
                f"cycle_timestamps_utc length ({len(cycle_timestamps_utc)}) must match max_cycles ({max_cycles})."
            )

        started_at = (
            _normalize_utc(cycle_timestamps_utc[0])
            if cycle_timestamps_utc
            else datetime.now(timezone.utc)
        )
        reports: list[RuntimeCycleReport] = []
        executed_count = 0
        dry_run_count = 0
        skipped_dup_count = 0
        no_sig_count = 0
        risk_rej_count = 0
        broker_rej_count = 0
        ambig_count = 0
        halted_by_emergency = False

        for idx in range(max_cycles):
            if self._shutdown_event.is_set():
                break

            cycle_ts = (
                _normalize_utc(cycle_timestamps_utc[idx])
                if cycle_timestamps_utc is not None
                else datetime.now(timezone.utc)
            )
            try:
                report = self.run_single_cycle(
                    reference_now_utc=cycle_ts,
                    economic_events=economic_events,
                    risk_parameters=risk_parameters,
                )
            except CircuitBreakerLatchedError:
                halted_by_emergency = True
                if stop_on_emergency_halt:
                    break
                raise

            reports.append(report)
            if report.status == RuntimeCycleStatus.EXECUTED_ORDER:
                executed_count += 1
            elif report.status == RuntimeCycleStatus.DRY_RUN_EVALUATED:
                dry_run_count += 1
            elif report.status == RuntimeCycleStatus.SKIPPED_DUPLICATE_BAR:
                skipped_dup_count += 1
            elif report.status == RuntimeCycleStatus.NO_EXECUTABLE_SIGNAL:
                no_sig_count += 1
            elif report.status == RuntimeCycleStatus.RISK_REJECTED:
                risk_rej_count += 1
            elif report.status == RuntimeCycleStatus.BROKER_REJECTED:
                broker_rej_count += 1
            elif report.status == RuntimeCycleStatus.AMBIGUOUS_EXECUTION_REQUIRES_RECONCILIATION:
                ambig_count += 1

            if self._shutdown_event.is_set():
                break

        completed_at = (
            reports[-1].cycle_timestamp_utc
            if reports
            else (
                _normalize_utc(cycle_timestamps_utc[-1])
                if cycle_timestamps_utc
                else datetime.now(timezone.utc)
            )
        )

        return RuntimeSessionSummary(
            started_at_utc=started_at,
            completed_at_utc=completed_at,
            total_cycles_run=len(reports),
            executed_order_cycles=executed_count,
            dry_run_evaluated_cycles=dry_run_count,
            skipped_duplicate_bar_cycles=skipped_dup_count,
            no_executable_signal_cycles=no_sig_count,
            risk_rejected_cycles=risk_rej_count,
            broker_rejected_cycles=broker_rej_count,
            ambiguous_execution_cycles=ambig_count,
            halted_by_emergency_stop=halted_by_emergency,
            graceful_shutdown_requested=self._shutdown_event.is_set(),
            cycle_reports=tuple(reports),
        )
