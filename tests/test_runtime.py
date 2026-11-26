"""Focused Phase 9.1 offline unit and concurrency tests for DemoRuntimeConfig,
EmergencyStopState, EmergencyAuditEvent, and persistent crash-safe EmergencyStopController.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError
import pytest

from atlas_ai.analytics import (
    CompletedTradeRecord,
    ContextAttributionStatus,
    DemoTradeReconstructor,
    DriftAndHealthReport,
    DriftDetectionEngine,
    DriftStatus,
    PersistentTradeJournal,
    PostTradeAnalyticsConfig,
    TradeReconstructionError,
    TradeSource,
)
from atlas_ai.backtest.models import BacktestMetrics, ExitReason
from atlas_ai.data.features import FeaturePipeline
from atlas_ai.data.integrity import get_timeframe_delta
from atlas_ai.domain.market import Timeframe
from atlas_ai.domain.risk import RiskParameters
from atlas_ai.domain.signals import SignalDirection
from atlas_ai.execution import (
    ExecutionSafetyConfig,
    LedgerHealthStatus,
    LedgerRecoveryLockError,
    MT5ConnectionAdapter,
    MT5ConnectionConfig,
    MT5ConnectionError,
    MT5DealEntry,
    MT5DealRecord,
    MT5DealType,
    MT5DemoExecutionEngine,
    MT5MarketDataAdapter,
    MT5OrderFillingMode,
    MT5OrderSendResult,
    MT5PositionRecord,
    MT5RateBar,
    MT5Retcode,
    MT5TerminalInfo,
    MockMT5TerminalBackend,
    OrderLedgerStatus,
    PersistenceAccountMismatchError,
    PersistenceCorruptionError,
    PersistentAccountRiskTracker,
    PersistentExecutionLedger,
    PreTradeSafetyGate,
    ReconciliationStatus,
    SymbolTick,
    TrustedBalanceAnchorConfig,
)
from atlas_ai.risk.engine import RiskEngine
from atlas_ai.runtime import (
    BarEvaluationState,
    CircuitBreakerLatchedError,
    DemoRuntimeConfig,
    DemoTradingOrchestrator,
    EmergencyAuditEvent,
    EmergencyEventType,
    EmergencyStopController,
    EmergencyStopError,
    EmergencyStopReason,
    EmergencyStopResetError,
    EmergencyStopState,
    EmergencyStopStatus,
    PersistentBarStateTracker,
    RuntimeCycleReport,
    RuntimeCycleStatus,
    RuntimeCycleStep,
    RuntimeOrchestrationError,
    RuntimeSessionSummary,
)
from atlas_ai.strategies.config import ConfluenceWeightsConfig, MultiTimeframeConfig
from atlas_ai.strategies.engine import StrategyEngine
from atlas_ai.strategies.exceptions import StrategyEvaluationError
from atlas_ai.strategies.models import RiskEvaluationStatus, SignalLifecycleState
from atlas_ai.strategies.signal_lifecycle import SignalLifecycleManager

FIXED_UTC = datetime(2026, 11, 25, 14, 0, 0, tzinfo=timezone.utc)
DEMO_LOGIN = 100200300
DEMO_SERVER = "AtlasBroker-Demo"


def _sample_backtest_metrics(total_trades: int = 12) -> BacktestMetrics:
    return BacktestMetrics(
        initial_balance=Decimal("10000.00"),
        final_balance=Decimal("9800.00"),
        final_equity=Decimal("9800.00"),
        total_trades=total_trades,
        winning_trades=4,
        losing_trades=total_trades - 4,
        breakeven_trades=0,
        win_rate_pct=Decimal("33.33"),
        gross_profit=Decimal("400.00"),
        gross_loss=Decimal("600.00"),
        total_commission=Decimal("12.00"),
        net_profit_usd=Decimal("-200.00"),
        net_profit_pct=Decimal("-2.0000"),
        profit_factor=Decimal("0.6667"),
        expectancy_usd=Decimal("-16.67"),
        expectancy_r=Decimal("-0.2500"),
        average_win_usd=Decimal("100.00"),
        average_loss_usd=Decimal("75.00"),
        payoff_ratio=Decimal("1.3333"),
        average_r_multiple=Decimal("-0.2500"),
        max_drawdown_usd=Decimal("350.00"),
        max_drawdown_pct=Decimal("3.5000"),
        sharpe_ratio=Decimal("-1.2000"),
        sortino_ratio=Decimal("-1.1000"),
    )


def test_demo_runtime_config_validation_and_initial_armed_state(tmp_path: Path) -> None:
    """Verify Requirement 1 & D:
    1. DemoRuntimeConfig validates symbol normalization, timeframe hierarchy (higher_timeframes > primary_timeframe),
       lookback bounds, dry_run_only mode, and circuit-breaker policies.
    2. Initial EmergencyStopController state is ARMED, is_halted == False, requires_human_reset == True,
       and assert_trading_authorized() succeeds.
    """
    cfg = DemoRuntimeConfig(
        symbol=" xauusd ",
        primary_timeframe=Timeframe.M15,
        higher_timeframes=(Timeframe.D1, Timeframe.H1, Timeframe.H4, Timeframe.H1),
        candle_lookback_bars=300,
        dry_run_only=True,
        emergency_state_path=tmp_path / "emergency_state.json",
    )
    assert cfg.symbol == "XAUUSD"
    # Higher timeframes deduplicated and sorted by ascending duration
    assert cfg.higher_timeframes == (Timeframe.H1, Timeframe.H4, Timeframe.D1)
    assert cfg.dry_run_only is True

    # Reject higher_timeframe <= primary_timeframe
    with pytest.raises(ValidationError, match="strictly larger than primary_timeframe"):
        DemoRuntimeConfig(
            primary_timeframe=Timeframe.H4,
            higher_timeframes=(Timeframe.H1, Timeframe.D1),
        )

    controller = EmergencyStopController(config=cfg)
    state = controller.load_or_initialize(
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
        now_utc=FIXED_UTC,
    )
    assert state.status == EmergencyStopStatus.ARMED
    assert state.is_halted is False
    assert state.requires_human_reset is True
    assert state.active_trigger_reason is None
    assert state.active_trigger_detail is None
    assert state.latched_at_utc is None
    assert len(state.audit_history) == 1
    assert state.audit_history[0].event_type == EmergencyEventType.INITIALIZED
    assert controller.is_halted(account_login=DEMO_LOGIN, server=DEMO_SERVER) is False

    auth_state = controller.assert_trading_authorized(
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
        now_utc=FIXED_UTC,
    )
    assert auth_state.status == EmergencyStopStatus.ARMED


def test_manual_emergency_latch_restart_persistence_and_no_auto_unlock(tmp_path: Path) -> None:
    """Verify Requirements 2, 7 & 13:
    1. trigger_manual_stop() latches LATCHED_HALTED and persists to disk immediately.
    2. Empty manual rationale is rejected with EmergencyStopError.
    3. assert_trading_authorized() raises CircuitBreakerLatchedError when latched.
    4. State survives process restart (new EmergencyStopController instance reading from disk)
       and NEVER automatically unlocks after restart.
    """
    state_path = tmp_path / "manual_emergency.json"
    cfg = DemoRuntimeConfig(emergency_state_path=state_path)
    controller = EmergencyStopController(config=cfg)

    with pytest.raises(EmergencyStopError, match="non-empty string"):
        controller.trigger_manual_stop(
            "   ",
            account_login=DEMO_LOGIN,
            server=DEMO_SERVER,
        )

    latched = controller.trigger_manual_stop(
        "Operator halted trading ahead of FOMC announcement.",
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
        now_utc=FIXED_UTC,
    )
    assert latched.status == EmergencyStopStatus.LATCHED_HALTED
    assert latched.is_halted is True
    assert latched.active_trigger_reason == EmergencyStopReason.MANUAL_EMERGENCY_STOP
    assert latched.latched_at_utc == FIXED_UTC
    assert controller.last_generation == 1
    assert state_path.exists()

    with pytest.raises(CircuitBreakerLatchedError, match="MANUAL_EMERGENCY_STOP"):
        controller.assert_trading_authorized(
            account_login=DEMO_LOGIN,
            server=DEMO_SERVER,
        )

    # Simulate process restart with a fresh EmergencyStopController instance
    restarted_controller = EmergencyStopController(config=cfg)
    reloaded = restarted_controller.load_or_initialize(
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
        now_utc=FIXED_UTC + timedelta(minutes=10),
    )
    assert reloaded.status == EmergencyStopStatus.LATCHED_HALTED
    assert reloaded.is_halted is True
    assert reloaded.active_trigger_reason == EmergencyStopReason.MANUAL_EMERGENCY_STOP
    assert reloaded.latched_at_utc == FIXED_UTC
    assert restarted_controller.last_generation == 1

    # Even if a healthy connection check occurs after restart, it MUST NOT auto-unlock!
    after_conn_ok = restarted_controller.record_connection_success(
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
        now_utc=FIXED_UTC + timedelta(minutes=11),
    )
    assert after_conn_ok.status == EmergencyStopStatus.LATCHED_HALTED
    assert after_conn_ok.is_halted is True
    with pytest.raises(CircuitBreakerLatchedError, match="MANUAL_EMERGENCY_STOP"):
        restarted_controller.assert_trading_authorized(
            account_login=DEMO_LOGIN,
            server=DEMO_SERVER,
        )


def test_automatic_drift_quarantine_execution_degradation_and_ledger_recovery_latches(
    tmp_path: Path,
) -> None:
    """Verify Requirements 3, 4 & 5:
    1. DriftStatus.QUARANTINE_RECOMMENDED automatically latches DRIFT_QUARANTINE_RECOMMENDED.
    2. DriftStatus.EXECUTION_DEGRADED automatically latches DRIFT_EXECUTION_DEGRADED.
    3. DriftStatus.HEALTHY / INSUFFICIENT_SAMPLE / WARNING_DRIFT_DETECTED do not latch.
    4. LedgerHealthStatus.RECOVERY_REQUIRED_LOCKED automatically latches LEDGER_RECOVERY_REQUIRED_LOCKED.
    """
    metrics = _sample_backtest_metrics(12)

    # 1. DriftStatus.QUARANTINE_RECOMMENDED
    ctrl_q = EmergencyStopController(
        DemoRuntimeConfig(emergency_state_path=tmp_path / "drift_q.json")
    )
    healthy_rep = DriftAndHealthReport(
        overall_status=DriftStatus.HEALTHY,
        total_trades=12,
        insufficient_sample=False,
        realized_expectancy_r=Decimal("0.3500"),
        max_consecutive_losses=1,
        overall_metrics=metrics,
    )
    st_healthy = ctrl_q.evaluate_drift_report(
        healthy_rep,
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
        now_utc=FIXED_UTC,
    )
    assert st_healthy.is_halted is False

    quarantine_rep = DriftAndHealthReport(
        overall_status=DriftStatus.QUARANTINE_RECOMMENDED,
        total_trades=12,
        insufficient_sample=False,
        realized_expectancy_r=Decimal("-0.2500"),
        max_consecutive_losses=5,
        quarantine_recommended_rule_ids=("rule-ict-bos-01",),
        warnings=("Knowledge rule quarantine recommended for 'rule-ict-bos-01'.",),
        overall_metrics=metrics,
    )
    st_quarantined = ctrl_q.evaluate_drift_report(
        quarantine_rep,
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
        now_utc=FIXED_UTC + timedelta(minutes=1),
    )
    assert st_quarantined.status == EmergencyStopStatus.LATCHED_HALTED
    assert st_quarantined.is_halted is True
    assert st_quarantined.active_trigger_reason == EmergencyStopReason.DRIFT_QUARANTINE_RECOMMENDED
    assert "rule-ict-bos-01" in (st_quarantined.active_trigger_detail or "")
    with pytest.raises(CircuitBreakerLatchedError, match="DRIFT_QUARANTINE_RECOMMENDED"):
        ctrl_q.assert_trading_authorized(account_login=DEMO_LOGIN, server=DEMO_SERVER)

    # 2. DriftStatus.EXECUTION_DEGRADED
    ctrl_exec = EmergencyStopController(
        DemoRuntimeConfig(emergency_state_path=tmp_path / "drift_exec.json")
    )
    exec_deg_rep = DriftAndHealthReport(
        overall_status=DriftStatus.EXECUTION_DEGRADED,
        total_trades=12,
        insufficient_sample=False,
        realized_expectancy_r=Decimal("0.1500"),
        max_consecutive_losses=2,
        mean_entry_slippage_points=Decimal("22.5000"),
        warnings=("Execution quality degraded: mean_entry_slippage_points (22.5000) > 10.0.",),
        overall_metrics=metrics,
    )
    st_exec_deg = ctrl_exec.evaluate_drift_report(
        exec_deg_rep,
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
        now_utc=FIXED_UTC + timedelta(minutes=2),
    )
    assert st_exec_deg.status == EmergencyStopStatus.LATCHED_HALTED
    assert st_exec_deg.is_halted is True
    assert st_exec_deg.active_trigger_reason == EmergencyStopReason.DRIFT_EXECUTION_DEGRADED
    with pytest.raises(CircuitBreakerLatchedError, match="DRIFT_EXECUTION_DEGRADED"):
        ctrl_exec.assert_trading_authorized(account_login=DEMO_LOGIN, server=DEMO_SERVER)

    # 3. LedgerHealthStatus.RECOVERY_REQUIRED_LOCKED
    ctrl_ledger = EmergencyStopController(
        DemoRuntimeConfig(emergency_state_path=tmp_path / "ledger_lock.json")
    )
    st_ledger_ok = ctrl_ledger.evaluate_ledger_health(
        LedgerHealthStatus.HEALTHY,
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
        now_utc=FIXED_UTC,
    )
    assert st_ledger_ok.is_halted is False

    st_ledger_locked = ctrl_ledger.evaluate_ledger_health(
        LedgerHealthStatus.RECOVERY_REQUIRED_LOCKED,
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
        detail="Interrupted .tmp write detected on execution_ledger.json.",
        now_utc=FIXED_UTC + timedelta(minutes=3),
    )
    assert st_ledger_locked.status == EmergencyStopStatus.LATCHED_HALTED
    assert st_ledger_locked.is_halted is True
    assert st_ledger_locked.active_trigger_reason == EmergencyStopReason.LEDGER_RECOVERY_REQUIRED_LOCKED
    with pytest.raises(CircuitBreakerLatchedError, match="LEDGER_RECOVERY_REQUIRED_LOCKED"):
        ctrl_ledger.assert_trading_authorized(account_login=DEMO_LOGIN, server=DEMO_SERVER)


def test_repeated_connection_and_persistence_failure_thresholds(tmp_path: Path) -> None:
    """Verify Requirement 6:
    1. Consecutive connection failures below threshold increment counter in ARMED state,
       reset to 0 on record_connection_success(), and latch REPEATED_CONNECTION_FAILURES
       immediately upon reaching max_consecutive_connection_failures.
    2. Consecutive persistence failures latch REPEATED_PERSISTENCE_FAILURES upon reaching
       max_consecutive_persistence_failures.
    """
    cfg = DemoRuntimeConfig(
        emergency_state_path=tmp_path / "failures.json",
        max_consecutive_connection_failures=3,
        max_consecutive_persistence_failures=2,
    )
    ctrl = EmergencyStopController(config=cfg)

    # 2 connection failures (< 3) -> still ARMED, counter=2
    s1 = ctrl.record_connection_failure(
        "IPC timeout #1",
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
        now_utc=FIXED_UTC,
    )
    assert s1.is_halted is False
    assert s1.consecutive_connection_failures == 1

    s2 = ctrl.record_connection_failure(
        "IPC timeout #2",
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
        now_utc=FIXED_UTC + timedelta(seconds=10),
    )
    assert s2.is_halted is False
    assert s2.consecutive_connection_failures == 2

    # Healthy connection check clears the counter back to 0 while ARMED
    s_cleared = ctrl.record_connection_success(
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
        now_utc=FIXED_UTC + timedelta(seconds=20),
    )
    assert s_cleared.is_halted is False
    assert s_cleared.consecutive_connection_failures == 0

    # Now record 3 consecutive connection failures -> latches on failure #3!
    for idx in range(1, 3):
        st = ctrl.record_connection_failure(
            f"Terminal disconnected #{idx}",
            account_login=DEMO_LOGIN,
            server=DEMO_SERVER,
            now_utc=FIXED_UTC + timedelta(seconds=30 + idx),
        )
        assert st.is_halted is False
        assert st.consecutive_connection_failures == idx

    s_latched_conn = ctrl.record_connection_failure(
        "Terminal disconnected #3",
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
        now_utc=FIXED_UTC + timedelta(seconds=35),
    )
    assert s_latched_conn.status == EmergencyStopStatus.LATCHED_HALTED
    assert s_latched_conn.is_halted is True
    assert s_latched_conn.active_trigger_reason == EmergencyStopReason.REPEATED_CONNECTION_FAILURES
    assert s_latched_conn.consecutive_connection_failures == 3

    # Test consecutive persistence failure threshold (max_consecutive_persistence_failures=2)
    ctrl_p = EmergencyStopController(
        DemoRuntimeConfig(
            emergency_state_path=tmp_path / "persist_failures.json",
            max_consecutive_persistence_failures=2,
        )
    )
    sp1 = ctrl_p.record_persistence_failure(
        "Lock timeout on account_risk_state.json #1",
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
        now_utc=FIXED_UTC,
    )
    assert sp1.is_halted is False
    assert sp1.consecutive_persistence_failures == 1

    sp2 = ctrl_p.record_persistence_failure(
        "Lock timeout on account_risk_state.json #2",
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
        now_utc=FIXED_UTC + timedelta(seconds=5),
    )
    assert sp2.status == EmergencyStopStatus.LATCHED_HALTED
    assert sp2.is_halted is True
    assert sp2.active_trigger_reason == EmergencyStopReason.REPEATED_PERSISTENCE_FAILURES
    assert sp2.consecutive_persistence_failures == 2


def test_corrupted_state_fail_closed_and_account_server_mismatch_rejection(
    tmp_path: Path,
) -> None:
    """Verify Requirements 8 & 9:
    1. Account login or server mismatch fails closed with PersistenceAccountMismatchError.
    2. Corrupted JSON, tampered payload checksum, or lingering interrupted .tmp.* write
       fails closed with PersistenceCorruptionError / LedgerRecoveryLockError.
    """
    state_path = tmp_path / "corrupt_and_binding.json"
    cfg = DemoRuntimeConfig(emergency_state_path=state_path)
    ctrl = EmergencyStopController(config=cfg)

    ctrl.trigger_manual_stop(
        "Initial manual stop for binding test",
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
        now_utc=FIXED_UTC,
    )

    # 1. Account login mismatch -> PersistenceAccountMismatchError
    with pytest.raises(PersistenceAccountMismatchError):
        ctrl.assert_trading_authorized(
            account_login=999999999,
            server=DEMO_SERVER,
        )

    # Server mismatch -> PersistenceAccountMismatchError
    with pytest.raises(PersistenceAccountMismatchError):
        ctrl.assert_trading_authorized(
            account_login=DEMO_LOGIN,
            server="WrongServer-Demo",
        )

    # 2. Interrupted .tmp.* write present -> fails closed with LedgerRecoveryLockError
    lingering_tmp = tmp_path / f"{state_path.name}.tmp.9999.deadbeef"
    lingering_tmp.write_text("interrupted write", encoding="utf-8")
    with pytest.raises(LedgerRecoveryLockError):
        ctrl.is_halted(account_login=DEMO_LOGIN, server=DEMO_SERVER)
    lingering_tmp.unlink()

    # 3. Corrupted primary state file -> fails closed with PersistenceCorruptionError
    saved_bytes = state_path.read_bytes()
    state_path.write_text("{corrupted_json_payload", encoding="utf-8")
    with pytest.raises(PersistenceCorruptionError):
        ctrl.assert_trading_authorized(account_login=DEMO_LOGIN, server=DEMO_SERVER)

    # Restore valid bytes and verify latched state is still intact
    state_path.write_bytes(saved_bytes)
    with pytest.raises(CircuitBreakerLatchedError):
        ctrl.assert_trading_authorized(account_login=DEMO_LOGIN, server=DEMO_SERVER)


def test_explicit_reset_with_rationale_empty_rejection_and_audit_history_preservation(
    tmp_path: Path,
) -> None:
    """Verify Requirements 10 & 11:
    1. reset_emergency_stop() rejects empty or whitespace operator_rationale with EmergencyStopResetError
       without unlatching the circuit breaker.
    2. Explicit reset with valid operator_rationale transitions LATCHED_HALTED -> ARMED, clears
       active trigger and failure counters, persists durably across restarts, and preserves all
       previous emergency events in audit_history.
    """
    state_path = tmp_path / "reset_audit.json"
    cfg = DemoRuntimeConfig(emergency_state_path=state_path)
    ctrl = EmergencyStopController(config=cfg)

    ctrl.trigger_manual_stop(
        "Emergency halt #1: volatile news spike.",
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
        now_utc=FIXED_UTC,
    )
    assert ctrl.is_halted(account_login=DEMO_LOGIN, server=DEMO_SERVER) is True

    # Empty or whitespace rationale must fail closed with EmergencyStopResetError
    for bad_rationale in ("", "   ", "\t\n"):
        with pytest.raises(EmergencyStopResetError, match="non-empty operator_rationale"):
            ctrl.reset_emergency_stop(
                bad_rationale,
                account_login=DEMO_LOGIN,
                server=DEMO_SERVER,
            )

    # Circuit breaker must STILL be latched after failed reset attempts
    assert ctrl.is_halted(account_login=DEMO_LOGIN, server=DEMO_SERVER) is True

    # Valid operator reset
    reset_ts = FIXED_UTC + timedelta(minutes=30)
    reset_state = ctrl.reset_emergency_stop(
        "Verified spread normalized and ledger reconciled cleanly after FOMC.",
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
        now_utc=reset_ts,
    )
    assert reset_state.status == EmergencyStopStatus.ARMED
    assert reset_state.is_halted is False
    assert reset_state.active_trigger_reason is None
    assert reset_state.active_trigger_detail is None
    assert reset_state.latched_at_utc is None
    assert reset_state.last_reset_at_utc == reset_ts
    assert (
        reset_state.last_reset_rationale
        == "Verified spread normalized and ledger reconciled cleanly after FOMC."
    )
    assert ctrl.last_generation == 2

    # Verify audit history preserved INITIALIZED (1), LATCHED (2), and OPERATOR_RESET (3)
    assert len(reset_state.audit_history) == 3
    assert [ev.event_type for ev in reset_state.audit_history] == [
        EmergencyEventType.INITIALIZED,
        EmergencyEventType.LATCHED,
        EmergencyEventType.OPERATOR_RESET,
    ]
    assert [ev.event_id for ev in reset_state.audit_history] == [1, 2, 3]
    assert reset_state.audit_history[1].trigger_reason == EmergencyStopReason.MANUAL_EMERGENCY_STOP
    assert (
        reset_state.audit_history[2].operator_rationale
        == "Verified spread normalized and ledger reconciled cleanly after FOMC."
    )

    # Verify second latch after reset appends event #4 while preserving events 1..3 across restart
    ctrl.evaluate_ledger_health(
        LedgerHealthStatus.RECOVERY_REQUIRED_LOCKED,
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
        detail="Ledger recovery lock on restart check.",
        now_utc=FIXED_UTC + timedelta(minutes=45),
    )

    restarted_ctrl = EmergencyStopController(config=cfg)
    reloaded = restarted_ctrl.get_state(account_login=DEMO_LOGIN, server=DEMO_SERVER)
    assert reloaded.is_halted is True
    assert reloaded.active_trigger_reason == EmergencyStopReason.LEDGER_RECOVERY_REQUIRED_LOCKED
    assert len(reloaded.audit_history) == 4
    assert [ev.event_id for ev in reloaded.audit_history] == [1, 2, 3, 4]
    assert reloaded.last_reset_at_utc == reset_ts


def test_concurrent_state_transition_safety(tmp_path: Path) -> None:
    """Verify Requirement 12:
    Concurrent threads recording connection failures and triggering emergency stops on
    EmergencyStopController serialize cleanly through RLock + OS FileLock without corrupting
    the persistence hash chain, losing generations, or violating audit_history event_id monotonicity.
    """
    state_path = tmp_path / "concurrent_emergency.json"
    cfg = DemoRuntimeConfig(
        emergency_state_path=state_path,
        max_consecutive_connection_failures=4,
        lock_timeout_seconds=10.0,
    )
    ctrl = EmergencyStopController(config=cfg)

    def _worker(worker_idx: int) -> EmergencyStopState:
        ts = FIXED_UTC + timedelta(seconds=worker_idx)
        if worker_idx % 3 == 0:
            return ctrl.trigger_manual_stop(
                f"Concurrent worker {worker_idx} manual stop",
                account_login=DEMO_LOGIN,
                server=DEMO_SERVER,
                now_utc=ts,
            )
        return ctrl.record_connection_failure(
            f"Concurrent worker {worker_idx} connection timeout",
            account_login=DEMO_LOGIN,
            server=DEMO_SERVER,
            now_utc=ts,
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(_worker, range(1, 13)))

    assert len(results) == 12

    # Reload via a fresh instance to verify on-disk envelope checksum, generation, and audit sequence
    verifier = EmergencyStopController(config=cfg)
    final_state = verifier.load_or_initialize(
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
        now_utc=FIXED_UTC + timedelta(minutes=5),
    )
    assert final_state.status == EmergencyStopStatus.LATCHED_HALTED
    assert final_state.is_halted is True
    assert verifier.last_generation == 12
    assert len(final_state.audit_history) == 13  # 1 INITIALIZED + 12 concurrent events
    assert [ev.event_id for ev in final_state.audit_history] == list(range(1, 14))


# =============================================================================
# PHASE 9.2 — CLOSED-LOOP DEMO TRADING ORCHESTRATOR OFFLINE TESTS (20 SCENARIOS)
# =============================================================================

ALL_14_CYCLE_STEPS: tuple[RuntimeCycleStep, ...] = (
    RuntimeCycleStep.STEP_01_VERIFY_CONNECTION,
    RuntimeCycleStep.STEP_02_RECONCILE_UNRESOLVED_ORDERS,
    RuntimeCycleStep.STEP_03_SYNC_CLOSED_TRADES,
    RuntimeCycleStep.STEP_04_RECONSTRUCT_ACCOUNT_STATE,
    RuntimeCycleStep.STEP_05_EVALUATE_DRIFT_AND_LEDGER_HEALTH,
    RuntimeCycleStep.STEP_06_EVALUATE_EMERGENCY_STOP_STATE,
    RuntimeCycleStep.STEP_07_ASSERT_TRADING_AUTHORIZED,
    RuntimeCycleStep.STEP_08_CHECK_BAR_CLOSE_DEDUPLICATION,
    RuntimeCycleStep.STEP_09_FETCH_SYNCHRONIZED_MARKET_DATA,
    RuntimeCycleStep.STEP_10_RUN_FEATURE_PIPELINE,
    RuntimeCycleStep.STEP_11_RUN_STRATEGY_ENGINE,
    RuntimeCycleStep.STEP_12_EVALUATE_RISK,
    RuntimeCycleStep.STEP_13_STAGE_JOURNAL_CONTEXT,
    RuntimeCycleStep.STEP_14_EXECUTE_OR_DRY_RUN_AND_COMMIT_BAR,
)


def _seed_trending_rates_and_tick(
    mock: MockMT5TerminalBackend,
    *,
    now_utc: datetime = FIXED_UTC,
    direction: str = "up",
    bars: int = 90,
) -> datetime:
    """Seed deterministic closed H1, H4, and D1 bars + synchronized live Bid/Ask tick into `mock`.

    Returns the latest closed H1 bar open UTC timestamp (`now_utc - 1h`).
    """
    for tf in (Timeframe.H1, Timeframe.H4, Timeframe.D1):
        delta = get_timeframe_delta(tf)
        rate_bars: list[MT5RateBar] = []
        # Generate `bars` closed candles ending at `now_utc - delta` (so bar_open + delta <= now_utc),
        # plus 1 forming candle at `now_utc` (index 0 in MT5 copy_rates_from_pos, excluded by start_pos=1).
        price = Decimal("2560.00") if direction == "up" else Decimal("2740.00")
        step = (
            Decimal("1.00")
            if direction == "up"
            else (Decimal("-1.00") if direction == "down" else Decimal("0.00"))
        )
        for idx in range(bars):
            bar_open = now_utc - delta * (bars - idx)
            if direction == "range":
                if idx == bars - 1:
                    o = Decimal("2650.00")
                    c = Decimal("2650.00")
                else:
                    o = Decimal("2650.00") + (Decimal("0.40") if idx % 2 == 0 else Decimal("-0.40"))
                    c = Decimal("2650.00") + (Decimal("-0.40") if idx % 2 == 0 else Decimal("0.40"))
                h = max(o, c) + Decimal("0.80")
                l = min(o, c) - Decimal("0.80")
            else:
                o = price
                c = price + step
                h = max(o, c) + Decimal("0.40")
                l = min(o, c) - Decimal("0.40")
                price = c

            rate_bars.append(
                MT5RateBar(
                    timestamp=bar_open,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    tick_volume=Decimal("250"),
                    spread=20,
                )
            )

        last_closed_c = rate_bars[-1].close
        # Append currently forming bar at index 0 (timestamp == now_utc)
        rate_bars.append(
            MT5RateBar(
                timestamp=now_utc,
                open=last_closed_c,
                high=last_closed_c + Decimal("0.30"),
                low=last_closed_c - Decimal("0.30"),
                close=last_closed_c,
                tick_volume=Decimal("50"),
                spread=20,
            )
        )
        mock.set_rates("XAUUSD", tf, rate_bars)

    h1_closed_close = mock._rates[("XAUUSD", Timeframe.H1)][-2].close
    mock.set_tick(
        SymbolTick(
            symbol="XAUUSD",
            timestamp=now_utc,
            bid=h1_closed_close - Decimal("0.10"),
            ask=h1_closed_close + Decimal("0.10"),
            last=h1_closed_close,
            volume=Decimal("100"),
        )
    )
    return now_utc - timedelta(hours=1)


def _build_phase92_orchestrator(
    tmp_path: Path,
    *,
    tag: str = "p92",
    dry_run_only: bool = False,
    direction: str = "up",
    now_utc: datetime = FIXED_UTC,
    risk_parameters: RiskParameters | None = None,
    max_consecutive_connection_failures: int = 3,
    max_consecutive_persistence_failures: int = 2,
    require_staged_context: bool = True,
) -> tuple[DemoTradingOrchestrator, MockMT5TerminalBackend, datetime]:
    """Construct a complete, 100% offline Phase 1–9.2 DemoTradingOrchestrator stack."""
    mock = MockMT5TerminalBackend()
    latest_closed_bar_ts = _seed_trending_rates_and_tick(
        mock,
        now_utc=now_utc,
        direction=direction,
        bars=90,
    )
    conn = MT5ConnectionAdapter(
        backend=mock,
        config=MT5ConnectionConfig(
            login=DEMO_LOGIN,
            server=DEMO_SERVER,
            expected_login=DEMO_LOGIN,
            expected_server=DEMO_SERVER,
        ),
    )
    conn.connect()

    origin_anchor = TrustedBalanceAnchorConfig(
        anchor_balance=Decimal("0.00"),
        anchor_timestamp_utc=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        require_initial_deposit_deal=True,
    )
    safety_cfg = ExecutionSafetyConfig(
        account_risk_state_path=tmp_path / f"{tag}_account_risk.json",
        ledger_path=tmp_path / f"{tag}_execution_ledger.json",
        trusted_balance_anchor=origin_anchor,
    )
    analytics_cfg = PostTradeAnalyticsConfig(
        journal_path=tmp_path / f"{tag}_trade_journal.json",
    )
    runtime_cfg = DemoRuntimeConfig(
        symbol="XAUUSD",
        primary_timeframe=Timeframe.H1,
        higher_timeframes=(Timeframe.H4, Timeframe.D1),
        candle_lookback_bars=80,
        dry_run_only=dry_run_only,
        max_consecutive_connection_failures=max_consecutive_connection_failures,
        max_consecutive_persistence_failures=max_consecutive_persistence_failures,
        emergency_state_path=tmp_path / f"{tag}_emergency_state.json",
    )

    md = MT5MarketDataAdapter(connection=conn, safety_config=safety_cfg)
    tracker = PersistentAccountRiskTracker(connection=conn, safety_config=safety_cfg)
    ledger = PersistentExecutionLedger(safety_config=safety_cfg)
    active_risk_params = risk_parameters or RiskParameters()
    risk_engine = RiskEngine(risk_parameters=active_risk_params)
    lifecycle = SignalLifecycleManager()

    # Configure StrategyEngine with EMA trend + MTF alignment enabled so a clean uptrend emits BUY
    confluence_cfg = ConfluenceWeightsConfig(
        enable_ema_trend=True,
        enable_market_structure=False,
        enable_liquidity_sweep=False,
        enable_fvg=False,
        enable_fvg_retest=False,
        enable_premium_discount=False,
        enable_pdh_pdl=False,
        enable_kill_zone=False,
        enable_mtf=True,
        enable_news_filter=True,
        weight_ema_trend=Decimal("0.60"),
        weight_mtf=Decimal("0.40"),
        minimum_confidence_threshold=Decimal("0.55"),
        minimum_directional_edge=Decimal("0.15"),
        minimum_agreeing_directional_modules=1,
    )
    mtf_cfg = MultiTimeframeConfig(
        htf_timeframe=Timeframe.D1,
        mtf_timeframe=Timeframe.H4,
        ltf_timeframe=Timeframe.H1,
        precision_timeframe=None,
        require_precision_timeframe=False,
    )
    strategy_engine = StrategyEngine(
        confluence_config=confluence_cfg,
        mtf_config=mtf_cfg,
    )

    gate = PreTradeSafetyGate(
        connection=conn,
        market_data=md,
        account_tracker=tracker,
        ledger=ledger,
        risk_engine=risk_engine,
        safety_config=safety_cfg,
        lifecycle_manager=lifecycle,
    )
    exec_engine = MT5DemoExecutionEngine(
        connection=conn,
        market_data=md,
        account_tracker=tracker,
        ledger=ledger,
        risk_engine=risk_engine,
        safety_config=safety_cfg,
        lifecycle_manager=lifecycle,
        safety_gate=gate,
    )
    journal = PersistentTradeJournal(config=analytics_cfg)
    reconstructor = DemoTradeReconstructor(
        connection=conn,
        ledger=ledger,
        journal=journal,
        lifecycle_manager=lifecycle,
        config=analytics_cfg,
        execution_engine=exec_engine,
    )
    drift_engine = DriftDetectionEngine(config=analytics_cfg)
    emergency_stop = EmergencyStopController(config=runtime_cfg)

    orchestrator = DemoTradingOrchestrator(
        config=runtime_cfg,
        connection=conn,
        market_data=md,
        account_tracker=tracker,
        ledger=ledger,
        execution_engine=exec_engine,
        journal=journal,
        trade_reconstructor=reconstructor,
        drift_engine=drift_engine,
        emergency_stop=emergency_stop,
        strategy_engine=strategy_engine,
        risk_engine=risk_engine,
        lifecycle_manager=lifecycle,
        bar_state_path=tmp_path / f"{tag}_bar_state.json",
        require_staged_context=require_staged_context,
    )
    return orchestrator, mock, latest_closed_bar_ts


def test_phase92_exact_14_step_cycle_ordering_and_full_offline_execution(
    tmp_path: Path,
) -> None:
    """Scenarios 1 & 20:
    1. Exact 14-step cycle ordering verified via step observer and report.executed_steps.
    20. Full offline end-to-end integration with MockMT5TerminalBackend: executes BUY order,
        persists write-ahead ledger entry, persists pre-send EntryContextSnapshot in journal,
        transitions SignalLifecycleManager to ACTIVE, and commits evaluated bar timestamp.
    """
    orchestrator, mock, latest_bar_ts = _build_phase92_orchestrator(
        tmp_path,
        tag="s01_order",
        direction="up",
    )
    observed_steps: list[RuntimeCycleStep] = []
    orchestrator.set_step_observer(observed_steps.append)

    report = orchestrator.run_single_cycle(
        reference_now_utc=FIXED_UTC,
        client_order_id="ord-cycle-01",
    )

    assert report.status == RuntimeCycleStatus.EXECUTED_ORDER
    assert report.executed_steps == ALL_14_CYCLE_STEPS
    assert tuple(observed_steps) == ALL_14_CYCLE_STEPS
    assert report.latest_closed_bar_timestamp_utc == latest_bar_ts
    assert report.previous_evaluated_bar_timestamp_utc is None
    assert report.committed_evaluated_bar_timestamp_utc == latest_bar_ts
    assert report.signal is not None and report.signal.direction == SignalDirection.BUY
    assert report.risk_status == RiskEvaluationStatus.RISK_APPROVED
    assert report.execution_result is not None
    assert report.execution_result.executed is True
    assert report.execution_result.client_order_id == "ord-cycle-01"
    assert report.execution_result.ledger_entry.status == OrderLedgerStatus.ACTIVE
    assert report.signal_id is not None
    assert (
        orchestrator.lifecycle_manager.get_record(report.signal_id).state
        == SignalLifecycleState.ACTIVE
    )

    # Verify EntryContextSnapshot was persisted to disk BEFORE order_send()
    persisted_ctx = orchestrator.journal.get_entry_context("ord-cycle-01")
    assert persisted_ctx is not None
    assert persisted_ctx.signal_id == report.signal_id
    assert len(mock.positions_get("XAUUSD") or ()) == 1


def test_phase92_startup_recovery_unresolved_order_reconciliation_closed_trade_sync_and_account_state(
    tmp_path: Path,
) -> None:
    """Scenarios 2, 3, 4 & 5:
    2. Startup recovery: fresh orchestrator loads persisted ledger, journal, account risk, and bar state.
    3. Unresolved order reconciliation: resolves an interrupted SUBMITTING order discovered on broker.
    4. Closed trade synchronization: reconstructs a newly closed trade into PersistentTradeJournal.
    5. Account state reconstruction: updates realized P&L and equity before strategy evaluation.
    """
    orchestrator, mock, latest_bar_ts = _build_phase92_orchestrator(
        tmp_path,
        tag="s02_recovery",
        direction="up",
    )

    # Seed an unresolved SUBMITTING intent in the ledger before startup cycle, plus its filled deal & closed exit deal on broker
    orchestrator.ledger.create_intent(
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
        client_order_id="ord-unresolved-1",
        signal_id="sig-unresolved-1",
        symbol="XAUUSD",
        timeframe=Timeframe.H1,
        direction=SignalDirection.BUY,
        magic=202607,
        requested_volume=Decimal("0.10"),
        requested_price=Decimal("2640.00"),
        stop_loss=Decimal("2630.00"),
        take_profit=Decimal("2660.00"),
        filling_mode=MT5OrderFillingMode.ORDER_FILLING_FOK,
        now_utc=FIXED_UTC - timedelta(hours=2),
    )
    orchestrator.ledger.transition_entry(
        "ord-unresolved-1",
        OrderLedgerStatus.VALIDATED,
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
        now_utc=FIXED_UTC - timedelta(hours=2),
        reason="Validated before crash.",
    )
    orchestrator.ledger.transition_entry(
        "ord-unresolved-1",
        OrderLedgerStatus.SUBMITTING,
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
        now_utc=FIXED_UTC - timedelta(hours=2),
        reconciliation_status=ReconciliationStatus.PENDING,
        reason="Crash right after SUBMITTING write-ahead.",
    )

    # Broker actually filled ord-unresolved-1 at 2640.00 and later hit TP at 2660.00 (+200.00 profit)
    base_deals = list(
        mock.history_deals_get(
            datetime(1970, 1, 1, tzinfo=timezone.utc),
            FIXED_UTC + timedelta(days=1),
        )
        or ()
    )
    entry_deal = MT5DealRecord(
        ticket=81001,
        order=80001,
        position_id=82001,
        time_utc=FIXED_UTC - timedelta(hours=2),
        deal_type=int(MT5DealType.DEAL_TYPE_BUY),
        entry=int(MT5DealEntry.DEAL_ENTRY_IN),
        magic=202607,
        symbol="XAUUSD",
        volume=Decimal("0.10"),
        price=Decimal("2640.00"),
        profit=Decimal("0.00"),
        comment="ord-unresolved-1",
    )
    exit_deal = MT5DealRecord(
        ticket=81002,
        order=80002,
        position_id=82001,
        time_utc=FIXED_UTC - timedelta(hours=1, minutes=15),
        deal_type=int(MT5DealType.DEAL_TYPE_SELL),
        entry=int(MT5DealEntry.DEAL_ENTRY_OUT),
        magic=202607,
        symbol="XAUUSD",
        volume=Decimal("0.10"),
        price=Decimal("2660.00"),
        profit=Decimal("200.00"),
        comment="ord-unresolved-1[tp]",
    )
    mock.set_deals([*base_deals, entry_deal, exit_deal])
    acct = mock.account_info()
    assert acct is not None
    mock.set_account(
        acct.model_copy(
            update={
                "balance": Decimal("10200.00"),
                "equity": Decimal("10200.00"),
                "margin_free": Decimal("10200.00"),
            }
        )
    )

    # Run startup cycle: Step 2 reconciles ord-unresolved-1, Step 3 reconstructs closed trade, Step 4 reconstructs account state (10200.00)
    report = orchestrator.run_single_cycle(
        reference_now_utc=FIXED_UTC,
        client_order_id="ord-after-recovery-1",
    )
    assert "ord-unresolved-1" in report.reconciliation_report.reconciled_executed_ids
    assert len(report.trade_reconstruction_report.newly_completed_trades) == 1
    completed_tr = report.trade_reconstruction_report.newly_completed_trades[0]
    assert completed_tr.client_order_id == "ord-unresolved-1"
    assert completed_tr.net_pnl == Decimal("200.00")
    assert report.account_state.balance == Decimal("10200.00")
    assert report.account_state.current_daily_pnl == Decimal("200.00")
    assert report.status == RuntimeCycleStatus.EXECUTED_ORDER
    assert report.committed_evaluated_bar_timestamp_utc == latest_bar_ts


def test_phase92_drift_triggered_emergency_stop_and_ledger_recovery_lock_halt(
    tmp_path: Path,
) -> None:
    """Scenarios 6, 7 & 11:
    6. Drift-triggered emergency stop: when DriftDetectionEngine reports QUARANTINE_RECOMMENDED
       (e.g., 5 consecutive losing trades >= max_consecutive_losses_threshold=4), Step 6 latches
       EmergencyStopController and Step 7 raises CircuitBreakerLatchedError before Step 8/14.
    7. Ledger recovery lock: lingering .tmp.* write on ledger triggers LedgerRecoveryLockError,
       latches LEDGER_RECOVERY_REQUIRED_LOCKED in EmergencyStopController, and blocks execution.
    11. Emergency stop blocks execution: zero order_send calls occur when emergency stop is latched.
    """
    orchestrator, mock, _ = _build_phase92_orchestrator(
        tmp_path,
        tag="s06_drift_halt",
        direction="up",
    )
    order_send_calls = 0

    def _counting_hook(req):
        nonlocal order_send_calls
        order_send_calls += 1
        return None

    mock.set_order_send_hook(_counting_hook)

    # Record 10 consecutive losing completed trades (>= min_trades_for_drift_evaluation=10)
    # in journal so DriftDetectionEngine emits QUARANTINE_RECOMMENDED
    for i in range(1, 11):
        t_in = FIXED_UTC - timedelta(hours=15 - i)
        t_out = t_in + timedelta(minutes=30)
        orchestrator.journal.record_completed_trade(
            CompletedTradeRecord(
                trade_id=f"tr-loss-{i}",
                source=TradeSource.MT5_DEMO,
                client_order_id=f"ord-loss-{i}",
                signal_id=f"sig-loss-{i}",
                position_ticket=90000 + i,
                entry_deal_tickets=(93000 + i,),
                exit_deal_tickets=(94000 + i,),
                symbol="XAUUSD",
                timeframe=Timeframe.H1,
                direction=SignalDirection.BUY,
                strategy_id="smc_confluence_v1",
                entry_timestamp_utc=t_in,
                exit_timestamp_utc=t_out,
                volume=Decimal("0.10"),
                requested_entry_price=Decimal("2650.00"),
                entry_fill_price=Decimal("2650.00"),
                requested_exit_price=Decimal("2640.00"),
                exit_fill_price=Decimal("2640.00"),
                stop_loss=Decimal("2640.00"),
                take_profit=Decimal("2670.00"),
                initial_risk_amount=Decimal("100.00"),
                gross_pnl=Decimal("-100.00"),
                commission=Decimal("0.00"),
                fee=Decimal("0.00"),
                swap=Decimal("0.00"),
                net_pnl=Decimal("-100.00"),
                realized_r_multiple=Decimal("-1.0000"),
                exit_reason=ExitReason.STOP_LOSS,
                context_attribution_status=ContextAttributionStatus.UNATTRIBUTED_MISSING_CONTEXT,
            ),
            account_login=DEMO_LOGIN,
            server=DEMO_SERVER,
            now_utc=t_out,
        )

    with pytest.raises(CircuitBreakerLatchedError, match="DRIFT_QUARANTINE_RECOMMENDED"):
        orchestrator.run_single_cycle(reference_now_utc=FIXED_UTC)

    assert order_send_calls == 0
    em_state = orchestrator.emergency_stop.get_state(
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
    )
    assert em_state.is_halted is True
    assert em_state.active_trigger_reason == EmergencyStopReason.DRIFT_QUARANTINE_RECOMMENDED

    # Subsequent cycle while latched must ALSO fail closed at Step 7 with zero order_send calls
    with pytest.raises(CircuitBreakerLatchedError, match="DRIFT_QUARANTINE_RECOMMENDED"):
        orchestrator.run_single_cycle(reference_now_utc=FIXED_UTC + timedelta(minutes=5))
    assert order_send_calls == 0

    # Scenario 7: Ledger recovery lock (.tmp.* file on ledger) latches LEDGER_RECOVERY_REQUIRED_LOCKED
    orch_lock, mock_lock, _ = _build_phase92_orchestrator(
        tmp_path,
        tag="s07_ledger_lock",
        direction="up",
    )
    ledger_path = orch_lock.ledger.safety_config.ledger_path
    lingering_tmp = ledger_path.parent / f"{ledger_path.name}.tmp.1234.cafebabe"
    lingering_tmp.write_text("interrupted write", encoding="utf-8")

    with pytest.raises((LedgerRecoveryLockError, PersistenceCorruptionError)):
        orch_lock.run_single_cycle(reference_now_utc=FIXED_UTC)

    lingering_tmp.unlink()
    locked_em_state = orch_lock.emergency_stop.get_state(
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
    )
    assert locked_em_state.is_halted is True
    assert (
        locked_em_state.active_trigger_reason
        == EmergencyStopReason.LEDGER_RECOVERY_REQUIRED_LOCKED
    )


def test_phase92_bar_close_deduplication_duplicate_polling_and_restart_safe_bar_state(
    tmp_path: Path,
) -> None:
    """Scenarios 8, 9 & 12:
    8. Bar-close deduplication: evaluating the same closed H1 bar twice skips Steps 9–14 and returns
       SKIPPED_DUPLICATE_BAR while STILL executing Steps 1–8 (connection, reconciliation, closed-trade
       sync, account state, drift, emergency stop).
    9. Restart-safe bar state: a new DemoTradingOrchestrator instance started after process restart
       reads `bar_state.json` from disk and refuses to re-evaluate the already-committed bar.
    12. Duplicate polling behavior: advances only when a new H1 candle closes (`FIXED_UTC + 1h`).
    """
    orchestrator, mock, first_bar_ts = _build_phase92_orchestrator(
        tmp_path,
        tag="s08_bar_dedup",
        direction="up",
    )

    # First poll at FIXED_UTC evaluates first_bar_ts (13:00 UTC) and executes order
    rep1 = orchestrator.run_single_cycle(
        reference_now_utc=FIXED_UTC,
        client_order_id="ord-bar-1",
    )
    assert rep1.status == RuntimeCycleStatus.EXECUTED_ORDER
    assert rep1.committed_evaluated_bar_timestamp_utc == first_bar_ts

    # Duplicate poll 15 minutes later (same closed H1 bar 13:00 UTC) -> SKIPPED_DUPLICATE_BAR
    rep2 = orchestrator.run_single_cycle(
        reference_now_utc=FIXED_UTC + timedelta(minutes=15),
    )
    assert rep2.status == RuntimeCycleStatus.SKIPPED_DUPLICATE_BAR
    assert rep2.executed_steps == ALL_14_CYCLE_STEPS[:8]
    assert rep2.latest_closed_bar_timestamp_utc == first_bar_ts
    assert rep2.previous_evaluated_bar_timestamp_utc == first_bar_ts
    assert len(mock.positions_get("XAUUSD") or ()) == 1

    # Simulate process restart with a brand-new PersistentBarStateTracker & DemoTradingOrchestrator reading disk
    restarted_bar_tracker = PersistentBarStateTracker(
        orchestrator.bar_state_tracker.state_path
    )
    already_eval, persisted_ts = restarted_bar_tracker.is_bar_already_evaluated(
        "XAUUSD",
        Timeframe.H1,
        first_bar_ts,
        account_login=DEMO_LOGIN,
        server=DEMO_SERVER,
    )
    assert already_eval is True
    assert persisted_ts == first_bar_ts

    # Verify rollback protection on PersistentBarStateTracker
    with pytest.raises(Exception, match="Cannot regress evaluated bar timestamp"):
        restarted_bar_tracker.commit_evaluated_bar_timestamp(
            "XAUUSD",
            Timeframe.H1,
            first_bar_ts - timedelta(hours=1),
            account_login=DEMO_LOGIN,
            server=DEMO_SERVER,
        )

    # Advance clock by 1 full hour (`FIXED_UTC + 1h`) and seed the new closed H1 bar (14:00 UTC)
    next_now = FIXED_UTC + timedelta(hours=1)
    second_bar_ts = _seed_trending_rates_and_tick(
        mock,
        now_utc=next_now,
        direction="range",  # Ranging on the new bar -> emits WAIT (NO_EXECUTABLE_SIGNAL)
        bars=90,
    )
    assert second_bar_ts == first_bar_ts + timedelta(hours=1)

    rep3 = orchestrator.run_single_cycle(reference_now_utc=next_now)
    assert rep3.status == RuntimeCycleStatus.NO_EXECUTABLE_SIGNAL
    assert rep3.previous_evaluated_bar_timestamp_utc == first_bar_ts
    assert rep3.committed_evaluated_bar_timestamp_utc == second_bar_ts


def test_phase92_dry_run_shadow_mode_zero_order_send_calls(tmp_path: Path) -> None:
    """Scenario 10:
    Dry-run / shadow mode (`config.dry_run_only == True`) executes Steps 1–13 (connection,
    reconciliation, trade sync, account reconstruction, drift, emergency check, market data,
    features, strategy analysis, risk approval, journal staging) and commits the evaluated bar
    timestamp at Step 14, while making ZERO calls to `backend.order_send()`.
    """
    orchestrator, mock, latest_bar_ts = _build_phase92_orchestrator(
        tmp_path,
        tag="s10_dry_run",
        dry_run_only=True,
        direction="up",
    )
    order_send_invocations = 0

    def _forbidden_send_hook(req):
        nonlocal order_send_invocations
        order_send_invocations += 1
        raise AssertionError("backend.order_send() must NEVER be called in dry_run_only mode!")

    mock.set_order_send_hook(_forbidden_send_hook)

    report = orchestrator.run_single_cycle(reference_now_utc=FIXED_UTC)
    assert report.status == RuntimeCycleStatus.DRY_RUN_EVALUATED
    assert report.dry_run_only is True
    assert report.executed_steps == ALL_14_CYCLE_STEPS
    assert report.signal is not None and report.signal.direction == SignalDirection.BUY
    assert report.risk_status == RiskEvaluationStatus.RISK_APPROVED
    assert report.execution_result is None
    assert report.committed_evaluated_bar_timestamp_utc == latest_bar_ts
    assert order_send_invocations == 0
    assert len(mock.positions_get("XAUUSD") or ()) == 0
    assert len(orchestrator.ledger.list_entries()) == 0


def test_phase92_connection_and_persistence_failure_handling_and_graceful_shutdown(
    tmp_path: Path,
) -> None:
    """Scenarios 13, 14 & 15:
    13. Connection failure handling: consecutive connection failures increment EmergencyStopController
        counter and latch REPEATED_CONNECTION_FAILURES at threshold.
    14. Persistence failure handling: corrupted bar state file increments EmergencyStopController
        persistence failure counter and latches REPEATED_PERSISTENCE_FAILURES at threshold without
        advancing bar state.
    15. Graceful shutdown: calling `request_shutdown()` during `run_session()` stops cleanly at the
        cycle boundary and sets `graceful_shutdown_requested == True`.
    """
    # Scenario 13: Consecutive connection failures (max_consecutive_connection_failures=2)
    orch_conn, mock_conn, _ = _build_phase92_orchestrator(
        tmp_path,
        tag="s13_conn_fail",
        max_consecutive_connection_failures=2,
    )
    # Mark terminal permanently disconnected so every reconnect attempt fails with MT5ConnectionError
    mock_conn.set_terminal(
        MT5TerminalInfo(
            connected=False,
            trade_allowed=True,
            tradeapi_disabled=False,
        )
    )
    with pytest.raises(MT5ConnectionError):
        orch_conn.run_single_cycle(reference_now_utc=FIXED_UTC)

    st1 = orch_conn.emergency_stop.get_state(account_login=DEMO_LOGIN, server=DEMO_SERVER)
    assert st1.consecutive_connection_failures == 1
    assert st1.is_halted is False

    with pytest.raises(MT5ConnectionError):
        orch_conn.run_single_cycle(reference_now_utc=FIXED_UTC + timedelta(seconds=10))

    st2 = orch_conn.emergency_stop.get_state(account_login=DEMO_LOGIN, server=DEMO_SERVER)
    assert st2.consecutive_connection_failures == 2
    assert st2.is_halted is True
    assert st2.active_trigger_reason == EmergencyStopReason.REPEATED_CONNECTION_FAILURES

    # Scenario 14: Persistence failure handling on corrupted bar_state.json (max_consecutive_persistence_failures=2)
    orch_p, _, _ = _build_phase92_orchestrator(
        tmp_path,
        tag="s14_persist_fail",
        max_consecutive_persistence_failures=2,
    )
    orch_p.bar_state_tracker.state_path.write_text("{corrupt_bar_state_json", encoding="utf-8")

    with pytest.raises(PersistenceCorruptionError):
        orch_p.run_single_cycle(reference_now_utc=FIXED_UTC)
    pst1 = orch_p.emergency_stop.get_state(account_login=DEMO_LOGIN, server=DEMO_SERVER)
    assert pst1.consecutive_persistence_failures == 1
    assert pst1.is_halted is False

    with pytest.raises(PersistenceCorruptionError):
        orch_p.run_single_cycle(reference_now_utc=FIXED_UTC + timedelta(seconds=5))
    pst2 = orch_p.emergency_stop.get_state(account_login=DEMO_LOGIN, server=DEMO_SERVER)
    assert pst2.consecutive_persistence_failures == 2
    assert pst2.is_halted is True
    assert pst2.active_trigger_reason == EmergencyStopReason.REPEATED_PERSISTENCE_FAILURES

    # Scenario 15: Graceful shutdown in run_session()
    orch_shut, _, _ = _build_phase92_orchestrator(
        tmp_path,
        tag="s15_shutdown",
        dry_run_only=True,
    )

    def _shutdown_on_step_14(step: RuntimeCycleStep) -> None:
        if step == RuntimeCycleStep.STEP_14_EXECUTE_OR_DRY_RUN_AND_COMMIT_BAR:
            orch_shut.request_shutdown()

    orch_shut.set_step_observer(_shutdown_on_step_14)
    session_summary = orch_shut.run_session(
        max_cycles=5,
        cycle_timestamps_utc=[FIXED_UTC + timedelta(minutes=i) for i in range(5)],
    )
    assert session_summary.total_cycles_run == 1
    assert session_summary.dry_run_evaluated_cycles == 1
    assert session_summary.graceful_shutdown_requested is True


def test_phase92_unknown_order_outcome_strategy_exception_risk_rejection_and_journal_staging_failure(
    tmp_path: Path,
) -> None:
    """Scenarios 16, 17, 18 & 19:
    16. Unknown order outcome handling: ambiguous timeout retcode (10012) returns
        AMBIGUOUS_EXECUTION_REQUIRES_RECONCILIATION without blind retry and marks ledger
        RECONCILIATION_REQUIRED.
    17. Strategy exceptions: an exception inside StrategyEngine aborts the cycle before Step 14
        without advancing `last_evaluated_bar_timestamp_utc` or submitting an order.
    18. Risk rejection: when `RiskParameters(allow_trading=False)` rejects the signal at Step 12,
        returns `RuntimeCycleStatus.RISK_REJECTED`, commits the evaluated bar timestamp, and makes
        zero `order_send` calls.
    19. Journal staging / pre-send context persistence failure: if journal persistence fails in the
        pre-send hook before `order_send()`, the order is safely aborted to `FAILED` in the ledger,
        `order_send()` is NEVER called, and `last_evaluated_bar_timestamp_utc` is NOT committed.
    """
    # Scenario 16: Ambiguous order_send timeout (TRADE_RETCODE_TIMEOUT = 10012)
    orch_ambig, mock_ambig, bar_ts_ambig = _build_phase92_orchestrator(
        tmp_path,
        tag="s16_ambig",
        direction="up",
    )
    send_attempts = 0

    def _timeout_hook(req):
        nonlocal send_attempts
        send_attempts += 1
        return MT5OrderSendResult(
            retcode=int(MT5Retcode.TRADE_RETCODE_TIMEOUT),
            comment="Broker IPC timeout during order_send",
        )

    mock_ambig.set_order_send_hook(_timeout_hook)
    rep_ambig = orch_ambig.run_single_cycle(
        reference_now_utc=FIXED_UTC,
        client_order_id="ord-ambig-1",
    )
    assert send_attempts == 1  # Strictly zero blind retries!
    assert rep_ambig.status == RuntimeCycleStatus.AMBIGUOUS_EXECUTION_REQUIRES_RECONCILIATION
    assert rep_ambig.execution_result is not None
    assert rep_ambig.execution_result.ambiguous_requires_reconciliation is True
    assert (
        rep_ambig.execution_result.ledger_entry.status
        == OrderLedgerStatus.RECONCILIATION_REQUIRED
    )
    assert rep_ambig.committed_evaluated_bar_timestamp_utc == bar_ts_ambig

    # Scenario 17: StrategyEngine exception aborts cleanly without advancing bar timestamp
    orch_strat_err, _, bar_ts_err = _build_phase92_orchestrator(
        tmp_path,
        tag="s17_strat_exc",
        direction="up",
    )

    def _boom_analyze(*args, **kwargs):
        raise StrategyEvaluationError("Synthetic strategy evaluation failure")

    orch_strat_err.strategy_engine.analyze = _boom_analyze  # type: ignore[method-assign]
    with pytest.raises(StrategyEvaluationError, match="Synthetic strategy evaluation failure"):
        orch_strat_err.run_single_cycle(reference_now_utc=FIXED_UTC)

    assert (
        orch_strat_err.bar_state_tracker.get_last_evaluated_bar_timestamp(
            "XAUUSD",
            Timeframe.H1,
            account_login=DEMO_LOGIN,
            server=DEMO_SERVER,
        )
        is None
    )

    # Scenario 18: Risk rejection (allow_trading=False) returns RISK_REJECTED with zero order_send calls
    orch_risk_rej, mock_risk_rej, bar_ts_rej = _build_phase92_orchestrator(
        tmp_path,
        tag="s18_risk_rej",
        direction="up",
        risk_parameters=RiskParameters(allow_trading=False),
    )
    rep_rej = orch_risk_rej.run_single_cycle(reference_now_utc=FIXED_UTC)
    assert rep_rej.status == RuntimeCycleStatus.RISK_REJECTED
    assert rep_rej.risk_status == RiskEvaluationStatus.RISK_REJECTED
    assert rep_rej.execution_result is None
    assert rep_rej.committed_evaluated_bar_timestamp_utc == bar_ts_rej
    assert len(mock_risk_rej.positions_get("XAUUSD") or ()) == 0

    # Scenario 19: Journal pre-send persistence failure aborts BEFORE backend.order_send()
    orch_j_fail, mock_j_fail, _ = _build_phase92_orchestrator(
        tmp_path,
        tag="s19_journal_fail",
        direction="up",
    )
    j_send_calls = 0

    def _j_send_hook(req):
        nonlocal j_send_calls
        j_send_calls += 1
        return None

    mock_j_fail.set_order_send_hook(_j_send_hook)

    def _broken_record_entry_context(*args, **kwargs):
        raise TradeReconstructionError("Simulated disk failure while persisting EntryContextSnapshot")

    orch_j_fail.journal.record_entry_context = _broken_record_entry_context  # type: ignore[method-assign]

    with pytest.raises(TradeReconstructionError, match="Simulated disk failure"):
        orch_j_fail.run_single_cycle(
            reference_now_utc=FIXED_UTC,
            client_order_id="ord-jfail-1",
        )

    assert j_send_calls == 0  # order_send() was NEVER invoked!
    aborted_entry = orch_j_fail.ledger.get_entry("ord-jfail-1")
    assert aborted_entry is not None
    assert aborted_entry.status == OrderLedgerStatus.FAILED
    assert (
        orch_j_fail.bar_state_tracker.get_last_evaluated_bar_timestamp(
            "XAUUSD",
            Timeframe.H1,
            account_login=DEMO_LOGIN,
            server=DEMO_SERVER,
        )
        is None
    )
