"""Persistent, Crash-Safe Emergency Circuit Breaker (`EmergencyStopController`) for ATLAS AI Phase 9.1."""

from datetime import datetime, timezone
import threading

from pydantic import ValidationError

from atlas_ai.analytics.models import DriftAndHealthReport, DriftStatus
from atlas_ai.core.logger import get_logger
from atlas_ai.execution.exceptions import (
    LedgerRecoveryLockError,
    PersistenceAccountMismatchError,
    PersistenceCorruptionError,
    PersistenceError,
    PersistenceRollbackDetectedError,
)
from atlas_ai.execution.models import LedgerHealthStatus
from atlas_ai.execution.persistence import CrashSafeJsonPersistence
from atlas_ai.runtime.config import DemoRuntimeConfig
from atlas_ai.runtime.exceptions import (
    CircuitBreakerLatchedError,
    EmergencyStopError,
    EmergencyStopResetError,
    RuntimeOrchestrationError,
)
from atlas_ai.runtime.models import (
    EmergencyAuditEvent,
    EmergencyEventType,
    EmergencyStopReason,
    EmergencyStopState,
    EmergencyStopStatus,
)

logger = get_logger("runtime.circuit_breaker")


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


class EmergencyStopController:
    """Persistent, crash-safe emergency circuit breaker that halts trading authorization
    whenever an emergency condition is detected.

    Safety Guarantees:
    1. Reuses Phase 7 `CrashSafeJsonPersistence` and OS-level `FileLock` (`msvcrt.locking` / `fcntl.flock`)
       with SHA-256 checksums, monotonic generations, `previous_checksum` hash chains, and strict
       `(account_login, server)` binding.
    2. Never auto-resets after a process restart; once latched in `LATCHED_HALTED`, remains halted
       across restarts until an explicit `reset_emergency_stop(operator_rationale=...)` call with
       a non-empty rationale is durably committed to disk.
    3. Preserves all historical `EmergencyAuditEvent` records across latches, resets, and restarts.
    4. Fails closed on corrupted, rolled-back, recovery-locked, or account-mismatched state files.
    5. Thread-safe (`threading.RLock`) and cross-process safe (`FileLock`).
    """

    def __init__(self, config: DemoRuntimeConfig | None = None) -> None:
        self._config = config or DemoRuntimeConfig()
        self._persistence = CrashSafeJsonPersistence(
            self._config.emergency_state_path,
            lock_timeout_seconds=self._config.lock_timeout_seconds,
        )
        self._mutex = threading.RLock()
        self._loaded_account: tuple[int, str] | None = None
        self._state: EmergencyStopState | None = None
        self._last_generation: int = 0

    @property
    def config(self) -> DemoRuntimeConfig:
        return self._config

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
    ) -> EmergencyStopState:
        """Load and verify the persistent emergency state from disk under an exclusive OS file lock,
        or initialize an `ARMED` state in memory if no persistence file exists yet.

        Fails closed (`LedgerRecoveryLockError`, `PersistenceCorruptionError`,
        `PersistenceAccountMismatchError`, `PersistenceRollbackDetectedError`) if the on-disk
        state file is corrupted, rolled back, mismatched, or has interrupted `.tmp.*` writes.
        """
        login, cleaned_server = _validate_account_identity(account_login, server)
        ts = _normalize_utc(now_utc)

        with self._mutex:
            env = self._persistence.load(
                expected_login=login,
                expected_server=cleaned_server,
            )
            self._loaded_account = (login, cleaned_server)

            if env is None:
                init_event = EmergencyAuditEvent(
                    event_id=1,
                    event_type=EmergencyEventType.INITIALIZED,
                    timestamp_utc=ts,
                    from_status=None,
                    to_status=EmergencyStopStatus.ARMED,
                    trigger_reason=None,
                    detail=f"EmergencyStopController initialized in ARMED state for {login}@{cleaned_server}.",
                    operator_rationale=None,
                    consecutive_connection_failures=0,
                    consecutive_persistence_failures=0,
                )
                self._state = EmergencyStopState(
                    status=EmergencyStopStatus.ARMED,
                    is_halted=False,
                    updated_at_utc=ts,
                    audit_history=(init_event,),
                )
                self._last_generation = 0
                return self._state

            try:
                loaded_state = EmergencyStopState.model_validate(env.payload)
            except ValidationError as exc:
                raise PersistenceCorruptionError(
                    f"Emergency stop state schema validation failed for '{self._config.emergency_state_path}': {exc}"
                ) from exc

            self._state = loaded_state
            self._last_generation = env.generation
            return self._state

    def get_state(
        self,
        *,
        account_login: int,
        server: str,
        now_utc: datetime | None = None,
    ) -> EmergencyStopState:
        """Reload (or initialize) and return the latest verified `EmergencyStopState` from disk."""
        return self.load_or_initialize(
            account_login=account_login,
            server=server,
            now_utc=now_utc,
        )

    def is_halted(
        self,
        *,
        account_login: int,
        server: str,
        now_utc: datetime | None = None,
    ) -> bool:
        """Return `True` if the emergency circuit breaker is currently `LATCHED_HALTED`.

        Always verifies the on-disk persistence state first and fails closed if corrupted.
        """
        state = self.load_or_initialize(
            account_login=account_login,
            server=server,
            now_utc=now_utc,
        )
        return state.is_halted

    def assert_trading_authorized(
        self,
        *,
        account_login: int,
        server: str,
        now_utc: datetime | None = None,
    ) -> EmergencyStopState:
        """Verify that the emergency circuit breaker is `ARMED` and not halted.

        Raises:
            CircuitBreakerLatchedError: If the circuit breaker is `LATCHED_HALTED`.
            PersistenceError / LedgerRecoveryLockError: If the persistent state is corrupted or locked.
        """
        state = self.load_or_initialize(
            account_login=account_login,
            server=server,
            now_utc=now_utc,
        )
        if state.is_halted or state.status == EmergencyStopStatus.LATCHED_HALTED:
            raise CircuitBreakerLatchedError(
                f"EMERGENCY CIRCUIT BREAKER LATCHED ({state.active_trigger_reason}): "
                f"{state.active_trigger_detail} (latched_at_utc="
                f"{state.latched_at_utc.isoformat() if state.latched_at_utc else 'unknown'}). "
                "Trading authorization is strictly blocked until explicit human operator reset."
            )
        return state

    def _persist_state_locked(
        self,
        new_state: EmergencyStopState,
        *,
        account_login: int,
        server: str,
        now_utc: datetime,
    ) -> EmergencyStopState:
        saved_env = self._persistence.save(
            new_state.model_dump(mode="json"),
            account_login=account_login,
            server=server,
            now_utc=now_utc,
        )
        self._state = new_state
        self._loaded_account = (account_login, server)
        self._last_generation = saved_env.generation
        return new_state

    def _latch_locked(
        self,
        current_state: EmergencyStopState,
        *,
        reason: EmergencyStopReason,
        detail: str,
        account_login: int,
        server: str,
        now_utc: datetime,
        conn_failures: int | None = None,
        persist_failures: int | None = None,
    ) -> EmergencyStopState:
        cleaned_detail = detail.strip()
        if not cleaned_detail:
            raise EmergencyStopError("Emergency latch detail must be a non-empty string.")

        c_fail = (
            conn_failures
            if conn_failures is not None
            else current_state.consecutive_connection_failures
        )
        p_fail = (
            persist_failures
            if persist_failures is not None
            else current_state.consecutive_persistence_failures
        )

        # Idempotent if already latched with the identical trigger reason, detail, and failure counts
        if (
            current_state.status == EmergencyStopStatus.LATCHED_HALTED
            and current_state.active_trigger_reason == reason
            and current_state.active_trigger_detail == cleaned_detail
            and current_state.consecutive_connection_failures == c_fail
            and current_state.consecutive_persistence_failures == p_fail
            and self._last_generation > 0
        ):
            return current_state

        next_event_id = len(current_state.audit_history) + 1
        latch_event = EmergencyAuditEvent(
            event_id=next_event_id,
            event_type=EmergencyEventType.LATCHED,
            timestamp_utc=now_utc,
            from_status=current_state.status,
            to_status=EmergencyStopStatus.LATCHED_HALTED,
            trigger_reason=reason,
            detail=cleaned_detail,
            operator_rationale=None,
            consecutive_connection_failures=c_fail,
            consecutive_persistence_failures=p_fail,
        )
        latched_ts = (
            current_state.latched_at_utc
            if (
                current_state.status == EmergencyStopStatus.LATCHED_HALTED
                and current_state.latched_at_utc is not None
            )
            else now_utc
        )
        new_state = EmergencyStopState(
            status=EmergencyStopStatus.LATCHED_HALTED,
            is_halted=True,
            active_trigger_reason=reason,
            active_trigger_detail=cleaned_detail,
            latched_at_utc=latched_ts,
            last_reset_at_utc=current_state.last_reset_at_utc,
            last_reset_rationale=current_state.last_reset_rationale,
            consecutive_connection_failures=c_fail,
            consecutive_persistence_failures=p_fail,
            updated_at_utc=now_utc,
            audit_history=(*current_state.audit_history, latch_event),
            requires_human_reset=True,
        )
        persisted = self._persist_state_locked(
            new_state,
            account_login=account_login,
            server=server,
            now_utc=now_utc,
        )
        logger.critical(
            "EMERGENCY CIRCUIT BREAKER LATCHED for %d@%s [reason=%s, generation=%d]: %s",
            account_login,
            server,
            reason.value,
            self._last_generation,
            cleaned_detail,
        )
        return persisted

    def trigger_manual_stop(
        self,
        rationale: str,
        *,
        account_login: int,
        server: str,
        now_utc: datetime | None = None,
    ) -> EmergencyStopState:
        """Immediately latch a manual emergency stop and durably persist it to disk."""
        if not isinstance(rationale, str) or not rationale.strip():
            raise EmergencyStopError("Manual emergency stop rationale must be a non-empty string.")
        login, cleaned_server = _validate_account_identity(account_login, server)
        ts = _normalize_utc(now_utc)

        with self._mutex:
            current = self.load_or_initialize(
                account_login=login,
                server=cleaned_server,
                now_utc=ts,
            )
            return self._latch_locked(
                current,
                reason=EmergencyStopReason.MANUAL_EMERGENCY_STOP,
                detail=f"Manual emergency stop triggered: {rationale.strip()}",
                account_login=login,
                server=cleaned_server,
                now_utc=ts,
            )

    def evaluate_drift_report(
        self,
        drift_report: DriftAndHealthReport,
        *,
        account_login: int,
        server: str,
        now_utc: datetime | None = None,
    ) -> EmergencyStopState:
        """Evaluate a Phase 8.3 `DriftAndHealthReport` and automatically latch the circuit breaker
        if `DriftStatus.QUARANTINE_RECOMMENDED` or `DriftStatus.EXECUTION_DEGRADED` is active
        under the configured circuit-breaker policies.
        """
        if not isinstance(drift_report, DriftAndHealthReport):
            raise RuntimeOrchestrationError(
                f"Expected DriftAndHealthReport, got {type(drift_report).__name__}."
            )
        login, cleaned_server = _validate_account_identity(account_login, server)
        ts = _normalize_utc(now_utc)

        with self._mutex:
            current = self.load_or_initialize(
                account_login=login,
                server=cleaned_server,
                now_utc=ts,
            )
            if (
                drift_report.overall_status == DriftStatus.QUARANTINE_RECOMMENDED
                and self._config.latch_on_quarantine_recommended
            ):
                rules_str = (
                    ", ".join(drift_report.quarantine_recommended_rule_ids)
                    if drift_report.quarantine_recommended_rule_ids
                    else "portfolio expectancy & loss-streak degradation"
                )
                return self._latch_locked(
                    current,
                    reason=EmergencyStopReason.DRIFT_QUARANTINE_RECOMMENDED,
                    detail=(
                        f"Phase 8.3 DriftDetectionEngine reported QUARANTINE_RECOMMENDED "
                        f"(expectancy_r={drift_report.realized_expectancy_r}, "
                        f"max_consecutive_losses={drift_report.max_consecutive_losses}, "
                        f"triggers={rules_str})."
                    ),
                    account_login=login,
                    server=cleaned_server,
                    now_utc=ts,
                )

            if (
                drift_report.overall_status == DriftStatus.EXECUTION_DEGRADED
                and self._config.latch_on_execution_degraded
            ):
                return self._latch_locked(
                    current,
                    reason=EmergencyStopReason.DRIFT_EXECUTION_DEGRADED,
                    detail=(
                        f"Phase 8.3 DriftDetectionEngine reported EXECUTION_DEGRADED "
                        f"(mean_entry_slippage_points={drift_report.mean_entry_slippage_points})."
                    ),
                    account_login=login,
                    server=cleaned_server,
                    now_utc=ts,
                )

            return current

    def evaluate_ledger_health(
        self,
        ledger_health: LedgerHealthStatus,
        *,
        account_login: int,
        server: str,
        detail: str | None = None,
        now_utc: datetime | None = None,
    ) -> EmergencyStopState:
        """Evaluate Phase 7 `LedgerHealthStatus` and automatically latch the circuit breaker
        if `LedgerHealthStatus.RECOVERY_REQUIRED_LOCKED` is reported.
        """
        if not isinstance(ledger_health, LedgerHealthStatus):
            raise RuntimeOrchestrationError(
                f"Expected LedgerHealthStatus, got {type(ledger_health).__name__}."
            )
        login, cleaned_server = _validate_account_identity(account_login, server)
        ts = _normalize_utc(now_utc)

        with self._mutex:
            current = self.load_or_initialize(
                account_login=login,
                server=cleaned_server,
                now_utc=ts,
            )
            if (
                ledger_health == LedgerHealthStatus.RECOVERY_REQUIRED_LOCKED
                and self._config.latch_on_ledger_recovery_lock
            ):
                msg = (
                    detail.strip()
                    if (detail is not None and detail.strip())
                    else "PersistentExecutionLedger reported RECOVERY_REQUIRED_LOCKED."
                )
                return self._latch_locked(
                    current,
                    reason=EmergencyStopReason.LEDGER_RECOVERY_REQUIRED_LOCKED,
                    detail=msg,
                    account_login=login,
                    server=cleaned_server,
                    now_utc=ts,
                )
            return current

    def record_connection_failure(
        self,
        error_detail: str,
        *,
        account_login: int,
        server: str,
        now_utc: datetime | None = None,
    ) -> EmergencyStopState:
        """Increment `consecutive_connection_failures` and automatically latch the circuit breaker
        when `consecutive_connection_failures >= config.max_consecutive_connection_failures`.
        """
        if not isinstance(error_detail, str) or not error_detail.strip():
            raise EmergencyStopError("Connection failure error_detail must be a non-empty string.")
        login, cleaned_server = _validate_account_identity(account_login, server)
        ts = _normalize_utc(now_utc)
        cleaned_err = error_detail.strip()

        with self._mutex:
            current = self.load_or_initialize(
                account_login=login,
                server=cleaned_server,
                now_utc=ts,
            )
            new_count = current.consecutive_connection_failures + 1
            threshold = self._config.max_consecutive_connection_failures

            if new_count >= threshold:
                return self._latch_locked(
                    current,
                    reason=EmergencyStopReason.REPEATED_CONNECTION_FAILURES,
                    detail=(
                        f"Consecutive MT5 connection failures ({new_count}) reached threshold "
                        f"({threshold}): {cleaned_err}"
                    ),
                    account_login=login,
                    server=cleaned_server,
                    now_utc=ts,
                    conn_failures=new_count,
                )

            next_event_id = len(current.audit_history) + 1
            fail_event = EmergencyAuditEvent(
                event_id=next_event_id,
                event_type=EmergencyEventType.FAILURE_RECORDED,
                timestamp_utc=ts,
                from_status=current.status,
                to_status=current.status,
                trigger_reason=EmergencyStopReason.REPEATED_CONNECTION_FAILURES,
                detail=f"Connection failure {new_count}/{threshold}: {cleaned_err}",
                operator_rationale=None,
                consecutive_connection_failures=new_count,
                consecutive_persistence_failures=current.consecutive_persistence_failures,
            )
            updated_state = current.model_copy(
                update={
                    "consecutive_connection_failures": new_count,
                    "updated_at_utc": ts,
                    "audit_history": (*current.audit_history, fail_event),
                }
            )
            return self._persist_state_locked(
                updated_state,
                account_login=login,
                server=cleaned_server,
                now_utc=ts,
            )

    def record_connection_success(
        self,
        *,
        account_login: int,
        server: str,
        now_utc: datetime | None = None,
    ) -> EmergencyStopState:
        """Reset `consecutive_connection_failures` to `0` on a healthy connection check
        WITHOUT ever unlatching a `LATCHED_HALTED` circuit breaker.
        """
        login, cleaned_server = _validate_account_identity(account_login, server)
        ts = _normalize_utc(now_utc)

        with self._mutex:
            current = self.load_or_initialize(
                account_login=login,
                server=cleaned_server,
                now_utc=ts,
            )
            if current.consecutive_connection_failures == 0:
                return current

            next_event_id = len(current.audit_history) + 1
            clear_event = EmergencyAuditEvent(
                event_id=next_event_id,
                event_type=EmergencyEventType.FAILURE_COUNTERS_CLEARED,
                timestamp_utc=ts,
                from_status=current.status,
                to_status=current.status,
                trigger_reason=None,
                detail=(
                    f"Cleared consecutive_connection_failures (was {current.consecutive_connection_failures}) "
                    "after healthy connection check."
                ),
                operator_rationale=None,
                consecutive_connection_failures=0,
                consecutive_persistence_failures=current.consecutive_persistence_failures,
            )
            updated_state = current.model_copy(
                update={
                    "consecutive_connection_failures": 0,
                    "updated_at_utc": ts,
                    "audit_history": (*current.audit_history, clear_event),
                }
            )
            return self._persist_state_locked(
                updated_state,
                account_login=login,
                server=cleaned_server,
                now_utc=ts,
            )

    def record_persistence_failure(
        self,
        error_detail: str,
        *,
        account_login: int,
        server: str,
        now_utc: datetime | None = None,
    ) -> EmergencyStopState:
        """Increment `consecutive_persistence_failures` and automatically latch the circuit breaker
        when `consecutive_persistence_failures >= config.max_consecutive_persistence_failures`.
        """
        if not isinstance(error_detail, str) or not error_detail.strip():
            raise EmergencyStopError("Persistence failure error_detail must be a non-empty string.")
        login, cleaned_server = _validate_account_identity(account_login, server)
        ts = _normalize_utc(now_utc)
        cleaned_err = error_detail.strip()

        with self._mutex:
            current = self.load_or_initialize(
                account_login=login,
                server=cleaned_server,
                now_utc=ts,
            )
            new_count = current.consecutive_persistence_failures + 1
            threshold = self._config.max_consecutive_persistence_failures

            if new_count >= threshold:
                return self._latch_locked(
                    current,
                    reason=EmergencyStopReason.REPEATED_PERSISTENCE_FAILURES,
                    detail=(
                        f"Consecutive persistence failures ({new_count}) reached threshold "
                        f"({threshold}): {cleaned_err}"
                    ),
                    account_login=login,
                    server=cleaned_server,
                    now_utc=ts,
                    persist_failures=new_count,
                )

            next_event_id = len(current.audit_history) + 1
            fail_event = EmergencyAuditEvent(
                event_id=next_event_id,
                event_type=EmergencyEventType.FAILURE_RECORDED,
                timestamp_utc=ts,
                from_status=current.status,
                to_status=current.status,
                trigger_reason=EmergencyStopReason.REPEATED_PERSISTENCE_FAILURES,
                detail=f"Persistence failure {new_count}/{threshold}: {cleaned_err}",
                operator_rationale=None,
                consecutive_connection_failures=current.consecutive_connection_failures,
                consecutive_persistence_failures=new_count,
            )
            updated_state = current.model_copy(
                update={
                    "consecutive_persistence_failures": new_count,
                    "updated_at_utc": ts,
                    "audit_history": (*current.audit_history, fail_event),
                }
            )
            return self._persist_state_locked(
                updated_state,
                account_login=login,
                server=cleaned_server,
                now_utc=ts,
            )

    def reset_emergency_stop(
        self,
        operator_rationale: str,
        *,
        account_login: int,
        server: str,
        now_utc: datetime | None = None,
    ) -> EmergencyStopState:
        """Explicitly unlatch the emergency circuit breaker with a mandatory human operator rationale.

        Requirements enforced:
        1. Non-empty `operator_rationale` (raises `EmergencyStopResetError` if empty or whitespace).
        2. Loads and verifies existing on-disk state first (fails closed if state file is corrupted).
        3. Preserves all prior `EmergencyAuditEvent` entries in `audit_history` and appends an
           `OPERATOR_RESET` audit event.
        4. Durably commits the `ARMED` state to disk under an exclusive OS file lock before returning.
        """
        if not isinstance(operator_rationale, str) or not operator_rationale.strip():
            raise EmergencyStopResetError(
                "reset_emergency_stop() requires a non-empty operator_rationale string."
            )
        cleaned_rationale = operator_rationale.strip()
        login, cleaned_server = _validate_account_identity(account_login, server)
        ts = _normalize_utc(now_utc)

        with self._mutex:
            current = self.load_or_initialize(
                account_login=login,
                server=cleaned_server,
                now_utc=ts,
            )
            prev_reason = (
                current.active_trigger_reason.value
                if current.active_trigger_reason is not None
                else "NONE"
            )
            next_event_id = len(current.audit_history) + 1
            reset_event = EmergencyAuditEvent(
                event_id=next_event_id,
                event_type=EmergencyEventType.OPERATOR_RESET,
                timestamp_utc=ts,
                from_status=current.status,
                to_status=EmergencyStopStatus.ARMED,
                trigger_reason=current.active_trigger_reason,
                detail=(
                    f"Explicit human operator reset from {current.status.value} "
                    f"(prior_reason={prev_reason}): {cleaned_rationale}"
                ),
                operator_rationale=cleaned_rationale,
                consecutive_connection_failures=0,
                consecutive_persistence_failures=0,
            )
            reset_state = EmergencyStopState(
                status=EmergencyStopStatus.ARMED,
                is_halted=False,
                active_trigger_reason=None,
                active_trigger_detail=None,
                latched_at_utc=None,
                last_reset_at_utc=ts,
                last_reset_rationale=cleaned_rationale,
                consecutive_connection_failures=0,
                consecutive_persistence_failures=0,
                updated_at_utc=ts,
                audit_history=(*current.audit_history, reset_event),
                requires_human_reset=True,
            )
            persisted = self._persist_state_locked(
                reset_state,
                account_login=login,
                server=cleaned_server,
                now_utc=ts,
            )
            logger.info(
                "EmergencyStopController reset to ARMED for %d@%s [generation=%d]: %s",
                login,
                cleaned_server,
                self._last_generation,
                cleaned_rationale,
            )
            return persisted


__all__ = [
    "CircuitBreakerLatchedError",
    "EmergencyStopController",
    "EmergencyStopError",
    "EmergencyStopResetError",
    "LedgerRecoveryLockError",
    "PersistenceAccountMismatchError",
    "PersistenceCorruptionError",
    "PersistenceError",
    "PersistenceRollbackDetectedError",
    "RuntimeOrchestrationError",
]
