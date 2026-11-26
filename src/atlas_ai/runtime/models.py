"""Immutable Pydantic domain models for ATLAS AI Phase 9.1 Persistent Emergency Circuit Breaker."""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EmergencyStopStatus(StrEnum):
    """Operational state of the persistent emergency circuit breaker."""

    ARMED = "ARMED"
    LATCHED_HALTED = "LATCHED_HALTED"


class EmergencyStopReason(StrEnum):
    """Deterministic trigger categories capable of latching the emergency circuit breaker."""

    MANUAL_EMERGENCY_STOP = "MANUAL_EMERGENCY_STOP"
    DRIFT_QUARANTINE_RECOMMENDED = "DRIFT_QUARANTINE_RECOMMENDED"
    DRIFT_EXECUTION_DEGRADED = "DRIFT_EXECUTION_DEGRADED"
    LEDGER_RECOVERY_REQUIRED_LOCKED = "LEDGER_RECOVERY_REQUIRED_LOCKED"
    REPEATED_CONNECTION_FAILURES = "REPEATED_CONNECTION_FAILURES"
    REPEATED_PERSISTENCE_FAILURES = "REPEATED_PERSISTENCE_FAILURES"


class EmergencyEventType(StrEnum):
    """Audit event classification inside the persistent emergency circuit-breaker history."""

    INITIALIZED = "INITIALIZED"
    LATCHED = "LATCHED"
    FAILURE_RECORDED = "FAILURE_RECORDED"
    FAILURE_COUNTERS_CLEARED = "FAILURE_COUNTERS_CLEARED"
    OPERATOR_RESET = "OPERATOR_RESET"


class EmergencyAuditEvent(BaseModel):
    """Immutable audit record of an emergency circuit-breaker state transition or failure event.

    All historical events are preserved across operator resets and process restarts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: int = Field(..., ge=1, description="Monotonically increasing 1-based audit sequence number.")
    event_type: EmergencyEventType
    timestamp_utc: datetime
    from_status: EmergencyStopStatus | None = Field(default=None)
    to_status: EmergencyStopStatus
    trigger_reason: EmergencyStopReason | None = Field(default=None)
    detail: str = Field(..., min_length=1)
    operator_rationale: str | None = Field(default=None)
    consecutive_connection_failures: int = Field(default=0, ge=0)
    consecutive_persistence_failures: int = Field(default=0, ge=0)

    @field_validator("timestamp_utc")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("EmergencyAuditEvent timestamp_utc must be timezone-aware.")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_event_invariants(self) -> "EmergencyAuditEvent":
        if self.event_type == EmergencyEventType.LATCHED:
            if self.to_status != EmergencyStopStatus.LATCHED_HALTED:
                raise ValueError("LATCHED event must transition to LATCHED_HALTED.")
            if self.trigger_reason is None:
                raise ValueError("LATCHED event requires a non-None trigger_reason.")
        if self.event_type == EmergencyEventType.OPERATOR_RESET:
            if self.to_status != EmergencyStopStatus.ARMED:
                raise ValueError("OPERATOR_RESET event must transition to ARMED.")
            if self.operator_rationale is None or not self.operator_rationale.strip():
                raise ValueError("OPERATOR_RESET event requires a non-empty operator_rationale.")
        return self


class EmergencyStopState(BaseModel):
    """Immutable snapshot of the persistent emergency circuit-breaker state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: EmergencyStopStatus = Field(default=EmergencyStopStatus.ARMED)
    is_halted: bool = Field(default=False)
    active_trigger_reason: EmergencyStopReason | None = Field(default=None)
    active_trigger_detail: str | None = Field(default=None)
    latched_at_utc: datetime | None = Field(default=None)
    last_reset_at_utc: datetime | None = Field(default=None)
    last_reset_rationale: str | None = Field(default=None)
    consecutive_connection_failures: int = Field(default=0, ge=0)
    consecutive_persistence_failures: int = Field(default=0, ge=0)
    updated_at_utc: datetime
    audit_history: tuple[EmergencyAuditEvent, ...] = Field(default=())
    requires_human_reset: Literal[True] = Field(
        default=True,
        description="Invariantly True: once latched, the circuit breaker can never auto-reset.",
    )

    @field_validator("updated_at_utc")
    @classmethod
    def ensure_updated_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("EmergencyStopState updated_at_utc must be timezone-aware.")
        return value.astimezone(timezone.utc)

    @field_validator("latched_at_utc", "last_reset_at_utc")
    @classmethod
    def ensure_optional_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("EmergencyStopState optional timestamps must be timezone-aware.")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_state_invariants(self) -> "EmergencyStopState":
        if self.requires_human_reset is not True:
            raise ValueError("requires_human_reset must invariantly be True.")
        if self.status == EmergencyStopStatus.LATCHED_HALTED:
            if not self.is_halted:
                raise ValueError("is_halted must be True when status is LATCHED_HALTED.")
            if self.active_trigger_reason is None:
                raise ValueError("active_trigger_reason is required when status is LATCHED_HALTED.")
            if self.active_trigger_detail is None or not self.active_trigger_detail.strip():
                raise ValueError("active_trigger_detail is required when status is LATCHED_HALTED.")
            if self.latched_at_utc is None:
                raise ValueError("latched_at_utc is required when status is LATCHED_HALTED.")
        else:
            if self.is_halted:
                raise ValueError("is_halted must be False when status is ARMED.")
            if (
                self.active_trigger_reason is not None
                or self.active_trigger_detail is not None
                or self.latched_at_utc is not None
            ):
                raise ValueError(
                    "active_trigger_reason, active_trigger_detail, and latched_at_utc must be None when status is ARMED."
                )

        # Verify monotonic event_id sequence in audit_history
        for idx, ev in enumerate(self.audit_history, start=1):
            if ev.event_id != idx:
                raise ValueError(
                    f"Non-sequential audit_history event_id at position {idx}: got {ev.event_id}."
                )
        return self
