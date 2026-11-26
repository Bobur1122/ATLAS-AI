"""Immutable runtime configuration schema for ATLAS AI Phase 9.1 Demo Runtime & Emergency Circuit Breaker."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atlas_ai.domain.market import Timeframe
from atlas_ai.data.integrity import get_timeframe_delta


class DemoRuntimeConfig(BaseModel):
    """Deterministic configuration for the closed-loop MT5 Demo Runtime and Persistent Emergency Circuit Breaker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(
        default="XAUUSD",
        min_length=3,
        max_length=16,
        description="Primary trading instrument symbol.",
    )
    primary_timeframe: Timeframe = Field(
        default=Timeframe.H1,
        description="Primary execution timeframe evaluated on each newly closed candle.",
    )
    higher_timeframes: tuple[Timeframe, ...] = Field(
        default=(Timeframe.H4, Timeframe.D1),
        description="Higher timeframes fetched for multi-timeframe directional alignment.",
    )
    candle_lookback_bars: int = Field(
        default=250,
        ge=50,
        le=5000,
        description="Number of closed historical candles (`start_pos=1`) fetched per timeframe.",
    )
    dry_run_only: bool = Field(
        default=False,
        description="When True (shadow evaluation mode), evaluates the full pipeline without submitting orders.",
    )
    latch_on_quarantine_recommended: bool = Field(
        default=True,
        description="Automatically latch the emergency circuit breaker on DriftStatus.QUARANTINE_RECOMMENDED.",
    )
    latch_on_execution_degraded: bool = Field(
        default=True,
        description="Automatically latch the emergency circuit breaker on DriftStatus.EXECUTION_DEGRADED.",
    )
    latch_on_ledger_recovery_lock: bool = Field(
        default=True,
        description="Automatically latch the emergency circuit breaker on LedgerHealthStatus.RECOVERY_REQUIRED_LOCKED.",
    )
    max_consecutive_connection_failures: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Consecutive MT5 connection failure count threshold that latches the emergency circuit breaker.",
    )
    max_consecutive_persistence_failures: int = Field(
        default=2,
        ge=1,
        le=100,
        description="Consecutive persistence/state error count threshold that latches the emergency circuit breaker.",
    )
    emergency_state_path: Path = Field(
        default=Path("data/processed/emergency_stop_state.json"),
        description="File path for the crash-safe persistent emergency circuit-breaker state.",
    )
    lock_timeout_seconds: float = Field(
        default=5.0,
        gt=0.0,
        le=60.0,
        description="Timeout in seconds for acquiring an exclusive OS file lock on the emergency state file.",
    )

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned or not cleaned.isalnum():
            raise ValueError(f"Symbol must be alphanumeric, got '{value}'.")
        return cleaned

    @field_validator("higher_timeframes")
    @classmethod
    def deduplicate_higher_timeframes(cls, values: tuple[Timeframe, ...]) -> tuple[Timeframe, ...]:
        ordered_unique = list(dict.fromkeys(values))
        ordered_unique.sort(key=lambda tf: get_timeframe_delta(tf))
        return tuple(ordered_unique)

    @model_validator(mode="after")
    def validate_timeframe_hierarchy(self) -> "DemoRuntimeConfig":
        primary_delta = get_timeframe_delta(self.primary_timeframe)
        for htf in self.higher_timeframes:
            if get_timeframe_delta(htf) <= primary_delta:
                raise ValueError(
                    f"Every higher_timeframe ({htf.value}) must be strictly larger than "
                    f"primary_timeframe ({self.primary_timeframe.value})."
                )
        return self
