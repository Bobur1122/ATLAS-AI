"""Configuration schemas for ATLAS AI Phase 7 MetaTrader 5 Integration & Demo Execution."""

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from atlas_ai.core.config import ExecutionMode
from atlas_ai.execution.exceptions import LiveTradingBlockedError


class TrustedBalanceAnchorConfig(BaseModel):
    """Independently verified account balance anchor used to prove deal history completeness."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    anchor_balance: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Verified account balance at anchor_timestamp_utc (e.g., 0.00 prior to initial deposit deal, or statement balance).",
    )
    anchor_timestamp_utc: datetime = Field(
        ...,
        description="Timezone-aware UTC timestamp corresponding to anchor_balance.",
    )
    require_initial_deposit_deal: bool = Field(
        default=False,
        description="If True (for origin anchor 0.00), requires at least one DEAL_TYPE_BALANCE deposit deal after anchor_timestamp_utc.",
    )
    anchor_deal_ticket: int | None = Field(
        default=None,
        gt=0,
        description="Optional explicit broker deal ticket at anchor_timestamp_utc whose post-deal balance equals anchor_balance.",
    )

    @field_validator("anchor_timestamp_utc")
    @classmethod
    def ensure_utc_timestamp(cls, value: datetime) -> datetime:
        """Require a timezone-aware timestamp and normalize to UTC."""
        if value.tzinfo is None:
            raise ValueError("anchor_timestamp_utc must be timezone-aware.")
        return value.astimezone(timezone.utc)


class MT5ConnectionConfig(BaseModel):
    """Configuration for connecting to a MetaTrader 5 Demo terminal with secret redaction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    terminal_path: Path | None = Field(
        default=None,
        description="Optional path to terminal64.exe.",
    )
    login: int | None = Field(
        default=None,
        gt=0,
        description="MT5 demo account login number.",
    )
    password: SecretStr | None = Field(
        default=None,
        repr=False,
        description="MT5 demo account password (redacted from logs and repr).",
    )
    server: str | None = Field(
        default=None,
        min_length=1,
        description="MT5 demo server name.",
    )
    expected_login: int | None = Field(
        default=None,
        gt=0,
        description="Mandatory expected demo login number verified against account_info().login.",
    )
    expected_server: str | None = Field(
        default=None,
        min_length=1,
        description="Mandatory expected demo server verified against account_info().server.",
    )
    timeout_ms: int = Field(
        default=10000,
        ge=1000,
        le=120000,
        description="Terminal initialization and IPC timeout in milliseconds.",
    )
    max_reconnect_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum bounded reconnection attempts on terminal disconnect.",
    )
    portable: bool = Field(
        default=False,
        description="Whether to launch the MT5 terminal in portable mode.",
    )

    def redacted_dict(self) -> dict[str, object]:
        """Return a dictionary safe for structured logging with password strictly masked."""
        return {
            "terminal_path": str(self.terminal_path) if self.terminal_path else None,
            "login": self.login,
            "password": "***REDACTED***" if self.password is not None else None,
            "server": self.server,
            "expected_login": self.expected_login,
            "expected_server": self.expected_server,
            "timeout_ms": self.timeout_ms,
            "max_reconnect_attempts": self.max_reconnect_attempts,
            "portable": self.portable,
        }

    def __repr__(self) -> str:
        return f"MT5ConnectionConfig({self.redacted_dict()})"

    def __str__(self) -> str:
        return self.__repr__()


class ExecutionSafetyConfig(BaseModel):
    """Deterministic pre-trade execution safety and persistence configuration for Phase 7."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.DEMO,
        description="Execution mode for MT5 operations (must be DEMO; LIVE is strictly blocked).",
    )
    live_trading_enabled: bool = Field(
        default=False,
        description="Must remain False; setting True raises LiveTradingBlockedError.",
    )
    magic_number: int = Field(
        default=202607,
        gt=0,
        description="Deterministic ATLAS AI magic number stamped on all demo orders.",
    )
    max_allowed_spread_points: Decimal = Field(
        default=Decimal("50"),
        gt=Decimal("0"),
        description="Maximum live Bid-Ask spread (in instrument tick_size points) permitted at order execution.",
    )
    max_entry_deviation_points: int = Field(
        default=20,
        ge=0,
        le=500,
        description="Maximum allowed live price drift / slippage in points from signal.entry_price.",
    )
    max_tick_age_seconds: int = Field(
        default=30,
        ge=1,
        le=3600,
        description="Maximum allowed age in seconds for the latest symbol tick before StaleMarketDataError is raised.",
    )
    day_boundary_timezone: str = Field(
        default="America/New_York",
        description="IANA timezone for daily risk rollover (default 5 PM America/New_York).",
    )
    day_rollover_hour: int = Field(
        default=17,
        ge=0,
        le=23,
        description="Hour (0..23) in day_boundary_timezone when trading day rolls over.",
    )
    reconciliation_tolerance_units: int = Field(
        default=5,
        ge=0,
        le=100,
        description="Balance reconciliation tolerance in minor currency units (e.g., 5 -> 0.05 for 2-digit currency, 5 for 0-digit JPY).",
    )
    trusted_balance_anchor: TrustedBalanceAnchorConfig | None = Field(
        default=None,
        description="Optional explicit balance anchor used when no prior persisted rollover snapshot exists.",
    )
    ledger_path: Path = Field(
        default=Path("data/processed/execution_ledger.json"),
        description="File path for the crash-safe persistent execution ledger.",
    )
    account_risk_state_path: Path = Field(
        default=Path("data/processed/account_risk_state.json"),
        description="File path for the crash-safe persistent account risk snapshot.",
    )
    lock_timeout_seconds: float = Field(
        default=5.0,
        gt=0.0,
        le=60.0,
        description="Timeout in seconds for acquiring an exclusive cross-process file lock.",
    )

    @field_validator("day_boundary_timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        cleaned = value.strip()
        try:
            ZoneInfo(cleaned)
        except (ZoneInfoNotFoundError, KeyError, ValueError) as exc:
            raise ValueError(f"Invalid IANA timezone '{value}'.") from exc
        return cleaned

    @model_validator(mode="after")
    def enforce_demo_only_safety_gate(self) -> "ExecutionSafetyConfig":
        """Block ExecutionMode.LIVE or live_trading_enabled=True unconditionally."""
        if self.live_trading_enabled or self.execution_mode == ExecutionMode.LIVE:
            raise LiveTradingBlockedError(
                "SAFETY VIOLATION: Live trading is strictly blocked in ATLAS AI Phase 7. "
                "Only ExecutionMode.DEMO (with live_trading_enabled=False) is permitted."
            )
        return self
