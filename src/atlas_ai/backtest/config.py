"""Configuration schemas for the ATLAS AI Phase 4 Historical Backtesting Engine."""

from decimal import Decimal
from enum import StrEnum
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atlas_ai.domain.market import InstrumentSpec, Timeframe
from atlas_ai.domain.risk import RiskParameters


class IntrabarTieBreakPolicy(StrEnum):
    """Tie-break policy when both Stop-Loss and Take-Profit are touched within the same OHLC bar."""

    CONSERVATIVE_SL_FIRST = "CONSERVATIVE_SL_FIRST"
    OPTIMISTIC_TP_FIRST = "OPTIMISTIC_TP_FIRST"


def default_xauusd_instrument_spec() -> InstrumentSpec:
    """Return the standard default XAUUSD contract specification (100 oz, $0.01 tick = $1.00/lot)."""
    return InstrumentSpec(
        symbol="XAUUSD",
        contract_size=Decimal("100"),
        tick_size=Decimal("0.01"),
        tick_value=Decimal("1.00"),
        volume_step=Decimal("0.01"),
        minimum_volume=Decimal("0.01"),
        maximum_volume=Decimal("50.00"),
        margin_rate=Decimal("0.01"),
    )


class BacktestConfig(BaseModel):
    """Validated configuration for deterministic historical backtesting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(
        default="XAUUSD",
        min_length=3,
        max_length=16,
        description="Primary instrument symbol under test.",
    )
    timeframe: Timeframe = Field(
        default=Timeframe.H1,
        description="Primary execution timeframe.",
    )
    initial_balance: Decimal = Field(
        default=Decimal("10000.00"),
        gt=Decimal("0"),
        description="Starting account balance in account currency.",
    )
    currency: str = Field(
        default="USD",
        min_length=3,
        max_length=3,
        description="3-letter ISO account currency code.",
    )
    spread_points: Decimal = Field(
        default=Decimal("20"),
        ge=Decimal("0"),
        description="Bid/Ask spread in instrument tick_size points (e.g., 20 points = $0.20 on XAUUSD).",
    )
    slippage_points: Decimal = Field(
        default=Decimal("5"),
        ge=Decimal("0"),
        description="Adverse execution slippage in instrument tick_size points per fill.",
    )
    commission_per_lot: Decimal = Field(
        default=Decimal("7.00"),
        ge=Decimal("0"),
        description="Round-turn commission in account currency per 1.0 standard lot.",
    )
    intrabar_tie_break: IntrabarTieBreakPolicy = Field(
        default=IntrabarTieBreakPolicy.CONSERVATIVE_SL_FIRST,
        description="Deterministic tie-breaker when a single bar's High/Low spans both SL and TP.",
    )
    enforce_min_rr_after_costs: bool = Field(
        default=True,
        description="Reject entry fill if spread and slippage reduce effective RR below risk_parameters.minimum_risk_reward_ratio.",
    )
    day_boundary_timezone: str = Field(
        default="America/New_York",
        description="IANA timezone used to determine trading-day rollover (default 5 PM New York).",
    )
    day_rollover_hour: int = Field(
        default=17,
        ge=0,
        le=23,
        description="Hour (0..23) in day_boundary_timezone when trading day rolls over.",
    )
    warmup_bars: int = Field(
        default=0,
        ge=0,
        description="Number of initial bars to skip before invoking strategy evaluation.",
    )
    close_open_positions_at_end: bool = Field(
        default=True,
        description="If True, force-close any remaining open positions on the final bar with ExitReason.END_OF_DATA.",
    )
    annualization_periods: int = Field(
        default=252,
        ge=1,
        description="Annualization factor for Sharpe and Sortino ratios.",
    )
    risk_parameters: RiskParameters = Field(
        default_factory=RiskParameters,
        description="Deterministic RiskParameters enforced by RiskEngine during the backtest.",
    )
    instrument_spec: InstrumentSpec = Field(
        default_factory=default_xauusd_instrument_spec,
        description="Broker contract specification for the backtested symbol.",
    )

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        """Normalize symbol to uppercase alphanumeric characters."""
        cleaned = value.strip().upper()
        if not cleaned.isalnum():
            raise ValueError(f"Symbol must be alphanumeric, got '{value}'.")
        return cleaned

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        """Normalize currency to 3 uppercase letters."""
        cleaned = value.strip().upper()
        if len(cleaned) != 3 or not cleaned.isalpha():
            raise ValueError(f"Currency must be a 3-letter alphabetic code, got '{value}'.")
        return cleaned

    @field_validator("day_boundary_timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        """Validate IANA timezone identifier."""
        cleaned = value.strip()
        try:
            ZoneInfo(cleaned)
        except (ZoneInfoNotFoundError, KeyError) as exc:
            raise ValueError(f"Invalid IANA timezone '{value}'.") from exc
        return cleaned

    @model_validator(mode="after")
    def validate_spec_symbol_alignment(self) -> "BacktestConfig":
        """Ensure instrument_spec symbol matches backtest symbol."""
        if self.instrument_spec.symbol != self.symbol:
            raise ValueError(
                f"instrument_spec.symbol ('{self.instrument_spec.symbol}') must match "
                f"BacktestConfig.symbol ('{self.symbol}')."
            )
        return self
