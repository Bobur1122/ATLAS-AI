"""Risk management domain models for ATLAS AI (AccountState, RiskParameters, RiskDecision)."""

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RiskRejectionCode(StrEnum):
    """Machine-readable rejection codes returned by the Deterministic Risk Engine."""

    TRADING_DISABLED = "TRADING_DISABLED"
    WAIT_SIGNAL_NOT_EXECUTABLE = "WAIT_SIGNAL_NOT_EXECUTABLE"
    MISSING_INSTRUMENT_SPEC = "MISSING_INSTRUMENT_SPEC"
    SYMBOL_MISMATCH = "SYMBOL_MISMATCH"
    INVALID_SIGNAL_PRICES = "INVALID_SIGNAL_PRICES"
    INVALID_STOP_DISTANCE = "INVALID_STOP_DISTANCE"
    INVALID_ACCOUNT_EQUITY = "INVALID_ACCOUNT_EQUITY"
    INSUFFICIENT_FREE_MARGIN = "INSUFFICIENT_FREE_MARGIN"
    MAX_OPEN_POSITIONS_REACHED = "MAX_OPEN_POSITIONS_REACHED"
    DAILY_LOSS_LIMIT_EXCEEDED = "DAILY_LOSS_LIMIT_EXCEEDED"
    TOTAL_DRAWDOWN_LIMIT_EXCEEDED = "TOTAL_DRAWDOWN_LIMIT_EXCEEDED"
    INSUFFICIENT_RISK_REWARD_RATIO = "INSUFFICIENT_RISK_REWARD_RATIO"
    POSITION_SIZE_BELOW_MINIMUM = "POSITION_SIZE_BELOW_MINIMUM"
    POSITION_SIZE_EXCEEDS_MAXIMUM = "POSITION_SIZE_EXCEEDS_MAXIMUM"
    RISK_PER_TRADE_EXCEEDED = "RISK_PER_TRADE_EXCEEDED"


class AccountState(BaseModel):
    """Snapshot of current trading account balances, equity, margin, and exposure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    balance: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Current closed-trade account balance (must be > 0).",
    )
    equity: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Current floating account equity (must be > 0).",
    )
    free_margin: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Available free margin for opening new positions (must be >= 0 and <= equity).",
    )
    currency: str = Field(
        default="USD",
        min_length=3,
        max_length=3,
        description="3-letter ISO account currency code (e.g., USD).",
    )
    open_positions_count: int = Field(
        default=0,
        ge=0,
        description="Number of currently open positions (must be >= 0).",
    )
    current_daily_pnl: Decimal = Field(
        default=Decimal("0"),
        description="Realized + unrealized PnL for the current trading day (negative for net loss).",
    )
    daily_starting_balance: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Account balance/equity at the start of the current trading day (must be > 0).",
    )
    peak_balance: Decimal | None = Field(
        default=None,
        gt=Decimal("0"),
        description="Optional high-water mark balance for total drawdown calculation. Defaults to max(balance, daily_starting_balance).",
    )

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        """Normalize currency code to 3 uppercase letters."""
        cleaned = value.strip().upper()
        if len(cleaned) != 3 or not cleaned.isalpha():
            raise ValueError(f"Currency must be a 3-letter alphabetic code, got '{value}'.")
        return cleaned

    @model_validator(mode="after")
    def validate_margin_constraints(self) -> "AccountState":
        """Ensure free_margin does not exceed total floating equity."""
        if self.free_margin > self.equity:
            raise ValueError(
                f"Invalid AccountState: free_margin ({self.free_margin}) cannot exceed equity ({self.equity})."
            )
        return self

    @property
    def effective_peak_balance(self) -> Decimal:
        """Return the reference high-water mark used to measure total account drawdown."""
        candidates = [self.balance, self.daily_starting_balance]
        if self.peak_balance is not None:
            candidates.append(self.peak_balance)
        return max(candidates)


class RiskParameters(BaseModel):
    """Deterministic risk limits and position sizing constraints."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    risk_per_trade_percent: Decimal = Field(
        default=Decimal("1.0"),
        gt=Decimal("0"),
        le=Decimal("10.0"),
        description="Percentage of account equity risked per trade (0 < x <= 10%).",
    )
    max_daily_loss_percent: Decimal = Field(
        default=Decimal("3.0"),
        gt=Decimal("0"),
        le=Decimal("50.0"),
        description="Maximum allowed daily loss as a percentage of daily_starting_balance.",
    )
    max_total_drawdown_percent: Decimal = Field(
        default=Decimal("10.0"),
        gt=Decimal("0"),
        le=Decimal("80.0"),
        description="Maximum allowed total drawdown as a percentage of reference peak balance.",
    )
    max_open_positions: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Maximum number of simultaneous open positions allowed.",
    )
    minimum_risk_reward_ratio: Decimal = Field(
        default=Decimal("1.5"),
        gt=Decimal("0"),
        le=Decimal("50.0"),
        description="Minimum required Take-Profit distance to Stop-Loss distance ratio.",
    )
    max_position_size: Decimal = Field(
        default=Decimal("10.0"),
        gt=Decimal("0"),
        description="Hard upper cap on position size in lots allowed by the risk policy.",
    )
    allow_trading: bool = Field(
        default=True,
        description="Master kill-switch to enable or disable new trade approvals.",
    )

    @model_validator(mode="after")
    def validate_risk_hierarchy(self) -> "RiskParameters":
        """Ensure per-trade risk does not exceed daily loss limit, and daily loss limit does not exceed total drawdown limit."""
        if self.risk_per_trade_percent > self.max_daily_loss_percent:
            raise ValueError(
                f"risk_per_trade_percent ({self.risk_per_trade_percent}%) cannot exceed "
                f"max_daily_loss_percent ({self.max_daily_loss_percent}%)."
            )
        if self.max_daily_loss_percent > self.max_total_drawdown_percent:
            raise ValueError(
                f"max_daily_loss_percent ({self.max_daily_loss_percent}%) cannot exceed "
                f"max_total_drawdown_percent ({self.max_total_drawdown_percent}%)."
            )
        return self


class PositionSizeResult(BaseModel):
    """Deterministic output of position size calculation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    approved: bool = Field(..., description="Whether a valid position size was computed.")
    rejection_reason: str | None = Field(
        default=None,
        description="Machine-readable reason if position sizing failed.",
    )
    raw_position_size: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Unrounded theoretical position size before step floor rounding.",
    )
    rounded_position_size: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Position size rounded down (floored) to the instrument volume_step.",
    )
    allowed_risk_amount: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Maximum monetary risk allowed by risk_per_trade_percent.",
    )
    actual_risk_amount: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Actual monetary risk at rounded_position_size (guaranteed <= allowed_risk_amount).",
    )
    actual_risk_percent: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Actual percentage of account equity risked at rounded_position_size.",
    )
    stop_loss_distance: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Price distance between entry_price and stop_loss.",
    )
    loss_per_lot: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Monetary loss per 1.0 standard lot for the stop_loss_distance.",
    )


class RiskDecision(BaseModel):
    """Structured, deterministic output of RiskEngine.evaluate()."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    approved: bool = Field(
        ...,
        description="True only if all deterministic risk and safety gates pass.",
    )
    rejection_reason: str | None = Field(
        default=None,
        description="Machine-readable rejection code and explanation when approved is False.",
    )
    calculated_risk_amount: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Monetary risk amount in account currency for the sized position.",
    )
    calculated_position_size: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Approved position size in lots (0 if rejected).",
    )
    risk_reward_ratio: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Calculated reward-to-risk ratio (take_profit_distance / stop_loss_distance).",
    )
    risk_percent: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Actual percentage of account equity risked by calculated_position_size.",
    )
    validation_details: dict[str, object] = Field(
        default_factory=dict,
        description="Detailed audit trail of every risk check performed.",
    )
