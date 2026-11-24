"""Trading signal domain models for ATLAS AI (SignalDirection, TradeSignal)."""

from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atlas_ai.domain.market import Timeframe


class SignalDirection(StrEnum):
    """Supported trading signal directions."""

    BUY = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"


class TradeSignal(BaseModel):
    """Structured trading decision emitted by a strategy or decision aggregator.

    Invariants:
    - Confidence must be in [0.0, 1.0].
    - BUY signals require positive entry_price, stop_loss, take_profit with:
      stop_loss < entry_price < take_profit.
    - SELL signals require positive entry_price, stop_loss, take_profit with:
      take_profit < entry_price < stop_loss.
    - WAIT signals represent an explicit decision not to trade and are never executable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(
        ...,
        min_length=3,
        max_length=16,
        description="Target trading instrument symbol (e.g., XAUUSD).",
    )
    direction: SignalDirection = Field(
        ...,
        description="Trading signal direction (BUY, SELL, or WAIT).",
    )
    entry_price: Decimal | None = Field(
        default=None,
        description="Proposed entry price (required and > 0 for BUY/SELL).",
    )
    stop_loss: Decimal | None = Field(
        default=None,
        description="Proposed stop-loss price (required and > 0 for BUY/SELL).",
    )
    take_profit: Decimal | None = Field(
        default=None,
        description="Proposed take-profit price (required and > 0 for BUY/SELL).",
    )
    timeframe: Timeframe = Field(
        ...,
        description="Timeframe on which the signal was generated.",
    )
    strategy_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Identifier of the strategy that generated this signal.",
    )
    confidence: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Deterministic or statistical confidence score between 0.0 and 1.0.",
    )
    timestamp: datetime = Field(
        ...,
        description="Timestamp when the signal was generated.",
    )
    rationale: str = Field(
        ...,
        min_length=1,
        description="Human-readable explanation of the rules or conditions that triggered the signal.",
    )

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        """Normalize symbol to uppercase alphanumeric characters."""
        cleaned = value.strip().upper()
        if not cleaned.isalnum():
            raise ValueError(f"Signal symbol must be alphanumeric, got '{value}'.")
        return cleaned

    @field_validator("strategy_id", "rationale")
    @classmethod
    def ensure_non_blank_text(cls, value: str) -> str:
        """Reject whitespace-only strings for strategy_id and rationale."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty or whitespace only.")
        return cleaned

    @model_validator(mode="after")
    def validate_signal_geometry(self) -> "TradeSignal":
        """Enforce directional price ordering for BUY/SELL and non-executability for WAIT."""
        if self.direction in (SignalDirection.BUY, SignalDirection.SELL):
            if self.entry_price is None or self.stop_loss is None or self.take_profit is None:
                raise ValueError(
                    f"{self.direction.value} signal requires entry_price, stop_loss, and take_profit."
                )
            if self.entry_price <= Decimal("0") or self.stop_loss <= Decimal("0") or self.take_profit <= Decimal("0"):
                raise ValueError(
                    f"{self.direction.value} signal prices (entry_price, stop_loss, take_profit) must all be > 0."
                )

            if self.direction == SignalDirection.BUY:
                if not (self.stop_loss < self.entry_price < self.take_profit):
                    raise ValueError(
                        f"Invalid BUY signal geometry: require stop_loss ({self.stop_loss}) "
                        f"< entry_price ({self.entry_price}) < take_profit ({self.take_profit})."
                    )
            elif self.direction == SignalDirection.SELL:
                if not (self.take_profit < self.entry_price < self.stop_loss):
                    raise ValueError(
                        f"Invalid SELL signal geometry: require take_profit ({self.take_profit}) "
                        f"< entry_price ({self.entry_price}) < stop_loss ({self.stop_loss})."
                    )
        else:
            # WAIT signal: if any optional price is provided, it must not be negative
            for field_name, price_val in (
                ("entry_price", self.entry_price),
                ("stop_loss", self.stop_loss),
                ("take_profit", self.take_profit),
            ):
                if price_val is not None and price_val <= Decimal("0"):
                    raise ValueError(
                        f"WAIT signal {field_name} must be None or > 0, got {price_val}."
                    )
        return self

    @property
    def is_executable(self) -> bool:
        """Return True only for BUY or SELL signals; WAIT is never executable."""
        return self.direction in (SignalDirection.BUY, SignalDirection.SELL)

    @property
    def stop_loss_distance(self) -> Decimal | None:
        """Return absolute price distance between entry_price and stop_loss."""
        if self.entry_price is None or self.stop_loss is None:
            return None
        return abs(self.entry_price - self.stop_loss)

    @property
    def take_profit_distance(self) -> Decimal | None:
        """Return absolute price distance between take_profit and entry_price."""
        if self.entry_price is None or self.take_profit is None:
            return None
        return abs(self.take_profit - self.entry_price)
