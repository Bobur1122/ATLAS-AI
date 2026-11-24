"""Market data domain models for ATLAS AI (Candle, Timeframe, InstrumentSpec)."""

from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Timeframe(StrEnum):
    """Standard Forex and metals trading timeframes."""

    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"
    MN1 = "MN1"


class Candle(BaseModel):
    """Validated OHLCV market candle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(
        ...,
        min_length=3,
        max_length=16,
        description="Trading instrument symbol (e.g., XAUUSD).",
    )
    timeframe: Timeframe = Field(
        ...,
        description="Candle timeframe (e.g., M1, M5, M15, H1, H4, D1).",
    )
    timestamp: datetime = Field(
        ...,
        description="Candle open timestamp.",
    )
    open: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Opening price (must be strictly positive).",
    )
    high: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Highest price during the period (must be strictly positive).",
    )
    low: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Lowest price during the period (must be strictly positive).",
    )
    close: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Closing price (must be strictly positive).",
    )
    volume: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Tick or real volume (must be non-negative).",
    )

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        """Normalize symbol to uppercase alphanumeric characters."""
        cleaned = value.strip().upper()
        if not cleaned.isalnum():
            raise ValueError(f"Candle symbol must be alphanumeric, got '{value}'.")
        return cleaned

    @model_validator(mode="after")
    def validate_ohlc_relationships(self) -> "Candle":
        """Enforce OHLC price invariants: high >= max(open, close), low <= min(open, close), high >= low."""
        if self.high < self.low:
            raise ValueError(
                f"Invalid Candle OHLC: high ({self.high}) cannot be less than low ({self.low})."
            )
        max_oc = max(self.open, self.close)
        min_oc = min(self.open, self.close)
        if self.high < max_oc:
            raise ValueError(
                f"Invalid Candle OHLC: high ({self.high}) must be >= max(open, close) ({max_oc})."
            )
        if self.low > min_oc:
            raise ValueError(
                f"Invalid Candle OHLC: low ({self.low}) must be <= min(open, close) ({min_oc})."
            )
        return self


class InstrumentSpec(BaseModel):
    """Explicit broker contract and volume specification for a trading instrument.

    Never hardcode XAUUSD contract specifications; brokers differ in contract size,
    tick size, tick value, and volume limits.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(
        default="XAUUSD",
        min_length=3,
        max_length=16,
        description="Instrument symbol to which this specification applies.",
    )
    contract_size: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Units per 1.0 standard lot (e.g., 100 troy oz for standard XAUUSD).",
    )
    tick_size: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Minimum price movement step (e.g., 0.01 for XAUUSD).",
    )
    tick_value: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Monetary value in account currency of 1 tick_size move for 1.0 lot (e.g., 1.00 USD).",
    )
    volume_step: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Minimum increment between allowed lot sizes (e.g., 0.01 lot).",
    )
    minimum_volume: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Minimum allowed order volume in lots (e.g., 0.01 lot).",
    )
    maximum_volume: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Maximum allowed order volume in lots per trade (e.g., 100.0 lots).",
    )
    margin_rate: Decimal = Field(
        default=Decimal("0.01"),
        gt=Decimal("0"),
        le=Decimal("1"),
        description="Initial margin requirement ratio per notional value (e.g., 0.01 for 1:100 leverage).",
    )

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        """Normalize symbol to uppercase alphanumeric characters."""
        cleaned = value.strip().upper()
        if not cleaned.isalnum():
            raise ValueError(f"Instrument symbol must be alphanumeric, got '{value}'.")
        return cleaned

    @model_validator(mode="after")
    def validate_volume_hierarchy(self) -> "InstrumentSpec":
        """Ensure minimum_volume <= maximum_volume and volume_step <= minimum_volume."""
        if self.minimum_volume > self.maximum_volume:
            raise ValueError(
                f"minimum_volume ({self.minimum_volume}) cannot exceed maximum_volume ({self.maximum_volume})."
            )
        if self.volume_step > self.minimum_volume:
            raise ValueError(
                f"volume_step ({self.volume_step}) cannot exceed minimum_volume ({self.minimum_volume})."
            )
        remainder = self.minimum_volume % self.volume_step
        if remainder != Decimal("0"):
            raise ValueError(
                f"minimum_volume ({self.minimum_volume}) must be an exact multiple of volume_step ({self.volume_step})."
            )
        return self
