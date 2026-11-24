"""Configuration and report models for the ATLAS AI market data and feature pipeline."""

from datetime import date, datetime
from enum import StrEnum
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MissingBarPolicy(StrEnum):
    """Policy controlling how unexpected missing bars (gaps) are handled."""

    STRICT = "STRICT"
    REPORT = "REPORT"
    ALLOW = "ALLOW"


class DataValidationIssueCode(StrEnum):
    """Machine-readable codes for OHLCV and time-series validation issues."""

    EMPTY_DATASET = "EMPTY_DATASET"
    MISSING_REQUIRED_COLUMNS = "MISSING_REQUIRED_COLUMNS"
    NON_NUMERIC_VALUE = "NON_NUMERIC_VALUE"
    NAN_OR_INFINITE_VALUE = "NAN_OR_INFINITE_VALUE"
    INVALID_TIMESTAMP = "INVALID_TIMESTAMP"
    NON_POSITIVE_PRICE = "NON_POSITIVE_PRICE"
    NEGATIVE_VOLUME = "NEGATIVE_VOLUME"
    INVALID_OHLC_RELATIONSHIP = "INVALID_OHLC_RELATIONSHIP"
    DUPLICATE_TIMESTAMP = "DUPLICATE_TIMESTAMP"
    OUT_OF_ORDER_TIMESTAMP = "OUT_OF_ORDER_TIMESTAMP"
    UNEXPECTED_TIMEFRAME_INTERVAL = "UNEXPECTED_TIMEFRAME_INTERVAL"
    MISSING_EXPECTED_BAR = "MISSING_EXPECTED_BAR"


class TradingSessionName(StrEnum):
    """Canonical Forex/metals trading session classifications."""

    ASIAN = "ASIAN"
    LONDON = "LONDON"
    LONDON_NY_OVERLAP = "LONDON_NY_OVERLAP"
    NEW_YORK = "NEW_YORK"
    OFF_HOURS = "OFF_HOURS"


class SessionBoundary(BaseModel):
    """Configurable hour boundary [start_hour, end_hour) in session_timezone."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: TradingSessionName = Field(..., description="Session classification name.")
    start_hour: int = Field(..., ge=0, le=23, description="Inclusive start hour (0-23).")
    end_hour: int = Field(..., ge=1, le=24, description="Exclusive end hour (1-24).")

    @model_validator(mode="after")
    def validate_hours(self) -> "SessionBoundary":
        """Ensure start_hour < end_hour."""
        if self.start_hour >= self.end_hour:
            raise ValueError(
                f"SessionBoundary {self.name}: start_hour ({self.start_hour}) must be < end_hour ({self.end_hour})."
            )
        return self


class MarketCalendarConfig(BaseModel):
    """Configurable Forex/metals market calendar, weekend closure, and session schedule.

    Default schedule reflects standard UTC Forex/XAUUSD hours:
    - Market closes Friday at 22:00 UTC and reopens Sunday at 22:00 UTC.
    - Daily rollover break (optional, default 22:00 to 23:00 UTC for XAUUSD CME/broker break,
      configurable via `daily_break_start_hour` / `daily_break_end_hour`).
    - Sessions are evaluated in `session_timezone` (default UTC) and are completely configurable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    market_timezone: str = Field(
        default="UTC",
        description="IANA timezone name for market open/close rules (e.g., 'UTC', 'America/New_York').",
    )
    session_timezone: str = Field(
        default="UTC",
        description="IANA timezone name used for hour_of_day, day_of_week, and trading session boundaries.",
    )
    friday_close_hour: int = Field(
        default=22,
        ge=0,
        le=24,
        description="Hour on Friday (in market_timezone) when the market closes for the weekend.",
    )
    sunday_open_hour: int = Field(
        default=22,
        ge=0,
        le=24,
        description="Hour on Sunday (in market_timezone) when the market reopens.",
    )
    daily_break_start_hour: int | None = Field(
        default=None,
        ge=0,
        le=23,
        description="Optional daily maintenance break start hour (in market_timezone).",
    )
    daily_break_end_hour: int | None = Field(
        default=None,
        ge=1,
        le=24,
        description="Optional daily maintenance break end hour (in market_timezone).",
    )
    closed_dates: frozenset[date] = Field(
        default_factory=frozenset,
        description="Explicit market holidays (in market_timezone) when legitimate closures occur.",
    )
    asian_start_hour: int = Field(default=0, ge=0, le=23)
    asian_end_hour: int = Field(default=7, ge=1, le=24)
    london_start_hour: int = Field(default=7, ge=0, le=23)
    london_end_hour: int = Field(default=12, ge=1, le=24)
    overlap_start_hour: int = Field(default=12, ge=0, le=23)
    overlap_end_hour: int = Field(default=16, ge=1, le=24)
    new_york_start_hour: int = Field(default=16, ge=0, le=23)
    new_york_end_hour: int = Field(default=21, ge=1, le=24)

    @field_validator("market_timezone", "session_timezone")
    @classmethod
    def validate_timezone_name(cls, value: str) -> str:
        """Ensure timezone string is a valid IANA ZoneInfo identifier."""
        cleaned = value.strip()
        try:
            ZoneInfo(cleaned)
        except (ZoneInfoNotFoundError, KeyError, ValueError) as exc:
            raise ValueError(f"Unknown or invalid timezone '{value}'.") from exc
        return cleaned

    @model_validator(mode="after")
    def validate_break_hours(self) -> "MarketCalendarConfig":
        """Validate daily maintenance break hours if configured."""
        if (self.daily_break_start_hour is None) != (self.daily_break_end_hour is None):
            raise ValueError(
                "Both daily_break_start_hour and daily_break_end_hour must be set together, or both None."
            )
        if (
            self.daily_break_start_hour is not None
            and self.daily_break_end_hour is not None
            and self.daily_break_start_hour >= self.daily_break_end_hour
        ):
            raise ValueError("daily_break_start_hour must be < daily_break_end_hour.")
        return self

    def is_market_open(self, ts: datetime) -> bool:
        """Return True if timestamp `ts` falls within open market trading hours."""
        local_ts = ts.astimezone(ZoneInfo(self.market_timezone))
        if local_ts.date() in self.closed_dates:
            return False

        weekday = local_ts.weekday()  # Monday=0 ... Friday=4, Saturday=5, Sunday=6
        hour = local_ts.hour

        # Saturday is always closed
        if weekday == 5:
            return False
        # Friday at or after friday_close_hour is closed
        if weekday == 4 and hour >= self.friday_close_hour:
            return False
        # Sunday before sunday_open_hour is closed
        if weekday == 6 and hour < self.sunday_open_hour:
            return False

        # Optional daily rollover break
        if (
            self.daily_break_start_hour is not None
            and self.daily_break_end_hour is not None
            and self.daily_break_start_hour <= hour < self.daily_break_end_hour
        ):
            return False

        return True

    def classify_session(self, ts: datetime) -> TradingSessionName:
        """Deterministically classify the trading session for a timestamp in `session_timezone`."""
        local_ts = ts.astimezone(ZoneInfo(self.session_timezone))
        hour = local_ts.hour

        if self.overlap_start_hour <= hour < self.overlap_end_hour:
            return TradingSessionName.LONDON_NY_OVERLAP
        if self.london_start_hour <= hour < self.london_end_hour:
            return TradingSessionName.LONDON
        if self.new_york_start_hour <= hour < self.new_york_end_hour:
            return TradingSessionName.NEW_YORK
        if self.asian_start_hour <= hour < self.asian_end_hour:
            return TradingSessionName.ASIAN
        return TradingSessionName.OFF_HOURS


class DataValidationIssue(BaseModel):
    """Single validation or integrity issue discovered in a market dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: DataValidationIssueCode = Field(..., description="Machine-readable issue code.")
    message: str = Field(..., description="Human-readable diagnostic explanation.")
    row_index: int | None = Field(default=None, description="Zero-based row index where the issue occurred.")
    timestamp: datetime | None = Field(default=None, description="Timestamp associated with the issue, if known.")
    is_error: bool = Field(
        default=True,
        description="True if this issue causes validation failure; False if informational/warning.",
    )


class MissingBarGap(BaseModel):
    """Record of an unexpected gap of missing bars between two consecutive candles."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    previous_timestamp: datetime = Field(..., description="Timestamp of the bar before the gap.")
    next_timestamp: datetime = Field(..., description="Timestamp of the bar after the gap.")
    missing_bars_count: int = Field(..., ge=1, description="Number of expected open-market bars missing.")
    first_missing_timestamp: datetime = Field(..., description="First expected bar timestamp that was missing.")


class DataValidationReport(BaseModel):
    """Structured report produced by OHLCV and time-series integrity validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_valid: bool = Field(..., description="True if the dataset passed validation under the active policy.")
    symbol: str = Field(..., description="Validated instrument symbol.")
    timeframe: str = Field(..., description="Validated timeframe string.")
    total_rows: int = Field(..., ge=0, description="Total rows inspected.")
    missing_bar_policy: MissingBarPolicy = Field(..., description="Policy applied for missing bars.")
    missing_bars_total: int = Field(default=0, ge=0, description="Total count of unexpected missing bars.")
    gaps: tuple[MissingBarGap, ...] = Field(default_factory=tuple, description="Detected unexpected missing-bar gaps.")
    issues: tuple[DataValidationIssue, ...] = Field(
        default_factory=tuple,
        description="All validation issues and warnings recorded.",
    )

    @property
    def error_codes(self) -> list[DataValidationIssueCode]:
        """Return all issue codes where `is_error=True`."""
        return [issue.code for issue in self.issues if issue.is_error]

    @property
    def all_issue_codes(self) -> list[DataValidationIssueCode]:
        """Return all issue codes (errors and warnings)."""
        return [issue.code for issue in self.issues]


class FeatureConfig(BaseModel):
    """Configuration for deterministic technical feature engineering."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    include_volatility: bool = Field(
        default=True,
        description="Compute True Range (tr), ATR (atr_14), and ATR percentage (atr_pct_14).",
    )
    atr_period: int = Field(
        default=14,
        ge=1,
        le=500,
        description="Lookback window for Average True Range (ATR).",
    )
    include_moving_averages: bool = Field(
        default=True,
        description="Compute configured SMA and EMA series.",
    )
    sma_periods: tuple[int, ...] = Field(
        default=(20, 50, 100, 200),
        description="Simple Moving Average lookback windows.",
    )
    ema_periods: tuple[int, ...] = Field(
        default=(9, 21, 50),
        description="Exponential Moving Average spans.",
    )
    include_candle_features: bool = Field(
        default=True,
        description="Compute candle_body, upper_wick, lower_wick, candle_range, candle_direction, body_to_range_ratio.",
    )
    include_structure_features: bool = Field(
        default=True,
        description="Compute rolling_high, rolling_low, prev_high, prev_low, session_high, session_low.",
    )
    structure_lookback: int = Field(
        default=20,
        ge=2,
        le=1000,
        description="Window length for rolling_high and rolling_low.",
    )
    include_time_features: bool = Field(
        default=True,
        description="Compute hour_of_day, day_of_week, and trading_session.",
    )
    calendar: MarketCalendarConfig = Field(
        default_factory=MarketCalendarConfig,
        description="Market calendar and session boundary configuration.",
    )

    @field_validator("sma_periods", "ema_periods")
    @classmethod
    def validate_positive_periods(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        """Ensure all moving average periods are strictly positive integers."""
        for p in values:
            if p < 1:
                raise ValueError(f"Moving average period must be >= 1, got {p}.")
        return values
