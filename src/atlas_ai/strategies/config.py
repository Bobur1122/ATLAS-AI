"""Configuration models for all ATLAS AI Phase 3 strategies, analyzers, confluence, and signal lifecycle."""

from decimal import Decimal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atlas_ai.domain.market import Timeframe
from atlas_ai.strategies.models import NewsImpactLevel


class RegimeDetectorConfig(BaseModel):
    """Configuration for MarketRegimeDetector."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    high_volatility_atr_pct: Decimal = Field(
        default=Decimal("0.60"),
        gt=Decimal("0"),
        description="ATR percentage threshold above which regime is HIGH_VOLATILITY (e.g., 0.60% on H1/M15).",
    )
    low_volatility_atr_pct: Decimal = Field(
        default=Decimal("0.08"),
        gt=Decimal("0"),
        description="ATR percentage threshold below which regime is LOW_VOLATILITY.",
    )
    min_ema_separation_atr_ratio: Decimal = Field(
        default=Decimal("0.15"),
        ge=Decimal("0"),
        description="Minimum |EMA9 - EMA50| / ATR_14 ratio required to classify a clean UPTREND/DOWNTREND.",
    )

    @model_validator(mode="after")
    def validate_volatility_bounds(self) -> "RegimeDetectorConfig":
        """Ensure low_volatility_atr_pct < high_volatility_atr_pct."""
        if self.low_volatility_atr_pct >= self.high_volatility_atr_pct:
            raise ValueError("low_volatility_atr_pct must be strictly less than high_volatility_atr_pct.")
        return self


class EmaTrendConfig(BaseModel):
    """Configuration for EmaTrendStrategy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fast_ema_col: str = Field(default="ema_9")
    mid_ema_col: str = Field(default="ema_21")
    slow_ema_col: str = Field(default="ema_50")
    atr_col: str = Field(default="atr_14")
    sl_atr_multiplier: Decimal = Field(default=Decimal("1.5"), gt=Decimal("0"))
    rr_ratio: Decimal = Field(default=Decimal("2.0"), ge=Decimal("1.0"))
    require_candle_direction_confirmation: bool = Field(default=True)


class MarketStructureConfig(BaseModel):
    """Configuration for causal swing detection and BOS/CHoCH classification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    left_bars: int = Field(
        default=2,
        ge=1,
        le=50,
        description="Number of bars to the left of a pivot required to form a swing high/low.",
    )
    right_bars: int = Field(
        default=2,
        ge=1,
        le=50,
        description="Number of closed bars to the right of a pivot required to confirm a swing without lookahead.",
    )
    equal_swing_tolerance_atr: Decimal = Field(
        default=Decimal("0.05"),
        ge=Decimal("0"),
        description="Tolerance as a fraction of ATR within which two swings are classified as Equal High/Low.",
    )
    sl_buffer_atr: Decimal = Field(default=Decimal("0.25"), ge=Decimal("0"))
    rr_ratio: Decimal = Field(default=Decimal("2.0"), ge=Decimal("1.0"))


class LiquidityConfig(BaseModel):
    """Configuration for EQH/EQL, swing liquidity pools, and sweep + rejection detection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    left_bars: int = Field(default=2, ge=1, le=50)
    right_bars: int = Field(default=2, ge=1, le=50)
    eq_tolerance_atr: Decimal = Field(
        default=Decimal("0.15"),
        gt=Decimal("0"),
        description="Maximum price difference (as a multiple of ATR) between two swing pivots to form EQH/EQL.",
    )
    min_sweep_penetration_atr: Decimal = Field(
        default=Decimal("0.02"),
        ge=Decimal("0"),
        description="Minimum wick excursion beyond the liquidity level (in ATR multiples) to count as a sweep.",
    )
    require_directional_close: bool = Field(
        default=True,
        description="Require bullish close on SSL/EQL sweep or bearish close on BSL/EQH sweep.",
    )
    sl_buffer_atr: Decimal = Field(default=Decimal("0.30"), ge=Decimal("0"))
    rr_ratio: Decimal = Field(default=Decimal("2.0"), ge=Decimal("1.0"))


class FVGConfig(BaseModel):
    """Configuration for Fair Value Gap detection and FVG Retest evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_gap_atr_ratio: Decimal = Field(
        default=Decimal("0.10"),
        ge=Decimal("0"),
        description="Minimum gap size as a fraction of ATR_14 to qualify as a significant FVG.",
    )
    max_age_bars: int = Field(
        default=40,
        ge=1,
        le=500,
        description="Maximum bars after formation before an unmitigated FVG expires.",
    )
    sl_buffer_atr: Decimal = Field(default=Decimal("0.25"), ge=Decimal("0"))
    rr_ratio: Decimal = Field(default=Decimal("2.0"), ge=Decimal("1.0"))


class PremiumDiscountConfig(BaseModel):
    """Configuration for dealing range Premium / Discount / Equilibrium analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    left_bars: int = Field(default=2, ge=1, le=50)
    right_bars: int = Field(default=2, ge=1, le=50)
    fallback_lookback_bars: int = Field(default=50, ge=5, le=500)
    equilibrium_tolerance: Decimal = Field(
        default=Decimal("0.05"),
        ge=Decimal("0"),
        le=Decimal("0.25"),
        description="Half-width around 0.50 classified as EQUILIBRIUM (e.g., 0.05 -> [0.45, 0.55]).",
    )


class PreviousDayLevelsConfig(BaseModel):
    """Configuration for Previous Day High/Low/Close (PDH/PDL/PDC) analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    day_boundary_timezone: str = Field(
        default="America/New_York",
        description="IANA timezone used to determine trading day boundaries (e.g., 17:00 NY close).",
    )
    day_rollover_hour: int = Field(
        default=17,
        ge=0,
        le=23,
        description="Hour in day_boundary_timezone when a new trading day starts (17 = 5 PM NY rollover, or 0 for midnight).",
    )
    proximity_atr_threshold: Decimal = Field(
        default=Decimal("0.50"),
        gt=Decimal("0"),
        description="Distance in ATR multiples considered 'near' PDH or PDL.",
    )

    @field_validator("day_boundary_timezone")
    @classmethod
    def validate_tz(cls, value: str) -> str:
        """Ensure timezone is a valid IANA ZoneInfo identifier."""
        cleaned = value.strip()
        try:
            ZoneInfo(cleaned)
        except (ZoneInfoNotFoundError, KeyError, ValueError) as exc:
            raise ValueError(f"Invalid timezone '{value}'.") from exc
        return cleaned


class KillZoneSessionConfig(BaseModel):
    """Configurable session and ICT Kill Zone schedule with DST-aware timezone support."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timezone: str = Field(
        default="America/New_York",
        description="IANA timezone for evaluating session and Kill Zone hours (DST-aware).",
    )
    asian_start_hour: int = Field(default=19, ge=0, le=23, description="Asian session start hour (19:00 NY).")
    asian_end_hour: int = Field(default=0, ge=0, le=23, description="Asian session end hour (00:00 NY).")
    london_start_hour: int = Field(default=2, ge=0, le=23, description="London session start hour (02:00 NY).")
    london_end_hour: int = Field(default=8, ge=0, le=23, description="London session end hour (08:00 NY).")
    london_killzone_start_hour: int = Field(default=2, ge=0, le=23, description="London Kill Zone start (02:00 NY).")
    london_killzone_end_hour: int = Field(default=5, ge=0, le=23, description="London Kill Zone end (05:00 NY).")
    ny_start_hour: int = Field(default=8, ge=0, le=23, description="New York session start (08:00 NY).")
    ny_end_hour: int = Field(default=17, ge=0, le=23, description="New York session end (17:00 NY).")
    ny_killzone_start_hour: int = Field(default=7, ge=0, le=23, description="New York Kill Zone start (07:00 NY).")
    ny_killzone_end_hour: int = Field(default=10, ge=0, le=23, description="New York Kill Zone end (10:00 NY).")
    overlap_start_hour: int = Field(default=8, ge=0, le=23, description="London/NY overlap start (08:00 NY).")
    overlap_end_hour: int = Field(default=12, ge=0, le=23, description="London/NY overlap end (12:00 NY).")
    require_killzone_for_entry: bool = Field(
        default=False,
        description="If True, signals outside London or NY Kill Zones are rejected.",
    )

    @field_validator("timezone")
    @classmethod
    def validate_tz(cls, value: str) -> str:
        """Ensure timezone is a valid IANA ZoneInfo identifier."""
        cleaned = value.strip()
        try:
            ZoneInfo(cleaned)
        except (ZoneInfoNotFoundError, KeyError, ValueError) as exc:
            raise ValueError(f"Invalid timezone '{value}'.") from exc
        return cleaned


class MultiTimeframeConfig(BaseModel):
    """Configuration for MultiTimeframeAnalyzer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    htf_timeframe: Timeframe = Field(default=Timeframe.H1, description="Higher timeframe for directional bias.")
    mtf_timeframe: Timeframe = Field(default=Timeframe.M15, description="Middle timeframe for structure & setup.")
    ltf_timeframe: Timeframe = Field(default=Timeframe.M5, description="Lower timeframe for entry confirmation.")
    precision_timeframe: Timeframe | None = Field(
        default=Timeframe.M1,
        description="Optional precision timeframe (M1).",
    )
    require_precision_timeframe: bool = Field(
        default=False,
        description="Do not require M1 precision confirmation by default.",
    )


class NewsFilterConfig(BaseModel):
    """Configuration for deterministic economic calendar news filtering."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(default=True)
    mandatory_news_data: bool = Field(
        default=False,
        description="If True, UNKNOWN news status (when events list is None) blocks trade approval.",
    )
    monitored_currencies: frozenset[str] = Field(
        default=frozenset({"USD"}),
        description="Currencies relevant to the instrument (e.g., USD for XAUUSD).",
    )
    minutes_before_high_impact: int = Field(default=30, ge=0, le=720)
    minutes_after_high_impact: int = Field(default=30, ge=0, le=720)
    minutes_before_medium_impact: int = Field(default=15, ge=0, le=720)
    minutes_after_medium_impact: int = Field(default=15, ge=0, le=720)
    block_on_high_impact: bool = Field(default=True)
    block_on_medium_impact: bool = Field(default=False)
    medium_impact_confidence_multiplier: Decimal = Field(
        default=Decimal("0.75"),
        ge=Decimal("0"),
        le=Decimal("1"),
    )
    minimum_impact_threshold: NewsImpactLevel = Field(default=NewsImpactLevel.MEDIUM)


class ConfluenceWeightsConfig(BaseModel):
    """Configurable module weights, activation flags, and thresholds for ConfluenceEngine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enable_ema_trend: bool = Field(default=True)
    enable_market_structure: bool = Field(default=True)
    enable_liquidity_sweep: bool = Field(default=True)
    enable_fvg: bool = Field(default=True)
    enable_fvg_retest: bool = Field(default=True)
    enable_premium_discount: bool = Field(default=True)
    enable_pdh_pdl: bool = Field(default=True)
    enable_kill_zone: bool = Field(default=True)
    enable_mtf: bool = Field(default=True)
    enable_news_filter: bool = Field(default=True)
    enable_knowledge_strategy: bool = Field(default=False)
    enable_ml_strategy: bool = Field(default=False)

    # Primary directional strategy weights
    weight_ema_trend: Decimal = Field(default=Decimal("0.15"), ge=Decimal("0"))
    weight_market_structure: Decimal = Field(default=Decimal("0.25"), ge=Decimal("0"))
    weight_liquidity_sweep: Decimal = Field(default=Decimal("0.20"), ge=Decimal("0"))
    # Note: FVG formation and FVG retest are correlated; FVG formation carries a smaller base weight
    # while FVG retest carries the actionable entry weight.
    weight_fvg: Decimal = Field(default=Decimal("0.05"), ge=Decimal("0"))
    weight_fvg_retest: Decimal = Field(default=Decimal("0.15"), ge=Decimal("0"))
    weight_mtf: Decimal = Field(default=Decimal("0.20"), ge=Decimal("0"))
    weight_knowledge_strategy: Decimal = Field(default=Decimal("0.20"), ge=Decimal("0"))
    weight_ml_strategy: Decimal = Field(default=Decimal("0.20"), ge=Decimal("0"))

    # Contextual confluence bonus/penalty adjustments (Premium/Discount, PDH/PDL, KillZone)
    contextual_alignment_bonus: Decimal = Field(
        default=Decimal("0.06"),
        ge=Decimal("0"),
        le=Decimal("0.25"),
        description="Additive score bonus when a contextual filter aligns with the primary directional bias.",
    )
    contextual_conflict_penalty: Decimal = Field(
        default=Decimal("0.08"),
        ge=Decimal("0"),
        le=Decimal("0.30"),
        description="Score penalty when a contextual filter contradicts the primary directional bias (e.g., BUY in deep PREMIUM).",
    )

    minimum_confidence_threshold: Decimal = Field(
        default=Decimal("0.55"),
        gt=Decimal("0"),
        le=Decimal("1.0"),
        description="Minimum final_confidence required to emit a BUY or SELL decision.",
    )
    minimum_directional_edge: Decimal = Field(
        default=Decimal("0.15"),
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Minimum |buy_score - sell_score| required so contradictory evidence returns WAIT.",
    )
    minimum_agreeing_directional_modules: int = Field(
        default=1,
        ge=1,
        description="Minimum number of primary directional modules that must actively vote BUY or SELL.",
    )


class SignalGeneratorConfig(BaseModel):
    """Configuration for structural/ATR TradeSignal construction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    default_rr_ratio: Decimal = Field(default=Decimal("2.0"), ge=Decimal("1.0"))
    atr_sl_multiplier: Decimal = Field(default=Decimal("1.5"), gt=Decimal("0"))
    min_stop_atr_multiplier: Decimal = Field(default=Decimal("0.25"), gt=Decimal("0"))
    max_stop_atr_multiplier: Decimal = Field(default=Decimal("5.0"), gt=Decimal("0"))


class SignalLifecycleConfig(BaseModel):
    """Configuration for SignalLifecycleManager."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    default_expiration_minutes: int = Field(default=240, ge=1, le=10080)
    entry_zone_dedup_tolerance: Decimal = Field(
        default=Decimal("2.00"),
        ge=Decimal("0"),
        description="Price tolerance within which an already-active signal on the same symbol/timeframe/direction is considered a duplicate.",
    )
