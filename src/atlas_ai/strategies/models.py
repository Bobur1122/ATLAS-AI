"""Domain and evaluation models for the ATLAS AI Phase 3 Strategy & SMC/ICT Engine."""

from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atlas_ai.domain.market import Timeframe
from atlas_ai.domain.risk import RiskDecision
from atlas_ai.domain.signals import SignalDirection, TradeSignal


class MarketRegime(StrEnum):
    """Market regime classifications produced by MarketRegimeDetector."""

    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    UNKNOWN = "UNKNOWN"


class SwingType(StrEnum):
    """Structural swing point classifications."""

    HH = "HH"  # Higher High
    LH = "LH"  # Lower High
    EH = "EH"  # Equal High
    HL = "HL"  # Higher Low
    LL = "LL"  # Lower Low
    EL = "EL"  # Equal Low
    HIGH = "HIGH"  # Initial Swing High
    LOW = "LOW"  # Initial Swing Low


class StructureEventType(StrEnum):
    """Market structure break classifications (BOS and CHoCH)."""

    BULLISH_BOS = "BULLISH_BOS"
    BEARISH_BOS = "BEARISH_BOS"
    BULLISH_CHOCH = "BULLISH_CHOCH"
    BEARISH_CHOCH = "BEARISH_CHOCH"
    NONE = "NONE"


class PriceZoneClassification(StrEnum):
    """Dealing range price zone classifications."""

    DISCOUNT = "DISCOUNT"
    PREMIUM = "PREMIUM"
    EQUILIBRIUM = "EQUILIBRIUM"
    UNKNOWN = "UNKNOWN"


class NewsImpactLevel(StrEnum):
    """Economic calendar event impact levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class NewsFilterStatus(StrEnum):
    """Status returned by deterministic NewsFilter evaluation."""

    CLEAR = "CLEAR"
    RESTRICTED_MEDIUM = "RESTRICTED_MEDIUM"
    BLOCKED_HIGH = "BLOCKED_HIGH"
    UNKNOWN = "UNKNOWN"


class RiskEvaluationStatus(StrEnum):
    """Status of RiskEngine evaluation on an analytical signal."""

    NOT_EVALUATED = "NOT_EVALUATED"
    RISK_APPROVED = "RISK_APPROVED"
    RISK_REJECTED = "RISK_REJECTED"


class SignalLifecycleState(StrEnum):
    """Auditable states of a trading signal managed by SignalLifecycleManager."""

    GENERATED = "GENERATED"
    PENDING_CONFIRMATION = "PENDING_CONFIRMATION"
    CONFIRMED = "CONFIRMED"
    RISK_APPROVED = "RISK_APPROVED"
    RISK_REJECTED = "RISK_REJECTED"
    ACTIVE = "ACTIVE"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    TARGET_HIT = "TARGET_HIT"
    STOP_LOSS_HIT = "STOP_LOSS_HIT"


class StrategyEvaluationResult(BaseModel):
    """Standardized evaluation result returned by every strategy and analyzer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy_id: str = Field(..., min_length=1, description="Unique strategy or analyzer identifier.")
    direction: SignalDirection = Field(..., description="BUY, SELL, or WAIT decision.")
    timestamp: datetime = Field(..., description="Evaluation timestamp (bar close/open timestamp).")
    entry_price: Decimal | None = Field(default=None, description="Suggested entry price if applicable.")
    stop_loss: Decimal | None = Field(default=None, description="Suggested structural/ATR stop-loss price.")
    take_profit: Decimal | None = Field(default=None, description="Suggested take-profit price.")
    confidence: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Deterministic score in [0.0, 1.0] (not a win probability).",
    )
    rationale: str = Field(..., min_length=1, description="Human-readable explanation of the evaluation.")
    supporting_evidence: dict[str, object] = Field(
        default_factory=dict,
        description="Structured evidence dictionary produced by the module.",
    )
    rejection_reasons: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Reasons why a BUY/SELL signal was not emitted or why confluence failed.",
    )
    is_contextual_filter: bool = Field(
        default=False,
        description="True if this module is a contextual confluence filter (e.g., Premium/Discount, PDH/PDL, KillZone) rather than a primary directional trigger.",
    )


class SwingPoint(BaseModel):
    """Confirmed historical swing high or swing low."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(..., ge=0, description="Bar index where the swing pivot occurred.")
    confirmed_at_index: int = Field(
        ...,
        ge=0,
        description="Bar index when right-side confirmation bars completed (confirmed_at_index >= index).",
    )
    timestamp: datetime = Field(..., description="Timestamp of the swing pivot candle.")
    confirmed_at_timestamp: datetime = Field(..., description="Timestamp when the swing was confirmed.")
    price: Decimal = Field(..., gt=Decimal("0"), description="Swing high or low price.")
    is_high: bool = Field(..., description="True for Swing High, False for Swing Low.")
    swing_type: SwingType = Field(..., description="HH, LH, EH, HL, LL, EL, HIGH, or LOW.")


class FairValueGap(BaseModel):
    """3-candle SMC/ICT Fair Value Gap (imbalance) record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    direction: SignalDirection = Field(..., description="BUY (Bullish FVG) or SELL (Bearish FVG).")
    upper_boundary: Decimal = Field(..., gt=Decimal("0"), description="Top price of the FVG zone.")
    lower_boundary: Decimal = Field(..., gt=Decimal("0"), description="Bottom price of the FVG zone.")
    formation_index: int = Field(..., ge=2, description="Bar index of the 3rd candle that completed the FVG.")
    formation_timestamp: datetime = Field(..., description="Timestamp when the 3rd candle closed.")
    gap_size: Decimal = Field(..., gt=Decimal("0"), description="upper_boundary - lower_boundary.")
    mitigated: bool = Field(default=False, description="True if subsequent price action fully filled the gap.")
    partially_filled: bool = Field(default=False, description="True if subsequent price entered the gap zone.")
    mitigation_timestamp: datetime | None = Field(default=None, description="Timestamp when fully mitigated.")
    age_bars: int = Field(default=0, ge=0, description="Number of bars elapsed since formation_index.")
    is_valid: bool = Field(default=True, description="True if unmitigated and within max_age_bars.")

    @model_validator(mode="after")
    def validate_boundaries(self) -> "FairValueGap":
        """Ensure upper_boundary > lower_boundary."""
        if self.upper_boundary <= self.lower_boundary:
            raise ValueError(
                f"FVG upper_boundary ({self.upper_boundary}) must be > lower_boundary ({self.lower_boundary})."
            )
        return self


class LiquidityPool(BaseModel):
    """Detected Equal Highs (EQH), Equal Lows (EQL), or swing liquidity pool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pool_type: str = Field(
        ...,
        description="'EQH' (Buy-Side Equal Highs), 'EQL' (Sell-Side Equal Lows), 'BSL' (Swing High), or 'SSL' (Swing Low).",
    )
    is_buy_side: bool = Field(..., description="True for highs (BSL/EQH), False for lows (SSL/EQL).")
    level_price: Decimal = Field(..., gt=Decimal("0"), description="Reference liquidity price level.")
    first_timestamp: datetime = Field(..., description="Timestamp of initial swing.")
    second_timestamp: datetime | None = Field(default=None, description="Timestamp of second touch for EQH/EQL.")
    swept: bool = Field(default=False, description="True if swept by subsequent price action.")
    sweep_timestamp: datetime | None = Field(default=None, description="Timestamp of the sweep candle.")
    rejected: bool = Field(default=False, description="True if candle swept level and closed back inside.")


class PremiumDiscountState(BaseModel):
    """Dealing range Premium / Discount / Equilibrium evaluation state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    range_high: Decimal | None = Field(default=None)
    range_low: Decimal | None = Field(default=None)
    equilibrium: Decimal | None = Field(default=None)
    current_price: Decimal | None = Field(default=None)
    position_in_range: Decimal | None = Field(
        default=None,
        description="Normalized position in [0, 1] (or <0 / >1 if outside dealing range).",
    )
    zone: PriceZoneClassification = Field(default=PriceZoneClassification.UNKNOWN)


class PreviousDayLevelsState(BaseModel):
    """Previous Day High/Low/Close (PDH/PDL/PDC) and current day interaction state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pdh: Decimal | None = Field(default=None, description="Previous trading day high.")
    pdl: Decimal | None = Field(default=None, description="Previous trading day low.")
    pdc: Decimal | None = Field(default=None, description="Previous trading day close.")
    current_day_high: Decimal | None = Field(default=None, description="Causal high of current trading day so far.")
    current_day_low: Decimal | None = Field(default=None, description="Causal low of current trading day so far.")
    distance_to_pdh: Decimal | None = Field(default=None, description="current_price - pdh.")
    distance_to_pdl: Decimal | None = Field(default=None, description="current_price - pdl.")
    pdh_swept_and_rejected: bool = Field(default=False, description="Bearish sweep of PDH with close below PDH.")
    pdl_swept_and_rejected: bool = Field(default=False, description="Bullish sweep of PDL with close above PDL.")
    pdh_broken_above: bool = Field(default=False, description="Close above PDH.")
    pdl_broken_below: bool = Field(default=False, description="Close below PDL.")


class EconomicEvent(BaseModel):
    """Structured economic calendar event supplied to NewsFilter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_name: str = Field(..., min_length=1, description="Name of the economic release (e.g., 'US Non-Farm Payrolls').")
    currency: str = Field(..., min_length=3, max_length=3, description="3-letter currency code (e.g., 'USD').")
    impact_level: NewsImpactLevel = Field(..., description="LOW, MEDIUM, or HIGH impact.")
    scheduled_time: datetime = Field(..., description="Scheduled release timestamp.")
    timezone: str = Field(default="UTC", description="IANA timezone name for scheduled_time.")
    actual_value: str | None = Field(default=None)
    forecast_value: str | None = Field(default=None)
    previous_value: str | None = Field(default=None)

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        """Normalize currency code to 3 uppercase letters."""
        cleaned = value.strip().upper()
        if len(cleaned) != 3 or not cleaned.isalpha():
            raise ValueError(f"Currency must be a 3-letter alphabetic code, got '{value}'.")
        return cleaned


class NewsFilterResult(BaseModel):
    """Result of evaluating scheduled economic events at a given timestamp."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: NewsFilterStatus = Field(..., description="CLEAR, RESTRICTED_MEDIUM, BLOCKED_HIGH, or UNKNOWN.")
    allow_new_entries: bool = Field(..., description="False if blocked by HIGH impact news or mandatory UNKNOWN status.")
    confidence_multiplier: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Multiplier applied to confluence confidence (e.g., 0.0 if blocked, 0.75 if restricted).",
    )
    active_events: tuple[EconomicEvent, ...] = Field(
        default_factory=tuple,
        description="Events falling within the configured blackout window around evaluation timestamp.",
    )
    rationale: str = Field(..., description="Explanation of news filter status.")


class TimeframeAnalysisSummary(BaseModel):
    """Summary of single-timeframe analysis within MultiTimeframeAnalyzer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timeframe: Timeframe
    last_closed_timestamp: datetime
    direction_bias: SignalDirection
    regime: MarketRegime
    structure_event: StructureEventType
    pd_zone: PriceZoneClassification
    has_fvg_retest: bool
    has_liquidity_sweep: bool
    confidence: Decimal
    rationale: str


class MultiTimeframeEvaluation(BaseModel):
    """Result of Multi-Timeframe (MTF) alignment evaluation across H1, M15, M5, and optional M1."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    aligned_direction: SignalDirection = Field(..., description="Overall MTF aligned direction (BUY, SELL, or WAIT).")
    alignment_score: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Normalized MTF alignment score in [0.0, 1.0].",
    )
    has_conflict: bool = Field(default=False, description="True if higher and middle/lower timeframes strongly disagree.")
    missing_timeframes: tuple[Timeframe, ...] = Field(
        default_factory=tuple,
        description="Configured required timeframes that had no closed candles available.",
    )
    summaries: dict[Timeframe, TimeframeAnalysisSummary] = Field(default_factory=dict)
    rationale: str = Field(..., description="Explanation of MTF alignment or conflict.")


class ConfluenceResult(BaseModel):
    """Structured output of ConfluenceEngine combining all enabled strategy and context modules."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    direction: SignalDirection = Field(..., description="Final aggregated decision: BUY, SELL, or WAIT.")
    buy_score: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), le=Decimal("1"))
    sell_score: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), le=Decimal("1"))
    wait_weight_ratio: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), le=Decimal("1"))
    net_directional_edge: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), le=Decimal("1"))
    final_confidence: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Deterministic weighted confluence score in [0.0, 1.0] (not a win probability).",
    )
    directional_agreement_count: int = Field(default=0, ge=0)
    conflicts: tuple[str, ...] = Field(default_factory=tuple)
    missing_evidence: tuple[str, ...] = Field(default_factory=tuple)
    rejection_reasons: tuple[str, ...] = Field(default_factory=tuple)
    component_evaluations: dict[str, StrategyEvaluationResult] = Field(default_factory=dict)
    news_result: NewsFilterResult | None = Field(default=None)
    rationale: str = Field(..., description="Comprehensive rationale of confluence scoring.")


class LifecycleEvent(BaseModel):
    """Single auditable state transition event for a managed signal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    from_state: SignalLifecycleState | None
    to_state: SignalLifecycleState
    timestamp: datetime
    reason: str


class ManagedSignalRecord(BaseModel):
    """Full lifecycle record for a generated signal managed by SignalLifecycleManager."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    signal_id: str = Field(..., min_length=1, description="Unique deterministic or UUID identifier.")
    setup_key: str = Field(..., min_length=1, description="Deduplication key (symbol:timeframe:direction:entry_zone).")
    created_at: datetime = Field(..., description="Timestamp when the signal was created.")
    expires_at: datetime | None = Field(default=None, description="Optional expiration timestamp.")
    signal: TradeSignal = Field(..., description="Canonical TradeSignal domain object.")
    state: SignalLifecycleState = Field(..., description="Current lifecycle state.")
    risk_status: RiskEvaluationStatus = Field(default=RiskEvaluationStatus.NOT_EVALUATED)
    risk_decision: RiskDecision | None = Field(default=None)
    invalidation_price: Decimal | None = Field(default=None, description="Price level that invalidates the setup before entry.")
    supporting_evidence: dict[str, object] = Field(default_factory=dict)
    history: tuple[LifecycleEvent, ...] = Field(default_factory=tuple)


class StrategyAnalysisReport(BaseModel):
    """Top-level structured result returned by StrategyEngine.analyze()."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    timeframe: Timeframe
    timestamp: datetime
    regime: MarketRegime
    regime_evaluation: StrategyEvaluationResult
    mtf_evaluation: MultiTimeframeEvaluation | None = None
    confluence: ConfluenceResult
    signal: TradeSignal
    lifecycle_record: ManagedSignalRecord | None = None
    risk_status: RiskEvaluationStatus = Field(default=RiskEvaluationStatus.NOT_EVALUATED)
    risk_decision: RiskDecision | None = Field(default=None)
