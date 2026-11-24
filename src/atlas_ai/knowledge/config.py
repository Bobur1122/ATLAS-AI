"""Configuration schemas and feature whitelist for ATLAS AI Phase 5 Knowledge Engine."""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atlas_ai.domain.market import Timeframe

# Strict whitelist of causal feature columns produced by Phase 2 FeaturePipeline
DEFAULT_WHITELISTED_FEATURES: frozenset[str] = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "tr",
        "atr_14",
        "atr_pct_14",
        "sma_20",
        "sma_50",
        "sma_100",
        "sma_200",
        "ema_9",
        "ema_21",
        "ema_50",
        "candle_body",
        "upper_wick",
        "lower_wick",
        "candle_range",
        "body_to_range_ratio",
        "hour_of_day",
        "day_of_week",
        "rolling_high_20",
        "rolling_low_20",
        "prev_high",
        "prev_low",
        "session_high",
        "session_low",
    }
)


class KnowledgeConfig(BaseModel):
    """Validated configuration for educational knowledge ingestion, rule validation, and retrieval."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_chunk_chars: int = Field(
        default=1200,
        ge=100,
        le=20000,
        description="Maximum character length of an individual KnowledgeChunk.",
    )
    min_chunk_chars: int = Field(
        default=10,
        ge=1,
        le=5000,
        description="Minimum character length for a non-empty chunk.",
    )
    whitelisted_features: frozenset[str] = Field(
        default=DEFAULT_WHITELISTED_FEATURES,
        description="Set of causal DataFrame column names permitted in FEATURE_COMPARISON conditions.",
    )
    min_allowed_sl_atr_multiplier: Decimal = Field(
        default=Decimal("0.25"),
        gt=Decimal("0"),
        description="Minimum permitted stop-loss ATR multiplier for extracted rules.",
    )
    max_allowed_sl_atr_multiplier: Decimal = Field(
        default=Decimal("5.00"),
        gt=Decimal("0"),
        description="Maximum permitted stop-loss ATR multiplier for extracted rules.",
    )
    min_allowed_rr_ratio: Decimal = Field(
        default=Decimal("1.00"),
        ge=Decimal("1.0"),
        description="Minimum permitted reward-to-risk ratio for extracted rules.",
    )
    max_allowed_rr_ratio: Decimal = Field(
        default=Decimal("10.00"),
        ge=Decimal("1.0"),
        description="Maximum permitted reward-to-risk ratio for extracted rules.",
    )
    default_symbol: str = Field(
        default="XAUUSD",
        min_length=3,
        max_length=16,
        description="Default target instrument symbol when not explicitly tagged in a document.",
    )
    default_timeframe: Timeframe = Field(
        default=Timeframe.H1,
        description="Default applicable timeframe when not explicitly tagged in a document.",
    )

    @field_validator("default_symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        """Normalize default_symbol to uppercase alphanumeric characters."""
        cleaned = value.strip().upper()
        if not cleaned.isalnum():
            raise ValueError(f"default_symbol must be alphanumeric, got '{value}'.")
        return cleaned

    @model_validator(mode="after")
    def validate_bounds(self) -> "KnowledgeConfig":
        """Ensure chunk and risk multiplier bounds are well-ordered."""
        if self.min_chunk_chars > self.max_chunk_chars:
            raise ValueError("min_chunk_chars cannot exceed max_chunk_chars.")
        if self.min_allowed_sl_atr_multiplier > self.max_allowed_sl_atr_multiplier:
            raise ValueError(
                "min_allowed_sl_atr_multiplier cannot exceed max_allowed_sl_atr_multiplier."
            )
        if self.min_allowed_rr_ratio > self.max_allowed_rr_ratio:
            raise ValueError("min_allowed_rr_ratio cannot exceed max_allowed_rr_ratio.")
        return self
