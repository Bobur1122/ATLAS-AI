"""Immutable Pydantic domain models for ATLAS AI Phase 5 Knowledge Ingestion & Rule Extraction."""

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
import hashlib

from pydantic import BaseModel, ConfigDict, Field, field_validator

from atlas_ai.domain.market import Timeframe
from atlas_ai.domain.signals import SignalDirection


class KnowledgeSourceType(StrEnum):
    """Supported educational document source formats."""

    TRANSCRIPT_SRT = "TRANSCRIPT_SRT"
    TRANSCRIPT_VTT = "TRANSCRIPT_VTT"
    TRANSCRIPT_TXT = "TRANSCRIPT_TXT"
    MARKDOWN_NOTES = "MARKDOWN_NOTES"
    TEXT_NOTES = "TEXT_NOTES"
    PLAYBOOK_JSON = "PLAYBOOK_JSON"
    PDF_TEXT_EXTRACT = "PDF_TEXT_EXTRACT"
    SCREENSHOT_ANNOTATION = "SCREENSHOT_ANNOTATION"


class KnowledgeCategory(StrEnum):
    """Classification categories for extracted trading knowledge rules."""

    SMC_ICT = "SMC_ICT"
    TREND_FOLLOWING = "TREND_FOLLOWING"
    LIQUIDITY = "LIQUIDITY"
    FAIR_VALUE_GAP = "FAIR_VALUE_GAP"
    SESSION_KILLZONE = "SESSION_KILLZONE"
    MULTI_TIMEFRAME = "MULTI_TIMEFRAME"
    RISK_GUIDELINE = "RISK_GUIDELINE"


class RuleApprovalStatus(StrEnum):
    """Lifecycle approval state of an extracted trading rule.

    CRITICAL: Only `APPROVED` rules may be evaluated by `KnowledgeBasedStrategy`.
    `DRAFT`, `VALIDATED`, and `REJECTED` rules can never generate executable signals.
    """

    DRAFT = "DRAFT"
    VALIDATED = "VALIDATED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class ConditionType(StrEnum):
    """Declarative condition types evaluated deterministically without eval() or exec()."""

    REQUIRED_REGIME = "REQUIRED_REGIME"
    REQUIRED_PRICE_ZONE = "REQUIRED_PRICE_ZONE"
    REQUIRED_STRUCTURE_EVENT = "REQUIRED_STRUCTURE_EVENT"
    REQUIRED_LIQUIDITY_SWEEP = "REQUIRED_LIQUIDITY_SWEEP"
    REQUIRED_FVG = "REQUIRED_FVG"
    REQUIRED_KILLZONE = "REQUIRED_KILLZONE"
    FEATURE_COMPARISON = "FEATURE_COMPARISON"


class ComparisonOperator(StrEnum):
    """Safe comparison operators for declarative condition evaluation."""

    GT = "GT"
    GE = "GE"
    LT = "LT"
    LE = "LE"
    EQ = "EQ"


def compute_sha256_hex(content: str) -> str:
    """Compute a deterministic SHA-256 hex digest of UTF-8 text."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class KnowledgeChunk(BaseModel):
    """Traceable text segment extracted from an educational KnowledgeDocument."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(..., min_length=1, description="Deterministic chunk identifier.")
    document_id: str = Field(..., min_length=1, description="Parent KnowledgeDocument identifier.")
    sequence_index: int = Field(..., ge=0, description="0-based chunk sequence index within the document.")
    text: str = Field(..., min_length=1, description="Sanitized educational text content of this chunk.")
    content_hash: str = Field(..., min_length=64, max_length=64, description="SHA-256 hex hash of chunk text.")
    section_title: str | None = Field(default=None, description="Markdown section heading or topic title.")
    start_timestamp: str | None = Field(
        default=None,
        description="Optional media transcript start timestamp (e.g., '00:01:15.000').",
    )
    end_timestamp: str | None = Field(
        default=None,
        description="Optional media transcript end timestamp (e.g., '00:02:05.000').",
    )
    symbol_tags: tuple[str, ...] = Field(default_factory=tuple)
    timeframe_tags: tuple[Timeframe, ...] = Field(default_factory=tuple)

    @field_validator("text")
    @classmethod
    def ensure_non_blank_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("KnowledgeChunk text cannot be empty or whitespace only.")
        return cleaned


class KnowledgeDocument(BaseModel):
    """Immutable representation of an ingested educational trading document or transcript."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(..., min_length=1, description="Deterministic or explicit document ID.")
    title: str = Field(..., min_length=1, description="Human-readable document or lesson title.")
    source_type: KnowledgeSourceType
    source_uri: str = Field(..., min_length=1, description="File path, lesson identifier, or URI.")
    author: str = Field(default="Unknown", min_length=1)
    ingested_at: datetime
    content_hash: str = Field(..., min_length=64, max_length=64, description="SHA-256 hash of normalized_text.")
    raw_char_count: int = Field(..., ge=1)
    normalized_text: str = Field(..., min_length=1)
    symbol_tags: tuple[str, ...] = Field(default=("XAUUSD",))
    timeframe_tags: tuple[Timeframe, ...] = Field(default=(Timeframe.H1,))
    chunks: tuple[KnowledgeChunk, ...] = Field(default_factory=tuple)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("document_id", "title", "source_uri", "author", "normalized_text")
    @classmethod
    def ensure_non_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Document string field cannot be empty or whitespace only.")
        return cleaned

    @field_validator("symbol_tags")
    @classmethod
    def normalize_symbols(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for sym in values:
            cleaned = sym.strip().upper()
            if not cleaned or not cleaned.isalnum():
                raise ValueError(f"Invalid symbol tag '{sym}'.")
            if cleaned not in normalized:
                normalized.append(cleaned)
        return tuple(normalized)


class KnowledgeRuleCondition(BaseModel):
    """Single declarative condition predicate within an ExtractedTradingRule."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    condition_type: ConditionType
    parameter: str = Field(
        ...,
        min_length=1,
        description=(
            "Target state or left-hand feature name (e.g., 'UPTREND', 'DISCOUNT', 'BULLISH_BOS', "
            "'SSL_OR_EQL_SWEEP', 'BULLISH_FVG_RETEST', 'LONDON_KILL_ZONE', or feature 'close')."
        ),
    )
    operator: ComparisonOperator = Field(default=ComparisonOperator.EQ)
    target_feature: str | None = Field(
        default=None,
        description="Optional right-hand whitelisted feature column name for FEATURE_COMPARISON (e.g., 'ema_21').",
    )
    threshold_value: Decimal | None = Field(
        default=None,
        description="Optional numeric threshold for FEATURE_COMPARISON (e.g., Decimal('0.15')).",
    )
    rationale: str = Field(default="", description="Human-readable explanation of this predicate.")

    @field_validator("parameter")
    @classmethod
    def strip_parameter(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Condition parameter cannot be empty.")
        return cleaned


class ExtractedTradingRule(BaseModel):
    """Immutable, provenance-tracked trading rule extracted from educational material."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: str = Field(..., min_length=1, description="Deterministic or explicit rule identifier.")
    version: int = Field(default=1, ge=1, description="Monotonically increasing rule version.")
    source_document_id: str = Field(..., min_length=1, description="Provenance link to KnowledgeDocument.")
    source_chunk_id: str = Field(..., min_length=1, description="Provenance link to KnowledgeChunk.")
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    category: KnowledgeCategory
    approval_status: RuleApprovalStatus = Field(
        default=RuleApprovalStatus.DRAFT,
        description="Rule approval status (defaults to DRAFT; never auto-approved on ingestion).",
    )
    target_symbols: tuple[str, ...] = Field(default=("XAUUSD",))
    applicable_timeframes: tuple[Timeframe, ...] = Field(default=(Timeframe.H1,))
    direction: SignalDirection = Field(
        ...,
        description="Target signal direction (BUY, SELL, or WAIT for non-executable educational notes).",
    )
    conditions: tuple[KnowledgeRuleCondition, ...] = Field(default_factory=tuple)
    stop_loss_atr_multiplier: Decimal = Field(
        default=Decimal("1.50"),
        description="Stop-loss distance in ATR_14 multiples.",
    )
    minimum_rr_ratio: Decimal = Field(
        default=Decimal("2.00"),
        description="Target Take-Profit to Stop-Loss reward-to-risk ratio.",
    )
    base_confidence: Decimal = Field(
        default=Decimal("0.75"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Deterministic rule base confidence score in [0.0, 1.0].",
    )
    extracted_at: datetime
    content_hash: str = Field(..., min_length=64, max_length=64)
    extraction_notes: tuple[str, ...] = Field(default_factory=tuple)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("rule_id", "source_document_id", "source_chunk_id", "title", "description")
    @classmethod
    def ensure_non_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("ExtractedTradingRule required string field cannot be blank.")
        return cleaned

    @field_validator("target_symbols")
    @classmethod
    def normalize_target_symbols(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("target_symbols must contain at least one symbol.")
        normalized: list[str] = []
        for sym in values:
            cleaned = sym.strip().upper()
            if not cleaned or not cleaned.isalnum():
                raise ValueError(f"Invalid target symbol '{sym}'.")
            if cleaned not in normalized:
                normalized.append(cleaned)
        return tuple(normalized)


class KnowledgeQuery(BaseModel):
    """Structured query parameters for filtering rules in KnowledgeRepository."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str | None = Field(default=None)
    timeframe: Timeframe | None = Field(default=None)
    category: KnowledgeCategory | None = Field(default=None)
    direction: SignalDirection | None = Field(default=None)
    approval_status: RuleApprovalStatus | None = Field(default=None)
    required_regime: str | None = Field(default=None)
    required_killzone: str | None = Field(default=None)

    @field_validator("symbol")
    @classmethod
    def normalize_optional_symbol(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().upper()
        return cleaned if cleaned else None
