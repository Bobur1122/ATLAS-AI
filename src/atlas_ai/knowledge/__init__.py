"""Educational Knowledge Ingestion & Structured Rule Extraction package for ATLAS AI Phase 5."""

from atlas_ai.knowledge.config import DEFAULT_WHITELISTED_FEATURES, KnowledgeConfig
from atlas_ai.knowledge.exceptions import (
    KnowledgeError,
    KnowledgeExtractionError,
    KnowledgeIngestionError,
    KnowledgeRepositoryError,
    KnowledgeSafetyError,
    KnowledgeValidationError,
)
from atlas_ai.knowledge.extractor import (
    BaseRuleExtractor,
    DeterministicRuleExtractor,
    LLMExtractorAdapter,
)
from atlas_ai.knowledge.loaders import KnowledgeDocumentLoader, TextChunker
from atlas_ai.knowledge.models import (
    ComparisonOperator,
    ConditionType,
    ExtractedTradingRule,
    KnowledgeCategory,
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeQuery,
    KnowledgeRuleCondition,
    KnowledgeSourceType,
    RuleApprovalStatus,
    compute_sha256_hex,
)
from atlas_ai.knowledge.repository import KnowledgeRepository
from atlas_ai.knowledge.strategy import KnowledgeBasedStrategy
from atlas_ai.knowledge.validator import KnowledgeRuleValidator

__all__ = [
    "BaseRuleExtractor",
    "ComparisonOperator",
    "ConditionType",
    "DEFAULT_WHITELISTED_FEATURES",
    "DeterministicRuleExtractor",
    "ExtractedTradingRule",
    "KnowledgeBasedStrategy",
    "KnowledgeCategory",
    "KnowledgeChunk",
    "KnowledgeConfig",
    "KnowledgeDocument",
    "KnowledgeDocumentLoader",
    "KnowledgeError",
    "KnowledgeExtractionError",
    "KnowledgeIngestionError",
    "KnowledgeQuery",
    "KnowledgeRepository",
    "KnowledgeRepositoryError",
    "KnowledgeRuleCondition",
    "KnowledgeRuleValidator",
    "KnowledgeSafetyError",
    "KnowledgeSourceType",
    "KnowledgeValidationError",
    "LLMExtractorAdapter",
    "RuleApprovalStatus",
    "TextChunker",
    "compute_sha256_hex",
]
