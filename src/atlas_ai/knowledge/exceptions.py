"""Exceptions for the ATLAS AI Phase 5 Educational Knowledge Ingestion & Rule Extraction Engine."""

from atlas_ai.core.exceptions import AtlasError, TradingSafetyError


class KnowledgeError(AtlasError):
    """Base exception for all educational knowledge ingestion, extraction, and storage errors."""


class KnowledgeIngestionError(KnowledgeError):
    """Raised when an educational document, transcript, or JSON playbook cannot be parsed."""


class KnowledgeExtractionError(KnowledgeError):
    """Raised when rule extraction fails or receives malformed extractor responses."""


class KnowledgeValidationError(KnowledgeError):
    """Raised when an extracted trading rule violates schema, provenance, or feature whitelist invariants."""


class KnowledgeSafetyError(TradingSafetyError, KnowledgeError):
    """Raised when an educational rule attempts to bypass RiskEngine, omit stop-losses,
    execute arbitrary code, reference future data, or enable live trading.
    """


class KnowledgeRepositoryError(KnowledgeError):
    """Raised when saving, loading, querying, or transitioning rules in KnowledgeRepository fails."""
