"""Custom exception hierarchy for ATLAS AI."""


class AtlasError(Exception):
    """Base exception for all ATLAS AI operational errors."""


class ConfigurationError(AtlasError):
    """Raised when application or environment configuration is invalid."""


class TradingSafetyError(AtlasError):
    """Raised when an operation violates deterministic trading safety gates."""


class DomainValidationError(AtlasError):
    """Raised when domain models violate business invariants."""


class RiskValidationError(TradingSafetyError):
    """Raised when risk calculation encounters missing or invalid parameters."""

