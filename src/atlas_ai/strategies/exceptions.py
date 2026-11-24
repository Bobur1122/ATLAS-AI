"""Exceptions for the ATLAS AI Phase 3 Strategy, SMC/ICT, Confluence, and Signal Lifecycle Engine."""

from atlas_ai.core.exceptions import AtlasError


class StrategyError(AtlasError):
    """Base exception for all strategy engine and signal generation errors."""


class StrategyConfigurationError(StrategyError):
    """Raised when strategy, confluence, session, or news filter configuration is invalid."""


class StrategyEvaluationError(StrategyError):
    """Raised when a strategy or analyzer encounters invalid or unsorted input data."""


class SignalLifecycleError(StrategyError):
    """Raised when an invalid state transition or duplicate active setup occurs in SignalLifecycleManager."""
