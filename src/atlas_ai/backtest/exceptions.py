"""Exceptions for the ATLAS AI Phase 4 Historical Backtesting Engine."""

from atlas_ai.core.exceptions import AtlasError


class BacktestError(AtlasError):
    """Base exception for all historical backtesting engine errors."""


class BacktestConfigurationError(BacktestError):
    """Raised when backtest configuration or instrument parameters are invalid."""


class BacktestExecutionError(BacktestError):
    """Raised when trade simulation, fill calculation, or portfolio accounting encounters an invalid state."""
