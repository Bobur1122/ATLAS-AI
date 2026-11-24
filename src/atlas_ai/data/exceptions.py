"""Exceptions for the ATLAS AI market data ingestion, validation, and feature pipeline."""

from atlas_ai.core.exceptions import AtlasError


class MarketDataError(AtlasError):
    """Base exception for market data loading, validation, and feature calculation errors."""


class DataLoadError(MarketDataError):
    """Raised when a CSV or raw dataset cannot be parsed or lacks required columns."""


class DataValidationError(MarketDataError):
    """Raised when market data fails OHLCV or time-series integrity validation."""


class FeatureCalculationError(MarketDataError):
    """Raised when feature computation receives invalid inputs or configuration."""
