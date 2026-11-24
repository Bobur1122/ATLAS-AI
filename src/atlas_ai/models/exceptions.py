"""Exceptions for the ATLAS AI Phase 6 Machine Learning & Statistical Intelligence Engine."""

from atlas_ai.core.exceptions import AtlasError, TradingSafetyError


class ModelError(AtlasError):
    """Base exception for all Phase 6 machine learning and statistical modeling errors."""


class ModelConfigurationError(ModelError):
    """Raised when labeling, feature, training, or walk-forward configuration is invalid."""


class ModelDataLeakageError(TradingSafetyError, ModelError):
    """Raised when a train/validation/test split, feature slice, or label window violates strict
    chronological ordering or purge/embargo anti-leakage guarantees.
    """


class ModelTrainingError(ModelError):
    """Raised when model fitting fails due to empty datasets, NaN matrices, or invalid targets."""


class ModelPredictionError(ModelError):
    """Raised when an untrained model is queried or input features do not match the artifact schema."""
