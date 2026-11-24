"""Machine Learning & Statistical Intelligence package for ATLAS AI Phase 6."""

from atlas_ai.models.classifiers import (
    BaseMLClassifier,
    DecisionStump,
    GradientBoostedStumpEnsemble,
    RegularizedLogisticClassifier,
    load_model_artifact,
)
from atlas_ai.models.config import (
    LabelingConfig,
    MLFeatureConfig,
    MLStrategyConfig,
    ModelTrainingConfig,
    WalkForwardConfig,
)
from atlas_ai.models.exceptions import (
    ModelConfigurationError,
    ModelDataLeakageError,
    ModelError,
    ModelPredictionError,
    ModelTrainingError,
)
from atlas_ai.models.features import FoldStandardScaler, MLFeatureBuilder
from atlas_ai.models.labeling import TripleBarrierLabeler
from atlas_ai.models.schemas import (
    ChronologicalSplitIndices,
    ClassClassificationMetrics,
    ModelArtifactMetadata,
    ModelEvaluationReport,
    ModelType,
    PredictionResult,
    TripleBarrierLabel,
    WalkForwardFoldResult,
    WalkForwardReport,
    class_index_to_direction,
    direction_to_class_index,
)
from atlas_ai.models.strategy import MLSignalStrategy
from atlas_ai.models.validation import (
    WalkForwardValidator,
    evaluate_predictions,
    validate_chronological_split,
)

__all__ = [
    "BaseMLClassifier",
    "ChronologicalSplitIndices",
    "ClassClassificationMetrics",
    "DecisionStump",
    "FoldStandardScaler",
    "GradientBoostedStumpEnsemble",
    "LabelingConfig",
    "MLFeatureBuilder",
    "MLFeatureConfig",
    "MLSignalStrategy",
    "MLStrategyConfig",
    "ModelArtifactMetadata",
    "ModelConfigurationError",
    "ModelDataLeakageError",
    "ModelError",
    "ModelEvaluationReport",
    "ModelPredictionError",
    "ModelTrainingConfig",
    "ModelTrainingError",
    "ModelType",
    "PredictionResult",
    "RegularizedLogisticClassifier",
    "TripleBarrierLabel",
    "TripleBarrierLabeler",
    "WalkForwardConfig",
    "WalkForwardFoldResult",
    "WalkForwardReport",
    "WalkForwardValidator",
    "class_index_to_direction",
    "direction_to_class_index",
    "evaluate_predictions",
    "load_model_artifact",
    "validate_chronological_split",
]
