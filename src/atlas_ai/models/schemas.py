"""Immutable Pydantic schemas for ATLAS AI Phase 6 Machine Learning & Statistical Intelligence."""

from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from atlas_ai.domain.signals import SignalDirection
from atlas_ai.models.config import WalkForwardConfig


class ModelType(StrEnum):
    """Supported classical statistical and machine learning model architectures."""

    REGULARIZED_LOGISTIC = "REGULARIZED_LOGISTIC"
    GRADIENT_BOOSTED_STUMPS = "GRADIENT_BOOSTED_STUMPS"


def direction_to_class_index(direction: SignalDirection) -> int:
    """Map SignalDirection to 0-based multiclass index: SELL=0, WAIT=1, BUY=2."""
    if direction == SignalDirection.SELL:
        return 0
    if direction == SignalDirection.WAIT:
        return 1
    return 2


def class_index_to_direction(index: int) -> SignalDirection:
    """Map 0-based multiclass index (0=SELL, 1=WAIT, 2=BUY) to SignalDirection."""
    if index == 0:
        return SignalDirection.SELL
    if index == 1:
        return SignalDirection.WAIT
    if index == 2:
        return SignalDirection.BUY
    raise ValueError(f"Invalid class index {index}; expected 0 (SELL), 1 (WAIT), or 2 (BUY).")


class TripleBarrierLabel(BaseModel):
    """Immutable record of a leakage-safe Triple-Barrier label at bar `bar_index`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bar_index: int = Field(..., ge=0)
    timestamp: datetime
    label: SignalDirection = Field(..., description="BUY, SELL, or WAIT.")
    class_index: int = Field(..., ge=0, le=2, description="0=SELL, 1=WAIT, 2=BUY.")
    entry_price: Decimal = Field(..., gt=Decimal("0"))
    stop_distance: Decimal = Field(..., gt=Decimal("0"))
    target_distance: Decimal = Field(..., gt=Decimal("0"))
    long_tp: Decimal = Field(..., gt=Decimal("0"))
    long_sl: Decimal = Field(..., gt=Decimal("0"))
    short_tp: Decimal = Field(..., gt=Decimal("0"))
    short_sl: Decimal = Field(..., gt=Decimal("0"))
    long_succeeded: bool
    short_succeeded: bool
    long_resolution_bar: int | None = Field(default=None)
    short_resolution_bar: int | None = Field(default=None)
    max_future_bar_inspected: int = Field(
        ...,
        ge=0,
        description="Highest bar index inspected to compute this label (guaranteed <= bar_index + horizon_bars).",
    )
    is_complete_horizon: bool = Field(
        ...,
        description="True only if all H future bars were available in the dataset.",
    )
    rationale: str


class ClassClassificationMetrics(BaseModel):
    """Precision, recall, F1, and support for an individual SignalDirection class (BUY, SELL, or WAIT)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    direction: SignalDirection
    support: int = Field(..., ge=0, description="Actual ground-truth count of this class.")
    predicted_count: int = Field(..., ge=0, description="Number of times the model predicted this class.")
    true_positives: int = Field(..., ge=0)
    false_positives: int = Field(..., ge=0)
    false_negatives: int = Field(..., ge=0)
    precision: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    recall: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    f1_score: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))


class PredictionResult(BaseModel):
    """Immutable single-bar probabilistic prediction emitted by a trained Phase 6 classifier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(..., min_length=1)
    model_type: ModelType
    timestamp: datetime
    predicted_direction: SignalDirection
    buy_probability: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    sell_probability: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    wait_probability: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    confidence: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    directional_margin: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="|buy_probability - sell_probability|.",
    )
    top_features: dict[str, Decimal] = Field(default_factory=dict)


class ModelEvaluationReport(BaseModel):
    """Comprehensive classification, probability calibration, and trading utility report for a split."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    split_name: str = Field(..., min_length=1)
    total_samples: int = Field(..., ge=0)
    class_distribution: dict[str, int] = Field(
        ...,
        description="Ground-truth sample counts for BUY, SELL, and WAIT.",
    )
    class_imbalance_ratio: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Ratio of majority class count to non-zero minority class count.",
    )
    buy_metrics: ClassClassificationMetrics
    sell_metrics: ClassClassificationMetrics
    wait_metrics: ClassClassificationMetrics
    macro_f1: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    active_directional_signals: int = Field(..., ge=0)
    active_directional_precision: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    multiclass_brier_score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Mean squared probability error across BUY, SELL, WAIT (lower is better calibrated).",
    )
    mean_winning_probability: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    mean_losing_probability: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    calibration_error_ece: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Expected Calibration Error (ECE) across probability bins.",
    )
    expected_r_per_signal: Decimal = Field(
        ...,
        description="Realized R-multiple expectancy across active BUY/SELL signals based on triple-barrier outcomes.",
    )
    simulated_profit_factor: Decimal = Field(..., ge=Decimal("0"))
    simulated_max_drawdown_r: Decimal = Field(..., ge=Decimal("0"))
    simulated_sharpe_ratio: Decimal


class ChronologicalSplitIndices(BaseModel):
    """Verified, purge-separated chronological index boundaries for train, validation, and test sets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    train_start: int = Field(..., ge=0)
    train_end: int = Field(..., ge=0, description="Inclusive end index of training set.")
    purge_1_start: int = Field(..., ge=0)
    purge_1_end: int = Field(..., ge=0)
    val_start: int = Field(..., ge=0)
    val_end: int = Field(..., ge=0, description="Inclusive end index of validation set.")
    purge_2_start: int = Field(..., ge=0)
    purge_2_end: int = Field(..., ge=0)
    test_start: int = Field(..., ge=0)
    test_end: int = Field(..., ge=0, description="Inclusive end index of out-of-sample test set.")
    horizon_bars: int = Field(..., ge=1)
    purge_bars: int = Field(..., ge=1)


class WalkForwardFoldResult(BaseModel):
    """Out-of-sample evaluation result for a single chronological walk-forward fold."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fold_index: int = Field(..., ge=0)
    train_start_index: int = Field(..., ge=0)
    train_end_index: int = Field(..., ge=0)
    purge_start_index: int = Field(..., ge=0)
    purge_end_index: int = Field(..., ge=0)
    test_start_index: int = Field(..., ge=0)
    test_end_index: int = Field(..., ge=0)
    model_id: str
    evaluation: ModelEvaluationReport


class WalkForwardReport(BaseModel):
    """Aggregated walk-forward validation report across all chronological folds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_type: ModelType
    config: WalkForwardConfig
    folds: tuple[WalkForwardFoldResult, ...]
    aggregated_out_of_sample: ModelEvaluationReport


class ModelArtifactMetadata(BaseModel):
    """Metadata header embedded inside deterministic JSON model artifacts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(..., min_length=1)
    model_type: ModelType
    feature_names: tuple[str, ...]
    feature_schema_hash: str = Field(..., min_length=64, max_length=64)
    train_samples: int = Field(..., ge=1)
    horizon_bars: int = Field(..., ge=1)
    random_seed: int
    standardize_features: bool
