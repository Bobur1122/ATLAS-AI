"""Configuration schemas for ATLAS AI Phase 6 Machine Learning & Statistical Intelligence."""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atlas_ai.models.exceptions import ModelDataLeakageError


class LabelingConfig(BaseModel):
    """Configuration for deterministic Triple-Barrier labeling."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    horizon_bars: int = Field(
        default=12,
        ge=1,
        le=500,
        description="Maximum forward holding horizon H (in bars) inspected for SL/TP barrier touches.",
    )
    sl_atr_multiplier: Decimal = Field(
        default=Decimal("1.50"),
        gt=Decimal("0"),
        le=Decimal("10.0"),
        description="Stop-loss barrier distance as a multiple of causal ATR_14 at bar t.",
    )
    rr_ratio: Decimal = Field(
        default=Decimal("2.00"),
        ge=Decimal("1.0"),
        le=Decimal("20.0"),
        description="Take-profit barrier distance to stop-loss barrier distance ratio.",
    )
    atr_column: str = Field(
        default="atr_14",
        min_length=1,
        description="Causal ATR column name from FeaturePipeline.",
    )


class MLFeatureConfig(BaseModel):
    """Configuration for stationary, scale-invariant ML feature construction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    return_lags: tuple[int, ...] = Field(
        default=(1, 3, 5, 10),
        description="Causal backward lag periods (in bars) for percentage and ATR-normalized returns.",
    )
    volume_window: int = Field(
        default=20,
        ge=2,
        le=500,
        description="Causal rolling window for relative volume normalization.",
    )
    atr_ratio_window: int = Field(
        default=20,
        ge=2,
        le=500,
        description="Causal rolling window for relative volatility regime ratio.",
    )
    standardize_features: bool = Field(
        default=True,
        description="Fit z-score standardization (mean/std) strictly on each training fold.",
    )
    clip_zscore: float = Field(
        default=5.0,
        gt=0.0,
        le=20.0,
        description="Symmetric z-score clipping bound to limit extreme outlier leverage.",
    )

    @field_validator("return_lags")
    @classmethod
    def validate_lags(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if not values:
            raise ValueError("return_lags must contain at least one positive lag.")
        for lag in values:
            if lag < 1:
                raise ValueError(f"All return_lags must be >= 1 (causal backward lag), got {lag}.")
        return tuple(sorted(set(values)))


class ModelTrainingConfig(BaseModel):
    """Hyperparameters for deterministic classical statistical and GBDT classifiers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    random_seed: int = Field(
        default=42,
        description="Fixed deterministic random seed for reproducibility.",
    )
    learning_rate: float = Field(
        default=0.15,
        gt=0.0,
        le=5.0,
        description="Gradient step size / shrinkage rate.",
    )
    l2_regularization: float = Field(
        default=0.02,
        ge=0.0,
        le=100.0,
        description="L2 ridge penalty weight for RegularizedLogisticClassifier and leaf weights.",
    )
    max_iterations: int = Field(
        default=250,
        ge=10,
        le=5000,
        description="Number of deterministic gradient descent optimization steps for logistic classifier.",
    )
    n_estimators: int = Field(
        default=35,
        ge=1,
        le=500,
        description="Number of boosting stages for GradientBoostedStumpEnsemble.",
    )
    n_thresholds_per_feature: int = Field(
        default=9,
        ge=3,
        le=50,
        description="Number of quantile split candidates evaluated per feature in decision stumps.",
    )
    class_weight_balanced: bool = Field(
        default=True,
        description="Reweight training samples inversely to training-fold class frequencies.",
    )


class WalkForwardConfig(BaseModel):
    """Configuration for leakage-safe chronological splitting and walk-forward validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    train_window_bars: int = Field(
        default=120,
        ge=20,
        description="Initial (or rolling) training window size in labeled bars.",
    )
    validation_window_bars: int = Field(
        default=40,
        ge=5,
        description="Validation window size in bars (used for hyperparameter/model selection, separate from final OOS test).",
    )
    test_window_bars: int = Field(
        default=40,
        ge=5,
        description="Out-of-sample test window size per walk-forward fold.",
    )
    step_bars: int = Field(
        default=40,
        ge=5,
        description="Chronological step size advancing the walk-forward window across folds.",
    )
    purge_bars: int = Field(
        default=12,
        ge=1,
        description="Mandatory purge/embargo gap (in bars) dropped between train, validation, and test splits.",
    )
    expanding_window: bool = Field(
        default=True,
        description="If True, training window expands from bar 0; if False, uses a rolling window of train_window_bars.",
    )
    probability_threshold: Decimal = Field(
        default=Decimal("0.45"),
        gt=Decimal("0"),
        le=Decimal("1.0"),
        description="Minimum class probability required to emit an active BUY or SELL prediction during evaluation.",
    )
    directional_margin: Decimal = Field(
        default=Decimal("0.10"),
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Minimum |P(BUY) - P(SELL)| required to emit an active BUY or SELL prediction.",
    )
    labeling: LabelingConfig = Field(default_factory=LabelingConfig)
    feature_config: MLFeatureConfig = Field(default_factory=MLFeatureConfig)
    training_config: ModelTrainingConfig = Field(default_factory=ModelTrainingConfig)

    @model_validator(mode="after")
    def enforce_purge_horizon_invariant(self) -> "WalkForwardConfig":
        """CRITICAL ANTI-LEAKAGE RULE: purge_bars must be >= labeling.horizon_bars."""
        if self.purge_bars < self.labeling.horizon_bars:
            raise ModelDataLeakageError(
                f"LEAKAGE VIOLATION: purge_bars ({self.purge_bars}) must be >= "
                f"labeling.horizon_bars ({self.labeling.horizon_bars}) to prevent train/test label overlap."
            )
        return self


class MLStrategyConfig(BaseModel):
    """Configuration for MLSignalStrategy(BaseStrategy)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_probability_threshold: Decimal = Field(
        default=Decimal("0.48"),
        gt=Decimal("0"),
        le=Decimal("1.0"),
        description="Minimum predicted BUY or SELL probability required to emit a directional signal.",
    )
    min_directional_margin: Decimal = Field(
        default=Decimal("0.12"),
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Minimum |P(BUY) - P(SELL)| probability edge required.",
    )
    sl_atr_multiplier: Decimal = Field(
        default=Decimal("1.50"),
        gt=Decimal("0"),
        le=Decimal("5.0"),
        description="Stop-loss distance in causal ATR_14 multiples.",
    )
    rr_ratio: Decimal = Field(
        default=Decimal("2.00"),
        ge=Decimal("1.0"),
        le=Decimal("10.0"),
        description="Take-profit reward-to-risk ratio.",
    )
