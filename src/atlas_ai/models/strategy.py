"""Causal MLSignalStrategy(BaseStrategy) integrating classical ML models into StrategyEngine."""

from decimal import Decimal, ROUND_DOWN

import numpy as np
import pandas as pd

from atlas_ai.domain.signals import SignalDirection
from atlas_ai.models.classifiers import BaseMLClassifier
from atlas_ai.models.config import MLFeatureConfig, MLStrategyConfig
from atlas_ai.models.features import MLFeatureBuilder
from atlas_ai.strategies.base import BaseStrategy
from atlas_ai.strategies.models import StrategyEvaluationResult


class MLSignalStrategy(BaseStrategy):
    """Evaluates a trained Phase 6 classical ML classifier causally on market data.

    Key Safety Invariants:
    1. Strictly implements `BaseStrategy` and slices `df.iloc[: as_of_index + 1]` via `prepare_causal_slice()`.
    2. Never refits or peeks at future bars during `evaluate()`; uses only the frozen training-fold
       `FoldStandardScaler` and weights embedded in the fitted `classifier`.
    3. Emits a standard `StrategyEvaluationResult` (`BUY`, `SELL`, or `WAIT`) that flows through
       `ConfluenceEngine` -> `SignalGenerator` -> `SignalLifecycleManager` -> Phase 1 `RiskEngine`.
    4. Can NEVER bypass `RiskEngine` or directly execute orders.
    """

    def __init__(
        self,
        classifier: BaseMLClassifier,
        *,
        strategy_config: MLStrategyConfig | None = None,
        feature_config: MLFeatureConfig | None = None,
    ) -> None:
        self._classifier = classifier
        self._config = strategy_config or MLStrategyConfig()
        self._feature_builder = MLFeatureBuilder(feature_config)

    @property
    def strategy_id(self) -> str:
        return "ml_signal_strategy"

    @property
    def name(self) -> str:
        return "Classical Machine Learning Signal Strategy"

    @property
    def required_features(self) -> tuple[str, ...]:
        return ("timestamp", "open", "high", "low", "close", "volume")

    @property
    def classifier(self) -> BaseMLClassifier:
        return self._classifier

    @property
    def config(self) -> MLStrategyConfig:
        return self._config

    def evaluate(
        self,
        df: pd.DataFrame,
        as_of_index: int | None = None,
    ) -> StrategyEvaluationResult:
        """Deterministically evaluate the fitted ML model at `as_of_index` (causal slice only)."""
        sliced = self.prepare_causal_slice(df, as_of_index)
        last_raw_row = sliced.iloc[-1]
        ts = self.extract_timestamp(last_raw_row)

        if not self._classifier.is_fitted or self._classifier.metadata is None:
            return self.build_wait_result(
                timestamp=ts,
                rationale="MLSignalStrategy classifier is not fitted; returning WAIT.",
                rejection_reasons=["MODEL_NOT_FITTED"],
            )

        ml_df = self._feature_builder.build_features(sliced)
        last_ml_row = ml_df.iloc[-1]
        feat_names = self._feature_builder.feature_names

        # Check warm-up NaNs across all stationary ML features and atr_14
        x_row = last_ml_row[list(feat_names)].to_numpy(dtype="float64")
        raw_atr = last_ml_row.get("atr_14")
        if (
            not np.all(np.isfinite(x_row))
            or raw_atr is None
            or pd.isna(raw_atr)
            or float(raw_atr) <= 0.0
        ):
            return self.build_wait_result(
                timestamp=ts,
                rationale="Warm-up NaN in stationary ML features or ATR_14; returning WAIT.",
                rejection_reasons=["WARMUP_NAN"],
            )

        pred = self._classifier.predict_single(
            x_row,
            timestamp=ts,
            probability_threshold=self._config.min_probability_threshold,
            directional_margin=self._config.min_directional_margin,
            feature_names=feat_names,
        )

        evidence: dict[str, object] = {
            "model_id": pred.model_id,
            "model_type": pred.model_type.value,
            "buy_probability": str(pred.buy_probability),
            "sell_probability": str(pred.sell_probability),
            "wait_probability": str(pred.wait_probability),
            "directional_margin": str(pred.directional_margin),
            "top_features": {k: str(v) for k, v in pred.top_features.items()},
        }

        if pred.predicted_direction == SignalDirection.WAIT:
            max_dir_p = max(pred.buy_probability, pred.sell_probability)
            reasons: list[str] = []
            if max_dir_p < self._config.min_probability_threshold:
                reasons.append("ML_PROBABILITY_BELOW_THRESHOLD")
            if pred.directional_margin < self._config.min_directional_margin:
                reasons.append("ML_DIRECTIONAL_MARGIN_BELOW_THRESHOLD")
            if not reasons:
                reasons.append("ML_WAIT_DOMINANT")
            return self.build_wait_result(
                timestamp=ts,
                rationale=(
                    f"ML model '{pred.model_id}' ({pred.model_type.value}) predicted WAIT "
                    f"(P_BUY={pred.buy_probability}, P_SELL={pred.sell_probability}, "
                    f"P_WAIT={pred.wait_probability})."
                ),
                rejection_reasons=reasons,
                supporting_evidence=evidence,
            )

        close_price = self.to_decimal(last_ml_row["close"])
        atr_val = self.to_decimal(last_ml_row["atr_14"])
        sl_dist = max(
            Decimal("0.10"),
            (atr_val * self._config.sl_atr_multiplier).quantize(
                Decimal("0.01"), rounding=ROUND_DOWN
            ),
        )
        tp_dist = max(
            Decimal("0.10"),
            (sl_dist * self._config.rr_ratio).quantize(
                Decimal("0.01"), rounding=ROUND_DOWN
            ),
        )

        if pred.predicted_direction == SignalDirection.BUY:
            sl_price = close_price - sl_dist
            tp_price = close_price + tp_dist
        else:
            sl_price = close_price + sl_dist
            tp_price = close_price - tp_dist

        return StrategyEvaluationResult(
            strategy_id=self.strategy_id,
            direction=pred.predicted_direction,
            timestamp=ts,
            entry_price=close_price,
            stop_loss=sl_price,
            take_profit=tp_price,
            confidence=pred.confidence,
            rationale=(
                f"ML model '{pred.model_id}' ({pred.model_type.value}) emitted "
                f"{pred.predicted_direction.value} with probability={pred.confidence} "
                f"(margin={pred.directional_margin})."
            ),
            supporting_evidence=evidence,
            rejection_reasons=(),
        )
