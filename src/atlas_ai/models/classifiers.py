"""Deterministic Classical ML Classifiers (RegularizedLogisticClassifier & GradientBoostedStumpEnsemble)."""

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from atlas_ai.domain.signals import SignalDirection
from atlas_ai.knowledge.models import compute_sha256_hex
from atlas_ai.models.config import ModelTrainingConfig
from atlas_ai.models.exceptions import ModelPredictionError, ModelTrainingError
from atlas_ai.models.features import FoldStandardScaler
from atlas_ai.models.schemas import (
    ModelArtifactMetadata,
    ModelType,
    PredictionResult,
    class_index_to_direction,
)

QUANTIZE_PROB = Decimal("0.0001")
N_CLASSES = 3  # 0=SELL, 1=WAIT, 2=BUY


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax over a 2D array `(n_samples, n_classes)`."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(np.clip(shifted, -40.0, 40.0))
    sums = np.sum(exps, axis=1, keepdims=True)
    return exps / sums


def _compute_balanced_sample_weights(y: np.ndarray, enabled: bool) -> np.ndarray:
    """Compute deterministic inverse-frequency sample weights normalized to mean 1.0."""
    n = len(y)
    if not enabled or n == 0:
        return np.ones(n, dtype="float64")
    weights = np.ones(n, dtype="float64")
    present_classes = [c for c in range(N_CLASSES) if np.any(y == c)]
    n_present = len(present_classes)
    if n_present <= 1:
        return weights
    for c in present_classes:
        mask = y == c
        cnt = int(np.sum(mask))
        weights[mask] = float(n) / float(n_present * cnt)
    mean_w = float(np.mean(weights))
    return weights / mean_w if mean_w > 0 else weights


class BaseMLClassifier(ABC):
    """Abstract base class for deterministic Phase 6 statistical and ML classifiers."""

    def __init__(
        self,
        config: ModelTrainingConfig | None = None,
        *,
        standardize_features: bool = True,
        clip_zscore: float = 5.0,
    ) -> None:
        self._config = config or ModelTrainingConfig()
        self._standardize = standardize_features
        self._scaler = FoldStandardScaler(clip_zscore=clip_zscore)
        self._feature_names: tuple[str, ...] = ()
        self._metadata: ModelArtifactMetadata | None = None

    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        """Return the ModelType identifier."""

    @property
    def config(self) -> ModelTrainingConfig:
        return self._config

    @property
    def is_fitted(self) -> bool:
        return self._metadata is not None

    @property
    def metadata(self) -> ModelArtifactMetadata | None:
        return self._metadata

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    def _validate_and_prepare_train_inputs(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Sequence[str],
    ) -> np.ndarray:
        if x_train.ndim != 2 or x_train.shape[0] < 3:
            raise ModelTrainingError(
                f"{self.model_type.value}: x_train must be a 2D array with >= 3 samples, got {x_train.shape}."
            )
        if y_train.ndim != 1 or len(y_train) != x_train.shape[0]:
            raise ModelTrainingError(
                f"{self.model_type.value}: y_train length ({len(y_train)}) must match x_train rows ({x_train.shape[0]})."
            )
        if x_train.shape[1] != len(feature_names):
            raise ModelTrainingError(
                f"{self.model_type.value}: x_train columns ({x_train.shape[1]}) must match "
                f"feature_names count ({len(feature_names)})."
            )
        if not np.all(np.isfinite(x_train)):
            raise ModelTrainingError(f"{self.model_type.value}: x_train contains NaN or infinite values.")
        if not np.all(np.isin(y_train, [0, 1, 2])):
            raise ModelTrainingError(
                f"{self.model_type.value}: y_train must contain class indices in {{0 (SELL), 1 (WAIT), 2 (BUY)}}."
            )

        self._feature_names = tuple(str(f) for f in feature_names)
        if self._standardize:
            return self._scaler.fit_transform(x_train.astype("float64"))
        return x_train.astype("float64")

    def _prepare_inference_inputs(
        self,
        x: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise ModelPredictionError(f"{self.model_type.value} is not fitted yet.")
        if feature_names is not None and tuple(feature_names) != self._feature_names:
            raise ModelPredictionError(
                f"Feature schema mismatch: model expects {self._feature_names}, got {tuple(feature_names)}."
            )
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != len(self._feature_names):
            raise ModelPredictionError(
                f"Expected 2D array with {len(self._feature_names)} features, got {x.shape}."
            )
        if not np.all(np.isfinite(x)):
            raise ModelPredictionError("Inference input matrix contains NaN or infinite values.")
        if self._standardize:
            return self._scaler.transform(x.astype("float64"))
        return x.astype("float64")

    @abstractmethod
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Sequence[str],
        *,
        horizon_bars: int = 12,
    ) -> "BaseMLClassifier":
        """Fit the classifier deterministically on training fold `(x_train, y_train)`."""

    @abstractmethod
    def predict_proba_matrix(
        self,
        x: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Return calibrated class probabilities `(n_samples, 3)` for `[SELL, WAIT, BUY]`."""

    @abstractmethod
    def feature_importances(self) -> dict[str, Decimal]:
        """Return normalized feature importances summing to 1.0."""

    def predict_single(
        self,
        x_row: np.ndarray,
        timestamp: datetime,
        *,
        probability_threshold: Decimal = Decimal("0.45"),
        directional_margin: Decimal = Decimal("0.10"),
        feature_names: Sequence[str] | None = None,
    ) -> PredictionResult:
        """Emit a structured `PredictionResult` for a single feature row."""
        assert self._metadata is not None
        probs = self.predict_proba_matrix(x_row.reshape(1, -1), feature_names=feature_names)[0]

        sell_p = Decimal(str(round(float(probs[0]), 6))).quantize(QUANTIZE_PROB, rounding=ROUND_HALF_UP)
        wait_p = Decimal(str(round(float(probs[1]), 6))).quantize(QUANTIZE_PROB, rounding=ROUND_HALF_UP)
        buy_p = Decimal(str(round(float(probs[2]), 6))).quantize(QUANTIZE_PROB, rounding=ROUND_HALF_UP)

        # Ensure exact sum to 1.0000 after quantization
        residual = Decimal("1.0000") - (sell_p + wait_p + buy_p)
        wait_p = max(Decimal("0.0000"), min(Decimal("1.0000"), wait_p + residual))

        margin = abs(buy_p - sell_p).quantize(QUANTIZE_PROB, rounding=ROUND_HALF_UP)

        if buy_p >= sell_p and buy_p >= wait_p and buy_p >= probability_threshold and margin >= directional_margin:
            pred_dir = SignalDirection.BUY
            conf = buy_p
        elif sell_p > buy_p and sell_p >= wait_p and sell_p >= probability_threshold and margin >= directional_margin:
            pred_dir = SignalDirection.SELL
            conf = sell_p
        else:
            pred_dir = SignalDirection.WAIT
            conf = wait_p

        importances = self.feature_importances()
        top_feats = dict(
            sorted(importances.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:5]
        )

        return PredictionResult(
            model_id=self._metadata.model_id,
            model_type=self.model_type,
            timestamp=timestamp,
            predicted_direction=pred_dir,
            buy_probability=buy_p,
            sell_probability=sell_p,
            wait_probability=wait_p,
            confidence=conf,
            directional_margin=margin,
            top_features=top_feats,
        )

    @abstractmethod
    def to_artifact_dict(self) -> dict[str, object]:
        """Serialize model parameters and metadata to a JSON-compatible dictionary."""

    def save_artifact(self, file_path: str | Path) -> Path:
        """Persist the fitted model artifact to a deterministic JSON file."""
        if not self.is_fitted:
            raise ModelPredictionError("Cannot save an unfitted model artifact.")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_artifact_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path


# ---------------------------------------------------------------------------
# 1. First Classical Model: RegularizedLogisticClassifier (L2 Multinomial Softmax)
# ---------------------------------------------------------------------------
class RegularizedLogisticClassifier(BaseMLClassifier):
    """Deterministic L2-regularized multinomial softmax logistic classifier.

    Trained via full-batch deterministic gradient descent with inverse-frequency
    class weighting and train-fold-only z-score standardization.
    """

    def __init__(
        self,
        config: ModelTrainingConfig | None = None,
        *,
        standardize_features: bool = True,
        clip_zscore: float = 5.0,
    ) -> None:
        super().__init__(
            config=config,
            standardize_features=standardize_features,
            clip_zscore=clip_zscore,
        )
        self._weights: np.ndarray | None = None  # shape: (n_features, 3)
        self._bias: np.ndarray | None = None     # shape: (3,)

    @property
    def model_type(self) -> ModelType:
        return ModelType.REGULARIZED_LOGISTIC

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Sequence[str],
        *,
        horizon_bars: int = 12,
    ) -> "RegularizedLogisticClassifier":
        x_prep = self._validate_and_prepare_train_inputs(x_train, y_train, feature_names)
        n_samples, n_features = x_prep.shape

        # One-hot target matrix Y: shape (n_samples, 3)
        y_onehot = np.zeros((n_samples, N_CLASSES), dtype="float64")
        for idx, cls_idx in enumerate(y_train.astype(int)):
            y_onehot[idx, cls_idx] = 1.0

        sample_weights = _compute_balanced_sample_weights(
            y_train.astype(int), self._config.class_weight_balanced
        )
        sw_col = sample_weights.reshape(-1, 1)

        # Deterministic zero initialization (convex L2 objective -> unique global minimum)
        w = np.zeros((n_features, N_CLASSES), dtype="float64")
        b = np.zeros(N_CLASSES, dtype="float64")

        lr = self._config.learning_rate
        l2 = self._config.l2_regularization

        for _ in range(self._config.max_iterations):
            logits = x_prep @ w + b
            probs = _softmax(logits)
            diff = (probs - y_onehot) * sw_col
            grad_w = (x_prep.T @ diff) / float(n_samples) + l2 * w
            grad_b = np.sum(diff, axis=0) / float(n_samples)
            w -= lr * grad_w
            b -= lr * grad_b

        self._weights = w
        self._bias = b

        schema_hash = compute_sha256_hex(",".join(self._feature_names))
        param_sig = compute_sha256_hex(
            f"{schema_hash}:{n_samples}:{self._config.random_seed}:{np.round(w, 6).tolist()}"
        )
        self._metadata = ModelArtifactMetadata(
            model_id=f"logreg-{param_sig[:12]}",
            model_type=self.model_type,
            feature_names=self._feature_names,
            feature_schema_hash=schema_hash,
            train_samples=n_samples,
            horizon_bars=horizon_bars,
            random_seed=self._config.random_seed,
            standardize_features=self._standardize,
        )
        return self

    def predict_proba_matrix(
        self,
        x: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        x_prep = self._prepare_inference_inputs(x, feature_names=feature_names)
        assert self._weights is not None and self._bias is not None
        logits = x_prep @ self._weights + self._bias
        return _softmax(logits)

    def feature_importances(self) -> dict[str, Decimal]:
        if not self.is_fitted or self._weights is None:
            raise ModelPredictionError("RegularizedLogisticClassifier is not fitted.")
        # L1 norm across the 3 class weight vectors per feature
        raw_imp = np.sum(np.abs(self._weights), axis=1)
        total = float(np.sum(raw_imp))
        result: dict[str, Decimal] = {}
        for idx, f_name in enumerate(self._feature_names):
            val = float(raw_imp[idx]) / total if total > 1e-12 else 1.0 / len(self._feature_names)
            result[f_name] = Decimal(str(round(val, 4))).quantize(
                QUANTIZE_PROB, rounding=ROUND_HALF_UP
            )
        return result

    def to_artifact_dict(self) -> dict[str, object]:
        if not self.is_fitted or self._metadata is None or self._weights is None or self._bias is None:
            raise ModelPredictionError("RegularizedLogisticClassifier is not fitted.")
        return {
            "metadata": self._metadata.model_dump(mode="json"),
            "training_config": self._config.model_dump(mode="json"),
            "scaler": self._scaler.to_dict(),
            "weights": self._weights.tolist(),
            "bias": self._bias.tolist(),
        }

    @classmethod
    def from_artifact_dict(cls, payload: dict[str, object]) -> "RegularizedLogisticClassifier":
        meta = ModelArtifactMetadata.model_validate(payload["metadata"])
        t_cfg = ModelTrainingConfig.model_validate(payload["training_config"])
        scaler_dict = payload["scaler"]
        assert isinstance(scaler_dict, dict)

        model = cls(
            config=t_cfg,
            standardize_features=meta.standardize_features,
        )
        model._scaler = FoldStandardScaler.from_dict(scaler_dict)
        model._feature_names = meta.feature_names
        model._metadata = meta
        model._weights = np.array(payload["weights"], dtype="float64")
        model._bias = np.array(payload["bias"], dtype="float64")
        return model

    @classmethod
    def load_artifact(cls, file_path: str | Path) -> "RegularizedLogisticClassifier":
        path = Path(file_path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_artifact_dict(raw)


# ---------------------------------------------------------------------------
# 2. Second Classical Model: GradientBoostedStumpEnsemble (Pure-NumPy GBDT)
# ---------------------------------------------------------------------------
class DecisionStump:
    """Single depth-1 regression tree stump fitted to Newton second-order gradient steps."""

    def __init__(
        self,
        *,
        feature_index: int,
        threshold: float,
        left_value: float,
        right_value: float,
        gain: float,
    ) -> None:
        self.feature_index = feature_index
        self.threshold = threshold
        self.left_value = left_value
        self.right_value = right_value
        self.gain = gain

    def predict(self, x: np.ndarray) -> np.ndarray:
        col = x[:, self.feature_index]
        return np.where(col <= self.threshold, self.left_value, self.right_value)

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_index": self.feature_index,
            "threshold": self.threshold,
            "left_value": self.left_value,
            "right_value": self.right_value,
            "gain": self.gain,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "DecisionStump":
        return cls(
            feature_index=int(str(data["feature_index"])),
            threshold=float(str(data["threshold"])),
            left_value=float(str(data["left_value"])),
            right_value=float(str(data["right_value"])),
            gain=float(str(data["gain"])),
        )


class GradientBoostedStumpEnsemble(BaseMLClassifier):
    """Deterministic multi-class Gradient Boosted Decision Stump Ensemble (GBDT).

    Uses second-order Newton-Raphson boosting updates (identical to XGBoost/LightGBM math
    on depth-1 trees) with L2 leaf regularization and quantile threshold candidates,
    requiring zero external C++ dependencies beyond NumPy.
    """

    def __init__(
        self,
        config: ModelTrainingConfig | None = None,
        *,
        standardize_features: bool = True,
        clip_zscore: float = 5.0,
    ) -> None:
        super().__init__(
            config=config,
            standardize_features=standardize_features,
            clip_zscore=clip_zscore,
        )
        self._init_logits: np.ndarray | None = None          # shape: (3,)
        self._stumps: list[tuple[DecisionStump, ...]] = []   # list of length n_estimators, each a 3-tuple of stumps

    @property
    def model_type(self) -> ModelType:
        return ModelType.GRADIENT_BOOSTED_STUMPS

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Sequence[str],
        *,
        horizon_bars: int = 12,
    ) -> "GradientBoostedStumpEnsemble":
        x_prep = self._validate_and_prepare_train_inputs(x_train, y_train, feature_names)
        n_samples, n_features = x_prep.shape
        y_int = y_train.astype(int)

        y_onehot = np.zeros((n_samples, N_CLASSES), dtype="float64")
        for idx, cls_idx in enumerate(y_int):
            y_onehot[idx, cls_idx] = 1.0

        sample_weights = _compute_balanced_sample_weights(
            y_int, self._config.class_weight_balanced
        )

        # Precompute candidate split thresholds per feature using deterministic quantiles on x_prep
        n_q = self._config.n_thresholds_per_feature
        quantiles = np.linspace(1.0 / (n_q + 1), n_q / (n_q + 1), n_q)
        candidate_thresholds: list[np.ndarray] = [
            np.unique(np.quantile(x_prep[:, f_idx], quantiles))
            for f_idx in range(n_features)
        ]

        init_logits = np.zeros(N_CLASSES, dtype="float64")
        f_scores = np.tile(init_logits, (n_samples, 1))
        stumps_stages: list[tuple[DecisionStump, ...]] = []

        lr = self._config.learning_rate
        l2 = max(1e-4, self._config.l2_regularization)

        for _ in range(self._config.n_estimators):
            probs = _softmax(f_scores)
            stage_stumps: list[DecisionStump] = []

            for k in range(N_CLASSES):
                p_k = probs[:, k]
                y_k = y_onehot[:, k]
                # First-order gradient g and second-order Hessian h (weighted by sample_weights)
                g = (p_k - y_k) * sample_weights
                h = np.maximum(p_k * (1.0 - p_k) * sample_weights, 1e-6)

                best_stump = self._fit_best_newton_stump(
                    x_prep=x_prep,
                    g=g,
                    h=h,
                    candidate_thresholds=candidate_thresholds,
                    l2=l2,
                )
                stage_stumps.append(best_stump)
                f_scores[:, k] += lr * best_stump.predict(x_prep)

            stumps_stages.append(tuple(stage_stumps))

        self._init_logits = init_logits
        self._stumps = stumps_stages

        schema_hash = compute_sha256_hex(",".join(self._feature_names))
        stump_sig = compute_sha256_hex(
            f"{schema_hash}:{n_samples}:{len(stumps_stages)}:"
            f"{[(s.feature_index, round(s.threshold, 4)) for stg in stumps_stages for s in stg]}"
        )
        self._metadata = ModelArtifactMetadata(
            model_id=f"gbdt-{stump_sig[:12]}",
            model_type=self.model_type,
            feature_names=self._feature_names,
            feature_schema_hash=schema_hash,
            train_samples=n_samples,
            horizon_bars=horizon_bars,
            random_seed=self._config.random_seed,
            standardize_features=self._standardize,
        )
        return self

    @staticmethod
    def _fit_best_newton_stump(
        *,
        x_prep: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        candidate_thresholds: list[np.ndarray],
        l2: float,
    ) -> DecisionStump:
        """Find the feature and threshold maximizing exact XGBoost/LightGBM Newton split gain."""
        g_total = float(np.sum(g))
        h_total = float(np.sum(h))
        parent_score = (g_total * g_total) / (h_total + l2)

        best_gain = -1.0
        best_feat = 0
        best_thresh = 0.0
        best_left_val = 0.0
        best_right_val = 0.0

        n_features = x_prep.shape[1]
        for f_idx in range(n_features):
            col = x_prep[:, f_idx]
            for thresh in candidate_thresholds[f_idx]:
                left_mask = col <= thresh
                if not np.any(left_mask) or np.all(left_mask):
                    continue
                g_left = float(np.sum(g[left_mask]))
                h_left = float(np.sum(h[left_mask]))
                g_right = g_total - g_left
                h_right = h_total - h_left

                gain = (
                    (g_left * g_left) / (h_left + l2)
                    + (g_right * g_right) / (h_right + l2)
                    - parent_score
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f_idx
                    best_thresh = float(thresh)
                    best_left_val = -g_left / (h_left + l2)
                    best_right_val = -g_right / (h_right + l2)

        return DecisionStump(
            feature_index=best_feat,
            threshold=best_thresh,
            left_value=best_left_val,
            right_value=best_right_val,
            gain=max(0.0, best_gain),
        )

    def predict_proba_matrix(
        self,
        x: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        x_prep = self._prepare_inference_inputs(x, feature_names=feature_names)
        assert self._init_logits is not None
        n_samples = x_prep.shape[0]
        logits = np.tile(self._init_logits, (n_samples, 1))
        lr = self._config.learning_rate

        for stage in self._stumps:
            for k in range(N_CLASSES):
                logits[:, k] += lr * stage[k].predict(x_prep)

        return _softmax(logits)

    def feature_importances(self) -> dict[str, Decimal]:
        if not self.is_fitted:
            raise ModelPredictionError("GradientBoostedStumpEnsemble is not fitted.")
        gains = np.zeros(len(self._feature_names), dtype="float64")
        for stage in self._stumps:
            for stump in stage:
                gains[stump.feature_index] += max(1e-6, stump.gain)
        total = float(np.sum(gains))
        result: dict[str, Decimal] = {}
        for idx, f_name in enumerate(self._feature_names):
            val = float(gains[idx]) / total if total > 1e-12 else 1.0 / len(self._feature_names)
            result[f_name] = Decimal(str(round(val, 4))).quantize(
                QUANTIZE_PROB, rounding=ROUND_HALF_UP
            )
        return result

    def to_artifact_dict(self) -> dict[str, object]:
        if not self.is_fitted or self._metadata is None or self._init_logits is None:
            raise ModelPredictionError("GradientBoostedStumpEnsemble is not fitted.")
        return {
            "metadata": self._metadata.model_dump(mode="json"),
            "training_config": self._config.model_dump(mode="json"),
            "scaler": self._scaler.to_dict(),
            "init_logits": self._init_logits.tolist(),
            "stages": [[s.to_dict() for s in stage] for stage in self._stumps],
        }

    @classmethod
    def from_artifact_dict(cls, payload: dict[str, object]) -> "GradientBoostedStumpEnsemble":
        meta = ModelArtifactMetadata.model_validate(payload["metadata"])
        t_cfg = ModelTrainingConfig.model_validate(payload["training_config"])
        scaler_dict = payload["scaler"]
        assert isinstance(scaler_dict, dict)

        model = cls(
            config=t_cfg,
            standardize_features=meta.standardize_features,
        )
        model._scaler = FoldStandardScaler.from_dict(scaler_dict)
        model._feature_names = meta.feature_names
        model._metadata = meta
        model._init_logits = np.array(payload["init_logits"], dtype="float64")
        raw_stages = payload["stages"]
        assert isinstance(raw_stages, list)
        model._stumps = [
            tuple(DecisionStump.from_dict(s_dict) for s_dict in stg)
            for stg in raw_stages
        ]
        return model

    @classmethod
    def load_artifact(cls, file_path: str | Path) -> "GradientBoostedStumpEnsemble":
        path = Path(file_path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_artifact_dict(raw)


def load_model_artifact(file_path: str | Path) -> BaseMLClassifier:
    """Load either a RegularizedLogisticClassifier or GradientBoostedStumpEnsemble from JSON."""
    path = Path(file_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    meta = raw.get("metadata", {})
    m_type = ModelType(str(meta.get("model_type")))
    if m_type == ModelType.REGULARIZED_LOGISTIC:
        return RegularizedLogisticClassifier.from_artifact_dict(raw)
    if m_type == ModelType.GRADIENT_BOOSTED_STUMPS:
        return GradientBoostedStumpEnsemble.from_artifact_dict(raw)
    raise ModelPredictionError(f"Unsupported model artifact type: {m_type}.")
