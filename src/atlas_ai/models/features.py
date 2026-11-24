"""Causal, stationary ML feature engineering and train-fold-only standardization for ATLAS AI Phase 6."""

import math

import numpy as np
import pandas as pd

from atlas_ai.data.features import FeaturePipeline
from atlas_ai.knowledge.models import compute_sha256_hex
from atlas_ai.models.config import MLFeatureConfig
from atlas_ai.models.exceptions import ModelConfigurationError, ModelDataLeakageError


class FoldStandardScaler:
    """Z-score feature standardizer fit EXCLUSIVELY on a single training fold.

    Prevents global mean/variance leakage from validation or out-of-sample test bars.
    """

    def __init__(self, clip_zscore: float = 5.0) -> None:
        self._clip = float(clip_zscore)
        self._means: np.ndarray | None = None
        self._stds: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self._means is not None and self._stds is not None

    @property
    def means(self) -> tuple[float, ...]:
        if self._means is None:
            return ()
        return tuple(float(x) for x in self._means)

    @property
    def stds(self) -> tuple[float, ...]:
        if self._stds is None:
            return ()
        return tuple(float(x) for x in self._stds)

    def fit(self, x_train: np.ndarray) -> "FoldStandardScaler":
        """Compute per-column mean and standard deviation strictly on `x_train`."""
        if x_train.ndim != 2 or x_train.shape[0] < 2:
            raise ModelConfigurationError("FoldStandardScaler.fit requires a 2D array with >= 2 rows.")
        means = np.mean(x_train, axis=0)
        stds = np.std(x_train, axis=0)
        # Replace near-zero std with 1.0 to avoid division by zero on constant features
        stds = np.where(stds < 1e-8, 1.0, stds)
        self._means = means.astype("float64")
        self._stds = stds.astype("float64")
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Standardize `x` using previously fitted training-fold statistics."""
        if self._means is None or self._stds is None:
            raise ModelConfigurationError("FoldStandardScaler must be fitted on training data before transform.")
        if x.ndim != 2 or x.shape[1] != len(self._means):
            raise ModelConfigurationError(
                f"Feature dimension mismatch in FoldStandardScaler: expected {len(self._means)}, got {x.shape}."
            )
        z = (x - self._means) / self._stds
        return np.clip(z, -self._clip, self._clip).astype("float64")

    def fit_transform(self, x_train: np.ndarray) -> np.ndarray:
        """Fit on `x_train` and return standardized `x_train`."""
        return self.fit(x_train).transform(x_train)

    def to_dict(self) -> dict[str, object]:
        return {
            "clip_zscore": self._clip,
            "means": list(self.means),
            "stds": list(self.stds),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "FoldStandardScaler":
        clip = float(str(data.get("clip_zscore", 5.0)))
        scaler = cls(clip_zscore=clip)
        means_raw = data.get("means")
        stds_raw = data.get("stds")
        if isinstance(means_raw, list) and isinstance(stds_raw, list) and means_raw:
            scaler._means = np.array([float(x) for x in means_raw], dtype="float64")
            scaler._stds = np.array([float(x) for x in stds_raw], dtype="float64")
        return scaler


class MLFeatureBuilder:
    """Constructs stationary, scale-invariant ML features causally from Phase 2 FeaturePipeline outputs.

    Key Anti-Leakage Guarantee:
    - Feature row at index `t` depends strictly on market data at indices `<= t`.
    - No forward shifts, global normalization, or future extrema are ever used.
    """

    def __init__(
        self,
        config: MLFeatureConfig | None = None,
        feature_pipeline: FeaturePipeline | None = None,
    ) -> None:
        self._config = config or MLFeatureConfig()
        self._feature_pipeline = feature_pipeline or FeaturePipeline()

    @property
    def config(self) -> MLFeatureConfig:
        return self._config

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Ordered tuple of stationary numeric ML feature column names."""
        names: list[str] = []
        for lag in self._config.return_lags:
            names.append(f"ret_pct_lag_{lag}")
            names.append(f"norm_ret_atr_lag_{lag}")
        names.extend(
            [
                "norm_close_ema9",
                "norm_close_ema21",
                "norm_close_ema50",
                "norm_ema9_ema21",
                "norm_ema21_ema50",
                "norm_close_sma20",
                "norm_close_sma50",
                "range_pos_20",
                "tr_to_atr",
                "atr_regime_ratio",
                "upper_wick_to_atr",
                "lower_wick_to_atr",
                "signed_body_to_range",
                "atr_pct_14",
                "rel_volume_20",
                "hour_sin",
                "hour_cos",
                "dow_norm",
            ]
        )
        return tuple(names)

    @property
    def feature_schema_hash(self) -> str:
        """Deterministic SHA-256 hash of ordered `feature_names`."""
        return compute_sha256_hex(",".join(self.feature_names))

    def _ensure_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        required = (
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tr",
            "atr_14",
            "atr_pct_14",
            "ema_9",
            "ema_21",
            "ema_50",
            "sma_20",
            "sma_50",
            "rolling_high_20",
            "rolling_low_20",
            "upper_wick",
            "lower_wick",
            "body_to_range_ratio",
            "hour_of_day",
            "day_of_week",
        )
        if all(col in df.columns for col in required):
            return df
        return self._feature_pipeline.transform(df)

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute causal stationary ML features and return DataFrame containing `timestamp`,
        base OHLC/ATR columns, and all `self.feature_names`.
        """
        if df.empty:
            raise ModelConfigurationError("Cannot build ML features from an empty DataFrame.")

        ts_series = pd.to_datetime(df["timestamp"])
        if not ts_series.is_monotonic_increasing or ts_series.duplicated().any():
            raise ModelDataLeakageError(
                "MLFeatureBuilder requires strictly increasing, non-duplicate timestamps."
            )

        base = self._ensure_base_features(df).copy()
        close_s = base["close"].astype("float64")
        open_s = base["open"].astype("float64")
        vol_s = base["volume"].astype("float64")
        atr_s = base["atr_14"].astype("float64")
        safe_atr = np.where((atr_s > 1e-8) & (~np.isnan(atr_s)), atr_s, np.nan)

        # 1. Causal backward lag returns (percentage and ATR-normalized)
        for lag in self._config.return_lags:
            prev_c = close_s.shift(lag)
            diff = close_s - prev_c
            base[f"ret_pct_lag_{lag}"] = (diff / prev_c) * 100.0
            base[f"norm_ret_atr_lag_{lag}"] = diff / safe_atr

        # 2. Stationary ATR-normalized moving average distances
        base["norm_close_ema9"] = (close_s - base["ema_9"].astype("float64")) / safe_atr
        base["norm_close_ema21"] = (close_s - base["ema_21"].astype("float64")) / safe_atr
        base["norm_close_ema50"] = (close_s - base["ema_50"].astype("float64")) / safe_atr
        base["norm_ema9_ema21"] = (
            base["ema_9"].astype("float64") - base["ema_21"].astype("float64")
        ) / safe_atr
        base["norm_ema21_ema50"] = (
            base["ema_21"].astype("float64") - base["ema_50"].astype("float64")
        ) / safe_atr
        base["norm_close_sma20"] = (close_s - base["sma_20"].astype("float64")) / safe_atr
        base["norm_close_sma50"] = (close_s - base["sma_50"].astype("float64")) / safe_atr

        # 3. Normalized 20-bar range position in [0, 1]
        r_high = base["rolling_high_20"].astype("float64")
        r_low = base["rolling_low_20"].astype("float64")
        r_span = r_high - r_low
        safe_span = np.where((r_span > 1e-8) & (~np.isnan(r_span)), r_span, np.nan)
        base["range_pos_20"] = (close_s - r_low) / safe_span

        # 4. Volatility and candle geometry ratios
        base["tr_to_atr"] = base["tr"].astype("float64") / safe_atr
        atr_roll_mean = atr_s.rolling(
            window=self._config.atr_ratio_window,
            min_periods=self._config.atr_ratio_window,
        ).mean()
        safe_atr_mean = np.where(
            (atr_roll_mean > 1e-8) & (~np.isnan(atr_roll_mean)), atr_roll_mean, np.nan
        )
        base["atr_regime_ratio"] = atr_s / safe_atr_mean
        base["upper_wick_to_atr"] = base["upper_wick"].astype("float64") / safe_atr
        base["lower_wick_to_atr"] = base["lower_wick"].astype("float64") / safe_atr

        sign = np.where(close_s > open_s, 1.0, np.where(close_s < open_s, -1.0, 0.0))
        base["signed_body_to_range"] = sign * base["body_to_range_ratio"].astype("float64")

        # 5. Relative volume ratio
        vol_mean = vol_s.rolling(
            window=self._config.volume_window,
            min_periods=self._config.volume_window,
        ).mean()
        safe_vol_mean = np.where((vol_mean > 1e-8) & (~np.isnan(vol_mean)), vol_mean, 1.0)
        base["rel_volume_20"] = np.where(
            np.isnan(vol_mean), np.nan, vol_s / safe_vol_mean
        )

        # 6. Cyclical hour and normalized day of week
        hours = base["hour_of_day"].astype("float64")
        two_pi_h = (2.0 * math.pi * hours) / 24.0
        base["hour_sin"] = np.sin(two_pi_h)
        base["hour_cos"] = np.cos(two_pi_h)
        base["dow_norm"] = base["day_of_week"].astype("float64") / 6.0

        return base

    def extract_valid_row_mask(self, ml_df: pd.DataFrame) -> np.ndarray:
        """Return a boolean mask of rows where all `self.feature_names` are finite (no warm-up NaNs)."""
        mat = ml_df[list(self.feature_names)].to_numpy(dtype="float64")
        return np.all(np.isfinite(mat), axis=1)
