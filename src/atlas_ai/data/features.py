"""Deterministic, lookahead-free technical feature engineering pipeline for ATLAS AI."""

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from atlas_ai.data.exceptions import FeatureCalculationError
from atlas_ai.data.models import FeatureConfig
from atlas_ai.data.validator import REQUIRED_OHLCV_COLUMNS


class FeaturePipeline:
    """Computes deterministic technical and time/session features from validated OHLCV DataFrames.

    Key Guarantees:
    1. **Zero Lookahead Bias:** Every indicator at bar `t` is computed strictly from bars `<= t`.
       `session_high` and `session_low` use expanding cumulative extrema within each trading session
       so future highs/lows later in the same session never leak into earlier bars.
    2. **Explicit Warm-Up Handling:** Rolling indicators (`atr_*`, `sma_*`, `ema_*`, `rolling_high_*`,
       `rolling_low_*`, `prev_high`, `prev_low`) retain `NaN` during their warm-up window (`min_periods=window`)
       and are never silently filled with `0`.
    3. **Non-Destructive:** Original OHLCV columns (`timestamp`, `open`, `high`, `low`, `close`, `volume`)
       are preserved unmodified.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        """Initialize the FeaturePipeline with a FeatureConfig."""
        self._config = config or FeatureConfig()

    @property
    def config(self) -> FeatureConfig:
        """Return active FeatureConfig."""
        return self._config

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute configured features on an OHLCV DataFrame in chronological order.

        Raises:
            FeatureCalculationError: If required OHLCV columns are missing or timestamps are not sorted.
        """
        missing = [col for col in REQUIRED_OHLCV_COLUMNS if col not in df.columns]
        if missing:
            raise FeatureCalculationError(
                f"FeaturePipeline requires OHLCV columns {REQUIRED_OHLCV_COLUMNS}; missing {missing}."
            )

        # Strip internal loader helper columns if present while preserving all public columns
        public_cols = [c for c in df.columns if not str(c).startswith("_raw_")]
        out = df[public_cols].copy()

        if out.empty:
            return out

        # Verify strictly monotonic increasing timestamps to guarantee chronological calculation
        ts_series = pd.to_datetime(out["timestamp"])
        if not ts_series.is_monotonic_increasing or ts_series.duplicated().any():
            raise FeatureCalculationError(
                "FeaturePipeline requires strictly increasing, non-duplicate timestamps."
            )

        open_s = out["open"].astype("float64")
        high_s = out["high"].astype("float64")
        low_s = out["low"].astype("float64")
        close_s = out["close"].astype("float64")

        # 1. Volatility features: True Range (tr), ATR (atr_{n}), ATR % (atr_pct_{n})
        if self._config.include_volatility:
            prev_close = close_s.shift(1)
            hl_range = high_s - low_s
            hc_dist = (high_s - prev_close).abs()
            lc_dist = (low_s - prev_close).abs()

            # For bar 0 (where prev_close is NaN), True Range is simply high - low
            tr = pd.concat([hl_range, hc_dist, lc_dist], axis=1).max(axis=1, skipna=True)
            out["tr"] = tr

            n_atr = self._config.atr_period
            atr = tr.rolling(window=n_atr, min_periods=n_atr).mean()
            out[f"atr_{n_atr}"] = atr
            out[f"atr_pct_{n_atr}"] = (atr / close_s) * 100.0

        # 2. Moving averages: SMA (20, 50, 100, 200) and EMA (9, 21, 50)
        if self._config.include_moving_averages:
            for period in self._config.sma_periods:
                out[f"sma_{period}"] = close_s.rolling(window=period, min_periods=period).mean()

            for span in self._config.ema_periods:
                out[f"ema_{span}"] = close_s.ewm(
                    span=span,
                    adjust=False,
                    min_periods=span,
                ).mean()

        # 3. Candle geometry features
        if self._config.include_candle_features:
            body = (close_s - open_s).abs()
            c_range = high_s - low_s
            max_oc = np.maximum(open_s, close_s)
            min_oc = np.minimum(open_s, close_s)
            upper_wick = high_s - max_oc
            lower_wick = min_oc - low_s

            direction = pd.Series("DOJI", index=out.index, dtype="object")
            direction = direction.mask(close_s > open_s, "BULLISH")
            direction = direction.mask(close_s < open_s, "BEARISH")

            # Avoid division by zero on flat 0-range doji candles
            body_to_range = np.where(c_range > 0.0, body / c_range, 0.0)

            out["candle_body"] = body
            out["upper_wick"] = upper_wick
            out["lower_wick"] = lower_wick
            out["candle_range"] = c_range
            out["candle_direction"] = direction
            out["body_to_range_ratio"] = pd.Series(body_to_range, index=out.index, dtype="float64")

        # 4. Time & configurable session classification features (computed before structure if needed)
        session_tz = ZoneInfo(self._config.calendar.session_timezone)
        if ts_series.dt.tz is None:
            localized_ts = ts_series.dt.tz_localize(session_tz)
        else:
            localized_ts = ts_series.dt.tz_convert(session_tz)

        session_labels = pd.Series(
            [self._config.calendar.classify_session(ts.to_pydatetime()).value for ts in localized_ts],
            index=out.index,
            dtype="object",
        )

        if self._config.include_time_features:
            out["hour_of_day"] = localized_ts.dt.hour.astype("int64")
            out["day_of_week"] = localized_ts.dt.day_of_week.astype("int64")
            out["trading_session"] = session_labels

        # 5. Market structure preparation features (strictly causal, zero lookahead)
        if self._config.include_structure_features:
            lookback = self._config.structure_lookback
            out[f"rolling_high_{lookback}"] = high_s.rolling(
                window=lookback, min_periods=lookback
            ).max()
            out[f"rolling_low_{lookback}"] = low_s.rolling(
                window=lookback, min_periods=lookback
            ).min()
            out["prev_high"] = high_s.shift(1)
            out["prev_low"] = low_s.shift(1)

            # Session High / Session Low:
            # Group by consecutive (calendar_date_in_session_tz, trading_session) blocks and
            # take cumulative expanding max/min so only bars up to `t` within the current session are used.
            session_dates = localized_ts.dt.date
            session_change = (session_labels != session_labels.shift(1)) | (
                session_dates != session_dates.shift(1)
            )
            session_group_id = session_change.cumsum()

            out["session_high"] = high_s.groupby(session_group_id).cummax()
            out["session_low"] = low_s.groupby(session_group_id).cummin()

        return out
