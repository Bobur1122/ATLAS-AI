"""Phase 2 tests for deterministic technical feature calculations, warm-up NaNs, and zero lookahead bias."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atlas_ai.data.exceptions import FeatureCalculationError
from atlas_ai.data.features import FeaturePipeline
from atlas_ai.data.loader import HistoricalCSVLoader
from atlas_ai.data.models import FeatureConfig, MarketCalendarConfig, TradingSessionName
from atlas_ai.domain.market import Timeframe

SYNTHETIC_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_xauusd_h1.csv"


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Load the 220-bar synthetic XAUUSD H1 dataset."""
    loader = HistoricalCSVLoader()
    return loader.load_csv(SYNTHETIC_FIXTURE_PATH, symbol="XAUUSD", timeframe=Timeframe.H1)


def test_true_range_and_atr_calculations() -> None:
    """Verify True Range (tr), ATR, and ATR percentage against exact hand-calculated values."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-05 00:00:00", periods=4, freq="1h", tz="UTC"),
            "open": [2650.0, 2655.0, 2648.0, 2660.0],
            "high": [2655.0, 2662.0, 2652.0, 2665.0],
            "low": [2645.0, 2652.0, 2640.0, 2655.0],
            "close": [2650.0, 2660.0, 2642.0, 2660.0],
            "volume": [100.0, 120.0, 150.0, 130.0],
        }
    )
    # Expected True Range:
    # Bar 0: high(2655) - low(2645) = 10.0
    # Bar 1: max(2662-2652=10, |2662-2650|=12, |2652-2650|=2) = 12.0
    # Bar 2: max(2652-2640=12, |2652-2660|=8, |2640-2660|=20) = 20.0
    # Bar 3: max(2665-2655=10, |2665-2642|=23, |2655-2642|=13) = 23.0
    cfg = FeatureConfig(atr_period=3)
    pipeline = FeaturePipeline(config=cfg)
    feat = pipeline.transform(df)

    assert list(feat["tr"]) == pytest.approx([10.0, 12.0, 20.0, 23.0])
    # Warm-up: bars 0 and 1 must be NaN for atr_3
    assert np.isnan(feat["atr_3"].iloc[0])
    assert np.isnan(feat["atr_3"].iloc[1])
    # Bar 2 ATR(3) = (10 + 12 + 20) / 3 = 14.0
    assert feat["atr_3"].iloc[2] == pytest.approx(14.0)
    # Bar 3 ATR(3) = (12 + 20 + 23) / 3 = 18.3333333
    assert feat["atr_3"].iloc[3] == pytest.approx(55.0 / 3.0)
    assert feat["atr_pct_3"].iloc[2] == pytest.approx((14.0 / 2642.0) * 100.0)


def test_sma_and_ema_calculations_and_warmup_nans(sample_ohlcv_df: pd.DataFrame) -> None:
    """Verify SMA (20, 50, 100, 200) and EMA (9, 21, 50) calculations and warm-up NaN counts."""
    pipeline = FeaturePipeline()
    feat = pipeline.transform(sample_ohlcv_df)

    for period in (20, 50, 100, 200):
        col = f"sma_{period}"
        assert col in feat.columns
        # Exactly (period - 1) warm-up NaNs at the start, and never replaced with 0
        assert feat[col].iloc[: period - 1].isna().all()
        assert feat[col].iloc[period - 1 :].notna().all()
        # Verify exact mean at index (period - 1)
        expected_mean = sample_ohlcv_df["close"].iloc[:period].mean()
        assert feat[col].iloc[period - 1] == pytest.approx(expected_mean)

    for span in (9, 21, 50):
        col = f"ema_{span}"
        assert col in feat.columns
        assert feat[col].iloc[: span - 1].isna().all()
        assert feat[col].iloc[span - 1 :].notna().all()


def test_candle_geometry_features() -> None:
    """Verify body size, upper/lower wicks, range, direction, and body-to-range ratio."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-05 00:00:00", periods=3, freq="1h", tz="UTC"),
            "open": [2650.0, 2660.0, 2655.0],
            "high": [2665.0, 2662.0, 2655.0],
            "low": [2648.0, 2645.0, 2655.0],
            "close": [2660.0, 2650.0, 2655.0],
            "volume": [100.0, 100.0, 0.0],
        }
    )
    feat = FeaturePipeline().transform(df)

    # Row 0: Bullish candle (open=2650, close=2660, high=2665, low=2648)
    assert feat["candle_body"].iloc[0] == pytest.approx(10.0)
    assert feat["upper_wick"].iloc[0] == pytest.approx(5.0)
    assert feat["lower_wick"].iloc[0] == pytest.approx(2.0)
    assert feat["candle_range"].iloc[0] == pytest.approx(17.0)
    assert feat["candle_direction"].iloc[0] == "BULLISH"
    assert feat["body_to_range_ratio"].iloc[0] == pytest.approx(10.0 / 17.0)

    # Row 1: Bearish candle (open=2660, close=2650, high=2662, low=2645)
    assert feat["candle_body"].iloc[1] == pytest.approx(10.0)
    assert feat["upper_wick"].iloc[1] == pytest.approx(2.0)
    assert feat["lower_wick"].iloc[1] == pytest.approx(5.0)
    assert feat["candle_range"].iloc[1] == pytest.approx(17.0)
    assert feat["candle_direction"].iloc[1] == "BEARISH"

    # Row 2: Zero-range Doji candle (open=high=low=close=2655) -> body_to_range_ratio must be 0.0, not NaN
    assert feat["candle_direction"].iloc[2] == "DOJI"
    assert feat["body_to_range_ratio"].iloc[2] == pytest.approx(0.0)


def test_rolling_and_session_highs_lows_and_configurable_session_timezone() -> None:
    """Verify rolling highs/lows, prev_high/prev_low, and causal session_high/session_low."""
    df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-01-05 06:00:00", tz="UTC"),  # ASIAN
                pd.Timestamp("2026-01-05 07:00:00", tz="UTC"),  # LONDON bar 1
                pd.Timestamp("2026-01-05 08:00:00", tz="UTC"),  # LONDON bar 2 (higher high)
                pd.Timestamp("2026-01-05 09:00:00", tz="UTC"),  # LONDON bar 3 (lower low)
            ],
            "open": [2650.0, 2652.0, 2655.0, 2658.0],
            "high": [2655.0, 2660.0, 2675.0, 2668.0],
            "low": [2648.0, 2650.0, 2654.0, 2642.0],
            "close": [2652.0, 2655.0, 2665.0, 2645.0],
            "volume": [100.0, 150.0, 200.0, 180.0],
        }
    )
    cfg = FeatureConfig(structure_lookback=2)
    feat = FeaturePipeline(config=cfg).transform(df)

    # Check session classification
    assert list(feat["trading_session"]) == [
        TradingSessionName.ASIAN.value,
        TradingSessionName.LONDON.value,
        TradingSessionName.LONDON.value,
        TradingSessionName.LONDON.value,
    ]

    # Check causal session_high and session_low within LONDON session (bars 1, 2, 3):
    # At bar 1 (07:00 UTC), session_high MUST be 2660.0 (NOT 2675.0 from bar 2!)
    assert feat["session_high"].iloc[1] == pytest.approx(2660.0)
    assert feat["session_high"].iloc[2] == pytest.approx(2675.0)
    assert feat["session_high"].iloc[3] == pytest.approx(2675.0)

    # At bar 1 & 2, session_low MUST be 2650.0 (NOT 2642.0 from bar 3!)
    assert feat["session_low"].iloc[1] == pytest.approx(2650.0)
    assert feat["session_low"].iloc[2] == pytest.approx(2650.0)
    assert feat["session_low"].iloc[3] == pytest.approx(2642.0)

    # Check prev_high and prev_low
    assert np.isnan(feat["prev_high"].iloc[0])
    assert feat["prev_high"].iloc[1] == pytest.approx(2655.0)
    assert feat["prev_low"].iloc[1] == pytest.approx(2648.0)

    # Check configurable session_timezone (e.g., America/New_York)
    ny_cal = MarketCalendarConfig(session_timezone="America/New_York")
    ny_feat = FeaturePipeline(config=FeatureConfig(calendar=ny_cal)).transform(df)
    # 06:00 UTC in January is 01:00 EST
    assert ny_feat["hour_of_day"].iloc[0] == 1


def test_strict_no_lookahead_bias_across_all_features(sample_ohlcv_df: pd.DataFrame) -> None:
    """Prove mathematically that future candles never alter any feature on earlier candles.

    Methodology:
    Compute features on the first N=120 bars (`prefix_features`), and compare against the first
    120 rows of features computed on the full 220-bar dataset (`full_features.iloc[:120]`), even
    after mutating future bars 120..219 with extreme price spikes.
    """
    pipeline = FeaturePipeline()

    prefix_df = sample_ohlcv_df.iloc[:120].copy().reset_index(drop=True)
    prefix_features = pipeline.transform(prefix_df)

    shocked_full_df = sample_ohlcv_df.copy()
    # Inject massive future price spike on bars 120..219
    shocked_full_df.loc[120:, "high"] = shocked_full_df.loc[120:, "high"] + 5000.0
    shocked_full_df.loc[120:, "close"] = shocked_full_df.loc[120:, "close"] + 4000.0

    full_features = pipeline.transform(shocked_full_df)
    truncated_full_features = full_features.iloc[:120].reset_index(drop=True)

    pd.testing.assert_frame_equal(prefix_features, truncated_full_features)


def test_feature_pipeline_determinism_and_empty_or_unsorted_handling(
    sample_ohlcv_df: pd.DataFrame,
) -> None:
    """FeaturePipeline must be deterministic, handle empty DataFrames, and reject unsorted inputs."""
    pipeline = FeaturePipeline()
    run1 = pipeline.transform(sample_ohlcv_df)
    run2 = pipeline.transform(sample_ohlcv_df)
    pd.testing.assert_frame_equal(run1, run2)

    # Empty DataFrame returns empty DataFrame cleanly
    empty_df = sample_ohlcv_df.iloc[0:0].copy()
    empty_out = pipeline.transform(empty_df)
    assert empty_out.empty

    # Unsorted DataFrame raises FeatureCalculationError
    unsorted_df = sample_ohlcv_df.iloc[[1, 0, 2]].copy()
    with pytest.raises(FeatureCalculationError, match="strictly increasing"):
        pipeline.transform(unsorted_df)
