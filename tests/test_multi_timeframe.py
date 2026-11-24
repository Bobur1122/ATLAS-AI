"""Phase 3 tests for MultiTimeframeAnalyzer and closed-candle anti-lookahead synchronization."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from atlas_ai.data.features import FeaturePipeline
from atlas_ai.domain.market import Timeframe
from atlas_ai.domain.signals import SignalDirection
from atlas_ai.strategies.multi_timeframe import (
    MultiTimeframeAnalyzer,
    filter_closed_candles_as_of,
)
from tests.test_strategies import _make_trend_df


def test_filter_closed_candles_prevents_incomplete_htf_candle_leakage() -> None:
    """An H1 candle opening at 10:00 UTC closes at 11:00 UTC and must NOT be visible at 10:30 UTC."""
    h1_df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-01-05 08:00:00", tz="UTC"),  # Closes 09:00 UTC
                pd.Timestamp("2026-01-05 09:00:00", tz="UTC"),  # Closes 10:00 UTC
                pd.Timestamp("2026-01-05 10:00:00", tz="UTC"),  # Closes 11:00 UTC (INCOMPLETE at 10:30!)
            ],
            "open": [2650.0, 2652.0, 2655.0],
            "high": [2655.0, 2658.0, 2700.0],
            "low": [2648.0, 2650.0, 2654.0],
            "close": [2652.0, 2655.0, 2695.0],
            "volume": [100.0, 100.0, 100.0],
        }
    )

    eval_at_10_30 = datetime(2026, 1, 5, 10, 30, tzinfo=timezone.utc)
    closed_at_10_30 = filter_closed_candles_as_of(h1_df, Timeframe.H1, eval_at_10_30)
    assert len(closed_at_10_30) == 2
    assert closed_at_10_30["timestamp"].iloc[-1] == pd.Timestamp("2026-01-05 09:00:00", tz="UTC")

    eval_at_11_00 = datetime(2026, 1, 5, 11, 0, tzinfo=timezone.utc)
    closed_at_11_00 = filter_closed_candles_as_of(h1_df, Timeframe.H1, eval_at_11_00)
    assert len(closed_at_11_00) == 3


def test_mtf_analyzer_aligned_vs_conflicting_and_missing_timeframes() -> None:
    """MultiTimeframeAnalyzer must approve aligned timeframes, flag conflicts as WAIT, and report missing timeframes."""
    analyzer = MultiTimeframeAnalyzer()

    h1_up = _make_trend_df("up", bars=70)
    m15_up = _make_trend_df("up", bars=70)
    m5_up = _make_trend_df("up", bars=70)
    m15_down = _make_trend_df("down", bars=70)

    eval_ts = datetime(2026, 1, 10, 0, 0, tzinfo=timezone.utc)

    # 1. All 3 timeframes aligned bullish -> BUY
    aligned_res = analyzer.analyze_mtf(
        {Timeframe.H1: h1_up, Timeframe.M15: m15_up, Timeframe.M5: m5_up},
        evaluation_close_timestamp=eval_ts,
    )
    assert aligned_res.aligned_direction == SignalDirection.BUY
    assert aligned_res.has_conflict is False
    assert aligned_res.alignment_score > 0

    # 2. H1 is BUY while M15 is SELL -> conflict -> WAIT
    conflict_res = analyzer.analyze_mtf(
        {Timeframe.H1: h1_up, Timeframe.M15: m15_down, Timeframe.M5: m5_up},
        evaluation_close_timestamp=eval_ts,
    )
    assert conflict_res.aligned_direction == SignalDirection.WAIT
    assert conflict_res.has_conflict is True

    # 3. Missing M5 timeframe -> explicitly reported in missing_timeframes -> WAIT
    missing_res = analyzer.analyze_mtf(
        {Timeframe.H1: h1_up, Timeframe.M15: m15_up},
        evaluation_close_timestamp=eval_ts,
    )
    assert missing_res.aligned_direction == SignalDirection.WAIT
    assert Timeframe.M5 in missing_res.missing_timeframes
