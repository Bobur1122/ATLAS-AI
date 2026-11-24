"""Phase 3 tests for MarketRegimeDetector, EmaTrendStrategy, and StrategyEngine orchestration.

NOTE: All market data and signals in these tests are strictly SYNTHETIC test fixtures.
"""

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from atlas_ai.data.features import FeaturePipeline
from atlas_ai.data.loader import HistoricalCSVLoader
from atlas_ai.domain.market import InstrumentSpec, Timeframe
from atlas_ai.domain.risk import AccountState, RiskParameters
from atlas_ai.domain.signals import SignalDirection
from atlas_ai.strategies.config import RegimeDetectorConfig
from atlas_ai.strategies.engine import StrategyEngine
from atlas_ai.strategies.models import MarketRegime, RiskEvaluationStatus
from atlas_ai.strategies.trend import EmaTrendStrategy, MarketRegimeDetector

SYNTHETIC_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_xauusd_h1.csv"


def _make_trend_df(direction: str = "up", bars: int = 80) -> pd.DataFrame:
    """Create a deterministic synthetic trending OHLCV + features DataFrame."""
    timestamps = pd.date_range("2026-01-05 00:00:00", periods=bars, freq="1h", tz="UTC")
    opens, highs, lows, closes, vols = [], [], [], [], []
    price = 2600.0
    step = 2.0 if direction == "up" else (-2.0 if direction == "down" else 0.0)

    for i in range(bars):
        if direction == "range":
            # Flat oscillation around 2600.0 with zero net drift and Doji close on the final bar
            if i == bars - 1:
                o = 2600.0
                c = 2600.0
            else:
                o = 2600.0 + (1.0 if i % 2 == 0 else -1.0)
                c = 2600.0 + (-1.0 if i % 2 == 0 else 1.0)
            h = max(o, c) + 2.0
            l = min(o, c) - 2.0
        else:
            o = price
            c = price + step
            h = max(o, c) + 0.5
            l = min(o, c) - 0.5
        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        vols.append(1000.0)
        price = c

    raw = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
        }
    )
    return FeaturePipeline().transform(raw)


def test_market_regime_detector_all_regimes() -> None:
    """MarketRegimeDetector must classify UNKNOWN (warm-up), UPTREND, DOWNTREND, RANGING, HIGH_VOLATILITY, and LOW_VOLATILITY."""
    detector = MarketRegimeDetector()

    up_df = _make_trend_df("up", bars=70)
    # Bar 5 has NaNs in ema_50 / sma_50 -> UNKNOWN
    reg_warmup, res_warmup = detector.detect_regime(up_df, as_of_index=5)
    assert reg_warmup == MarketRegime.UNKNOWN
    assert res_warmup.direction == SignalDirection.WAIT

    # Bar 69 has full warm-up and clean EMA9 > EMA21 > EMA50 -> UPTREND
    reg_up, res_up = detector.detect_regime(up_df)
    assert reg_up == MarketRegime.UPTREND
    assert res_up.direction == SignalDirection.BUY

    # Downtrend
    down_df = _make_trend_df("down", bars=70)
    reg_down, res_down = detector.detect_regime(down_df)
    assert reg_down == MarketRegime.DOWNTREND
    assert res_down.direction == SignalDirection.SELL

    # Ranging
    range_df = _make_trend_df("range", bars=70)
    reg_range, _ = detector.detect_regime(range_df)
    assert reg_range == MarketRegime.RANGING

    # High volatility override
    hv_detector = MarketRegimeDetector(
        RegimeDetectorConfig(high_volatility_atr_pct=Decimal("0.05"), low_volatility_atr_pct=Decimal("0.01"))
    )
    reg_hv, _ = hv_detector.detect_regime(up_df)
    assert reg_hv == MarketRegime.HIGH_VOLATILITY

    # Low volatility override
    lv_detector = MarketRegimeDetector(
        RegimeDetectorConfig(high_volatility_atr_pct=Decimal("5.0"), low_volatility_atr_pct=Decimal("2.0"))
    )
    reg_lv, _ = lv_detector.detect_regime(up_df)
    assert reg_lv == MarketRegime.LOW_VOLATILITY


def test_ema_trend_strategy_buy_sell_wait_and_warmup() -> None:
    """EmaTrendStrategy must return BUY on bullish stack, SELL on bearish stack, and WAIT during warm-up or range."""
    strat = EmaTrendStrategy()

    up_df = _make_trend_df("up", bars=70)
    assert strat.evaluate(up_df, as_of_index=10).direction == SignalDirection.WAIT

    buy_res = strat.evaluate(up_df)
    assert buy_res.direction == SignalDirection.BUY
    assert buy_res.entry_price is not None and buy_res.stop_loss is not None and buy_res.take_profit is not None
    assert buy_res.stop_loss < buy_res.entry_price < buy_res.take_profit

    down_df = _make_trend_df("down", bars=70)
    sell_res = strat.evaluate(down_df)
    assert sell_res.direction == SignalDirection.SELL
    assert sell_res.entry_price is not None and sell_res.stop_loss is not None and sell_res.take_profit is not None
    assert sell_res.take_profit < sell_res.entry_price < sell_res.stop_loss

    range_df = _make_trend_df("range", bars=70)
    wait_res = strat.evaluate(range_df)
    assert wait_res.direction == SignalDirection.WAIT


def test_strategy_engine_orchestration_and_future_candle_lookahead_prevention() -> None:
    """StrategyEngine must produce deterministic reports and never alter historical evaluations when future bars are mutated."""
    up_df = _make_trend_df("up", bars=100)
    engine = StrategyEngine()

    # Evaluate at bar index 75 on the 76-bar prefix
    prefix_df = up_df.iloc[:76].copy().reset_index(drop=True)
    report_prefix = engine.analyze(prefix_df, symbol="XAUUSD", timeframe=Timeframe.H1, register_in_lifecycle=False)

    # Mutate future bars 76..99 with massive crash
    shocked_df = up_df.copy()
    shocked_df.loc[76:, "close"] = 1000.0
    shocked_df.loc[76:, "low"] = 900.0
    shocked_df.loc[76:, "high"] = 1100.0

    report_with_future = engine.analyze(
        shocked_df,
        symbol="XAUUSD",
        timeframe=Timeframe.H1,
        as_of_index=75,
        register_in_lifecycle=False,
    )

    assert report_prefix.signal == report_with_future.signal
    assert report_prefix.confluence.final_confidence == report_with_future.confluence.final_confidence
    assert report_prefix.regime == report_with_future.regime
