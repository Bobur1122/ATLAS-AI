"""Phase 3 tests for SMC/ICT strategies and analyzers:
- MarketStructureStrategy (Swings, HH/HL/LH/LL, BOS, CHoCH)
- LiquiditySweepStrategy (EQH, EQL, BSL/SSL sweeps and rejection)
- FVGDetectionStrategy & FVGRetestStrategy (Bullish/Bearish FVG, mitigation, retest)
- PremiumDiscountAnalyzer (Discount, Premium, Equilibrium)
- PreviousDayLevelsAnalyzer (PDH, PDL, PDC, sweeps)
- KillZoneSessionAnalyzer (Asian, London, NY, Overlap, Kill Zones, DST awareness)
"""

from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest

from atlas_ai.domain.signals import SignalDirection
from atlas_ai.strategies.config import KillZoneSessionConfig
from atlas_ai.strategies.fvg import FVGDetectionStrategy, FVGRetestStrategy, scan_historical_fvgs
from atlas_ai.strategies.liquidity import LiquiditySweepStrategy
from atlas_ai.strategies.market_structure import MarketStructureStrategy, detect_confirmed_swings
from atlas_ai.strategies.models import PriceZoneClassification, StructureEventType
from atlas_ai.strategies.premium_discount import PremiumDiscountAnalyzer
from atlas_ai.strategies.sessions import KillZoneSessionAnalyzer, PreviousDayLevelsAnalyzer


def test_market_structure_bos_choch_and_causal_swing_confirmation() -> None:
    """Verify causal swing confirmation (right_bars=2) and Bullish BOS detection."""
    # Construct a 10-bar series with a clear swing low at index 2 and swing high at index 5,
    # followed by a breakout close above the swing high on the final bar (index 9).
    rows = [
        # idx, open, high, low, close
        (2650, 2652, 2648, 2650),  # 0
        (2650, 2651, 2645, 2646),  # 1
        (2646, 2648, 2640, 2647),  # 2 -> Swing Low (2640), confirmed at idx 4
        (2647, 2654, 2646, 2653),  # 3
        (2653, 2658, 2650, 2657),  # 4
        (2657, 2665, 2655, 2660),  # 5 -> Swing High (2665), confirmed at idx 7
        (2660, 2661, 2652, 2654),  # 6
        (2654, 2658, 2650, 2656),  # 7 (Swing High 2665 is now confirmed!)
        (2656, 2663, 2655, 2662),  # 8
        (2662, 2672, 2661, 2670),  # 9 -> Closes at 2670 > 2665 (Bullish BOS!)
    ]
    ts = pd.date_range("2026-01-05 00:00:00", periods=len(rows), freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [1000.0] * len(rows),
        }
    )

    # At index 6 (only 1 bar after pivot index 5), Swing High at index 5 must NOT be confirmed yet!
    sh_at_6, _ = detect_confirmed_swings(df.iloc[:7], left_bars=2, right_bars=2)
    assert len(sh_at_6) == 0

    # At index 7 (2 bars after pivot index 5), Swing High at index 5 IS confirmed
    sh_at_7, sl_at_7 = detect_confirmed_swings(df.iloc[:8], left_bars=2, right_bars=2)
    assert len(sh_at_7) == 1
    assert sh_at_7[0].price == Decimal("2665.00")
    assert len(sl_at_7) == 1
    assert sl_at_7[0].price == Decimal("2640.00")

    # Evaluate MarketStructureStrategy at bar 9 -> Bullish BOS
    strat = MarketStructureStrategy()
    res = strat.evaluate(df)
    assert res.direction == SignalDirection.BUY
    assert res.supporting_evidence["structure_event"] == StructureEventType.BULLISH_BOS.value
    assert res.stop_loss is not None and res.entry_price is not None and res.take_profit is not None
    assert res.stop_loss < res.entry_price < res.take_profit


def test_liquidity_equal_lows_sweep_and_rejection() -> None:
    """Verify Equal Lows (EQL) detection and Bullish sweep + close back above level."""
    # Two swing lows at 2640.00 (idx 2 and idx 7), confirmed by idx 9, then swept at idx 10 (low=2638.50, close=2643.00)
    rows = [
        (2650, 2655, 2648, 2652),  # 0
        (2652, 2653, 2645, 2646),  # 1
        (2646, 2649, 2640, 2648),  # 2 -> Swing Low #1 at 2640.00
        (2648, 2656, 2646, 2654),  # 3
        (2654, 2660, 2650, 2655),  # 4 -> Swing High at 2660.00
        (2655, 2656, 2647, 2648),  # 5
        (2648, 2650, 2644, 2645),  # 6
        (2645, 2649, 2640.1, 2647),  # 7 -> Swing Low #2 at 2640.10 (EQL with 2640.00!)
        (2647, 2653, 2645, 2651),  # 8
        (2651, 2655, 2646, 2652),  # 9 (Swing Low #2 is now confirmed!)
        (2642, 2646, 2638.5, 2644),  # 10 -> Sweeps below 2640.00 (low=2638.5) and closes bullish at 2644.0!
    ]
    ts = pd.date_range("2026-01-05 00:00:00", periods=len(rows), freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [1000.0] * len(rows),
        }
    )

    strat = LiquiditySweepStrategy()
    res = strat.evaluate(df)
    assert res.direction == SignalDirection.BUY
    assert res.supporting_evidence["swept_pool_type"] in ("EQL", "SSL")
    assert len(res.supporting_evidence["eql_levels"]) >= 1
    assert res.stop_loss is not None and res.entry_price is not None
    assert res.stop_loss < Decimal("2638.50") < res.entry_price


def test_fvg_detection_mitigation_and_retest() -> None:
    """Verify Bullish FVG detection (low[2] > high[0]), retest entry, and full mitigation tracking."""
    # Bar 0: high = 2650.0
    # Bar 1: large displacement candle (open=2650, high=2665, low=2649, close=2664)
    # Bar 2: low = 2656.0 > Bar 0 high (2650.0) -> Bullish FVG [2650.0, 2656.0] formed at Bar 2!
    # Bar 3: price dips to 2653.0 (inside [2650, 2656]) and closes bullish at 2660.0 (open=2655.0) -> FVG Retest BUY!
    # Bar 4: price crashes to 2648.0 (<= 2650.0) -> FVG is now mitigated!
    rows = [
        (2645.0, 2650.0, 2644.0, 2649.0),  # 0
        (2649.0, 2665.0, 2648.0, 2664.0),  # 1
        (2664.0, 2668.0, 2656.0, 2666.0),  # 2 -> Forms Bullish FVG [2650.0, 2656.0]
        (2655.0, 2662.0, 2653.0, 2660.0),  # 3 -> Retests FVG (low=2653.0) and closes bullish (2660 > 2655)
        (2660.0, 2661.0, 2648.0, 2649.0),  # 4 -> Mitigates FVG (low=2648 <= 2650)
    ]
    ts = pd.date_range("2026-01-05 00:00:00", periods=len(rows), freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [1000.0] * len(rows),
        }
    )

    fvg_det = FVGDetectionStrategy()
    # At index 2, new Bullish FVG is detected
    det_res = fvg_det.evaluate(df, as_of_index=2)
    assert det_res.direction == SignalDirection.BUY
    assert det_res.supporting_evidence["newly_formed_on_current_bar"] is True

    # At index 3, FVGRetestStrategy triggers BUY
    fvg_ret = FVGRetestStrategy()
    ret_res = fvg_ret.evaluate(df, as_of_index=3)
    assert ret_res.direction == SignalDirection.BUY
    assert ret_res.stop_loss is not None and ret_res.stop_loss < Decimal("2650.00")

    # At index 4, FVG is mitigated
    all_fvgs = scan_historical_fvgs(df)
    bull_fvgs = [g for g in all_fvgs if g.direction == SignalDirection.BUY]
    assert len(bull_fvgs) == 1
    assert bull_fvgs[0].mitigated is True
    assert bull_fvgs[0].is_valid is False


def test_premium_discount_analyzer_zones() -> None:
    """Verify Discount (<45%), Equilibrium (45%-55%), and Premium (>55%) dealing range classifications."""
    # Swing Low at idx 2 (2600.0), Swing High at idx 5 (2700.0), confirmed at idx 7. Equilibrium = 2650.0.
    base_rows = [
        (2620, 2625, 2610, 2615),  # 0
        (2615, 2618, 2605, 2608),  # 1
        (2608, 2612, 2600, 2610),  # 2 -> Swing Low = 2600.0
        (2610, 2650, 2609, 2645),  # 3
        (2645, 2680, 2640, 2675),  # 4
        (2675, 2700, 2670, 2690),  # 5 -> Swing High = 2700.0
        (2690, 2692, 2660, 2665),  # 6
        (2665, 2670, 2625, 2630),  # 7 -> Close = 2630.0 (30% of [2600, 2700] -> DISCOUNT)
        (2630, 2655, 2628, 2650),  # 8 -> Close = 2650.0 (50% of [2600, 2700] -> EQUILIBRIUM)
        (2650, 2685, 2648, 2680),  # 9 -> Close = 2680.0 (80% of [2600, 2700] -> PREMIUM)
    ]
    ts = pd.date_range("2026-01-05 00:00:00", periods=len(base_rows), freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [r[0] for r in base_rows],
            "high": [r[1] for r in base_rows],
            "low": [r[2] for r in base_rows],
            "close": [r[3] for r in base_rows],
            "volume": [1000.0] * len(base_rows),
        }
    )

    analyzer = PremiumDiscountAnalyzer()
    state_disc, res_disc = analyzer.analyze_state(df, as_of_index=7)
    assert state_disc.zone == PriceZoneClassification.DISCOUNT
    assert res_disc.direction == SignalDirection.BUY

    state_eq, res_eq = analyzer.analyze_state(df, as_of_index=8)
    assert state_eq.zone == PriceZoneClassification.EQUILIBRIUM
    assert res_eq.direction == SignalDirection.WAIT

    state_prem, res_prem = analyzer.analyze_state(df, as_of_index=9)
    assert state_prem.zone == PriceZoneClassification.PREMIUM
    assert res_prem.direction == SignalDirection.SELL


def test_pdh_pdl_analyzer_and_killzone_dst_awareness() -> None:
    """Verify Previous Day High/Low calculation across day boundaries and DST-aware Kill Zone detection."""
    # Day 1 (2026-01-05 UTC): High=2660, Low=2640, Close=2650
    # Day 2 (2026-01-06 UTC): Sweeps below PDL (2640) with low=2638 and closes at 2645 -> BUY context!
    ts = [
        pd.Timestamp("2026-01-05 10:00:00", tz="UTC"),
        pd.Timestamp("2026-01-05 14:00:00", tz="UTC"),
        pd.Timestamp("2026-01-06 08:00:00", tz="UTC"),  # 03:00 EST (London Kill Zone!)
    ]
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [2645.0, 2655.0, 2642.0],
            "high": [2660.0, 2658.0, 2648.0],
            "low": [2640.0, 2648.0, 2638.0],
            "close": [2655.0, 2650.0, 2645.0],
            "volume": [1000.0, 1000.0, 1200.0],
        }
    )

    pdh_analyzer = PreviousDayLevelsAnalyzer()
    pd_state, pd_res = pdh_analyzer.analyze_levels(df)
    assert pd_state.pdh == Decimal("2660.00")
    assert pd_state.pdl == Decimal("2640.00")
    assert pd_state.pdc == Decimal("2650.00")
    assert pd_state.pdl_swept_and_rejected is True
    assert pd_res.direction == SignalDirection.BUY

    # Verify DST-aware KillZoneSessionAnalyzer in America/New_York:
    # Winter (EST, UTC-5): 08:00 UTC == 03:00 EST -> Inside London Kill Zone (02:00-05:00 NY)
    # Summer (EDT, UTC-4): 08:00 UTC == 04:00 EDT -> Inside London Kill Zone (02:00-05:00 NY), with is_dst=True
    kz_analyzer = KillZoneSessionAnalyzer(KillZoneSessionConfig(timezone="America/New_York"))
    winter_info = kz_analyzer.classify_bar_time(datetime(2026, 1, 6, 8, 0, tzinfo=timezone.utc))
    assert winter_info["is_dst"] is False
    assert winter_info["local_hour"] == 3
    assert winter_info["in_london_killzone"] is True

    summer_info = kz_analyzer.classify_bar_time(datetime(2026, 7, 6, 8, 0, tzinfo=timezone.utc))
    assert summer_info["is_dst"] is True
    assert summer_info["local_hour"] == 4
    assert summer_info["in_london_killzone"] is True
