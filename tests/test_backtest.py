"""Comprehensive Phase 4 unit & integration tests for the Historical Backtesting Engine."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import math

import pandas as pd
import pytest

from atlas_ai.backtest import (
    BacktestConfig,
    BacktestEngine,
    BacktestTradeRecord,
    EquityPoint,
    ExecutionSimulator,
    ExitReason,
    IntrabarTieBreakPolicy,
    PortfolioTracker,
    calculate_backtest_metrics,
    default_xauusd_instrument_spec,
)
from atlas_ai.domain.market import Timeframe
from atlas_ai.domain.risk import AccountState, RiskDecision, RiskParameters, RiskRejectionCode
from atlas_ai.domain.signals import SignalDirection, TradeSignal
from atlas_ai.strategies.config import ConfluenceWeightsConfig, SignalLifecycleConfig
from atlas_ai.strategies.engine import StrategyEngine
from atlas_ai.strategies.models import SignalLifecycleState


def _make_ohlcv_df(
    bars: list[tuple[datetime, float, float, float, float, float]],
) -> pd.DataFrame:
    """Build a chronological OHLCV DataFrame from `(timestamp, open, high, low, close, volume)` tuples."""
    return pd.DataFrame(
        bars,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )


def _make_wait_signal(ts: datetime, tf: Timeframe = Timeframe.H1) -> TradeSignal:
    return TradeSignal(
        symbol="XAUUSD",
        direction=SignalDirection.WAIT,
        timeframe=tf,
        strategy_id="test_provider",
        confidence=Decimal("0.00"),
        timestamp=ts,
        rationale="Explicit WAIT signal.",
    )


# ---------------------------------------------------------------------------
# 1 & 2. Spread, Slippage, Commission, BUY/SELL Fill Correctness & Post-Cost RR
# ---------------------------------------------------------------------------
def test_execution_simulator_spread_slippage_commission_and_buy_sell_fills() -> None:
    """Verify exact Decimal BUY/SELL entry & exit fills, spread, slippage, commission, and post-cost RR checks."""
    cfg = BacktestConfig(
        symbol="XAUUSD",
        timeframe=Timeframe.H1,
        spread_points=Decimal("20"),      # 20 * 0.01 = $0.20 spread ($0.10 half-spread)
        slippage_points=Decimal("5"),     # 5 * 0.01 = $0.05 slippage
        commission_per_lot=Decimal("7.00"),
        enforce_min_rr_after_costs=True,
        risk_parameters=RiskParameters(minimum_risk_reward_ratio=Decimal("1.5")),
    )
    sim = ExecutionSimulator(cfg)

    # Half-spread ($0.10) + slippage ($0.05) = $0.15 drag per fill
    assert sim.spread_price == Decimal("0.20")
    assert sim.half_spread_price == Decimal("0.10")
    assert sim.slippage_price == Decimal("0.05")

    # BUY entry fills higher (Ask + slippage); BUY exit fills lower (Bid - slippage)
    assert sim.calculate_entry_fill_price(SignalDirection.BUY, Decimal("2600.00")) == Decimal("2600.15")
    assert sim.calculate_exit_fill_price(SignalDirection.BUY, Decimal("2620.00")) == Decimal("2619.85")

    # SELL entry fills lower (Bid - slippage); SELL exit fills higher (Ask + slippage)
    assert sim.calculate_entry_fill_price(SignalDirection.SELL, Decimal("2600.00")) == Decimal("2599.85")
    assert sim.calculate_exit_fill_price(SignalDirection.SELL, Decimal("2580.00")) == Decimal("2580.15")

    # Commission for 0.25 lots = 0.25 * $7.00 = $1.75
    assert sim.calculate_commission(Decimal("0.25")) == Decimal("1.75")

    ts = datetime(2026, 3, 2, 14, 0, tzinfo=timezone.utc)
    buy_sig = TradeSignal(
        symbol="XAUUSD",
        direction=SignalDirection.BUY,
        entry_price=Decimal("2600.00"),
        stop_loss=Decimal("2590.00"),
        take_profit=Decimal("2620.00"),
        timeframe=Timeframe.H1,
        strategy_id="test_buy",
        confidence=Decimal("0.85"),
        timestamp=ts,
        rationale="Valid BUY setup.",
    )
    approved_decision = RiskDecision(
        approved=True,
        calculated_risk_amount=Decimal("100.00"),
        calculated_position_size=Decimal("0.10"),
        risk_reward_ratio=Decimal("2.0000"),
        risk_percent=Decimal("1.0000"),
    )

    eval_res = sim.open_position(
        position_id="pos-00001",
        signal_id="sig-00001",
        signal=buy_sig,
        risk_decision=approved_decision,
        entry_bar_index=5,
        entry_timestamp=ts,
    )
    assert eval_res.approved is True
    assert eval_res.position is not None
    pos = eval_res.position
    assert pos.entry_fill_price == Decimal("2600.15")
    assert pos.effective_stop_distance == Decimal("10.15")
    assert pos.effective_reward_distance == Decimal("19.85")
    assert pos.commission == Decimal("0.70")

    # Close at target (2620.00 -> exit fill 2619.85 -> +19.70 points = 1970 ticks * $1 * 0.10 lots = $197.00 gross)
    trade = sim.close_position(
        pos,
        trade_id="trd-00001",
        exit_bar_index=8,
        exit_timestamp=ts + timedelta(hours=3),
        requested_exit_price=Decimal("2620.00"),
        exit_reason=ExitReason.TAKE_PROFIT,
    )
    assert trade.exit_fill_price == Decimal("2619.85")
    assert trade.gross_pnl == Decimal("197.00")
    assert trade.commission == Decimal("0.70")
    assert trade.net_pnl == Decimal("196.30")
    assert trade.bars_held == 3

    # Verify post-cost effective RR rejection when spread/slippage degrades a tight setup below minimum RR
    tight_sig = TradeSignal(
        symbol="XAUUSD",
        direction=SignalDirection.BUY,
        entry_price=Decimal("2600.00"),
        stop_loss=Decimal("2599.50"),   # $0.50 stop -> after $0.15 drag, eff stop = $0.65
        take_profit=Decimal("2600.76"), # $0.76 reward -> after $0.15 drag, eff reward = $0.61 (RR = 0.9384 < 1.5)
        timeframe=Timeframe.H1,
        strategy_id="tight_buy",
        confidence=Decimal("0.80"),
        timestamp=ts,
        rationale="Tight setup degraded by spread/slippage.",
    )
    rejected_fill = sim.open_position(
        position_id="pos-00002",
        signal_id="sig-00002",
        signal=tight_sig,
        risk_decision=approved_decision,
        entry_bar_index=6,
        entry_timestamp=ts,
    )
    assert rejected_fill.approved is False
    assert rejected_fill.rejection_reason is not None
    assert "EXECUTION_INSUFFICIENT_EFFECTIVE_RR" in rejected_fill.rejection_reason


# ---------------------------------------------------------------------------
# 3. Intrabar SL/TP Detection, Conservative SL Priority & No Same-Bar Execution
# ---------------------------------------------------------------------------
def test_intrabar_sl_tp_priority_and_no_same_candle_hindsight() -> None:
    """Verify conservative STOP_LOSS priority when a bar touches both SL & TP, and no same-bar execution."""
    cfg_conservative = BacktestConfig(
        spread_points=Decimal("0"),
        slippage_points=Decimal("0"),
        commission_per_lot=Decimal("0"),
        intrabar_tie_break=IntrabarTieBreakPolicy.CONSERVATIVE_SL_FIRST,
    )
    sim_cons = ExecutionSimulator(cfg_conservative)
    ts = datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc)

    sig = TradeSignal(
        symbol="XAUUSD",
        direction=SignalDirection.BUY,
        entry_price=Decimal("2600.00"),
        stop_loss=Decimal("2590.00"),
        take_profit=Decimal("2620.00"),
        timeframe=Timeframe.H1,
        strategy_id="test_tie",
        confidence=Decimal("0.85"),
        timestamp=ts,
        rationale="Tie-break test.",
    )
    dec = RiskDecision(
        approved=True,
        calculated_risk_amount=Decimal("100.00"),
        calculated_position_size=Decimal("0.10"),
        risk_reward_ratio=Decimal("2.0000"),
        risk_percent=Decimal("1.0000"),
    )
    pos = sim_cons.open_position(
        position_id="pos-1",
        signal_id="sig-1",
        signal=sig,
        risk_decision=dec,
        entry_bar_index=10,
        entry_timestamp=ts,
    ).position
    assert pos is not None

    # Same bar (bar_index == 10) must NEVER trigger SL or TP even if High/Low span both
    same_bar_pos, same_bar_trade = sim_cons.evaluate_bar_for_position(
        pos,
        bar_index=10,
        bar_timestamp=ts,
        open_price=Decimal("2600.00"),
        high_price=Decimal("2650.00"),
        low_price=Decimal("2550.00"),
        close_price=Decimal("2600.00"),
        trade_id_Trigger="trd-same",
    )
    assert same_bar_trade is None
    assert same_bar_pos == pos

    # Next bar (bar_index == 11) spans both SL (2590) and TP (2620) -> Conservative policy chooses STOP_LOSS
    _, closed_cons = sim_cons.evaluate_bar_for_position(
        pos,
        bar_index=11,
        bar_timestamp=ts + timedelta(hours=1),
        open_price=Decimal("2600.00"),
        high_price=Decimal("2625.00"),
        low_price=Decimal("2585.00"),
        close_price=Decimal("2605.00"),
        trade_id_Trigger="trd-cons",
    )
    assert closed_cons is not None
    assert closed_cons.exit_reason == ExitReason.STOP_LOSS
    assert closed_cons.exit_fill_price == Decimal("2590.00")
    assert closed_cons.net_pnl == Decimal("-100.00")
    assert closed_cons.mfe_price_distance == Decimal("25.00")
    assert closed_cons.mae_price_distance == Decimal("15.00")


# ---------------------------------------------------------------------------
# 4, 5 & 6. Portfolio Accounting, Margin Reservation/Release & 5 PM NY Rollover
# ---------------------------------------------------------------------------
def test_portfolio_accounting_margin_and_ny_5pm_dst_rollover() -> None:
    """Verify balance, floating PnL, margin reservation/release, and 5 PM NY trading-day rollover across DST."""
    cfg = BacktestConfig(
        initial_balance=Decimal("10000.00"),
        spread_points=Decimal("0"),
        slippage_points=Decimal("0"),
        commission_per_lot=Decimal("0"),
        day_boundary_timezone="America/New_York",
        day_rollover_hour=17,
    )
    tracker = PortfolioTracker(cfg)
    sim = ExecutionSimulator(cfg)

    # 2026-03-09 is the first Monday of US Daylight Saving Time (EDT = UTC-4).
    # 16:00 EDT = 20:00 UTC (still trading day 2026-03-09)
    # 17:00 EDT = 21:00 UTC (rolls over to trading day 2026-03-10!)
    ts_before_rollover = datetime(2026, 3, 9, 20, 0, tzinfo=timezone.utc)
    ts_at_rollover = datetime(2026, 3, 9, 21, 0, tzinfo=timezone.utc)

    rolled_1 = tracker.update_trading_day_rollover(ts_before_rollover)
    assert rolled_1 is False
    assert str(tracker.current_trading_day) == "2026-03-09"

    # Open a 0.20 lot BUY position at 2600.00 (margin = 0.20 * 100 * 2600 * 0.01 = $520.00)
    sig = TradeSignal(
        symbol="XAUUSD",
        direction=SignalDirection.BUY,
        entry_price=Decimal("2600.00"),
        stop_loss=Decimal("2590.00"),
        take_profit=Decimal("2620.00"),
        timeframe=Timeframe.H1,
        strategy_id="test_margin",
        confidence=Decimal("0.90"),
        timestamp=ts_before_rollover,
        rationale="Margin test.",
    )
    dec = RiskDecision(
        approved=True,
        calculated_risk_amount=Decimal("200.00"),
        calculated_position_size=Decimal("0.20"),
        risk_reward_ratio=Decimal("2.0000"),
        risk_percent=Decimal("2.0000"),
    )
    pos = sim.open_position(
        position_id="pos-m1",
        signal_id="sig-m1",
        signal=sig,
        risk_decision=dec,
        entry_bar_index=0,
        entry_timestamp=ts_before_rollover,
    ).position
    assert pos is not None
    tracker.add_open_position(pos)

    assert tracker.used_margin == Decimal("520.00")
    assert tracker.free_margin == Decimal("9480.00")
    assert tracker.open_positions_count == 1

    # Mark-to-market at 2595.00 (-$100.00 floating loss) before rollover
    updated_pos = pos.model_copy(update={"floating_pnl": Decimal("-100.00")})
    tracker.update_open_position(updated_pos)
    assert tracker.equity == Decimal("9900.00")
    assert tracker.current_daily_pnl == Decimal("-100.00")

    # Close the losing trade before 5 PM NY rollover
    closed_tr = sim.close_position(
        updated_pos,
        trade_id="trd-m1",
        exit_bar_index=1,
        exit_timestamp=ts_before_rollover + timedelta(minutes=30),
        requested_exit_price=Decimal("2595.00"),
        exit_reason=ExitReason.MANUAL_CANCEL,
    )
    tracker.record_closed_trade(closed_tr)
    assert tracker.used_margin == Decimal("0.00")
    assert tracker.open_positions_count == 0
    assert tracker.balance == Decimal("9900.00")
    assert tracker.free_margin == Decimal("9900.00")
    assert tracker.current_daily_pnl == Decimal("-100.00")

    # Now cross 5 PM EDT (21:00 UTC) -> Trading day rolls over to 2026-03-10
    rolled_2 = tracker.update_trading_day_rollover(ts_at_rollover)
    assert rolled_2 is True
    assert str(tracker.current_trading_day) == "2026-03-10"
    assert tracker.daily_starting_balance == Decimal("9900.00")
    assert tracker.current_daily_pnl == Decimal("0.00")
    assert tracker.peak_balance == Decimal("10000.00")


# ---------------------------------------------------------------------------
# 7, 8 & 9. Risk Engine Integration: Daily Loss, Max Drawdown & Max Positions
# ---------------------------------------------------------------------------
def test_backtest_enforces_daily_loss_limit_and_resets_next_trading_day() -> None:
    """Verify RiskEngine halts new entries once daily loss limit is reached and resumes next trading day."""
    # 1.0% risk per trade, 2.0% max daily loss -> after 2 consecutive 1R (-$100) losses on the same day,
    # the 3rd signal on that day must be rejected with DAILY_LOSS_LIMIT_EXCEEDED.
    # On the next trading day (after 5 PM NY rollover), a new trade is approved again.
    cfg = BacktestConfig(
        initial_balance=Decimal("10000.00"),
        spread_points=Decimal("0"),
        slippage_points=Decimal("0"),
        commission_per_lot=Decimal("0"),
        risk_parameters=RiskParameters(
            risk_per_trade_percent=Decimal("1.0"),
            max_daily_loss_percent=Decimal("2.0"),
            max_total_drawdown_percent=Decimal("10.0"),
            max_open_positions=1,
            minimum_risk_reward_ratio=Decimal("1.5"),
        ),
    )
    engine = BacktestEngine(cfg)

    # Construct 8 H1 bars starting 2026-01-05 12:00 UTC (07:00 EST):
    # Bar 0 (12:00 UTC): Signal 1 BUY @ 2600, SL=2590, TP=2620
    # Bar 1 (13:00 UTC): Low hits 2588 -> Trade 1 closes at SL (-$100, -1%)
    # Bar 2 (14:00 UTC): Signal 2 BUY @ 2600, SL=2590, TP=2620
    # Bar 3 (15:00 UTC): Low hits 2588 -> Trade 2 closes at SL (-$99, total daily loss -$199)
    # Bar 4 (16:00 UTC): Signal 3 BUY @ 2600 -> Projected daily loss > $200 (2% of $10,000) -> REJECTED!
    # Bar 5 (22:00 UTC = 17:00 EST): 5 PM NY Rollover! Signal 4 BUY @ 2600 -> APPROVED!
    # Bar 6 (23:00 UTC): High hits 2625 -> Trade 3 closes at TP!
    t0 = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
    bars = [
        (t0, 2600.0, 2602.0, 2598.0, 2600.0, 100.0),
        (t0 + timedelta(hours=1), 2600.0, 2601.0, 2588.0, 2595.0, 100.0),
        (t0 + timedelta(hours=2), 2595.0, 2602.0, 2594.0, 2600.0, 100.0),
        (t0 + timedelta(hours=3), 2600.0, 2601.0, 2588.0, 2595.0, 100.0),
        (t0 + timedelta(hours=4), 2595.0, 2602.0, 2594.0, 2600.0, 100.0),
        (datetime(2026, 1, 5, 22, 0, tzinfo=timezone.utc), 2600.0, 2602.0, 2598.0, 2600.0, 100.0),
        (datetime(2026, 1, 5, 23, 0, tzinfo=timezone.utc), 2600.0, 2625.0, 2599.0, 2622.0, 100.0),
    ]
    df = _make_ohlcv_df(bars)

    def provider(causal_df: pd.DataFrame, idx: int, account: AccountState) -> TradeSignal:
        ts = causal_df.iloc[-1]["timestamp"].to_pydatetime()
        if idx in (0, 2, 4, 5):
            return TradeSignal(
                symbol="XAUUSD",
                direction=SignalDirection.BUY,
                entry_price=Decimal("2600.00"),
                stop_loss=Decimal("2590.00"),
                take_profit=Decimal("2620.00"),
                timeframe=Timeframe.H1,
                strategy_id="daily_loss_test",
                confidence=Decimal("0.85"),
                timestamp=ts,
                rationale=f"Test entry at bar {idx}",
            )
        return _make_wait_signal(ts)

    res = engine.run(df, signal_provider=provider)
    assert len(res.trades) == 3
    assert res.trades[0].exit_reason == ExitReason.STOP_LOSS
    assert res.trades[1].exit_reason == ExitReason.STOP_LOSS
    assert res.trades[2].exit_reason == ExitReason.TAKE_PROFIT

    # Bar 4 signal was rejected due to DAILY_LOSS_LIMIT_EXCEEDED
    assert len(res.risk_rejections) == 1
    assert res.risk_rejections[0].bar_index == 4
    assert RiskRejectionCode.DAILY_LOSS_LIMIT_EXCEEDED.value in res.risk_rejections[0].rejection_reason


def test_backtest_enforces_max_total_drawdown_and_max_open_positions() -> None:
    """Verify MAX_OPEN_POSITIONS_REACHED and multi-day TOTAL_DRAWDOWN_LIMIT_EXCEEDED during chronological simulation."""
    cfg = BacktestConfig(
        initial_balance=Decimal("10000.00"),
        spread_points=Decimal("0"),
        slippage_points=Decimal("0"),
        commission_per_lot=Decimal("0"),
        risk_parameters=RiskParameters(
            risk_per_trade_percent=Decimal("2.0"),
            max_daily_loss_percent=Decimal("3.5"),
            max_total_drawdown_percent=Decimal("4.0"),
            max_open_positions=1,
            minimum_risk_reward_ratio=Decimal("1.5"),
        ),
    )
    engine = BacktestEngine(cfg)
    t_day1 = datetime(2026, 1, 6, 10, 0, tzinfo=timezone.utc)
    t_day2 = datetime(2026, 1, 7, 10, 0, tzinfo=timezone.utc)
    t_day3 = datetime(2026, 1, 8, 10, 0, tzinfo=timezone.utc)
    bars = [
        # Bar 0 (Day 1): Open Trade 1 @ 2600 (2% risk = $200)
        (t_day1, 2600.0, 2601.0, 2599.0, 2600.0, 100.0),
        # Bar 1 (Day 1): Price stays at 2600 (Trade 1 still open); attempt second signal -> MAX_OPEN_POSITIONS_REACHED
        (t_day1 + timedelta(hours=1), 2600.0, 2601.0, 2599.0, 2600.0, 100.0),
        # Bar 2 (Day 1): Low hits 2585 -> Trade 1 stops out (-$200, DD = 2.0%)
        (t_day1 + timedelta(hours=2), 2600.0, 2601.0, 2585.0, 2600.0, 100.0),
        # Bar 3 (Day 2, after 5 PM NY rollover): Daily loss is 0%; Open Trade 2 @ 2600 (2% risk = $196)
        (t_day2, 2600.0, 2601.0, 2599.0, 2600.0, 100.0),
        # Bar 4 (Day 2): Low hits 2585 -> Trade 2 stops out (-$196, cumulative DD = $396 = 3.96%)
        (t_day2 + timedelta(hours=1), 2600.0, 2601.0, 2585.0, 2600.0, 100.0),
        # Bar 5 (Day 3): Daily loss is 0%, but cumulative DD = 3.96%; new 2% trade -> TOTAL_DRAWDOWN_LIMIT_EXCEEDED!
        (t_day3, 2600.0, 2601.0, 2599.0, 2600.0, 100.0),
        (t_day3 + timedelta(hours=1), 2600.0, 2601.0, 2599.0, 2600.0, 100.0),
    ]
    df = _make_ohlcv_df(bars)

    def provider(causal_df: pd.DataFrame, idx: int, account: AccountState) -> TradeSignal:
        ts = causal_df.iloc[-1]["timestamp"].to_pydatetime()
        close_val = Decimal(str(causal_df.iloc[-1]["close"]))
        if idx == 1:
            # Distinct setup (SELL while Trade 1 BUY is still open) -> triggers MAX_OPEN_POSITIONS_REACHED
            return TradeSignal(
                symbol="XAUUSD",
                direction=SignalDirection.SELL,
                entry_price=close_val,
                stop_loss=close_val + Decimal("10.00"),
                take_profit=close_val - Decimal("20.00"),
                timeframe=Timeframe.H1,
                strategy_id="dd_and_pos_test",
                confidence=Decimal("0.85"),
                timestamp=ts,
                rationale="Second position attempt while max_open_positions=1 is full.",
            )
        if idx in (0, 3, 5):
            return TradeSignal(
                symbol="XAUUSD",
                direction=SignalDirection.BUY,
                entry_price=close_val,
                stop_loss=close_val - Decimal("10.00"),
                take_profit=close_val + Decimal("20.00"),
                timeframe=Timeframe.H1,
                strategy_id="dd_and_pos_test",
                confidence=Decimal("0.85"),
                timestamp=ts,
                rationale=f"Signal at bar {idx}",
            )
        return _make_wait_signal(ts)

    res = engine.run(df, signal_provider=provider)
    assert len(res.trades) == 2
    rejection_codes = [r.rejection_reason for r in res.risk_rejections]
    assert any(RiskRejectionCode.MAX_OPEN_POSITIONS_REACHED.value in c for c in rejection_codes)
    assert any(RiskRejectionCode.TOTAL_DRAWDOWN_LIMIT_EXCEEDED.value in c for c in rejection_codes)


# ---------------------------------------------------------------------------
# 10 & 11. Performance Metrics Accuracy & Edge Cases (0 trades, 0 wins, 0 losses)
# ---------------------------------------------------------------------------
def test_performance_metrics_accuracy_and_edge_cases() -> None:
    """Verify exact metric calculations and safe finite edge-case handling (no NaN or Infinity)."""
    t0 = datetime(2026, 2, 1, 10, 0, tzinfo=timezone.utc)

    # Edge case 1: Zero trades
    m_zero = calculate_backtest_metrics(
        initial_balance=Decimal("10000.00"),
        final_balance=Decimal("10000.00"),
        final_equity=Decimal("10000.00"),
        trades=(),
        equity_curve=(),
    )
    assert m_zero.total_trades == 0
    assert m_zero.win_rate_pct == Decimal("0.00")
    assert m_zero.profit_factor == Decimal("0.0000")
    assert m_zero.expectancy_usd == Decimal("0.00")
    assert m_zero.sharpe_ratio == Decimal("0.0000")
    assert m_zero.sortino_ratio == Decimal("0.0000")

    # Known 3-trade sequence:
    # Trade 1: Win +$200 gross, $10 comm -> +$190 net (+1.90R on $100 risk)
    # Trade 2: Loss -$100 gross, $10 comm -> -$110 net (-1.10R on $100 risk)
    # Trade 3: Win +$300 gross, $10 comm -> +$290 net (+2.90R on $100 risk)
    trades = [
        BacktestTradeRecord(
            trade_id="t1",
            position_id="p1",
            signal_id="s1",
            symbol="XAUUSD",
            timeframe=Timeframe.H1,
            direction=SignalDirection.BUY,
            strategy_id="strat",
            entry_bar_index=0,
            exit_bar_index=1,
            entry_timestamp=t0,
            exit_timestamp=t0 + timedelta(hours=1),
            requested_entry_price=Decimal("2600"),
            entry_fill_price=Decimal("2600"),
            requested_exit_price=Decimal("2620"),
            exit_fill_price=Decimal("2620"),
            stop_loss=Decimal("2590"),
            take_profit=Decimal("2620"),
            lots=Decimal("0.10"),
            initial_risk_amount=Decimal("100.00"),
            gross_pnl=Decimal("200.00"),
            commission=Decimal("10.00"),
            net_pnl=Decimal("190.00"),
            realized_r_multiple=Decimal("1.9000"),
            mfe_price_distance=Decimal("20"),
            mae_price_distance=Decimal("2"),
            mfe_usd=Decimal("200"),
            mae_usd=Decimal("20"),
            bars_held=1,
            exit_reason=ExitReason.TAKE_PROFIT,
        ),
        BacktestTradeRecord(
            trade_id="t2",
            position_id="p2",
            signal_id="s2",
            symbol="XAUUSD",
            timeframe=Timeframe.H1,
            direction=SignalDirection.SELL,
            strategy_id="strat",
            entry_bar_index=2,
            exit_bar_index=3,
            entry_timestamp=t0 + timedelta(hours=2),
            exit_timestamp=t0 + timedelta(hours=3),
            requested_entry_price=Decimal("2620"),
            entry_fill_price=Decimal("2620"),
            requested_exit_price=Decimal("2630"),
            exit_fill_price=Decimal("2630"),
            stop_loss=Decimal("2630"),
            take_profit=Decimal("2600"),
            lots=Decimal("0.10"),
            initial_risk_amount=Decimal("100.00"),
            gross_pnl=Decimal("-100.00"),
            commission=Decimal("10.00"),
            net_pnl=Decimal("-110.00"),
            realized_r_multiple=Decimal("-1.1000"),
            mfe_price_distance=Decimal("3"),
            mae_price_distance=Decimal("10"),
            mfe_usd=Decimal("30"),
            mae_usd=Decimal("100"),
            bars_held=1,
            exit_reason=ExitReason.STOP_LOSS,
        ),
        BacktestTradeRecord(
            trade_id="t3",
            position_id="p3",
            signal_id="s3",
            symbol="XAUUSD",
            timeframe=Timeframe.H1,
            direction=SignalDirection.BUY,
            strategy_id="strat",
            entry_bar_index=4,
            exit_bar_index=5,
            entry_timestamp=t0 + timedelta(hours=4),
            exit_timestamp=t0 + timedelta(hours=5),
            requested_entry_price=Decimal("2600"),
            entry_fill_price=Decimal("2600"),
            requested_exit_price=Decimal("2630"),
            exit_fill_price=Decimal("2630"),
            stop_loss=Decimal("2590"),
            take_profit=Decimal("2630"),
            lots=Decimal("0.10"),
            initial_risk_amount=Decimal("100.00"),
            gross_pnl=Decimal("300.00"),
            commission=Decimal("10.00"),
            net_pnl=Decimal("290.00"),
            realized_r_multiple=Decimal("2.9000"),
            mfe_price_distance=Decimal("30"),
            mae_price_distance=Decimal("1"),
            mfe_usd=Decimal("300"),
            mae_usd=Decimal("10"),
            bars_held=1,
            exit_reason=ExitReason.TAKE_PROFIT,
        ),
    ]

    eq_curve = [
        EquityPoint(
            bar_index=0,
            timestamp=t0,
            trading_day="2026-02-01",
            balance=Decimal("10190.00"),
            floating_pnl=Decimal("0.00"),
            equity=Decimal("10190.00"),
            used_margin=Decimal("0.00"),
            free_margin=Decimal("10190.00"),
            open_positions_count=0,
            current_daily_pnl=Decimal("190.00"),
            daily_starting_balance=Decimal("10000.00"),
            peak_balance=Decimal("10190.00"),
            drawdown_usd=Decimal("0.00"),
            drawdown_pct=Decimal("0.0000"),
        ),
        EquityPoint(
            bar_index=1,
            timestamp=t0 + timedelta(hours=1),
            trading_day="2026-02-01",
            balance=Decimal("10080.00"),
            floating_pnl=Decimal("0.00"),
            equity=Decimal("10080.00"),
            used_margin=Decimal("0.00"),
            free_margin=Decimal("10080.00"),
            open_positions_count=0,
            current_daily_pnl=Decimal("80.00"),
            daily_starting_balance=Decimal("10000.00"),
            peak_balance=Decimal("10190.00"),
            drawdown_usd=Decimal("110.00"),
            drawdown_pct=Decimal("1.0795"),
        ),
        EquityPoint(
            bar_index=2,
            timestamp=t0 + timedelta(hours=2),
            trading_day="2026-02-01",
            balance=Decimal("10370.00"),
            floating_pnl=Decimal("0.00"),
            equity=Decimal("10370.00"),
            used_margin=Decimal("0.00"),
            free_margin=Decimal("10370.00"),
            open_positions_count=0,
            current_daily_pnl=Decimal("370.00"),
            daily_starting_balance=Decimal("10000.00"),
            peak_balance=Decimal("10370.00"),
            drawdown_usd=Decimal("0.00"),
            drawdown_pct=Decimal("0.0000"),
        ),
    ]

    m = calculate_backtest_metrics(
        initial_balance=Decimal("10000.00"),
        final_balance=Decimal("10370.00"),
        final_equity=Decimal("10370.00"),
        trades=trades,
        equity_curve=eq_curve,
    )
    assert m.total_trades == 3
    assert m.winning_trades == 2
    assert m.losing_trades == 1
    assert m.breakeven_trades == 0
    assert m.win_rate_pct == Decimal("66.67")
    assert m.gross_profit == Decimal("500.00")
    assert m.gross_loss == Decimal("100.00")
    assert m.total_commission == Decimal("30.00")
    assert m.net_profit_usd == Decimal("370.00")
    assert m.net_profit_pct == Decimal("3.7000")
    assert m.profit_factor == Decimal("5.0000")
    assert m.expectancy_usd == Decimal("123.33")
    assert m.average_win_usd == Decimal("240.00")
    assert m.average_loss_usd == Decimal("110.00")
    assert m.payoff_ratio == Decimal("2.1818")
    assert m.average_r_multiple == Decimal("1.2333")
    assert m.max_drawdown_usd == Decimal("110.00")
    assert m.max_drawdown_pct == Decimal("1.0795")
    assert m.sharpe_ratio > Decimal("0")
    assert m.sortino_ratio > Decimal("0")

    # Edge case 2: Winning trades only (0 losses, 0 downside deviation) -> finite profit factor & sortino
    m_wins_only = calculate_backtest_metrics(
        initial_balance=Decimal("10000.00"),
        final_balance=Decimal("10480.00"),
        final_equity=Decimal("10480.00"),
        trades=[trades[0], trades[2]],
        equity_curve=[eq_curve[0], eq_curve[2]],
    )
    assert m_wins_only.losing_trades == 0
    assert m_wins_only.win_rate_pct == Decimal("100.00")
    assert m_wins_only.profit_factor == Decimal("500.0000")
    assert m_wins_only.sortino_ratio == Decimal("0.0000")
    assert not math.isnan(float(m_wins_only.sortino_ratio))


# ---------------------------------------------------------------------------
# 12, 13, 14, 15 & 16. StrategyEngine Integration, MTF Causal Sync, Lifecycle,
# Determinism, Future-Data Append Invariance & End-of-Data Handling
# ---------------------------------------------------------------------------
def test_end_to_end_strategy_backtest_determinism_lookahead_invariance_and_lifecycle() -> None:
    """Verify full StrategyEngine backtest determinism, future-data invariance, MTF sync, and lifecycle states."""
    # Build an 80-bar trending + pullback + rally XAUUSD dataset
    t0 = datetime(2026, 2, 2, 0, 0, tzinfo=timezone.utc)
    bars: list[tuple[datetime, float, float, float, float, float]] = []
    price = 2600.0
    for i in range(80):
        ts = t0 + timedelta(hours=i)
        # Steady uptrend with periodic structural swing lows/highs
        step = 1.8 if (i % 7 not in (3, 4)) else -1.1
        open_p = round(price, 2)
        close_p = round(price + step, 2)
        high_p = round(max(open_p, close_p) + 1.2, 2)
        low_p = round(min(open_p, close_p) - 0.8, 2)
        bars.append((ts, open_p, high_p, low_p, close_p, 150.0 + i))
        price = close_p

    df_full = _make_ohlcv_df(bars)

    # Configure StrategyEngine with lower confluence threshold so multiple trades trigger across 80 bars
    strat_engine = StrategyEngine(
        confluence_config=ConfluenceWeightsConfig(
            minimum_confidence_threshold=Decimal("0.20"),
            minimum_directional_edge=Decimal("0.05"),
        ),
        lifecycle_config=SignalLifecycleConfig(entry_zone_dedup_tolerance=Decimal("1.00")),
    )
    cfg = BacktestConfig(
        symbol="XAUUSD",
        timeframe=Timeframe.H1,
        initial_balance=Decimal("10000.00"),
        spread_points=Decimal("10"),
        slippage_points=Decimal("2"),
        commission_per_lot=Decimal("5.00"),
        warmup_bars=52,
        close_open_positions_at_end=False,
    )

    bt_engine_1 = BacktestEngine(cfg, strategy_engine=strat_engine)
    bt_engine_2 = BacktestEngine(cfg, strategy_engine=strat_engine)

    res_1 = bt_engine_1.run(df_full)
    res_2 = bt_engine_2.run(df_full)

    # 14. Deterministic repeated backtest results (bit-for-bit equality)
    assert res_1 == res_2
    assert res_1.total_bars == 80
    assert len(res_1.equity_curve) == 80
    assert len(res_1.lifecycle_records) >= 1

    # 13. Signal lifecycle synchronization check: every lifecycle record went through RISK_APPROVED or RISK_REJECTED
    for rec in res_1.lifecycle_records:
        states_visited = [ev.to_state for ev in rec.history]
        assert SignalLifecycleState.GENERATED in states_visited
        if rec.state in (
            SignalLifecycleState.ACTIVE,
            SignalLifecycleState.TARGET_HIT,
            SignalLifecycleState.STOP_LOSS_HIT,
        ):
            assert SignalLifecycleState.RISK_APPROVED in states_visited
            assert SignalLifecycleState.ACTIVE in states_visited

    # 15. Future-data append invariance:
    # Run on first 68 bars vs full 80 bars (with close_open_positions_at_end=False).
    # Every equity point on bars 0..67 and every trade closed on or before bar 67 must be identical!
    df_prefix = df_full.iloc[:68].copy()
    res_prefix = bt_engine_1.run(df_prefix)

    assert res_prefix.equity_curve == res_1.equity_curve[:68]
    trades_closed_by_67 = tuple(t for t in res_1.trades if t.exit_bar_index < 68)
    assert res_prefix.trades == trades_closed_by_67


def test_end_of_dataset_position_handling_and_mtf_causal_filtering() -> None:
    """Verify close_open_positions_at_end=True vs False and MTF closed-candle filtering in BacktestEngine."""
    t0 = datetime(2026, 2, 10, 10, 0, tzinfo=timezone.utc)
    # 4 M15 bars from 10:00 to 10:45 UTC, plus 2 H1 bars (10:00 and 11:00 UTC)
    m15_bars = [
        (t0 + timedelta(minutes=15 * i), 2600.0 + i, 2602.0 + i, 2599.0 + i, 2601.0 + i, 100.0)
        for i in range(4)
    ]
    h1_bars = [
        (t0 - timedelta(hours=1), 2595.0, 2601.0, 2594.0, 2600.0, 400.0),
        # This 10:00 H1 bar has a huge spike, but it only closes at 11:00 UTC!
        (t0, 2600.0, 2700.0, 2500.0, 2650.0, 400.0),
    ]
    m15_df = _make_ohlcv_df(m15_bars)
    h1_df = _make_ohlcv_df(h1_bars)

    def provider(causal_df: pd.DataFrame, idx: int, account: AccountState) -> TradeSignal:
        ts = causal_df.iloc[-1]["timestamp"].to_pydatetime()
        if idx == 0:
            # Open a BUY trade with wide SL/TP so it stays open until the end of the 4-bar dataset
            return TradeSignal(
                symbol="XAUUSD",
                direction=SignalDirection.BUY,
                entry_price=Decimal("2601.00"),
                stop_loss=Decimal("2550.00"),
                take_profit=Decimal("2700.00"),
                timeframe=Timeframe.M15,
                strategy_id="eod_test",
                confidence=Decimal("0.85"),
                timestamp=ts,
                rationale="Open position held to end of data.",
            )
        return _make_wait_signal(ts, tf=Timeframe.M15)

    # Case A: close_open_positions_at_end=True -> closes on final bar with ExitReason.END_OF_DATA
    cfg_close = BacktestConfig(
        symbol="XAUUSD",
        timeframe=Timeframe.M15,
        spread_points=Decimal("0"),
        slippage_points=Decimal("0"),
        commission_per_lot=Decimal("0"),
        close_open_positions_at_end=True,
    )
    res_closed = BacktestEngine(cfg_close).run(
        m15_df,
        mtf_data={Timeframe.H1: h1_df},
        signal_provider=provider,
    )
    assert len(res_closed.open_positions) == 0
    assert len(res_closed.trades) == 1
    assert res_closed.trades[0].exit_reason == ExitReason.END_OF_DATA
    assert res_closed.trades[0].exit_bar_index == 3

    # Case B: close_open_positions_at_end=False -> position remains in open_positions
    cfg_keep = cfg_close.model_copy(update={"close_open_positions_at_end": False})
    res_open = BacktestEngine(cfg_keep).run(
        m15_df,
        mtf_data={Timeframe.H1: h1_df},
        signal_provider=provider,
    )
    assert len(res_open.open_positions) == 1
    assert len(res_open.trades) == 0
    assert res_open.open_positions[0].floating_pnl > Decimal("0")
