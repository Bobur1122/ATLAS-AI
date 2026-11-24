"""Deterministic Backtest Performance Metrics Calculator for ATLAS AI Phase 4."""

from decimal import Decimal, ROUND_HALF_UP
import math
from typing import Sequence

from atlas_ai.backtest.models import BacktestMetrics, BacktestTradeRecord, EquityPoint

QUANTIZE_MONEY = Decimal("0.01")
QUANTIZE_RATIO = Decimal("0.0001")
HUNDRED = Decimal("100")


def calculate_backtest_metrics(
    *,
    initial_balance: Decimal,
    final_balance: Decimal,
    final_equity: Decimal,
    trades: Sequence[BacktestTradeRecord],
    equity_curve: Sequence[EquityPoint],
    annualization_periods: int = 252,
) -> BacktestMetrics:
    """Compute comprehensive, deterministic performance metrics from closed trades and equity curve.

    Edge-case guarantees:
    - Never returns `NaN` or `Infinity`.
    - Handles 0 trades, 0 wins, 0 losses, 0 drawdown, and 0 downside deviation safely.
    """
    init_bal = initial_balance.quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
    fin_bal = final_balance.quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
    fin_eq = final_equity.quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)

    total_trades = len(trades)
    wins = [t for t in trades if t.net_pnl > Decimal("0")]
    losses = [t for t in trades if t.net_pnl < Decimal("0")]
    breakevens = [t for t in trades if t.net_pnl == Decimal("0")]

    winning_trades = len(wins)
    losing_trades = len(losses)
    breakeven_trades = len(breakevens)

    if total_trades > 0:
        win_rate_pct = (
            (Decimal(winning_trades) / Decimal(total_trades)) * HUNDRED
        ).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
    else:
        win_rate_pct = Decimal("0.00")

    # Gross profit & loss across positive/negative gross PnL trades, plus total commission
    gross_profit = sum(
        (t.gross_pnl for t in trades if t.gross_pnl > Decimal("0")),
        Decimal("0.00"),
    ).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)

    gross_loss = sum(
        (abs(t.gross_pnl) for t in trades if t.gross_pnl < Decimal("0")),
        Decimal("0.00"),
    ).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)

    total_commission = sum(
        (t.commission for t in trades),
        Decimal("0.00"),
    ).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)

    net_profit_usd = sum(
        (t.net_pnl for t in trades),
        Decimal("0.00"),
    ).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)

    net_profit_pct = (
        ((net_profit_usd / init_bal) * HUNDRED).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)
        if init_bal > Decimal("0")
        else Decimal("0.0000")
    )

    # Profit factor: gross_profit / gross_loss (safe finite fallback when gross_loss == 0)
    if gross_loss > Decimal("0"):
        profit_factor = (gross_profit / gross_loss).quantize(
            QUANTIZE_RATIO, rounding=ROUND_HALF_UP
        )
    elif gross_profit > Decimal("0"):
        profit_factor = gross_profit.quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)
    else:
        profit_factor = Decimal("0.0000")

    if total_trades > 0:
        expectancy_usd = (net_profit_usd / Decimal(total_trades)).quantize(
            QUANTIZE_MONEY, rounding=ROUND_HALF_UP
        )
        avg_r = (
            sum((t.realized_r_multiple for t in trades), Decimal("0")) / Decimal(total_trades)
        ).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)
        expectancy_r = avg_r
    else:
        expectancy_usd = Decimal("0.00")
        avg_r = Decimal("0.0000")
        expectancy_r = Decimal("0.0000")

    if winning_trades > 0:
        average_win_usd = (
            sum((t.net_pnl for t in wins), Decimal("0.00")) / Decimal(winning_trades)
        ).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
    else:
        average_win_usd = Decimal("0.00")

    if losing_trades > 0:
        average_loss_usd = (
            sum((abs(t.net_pnl) for t in losses), Decimal("0.00")) / Decimal(losing_trades)
        ).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
    else:
        average_loss_usd = Decimal("0.00")

    if average_loss_usd > Decimal("0"):
        payoff_ratio = (average_win_usd / average_loss_usd).quantize(
            QUANTIZE_RATIO, rounding=ROUND_HALF_UP
        )
    elif average_win_usd > Decimal("0"):
        payoff_ratio = average_win_usd.quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)
    else:
        payoff_ratio = Decimal("0.0000")

    # Max Drawdown from equity curve (computing peak-to-trough across the sequence)
    max_dd_usd = Decimal("0.00")
    max_dd_pct = Decimal("0.0000")
    running_peak = init_bal

    for pt in equity_curve:
        if pt.equity > running_peak:
            running_peak = pt.equity
        dd_u = max(Decimal("0.00"), running_peak - pt.equity)
        dd_p = (dd_u / running_peak) * HUNDRED if running_peak > Decimal("0") else Decimal("0")
        if dd_u > max_dd_usd:
            max_dd_usd = dd_u
        if dd_p > max_dd_pct:
            max_dd_pct = dd_p
        # Also consider pt.drawdown_usd / pt.drawdown_pct if higher
        if pt.drawdown_usd > max_dd_usd:
            max_dd_usd = pt.drawdown_usd
        if pt.drawdown_pct > max_dd_pct:
            max_dd_pct = pt.drawdown_pct

    max_dd_usd = max_dd_usd.quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
    max_dd_pct = max_dd_pct.quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)

    # Sharpe & Sortino ratios from bar-by-bar equity returns (or trade returns if equity curve is short)
    returns: list[float] = []
    if len(equity_curve) >= 2:
        prev_eq = float(init_bal)
        for pt in equity_curve:
            curr_eq = float(pt.equity)
            if prev_eq > 0.0:
                returns.append((curr_eq - prev_eq) / prev_eq)
            prev_eq = curr_eq
    elif len(trades) >= 2 and init_bal > Decimal("0"):
        running_bal = float(init_bal)
        for tr in trades:
            if running_bal > 0.0:
                returns.append(float(tr.net_pnl) / running_bal)
                running_bal += float(tr.net_pnl)

    sharpe_ratio = _compute_sharpe(returns, annualization_periods)
    sortino_ratio = _compute_sortino(returns, annualization_periods)

    return BacktestMetrics(
        initial_balance=init_bal,
        final_balance=fin_bal,
        final_equity=fin_eq,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        breakeven_trades=breakeven_trades,
        win_rate_pct=win_rate_pct,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        total_commission=total_commission,
        net_profit_usd=net_profit_usd,
        net_profit_pct=net_profit_pct,
        profit_factor=profit_factor,
        expectancy_usd=expectancy_usd,
        expectancy_r=expectancy_r,
        average_win_usd=average_win_usd,
        average_loss_usd=average_loss_usd,
        payoff_ratio=payoff_ratio,
        average_r_multiple=avg_r,
        max_drawdown_usd=max_dd_usd,
        max_drawdown_pct=max_dd_pct,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
    )


def _compute_sharpe(returns: list[float], annualization_periods: int) -> Decimal:
    """Compute annualized Sharpe ratio safely without NaN or Infinity."""
    n = len(returns)
    if n < 2:
        return Decimal("0.0000")
    mean_ret = sum(returns) / n
    variance = sum((r - mean_ret) ** 2 for r in returns) / (n - 1)
    std_dev = math.sqrt(variance)
    if std_dev <= 1e-12 or math.isnan(std_dev) or math.isinf(std_dev):
        return Decimal("0.0000")
    raw_sharpe = (mean_ret / std_dev) * math.sqrt(float(annualization_periods))
    if math.isnan(raw_sharpe) or math.isinf(raw_sharpe):
        return Decimal("0.0000")
    return Decimal(str(round(raw_sharpe, 4))).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)


def _compute_sortino(returns: list[float], annualization_periods: int) -> Decimal:
    """Compute annualized Sortino ratio safely without NaN or Infinity."""
    n = len(returns)
    if n < 2:
        return Decimal("0.0000")
    mean_ret = sum(returns) / n
    downside_sq = [min(0.0, r) ** 2 for r in returns]
    downside_var = sum(downside_sq) / n
    downside_dev = math.sqrt(downside_var)
    if downside_dev <= 1e-12 or math.isnan(downside_dev) or math.isinf(downside_dev):
        return Decimal("0.0000")
    raw_sortino = (mean_ret / downside_dev) * math.sqrt(float(annualization_periods))
    if math.isnan(raw_sortino) or math.isinf(raw_sortino):
        return Decimal("0.0000")
    return Decimal(str(round(raw_sortino, 4))).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)
