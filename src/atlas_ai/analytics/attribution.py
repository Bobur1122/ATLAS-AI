"""Multi-Dimensional Performance Attribution & Execution Quality Engine for ATLAS AI Phase 8.2."""

from decimal import Decimal, ROUND_HALF_UP
import math
from typing import Sequence

from atlas_ai.analytics.config import PostTradeAnalyticsConfig
from atlas_ai.analytics.exceptions import AnalyticsError
from atlas_ai.analytics.journal import PersistentTradeJournal
from atlas_ai.analytics.models import (
    CompletedTradeRecord,
    ContextAttributionStatus,
    ExecutionQualityReport,
    MULTI_LABEL_DIMENSIONS,
    PerformanceAttributionReport,
    SINGLE_LABEL_DIMENSIONS,
    SegmentPerformanceBreakdown,
    TradeSource,
)
from atlas_ai.backtest.metrics import _compute_sharpe, _compute_sortino
from atlas_ai.backtest.models import BacktestMetrics, EquityPoint
from atlas_ai.domain.signals import SignalDirection

QUANTIZE_MONEY = Decimal("0.01")
QUANTIZE_RATIO = Decimal("0.0001")
HUNDRED = Decimal("100")

MISSING_CONTEXT_SEGMENT_KEY = "UNATTRIBUTED_MISSING_CONTEXT"
NO_KNOWLEDGE_RULE_SEGMENT_KEY = "NO_KNOWLEDGE_RULE"


def _is_finite_decimal(val: object) -> bool:
    return isinstance(val, Decimal) and val.is_finite()


def validate_completed_trade_record(
    trade: CompletedTradeRecord,
    *,
    symbol_point: Decimal = Decimal("0.01"),
) -> None:
    """Strictly validate per-trade cash-flow identity, R-multiple consistency, slippage sign & magnitude,
    positive initial risk, and Decimal finiteness/precision before any Phase 8.2 aggregation.

    Fails closed by raising `AnalyticsError` on any violation.
    """
    if not _is_finite_decimal(symbol_point) or symbol_point <= Decimal("0"):
        raise AnalyticsError(f"symbol_point must be a finite positive Decimal, got {symbol_point!r}.")

    decimal_fields = (
        ("requested_entry_price", trade.requested_entry_price),
        ("entry_fill_price", trade.entry_fill_price),
        ("requested_exit_price", trade.requested_exit_price),
        ("exit_fill_price", trade.exit_fill_price),
        ("stop_loss", trade.stop_loss),
        ("take_profit", trade.take_profit),
        ("volume", trade.volume),
        ("entry_slippage_points", trade.entry_slippage_points),
        ("exit_slippage_points", trade.exit_slippage_points),
        ("initial_risk_amount", trade.initial_risk_amount),
        ("gross_pnl", trade.gross_pnl),
        ("commission", trade.commission),
        ("fee", trade.fee),
        ("swap", trade.swap),
        ("net_pnl", trade.net_pnl),
        ("realized_r_multiple", trade.realized_r_multiple),
        ("mfe_usd", trade.mfe_usd),
        ("mae_usd", trade.mae_usd),
    )
    for field_name, val in decimal_fields:
        if not _is_finite_decimal(val):
            raise AnalyticsError(
                f"Non-finite or non-Decimal field '{field_name}' ({val!r}) on trade '{trade.trade_id}'."
            )

    if trade.direction not in (SignalDirection.BUY, SignalDirection.SELL):
        raise AnalyticsError(
            f"Invalid direction '{trade.direction}' on trade '{trade.trade_id}'; expected BUY or SELL."
        )

    if (
        trade.requested_entry_price <= Decimal("0")
        or trade.entry_fill_price <= Decimal("0")
        or trade.requested_exit_price <= Decimal("0")
        or trade.exit_fill_price <= Decimal("0")
        or trade.stop_loss <= Decimal("0")
        or trade.take_profit <= Decimal("0")
    ):
        raise AnalyticsError(
            f"Non-positive price level detected on trade '{trade.trade_id}'."
        )

    if trade.volume <= Decimal("0"):
        raise AnalyticsError(
            f"Non-positive volume ({trade.volume}) on trade '{trade.trade_id}'."
        )

    if trade.initial_risk_amount <= Decimal("0"):
        raise AnalyticsError(
            f"Invalid non-positive initial_risk_amount ({trade.initial_risk_amount}) on trade '{trade.trade_id}'."
        )

    if trade.commission < Decimal("0") or trade.fee < Decimal("0"):
        raise AnalyticsError(
            f"Negative commission ({trade.commission}) or fee ({trade.fee}) on trade '{trade.trade_id}'."
        )

    if trade.exit_timestamp_utc < trade.entry_timestamp_utc:
        raise AnalyticsError(
            f"Chronological violation on trade '{trade.trade_id}': exit_timestamp_utc "
            f"({trade.exit_timestamp_utc.isoformat()}) < entry_timestamp_utc ({trade.entry_timestamp_utc.isoformat()})."
        )

    if trade.partial_exit_count < 1:
        raise AnalyticsError(
            f"Invalid partial_exit_count ({trade.partial_exit_count}) on trade '{trade.trade_id}'."
        )

    # 1. Exact monetary quantization (0.01) check with zero tolerance
    monetary_fields = (
        ("initial_risk_amount", trade.initial_risk_amount),
        ("gross_pnl", trade.gross_pnl),
        ("commission", trade.commission),
        ("fee", trade.fee),
        ("swap", trade.swap),
        ("net_pnl", trade.net_pnl),
    )
    for field_name, val in monetary_fields:
        q_val = val.quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
        if val != q_val:
            raise AnalyticsError(
                f"Monetary precision violation on trade '{trade.trade_id}': {field_name}={val} "
                f"is not quantized to 0.01 ({q_val})."
            )

    # 2. Exact cash-flow identity check (zero tolerance)
    expected_net = trade.gross_pnl - trade.commission - trade.fee + trade.swap
    if trade.net_pnl != expected_net:
        raise AnalyticsError(
            f"Cash-flow identity violation on trade '{trade.trade_id}': net_pnl ({trade.net_pnl}) != "
            f"gross_pnl ({trade.gross_pnl}) - commission ({trade.commission}) - fee ({trade.fee}) "
            f"+ swap ({trade.swap}) = {expected_net}."
        )

    # 3. Exact R-multiple identity check (0.0001 quantization, zero tolerance)
    expected_r = (trade.net_pnl / trade.initial_risk_amount).quantize(
        QUANTIZE_RATIO, rounding=ROUND_HALF_UP
    )
    actual_r_q = trade.realized_r_multiple.quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)
    if trade.realized_r_multiple != actual_r_q or actual_r_q != expected_r:
        raise AnalyticsError(
            f"R-multiple consistency violation on trade '{trade.trade_id}': "
            f"realized_r_multiple ({trade.realized_r_multiple}) != quantize(net_pnl / initial_risk_amount, 0.0001) "
            f"({expected_r})."
        )

    # 4. Exact direction-normalized entry & exit slippage sign and magnitude check
    if trade.direction == SignalDirection.BUY:
        expected_entry_slip = (
            (trade.entry_fill_price - trade.requested_entry_price) / symbol_point
        ).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)
        expected_exit_slip = (
            (trade.requested_exit_price - trade.exit_fill_price) / symbol_point
        ).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)
    else:
        expected_entry_slip = (
            (trade.requested_entry_price - trade.entry_fill_price) / symbol_point
        ).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)
        expected_exit_slip = (
            (trade.exit_fill_price - trade.requested_exit_price) / symbol_point
        ).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)

    actual_entry_slip_q = trade.entry_slippage_points.quantize(
        QUANTIZE_RATIO, rounding=ROUND_HALF_UP
    )
    actual_exit_slip_q = trade.exit_slippage_points.quantize(
        QUANTIZE_RATIO, rounding=ROUND_HALF_UP
    )
    if trade.entry_slippage_points != actual_entry_slip_q or actual_entry_slip_q != expected_entry_slip:
        raise AnalyticsError(
            f"Entry slippage violation on {trade.direction.value} trade '{trade.trade_id}': "
            f"entry_slippage_points ({trade.entry_slippage_points}) != expected {expected_entry_slip} "
            f"(requested={trade.requested_entry_price}, fill={trade.entry_fill_price}, point={symbol_point})."
        )
    if trade.exit_slippage_points != actual_exit_slip_q or actual_exit_slip_q != expected_exit_slip:
        raise AnalyticsError(
            f"Exit slippage violation on {trade.direction.value} trade '{trade.trade_id}': "
            f"exit_slippage_points ({trade.exit_slippage_points}) != expected {expected_exit_slip} "
            f"(requested={trade.requested_exit_price}, fill={trade.exit_fill_price}, point={symbol_point})."
        )

    # 5. Context attribution consistency check
    if trade.entry_context is not None:
        if trade.context_attribution_status != ContextAttributionStatus.VERIFIED_ENTRY_CONTEXT:
            raise AnalyticsError(
                f"Context attribution status mismatch on trade '{trade.trade_id}': entry_context is present "
                f"but status is {trade.context_attribution_status}."
            )
    else:
        if trade.context_attribution_status != ContextAttributionStatus.UNATTRIBUTED_MISSING_CONTEXT:
            raise AnalyticsError(
                f"Context attribution status mismatch on trade '{trade.trade_id}': entry_context is None "
                f"but status is {trade.context_attribution_status}."
            )


def _validate_and_normalize_trades(
    trades: Sequence[CompletedTradeRecord],
    *,
    symbol_point: Decimal = Decimal("0.01"),
    preserve_input_order: bool = False,
) -> tuple[CompletedTradeRecord, ...]:
    """Validate every trade record and ensure unique `trade_id` across the sequence."""
    seen_ids: set[str] = set()
    for tr in trades:
        validate_completed_trade_record(tr, symbol_point=symbol_point)
        if tr.trade_id in seen_ids:
            raise AnalyticsError(
                f"Duplicate trade_id '{tr.trade_id}' in trade sequence; refusing to double-count."
            )
        seen_ids.add(tr.trade_id)

    if preserve_input_order:
        return tuple(trades)

    return tuple(
        sorted(
            trades,
            key=lambda item: (item.exit_timestamp_utc, item.entry_timestamp_utc, item.trade_id),
        )
    )


def compute_r_drawdown(trades: Sequence[CompletedTradeRecord]) -> Decimal:
    """Compute peak-to-trough cumulative R-multiple drawdown (`max_drawdown_r >= 0.0000`)
    across a chronologically ordered sequence of trades.
    """
    if not trades:
        return Decimal("0.0000")

    cum_r = Decimal("0.0000")
    peak_r = Decimal("0.0000")
    max_dd_r = Decimal("0.0000")

    for tr in trades:
        cum_r += tr.realized_r_multiple
        if cum_r > peak_r:
            peak_r = cum_r
        dd = peak_r - cum_r
        if dd > max_dd_r:
            max_dd_r = dd

    return max_dd_r.quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)


def compute_r_sharpe_and_sortino(
    trades: Sequence[CompletedTradeRecord],
    *,
    annualization_periods: int = 252,
) -> tuple[Decimal, Decimal]:
    """Compute trade-based R-multiple Sharpe and Sortino ratios scaled by `sqrt(annualization_periods)`.

    Statistical note:
    - Uses per-trade realized R-multiples `r_i = float(trade.realized_r_multiple)`.
    - Returns `(Decimal("0.0000"), Decimal("0.0000"))` when `len(trades) < 2`.
    - Returns `Decimal("0.0000")` for Sharpe when sample standard deviation `s <= 1e-12` (zero variance).
    - Returns `Decimal("0.0000")` for Sortino when zero-target downside deviation `sigma_d <= 1e-12` (zero losses).
    - Scaling by `sqrt(252)` provides a fixed-scale standardized risk-adjusted return index
      directly comparable to Phase 6 `ModelEvaluationReport.simulated_sharpe_ratio`.
    """
    if annualization_periods < 1:
        raise AnalyticsError(f"annualization_periods must be >= 1, got {annualization_periods}.")

    r_series = [float(tr.realized_r_multiple) for tr in trades]
    n = len(r_series)
    if n < 2:
        return Decimal("0.0000"), Decimal("0.0000")

    mean_r = sum(r_series) / n
    ann_sqrt = math.sqrt(float(annualization_periods))

    # Sample standard deviation (n - 1 denominator)
    variance = sum((r - mean_r) ** 2 for r in r_series) / (n - 1)
    std_dev = math.sqrt(variance) if variance > 0.0 else 0.0
    if std_dev <= 1e-12 or math.isnan(std_dev) or math.isinf(std_dev):
        sharpe_dec = Decimal("0.0000")
    else:
        raw_sharpe = (mean_r / std_dev) * ann_sqrt
        if math.isnan(raw_sharpe) or math.isinf(raw_sharpe):
            sharpe_dec = Decimal("0.0000")
        else:
            sharpe_dec = Decimal(str(round(raw_sharpe, 4))).quantize(
                QUANTIZE_RATIO, rounding=ROUND_HALF_UP
            )

    # Zero-target downside deviation (n denominator)
    downside_sq = [min(0.0, r) ** 2 for r in r_series]
    downside_var = sum(downside_sq) / n
    downside_dev = math.sqrt(downside_var) if downside_var > 0.0 else 0.0
    if downside_dev <= 1e-12 or math.isnan(downside_dev) or math.isinf(downside_dev):
        sortino_dec = Decimal("0.0000")
    else:
        raw_sortino = (mean_r / downside_dev) * ann_sqrt
        if math.isnan(raw_sortino) or math.isinf(raw_sortino):
            sortino_dec = Decimal("0.0000")
        else:
            sortino_dec = Decimal(str(round(raw_sortino, 4))).quantize(
                QUANTIZE_RATIO, rounding=ROUND_HALF_UP
            )

    return sharpe_dec, sortino_dec


def compute_execution_quality_report(
    trades: Sequence[CompletedTradeRecord],
    *,
    symbol_point: Decimal = Decimal("0.01"),
) -> ExecutionQualityReport:
    """Compute fill quality, direction-normalized entry/exit slippage, partial-exit frequency,
    and commission/fee/swap cost drag across `trades`.
    """
    validated = _validate_and_normalize_trades(trades, symbol_point=symbol_point)
    n = len(validated)
    if n == 0:
        return ExecutionQualityReport(total_evaluated_trades=0)

    n_dec = Decimal(n)
    mean_entry_slip = (
        sum((t.entry_slippage_points for t in validated), Decimal("0")) / n_dec
    ).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)

    max_adverse_entry_slip = max(
        Decimal("0.0000"),
        max(t.entry_slippage_points for t in validated),
    ).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)

    adverse_cnt = sum(1 for t in validated if t.entry_slippage_points > Decimal("0"))
    favorable_cnt = sum(1 for t in validated if t.entry_slippage_points < Decimal("0"))
    zero_cnt = sum(1 for t in validated if t.entry_slippage_points == Decimal("0"))

    mean_exit_slip = (
        sum((t.exit_slippage_points for t in validated), Decimal("0")) / n_dec
    ).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)

    partial_exit_cnt = sum(1 for t in validated if t.partial_exit_count > 1)

    total_gp = sum(
        (t.gross_pnl for t in validated if t.gross_pnl > Decimal("0")),
        Decimal("0.00"),
    ).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
    total_comm = sum(
        (t.commission for t in validated),
        Decimal("0.00"),
    ).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
    total_fee = sum(
        (t.fee for t in validated),
        Decimal("0.00"),
    ).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
    total_swap = sum(
        (t.swap for t in validated),
        Decimal("0.00"),
    ).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
    total_cost_drag = (total_comm + total_fee - total_swap).quantize(
        QUANTIZE_MONEY, rounding=ROUND_HALF_UP
    )

    per_trade_drag_r_sum = sum(
        (
            (t.commission + t.fee - t.swap) / t.initial_risk_amount
            for t in validated
        ),
        Decimal("0"),
    )
    cost_drag_r = (per_trade_drag_r_sum / n_dec).quantize(
        QUANTIZE_RATIO, rounding=ROUND_HALF_UP
    )

    return ExecutionQualityReport(
        total_evaluated_trades=n,
        mean_entry_slippage_points=mean_entry_slip,
        max_adverse_entry_slippage_points=max_adverse_entry_slip,
        adverse_entry_slippage_count=adverse_cnt,
        favorable_entry_slippage_count=favorable_cnt,
        zero_entry_slippage_count=zero_cnt,
        mean_exit_slippage_points=mean_exit_slip,
        partial_exit_trade_count=partial_exit_cnt,
        total_gross_profit_usd=total_gp,
        total_commission_usd=total_comm,
        total_fee_usd=total_fee,
        total_swap_usd=total_swap,
        total_cost_drag_usd=total_cost_drag,
        cost_drag_r_per_trade=cost_drag_r,
    )


def compute_segment_breakdown(
    dimension_name: str,
    segment_key: str,
    trades: Sequence[CompletedTradeRecord],
    *,
    min_trades_threshold: int = 10,
    annualization_periods: int = 252,
    symbol_point: Decimal = Decimal("0.01"),
) -> SegmentPerformanceBreakdown:
    """Compute a deterministic `SegmentPerformanceBreakdown` for a single dimensional slice."""
    cleaned_dim = dimension_name.strip()
    cleaned_key = segment_key.strip()
    if not cleaned_dim or not cleaned_key:
        raise AnalyticsError("dimension_name and segment_key must be non-empty strings.")
    if min_trades_threshold < 1:
        raise AnalyticsError(f"min_trades_threshold must be >= 1, got {min_trades_threshold}.")

    validated = _validate_and_normalize_trades(trades, symbol_point=symbol_point)
    total_trades = len(validated)
    wins = [t for t in validated if t.net_pnl > Decimal("0")]
    losses = [t for t in validated if t.net_pnl < Decimal("0")]
    breakevens = [t for t in validated if t.net_pnl == Decimal("0")]

    winning_trades = len(wins)
    losing_trades = len(losses)
    breakeven_trades = len(breakevens)

    if total_trades > 0:
        win_rate_pct = (
            (Decimal(winning_trades) / Decimal(total_trades)) * HUNDRED
        ).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
    else:
        win_rate_pct = Decimal("0.00")

    gross_profit_usd = sum(
        (t.gross_pnl for t in validated if t.gross_pnl > Decimal("0")),
        Decimal("0.00"),
    ).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
    gross_loss_usd = sum(
        (abs(t.gross_pnl) for t in validated if t.gross_pnl < Decimal("0")),
        Decimal("0.00"),
    ).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
    net_profit_usd = sum(
        (t.net_pnl for t in validated),
        Decimal("0.00"),
    ).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)

    if gross_loss_usd > Decimal("0"):
        profit_factor = (gross_profit_usd / gross_loss_usd).quantize(
            QUANTIZE_RATIO, rounding=ROUND_HALF_UP
        )
    elif gross_profit_usd > Decimal("0"):
        profit_factor = gross_profit_usd.quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)
    else:
        profit_factor = Decimal("0.0000")

    if total_trades > 0:
        n_dec = Decimal(total_trades)
        expectancy_usd = (net_profit_usd / n_dec).quantize(
            QUANTIZE_MONEY, rounding=ROUND_HALF_UP
        )
        expectancy_r = (
            sum((t.realized_r_multiple for t in validated), Decimal("0")) / n_dec
        ).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)
        mean_entry_slip = (
            sum((t.entry_slippage_points for t in validated), Decimal("0")) / n_dec
        ).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)
    else:
        expectancy_usd = Decimal("0.00")
        expectancy_r = Decimal("0.0000")
        mean_entry_slip = Decimal("0.0000")

    max_dd_r = compute_r_drawdown(validated)
    sharpe_r, sortino_r = compute_r_sharpe_and_sortino(
        validated,
        annualization_periods=annualization_periods,
    )

    return SegmentPerformanceBreakdown(
        dimension_name=cleaned_dim,
        segment_key=cleaned_key,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        breakeven_trades=breakeven_trades,
        win_rate_pct=win_rate_pct,
        gross_profit_usd=gross_profit_usd,
        gross_loss_usd=gross_loss_usd,
        net_profit_usd=net_profit_usd,
        profit_factor=profit_factor,
        expectancy_usd=expectancy_usd,
        expectancy_r=expectancy_r,
        max_drawdown_r=max_dd_r,
        sharpe_ratio=sharpe_r,
        sortino_ratio=sortino_r,
        mean_entry_slippage_points=mean_entry_slip,
        insufficient_sample=total_trades < min_trades_threshold,
    )


def compute_overall_backtest_metrics(
    trades: Sequence[CompletedTradeRecord],
    *,
    initial_balance: Decimal = Decimal("10000.00"),
    final_balance: Decimal | None = None,
    final_equity: Decimal | None = None,
    equity_curve: Sequence[EquityPoint] | None = None,
    annualization_periods: int = 252,
    symbol_point: Decimal = Decimal("0.01"),
    preserve_trade_order: bool = False,
) -> tuple[BacktestMetrics, Decimal]:
    """Compute portfolio-level `BacktestMetrics` and `overall_max_drawdown_r` from `CompletedTradeRecord` items.

    Phase 4 Compatibility Guarantee:
    - Reproduces all 24 `BacktestMetrics` fields identically to `calculate_backtest_metrics()` when given
      converted Phase 4 `BacktestTradeRecord` items and matching `initial_balance` / `equity_curve`.
    - When `equity_curve` is omitted (`None`, e.g., pure MT5 DEMO journal analysis), computes realized
      closed-trade peak-to-trough `max_drawdown_usd` and `max_drawdown_pct` from the sequential closed-trade
      balance progression `B_k = B_{k-1} + net_pnl_k`, and uses Phase 4's sequential account-return
      series `float(tr.net_pnl) / running_bal` for `sharpe_ratio` and `sortino_ratio`.
    """
    if not _is_finite_decimal(initial_balance) or initial_balance <= Decimal("0"):
        raise AnalyticsError(f"initial_balance must be a finite positive Decimal, got {initial_balance!r}.")
    if annualization_periods < 1:
        raise AnalyticsError(f"annualization_periods must be >= 1, got {annualization_periods}.")

    validated = _validate_and_normalize_trades(
        trades,
        symbol_point=symbol_point,
        preserve_input_order=preserve_trade_order,
    )

    init_bal = initial_balance.quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
    total_trades = len(validated)
    wins = [t for t in validated if t.net_pnl > Decimal("0")]
    losses = [t for t in validated if t.net_pnl < Decimal("0")]
    breakevens = [t for t in validated if t.net_pnl == Decimal("0")]

    winning_trades = len(wins)
    losing_trades = len(losses)
    breakeven_trades = len(breakevens)

    if total_trades > 0:
        win_rate_pct = (
            (Decimal(winning_trades) / Decimal(total_trades)) * HUNDRED
        ).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
    else:
        win_rate_pct = Decimal("0.00")

    gross_profit = sum(
        (t.gross_pnl for t in validated if t.gross_pnl > Decimal("0")),
        Decimal("0.00"),
    ).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
    gross_loss = sum(
        (abs(t.gross_pnl) for t in validated if t.gross_pnl < Decimal("0")),
        Decimal("0.00"),
    ).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
    total_commission = sum(
        (t.commission for t in validated),
        Decimal("0.00"),
    ).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
    net_profit_usd = sum(
        (t.net_pnl for t in validated),
        Decimal("0.00"),
    ).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)

    fin_bal = (
        final_balance.quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
        if final_balance is not None
        else (init_bal + net_profit_usd).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
    )
    fin_eq = (
        final_equity.quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
        if final_equity is not None
        else fin_bal
    )

    net_profit_pct = (
        ((net_profit_usd / init_bal) * HUNDRED).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)
        if init_bal > Decimal("0")
        else Decimal("0.0000")
    )

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
            sum((t.realized_r_multiple for t in validated), Decimal("0")) / Decimal(total_trades)
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

    max_dd_usd = Decimal("0.00")
    max_dd_pct = Decimal("0.0000")
    running_peak = init_bal

    if equity_curve is not None:
        for pt in equity_curve:
            if pt.equity > running_peak:
                running_peak = pt.equity
            dd_u = max(Decimal("0.00"), running_peak - pt.equity)
            dd_p = (dd_u / running_peak) * HUNDRED if running_peak > Decimal("0") else Decimal("0")
            if dd_u > max_dd_usd:
                max_dd_usd = dd_u
            if dd_p > max_dd_pct:
                max_dd_pct = dd_p
            if pt.drawdown_usd > max_dd_usd:
                max_dd_usd = pt.drawdown_usd
            if pt.drawdown_pct > max_dd_pct:
                max_dd_pct = pt.drawdown_pct
    else:
        curr_bal = init_bal
        for tr in validated:
            curr_bal += tr.net_pnl
            if curr_bal > running_peak:
                running_peak = curr_bal
            dd_u = max(Decimal("0.00"), running_peak - curr_bal)
            dd_p = (dd_u / running_peak) * HUNDRED if running_peak > Decimal("0") else Decimal("0")
            if dd_u > max_dd_usd:
                max_dd_usd = dd_u
            if dd_p > max_dd_pct:
                max_dd_pct = dd_p

    max_dd_usd = max_dd_usd.quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
    max_dd_pct = max_dd_pct.quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)

    # Account-return Sharpe & Sortino (strictly separated from segment R-multiple Sharpe & Sortino)
    returns: list[float] = []
    if equity_curve is not None and len(equity_curve) >= 2:
        prev_eq = float(init_bal)
        for pt in equity_curve:
            curr_eq = float(pt.equity)
            if prev_eq > 0.0:
                returns.append((curr_eq - prev_eq) / prev_eq)
            prev_eq = curr_eq
    elif len(validated) >= 2 and init_bal > Decimal("0"):
        running_bal = float(init_bal)
        for tr in validated:
            if running_bal > 0.0:
                returns.append(float(tr.net_pnl) / running_bal)
                running_bal += float(tr.net_pnl)

    sharpe_ratio = _compute_sharpe(returns, annualization_periods)
    sortino_ratio = _compute_sortino(returns, annualization_periods)
    overall_max_dd_r = compute_r_drawdown(validated)

    metrics = BacktestMetrics(
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
    return metrics, overall_max_dd_r


class PerformanceAttributionEngine:
    """Deterministic Phase 8.2 engine that validates closed trades and produces
    multi-dimensional `SegmentPerformanceBreakdown` slices, `ExecutionQualityReport`,
    and portfolio-level `BacktestMetrics`.
    """

    def __init__(
        self,
        *,
        config: PostTradeAnalyticsConfig | None = None,
        journal: PersistentTradeJournal | None = None,
        symbol_point: Decimal = Decimal("0.01"),
        annualization_periods: int = 252,
    ) -> None:
        if not _is_finite_decimal(symbol_point) or symbol_point <= Decimal("0"):
            raise AnalyticsError(f"symbol_point must be a finite positive Decimal, got {symbol_point!r}.")
        if annualization_periods < 1:
            raise AnalyticsError(f"annualization_periods must be >= 1, got {annualization_periods}.")
        self._journal = journal
        self._config = config or (journal.config if journal is not None else PostTradeAnalyticsConfig())
        self._symbol_point = symbol_point
        self._annualization_periods = annualization_periods

    @property
    def config(self) -> PostTradeAnalyticsConfig:
        return self._config

    @property
    def journal(self) -> PersistentTradeJournal | None:
        return self._journal

    @property
    def symbol_point(self) -> Decimal:
        return self._symbol_point

    @property
    def annualization_periods(self) -> int:
        return self._annualization_periods

    def analyze_trades(
        self,
        trades: Sequence[CompletedTradeRecord],
        *,
        initial_balance: Decimal = Decimal("10000.00"),
        final_balance: Decimal | None = None,
        final_equity: Decimal | None = None,
        equity_curve: Sequence[EquityPoint] | None = None,
        symbol_point: Decimal | None = None,
    ) -> PerformanceAttributionReport:
        """Validate `trades` and compute a complete `PerformanceAttributionReport`."""
        pt = symbol_point if symbol_point is not None else self._symbol_point
        validated = _validate_and_normalize_trades(trades, symbol_point=pt)

        overall_metrics, overall_max_dd_r = compute_overall_backtest_metrics(
            validated,
            initial_balance=initial_balance,
            final_balance=final_balance,
            final_equity=final_equity,
            equity_curve=equity_curve,
            annualization_periods=self._annualization_periods,
            symbol_point=pt,
        )
        exec_quality = compute_execution_quality_report(validated, symbol_point=pt)

        verified_cnt = sum(
            1
            for t in validated
            if t.context_attribution_status == ContextAttributionStatus.VERIFIED_ENTRY_CONTEXT
        )
        unattributed_cnt = sum(
            1
            for t in validated
            if t.context_attribution_status == ContextAttributionStatus.UNATTRIBUTED_MISSING_CONTEXT
        )

        # Group trades across single-label and multi-label dimensions
        buckets: dict[tuple[str, str], list[CompletedTradeRecord]] = {}

        for tr in validated:
            # 1. Single-label dimensions (mutually exclusive & collectively exhaustive)
            single_pairs = (
                ("source", tr.source.value),
                ("timeframe", tr.timeframe.value),
                ("direction", tr.direction.value),
                ("strategy_id", tr.strategy_id),
                (
                    "regime",
                    tr.entry_context.regime.value
                    if tr.entry_context is not None
                    else MISSING_CONTEXT_SEGMENT_KEY,
                ),
                (
                    "killzone_session",
                    tr.entry_context.killzone_session.value
                    if tr.entry_context is not None
                    else MISSING_CONTEXT_SEGMENT_KEY,
                ),
            )
            for dim_name, seg_key in single_pairs:
                buckets.setdefault((dim_name, seg_key), []).append(tr)

            # 2. Multi-label dimension: contributing_strategy_id (deduplicated per trade)
            if tr.entry_context is None:
                contrib_strats: tuple[str, ...] = (MISSING_CONTEXT_SEGMENT_KEY,)
            elif tr.entry_context.contributing_strategy_ids:
                dedup_strats = tuple(
                    dict.fromkeys(
                        s.strip()
                        for s in tr.entry_context.contributing_strategy_ids
                        if s.strip()
                    )
                )
                contrib_strats = dedup_strats if dedup_strats else (tr.strategy_id,)
            else:
                contrib_strats = (tr.strategy_id,)

            for strat_key in contrib_strats:
                buckets.setdefault(("contributing_strategy_id", strat_key), []).append(tr)

            # 3. Multi-label dimension: knowledge_rule_id (deduplicated per trade)
            if tr.entry_context is None:
                rule_keys: tuple[str, ...] = (MISSING_CONTEXT_SEGMENT_KEY,)
            elif tr.entry_context.contributing_knowledge_rule_ids:
                dedup_rules = tuple(
                    dict.fromkeys(
                        r.strip()
                        for r in tr.entry_context.contributing_knowledge_rule_ids
                        if r.strip()
                    )
                )
                rule_keys = dedup_rules if dedup_rules else (NO_KNOWLEDGE_RULE_SEGMENT_KEY,)
            else:
                rule_keys = (NO_KNOWLEDGE_RULE_SEGMENT_KEY,)

            for rule_key in rule_keys:
                buckets.setdefault(("knowledge_rule_id", rule_key), []).append(tr)

        dim_order = (*SINGLE_LABEL_DIMENSIONS, *MULTI_LABEL_DIMENSIONS)
        dim_rank = {name: idx for idx, name in enumerate(dim_order)}

        sorted_keys = sorted(
            buckets.keys(),
            key=lambda pair: (dim_rank.get(pair[0], 999), pair[0], pair[1]),
        )

        breakdowns = tuple(
            compute_segment_breakdown(
                dim_name,
                seg_key,
                buckets[(dim_name, seg_key)],
                min_trades_threshold=self._config.min_trades_for_drift_evaluation,
                annualization_periods=self._annualization_periods,
                symbol_point=pt,
            )
            for dim_name, seg_key in sorted_keys
        )

        return PerformanceAttributionReport(
            overall_metrics=overall_metrics,
            overall_max_drawdown_r=overall_max_dd_r,
            verified_context_trades=verified_cnt,
            unattributed_missing_context_trades=unattributed_cnt,
            execution_quality=exec_quality,
            segment_breakdowns=breakdowns,
        )

    def analyze_journal(
        self,
        *,
        initial_balance: Decimal = Decimal("10000.00"),
        source_filter: TradeSource | None = None,
        symbol_point: Decimal | None = None,
    ) -> PerformanceAttributionReport:
        """Analyze all completed trades currently stored in the bound `PersistentTradeJournal`."""
        if self._journal is None:
            raise AnalyticsError(
                "PerformanceAttributionEngine.analyze_journal() requires a bound PersistentTradeJournal."
            )
        all_trades = self._journal.list_completed_trades()
        if source_filter is not None:
            filtered = tuple(t for t in all_trades if t.source == source_filter)
        else:
            filtered = all_trades
        return self.analyze_trades(
            filtered,
            initial_balance=initial_balance,
            symbol_point=symbol_point,
        )
