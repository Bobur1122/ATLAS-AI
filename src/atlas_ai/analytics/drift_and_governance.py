"""Analytics Drift Detection & Human-Gated Walk-Forward Promotion Governance for ATLAS AI Phase 8.3."""

from decimal import Decimal, ROUND_HALF_UP
from typing import Sequence

from atlas_ai.analytics.attribution import (
    MISSING_CONTEXT_SEGMENT_KEY,
    NO_KNOWLEDGE_RULE_SEGMENT_KEY,
    PerformanceAttributionEngine,
    validate_completed_trade_record,
)
from atlas_ai.analytics.config import PostTradeAnalyticsConfig
from atlas_ai.analytics.exceptions import (
    AnalyticsError,
    FeedbackLeakageError,
    GovernancePromotionRejectedError,
)
from atlas_ai.analytics.journal import PersistentTradeJournal
from atlas_ai.analytics.models import (
    CompletedTradeRecord,
    DriftAndHealthReport,
    DriftStatus,
    GovernancePromotionDecision,
    TradeSource,
)
from atlas_ai.domain.risk import RiskParameters
from atlas_ai.domain.signals import SignalDirection
from atlas_ai.models.exceptions import ModelDataLeakageError
from atlas_ai.models.schemas import (
    ChronologicalSplitIndices,
    ModelEvaluationReport,
    WalkForwardReport,
)
from atlas_ai.models.validation import validate_chronological_split

QUANTIZE_RATIO = Decimal("0.0001")
PROBABILITY_SUM_TOLERANCE = Decimal("0.0010")


def _is_finite_decimal(val: object) -> bool:
    return isinstance(val, Decimal) and val.is_finite()


def _validate_and_order_observation_window(
    trades: Sequence[CompletedTradeRecord],
    *,
    observation_window_size: int | None = None,
    symbol_point: Decimal = Decimal("0.01"),
) -> tuple[CompletedTradeRecord, ...]:
    """Validate `observation_window_size`, validate every trade record, ensure unique `trade_id`,
    sort chronologically, and slice the observation window safely.
    """
    if observation_window_size is not None:
        if (
            isinstance(observation_window_size, bool)
            or not isinstance(observation_window_size, int)
            or observation_window_size <= 0
        ):
            raise AnalyticsError(
                f"observation_window_size must be a positive integer (>= 1) or None, got {observation_window_size!r}."
            )

    seen_ids: set[str] = set()
    for tr in trades:
        validate_completed_trade_record(tr, symbol_point=symbol_point)
        if tr.trade_id in seen_ids:
            raise AnalyticsError(
                f"Duplicate trade_id '{tr.trade_id}' in trade sequence; refusing to double-count."
            )
        seen_ids.add(tr.trade_id)

    ordered = tuple(
        sorted(
            trades,
            key=lambda item: (item.exit_timestamp_utc, item.entry_timestamp_utc, item.trade_id),
        )
    )
    if observation_window_size is not None and len(ordered) > observation_window_size:
        return ordered[-observation_window_size:]
    return ordered


def compute_max_consecutive_losses(
    trades: Sequence[CompletedTradeRecord],
    *,
    symbol_point: Decimal = Decimal("0.01"),
) -> int:
    """Compute the maximum streak of consecutive losing trades (`net_pnl < Decimal("0.00")`)
    across a chronologically ordered sequence of validated trades.

    Note: Both winning trades (`net_pnl > 0`) and breakeven trades (`net_pnl == 0`)
    reset the consecutive loss counter to `0`.
    """
    ordered = _validate_and_order_observation_window(trades, symbol_point=symbol_point)
    current_streak = 0
    max_streak = 0
    for tr in ordered:
        if tr.net_pnl < Decimal("0.00"):
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
        else:
            current_streak = 0
    return max_streak


def compute_post_trade_calibration_metrics(
    trades: Sequence[CompletedTradeRecord],
    *,
    n_bins: int = 5,
    symbol_point: Decimal = Decimal("0.01"),
) -> tuple[Decimal, Decimal, int]:
    """Compute `(realized_brier_score, realized_ece, evaluated_ml_trade_count)` across closed trades
    whose `entry_context` contains ML probability predictions.

    Validation & Post-Trade Ground-Truth Mapping:
    - If a trade's `entry_context` has any of `(ml_sell_probability, ml_wait_probability, ml_buy_probability)`
      populated, all three must be non-None finite Decimals in `[0, 1]` summing to `1.0 +/- 0.0010`
      (else raises `AnalyticsError`).
    - For an executed directional trade (`BUY` or `SELL`):
      - If `trade.net_pnl > 0`: ground-truth class is the executed direction (`BUY` -> class 2, `SELL` -> class 0)
        and directional accuracy indicator `a_i = 1`.
      - If `trade.net_pnl <= 0`: the directional call did not profit (should have been filtered as `WAIT`),
        so ground-truth class is `WAIT` (class 1) and directional accuracy indicator `a_i = 0`.
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise AnalyticsError(f"n_bins must be a positive integer (>= 1), got {n_bins!r}.")

    ordered = _validate_and_order_observation_window(trades, symbol_point=symbol_point)

    ml_samples: list[tuple[Decimal, Decimal, Decimal, int, Decimal, Decimal]] = []
    # Each element: (p_sell, p_wait, p_buy, true_class_idx, exec_conf, acc_indicator)

    for tr in ordered:
        ctx = tr.entry_context
        if ctx is None:
            continue

        probs = (ctx.ml_sell_probability, ctx.ml_wait_probability, ctx.ml_buy_probability)
        non_none_count = sum(1 for p in probs if p is not None)
        if non_none_count == 0:
            if ctx.ml_predicted_direction is not None:
                raise AnalyticsError(
                    f"Inconsistent ML metadata on trade '{tr.trade_id}': ml_predicted_direction="
                    f"{ctx.ml_predicted_direction!r} is set but ML probabilities are None."
                )
            continue
        if non_none_count != 3:
            raise AnalyticsError(
                f"Partially populated ML probabilities on trade '{tr.trade_id}': all three of "
                f"(ml_sell_probability, ml_wait_probability, ml_buy_probability) must be present or all None."
            )

        if ctx.ml_predicted_direction == SignalDirection.WAIT:
            raise AnalyticsError(
                f"Invalid ml_predicted_direction='WAIT' on executed directional trade '{tr.trade_id}'; "
                "post-trade directional calibration requires BUY or SELL."
            )
        if (
            ctx.ml_predicted_direction is not None
            and ctx.ml_predicted_direction != tr.direction
        ):
            raise AnalyticsError(
                f"Inconsistent ml_predicted_direction='{ctx.ml_predicted_direction.value}' on "
                f"{tr.direction.value} trade '{tr.trade_id}'."
            )

        p_sell, p_wait, p_buy = probs[0], probs[1], probs[2]
        assert p_sell is not None and p_wait is not None and p_buy is not None
        for p_name, p_val in (
            ("ml_sell_probability", p_sell),
            ("ml_wait_probability", p_wait),
            ("ml_buy_probability", p_buy),
        ):
            if not _is_finite_decimal(p_val) or p_val < Decimal("0") or p_val > Decimal("1"):
                raise AnalyticsError(
                    f"Invalid {p_name}={p_val!r} on trade '{tr.trade_id}'; expected finite Decimal in [0, 1]."
                )

        prob_sum = p_sell + p_wait + p_buy
        if abs(prob_sum - Decimal("1.0000")) > PROBABILITY_SUM_TOLERANCE:
            raise AnalyticsError(
                f"ML probabilities on trade '{tr.trade_id}' sum to {prob_sum} "
                f"(expected 1.0000 +/- {PROBABILITY_SUM_TOLERANCE})."
            )

        is_win = tr.net_pnl > Decimal("0.00")
        if tr.direction == SignalDirection.BUY:
            exec_conf = p_buy
            true_cls = 2 if is_win else 1
        else:
            exec_conf = p_sell
            true_cls = 0 if is_win else 1

        acc_ind = Decimal("1.0000") if is_win else Decimal("0.0000")
        ml_samples.append((p_sell, p_wait, p_buy, true_cls, exec_conf, acc_ind))

    n_ml = len(ml_samples)
    if n_ml == 0:
        return Decimal("0.0000"), Decimal("0.0000"), 0

    n_ml_dec = Decimal(n_ml)

    # 1. Multiclass Brier Score: mean over samples of sum_{k in {0,1,2}} (p_{i,k} - y_{i,k})^2
    brier_total = Decimal("0")
    for p_sell, p_wait, p_buy, true_cls, _, _ in ml_samples:
        y0 = Decimal("1") if true_cls == 0 else Decimal("0")
        y1 = Decimal("1") if true_cls == 1 else Decimal("0")
        y2 = Decimal("1") if true_cls == 2 else Decimal("0")
        brier_total += (p_sell - y0) ** 2 + (p_wait - y1) ** 2 + (p_buy - y2) ** 2

    brier_score = (brier_total / n_ml_dec).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)

    # 2. Expected Calibration Error (ECE) across n_bins equal-width bins in [0, 1]
    ece_total = Decimal("0")
    n_bins_dec = Decimal(n_bins)
    for b in range(n_bins):
        lo = Decimal(b) / n_bins_dec
        hi = Decimal(b + 1) / n_bins_dec
        is_last = b == n_bins - 1
        bin_items = [
            (exec_conf, acc_ind)
            for _, _, _, _, exec_conf, acc_ind in ml_samples
            if (lo <= exec_conf <= hi if is_last else lo <= exec_conf < hi)
        ]
        if not bin_items:
            continue
        bin_cnt = Decimal(len(bin_items))
        bin_conf = sum((c for c, _ in bin_items), Decimal("0")) / bin_cnt
        bin_acc = sum((a for _, a in bin_items), Decimal("0")) / bin_cnt
        bin_weight = bin_cnt / n_ml_dec
        ece_total += bin_weight * abs(bin_acc - bin_conf)

    ece_score = min(
        Decimal("1.0000"),
        ece_total.quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP),
    )
    return brier_score, ece_score, n_ml


def validate_walk_forward_leakage_safety(
    report_or_split: WalkForwardReport | ChronologicalSplitIndices,
) -> tuple[int, int]:
    """Verify chronological ordering, non-empty folds, and mandatory purge separation
    (`purge_bars >= horizon_bars`) on a `WalkForwardReport` or `ChronologicalSplitIndices`.

    Returns `(purge_bars, horizon_bars)` if valid; raises `FeedbackLeakageError` if any
    anti-leakage invariant is violated.
    """
    if isinstance(report_or_split, ChronologicalSplitIndices):
        sp = report_or_split
        try:
            validate_chronological_split(
                train_start=sp.train_start,
                train_end=sp.train_end,
                val_start=sp.val_start,
                val_end=sp.val_end,
                test_start=sp.test_start,
                test_end=sp.test_end,
                horizon_bars=sp.horizon_bars,
                purge_bars=sp.purge_bars,
            )
        except ModelDataLeakageError as exc:
            raise FeedbackLeakageError(str(exc)) from exc

        if (
            sp.purge_1_start != sp.train_end + 1
            or sp.purge_1_end != sp.val_start - 1
            or (sp.purge_1_end - sp.purge_1_start + 1) < sp.purge_bars
        ):
            raise FeedbackLeakageError(
                f"LEAKAGE VIOLATION in purge_1 boundaries [{sp.purge_1_start}, {sp.purge_1_end}] "
                f"between train_end={sp.train_end} and val_start={sp.val_start} (purge_bars={sp.purge_bars})."
            )
        if (
            sp.purge_2_start != sp.val_end + 1
            or sp.purge_2_end != sp.test_start - 1
            or (sp.purge_2_end - sp.purge_2_start + 1) < sp.purge_bars
        ):
            raise FeedbackLeakageError(
                f"LEAKAGE VIOLATION in purge_2 boundaries [{sp.purge_2_start}, {sp.purge_2_end}] "
                f"between val_end={sp.val_end} and test_start={sp.test_start} (purge_bars={sp.purge_bars})."
            )
        return sp.purge_bars, sp.horizon_bars

    if isinstance(report_or_split, WalkForwardReport):
        wf = report_or_split
        horizon_bars = wf.config.labeling.horizon_bars
        purge_bars = wf.config.purge_bars

        if horizon_bars < 1 or purge_bars < 1:
            raise FeedbackLeakageError(
                f"LEAKAGE VIOLATION: horizon_bars ({horizon_bars}) and purge_bars ({purge_bars}) must be >= 1."
            )
        if purge_bars < horizon_bars:
            raise FeedbackLeakageError(
                f"LEAKAGE VIOLATION: purge_bars ({purge_bars}) must be >= horizon_bars ({horizon_bars})."
            )
        if not wf.folds:
            raise FeedbackLeakageError(
                "LEAKAGE VIOLATION: WalkForwardReport must contain at least one walk-forward fold (got 0 folds)."
            )

        prev_fold_idx: int | None = None
        prev_train_start: int | None = None
        prev_train_end: int | None = None
        prev_test_start: int | None = None
        prev_test_end: int | None = None

        for fold in wf.folds:
            if not (
                0
                <= fold.train_start_index
                <= fold.train_end_index
                < fold.purge_start_index
                <= fold.purge_end_index
                < fold.test_start_index
                <= fold.test_end_index
            ):
                raise FeedbackLeakageError(
                    f"LEAKAGE VIOLATION in fold {fold.fold_index}: non-chronological or overlapping indices "
                    f"train=[{fold.train_start_index}, {fold.train_end_index}], "
                    f"purge=[{fold.purge_start_index}, {fold.purge_end_index}], "
                    f"test=[{fold.test_start_index}, {fold.test_end_index}]."
                )
            if (
                fold.purge_start_index != fold.train_end_index + 1
                or fold.purge_end_index != fold.test_start_index - 1
            ):
                raise FeedbackLeakageError(
                    f"LEAKAGE VIOLATION in fold {fold.fold_index}: purge window "
                    f"[{fold.purge_start_index}, {fold.purge_end_index}] is not contiguous with "
                    f"train_end_index={fold.train_end_index} and test_start_index={fold.test_start_index}."
                )
            purged_gap = fold.test_start_index - fold.train_end_index - 1
            purge_span = fold.purge_end_index - fold.purge_start_index + 1
            if purged_gap < purge_bars or purge_span < purge_bars:
                raise FeedbackLeakageError(
                    f"LEAKAGE VIOLATION in fold {fold.fold_index}: purged gap ({purged_gap}) < "
                    f"required purge_bars ({purge_bars}) for horizon_bars ({horizon_bars})."
                )
            if prev_fold_idx is not None and fold.fold_index <= prev_fold_idx:
                raise FeedbackLeakageError(
                    f"LEAKAGE VIOLATION: non-increasing fold_index ({fold.fold_index} <= {prev_fold_idx})."
                )
            if prev_train_start is not None and fold.train_start_index < prev_train_start:
                raise FeedbackLeakageError(
                    f"LEAKAGE VIOLATION: fold {fold.fold_index} train_start_index ({fold.train_start_index}) "
                    f"regressed before previous fold train_start_index ({prev_train_start})."
                )
            if prev_train_end is not None and fold.train_end_index <= prev_train_end:
                raise FeedbackLeakageError(
                    f"LEAKAGE VIOLATION: fold {fold.fold_index} train_end_index ({fold.train_end_index}) "
                    f"does not advance past previous fold train_end_index ({prev_train_end})."
                )
            if prev_test_start is not None and fold.test_start_index <= prev_test_start:
                raise FeedbackLeakageError(
                    f"LEAKAGE VIOLATION: fold {fold.fold_index} test_start_index ({fold.test_start_index}) "
                    f"does not advance past previous fold test_start_index ({prev_test_start})."
                )
            if prev_test_end is not None and fold.test_start_index <= prev_test_end:
                raise FeedbackLeakageError(
                    f"LEAKAGE VIOLATION: overlapping out-of-sample test windows across folds; "
                    f"fold {fold.fold_index} test_start_index ({fold.test_start_index}) <= "
                    f"previous fold test_end_index ({prev_test_end})."
                )
            prev_fold_idx = fold.fold_index
            prev_train_start = fold.train_start_index
            prev_train_end = fold.train_end_index
            prev_test_start = fold.test_start_index
            prev_test_end = fold.test_end_index

        return purge_bars, horizon_bars

    raise AnalyticsError(
        f"Expected WalkForwardReport or ChronologicalSplitIndices, got {type(report_or_split).__name__}."
    )


def validate_risk_parameters_no_relaxation(
    *,
    baseline: RiskParameters,
    candidate: RiskParameters,
) -> None:
    """Ensure `candidate` RiskParameters do not relax any of the 7 Phase 1 safety ceilings
    relative to `baseline`.

    Raises `GovernancePromotionRejectedError` immediately if any ceiling is relaxed.
    """
    violations: list[str] = []

    if candidate.risk_per_trade_percent > baseline.risk_per_trade_percent:
        violations.append(
            f"risk_per_trade_percent increased from {baseline.risk_per_trade_percent}% "
            f"to {candidate.risk_per_trade_percent}%"
        )
    if candidate.max_daily_loss_percent > baseline.max_daily_loss_percent:
        violations.append(
            f"max_daily_loss_percent increased from {baseline.max_daily_loss_percent}% "
            f"to {candidate.max_daily_loss_percent}%"
        )
    if candidate.max_total_drawdown_percent > baseline.max_total_drawdown_percent:
        violations.append(
            f"max_total_drawdown_percent increased from {baseline.max_total_drawdown_percent}% "
            f"to {candidate.max_total_drawdown_percent}%"
        )
    if candidate.max_open_positions > baseline.max_open_positions:
        violations.append(
            f"max_open_positions increased from {baseline.max_open_positions} "
            f"to {candidate.max_open_positions}"
        )
    if candidate.minimum_risk_reward_ratio < baseline.minimum_risk_reward_ratio:
        violations.append(
            f"minimum_risk_reward_ratio reduced from {baseline.minimum_risk_reward_ratio} "
            f"to {candidate.minimum_risk_reward_ratio}"
        )
    if candidate.max_position_size > baseline.max_position_size:
        violations.append(
            f"max_position_size increased from {baseline.max_position_size} lots "
            f"to {candidate.max_position_size} lots"
        )
    if not baseline.allow_trading and candidate.allow_trading:
        violations.append("allow_trading kill-switch relaxed from False to True")

    if violations:
        raise GovernancePromotionRejectedError(
            "SAFETY CEILING VIOLATION: Candidate RiskParameters attempt to relax immutable "
            f"Phase 1 risk boundaries: {'; '.join(violations)}."
        )


class DriftDetectionEngine:
    """Deterministic Phase 8.3 engine evaluating expectancy drift, consecutive loss streaks,
    execution slippage degradation, ML probability calibration drift (Brier & ECE),
    and advisory knowledge-rule quarantine recommendations.

    Read-Only Guarantee:
    - Never deletes, disables, or mutates rules in `KnowledgeRepository`.
    - Never modifies execution or strategy state.
    """

    def __init__(
        self,
        *,
        config: PostTradeAnalyticsConfig | None = None,
        journal: PersistentTradeJournal | None = None,
        attribution_engine: PerformanceAttributionEngine | None = None,
        symbol_point: Decimal = Decimal("0.01"),
        annualization_periods: int = 252,
    ) -> None:
        if not _is_finite_decimal(symbol_point) or symbol_point <= Decimal("0"):
            raise AnalyticsError(f"symbol_point must be a finite positive Decimal, got {symbol_point!r}.")
        if annualization_periods < 1:
            raise AnalyticsError(f"annualization_periods must be >= 1, got {annualization_periods}.")

        self._journal = journal
        self._config = config or (
            journal.config if journal is not None else PostTradeAnalyticsConfig()
        )
        self._symbol_point = symbol_point
        self._annualization_periods = annualization_periods
        self._attribution_engine = attribution_engine or PerformanceAttributionEngine(
            config=self._config,
            journal=self._journal,
            symbol_point=self._symbol_point,
            annualization_periods=self._annualization_periods,
        )

    @property
    def config(self) -> PostTradeAnalyticsConfig:
        return self._config

    @property
    def journal(self) -> PersistentTradeJournal | None:
        return self._journal

    @property
    def attribution_engine(self) -> PerformanceAttributionEngine:
        return self._attribution_engine

    def evaluate_trades(
        self,
        trades: Sequence[CompletedTradeRecord],
        *,
        baseline_evaluation: ModelEvaluationReport | WalkForwardReport | None = None,
        observation_window_size: int | None = None,
        initial_balance: Decimal = Decimal("10000.00"),
        symbol_point: Decimal | None = None,
    ) -> DriftAndHealthReport:
        """Evaluate an observation window of closed trades and return a deterministic `DriftAndHealthReport`."""
        pt = symbol_point if symbol_point is not None else self._symbol_point
        window_trades = _validate_and_order_observation_window(
            trades,
            observation_window_size=observation_window_size,
            symbol_point=pt,
        )

        attr_report = self._attribution_engine.analyze_trades(
            window_trades,
            initial_balance=initial_balance,
            symbol_point=pt,
        )
        overall_metrics = attr_report.overall_metrics
        total_trades = overall_metrics.total_trades
        min_trades = self._config.min_trades_for_drift_evaluation
        insufficient_sample = total_trades < min_trades

        realized_expectancy_r = overall_metrics.expectancy_r
        mean_entry_slippage = attr_report.execution_quality.mean_entry_slippage_points
        max_consec_losses = compute_max_consecutive_losses(window_trades, symbol_point=pt)

        realized_brier, realized_ece, n_ml = compute_post_trade_calibration_metrics(
            window_trades,
            symbol_point=pt,
        )

        base_eval: ModelEvaluationReport | None = None
        if isinstance(baseline_evaluation, WalkForwardReport):
            base_eval = baseline_evaluation.aggregated_out_of_sample
        elif isinstance(baseline_evaluation, ModelEvaluationReport):
            base_eval = baseline_evaluation
        elif baseline_evaluation is not None:
            raise AnalyticsError(
                f"Invalid baseline_evaluation type {type(baseline_evaluation).__name__}; "
                "expected ModelEvaluationReport, WalkForwardReport, or None."
            )

        # Compute calibration drift vs baseline only when N_ML >= min_trades_for_drift_evaluation
        if base_eval is not None and n_ml >= min_trades:
            brier_drift = (realized_brier - base_eval.multiclass_brier_score).quantize(
                QUANTIZE_RATIO, rounding=ROUND_HALF_UP
            )
            ece_drift = (realized_ece - base_eval.calibration_error_ece).quantize(
                QUANTIZE_RATIO, rounding=ROUND_HALF_UP
            )
        else:
            brier_drift = Decimal("0.0000")
            ece_drift = Decimal("0.0000")

        # Minimum sample gate suppresses false positives when portfolio total_trades < min_trades
        if insufficient_sample:
            return DriftAndHealthReport(
                overall_status=DriftStatus.INSUFFICIENT_SAMPLE,
                total_trades=total_trades,
                insufficient_sample=True,
                realized_expectancy_r=realized_expectancy_r,
                max_consecutive_losses=max_consec_losses,
                realized_brier_score=realized_brier,
                realized_ece=realized_ece,
                brier_drift_from_baseline=brier_drift,
                ece_drift_from_baseline=ece_drift,
                mean_entry_slippage_points=mean_entry_slippage,
                quarantine_recommended_rule_ids=(),
                warnings=(),
                overall_metrics=overall_metrics,
            )

        warnings: list[str] = []
        expectancy_degraded = False
        streak_degraded = False
        execution_degraded = False

        # 1. Expectancy drift check
        if realized_expectancy_r < self._config.min_expectancy_r_threshold:
            expectancy_degraded = True
            warnings.append(
                f"Expectancy drift detected: realized_expectancy_r ({realized_expectancy_r}) "
                f"< min_expectancy_r_threshold ({self._config.min_expectancy_r_threshold})."
            )

        # 2. Consecutive losing streak check
        if max_consec_losses >= self._config.max_consecutive_losses_alert:
            streak_degraded = True
            warnings.append(
                f"Consecutive loss alert: max_consecutive_losses ({max_consec_losses}) "
                f">= max_consecutive_losses_alert ({self._config.max_consecutive_losses_alert})."
            )

        # 3. Execution quality (adverse entry slippage) check
        if mean_entry_slippage > self._config.max_acceptable_mean_slippage_points:
            execution_degraded = True
            warnings.append(
                f"Execution quality degraded: mean_entry_slippage_points ({mean_entry_slippage}) "
                f"> max_acceptable_mean_slippage_points ({self._config.max_acceptable_mean_slippage_points})."
            )

        # 4. ML calibration drift checks (only when N_ML >= min_trades and baseline provided)
        if base_eval is not None and n_ml >= min_trades:
            if ece_drift > self._config.max_acceptable_ece_drift:
                warnings.append(
                    f"ML ECE calibration drift detected: ece_drift_from_baseline ({ece_drift}) "
                    f"> max_acceptable_ece_drift ({self._config.max_acceptable_ece_drift}) "
                    f"(realized_ece={realized_ece}, baseline_ece={base_eval.calibration_error_ece})."
                )
            if brier_drift > self._config.max_acceptable_brier_drift:
                warnings.append(
                    f"ML Brier calibration drift detected: brier_drift_from_baseline ({brier_drift}) "
                    f"> max_acceptable_brier_drift ({self._config.max_acceptable_brier_drift}) "
                    f"(realized_brier={realized_brier}, baseline_brier={base_eval.multiclass_brier_score})."
                )

        # 5. Knowledge rule quarantine recommendations (advisory only)
        quarantined_rules: list[str] = []
        for rule_seg in attr_report.list_dimension("knowledge_rule_id"):
            if rule_seg.segment_key in (
                MISSING_CONTEXT_SEGMENT_KEY,
                NO_KNOWLEDGE_RULE_SEGMENT_KEY,
            ):
                continue
            if rule_seg.insufficient_sample:
                continue
            if (
                rule_seg.expectancy_r < self._config.min_expectancy_r_threshold
                or (
                    self._config.min_expectancy_r_threshold == Decimal("0.0000")
                    and rule_seg.net_profit_usd < Decimal("0.00")
                )
            ):
                quarantined_rules.append(rule_seg.segment_key)
                warnings.append(
                    f"Knowledge rule quarantine recommended for '{rule_seg.segment_key}': "
                    f"total_trades={rule_seg.total_trades}, expectancy_r={rule_seg.expectancy_r}, "
                    f"net_profit_usd={rule_seg.net_profit_usd}."
                )

        sorted_quarantined = tuple(sorted(set(quarantined_rules)))

        # Deterministic severity hierarchy
        if sorted_quarantined or (expectancy_degraded and streak_degraded):
            overall_status = DriftStatus.QUARANTINE_RECOMMENDED
        elif execution_degraded:
            overall_status = DriftStatus.EXECUTION_DEGRADED
        elif warnings:
            overall_status = DriftStatus.WARNING_DRIFT_DETECTED
        else:
            overall_status = DriftStatus.HEALTHY

        return DriftAndHealthReport(
            overall_status=overall_status,
            total_trades=total_trades,
            insufficient_sample=False,
            realized_expectancy_r=realized_expectancy_r,
            max_consecutive_losses=max_consec_losses,
            realized_brier_score=realized_brier,
            realized_ece=realized_ece,
            brier_drift_from_baseline=brier_drift,
            ece_drift_from_baseline=ece_drift,
            mean_entry_slippage_points=mean_entry_slippage,
            quarantine_recommended_rule_ids=sorted_quarantined,
            warnings=tuple(warnings),
            overall_metrics=overall_metrics,
        )

    def evaluate_journal(
        self,
        *,
        baseline_evaluation: ModelEvaluationReport | WalkForwardReport | None = None,
        observation_window_size: int | None = None,
        source_filter: TradeSource | None = None,
        initial_balance: Decimal = Decimal("10000.00"),
        symbol_point: Decimal | None = None,
    ) -> DriftAndHealthReport:
        """Evaluate completed trades stored in the bound `PersistentTradeJournal`."""
        if self._journal is None:
            raise AnalyticsError(
                "DriftDetectionEngine.evaluate_journal() requires a bound PersistentTradeJournal."
            )
        all_trades = self._journal.list_completed_trades()
        if source_filter is not None:
            filtered = tuple(t for t in all_trades if t.source == source_filter)
        else:
            filtered = all_trades
        return self.evaluate_trades(
            filtered,
            baseline_evaluation=baseline_evaluation,
            observation_window_size=observation_window_size,
            initial_balance=initial_balance,
            symbol_point=symbol_point,
        )


class GovernancePromotionEvaluator:
    """Human-gated Phase 8.3 promotion governance evaluator for candidate Phase 6 classifiers
    and configuration updates.

    Core Safety Invariants:
    1. Mandatory Anti-Leakage Gate: verifies `purge_bars >= horizon_bars`, non-empty folds,
       and strict chronological train/purge/test index separation (raises `FeedbackLeakageError`).
    2. Mandatory Immutable Risk Ceiling Gate: blocks any relaxation of Phase 1 `RiskParameters`
       (raises `GovernancePromotionRejectedError`).
    3. Out-of-Sample Walk-Forward Quality Gates: checks active signal count, positive & superior
       expectancy, ECE calibration, Brier score stability, and maximum R drawdown.
    4. Mandatory Human Approval: `GovernancePromotionDecision.requires_human_approval` is
       invariantly `True`; never auto-promotes or overwrites active model artifacts.
    """

    def __init__(self, config: PostTradeAnalyticsConfig | None = None) -> None:
        self._config = config or PostTradeAnalyticsConfig()

    @property
    def config(self) -> PostTradeAnalyticsConfig:
        return self._config

    def evaluate_candidate(
        self,
        *,
        candidate_id: str,
        baseline_walk_forward: WalkForwardReport | ModelEvaluationReport,
        candidate_walk_forward: WalkForwardReport | ModelEvaluationReport,
        split_indices: ChronologicalSplitIndices | None = None,
        baseline_risk_parameters: RiskParameters | None = None,
        candidate_risk_parameters: RiskParameters | None = None,
        max_acceptable_drawdown_r: Decimal | None = None,
        strict_raise: bool = False,
    ) -> GovernancePromotionDecision:
        """Evaluate a candidate model's out-of-sample walk-forward report against baseline
        and emit an advisory, human-gated `GovernancePromotionDecision`.
        """
        cleaned_id = candidate_id.strip()
        if not cleaned_id:
            raise AnalyticsError("candidate_id must be a non-empty string.")

        # 1. Immutable Risk-Parameter Ceiling Gate (M-1: require both together if either is supplied)
        if (baseline_risk_parameters is None) != (candidate_risk_parameters is None):
            raise AnalyticsError(
                "baseline_risk_parameters and candidate_risk_parameters must both be provided together "
                "(or both omitted)."
            )
        if baseline_risk_parameters is not None and candidate_risk_parameters is not None:
            validate_risk_parameters_no_relaxation(
                baseline=baseline_risk_parameters,
                candidate=candidate_risk_parameters,
            )

        # 2. Anti-Leakage Chronological & Purge Separation Gate (M-3)
        sp_purge: int | None = None
        sp_horizon: int | None = None
        if split_indices is not None:
            sp_purge, sp_horizon = validate_walk_forward_leakage_safety(split_indices)

        if isinstance(baseline_walk_forward, WalkForwardReport):
            base_purge, base_horizon = validate_walk_forward_leakage_safety(baseline_walk_forward)
            if sp_purge is not None and (
                sp_purge != base_purge or sp_horizon != base_horizon
            ):
                raise FeedbackLeakageError(
                    f"LEAKAGE VIOLATION: split_indices (purge_bars={sp_purge}, horizon_bars={sp_horizon}) "
                    f"conflicts with baseline WalkForwardReport (purge_bars={base_purge}, horizon_bars={base_horizon})."
                )
            base_eval = baseline_walk_forward.aggregated_out_of_sample
        elif isinstance(baseline_walk_forward, ModelEvaluationReport):
            if split_indices is None:
                raise FeedbackLeakageError(
                    "LEAKAGE VIOLATION: Evaluating a standalone baseline ModelEvaluationReport "
                    "requires verified ChronologicalSplitIndices (split_indices) to prove purge_bars >= horizon_bars."
                )
            base_eval = baseline_walk_forward
        else:
            raise AnalyticsError(
                f"Unexpected baseline_walk_forward type: {type(baseline_walk_forward).__name__}."
            )

        if isinstance(candidate_walk_forward, WalkForwardReport):
            purge_bars, horizon_bars = validate_walk_forward_leakage_safety(
                candidate_walk_forward
            )
            if sp_purge is not None and (
                sp_purge != purge_bars or sp_horizon != horizon_bars
            ):
                raise FeedbackLeakageError(
                    f"LEAKAGE VIOLATION: split_indices (purge_bars={sp_purge}, horizon_bars={sp_horizon}) "
                    f"conflicts with candidate WalkForwardReport (purge_bars={purge_bars}, horizon_bars={horizon_bars})."
                )
            cand_eval = candidate_walk_forward.aggregated_out_of_sample
        elif isinstance(candidate_walk_forward, ModelEvaluationReport):
            if split_indices is None or sp_purge is None or sp_horizon is None:
                raise FeedbackLeakageError(
                    "LEAKAGE VIOLATION: Evaluating a standalone candidate ModelEvaluationReport "
                    "requires verified ChronologicalSplitIndices (split_indices) to prove purge_bars >= horizon_bars."
                )
            purge_bars = sp_purge
            horizon_bars = sp_horizon
            cand_eval = candidate_walk_forward
        else:
            raise AnalyticsError(
                f"Unexpected candidate_walk_forward type: {type(candidate_walk_forward).__name__}."
            )

        # 3. Out-of-Sample Walk-Forward Promotion Quality Gates
        rejection_reasons: list[str] = []
        min_signals = self._config.min_trades_for_drift_evaluation

        # Gate A: Minimum out-of-sample active directional signals
        if cand_eval.active_directional_signals < min_signals:
            rejection_reasons.append(
                f"Insufficient out-of-sample active directional signals: "
                f"{cand_eval.active_directional_signals} < required {min_signals}."
            )

        # Gate B: Positive & superior R-multiple expectancy
        required_exp_r = max(
            self._config.min_expectancy_r_threshold,
            base_eval.expected_r_per_signal,
        )
        if cand_eval.expected_r_per_signal <= required_exp_r:
            rejection_reasons.append(
                f"Candidate expected_r_per_signal ({cand_eval.expected_r_per_signal}) does not exceed "
                f"required baseline/threshold ({required_exp_r})."
            )

        # Gate C: Expected Calibration Error (ECE) gate
        max_allowed_ece = (
            base_eval.calibration_error_ece + self._config.max_acceptable_ece_drift
        ).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)
        if cand_eval.calibration_error_ece > max_allowed_ece:
            rejection_reasons.append(
                f"Candidate calibration_error_ece ({cand_eval.calibration_error_ece}) exceeds "
                f"maximum allowed ECE ({max_allowed_ece} = baseline {base_eval.calibration_error_ece} "
                f"+ tolerance {self._config.max_acceptable_ece_drift})."
            )

        # Gate D: Multiclass Brier Score stability gate
        max_allowed_brier = (
            base_eval.multiclass_brier_score + self._config.max_acceptable_brier_drift
        ).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)
        if cand_eval.multiclass_brier_score > max_allowed_brier:
            rejection_reasons.append(
                f"Candidate multiclass_brier_score ({cand_eval.multiclass_brier_score}) exceeds "
                f"maximum allowed Brier score ({max_allowed_brier} = baseline {base_eval.multiclass_brier_score} "
                f"+ tolerance {self._config.max_acceptable_brier_drift})."
            )

        # Gate E: Simulated maximum R drawdown gate
        if max_acceptable_drawdown_r is not None:
            if not _is_finite_decimal(max_acceptable_drawdown_r) or max_acceptable_drawdown_r < Decimal("0"):
                raise AnalyticsError(
                    f"max_acceptable_drawdown_r must be a non-negative finite Decimal, got {max_acceptable_drawdown_r!r}."
                )
            dd_cap = max_acceptable_drawdown_r.quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)
        else:
            dd_cap = max(
                base_eval.simulated_max_drawdown_r,
                Decimal("1.0000"),
            ).quantize(QUANTIZE_RATIO, rounding=ROUND_HALF_UP)

        if cand_eval.simulated_max_drawdown_r > dd_cap:
            rejection_reasons.append(
                f"Candidate simulated_max_drawdown_r ({cand_eval.simulated_max_drawdown_r}) exceeds "
                f"maximum acceptable R drawdown ({dd_cap})."
            )

        approved = len(rejection_reasons) == 0
        if approved:
            rationale = (
                f"Candidate '{cleaned_id}' passed all 5 leakage-safe out-of-sample walk-forward "
                f"promotion gates (expectancy_r={cand_eval.expected_r_per_signal} > {required_exp_r}, "
                f"ece={cand_eval.calibration_error_ece} <= {max_allowed_ece}, "
                f"max_dd_r={cand_eval.simulated_max_drawdown_r} <= {dd_cap}, "
                f"purge_bars={purge_bars} >= horizon_bars={horizon_bars}). "
                "Eligible for human review; explicit human approval is required prior to any deployment."
            )
        else:
            rationale = (
                f"Candidate '{cleaned_id}' rejected by out-of-sample walk-forward promotion gates: "
                f"{' '.join(rejection_reasons)}"
            )

        decision = GovernancePromotionDecision(
            candidate_id=cleaned_id,
            approved_for_human_review=approved,
            requires_human_approval=True,
            baseline_expectancy_r=base_eval.expected_r_per_signal,
            candidate_expectancy_r=cand_eval.expected_r_per_signal,
            baseline_ece=base_eval.calibration_error_ece,
            candidate_ece=cand_eval.calibration_error_ece,
            candidate_max_drawdown_r=cand_eval.simulated_max_drawdown_r,
            purge_bars=purge_bars,
            horizon_bars=horizon_bars,
            rejection_reasons=tuple(rejection_reasons),
            rationale=rationale,
        )

        if strict_raise and not approved:
            raise GovernancePromotionRejectedError(decision.rationale)

        return decision
