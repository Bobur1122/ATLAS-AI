"""Leakage-safe chronological splitting, walk-forward validation, and trading-aware ML evaluation."""

from decimal import Decimal, ROUND_HALF_UP
import math
from typing import Sequence

import numpy as np
import pandas as pd

from atlas_ai.domain.signals import SignalDirection
from atlas_ai.models.classifiers import (
    BaseMLClassifier,
    GradientBoostedStumpEnsemble,
    RegularizedLogisticClassifier,
)
from atlas_ai.models.config import ModelTrainingConfig, WalkForwardConfig
from atlas_ai.models.exceptions import ModelConfigurationError, ModelDataLeakageError
from atlas_ai.models.features import MLFeatureBuilder
from atlas_ai.models.labeling import TripleBarrierLabeler
from atlas_ai.models.schemas import (
    ChronologicalSplitIndices,
    ClassClassificationMetrics,
    ModelEvaluationReport,
    ModelType,
    TripleBarrierLabel,
    WalkForwardFoldResult,
    WalkForwardReport,
    direction_to_class_index,
)

QUANTIZE_METRIC = Decimal("0.0001")


def validate_chronological_split(
    *,
    train_start: int,
    train_end: int,
    val_start: int,
    val_end: int,
    test_start: int,
    test_end: int,
    horizon_bars: int,
    purge_bars: int,
    total_labelable_bars: int | None = None,
) -> ChronologicalSplitIndices:
    """Validate chronological train/validation/test index boundaries and enforce anti-leakage purging.

    Invariants enforced:
    1. `0 <= train_start <= train_end < val_start <= val_end < test_start <= test_end`.
    2. `purge_bars >= horizon_bars`.
    3. `val_start - train_end > purge_bars` (i.e. at least `purge_bars` indices dropped between
       `train_end` and `val_start` so `train_end + horizon_bars < val_start`).
    4. `test_start - val_end > purge_bars` (at least `purge_bars` indices dropped between
       `val_end` and `test_start` so `val_end + horizon_bars < test_start`).
    """
    if purge_bars < horizon_bars:
        raise ModelDataLeakageError(
            f"LEAKAGE VIOLATION: purge_bars ({purge_bars}) must be >= horizon_bars ({horizon_bars})."
        )
    if train_start < 0 or train_end < train_start:
        raise ModelDataLeakageError(
            f"Invalid training range [{train_start}, {train_end}]."
        )
    if val_end < val_start:
        raise ModelDataLeakageError(
            f"Invalid validation range [{val_start}, {val_end}]."
        )
    if test_end < test_start:
        raise ModelDataLeakageError(
            f"Invalid test range [{test_start}, {test_end}]."
        )

    gap_train_val = val_start - train_end - 1
    if gap_train_val < purge_bars:
        raise ModelDataLeakageError(
            f"LEAKAGE VIOLATION between train_end ({train_end}) and val_start ({val_start}): "
            f"purged gap is {gap_train_val} bars, which is less than required purge_bars ({purge_bars}) "
            f"for horizon_bars ({horizon_bars})."
        )

    gap_val_test = test_start - val_end - 1
    if gap_val_test < purge_bars:
        raise ModelDataLeakageError(
            f"LEAKAGE VIOLATION between val_end ({val_end}) and test_start ({test_start}): "
            f"purged gap is {gap_val_test} bars, which is less than required purge_bars ({purge_bars}) "
            f"for horizon_bars ({horizon_bars})."
        )

    if total_labelable_bars is not None and test_end >= total_labelable_bars:
        raise ModelDataLeakageError(
            f"test_end ({test_end}) exceeds maximum complete-horizon labelable index "
            f"({total_labelable_bars - 1})."
        )

    return ChronologicalSplitIndices(
        train_start=train_start,
        train_end=train_end,
        purge_1_start=train_end + 1,
        purge_1_end=val_start - 1,
        val_start=val_start,
        val_end=val_end,
        purge_2_start=val_end + 1,
        purge_2_end=test_start - 1,
        test_start=test_start,
        test_end=test_end,
        horizon_bars=horizon_bars,
        purge_bars=purge_bars,
    )


def evaluate_predictions(
    *,
    split_name: str,
    y_true: np.ndarray,
    proba_matrix: np.ndarray,
    labels: Sequence[TripleBarrierLabel],
    rr_ratio: Decimal = Decimal("2.00"),
    probability_threshold: Decimal = Decimal("0.45"),
    directional_margin: Decimal = Decimal("0.10"),
) -> ModelEvaluationReport:
    """Compute per-class (BUY, SELL, WAIT) precision/recall/F1, class imbalance, Brier score,
    Expected Calibration Error (ECE), and R-multiple trading expectancy / drawdown / Sharpe.
    """
    n = len(y_true)
    if n == 0 or proba_matrix.shape[0] != n:
        raise ModelConfigurationError("evaluate_predictions requires non-empty aligned y_true and proba_matrix.")

    thresh_f = float(probability_threshold)
    margin_f = float(directional_margin)

    # Convert probability matrix into predicted class indices applying threshold and margin gates
    y_pred = np.ones(n, dtype=int)  # Default 1 = WAIT
    for i in range(n):
        p_sell, p_wait, p_buy = float(proba_matrix[i, 0]), float(proba_matrix[i, 1]), float(proba_matrix[i, 2])
        margin = abs(p_buy - p_sell)
        if p_buy >= p_sell and p_buy >= p_wait and p_buy >= thresh_f and margin >= margin_f:
            y_pred[i] = 2  # BUY
        elif p_sell > p_buy and p_sell >= p_wait and p_sell >= thresh_f and margin >= margin_f:
            y_pred[i] = 0  # SELL
        else:
            y_pred[i] = 1  # WAIT

    sell_metrics = _compute_single_class_metrics(SignalDirection.SELL, 0, y_true, y_pred)
    wait_metrics = _compute_single_class_metrics(SignalDirection.WAIT, 1, y_true, y_pred)
    buy_metrics = _compute_single_class_metrics(SignalDirection.BUY, 2, y_true, y_pred)

    macro_f1 = (
        (sell_metrics.f1_score + wait_metrics.f1_score + buy_metrics.f1_score) / Decimal("3")
    ).quantize(QUANTIZE_METRIC, rounding=ROUND_HALF_UP)

    class_dist = {
        SignalDirection.BUY.value: buy_metrics.support,
        SignalDirection.SELL.value: sell_metrics.support,
        SignalDirection.WAIT.value: wait_metrics.support,
    }
    non_zero_counts = [c for c in class_dist.values() if c > 0]
    if len(non_zero_counts) >= 2:
        imbalance = (
            Decimal(max(non_zero_counts)) / Decimal(min(non_zero_counts))
        ).quantize(QUANTIZE_METRIC, rounding=ROUND_HALF_UP)
    else:
        imbalance = Decimal("1.0000")

    # Multiclass Brier Score: mean over samples of sum_k (p_{i,k} - y_{i,k})^2
    y_onehot = np.zeros((n, 3), dtype="float64")
    for i, c_idx in enumerate(y_true.astype(int)):
        y_onehot[i, c_idx] = 1.0
    brier_raw = float(np.mean(np.sum((proba_matrix - y_onehot) ** 2, axis=1)))
    brier_dec = Decimal(str(round(brier_raw, 6))).quantize(QUANTIZE_METRIC, rounding=ROUND_HALF_UP)

    # Expected Calibration Error (ECE) across 5 probability bins on predicted class confidence
    max_probs = np.max(proba_matrix, axis=1)
    argmax_cls = np.argmax(proba_matrix, axis=1)
    correct_mask = (argmax_cls == y_true).astype("float64")
    ece_val = _compute_ece(max_probs, correct_mask, n_bins=5)
    ece_dec = Decimal(str(round(ece_val, 6))).quantize(QUANTIZE_METRIC, rounding=ROUND_HALF_UP)

    # Active directional trading evaluation (BUY and SELL calls only)
    active_indices = [i for i in range(n) if y_pred[i] in (0, 2)]
    active_count = len(active_indices)
    win_probs: list[float] = []
    loss_probs: list[float] = []
    r_outcomes: list[float] = []
    rr_f = float(rr_ratio)

    active_hits = 0
    for i in active_indices:
        pred_cls = int(y_pred[i])
        true_cls = int(y_true[i])
        conf_p = float(proba_matrix[i, pred_cls])
        lbl = labels[i]

        # Determine realized R outcome under Triple-Barrier rules:
        # If predicted BUY (2) and long_succeeded -> +RR; if long_resolution_bar hit SL -> -1.0R; else 0.0R
        if pred_cls == 2:
            if lbl.long_succeeded and true_cls == 2:
                active_hits += 1
                win_probs.append(conf_p)
                r_outcomes.append(rr_f)
            elif lbl.long_resolution_bar is not None and not lbl.long_succeeded:
                loss_probs.append(conf_p)
                r_outcomes.append(-1.0)
            else:
                loss_probs.append(conf_p)
                r_outcomes.append(-0.25)
        else:
            if lbl.short_succeeded and true_cls == 0:
                active_hits += 1
                win_probs.append(conf_p)
                r_outcomes.append(rr_f)
            elif lbl.short_resolution_bar is not None and not lbl.short_succeeded:
                loss_probs.append(conf_p)
                r_outcomes.append(-1.0)
            else:
                loss_probs.append(conf_p)
                r_outcomes.append(-0.25)

    active_prec = (
        (Decimal(active_hits) / Decimal(active_count)).quantize(QUANTIZE_METRIC, rounding=ROUND_HALF_UP)
        if active_count > 0
        else Decimal("0.0000")
    )
    mean_win_p = (
        Decimal(str(round(sum(win_probs) / len(win_probs), 4))).quantize(QUANTIZE_METRIC)
        if win_probs
        else Decimal("0.0000")
    )
    mean_loss_p = (
        Decimal(str(round(sum(loss_probs) / len(loss_probs), 4))).quantize(QUANTIZE_METRIC)
        if loss_probs
        else Decimal("0.0000")
    )

    if r_outcomes:
        exp_r = Decimal(str(round(sum(r_outcomes) / len(r_outcomes), 4))).quantize(
            QUANTIZE_METRIC, rounding=ROUND_HALF_UP
        )
        gross_pos_r = sum(r for r in r_outcomes if r > 0.0)
        gross_neg_r = abs(sum(r for r in r_outcomes if r < 0.0))
        if gross_neg_r > 1e-9:
            pf_r = Decimal(str(round(gross_pos_r / gross_neg_r, 4))).quantize(QUANTIZE_METRIC)
        elif gross_pos_r > 0.0:
            pf_r = Decimal(str(round(gross_pos_r, 4))).quantize(QUANTIZE_METRIC)
        else:
            pf_r = Decimal("0.0000")

        # Cumulative R drawdown and Sharpe ratio
        cum_r = 0.0
        peak_r = 0.0
        max_dd_r = 0.0
        for r in r_outcomes:
            cum_r += r
            if cum_r > peak_r:
                peak_r = cum_r
            dd = peak_r - cum_r
            if dd > max_dd_r:
                max_dd_r = dd
        max_dd_dec = Decimal(str(round(max_dd_r, 4))).quantize(QUANTIZE_METRIC)

        if len(r_outcomes) >= 2:
            mean_r = sum(r_outcomes) / len(r_outcomes)
            var_r = sum((r - mean_r) ** 2 for r in r_outcomes) / (len(r_outcomes) - 1)
            std_r = math.sqrt(var_r)
            sharpe_val = (mean_r / std_r) * math.sqrt(252.0) if std_r > 1e-9 else 0.0
            sharpe_dec = Decimal(str(round(sharpe_val, 4))).quantize(QUANTIZE_METRIC)
        else:
            sharpe_dec = Decimal("0.0000")
    else:
        exp_r = Decimal("0.0000")
        pf_r = Decimal("0.0000")
        max_dd_dec = Decimal("0.0000")
        sharpe_dec = Decimal("0.0000")

    return ModelEvaluationReport(
        split_name=split_name,
        total_samples=n,
        class_distribution=class_dist,
        class_imbalance_ratio=imbalance,
        buy_metrics=buy_metrics,
        sell_metrics=sell_metrics,
        wait_metrics=wait_metrics,
        macro_f1=macro_f1,
        active_directional_signals=active_count,
        active_directional_precision=active_prec,
        multiclass_brier_score=brier_dec,
        mean_winning_probability=mean_win_p,
        mean_losing_probability=mean_loss_p,
        calibration_error_ece=ece_dec,
        expected_r_per_signal=exp_r,
        simulated_profit_factor=pf_r,
        simulated_max_drawdown_r=max_dd_dec,
        simulated_sharpe_ratio=sharpe_dec,
    )


def _compute_single_class_metrics(
    direction: SignalDirection,
    cls_idx: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> ClassClassificationMetrics:
    tp = int(np.sum((y_true == cls_idx) & (y_pred == cls_idx)))
    fp = int(np.sum((y_true != cls_idx) & (y_pred == cls_idx)))
    fn = int(np.sum((y_true == cls_idx) & (y_pred != cls_idx)))
    support = tp + fn
    pred_cnt = tp + fp

    prec = Decimal(tp) / Decimal(pred_cnt) if pred_cnt > 0 else Decimal("0")
    rec = Decimal(tp) / Decimal(support) if support > 0 else Decimal("0")
    f1 = (Decimal("2") * prec * rec) / (prec + rec) if (prec + rec) > Decimal("0") else Decimal("0")

    return ClassClassificationMetrics(
        direction=direction,
        support=support,
        predicted_count=pred_cnt,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=prec.quantize(QUANTIZE_METRIC, rounding=ROUND_HALF_UP),
        recall=rec.quantize(QUANTIZE_METRIC, rounding=ROUND_HALF_UP),
        f1_score=f1.quantize(QUANTIZE_METRIC, rounding=ROUND_HALF_UP),
    )


def _compute_ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 5) -> float:
    n = len(confidences)
    if n == 0:
        return 0.0
    bin_Boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bin_Boundaries[b], bin_Boundaries[b + 1]
        mask = (confidences >= lo) & (confidences <= hi if b == n_bins - 1 else confidences < hi)
        if np.any(mask):
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            bin_weight = float(np.sum(mask)) / float(n)
            ece += bin_weight * abs(bin_acc - bin_conf)
    return ece


class WalkForwardValidator:
    """Executes leakage-safe chronological train/validation/test splits and walk-forward evaluation.

    Key Anti-Leakage & Evaluation Rules:
    1. Enforces `purge_bars >= labeling.horizon_bars` between every adjacent split.
    2. Fits feature standardization (`FoldStandardScaler`) and model parameters exclusively
       on each fold's training set.
    3. Model/hyperparameter selection (`select_and_evaluate_chronological`) uses ONLY the
       purged `VALIDATION` split to choose the winning model, and evaluates the selected model
       once on the untouched `OUT_OF_SAMPLE_TEST` split.
    """

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self._config = config or WalkForwardConfig()
        self._feature_builder = MLFeatureBuilder(self._config.feature_config)
        self._labeler = TripleBarrierLabeler(self._config.labeling)

    @property
    def config(self) -> WalkForwardConfig:
        return self._config

    @property
    def feature_builder(self) -> MLFeatureBuilder:
        return self._feature_builder

    @property
    def labeler(self) -> TripleBarrierLabeler:
        return self._labeler

    def _build_classifier(
        self,
        model_type: ModelType,
        training_config: ModelTrainingConfig | None = None,
    ) -> BaseMLClassifier:
        t_cfg = training_config or self._config.training_config
        f_cfg = self._config.feature_config
        if model_type == ModelType.REGULARIZED_LOGISTIC:
            return RegularizedLogisticClassifier(
                t_cfg,
                standardize_features=f_cfg.standardize_features,
                clip_zscore=f_cfg.clip_zscore,
            )
        if model_type == ModelType.GRADIENT_BOOSTED_STUMPS:
            return GradientBoostedStumpEnsemble(
                t_cfg,
                standardize_features=f_cfg.standardize_features,
                clip_zscore=f_cfg.clip_zscore,
            )
        raise ModelConfigurationError(f"Unsupported ModelType: {model_type}.")

    def prepare_supervised_arrays(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, tuple[TripleBarrierLabel, ...], tuple[int, ...]]:
        """Compute causal features and triple-barrier labels, filtering out warm-up NaN rows
        and incomplete tail horizon rows.

        Returns:
            `(x_matrix, y_array, valid_labels, original_bar_indices)`
        """
        ml_df = self._feature_builder.build_features(df)
        all_labels = self._labeler.label_dataframe(ml_df)
        valid_feat_mask = self._feature_builder.extract_valid_row_mask(ml_df)

        feat_cols = list(self._feature_builder.feature_names)
        raw_x = ml_df[feat_cols].to_numpy(dtype="float64")

        valid_indices: list[int] = []
        valid_labels: list[TripleBarrierLabel] = []
        y_list: list[int] = []

        for idx, lbl in enumerate(all_labels):
            if valid_feat_mask[idx] and lbl.is_complete_horizon:
                valid_indices.append(idx)
                valid_labels.append(lbl)
                y_list.append(lbl.class_index)

        if not valid_indices:
            raise ModelConfigurationError(
                "No valid supervised samples remain after warm-up and horizon filtering."
            )

        x_mat = raw_x[valid_indices]
        y_arr = np.array(y_list, dtype=int)
        return x_mat, y_arr, tuple(valid_labels), tuple(valid_indices)

    def run_walk_forward(
        self,
        df: pd.DataFrame,
        model_type: ModelType = ModelType.REGULARIZED_LOGISTIC,
    ) -> WalkForwardReport:
        """Execute multi-fold chronological walk-forward validation with mandatory purging."""
        cfg = self._config
        x_mat, y_arr, valid_labels, _ = self.prepare_supervised_arrays(df)
        n_total = len(y_arr)
        feat_names = self._feature_builder.feature_names

        min_required = cfg.train_window_bars + cfg.purge_bars + cfg.test_window_bars
        if n_total < min_required:
            raise ModelConfigurationError(
                f"Dataset has {n_total} valid supervised samples, fewer than minimum required "
                f"({min_required}) for walk-forward validation."
            )

        folds: list[WalkForwardFoldResult] = []
        oos_y_true_parts: list[np.ndarray] = []
        oos_proba_parts: list[np.ndarray] = []
        oos_labels_all: list[TripleBarrierLabel] = []

        cursor_train_end = cfg.train_window_bars - 1
        fold_idx = 0

        while True:
            purge_start = cursor_train_end + 1
            purge_end = cursor_train_end + cfg.purge_bars
            test_start = purge_end + 1
            if test_start >= n_total:
                break
            test_end = min(n_total - 1, test_start + cfg.test_window_bars - 1)
            if test_end - test_start + 1 < 5:
                break

            train_start = (
                0
                if cfg.expanding_window
                else max(0, cursor_train_end - cfg.train_window_bars + 1)
            )

            # Verify zero label overlap between training end and test start
            if test_start - cursor_train_end - 1 < cfg.purge_bars:
                raise ModelDataLeakageError("Walk-forward fold violated purge_bars constraint.")

            x_train = x_mat[train_start : cursor_train_end + 1]
            y_train = y_arr[train_start : cursor_train_end + 1]
            x_test = x_mat[test_start : test_end + 1]
            y_test = y_arr[test_start : test_end + 1]
            lbl_test = valid_labels[test_start : test_end + 1]

            clf = self._build_classifier(model_type)
            clf.fit(
                x_train,
                y_train,
                feat_names,
                horizon_bars=cfg.labeling.horizon_bars,
            )
            proba_test = clf.predict_proba_matrix(x_test, feature_names=feat_names)

            fold_eval = evaluate_predictions(
                split_name=f"FOLD_{fold_idx}_OOS_TEST",
                y_true=y_test,
                proba_matrix=proba_test,
                labels=lbl_test,
                rr_ratio=cfg.labeling.rr_ratio,
                probability_threshold=cfg.probability_threshold,
                directional_margin=cfg.directional_margin,
            )

            assert clf.metadata is not None
            folds.append(
                WalkForwardFoldResult(
                    fold_index=fold_idx,
                    train_start_index=train_start,
                    train_end_index=cursor_train_end,
                    purge_start_index=purge_start,
                    purge_end_index=purge_end,
                    test_start_index=test_start,
                    test_end_index=test_end,
                    model_id=clf.metadata.model_id,
                    evaluation=fold_eval,
                )
            )

            oos_y_true_parts.append(y_test)
            oos_proba_parts.append(proba_test)
            oos_labels_all.extend(lbl_test)

            fold_idx += 1
            cursor_train_end += cfg.step_bars

        if not folds:
            raise ModelConfigurationError("Walk-forward validation produced zero folds.")

        agg_y = np.concatenate(oos_y_true_parts, axis=0)
        agg_proba = np.concatenate(oos_proba_parts, axis=0)
        agg_eval = evaluate_predictions(
            split_name="AGGREGATED_OUT_OF_SAMPLE",
            y_true=agg_y,
            proba_matrix=agg_proba,
            labels=oos_labels_all,
            rr_ratio=cfg.labeling.rr_ratio,
            probability_threshold=cfg.probability_threshold,
            directional_margin=cfg.directional_margin,
        )

        return WalkForwardReport(
            model_type=model_type,
            config=cfg,
            folds=tuple(folds),
            aggregated_out_of_sample=agg_eval,
        )

    def select_on_validation_and_evaluate_test(
        self,
        df: pd.DataFrame,
        candidate_models: Sequence[ModelType] = (
            ModelType.REGULARIZED_LOGISTIC,
            ModelType.GRADIENT_BOOSTED_STUMPS,
        ),
    ) -> tuple[BaseMLClassifier, ModelEvaluationReport, ModelEvaluationReport, ChronologicalSplitIndices]:
        """Train candidates on TRAIN, select the winner STRICTLY on purged VALIDATION (never touching TEST),
        and evaluate the selected model once on the purged OUT_OF_SAMPLE_TEST split.
        """
        cfg = self._config
        x_mat, y_arr, valid_labels, _ = self.prepare_supervised_arrays(df)
        n_total = len(y_arr)
        feat_names = self._feature_builder.feature_names

        train_start = 0
        train_end = cfg.train_window_bars - 1
        val_start = train_end + 1 + cfg.purge_bars
        val_end = val_start + cfg.validation_window_bars - 1
        test_start = val_end + 1 + cfg.purge_bars
        test_end = min(n_total - 1, test_start + cfg.test_window_bars - 1)

        split_indices = validate_chronological_split(
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            test_end=test_end,
            horizon_bars=cfg.labeling.horizon_bars,
            purge_bars=cfg.purge_bars,
            total_labelable_bars=n_total,
        )

        x_train = x_mat[split_indices.train_start : split_indices.train_end + 1]
        y_train = y_arr[split_indices.train_start : split_indices.train_end + 1]
        x_val = x_mat[split_indices.val_start : split_indices.val_end + 1]
        y_val = y_arr[split_indices.val_start : split_indices.val_end + 1]
        lbl_val = valid_labels[split_indices.val_start : split_indices.val_end + 1]

        best_clf: BaseMLClassifier | None = None
        best_val_report: ModelEvaluationReport | None = None
        best_val_score: tuple[Decimal, Decimal] | None = None

        for m_type in candidate_models:
            clf = self._build_classifier(m_type)
            clf.fit(x_train, y_train, feat_names, horizon_bars=cfg.labeling.horizon_bars)
            val_proba = clf.predict_proba_matrix(x_val, feature_names=feat_names)
            val_report = evaluate_predictions(
                split_name=f"VALIDATION_{m_type.value}",
                y_true=y_val,
                proba_matrix=val_proba,
                labels=lbl_val,
                rr_ratio=cfg.labeling.rr_ratio,
                probability_threshold=cfg.probability_threshold,
                directional_margin=cfg.directional_margin,
            )
            # Rank strictly on validation (macro_f1, -multiclass_brier_score)
            score = (val_report.macro_f1, -val_report.multiclass_brier_score)
            if best_val_score is None or score > best_val_score:
                best_val_score = score
                best_clf = clf
                best_val_report = val_report

        assert best_clf is not None and best_val_report is not None

        # Evaluate the validation-selected model ONCE on the untouched out-of-sample TEST split
        x_test = x_mat[split_indices.test_start : split_indices.test_end + 1]
        y_test = y_arr[split_indices.test_start : split_indices.test_end + 1]
        lbl_test = valid_labels[split_indices.test_start : split_indices.test_end + 1]

        test_proba = best_clf.predict_proba_matrix(x_test, feature_names=feat_names)
        test_report = evaluate_predictions(
            split_name="FINAL_OUT_OF_SAMPLE_TEST",
            y_true=y_test,
            proba_matrix=test_proba,
            labels=lbl_test,
            rr_ratio=cfg.labeling.rr_ratio,
            probability_threshold=cfg.probability_threshold,
            directional_margin=cfg.directional_margin,
        )

        return best_clf, best_val_report, test_report, split_indices
