"""Comprehensive Phase 6 unit & walk-forward integration tests for Machine Learning & Statistical Intelligence."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atlas_ai.backtest import BacktestConfig, BacktestEngine
from atlas_ai.data import MarketDataPipeline, MissingBarPolicy
from atlas_ai.data.features import FeaturePipeline
from atlas_ai.domain.market import InstrumentSpec, Timeframe
from atlas_ai.domain.risk import AccountState, RiskParameters, RiskRejectionCode
from atlas_ai.domain.signals import SignalDirection
from atlas_ai.models import (
    FoldStandardScaler,
    GradientBoostedStumpEnsemble,
    LabelingConfig,
    MLFeatureBuilder,
    MLFeatureConfig,
    MLSignalStrategy,
    MLStrategyConfig,
    ModelDataLeakageError,
    ModelTrainingConfig,
    ModelType,
    RegularizedLogisticClassifier,
    TripleBarrierLabeler,
    WalkForwardConfig,
    WalkForwardValidator,
    load_model_artifact,
    validate_chronological_split,
)
from atlas_ai.strategies.config import ConfluenceWeightsConfig
from atlas_ai.strategies.engine import StrategyEngine
from atlas_ai.strategies.models import RiskEvaluationStatus, SignalLifecycleState


def _make_multi_regime_xauusd_df(n_bars: int = 360) -> pd.DataFrame:
    """Build a deterministic multi-regime synthetic XAUUSD H1 dataset for offline unit testing only.

    Cycles through Bullish Trend, Ranging/Consolidation, and Bearish Trend regimes so BUY, SELL,
    and WAIT triple-barrier outcomes are all well-represented.
    """
    t0 = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    rows: list[dict[str, object]] = []
    price = 2600.0
    for i in range(n_bars):
        ts = t0 + timedelta(hours=i)
        cycle_pos = i % 90
        if cycle_pos < 32:
            # Strong uptrend regime
            drift = 1.65 if (i % 5 != 2) else -0.45
        elif cycle_pos < 58:
            # Tight ranging oscillation regime (produces WAIT labels)
            drift = 0.35 if (i % 2 == 0) else -0.35
        else:
            # Strong downtrend regime
            drift = -1.65 if (i % 5 != 2) else 0.45

        open_p = round(price, 2)
        close_p = round(max(100.0, price + drift), 2)
        high_p = round(max(open_p, close_p) + 0.70, 2)
        low_p = round(min(open_p, close_p) - 0.70, 2)
        vol = 150.0 + float((i * 17) % 90)
        rows.append(
            {
                "timestamp": ts,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": vol,
            }
        )
        price = close_p

    return FeaturePipeline().transform(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# 1. TripleBarrierLabeler: BUY/SELL/WAIT, Dual-Success Resolution, SL-First & Horizon Cap
# ---------------------------------------------------------------------------
def test_triple_barrier_labeler_buy_sell_wait_conflict_and_sl_first_priority() -> None:
    """Verify exact BUY/SELL/WAIT labeling, dual-outcome conflict resolution, conservative SL-first,
    and strict prevention of using bars beyond t + horizon_bars.
    """
    labeler = TripleBarrierLabeler(
        LabelingConfig(
            horizon_bars=4,
            sl_atr_multiplier=Decimal("1.00"),
            rr_ratio=Decimal("2.00"),
        )
    )

    # Test deterministic dual-outcome resolution logic explicitly:
    # 1A. Long succeeds earlier than Short -> BUY
    dir_long_first, _ = TripleBarrierLabeler.resolve_barrier_outcomes(
        long_succeeded=True,
        short_succeeded=True,
        long_resolution_bar=12,
        short_resolution_bar=15,
    )
    assert dir_long_first == SignalDirection.BUY

    # 1B. Short succeeds earlier than Long -> SELL
    dir_short_first, _ = TripleBarrierLabeler.resolve_barrier_outcomes(
        long_succeeded=True,
        short_succeeded=True,
        long_resolution_bar=16,
        short_resolution_bar=13,
    )
    assert dir_short_first == SignalDirection.SELL

    # 1C. Ambiguous same-bar dual success -> conservatively assigned WAIT
    dir_same_bar, _ = TripleBarrierLabeler.resolve_barrier_outcomes(
        long_succeeded=True,
        short_succeeded=True,
        long_resolution_bar=14,
        short_resolution_bar=14,
    )
    assert dir_same_bar == SignalDirection.WAIT

    # Construct an explicit candle sequence with fixed atr_14 = 10.00 (d_SL = 10.00, d_TP = 20.00):
    # Bar 0 (close=2600): Bar 1 rallies to 2625 without dropping below 2595 -> BUY
    # Bar 1 (close=2620): Bar 2 spans BOTH Long SL (2610) and Long TP (2640) [low=2605, high=2645]
    #                     AND spans Short SL (2630) and Short TP (2600) -> Conservative SL-First fails both -> WAIT!
    # Bar 2 (close=2620): Bar 3 drops to 2598 without exceeding 2625 -> SELL
    t0 = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
    explicit_rows = [
        {"timestamp": t0, "open": 2600.0, "high": 2601.0, "low": 2599.0, "close": 2600.0, "volume": 100.0, "atr_14": 10.0},
        {"timestamp": t0 + timedelta(hours=1), "open": 2600.0, "high": 2625.0, "low": 2596.0, "close": 2620.0, "volume": 100.0, "atr_14": 10.0},
        {"timestamp": t0 + timedelta(hours=2), "open": 2620.0, "high": 2645.0, "low": 2595.0, "close": 2620.0, "volume": 100.0, "atr_14": 10.0},
        {"timestamp": t0 + timedelta(hours=3), "open": 2620.0, "high": 2622.0, "low": 2595.0, "close": 2598.0, "volume": 100.0, "atr_14": 10.0},
        {"timestamp": t0 + timedelta(hours=4), "open": 2598.0, "high": 2600.0, "low": 2597.0, "close": 2599.0, "volume": 100.0, "atr_14": 10.0},
        {"timestamp": t0 + timedelta(hours=5), "open": 2599.0, "high": 2600.0, "low": 2598.0, "close": 2599.0, "volume": 100.0, "atr_14": 10.0},
        {"timestamp": t0 + timedelta(hours=6), "open": 2599.0, "high": 2600.0, "low": 2598.0, "close": 2599.0, "volume": 100.0, "atr_14": 10.0},
        {"timestamp": t0 + timedelta(hours=7), "open": 2599.0, "high": 2600.0, "low": 2598.0, "close": 2599.0, "volume": 100.0, "atr_14": 10.0},
    ]
    df = pd.DataFrame(explicit_rows)
    labels = labeler.label_dataframe(df)

    # Bar 0: Long TP (2620) hit on Bar 1 (high=2625, low=2596 > long_sl=2590) -> BUY
    assert labels[0].label == SignalDirection.BUY
    assert labels[0].long_succeeded is True
    assert labels[0].long_resolution_bar == 1
    assert labels[0].max_future_bar_inspected <= 0 + 4

    # Bar 1: On Bar 2 (high=2645, low=2595), BOTH Long SL (2610) & Long TP (2640) are touched,
    # and BOTH Short SL (2630) & Short TP (2600) are touched. Conservative SL-first rejects both -> WAIT!
    assert labels[1].label == SignalDirection.WAIT
    assert labels[1].long_succeeded is False
    assert labels[1].short_succeeded is False

    # Bar 2: On Bar 3 (high=2622 < short_sl=2630, low=2595 <= short_tp=2600) -> SELL
    assert labels[2].label == SignalDirection.SELL
    assert labels[2].short_succeeded is True
    assert labels[2].short_resolution_bar == 3

    # Verify labels never use data beyond t + horizon_bars (4):
    # Mutating Bar 6 and Bar 7 (which are > 0 + 4 = 4) must not change Label 0 or Label 1!
    df_mut = df.copy()
    df_mut.loc[6, "high"] = 5000.0
    df_mut.loc[6, "low"] = 1000.0
    df_mut.loc[7, "high"] = 5000.0
    df_mut.loc[7, "low"] = 1000.0
    labels_mut = labeler.label_dataframe(df_mut)
    assert labels_mut[0] == labels[0]
    assert labels_mut[1] == labels[1]


# ---------------------------------------------------------------------------
# 2 & 3. Leakage Prevention: Causal Features, Train-Only Scaling & Purge Enforcement
# ---------------------------------------------------------------------------
def test_feature_causality_train_only_scaler_and_purge_enforcement() -> None:
    """Verify features at index t use only bars <= t, scaler fits only on train, and purge_bars >= horizon_bars."""
    df_full = _make_multi_regime_xauusd_df(120)
    builder = MLFeatureBuilder()
    ml_full = builder.build_features(df_full)

    # 2A. Prefix vs Mutated Future Feature Invariance at bar t = 80
    df_mutated = df_full.copy()
    for idx in range(81, 120):
        df_mutated.loc[idx, "open"] = 4500.0
        df_mutated.loc[idx, "high"] = 4600.0
        df_mutated.loc[idx, "low"] = 4400.0
        df_mutated.loc[idx, "close"] = 4550.0
        df_mutated.loc[idx, "volume"] = 99999.0

    ml_mutated = builder.build_features(df_mutated)
    feat_cols = list(builder.feature_names)
    pd.testing.assert_frame_equal(
        ml_full.loc[:80, feat_cols],
        ml_mutated.loc[:80, feat_cols],
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )

    # 2B. FoldStandardScaler fits strictly on training fold
    valid_mask = builder.extract_valid_row_mask(ml_full)
    x_valid = ml_full.loc[valid_mask, feat_cols].to_numpy(dtype="float64")
    x_train = x_valid[:40]
    x_test_1 = x_valid[50:65].copy()
    x_test_2 = x_test_1.copy()
    x_test_2[-1, :] = 9999.0  # Extreme outlier in test fold

    scaler_1 = FoldStandardScaler().fit(x_train)
    scaler_2 = FoldStandardScaler().fit(x_train)
    assert scaler_1.means == scaler_2.means
    assert scaler_1.stds == scaler_2.stds
    # Transforming earlier test rows 0..10 is completely unaffected by row 14 in test fold
    np.testing.assert_allclose(
        scaler_1.transform(x_test_1)[:10],
        scaler_2.transform(x_test_2)[:10],
    )

    # 2C. Enforce purge_bars >= horizon_bars in WalkForwardConfig and validate_chronological_split
    with pytest.raises(ModelDataLeakageError, match="LEAKAGE VIOLATION"):
        WalkForwardConfig(
            labeling=LabelingConfig(horizon_bars=12),
            purge_bars=11,  # < 12 -> must raise ModelDataLeakageError!
        )

    with pytest.raises(ModelDataLeakageError, match="LEAKAGE VIOLATION"):
        validate_chronological_split(
            train_start=0,
            train_end=99,
            val_start=105,  # Only 5 bars purged (< 12) -> must raise ModelDataLeakageError!
            val_end=140,
            test_start=155,
            test_end=190,
            horizon_bars=12,
            purge_bars=12,
        )

    valid_split = validate_chronological_split(
        train_start=0,
        train_end=99,
        val_start=112,   # 12 bars purged (indices 100..111)
        val_end=150,
        test_start=163,  # 12 bars purged (indices 151..162)
        test_end=200,
        horizon_bars=12,
        purge_bars=12,
        total_labelable_bars=210,
    )
    assert valid_split.purge_1_start == 100
    assert valid_split.purge_1_end == 111
    assert valid_split.purge_2_start == 151
    assert valid_split.purge_2_end == 162


# ---------------------------------------------------------------------------
# 4. Model 1 (First): RegularizedLogisticClassifier & Model 2 (Second): GradientBoostedStumpEnsemble
# ---------------------------------------------------------------------------
def test_classical_models_logistic_first_gbdt_second_determinism_and_artifacts(tmp_path: Path) -> None:
    """Test RegularizedLogisticClassifier first and GradientBoostedStumpEnsemble second."""
    df = _make_multi_regime_xauusd_df(260)
    wf_validator = WalkForwardValidator(
        WalkForwardConfig(
            train_window_bars=120,
            validation_window_bars=30,
            test_window_bars=30,
            purge_bars=8,
            labeling=LabelingConfig(horizon_bars=8),
        )
    )
    x_mat, y_arr, _, _ = wf_validator.prepare_supervised_arrays(df)
    feat_names = wf_validator.feature_builder.feature_names

    x_train, y_train = x_mat[:130], y_arr[:130]
    x_test = x_mat[140:170]
    ts = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)

    # --- STEP 1: RegularizedLogisticClassifier ---
    logreg_1 = RegularizedLogisticClassifier(ModelTrainingConfig(random_seed=42, max_iterations=200))
    logreg_2 = RegularizedLogisticClassifier(ModelTrainingConfig(random_seed=42, max_iterations=200))
    logreg_1.fit(x_train, y_train, feat_names, horizon_bars=8)
    logreg_2.fit(x_train, y_train, feat_names, horizon_bars=8)

    probs_lr_1 = logreg_1.predict_proba_matrix(x_test, feature_names=feat_names)
    probs_lr_2 = logreg_2.predict_proba_matrix(x_test, feature_names=feat_names)
    np.testing.assert_allclose(probs_lr_1, probs_lr_2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.sum(probs_lr_1, axis=1), np.ones(len(x_test)), atol=1e-6)

    pred_lr = logreg_1.predict_single(x_test[0], timestamp=ts, feature_names=feat_names)
    assert pred_lr.model_type == ModelType.REGULARIZED_LOGISTIC
    assert pred_lr.buy_probability + pred_lr.sell_probability + pred_lr.wait_probability == Decimal("1.0000")

    lr_path = tmp_path / "logreg_artifact.json"
    logreg_1.save_artifact(lr_path)
    loaded_lr = load_model_artifact(lr_path)
    assert isinstance(loaded_lr, RegularizedLogisticClassifier)
    np.testing.assert_allclose(
        loaded_lr.predict_proba_matrix(x_test, feature_names=feat_names),
        probs_lr_1,
        rtol=1e-12,
        atol=1e-12,
    )

    # --- STEP 2: GradientBoostedStumpEnsemble ---
    gbdt_1 = GradientBoostedStumpEnsemble(ModelTrainingConfig(random_seed=42, n_estimators=25))
    gbdt_2 = GradientBoostedStumpEnsemble(ModelTrainingConfig(random_seed=42, n_estimators=25))
    gbdt_1.fit(x_train, y_train, feat_names, horizon_bars=8)
    gbdt_2.fit(x_train, y_train, feat_names, horizon_bars=8)

    probs_gb_1 = gbdt_1.predict_proba_matrix(x_test, feature_names=feat_names)
    probs_gb_2 = gbdt_2.predict_proba_matrix(x_test, feature_names=feat_names)
    np.testing.assert_allclose(probs_gb_1, probs_gb_2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.sum(probs_gb_1, axis=1), np.ones(len(x_test)), atol=1e-6)

    gb_path = tmp_path / "gbdt_artifact.json"
    gbdt_1.save_artifact(gb_path)
    loaded_gb = load_model_artifact(gb_path)
    assert isinstance(loaded_gb, GradientBoostedStumpEnsemble)
    np.testing.assert_allclose(
        loaded_gb.predict_proba_matrix(x_test, feature_names=feat_names),
        probs_gb_1,
        rtol=1e-12,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# 5. Walk-Forward Evaluation, Separate BUY/SELL/WAIT Metrics & Phase 2 CSV Compatibility
# ---------------------------------------------------------------------------
def test_walk_forward_validation_metrics_and_phase2_csv_pipeline_support() -> None:
    """Verify walk-forward validation, separate BUY/SELL/WAIT metrics, validation-only model selection,
    and compatibility with the Phase 2 Historical CSV MarketDataPipeline.
    """
    df_multi = _make_multi_regime_xauusd_df(340)
    wf_cfg = WalkForwardConfig(
        train_window_bars=110,
        validation_window_bars=40,
        test_window_bars=45,
        step_bars=45,
        purge_bars=8,
        labeling=LabelingConfig(horizon_bars=8, sl_atr_multiplier=Decimal("1.20"), rr_ratio=Decimal("2.00")),
        probability_threshold=Decimal("0.38"),
        directional_margin=Decimal("0.05"),
    )
    validator = WalkForwardValidator(wf_cfg)

    # 5A. Walk-forward validation for both RegularizedLogisticClassifier and GradientBoostedStumpEnsemble
    wf_lr = validator.run_walk_forward(df_multi, ModelType.REGULARIZED_LOGISTIC)
    wf_gb = validator.run_walk_forward(df_multi, ModelType.GRADIENT_BOOSTED_STUMPS)

    assert len(wf_lr.folds) >= 2
    assert len(wf_gb.folds) >= 2
    for fold in wf_lr.folds:
        assert fold.test_start_index - fold.train_end_index - 1 >= wf_cfg.purge_bars

    agg = wf_gb.aggregated_out_of_sample
    assert agg.buy_metrics.direction == SignalDirection.BUY
    assert agg.sell_metrics.direction == SignalDirection.SELL
    assert agg.wait_metrics.direction == SignalDirection.WAIT
    assert agg.total_samples == (
        agg.class_distribution["BUY"]
        + agg.class_distribution["SELL"]
        + agg.class_distribution["WAIT"]
    )
    assert agg.multiclass_brier_score >= Decimal("0")
    assert Decimal("0") <= agg.calibration_error_ece <= Decimal("1")

    # 5B. Model selection strictly on purged VALIDATION split before single evaluation on TEST split
    selected_clf, val_rep, test_rep, split_idx = validator.select_on_validation_and_evaluate_test(df_multi)
    assert selected_clf.is_fitted
    assert val_rep.split_name.startswith("VALIDATION_")
    assert test_rep.split_name == "FINAL_OUT_OF_SAMPLE_TEST"
    assert split_idx.val_start - split_idx.train_end - 1 == wf_cfg.purge_bars
    assert split_idx.test_start - split_idx.val_end - 1 == wf_cfg.purge_bars

    # 5C. Verify compatibility with Phase 2 MarketDataPipeline on tests/fixtures/synthetic_xauusd_h1.csv
    pipeline_res = MarketDataPipeline(
        missing_bar_policy=MissingBarPolicy.ALLOW
    ).run_from_csv("tests/fixtures/synthetic_xauusd_h1.csv", symbol="XAUUSD", timeframe=Timeframe.H1)
    csv_validator = WalkForwardValidator(
        WalkForwardConfig(
            train_window_bars=70,
            validation_window_bars=25,
            test_window_bars=25,
            step_bars=25,
            purge_bars=6,
            labeling=LabelingConfig(horizon_bars=6),
        )
    )
    csv_wf = csv_validator.run_walk_forward(
        pipeline_res.feature_dataframe,
        ModelType.REGULARIZED_LOGISTIC,
    )
    assert len(csv_wf.folds) >= 1


# ---------------------------------------------------------------------------
# 6. MLSignalStrategy Integration with StrategyEngine, SignalLifecycleManager, RiskEngine & BacktestEngine
# ---------------------------------------------------------------------------
def test_ml_signal_strategy_integration_and_mandatory_risk_engine_gates() -> None:
    """Verify MLSignalStrategy follows BaseStrategy, has zero lookahead, and never bypasses RiskEngine."""
    df = _make_multi_regime_xauusd_df(260)
    wf_validator = WalkForwardValidator(
        WalkForwardConfig(
            train_window_bars=120,
            validation_window_bars=30,
            test_window_bars=30,
            purge_bars=8,
            labeling=LabelingConfig(horizon_bars=8),
        )
    )
    x_mat, y_arr, _, orig_indices = wf_validator.prepare_supervised_arrays(df)
    feat_names = wf_validator.feature_builder.feature_names

    # Train classifier strictly on the first 100 valid samples (up to orig_indices[99])
    train_cutoff_bar = orig_indices[99]
    clf = GradientBoostedStumpEnsemble(ModelTrainingConfig(random_seed=42, n_estimators=25))
    clf.fit(x_mat[:100], y_arr[:100], feat_names, horizon_bars=8)

    ml_strategy = MLSignalStrategy(
        clf,
        strategy_config=MLStrategyConfig(
            min_probability_threshold=Decimal("0.34"),
            min_directional_margin=Decimal("0.01"),
            sl_atr_multiplier=Decimal("1.50"),
            rr_ratio=Decimal("2.00"),
        ),
    )

    # Find an out-of-sample bar index strictly after train_cutoff_bar + 8 where ML emits BUY or SELL
    eval_bar_idx = train_cutoff_bar + 15
    for candidate_idx in range(train_cutoff_bar + 10, len(df) - 10):
        ev_cand = ml_strategy.evaluate(df, as_of_index=candidate_idx)
        if ev_cand.direction in (SignalDirection.BUY, SignalDirection.SELL):
            eval_bar_idx = candidate_idx
            break

    res_causal = ml_strategy.evaluate(df, as_of_index=eval_bar_idx)
    assert res_causal.direction in (SignalDirection.BUY, SignalDirection.SELL)
    assert "model_id" in res_causal.supporting_evidence
    assert "buy_probability" in res_causal.supporting_evidence

    # 6A. Strict Future-Data Invariance (mutating bars after eval_bar_idx does not change evaluation)
    df_mut = df.copy()
    for idx in range(eval_bar_idx + 1, len(df)):
        df_mut.loc[idx, "close"] = 1200.0
        df_mut.loc[idx, "high"] = 1210.0
        df_mut.loc[idx, "low"] = 1190.0
    assert ml_strategy.evaluate(df_mut, as_of_index=eval_bar_idx) == res_causal

    # 6B. Route MLSignalStrategy through StrategyEngine, SignalLifecycleManager, and RiskEngine
    conf_cfg = ConfluenceWeightsConfig(
        enable_ema_trend=False,
        enable_market_structure=False,
        enable_liquidity_sweep=False,
        enable_fvg=False,
        enable_fvg_retest=False,
        enable_mtf=False,
        enable_ml_strategy=True,
        weight_ml_strategy=Decimal("1.00"),
        minimum_confidence_threshold=Decimal("0.30"),
    )
    spec = InstrumentSpec(
        symbol="XAUUSD",
        contract_size=Decimal("100"),
        tick_size=Decimal("0.01"),
        tick_value=Decimal("1.00"),
        volume_step=Decimal("0.01"),
        minimum_volume=Decimal("0.01"),
        maximum_volume=Decimal("50.00"),
        margin_rate=Decimal("0.01"),
    )
    risk_params = RiskParameters(
        risk_per_trade_percent=Decimal("1.0"),
        max_daily_loss_percent=Decimal("3.0"),
        max_total_drawdown_percent=Decimal("10.0"),
        max_open_positions=1,
        minimum_risk_reward_ratio=Decimal("1.5"),
    )

    # Healthy account -> RiskEngine approves ML signal (GENERATED -> RISK_APPROVED)
    healthy_account = AccountState(
        balance=Decimal("10000.00"),
        equity=Decimal("10000.00"),
        free_margin=Decimal("10000.00"),
        daily_starting_balance=Decimal("10000.00"),
        current_daily_pnl=Decimal("0.00"),
        open_positions_count=0,
    )
    strat_engine_ok = StrategyEngine(confluence_config=conf_cfg, ml_strategy=ml_strategy)
    report_ok = strat_engine_ok.analyze(
        df,
        symbol="XAUUSD",
        timeframe=Timeframe.H1,
        as_of_index=eval_bar_idx,
        account_state=healthy_account,
        risk_parameters=risk_params,
        instrument_spec=spec,
    )
    assert report_ok.signal.is_executable is True
    assert report_ok.risk_status == RiskEvaluationStatus.RISK_APPROVED
    assert report_ok.lifecycle_record is not None
    assert report_ok.lifecycle_record.state == SignalLifecycleState.RISK_APPROVED

    # Breached daily loss account -> RiskEngine REJECTS the ML signal (GENERATED -> RISK_REJECTED)!
    breached_account = AccountState(
        balance=Decimal("9600.00"),
        equity=Decimal("9600.00"),
        free_margin=Decimal("9600.00"),
        daily_starting_balance=Decimal("10000.00"),
        current_daily_pnl=Decimal("-400.00"),
        open_positions_count=0,
    )
    strat_engine_rej = StrategyEngine(confluence_config=conf_cfg, ml_strategy=ml_strategy)
    report_rej = strat_engine_rej.analyze(
        df,
        symbol="XAUUSD",
        timeframe=Timeframe.H1,
        as_of_index=eval_bar_idx,
        account_state=breached_account,
        risk_parameters=risk_params,
        instrument_spec=spec,
    )
    assert report_rej.risk_status == RiskEvaluationStatus.RISK_REJECTED
    assert report_rej.risk_decision is not None and report_rej.risk_decision.approved is False
    assert (
        report_rej.risk_decision.rejection_reason is not None
        and RiskRejectionCode.DAILY_LOSS_LIMIT_EXCEEDED.value in report_rej.risk_decision.rejection_reason
    )
    assert report_rej.lifecycle_record is not None
    assert report_rej.lifecycle_record.state == SignalLifecycleState.RISK_REJECTED

    # 6C. Out-of-Sample BacktestEngine verification
    bt_engine = BacktestEngine(
        BacktestConfig(
            warmup_bars=train_cutoff_bar + 10,
            risk_parameters=risk_params,
            instrument_spec=spec,
        ),
        strategy_engine=StrategyEngine(confluence_config=conf_cfg, ml_strategy=ml_strategy),
    )
    bt_res = bt_engine.run(df)
    assert bt_res.total_bars == len(df)
    assert len(bt_res.lifecycle_records) >= 1
