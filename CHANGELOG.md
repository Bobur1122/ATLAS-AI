# Changelog

All notable changes to **ATLAS AI** are documented in this file.

## [0.1.0] - 2026-11-20 (Phase 6: Machine Learning & Statistical Intelligence)

### Added
- `atlas_ai.models.exceptions`: `ModelError`, `ModelConfigurationError`, `ModelDataLeakageError`, `ModelTrainingError`, and `ModelPredictionError`.
- `atlas_ai.models.config`: `LabelingConfig`, `MLFeatureConfig`, `ModelTrainingConfig`, `WalkForwardConfig` (enforcing `purge_bars >= labeling.horizon_bars`), and `MLStrategyConfig`.
- `atlas_ai.models.schemas`: Immutable Pydantic schemas `TripleBarrierLabel`, `ClassClassificationMetrics`, `PredictionResult`, `ModelEvaluationReport`, `ChronologicalSplitIndices`, `WalkForwardFoldResult`, `WalkForwardReport`, `ModelArtifactMetadata`, and `ModelType`.
- `atlas_ai.models.labeling`: `TripleBarrierLabeler` with bounded forward horizon $H$, conservative `STOP_LOSS`-first intrabar rule, and deterministic dual-outcome conflict resolution.
- `atlas_ai.models.features`: Causal stationary `MLFeatureBuilder` and train-fold-only `FoldStandardScaler`.
- `atlas_ai.models.classifiers`: `BaseMLClassifier`, `RegularizedLogisticClassifier`, `GradientBoostedStumpEnsemble`, and `load_model_artifact()`.
- `atlas_ai.models.validation`: `validate_chronological_split()`, `evaluate_predictions()`, and `WalkForwardValidator` with purged walk-forward folds and validation-only model selection.
- `atlas_ai.models.strategy`: `MLSignalStrategy(BaseStrategy)` integrating with `StrategyEngine`, `ConfluenceEngine`, `SignalGenerator`, `SignalLifecycleManager`, `RiskEngine`, and `BacktestEngine`.
- `tests/test_models.py`: 5 comprehensive Phase 6 test suites (95 total tests passing).

### Changed
- `atlas_ai.strategies`: Added optional `ml_strategy` / `ml_classifier` support to `StrategyEngine`, `enable_ml_strategy` / `weight_ml_strategy` to `ConfluenceWeightsConfig` and `ConfluenceEngine`, and `knowledge_based_strategy` / `ml_signal_strategy` stop-loss priority support to `SignalGenerator`.
- `atlas_ai.main`: Updated CLI readiness status message to reflect completed Phases 0, 1, 2, 3, 4, 5, and 6 and readiness for Phase 7.

## [0.1.0] - 2026-11-20 (Phase 5: Educational Knowledge Ingestion & Structured Rule Extraction)

### Added
- `atlas_ai.knowledge` package and 6 test suites in `tests/test_knowledge.py` (90 total tests passing).

## [0.1.0] - 2026-11-15 (Phase 4: Historical Backtesting Engine)

### Added
- `atlas_ai.backtest` package and 8 test suites in `tests/test_backtest.py` (84 total tests passing).

## [0.1.0] - 2026-11-15 (Phase 3: Advanced Strategy Engine & Multi-Rule SMC/ICT Signal Generation)

### Added
- `atlas_ai.strategies` package and 7 Phase 3 test suites (76 total tests passing).

## [0.1.0] - 2026-11-15 (Phase 2: Market Data Ingestion, Validation & Feature Pipeline)

### Added
- `atlas_ai.data` package and 15 unit tests (`test_market_data.py`, `test_features.py`).

## [0.1.0] - 2026-11-15 (Phase 1: Core Domain Schemas & Deterministic Risk Engine Foundation)

### Added
- `atlas_ai.domain` and `atlas_ai.risk` packages and 38 unit tests (`test_domain_and_risk.py`).

## [0.1.0] - 2026-11-14 (Phase 0: Project Foundation)

### Added
- Initial `src/atlas_ai` package, `atlas_ai.core`, `atlas_ai.main`, `tests/test_foundation.py`, and project documentation suite.
