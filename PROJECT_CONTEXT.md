# ATLAS AI — PROJECT CONTEXT & AI HANDOFF DOCUMENT

> **CRITICAL FOR ANY NEW AI OR HUMAN DEVELOPER:**  
> Read this entire file before writing or modifying any code. Compare the claims in this document against the actual files in `src/` and `tests/` before taking action.

---

## 1. Full Project Purpose

ATLAS AI is a modular autonomous Forex trading intelligence system focused initially on **XAUUSD** (Gold vs. USD) on Windows 11.

Its long-term architecture is designed to:
1. Ingest educational materials (YouTube transcripts, Markdown notes, extracted PDF text, playbooks) and extract structured trading rules.
2. Store, validate, approve, and retrieve learned trading knowledge (`atlas_ai.knowledge`).
3. Analyze Forex/metals market data using deterministic technical/SMC/ICT/knowledge rules and walk-forward classical statistical/ML models (`atlas_ai.models`).
4. Generate structured `BUY`, `SELL`, or `WAIT` decisions.
5. Enforce a **deterministic, LLM-independent and ML-independent risk management and validation pipeline** before any order can be considered.
6. Simulate historical execution via a deterministic, lookahead-free backtesting engine (`atlas_ai.backtest`).
7. Integrate with **MetaTrader 5 (MT5)** for demo execution and post-trade self-improvement.

---

## 2. Current Architecture (Phases 0, 1, 2, 3, 4, 5 & 6 Completed)

- **Phase 0 (Project Foundation)**: `src/atlas_ai/core/` & `src/atlas_ai/main.py`
- **Phase 1 (Core Domain Schemas & Deterministic Risk Engine)**: `src/atlas_ai/domain/` & `src/atlas_ai/risk/`
- **Phase 2 (Market Data Ingestion, Validation & Feature Pipeline)**: `src/atlas_ai/data/`
- **Phase 3 (Advanced Multi-Strategy SMC/ICT Trading Intelligence Engine)**: `src/atlas_ai/strategies/`
- **Phase 4 (Historical Backtesting Engine)**: `src/atlas_ai/backtest/`
- **Phase 5 (Educational Knowledge Ingestion & Structured Rule Extraction)**: `src/atlas_ai/knowledge/`
- **Phase 6 (Machine Learning & Statistical Intelligence)**: `src/atlas_ai/models/`

---

## 3. How the Current Modules Work Together

1. **`src/atlas_ai/core/`**: Provides `AppSettings` (blocking `ExecutionMode.LIVE` via `TradingSafetyError`), custom exception types (`AtlasError` hierarchy), and rotating file/console logging.
2. **`src/atlas_ai/domain/`**: Defines canonical immutable Pydantic models (`Candle`, `Timeframe`, `InstrumentSpec`, `SignalDirection`, `TradeSignal`, `AccountState`, `RiskParameters`, `PositionSizeResult`, `RiskDecision`, `RiskRejectionCode`).
3. **`src/atlas_ai/risk/engine.py`**: Evaluates `TradeSignal` against `AccountState`, `RiskParameters`, and `InstrumentSpec` across 10 deterministic gates and floors lot size to `volume_step`.
4. **`src/atlas_ai/data/`**: Loads historical CSVs (`HistoricalCSVLoader`), validates OHLCV and calendar-aware time-series integrity (`MarketDataValidator`, `check_timeseries_integrity`), converts to/from `Candle` objects, and computes causal technical features (`FeaturePipeline`).
5. **`src/atlas_ai/strategies/`**: Implements 10 SMC/ICT and trend strategies/analyzers, `NewsFilter`, `ConfluenceEngine`, `SignalGenerator`, `SignalLifecycleManager`, and `StrategyEngine` (supporting optional `knowledge_strategy`/`knowledge_repository` and `ml_strategy`/`ml_classifier` integration).
6. **`src/atlas_ai/backtest/`**: Implements `BacktestConfig`, `ExecutionSimulator`, `PortfolioTracker`, `calculate_backtest_metrics()`, and `BacktestEngine`.
7. **`src/atlas_ai/knowledge/`**: Implements `KnowledgeDocumentLoader`, `TextChunker`, `KnowledgeRuleValidator`, `DeterministicRuleExtractor`, `LLMExtractorAdapter`, `KnowledgeRepository`, and `KnowledgeBasedStrategy(BaseStrategy)`.
8. **`src/atlas_ai/models/`**:
   - `exceptions.py`: `ModelError`, `ModelConfigurationError`, `ModelDataLeakageError`, `ModelTrainingError`, `ModelPredictionError`.
   - `config.py`: `LabelingConfig`, `MLFeatureConfig`, `ModelTrainingConfig`, `WalkForwardConfig` (enforcing `purge_bars >= labeling.horizon_bars`), `MLStrategyConfig`.
   - `schemas.py`: `ModelType`, `TripleBarrierLabel`, `ClassClassificationMetrics`, `PredictionResult`, `ModelEvaluationReport`, `ChronologicalSplitIndices`, `WalkForwardFoldResult`, `WalkForwardReport`, `ModelArtifactMetadata`.
   - `labeling.py`: `TripleBarrierLabeler` computing deterministic `BUY`, `SELL`, and `WAIT` labels within forward horizon $H$ with conservative `STOP_LOSS`-first priority and dual-outcome tie-breaking.
   - `features.py`: `MLFeatureBuilder` (stationary ATR-normalized distances, lagged returns, range positions, volatility/volume ratios, cyclical time encodings) and train-fold-only `FoldStandardScaler`.
   - `classifiers.py`: `BaseMLClassifier`, `RegularizedLogisticClassifier` (implemented first), `GradientBoostedStumpEnsemble` (implemented second), and `load_model_artifact()`.
   - `validation.py`: `validate_chronological_split()`, `evaluate_predictions()`, and `WalkForwardValidator`.
   - `strategy.py`: `MLSignalStrategy(BaseStrategy)`.

---

## 4. What Has Already Been Implemented & Verified

- [x] **Phase 0 — Project Foundation** (`tests/test_foundation.py` — 5/5 passing).
- [x] **Phase 1 — Core Domain Schemas & Deterministic Risk Engine Foundation** (`tests/test_domain_and_risk.py` — 38/38 passing).
- [x] **Phase 2 — Market Data Ingestion, Validation & Feature Pipeline** (`tests/test_market_data.py` & `tests/test_features.py` — 15/15 passing).
- [x] **Phase 3 — Advanced Strategy Engine & Multi-Rule SMC/ICT Signal Generation** (`tests/test_strategies.py`, `tests/test_smc_ict.py`, `tests/test_multi_timeframe.py`, `tests/test_news_filter.py`, `tests/test_confluence.py`, `tests/test_signal_generation.py`, `tests/test_signal_lifecycle.py` — 18/18 passing).
- [x] **Phase 4 — Historical Backtesting Engine** (`tests/test_backtest.py` — 8/8 passing).
- [x] **Phase 5 — Educational Knowledge Ingestion & Structured Rule Extraction** (`tests/test_knowledge.py` — 6/6 passing).
- [x] **Phase 6 — Machine Learning & Statistical Intelligence** (`tests/test_models.py` — 5/5 passing; **95/95 total tests passing with 0 warnings**).
- [x] All required documentation files updated (`README.md`, `PROJECT_CONTEXT.md`, `DEVELOPMENT_LOG.md`, `ARCHITECTURE.md`, `NEXT_STEPS.md`, `CHANGELOG.md`, `.env.example`).

---

## 5. What Has NOT Been Implemented Yet

- **Phase 7:** MetaTrader 5 (`MetaTrader5`) Demo Environment Integration & Pre-Trade Gates.
- **Phase 8:** Post-Trade Analytics & Self-Improvement Feedback Loop.

---

## 6. Important Development Decisions & Why They Were Made

1. **Mandatory Purged Embargo Gap (`purge_bars >= labeling.horizon_bars`)**: Because a Triple-Barrier label at bar $t$ inspects future prices up to $t + H$, any chronological split without at least $H$ purged bars between `train_end` and `val_start` (or `val_end` and `test_start`) leaks validation/test prices into training labels. `WalkForwardConfig` and `validate_chronological_split()` enforce `purge_bars >= horizon_bars` and raise `ModelDataLeakageError` on any violation.
2. **Train-Fold-Only Feature Standardization (`FoldStandardScaler`)**: Feature z-score parameters $(\mu, \sigma)$ are fit strictly on each fold's training matrix and embedded inside the saved model artifact so test and live evaluation bars never influence scaling statistics.
3. **Validation-Only Model Selection (`select_on_validation_and_evaluate_test`)**: Model candidates (`RegularizedLogisticClassifier` vs. `GradientBoostedStumpEnsemble`) are ranked exclusively on the purged `VALIDATION` split before evaluating the winning model once on the untouched `OUT_OF_SAMPLE_TEST` split.

---

## 7. Exact Next Development Task (Phase 7)

**Phase 7: MetaTrader 5 (MT5) Demo Environment Integration & Pre-Trade Gates**
- Wait for explicit user instructions before beginning Phase 7.
