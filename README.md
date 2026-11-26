# ATLAS AI — Modular Autonomous Forex Trading Intelligence System

**Current Version:** `0.1.0`  
**Current Phase:** Phase 9 — Closed-Loop Demo Runtime Orchestration & Persistent Emergency Circuit Breaker (Completed through Phase 9.3)  
**Primary Instrument:** `XAUUSD` (Gold vs. US Dollar)

---

## 1. What ATLAS AI Is

ATLAS AI is a modular, safety-first autonomous Forex trading intelligence system designed to learn structured trading rules from educational materials, analyze market structure and statistical patterns on historical and live market data, generate disciplined `BUY`, `SELL`, or `WAIT` decisions, enforce deterministic risk management independent of any LLM or ML model, evaluate post-trade performance attribution and drift governance, and integrate with MetaTrader 5 (MT5) for backtesting and strictly gated **DEMO-only** closed-loop execution.

---

## 2. Project Capabilities & Completed Roadmap (Phases 0–9)

### Currently Implemented (Phases 0 through 9.3)
- **Phase 0 — Project Foundation (`src/atlas_ai/core/`):**
  - Standardized Python `src/` package layout (`atlas_ai`), `AppSettings` (`config.py`) with a deterministic safety gate blocking unauthorized `live` trading mode, rotating file logging (`logger.py`), and CLI readiness entrypoint (`src/atlas_ai/main.py`).
- **Phase 1 — Core Domain Schemas & Deterministic Risk Engine Foundation (`src/atlas_ai/domain/`, `src/atlas_ai/risk/`):**
  - Immutable (`frozen=True`) Decimal-based domain models: `Candle`, `Timeframe`, `InstrumentSpec` (`market.py`), `SignalDirection`, `TradeSignal` (`signals.py`), `AccountState`, `RiskParameters`, `PositionSizeResult`, `RiskDecision`, `RiskRejectionCode` (`risk.py`).
  - Standalone, LLM-independent `RiskEngine` and deterministic `calculate_position_size()` (`src/atlas_ai/risk/engine.py`) enforcing 10 sequential pre-trade safety gates and downward (`ROUND_DOWN`) volume step rounding.
- **Phase 2 — Market Data Ingestion, Validation & Feature Pipeline (`src/atlas_ai/data/`, `src/atlas_ai/features/`):**
  - Historical CSV OHLCV ingestion (`HistoricalCSVLoader`), OHLCV & calendar-aware time-series integrity validator (`MarketDataValidator`, `check_timeseries_integrity`), and lookahead-free `FeaturePipeline`.
- **Phase 3 — Advanced Strategy Engine & Multi-Rule SMC/ICT Signal Generation (`src/atlas_ai/strategies/`):**
  - `MarketRegimeDetector`, `EmaTrendStrategy`, `MarketStructureStrategy`, `LiquiditySweepStrategy`, `FVGDetectionStrategy`, `FVGRetestStrategy`, `PremiumDiscountAnalyzer`, `PreviousDayLevelsAnalyzer`, `KillZoneSessionAnalyzer`, `MultiTimeframeAnalyzer` (`filter_closed_candles_as_of`), `NewsFilter`, `ConfluenceEngine`, `SignalGenerator`, `SignalLifecycleManager`, and `StrategyEngine`.
- **Phase 4 — Historical Backtesting Engine (`src/atlas_ai/backtest/`):**
  - Chronological event-driven `BacktestEngine`, `ExecutionSimulator` (spread, adverse slippage, commission, post-cost effective RR validation, conservative `STOP_LOSS` intrabar priority), `PortfolioTracker` (Decimal accounting & 5 PM NY DST-aware rollover), and `calculate_backtest_metrics()`.
- **Phase 5 — Educational Knowledge Ingestion & Structured Rule Extraction (`src/atlas_ai/knowledge/`):**
  - Multi-format `KnowledgeDocumentLoader` & `TextChunker` (`.txt`, `.md`, `.srt`, `.vtt`, `.json`), offline `DeterministicRuleExtractor` & `LLMExtractorAdapter`, `KnowledgeRuleValidator`, `KnowledgeRepository`, and `KnowledgeBasedStrategy(BaseStrategy)`.
- **Phase 6 — Machine Learning & Statistical Intelligence (`src/atlas_ai/models/`):**
  - Leakage-safe `TripleBarrierLabeler` (`labeling.py`) computing `BUY`, `SELL`, and `WAIT` labels over a bounded forward horizon $H$ with conservative `STOP_LOSS`-first intrabar priority and deterministic dual-outcome conflict resolution.
  - Causal, stationary `MLFeatureBuilder` and train-fold-only `FoldStandardScaler` (`features.py`).
  - Classical statistical & ML classifiers (`classifiers.py`): `RegularizedLogisticClassifier` (multinomial L2 softmax) and `GradientBoostedStumpEnsemble` (second-order Newton GBDT) with deterministic JSON artifact persistence (`save_artifact` / `load_model_artifact`).
  - Chronological `WalkForwardValidator` (`validation.py`) enforcing `purge_bars >= horizon_bars` (`ModelDataLeakageError`), validation-only model selection, separate `BUY`/`SELL`/`WAIT` classification metrics, Brier/ECE calibration, and R-multiple trading expectancy.
  - `MLSignalStrategy(BaseStrategy)` (`strategy.py`) integrating causally with `StrategyEngine`, `ConfluenceEngine`, `SignalGenerator`, `SignalLifecycleManager`, `RiskEngine`, and `BacktestEngine`.
- **Phase 7 — MetaTrader 5 (MT5) Integration & Demo Execution Engine (`src/atlas_ai/execution/`):**
  - **Pluggable Backend & Lazy MT5 Loading (`backend.py`, `models.py`, `config.py`, `exceptions.py`):** `@runtime_checkable` `MT5TerminalBackend` protocol, `RealMetaTrader5Backend` with lazy `importlib.import_module("MetaTrader5")` and hardware-level `trade_mode == 0` guard, and deterministic offline `MockMT5TerminalBackend`.
  - **Connection & Market Data Adapters (`connection.py`, `market_data.py`):** `MT5ConnectionAdapter` enforcing `account_info.trade_mode == 0 (ACCOUNT_TRADE_MODE_DEMO)`, expected login/server validation, bounded reconnection with DEMO re-verification, execution-mode-aware filling policy resolution (`resolve_supported_filling_mode`), and `MT5MarketDataAdapter` retrieving UTC-normalized Bid/Ask ticks and closed candles (`start_pos=1`) integrated with Phase 2 `MarketDataPipeline`.
  - **Crash-Safe OS-Locked Persistence & Account Risk Accounting (`persistence.py`, `account_state.py`):** `FileLock` (`msvcrt.locking` on Windows, `fcntl.flock` on POSIX) with automatic OS kernel crash release, `CrashSafeJsonPersistence` with atomic temp-file + `os.fsync` + `os.replace`, SHA-256 checksums, monotonic generations, `previous_checksum` hash chains, valid `.tmp.*` envelope recovery, and stale `.bak` anti-rollback protection; `PersistentAccountRiskTracker` enforcing mandatory `TrustedBalanceAnchor` balance completeness verification, all 18 official MQL5 (`0..17`) deal and commission classifications, broker credit filtering, and DST-aware 5 PM `America/New_York` rollover.
  - **14-Step Pre-Trade Safety Gate, Write-Ahead Execution Ledger & Demo Engine (`safety.py`, `ledger.py`, `engine.py`):** `PreTradeSafetyGate` enforcing 14 deterministic pre-trade checks; `PersistentExecutionLedger` persisting `CREATED` $\rightarrow$ `VALIDATED` $\rightarrow$ `SUBMITTING` before `order_send()`, blocking new submissions on any unresolved intent or recovery lock, and reconciling against broker positions, working orders, and historical deals; and `MT5DemoExecutionEngine` enforcing pre-send TOCTOU re-checks, complete MQL5 retcode handling, post-accept broker verification, and immediate terminal shutdown on `10032 (TRADE_RETCODE_ONLY_REAL)` or non-demo accounts.
- **Phase 8 — Post-Trade Analytics, Performance Attribution & Drift/Promotion Governance (`src/atlas_ai/analytics/`):**
  - **Phase 8.1 & 8.1.1 — Trade Reconstruction & Persistent Trade Journal (`trade_reconstructor.py`, `journal.py`, `models.py`, `config.py`, `exceptions.py`):** `PersistentTradeJournal` using Phase 7 `CrashSafeJsonPersistence` with mandatory pre-send `EntryContextSnapshot` hook bound to `MT5DemoExecutionEngine` (aborting orders to `FAILED` before `order_send()` if entry-context persistence fails); `DemoTradeReconstructor` reconstructing closed and partially closed MT5 DEMO positions from deals (`DEAL_ENTRY_IN`, `DEAL_ENTRY_OUT`, `DEAL_TYPE_COMMISSION`), enforcing `net_pnl == gross_pnl - commission - fee + swap` without double-counting, tracking `PartialExitProgressRecord`, and failing closed on `DEAL_ENTRY_INOUT (2)` or `DEAL_ENTRY_OUT_BY (3)`.
  - **Phase 8.2 — Multi-Dimensional Performance Attribution & Execution Quality (`attribution.py`):** `PerformanceAttributionEngine` computing Phase 4-compatible `BacktestMetrics`, R-multiple Sharpe/Sortino/drawdown, `ExecutionQualityReport` (signed entry/exit slippage in `symbol_info.point`, adverse slippage rate, commission/swap cost drag), and non-overlapping single-label (`symbol`, `timeframe`, `direction`, `regime`, `killzone_session`, `strategy_id`, `ml_model_id`, `context_attribution_status`) vs. multi-label (`contributing_strategy_ids`, `contributing_knowledge_rule_ids`) segment breakdowns.
  - **Phase 8.3 — Drift Detection, Calibration Monitoring & Human-Gated Promotion Governance (`drift_and_governance.py`):** `DriftDetectionEngine` evaluating realized R-expectancy degradation, consecutive loss streaks, execution slippage degradation (`EXECUTION_DEGRADED`), post-trade ML Brier/ECE calibration drift, and advisory-only knowledge-rule quarantine recommendations (`QUARANTINE_RECOMMENDED`) without mutating `KnowledgeRepository`; `GovernancePromotionEvaluator` enforcing chronological walk-forward anti-leakage (`purge_bars >= horizon_bars`, strictly disjoint non-overlapping test folds), immutable Phase 1 `RiskParameters` ceilings (rejecting any risk relaxation or partial risk parameter arguments), and mandatory human approval (`requires_human_approval == True`).
- **Phase 9 — Closed-Loop Demo Runtime Orchestration & Persistent Emergency Circuit Breaker (`src/atlas_ai/runtime/`):**
  - **Phase 9.1 — Persistent Emergency Circuit Breaker & Runtime Config (`config.py`, `models.py`, `exceptions.py`, `circuit_breaker.py`):** Immutable `DemoRuntimeConfig` validating timeframe hierarchies (`higher_timeframes > primary_timeframe`) and circuit-breaker thresholds; crash-safe `EmergencyStopController` persisted via `CrashSafeJsonPersistence` and bound to `(account_login, server)`, latching `LATCHED_HALTED` on manual operator triggers, `DriftStatus.QUARANTINE_RECOMMENDED`, `DriftStatus.EXECUTION_DEGRADED`, `LedgerHealthStatus.RECOVERY_REQUIRED_LOCKED`, or consecutive connection/persistence failure thresholds, surviving restarts without auto-unlocking, and requiring explicit human reset with a non-empty rationale (`reset_emergency_stop()`).
  - **Phase 9.2 — Closed-Loop Demo Trading Orchestrator (`orchestrator.py`):** `PersistentBarStateTracker` providing crash-safe, account/server-bound `last_evaluated_bar_timestamp_utc` deduplication per `(symbol, timeframe)` with anti-regression protection; `DemoTradingOrchestrator` executing a deterministic 14-step closed-loop cycle (`run_single_cycle()`, `run_session()`, `request_shutdown()`) that always runs connection verification, unresolved order reconciliation, closed-trade journal synchronization, account risk state reconstruction, drift/ledger health evaluation, and emergency circuit-breaker authorization (Steps 1–7) before bar-close deduplication (Step 8), market data/feature/strategy/risk evaluation (Steps 9–12), journal context staging (Step 13), and dry-run or live-demo execution + bar-timestamp commit (Step 14).

---

## 3. Installation, Running Tests & Offline Verification (Windows PowerShell)

Run the following commands from the project root (`c:\Users\User\OneDrive\Desktop\ATLAS AI`):

```powershell
# 1. Run the complete automated offline test suite (Phases 0 through 9 — 153 tests)
.\.venv\Scripts\pytest.exe -v

# 2. Run the focused Phase 9 runtime & emergency circuit-breaker test suite (14 tests)
.\.venv\Scripts\pytest.exe tests/test_runtime.py -v

# 3. Run the focused Phase 8 post-trade analytics, attribution & governance test suite (15 tests)
.\.venv\Scripts\pytest.exe tests/test_analytics.py -v

# 4. Run the focused Phase 7 MT5 execution & safety test suite (29 tests)
.\.venv\Scripts\pytest.exe tests/test_execution.py -v

# 5. Run the ATLAS AI CLI readiness check
.\.venv\Scripts\python.exe -m atlas_ai.main

# 6. Run a deterministic historical XAUUSD backtest on the synthetic H1 fixture
.\.venv\Scripts\python.exe -c "from atlas_ai.data import HistoricalCSVLoader; from atlas_ai.backtest import BacktestEngine, BacktestConfig; df = HistoricalCSVLoader().load_csv('tests/fixtures/synthetic_xauusd_h1.csv'); res = BacktestEngine(BacktestConfig(warmup_bars=50)).run(df); print(res.metrics.model_dump())"
```

See [docs/architecture.md](docs/architecture.md) for the complete system architecture and Phase 7–9 execution, analytics, and runtime safety specifications.

---

## 4. Important Warnings, Current Limitations & Pre-Demo Operational Requirements

1. **No Unrestricted LLM or ML Execution:** Neither an LLM, an educational rule, nor a statistical/ML model has direct authority to place trades or override risk limits. Every signal must pass through `EmergencyStopController`, `SignalLifecycleManager`, `PreTradeSafetyGate`, and Phase 1 `RiskEngine.evaluate()`.
2. **Live Trading Strictly Blocked (DEMO-Only Enforcement):** `ExecutionMode.LIVE` and `live_trading_enabled=True` are blocked by `TradingSafetyError` and `LiveTradingBlockedError`. Any connection to an MT5 `ACCOUNT_TRADE_MODE_REAL (2)` or `ACCOUNT_TRADE_MODE_CONTEST (1)` account, or any `TRADE_RETCODE_ONLY_REAL (10032)` response, immediately shuts down the terminal connection and raises `LiveTradingBlockedError`.
3. **Not Production-Ready & No Profitability Guarantee:** ATLAS AI `v0.1.0` is an offline-verified research and **DEMO-only** engineering framework. It has not been validated on a live or demo broker feed and carries zero guarantee of trading profitability.
4. **Synthetic Test Data Notice:** All CSV, mock terminal (`MockMT5TerminalBackend`), and inline multi-regime test fixtures under `tests/` are strictly synthetic test data created for deterministic offline verification and must never be interpreted as evidence of real-world trading performance.
5. **Remaining Requirements Before Supervised MT5 Demo Operation:**
   - **Windows Host with MetaTrader 5 Terminal & Package:** Running against an actual MT5 demo account requires installing the official `MetaTrader5` Python package on a Windows host with a logged-in MT5 terminal (`ACCOUNT_TRADE_MODE_DEMO = 0`) and algorithmic trading enabled.
   - **Explicit `TrustedBalanceAnchorConfig` Configuration:** Before running `PersistentAccountRiskTracker` against an actual MT5 demo account, operators must configure `TrustedBalanceAnchorConfig` matching the demo account's initial deposit or verified rollover balance.
   - **Supervised Dry-Run Shadow Validation (`dry_run_only=True`):** Operators should first run `DemoTradingOrchestrator` with `DemoRuntimeConfig(dry_run_only=True)` to validate live demo market-data ingestion, candle-close deduplication, strategy signals, and risk decisions with zero broker order submissions before enabling `dry_run_only=False` on a demo account.
