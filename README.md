# ATLAS AI — Modular Autonomous Forex Trading Intelligence System

**Current Version:** `0.1.0`  
**Current Phase:** Phase 7 — MetaTrader 5 (MT5) Integration & Demo Execution Engine (Completed)  
**Primary Instrument:** `XAUUSD` (Gold vs. US Dollar)

---

## 1. What ATLAS AI Is

ATLAS AI is a modular, safety-first autonomous Forex trading intelligence system designed to learn structured trading rules from educational materials, analyze market structure and statistical patterns on historical and live market data, generate disciplined `BUY`, `SELL`, or `WAIT` decisions, enforce deterministic risk management independent of any LLM or ML model, and integrate with MetaTrader 5 (MT5) for backtesting and strictly gated **DEMO-only** order execution.

---

## 2. Project Capabilities & Roadmap Overview

### Currently Implemented (Phases 0, 1, 2, 3, 4, 5, 6 & 7)
- **Phase 0 — Project Foundation:**
  - Standardized Python `src/` package layout (`atlas_ai`), `AppSettings` (`src/atlas_ai/core/config.py`) with a deterministic safety gate blocking unauthorized `live` trading mode, rotating file logging (`src/atlas_ai/core/logger.py`), and CLI entrypoint (`src/atlas_ai/main.py`).
- **Phase 1 — Core Domain Schemas & Deterministic Risk Engine Foundation:**
  - Immutable (`frozen=True`) Decimal-based domain models: `Candle`, `Timeframe`, `InstrumentSpec` (`market.py`), `SignalDirection`, `TradeSignal` (`signals.py`), `AccountState`, `RiskParameters`, `PositionSizeResult`, `RiskDecision`, `RiskRejectionCode` (`risk.py`).
  - Standalone, LLM-independent `RiskEngine` and deterministic `calculate_position_size()` (`src/atlas_ai/risk/engine.py`) enforcing 10 sequential pre-trade safety gates and downward (`ROUND_DOWN`) volume step rounding.
- **Phase 2 — Market Data Ingestion, Validation & Feature Pipeline:**
  - Historical CSV OHLCV ingestion (`HistoricalCSVLoader`), OHLCV & calendar-aware time-series integrity validator (`MarketDataValidator`, `check_timeseries_integrity`), and lookahead-free `FeaturePipeline` (`src/atlas_ai/data/`, `src/atlas_ai/features/`).
- **Phase 3 — Advanced Strategy Engine & Multi-Rule SMC/ICT Signal Generation:**
  - `MarketRegimeDetector`, `EmaTrendStrategy`, `MarketStructureStrategy`, `LiquiditySweepStrategy`, `FVGDetectionStrategy`, `FVGRetestStrategy`, `PremiumDiscountAnalyzer`, `PreviousDayLevelsAnalyzer`, `KillZoneSessionAnalyzer`, `MultiTimeframeAnalyzer` (`filter_closed_candles_as_of`), `NewsFilter`, `ConfluenceEngine`, `SignalGenerator`, `SignalLifecycleManager`, and `StrategyEngine` (`src/atlas_ai/strategies/`).
- **Phase 4 — Historical Backtesting Engine:**
  - Chronological event-driven `BacktestEngine`, `ExecutionSimulator` (spread, adverse slippage, commission, post-cost effective RR validation, conservative `STOP_LOSS` intrabar priority), `PortfolioTracker` (Decimal accounting & 5 PM NY DST-aware rollover), and `calculate_backtest_metrics()` (`src/atlas_ai/backtest/`).
- **Phase 5 — Educational Knowledge Ingestion & Structured Rule Extraction:**
  - Multi-format `KnowledgeDocumentLoader` & `TextChunker` (`.txt`, `.md`, `.srt`, `.vtt`, `.json`), offline `DeterministicRuleExtractor` & `LLMExtractorAdapter`, `KnowledgeRuleValidator`, `KnowledgeRepository`, and `KnowledgeBasedStrategy(BaseStrategy)` (`src/atlas_ai/knowledge/`).
- **Phase 6 — Machine Learning & Statistical Intelligence:**
  - Leakage-safe `TripleBarrierLabeler` (`src/atlas_ai/models/labeling.py`) computing `BUY`, `SELL`, and `WAIT` labels over a bounded forward horizon $H$ with conservative `STOP_LOSS`-first intrabar priority and deterministic dual-outcome conflict resolution.
  - Causal, stationary `MLFeatureBuilder` and train-fold-only `FoldStandardScaler` (`features.py`).
  - Classical statistical & ML classifiers (`classifiers.py`): `RegularizedLogisticClassifier` (multinomial L2 softmax) and `GradientBoostedStumpEnsemble` (second-order Newton GBDT) with deterministic JSON artifact persistence (`save_artifact` / `load_model_artifact`).
  - Chronological `WalkForwardValidator` (`validation.py`) enforcing `purge_bars >= horizon_bars` (`ModelDataLeakageError`), validation-only model selection, separate `BUY`/`SELL`/`WAIT` classification metrics, Brier/ECE calibration, and R-multiple trading expectancy.
  - `MLSignalStrategy(BaseStrategy)` (`strategy.py`) integrating causally with `StrategyEngine`, `ConfluenceEngine`, `SignalGenerator`, `SignalLifecycleManager`, `RiskEngine`, and `BacktestEngine`.
- **Phase 7 — MetaTrader 5 (MT5) Integration & Demo Execution Engine (`src/atlas_ai/execution/`):**
  - **Pluggable Backend & Lazy MT5 Loading (`backend.py`, `models.py`, `config.py`, `exceptions.py`):** `@runtime_checkable` `MT5TerminalBackend` protocol, `RealMetaTrader5Backend` with lazy `importlib.import_module("MetaTrader5")` and hardware-level `trade_mode == 0` guard, and deterministic offline `MockMT5TerminalBackend`.
  - **Connection & Market Data Adapters (`connection.py`, `market_data.py`):** `MT5ConnectionAdapter` enforcing `account_info.trade_mode == 0 (ACCOUNT_TRADE_MODE_DEMO)`, expected login/server validation, bounded reconnection with DEMO re-verification, execution-mode-aware filling policy resolution (`resolve_supported_filling_mode`), and `MT5MarketDataAdapter` retrieving UTC-normalized Bid/Ask ticks and closed candles (`start_pos=1`) integrated with Phase 2 `MarketDataPipeline`.
  - **Crash-Safe OS-Locked Persistence & Account Risk Accounting (`persistence.py`, `account_state.py`):** `FileLock` (`msvcrt.locking` on Windows, `fcntl.flock` on POSIX) with automatic OS kernel crash release, `CrashSafeJsonPersistence` with atomic temp-file + `os.fsync` + `os.replace`, SHA-256 checksums, monotonic generations, `previous_checksum` hash chains, valid `.tmp.*` envelope recovery, and stale `.bak` anti-rollback protection; `PersistentAccountRiskTracker` enforcing mandatory `TrustedBalanceAnchor` balance completeness verification, all 18 official MQL5 (`0..17`) deal and commission classifications, broker credit filtering, and DST-aware 5 PM `America/New_York` rollover.
  - **14-Step Pre-Trade Safety Gate, Write-Ahead Execution Ledger & Demo Engine (`safety.py`, `ledger.py`, `engine.py`):** `PreTradeSafetyGate` enforcing 14 deterministic pre-trade checks (including `MT5SymbolTradeMode`, spread in `symbol_info.point`, price drift, stop/freeze levels, `RiskEngine.evaluate()`, volume/filling rules, and `order_check()`); `PersistentExecutionLedger` persisting `CREATED` $\rightarrow$ `VALIDATED` $\rightarrow$ `SUBMITTING` before `order_send()`, blocking new submissions on any unresolved intent or recovery lock, and reconciling against broker positions, working orders, and historical deals via exact token matching; and `MT5DemoExecutionEngine` enforcing pre-send TOCTOU re-checks, complete MQL5 retcode handling (`DETERMINISTIC_REJECTION_RETCODES`, ambiguous `10011`/`10012`/`10023`/`10031`/unknown, `10008 PLACED`, `10010 DONE_PARTIAL`, `10009 DONE`), post-accept broker verification, and immediate terminal shutdown on `10032 (TRADE_RETCODE_ONLY_REAL)` or non-demo accounts.

### Planned Capabilities (Future Phases)
- **Phase 8 — Post-Trade Analytics & Continuous Feedback Loop:** Evaluating completed demo/backtest trades to refine datasets, rules, and statistical models.

---

## 3. Installation, Running Tests & Offline Verification (Windows PowerShell)

Run the following commands from the project root (`c:\Users\User\OneDrive\Desktop\ATLAS AI`):

```powershell
# 1. Run the complete automated offline test suite (Phases 0 through 7 — 124 tests)
.\.venv\Scripts\pytest.exe

# 2. Run the focused Phase 7 MT5 execution & safety test suite (29 tests)
.\.venv\Scripts\pytest.exe tests/test_execution.py -v

# 3. Run the ATLAS AI CLI readiness check
.\.venv\Scripts\python.exe -m atlas_ai.main

# 4. Run a deterministic historical XAUUSD backtest on the synthetic H1 fixture
.\.venv\Scripts\python.exe -c "from atlas_ai.data import HistoricalCSVLoader; from atlas_ai.backtest import BacktestEngine, BacktestConfig; df = HistoricalCSVLoader().load_csv('tests/fixtures/synthetic_xauusd_h1.csv'); res = BacktestEngine(BacktestConfig(warmup_bars=50)).run(df); print(res.metrics.model_dump())"
```

See [docs/architecture.md](docs/architecture.md) for the complete system architecture and Phase 7 execution safety specification.

---

## 4. Important Warnings & Trading Safety Limitations

1. **No Unrestricted LLM or ML Execution:** Neither an LLM, an educational rule, nor a statistical/ML model has direct authority to place trades or override risk limits. Every signal must pass through `SignalLifecycleManager`, `PreTradeSafetyGate`, and Phase 1 `RiskEngine.evaluate()`.
2. **Live Trading Strictly Blocked (DEMO-Only Enforcement):** `ExecutionMode.LIVE` and `live_trading_enabled=True` are blocked by `TradingSafetyError` and `LiveTradingBlockedError`. Any connection to an MT5 `ACCOUNT_TRADE_MODE_REAL (2)` or `ACCOUNT_TRADE_MODE_CONTEST (1)` account, or any `TRADE_RETCODE_ONLY_REAL (10032)` response, immediately shuts down the terminal connection and raises `LiveTradingBlockedError`.
3. **Synthetic Test Data Notice:** All CSV, mock terminal, and inline multi-regime test fixtures under `tests/` are strictly synthetic test data created for deterministic offline verification and must never be interpreted as evidence of real-world trading profitability.
