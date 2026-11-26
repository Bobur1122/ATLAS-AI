# ATLAS AI — System Architecture & Phases 7–9 Safety Reference

**Version:** `0.1.0`  
**Completed Phases:** Phase 0 through Phase 9 (Steps 9.1–9.3)  
**Primary Instrument:** `XAUUSD`

---

## 1. Architectural Principles

1. **Deterministic Risk & Circuit-Breaker Supremacy:** No LLM, educational rule (`KnowledgeBasedStrategy`), or machine-learning classifier (`MLSignalStrategy`) can bypass Phase 9 `EmergencyStopController.assert_trading_authorized()`, Phase 1 `RiskEngine.evaluate()`, or Phase 7 `PreTradeSafetyGate.validate_and_build_order()`.
2. **Strict DEMO-Only Execution:** Live trading (`ExecutionMode.LIVE`, `live_trading_enabled=True`, `ACCOUNT_TRADE_MODE_REAL = 2`, `ACCOUNT_TRADE_MODE_CONTEST = 1`, or MQL5 `TRADE_RETCODE_ONLY_REAL = 10032`) is blocked across configuration, connection, pre-trade validation, execution engine, and hardware-level backend layers.
3. **Fail-Closed Accounting, Analytics & Persistence:** Unverified account history, empty broker deal cache (`history_deals_get() == ()`), ambiguous commission/dividend attribution, `DEAL_ENTRY_INOUT` / `DEAL_ENTRY_OUT_BY` position reversals, corrupted persistence files, stale backup rollbacks, unresolved order intents, or latched emergency circuit-breaker states immediately block order submission.
4. **Offline-First Determinism:** Importing `atlas_ai` and running the entire test suite (`153` tests across 15 test modules) never requires an installed `MetaTrader5` package or a running MT5 terminal.

---

## 2. Package Structure (`src/atlas_ai/`)

| Package / Module | Phase | Responsibility |
| :--- | :---: | :--- |
| `core/` (`config.py`, `logger.py`, `exceptions.py`) | Phase 0 | Environment configuration (`AppSettings`), rotating file logging, base exception hierarchy (`AtlasError`, `TradingSafetyError`), and live-mode safety gate. |
| `domain/` (`market.py`, `signals.py`, `risk.py`) | Phase 1 | Immutable (`frozen=True`) Decimal domain schemas (`Candle`, `Timeframe`, `InstrumentSpec`, `TradeSignal`, `AccountState`, `RiskParameters`, `RiskDecision`). |
| `risk/` (`engine.py`) | Phase 1 | Deterministic `RiskEngine` and `calculate_position_size()` with downward (`ROUND_DOWN`) lot-step rounding and 10 sequential risk checks. |
| `data/` & `features/` | Phase 2 | `HistoricalCSVLoader`, `MarketDataValidator`, `check_timeseries_integrity()`, `FeaturePipeline`, and causal technical indicators. |
| `strategies/` | Phase 3 | Market regime detection, SMC/ICT analyzers (`BOS`/`CHoCH`, liquidity sweeps, FVG detection/retest, premium/discount, PDH/PDL, killzones), `MultiTimeframeAnalyzer`, `NewsFilter`, `ConfluenceEngine`, `SignalGenerator`, `SignalLifecycleManager`, and `StrategyEngine`. |
| `backtest/` | Phase 4 | Event-driven `BacktestEngine`, `ExecutionSimulator` (spread, slippage, commissions, intrabar SL-first priority), `PortfolioTracker` (5 PM NY DST rollover), and performance metrics. |
| `knowledge/` | Phase 5 | Multi-format educational document loader, text chunker, deterministic/LLM rule extraction, `KnowledgeRuleValidator`, `KnowledgeRepository`, and `KnowledgeBasedStrategy`. |
| `models/` | Phase 6 | `TripleBarrierLabeler`, `MLFeatureBuilder`, `FoldStandardScaler`, `RegularizedLogisticClassifier`, `GradientBoostedStumpEnsemble`, `WalkForwardValidator`, and `MLSignalStrategy`. |
| `execution/` | Phase 7 | MetaTrader 5 terminal abstraction, connection/market-data adapters, OS-locked crash-safe persistence, `TrustedBalanceAnchor` account risk tracker, write-ahead execution ledger, 14-step `PreTradeSafetyGate`, and `MT5DemoExecutionEngine`. |
| `analytics/` | Phase 8 | `PersistentTradeJournal` (with mandatory pre-send `EntryContextSnapshot` hook), `DemoTradeReconstructor`, `PerformanceAttributionEngine`, `DriftDetectionEngine`, and human-gated `GovernancePromotionEvaluator`. |
| `runtime/` | Phase 9 | `DemoRuntimeConfig`, persistent crash-safe `EmergencyStopController`, `PersistentBarStateTracker` (closed-bar evaluation deduplication), and 14-step closed-loop `DemoTradingOrchestrator`. |

---

## 3. Phase 7 MetaTrader 5 Demo Execution Architecture

### 3.1 Five-Layer DEMO-Only Safety Barrier
1. **Configuration Gate (`ExecutionSafetyConfig` & `AppSettings`):** Raising `LiveTradingBlockedError` / `TradingSafetyError` if `execution_mode == ExecutionMode.LIVE` or `live_trading_enabled == True`.
2. **Connection & Reconnection Gate (`MT5ConnectionAdapter.verify_demo_account_only`):** Requiring `account_info().trade_mode == 0 (ACCOUNT_TRADE_MODE_DEMO)` (never server-name heuristics) and matching `expected_login` and `expected_server`. Any `REAL (2)` or `CONTEST (1)` account triggers immediate `backend.shutdown()` and `LiveTradingBlockedError`.
3. **Pre-Trade Safety Gate (`PreTradeSafetyGate.validate_and_build_order`):** Re-verifying DEMO account status, terminal/account algorithmic trading permissions, and all 14 deterministic pre-trade checks before building an `MT5OrderRequest`.
4. **Execution Engine Pre-Send & Retcode Gate (`MT5DemoExecutionEngine.execute_signal`):** Re-verifying `verify_demo_account_only()` immediately before `backend.order_send()` and shutting down the terminal if the broker returns `TRADE_RETCODE_ONLY_REAL (10032)`.
5. **Backend Hardware Guard (`RealMetaTrader5Backend.order_check` & `order_send`):** Querying `mt5.account_info().trade_mode` immediately prior to calling the underlying MT5 API and shutting down on any non-demo account.

### 3.2 Crash-Safe OS-Locked Persistence (`persistence.py`)
- **Cross-Process OS File Lock (`FileLock`):** Uses `msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)` on Windows and `fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)` on POSIX on `<target>.lock`, initializing the 1-byte marker only after acquiring the exclusive kernel lock and never unlinking `.lock` on release. Process crashes release the lock automatically via the OS kernel file-descriptor table.
- **Hash-Chained Envelopes (`PersistedEnvelope`):** Every write increments `generation` monotonically ($1, 2, 3, \dots$), chains `previous_checksum` to generation $G - 1$'s `payload_checksum` (or `GENESIS_CHECKSUM` for generation 1), binds `(account_login, server)`, and writes via `<target>.tmp.<pid>.<uuid>` + `os.fsync()` + `os.replace()`.
- **Safe Crash Recovery:** Detects both `<primary>.tmp.*` and `<primary>.bak.tmp.*` interrupted writes, recovers valid higher-generation `.tmp.*` envelopes, never silently restores a stale `.bak` execution ledger, and fails closed (`PersistenceCorruptionError` / `LedgerRecoveryLockError`) if all candidate envelopes are corrupted or locked.

### 3.3 `TrustedBalanceAnchor` & Official MQL5 (`0..17`) Deal Accounting (`account_state.py`)
- **Mandatory Balance Completeness Proof:** Reconstructs `daily_starting_balance` from either a verified `PERSISTED_ROLLOVER_ANCHOR` or an explicit `CONFIGURED_BALANCE_ANCHOR`, rejecting `None` or empty (`len == 0`) deal history and future-dated deals (`time_utc > reference_now_utc`).
- **Complete MQL5 Deal Classification (`0..17`):**
  - Execution deals (`DEAL_TYPE_BUY = 0`, `DEAL_TYPE_SELL = 1`), charges (`4`), commissions (`7..11`), negative interest (`12`), position-verified dividends (`DEAL_DIVIDEND = 15`, `DEAL_DIVIDEND_FRANKED = 16`), and taxes (`DEAL_TAX = 17`) are attributed to `TRADING_PNL`.
  - Account-level deposits/withdrawals (`DEAL_TYPE_BALANCE = 2`), credits (`3`), bonuses (`6`), and positive free-margin interest (`12`) are attributed to `NON_TRADING_CASH_FLOW` so deposits/credits never mask daily trading losses.
  - Ambiguous or retroactive types (`DEAL_TYPE_CORRECTION = 5`, `DEAL_TYPE_BUY_CANCELED = 13`, `DEAL_TYPE_SELL_CANCELED = 14`, unknown codes) and inline (`commission`/`fee`) + separate `DEAL_TYPE_COMMISSION` duplicates fail closed with `AccountStateReconstructionError`.
- **5 PM `America/New_York` DST-Aware Rollover:** Computes UTC trading-day bounds using IANA `ZoneInfo("America/New_York")` across EST (`UTC-5`) and EDT (`UTC-4`), strips consistent `account.credit` from floating PnL and effective cash equity, counts account-wide open positions via `positions_get(None)`, and preserves `peak_balance` high-water marks across rollovers.

### 3.4 14-Step `PreTradeSafetyGate` (`safety.py`) & Write-Ahead `PersistentExecutionLedger` (`ledger.py`, `engine.py`)
- Validates DEMO mode, terminal connectivity, signal prices, ledger readiness, live broker position/working-order conflicts, `SignalLifecycleManager` state, `MT5SymbolTradeMode`, tick freshness, Bid-Ask spread in `symbol_info.point`, live price drift, stop/freeze levels, live `RiskEngine.evaluate()`, lot volume bounds/steps, execution-mode filling policy, and MT5 `order_check()`.
- Persists `CREATED` $\rightarrow$ `VALIDATED` $\rightarrow$ `SUBMITTING` before `backend.order_send()`, transitions ambiguous retcodes (`10011`, `10012`, `10023`, `10031`, `None`, exceptions) to `UNKNOWN` $\rightarrow$ `RECONCILIATION_REQUIRED` without blind retries, and verifies fills against broker positions/deals.

---

## 4. Phase 8 Post-Trade Analytics, Attribution & Drift/Promotion Governance (`src/atlas_ai/analytics/`)

### 4.1 Closed-Trade Reconstruction & Mandatory Pre-Send Entry Context (`trade_reconstructor.py`, `journal.py`)
- **Mandatory Pre-Send `EntryContextSnapshot` Persistence:** `PersistentTradeJournal.bind_execution_engine()` hooks into `MT5DemoExecutionEngine` after `SUBMITTING` persistence and before `backend.order_send()`, binding the staged `StrategyAnalysisReport` to the exact approved volume, live price, and 1R `initial_risk_amount`. If journal persistence fails, the order is aborted to `FAILED` before `order_send()` is invoked.
- **Exact Cash-Flow Identity & Partial Exits:** `DemoTradeReconstructor.sync_closed_demo_trades()` groups `DEAL_ENTRY_IN`, `DEAL_ENTRY_OUT`, and `DEAL_TYPE_COMMISSION` deals by position, enforces `net_pnl == gross_pnl - commission - fee + swap` without double-counting, records `PartialExitProgressRecord` for partially closed positions without prematurely terminating `SignalLifecycleManager`, and fails closed on `DEAL_ENTRY_INOUT (2)` or `DEAL_ENTRY_OUT_BY (3)`.

### 4.2 Performance Attribution, Drift Detection & Promotion Governance (`attribution.py`, `drift_and_governance.py`)
- **Multi-Dimensional Attribution (`PerformanceAttributionEngine`):** Computes Phase 4-compatible `BacktestMetrics`, R-multiple Sharpe/Sortino/drawdown, `ExecutionQualityReport`, and single-label vs. multi-label segment breakdowns.
- **Drift & Calibration Monitoring (`DriftDetectionEngine`):** Evaluates realized R-expectancy, consecutive loss streaks, execution slippage degradation (`EXECUTION_DEGRADED`), ML Brier/ECE calibration drift, and advisory-only knowledge-rule quarantine recommendations (`QUARANTINE_RECOMMENDED`) with strict severity precedence (`QUARANTINE_RECOMMENDED` $>$ `EXECUTION_DEGRADED` $>$ `WARNING_DRIFT_DETECTED` $>$ `INSUFFICIENT_SAMPLE` $>$ `HEALTHY`).
- **Human-Gated Promotion Governance (`GovernancePromotionEvaluator`):** Enforces chronological walk-forward anti-leakage (`purge_bars >= horizon_bars`, non-overlapping chronological test folds), rejects any relaxation of Phase 1 `RiskParameters` (requiring baseline and candidate risk parameters together when provided), and enforces `requires_human_approval == True`.

---

## 5. Phase 9 Closed-Loop Demo Runtime & Persistent Emergency Circuit Breaker (`src/atlas_ai/runtime/`)

### 5.1 Persistent Crash-Safe `EmergencyStopController` (`circuit_breaker.py`, `config.py`, `models.py`)
- **Crash-Safe Latching:** Persists `EmergencyStopState` via Phase 7 `CrashSafeJsonPersistence` bound to `(account_login, server)`. Automatically latches `LATCHED_HALTED` on manual operator stops (`trigger_manual_stop`), `DriftStatus.QUARANTINE_RECOMMENDED`, `DriftStatus.EXECUTION_DEGRADED`, `LedgerHealthStatus.RECOVERY_REQUIRED_LOCKED`, or consecutive connection/persistence failure thresholds (`max_consecutive_connection_failures`, `max_consecutive_persistence_failures`).
- **No Auto-Unlock Across Restarts:** Once latched, `assert_trading_authorized()` raises `CircuitBreakerLatchedError` across process restarts until an explicit human reset (`reset_emergency_stop(operator_rationale=...)`) with a non-empty rationale is committed to disk, preserving all historical `EmergencyAuditEvent` records.

### 5.2 Persistent Bar-Close Deduplication (`PersistentBarStateTracker`)
- Persists `BarEvaluationState` (`last_evaluated_bar_timestamps_utc` keyed by `<SYMBOL>:<TIMEFRAME>`) via `CrashSafeJsonPersistence` bound to `(account_login, server)`.
- Rejects corrupted state (`PersistenceCorruptionError`), account/server mismatch (`PersistenceAccountMismatchError`), lingering `.tmp.*` recovery locks (`LedgerRecoveryLockError`), and backward bar-timestamp regression (`PersistenceRollbackDetectedError`).

### 5.3 14-Step Closed-Loop `DemoTradingOrchestrator` (`orchestrator.py`)
Every call to `DemoTradingOrchestrator.run_single_cycle()` executes in strict order:
1. `01_VERIFY_CONNECTION` (`connection.ensure_connected()`, recording connection failure/success in `EmergencyStopController`).
2. `02_RECONCILE_UNRESOLVED_ORDERS` (`execution_engine.reconcile_unresolved_orders()`).
3. `03_SYNC_CLOSED_TRADES` (`trade_reconstructor.sync_closed_demo_trades()`).
4. `04_RECONSTRUCT_ACCOUNT_STATE` (`account_tracker.reconstruct_account_state()`).
5. `05_EVALUATE_DRIFT_AND_LEDGER_HEALTH` (`drift_engine.evaluate_trades()` and `reconciliation_report.health_status`).
6. `06_EVALUATE_EMERGENCY_STOP_STATE` (`emergency_stop.evaluate_ledger_health()` and `evaluate_drift_report()`).
7. `07_ASSERT_TRADING_AUTHORIZED` (`emergency_stop.assert_trading_authorized()`).
8. `08_CHECK_BAR_CLOSE_DEDUPLICATION` (`market_data.fetch_closed_candles()` with `start_pos=1` and `bar_state_tracker.is_bar_already_evaluated()`, returning `SKIPPED_DUPLICATE_BAR` after Steps 1–7 if already evaluated).
9. `09_FETCH_SYNCHRONIZED_MARKET_DATA` (`fetch_market_data_pipeline_result()` across `primary_timeframe` and `higher_timeframes`).
10. `10_RUN_FEATURE_PIPELINE` (`feature_pipeline.transform()`).
11. `11_RUN_STRATEGY_ENGINE` (`strategy_engine.analyze()`).
12. `12_EVALUATE_RISK` (checking `WAIT` $\rightarrow$ `NO_EXECUTABLE_SIGNAL` and `RiskDecision` $\rightarrow$ `RISK_REJECTED`).
13. `13_STAGE_JOURNAL_CONTEXT` (`journal.stage_analysis_report()`).
14. `14_EXECUTE_OR_DRY_RUN_AND_COMMIT_BAR`:
    - When `config.dry_run_only == True`: structurally bypasses `execution_engine.execute_signal()` and `backend.order_send()`, commits `last_evaluated_bar_timestamp_utc`, and returns `DRY_RUN_EVALUATED`.
    - When `config.dry_run_only == False`: invokes `execution_engine.execute_signal()` and commits `last_evaluated_bar_timestamp_utc` only after the execution outcome (`EXECUTED_ORDER`, `BROKER_REJECTED`, or `AMBIGUOUS_EXECUTION_REQUIRES_RECONCILIATION`) is durably recorded in the ledger.

---

## 6. Known Limitations & Operational Requirements

1. **DEMO-Only & Non-Production Status:** ATLAS AI `v0.1.0` is strictly a DEMO-only research and engineering system. Live trading is disabled in code and configuration, and no guarantee of profitability is made.
2. **Drift Report Initial Balance Reference (`L-9.2-Obs`):** During Step 5 of `DemoTradingOrchestrator.run_single_cycle()`, `drift_engine.evaluate_trades()` receives `initial_balance=account_state.daily_starting_balance`. R-multiple expectancy, loss streaks, slippage, and Brier/ECE calibration are unaffected; embedded `overall_metrics.net_profit_pct` reflects return relative to `daily_starting_balance`.
3. **Audit History Growth (`L-9.1-1`):** `EmergencyStopController` retains the full `audit_history` tuple in `emergency_stop_state.json` across resets.
4. **External MT5 Terminal & `TrustedBalanceAnchorConfig` Setup:** Supervised demo execution requires a Windows host with the `MetaTrader5` Python package installed, an authenticated MT5 demo terminal (`trade_mode == 0`), and a configured `TrustedBalanceAnchorConfig` matching the demo account's initial deposit or verified rollover balance.
