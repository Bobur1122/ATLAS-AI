# ATLAS AI — System Architecture & Phase 7 Execution Safety Reference

**Version:** `0.1.0`  
**Completed Phases:** Phase 0 through Phase 7  
**Primary Instrument:** `XAUUSD`

---

## 1. Architectural Principles

1. **Deterministic Risk Supremacy:** No LLM, educational rule (`KnowledgeBasedStrategy`), or machine-learning classifier (`MLSignalStrategy`) can bypass Phase 1 `RiskEngine.evaluate()` or Phase 7 `PreTradeSafetyGate.validate_and_build_order()`.
2. **Strict DEMO-Only Execution:** Live trading (`ExecutionMode.LIVE`, `live_trading_enabled=True`, `ACCOUNT_TRADE_MODE_REAL = 2`, `ACCOUNT_TRADE_MODE_CONTEST = 1`, or MQL5 `TRADE_RETCODE_ONLY_REAL = 10032`) is blocked across configuration, connection, pre-trade validation, execution engine, and hardware-level backend layers.
3. **Fail-Closed Accounting & Persistence:** Unverified account history, empty broker deal cache (`history_deals_get() == ()`), ambiguous commission/dividend attribution, corrupted persistence files, stale backup rollbacks, or unresolved order intents immediately block order submission.
4. **Offline-First Determinism:** Importing `atlas_ai` and running the entire test suite (`124` tests) never requires an installed `MetaTrader5` package or a running MT5 terminal.

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
- **Safe Crash Recovery:** Detects both `<primary>.tmp.*` and `<primary>.bak.tmp.*` interrupted writes, recovers valid higher-generation `.tmp.*` envelopes, never silently restores a stale `.bak` execution ledger, and fails closed (`PersistenceCorruptionError`) if all candidate envelopes are corrupted.

### 3.3 `TrustedBalanceAnchor` & Official MQL5 (`0..17`) Deal Accounting (`account_state.py`)
- **Mandatory Balance Completeness Proof:** Reconstructs `daily_starting_balance` from either a verified `PERSISTED_ROLLOVER_ANCHOR` or an explicit `CONFIGURED_BALANCE_ANCHOR`, rejecting `None` or empty (`len == 0`) deal history and future-dated deals (`time_utc > reference_now_utc`).
- **Complete MQL5 Deal Classification (`0..17`):**
  - Execution deals (`DEAL_TYPE_BUY = 0`, `DEAL_TYPE_SELL = 1`), charges (`4`), commissions (`7..11`), negative interest (`12`), position-verified dividends (`DEAL_DIVIDEND = 15`, `DEAL_DIVIDEND_FRANKED = 16`), and taxes (`DEAL_TAX = 17`) are attributed to `TRADING_PNL`.
  - Account-level deposits/withdrawals (`DEAL_TYPE_BALANCE = 2`), credits (`3`), bonuses (`6`), and positive free-margin interest (`12`) are attributed to `NON_TRADING_CASH_FLOW` so deposits/credits never mask daily trading losses.
  - Ambiguous or retroactive types (`DEAL_TYPE_CORRECTION = 5`, `DEAL_TYPE_BUY_CANCELED = 13`, `DEAL_TYPE_SELL_CANCELED = 14`, unknown codes) and inline (`commission`/`fee`) + separate `DEAL_TYPE_COMMISSION` duplicates fail closed with `AccountStateReconstructionError`.
- **5 PM `America/New_York` DST-Aware Rollover:** Computes UTC trading-day bounds using IANA `ZoneInfo("America/New_York")` across EST (`UTC-5`) and EDT (`UTC-4`), strips consistent `account.credit` from floating PnL and effective cash equity, counts account-wide open positions via `positions_get(None)`, and preserves `peak_balance` high-water marks across rollovers.

### 3.4 14-Step `PreTradeSafetyGate` (`safety.py`)
1. `ExecutionSafetyConfig` DEMO-only mode check.
2. Terminal connection health, API permissions, and DEMO account (`trade_mode == 0`) check.
3. Signal executability (`BUY`/`SELL` only, positive entry/SL/TP prices).
4. `PersistentExecutionLedger` readiness (`RECOVERY_REQUIRED_LOCKED` block, unresolved intent block, duplicate ID block, and opposite-direction `ACTIVE`/`PARTIALLY_FILLED` block).
5. Live broker position (`positions_get`) and working-order (`orders_get`) conflict checks (blocking opposite-direction positions regardless of `magic` and conflicting working orders).
6. `SignalLifecycleManager` state/expiration validation + `MT5SymbolTradeMode` (`DISABLED=0`, `LONGONLY=1`, `SHORTONLY=2`, `CLOSEONLY=3`, `FULL=4`) enforcement + `InstrumentSpec` mapping.
7. Real-time tick freshness check (`max_tick_age_seconds`).
8. Live Bid-Ask spread limit check in `symbol_info.point` units (`max_allowed_spread_points`).
9. Live execution price drift check (`max_entry_deviation_points`) and directional SL/TP geometry at live market price.
10. Broker stop-level (`trade_stops_level`) and freeze-level (`trade_freeze_level`) minimum distance checks.
11. Verified `AccountState` reconstruction and Phase 1 `RiskEngine.evaluate()` at live execution price.
12. Volume bounds, `volume_step` multiple, and `RiskEngine` cap check (synchronizing `RiskDecision` metadata when `requested_volume` overrides `calculated_position_size`).
13. Execution-mode-aware filling mode resolution (`resolve_supported_filling_mode`).
14. Pre-flight MT5 `order_check()` acceptance and post-trade free-margin verification.

### 3.5 Write-Ahead `PersistentExecutionLedger` & `MT5DemoExecutionEngine` (`ledger.py`, `engine.py`)
- **Write-Ahead Lifecycle:** Every order is durably persisted through `CREATED` $\rightarrow$ `VALIDATED` $\rightarrow$ `SUBMITTING` before `backend.order_send()` is invoked, followed by a pre-send TOCTOU re-check of DEMO account status, tick freshness, spread, and price drift.
- **Retcode Classification & Post-Accept Verification:**
  - Only retcodes in `DETERMINISTIC_REJECTION_RETCODES` transition from `SUBMITTING` to `REJECTED` (or `CANCELLED` for `10007`).
  - `10011 (ERROR)`, `10012 (TIMEOUT)`, `10023 (ORDER_CHANGED)`, `10031 (CONNECTION)`, `0`, negative, and unknown retcodes transition `SUBMITTING` $\rightarrow$ `UNKNOWN` $\rightarrow$ `RECONCILIATION_REQUIRED` and never trigger blind retries.
  - `10009 (DONE)`, `10010 (DONE_PARTIAL)`, and `10008 (PLACED)` transition to `SUBMITTED` and run `_verify_broker_execution_after_accept()` against `positions_get()`, `orders_get()`, and `history_deals_get()` using exact token matching (`extract_comment_identifiers`).
- **Broker Reconciliation (`reconcile_with_broker`):** Fails closed on `None` or empty deal history (`len(deals) == 0`), matches entries by exact comment tokens, `magic`, `symbol`, `direction`, and unique tickets, discovers orphan open positions (`ACTIVE`), orphan working orders (`RECONCILIATION_REQUIRED`), and closed orphan execution deals (`FILLED`), and keeps `RECOVERY_REQUIRED_LOCKED` locked while any working order or unresolved entry remains.
