# ATLAS AI — SYSTEM ARCHITECTURE

---

## 1. Architectural Philosophy

ATLAS AI is built around **strict separation of concerns**, **zero lookahead bias**, and **deterministic trading safety**:
- **Knowledge, Statistical/ML & Strategy Engines Identify Opportunities; Risk Engine Decides Permissibility.** Every generated signal starts in `GENERATED` state and can never become `ACTIVE` without passing through the standalone Phase 1 `RiskEngine`.
- **Causal & Purged Evaluation Only:** Every feature, strategy, knowledge rule, and ML inference step at bar $t$ operates strictly on closed historical bars $\le t$, and every supervised ML training split enforces `purge_bars >= horizon_bars`.

---

## 2. High-Level System Data Flow (Phases 0–6)

```mermaid
flowchart TD
    subgraph Knowledge_Layer["Phase 5: Educational Knowledge Engine"]
        DOCS["Transcripts / Markdown / JSON Playbooks"] --> K_REPO["KnowledgeRepository (Only APPROVED Rules)"]
    end

    subgraph Market_Layer["Phase 2: Market Data & Feature Pipeline"]
        CSV["Historical CSV / OHLCV"] --> LOADER["HistoricalCSVLoader & MarketDataValidator"]
        LOADER --> FEAT["FeaturePipeline (TR, ATR, SMA, EMA, Structure, Session)"]
    end

    subgraph ML_Layer["Phase 6: Machine Learning & Statistical Intelligence"]
        FEAT --> ML_FEAT["MLFeatureBuilder (Stationary ATR-Normalized Features)"]
        FEAT --> LABELER["TripleBarrierLabeler (SL-First Horizon H)"]
        ML_FEAT --> WF["WalkForwardValidator (Purged Train/Val/Test + FoldStandardScaler)"]
        LABELER --> WF
        WF --> CLF["RegularizedLogisticClassifier & GradientBoostedStumpEnsemble"]
    end

    subgraph Strategy_Layer["Phase 3, 5 & 6: Multi-Strategy Confluence Engine"]
        K_REPO --> KB_STRAT["KnowledgeBasedStrategy"]
        CLF --> ML_STRAT["MLSignalStrategy"]
        FEAT --> KB_STRAT
        ML_FEAT --> ML_STRAT
        FEAT --> PRIMARY["Phase 3 Strategies: Regime, MTF, EMA Trend, Structure, Liquidity, FVG, Context, NewsFilter"]

        KB_STRAT --> CONF["ConfluenceEngine"]
        ML_STRAT --> CONF
        PRIMARY --> CONF

        CONF --> SIG_GEN["SignalGenerator -> TradeSignal"]
        SIG_GEN --> LIFECYCLE["SignalLifecycleManager (GENERATED -> RISK_APPROVED -> ACTIVE)"]
    end

    subgraph Safety_Layer["Phase 1: Deterministic Risk Engine"]
        LIFECYCLE --> RISK["RiskEngine (10 Safety Gates & Floor Volume Sizing) -> RiskDecision"]
    end

    subgraph Backtest_Layer["Phase 4: Historical Backtesting Engine"]
        RISK --> EXEC_SIM["ExecutionSimulator & PortfolioTracker -> BacktestResult"]
    end
```

---

## 3. Current Folder Structure (Phase 6)

```text
c:\Users\User\OneDrive\Desktop\ATLAS AI\
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
├── PROJECT_CONTEXT.md
├── ARCHITECTURE.md
├── DEVELOPMENT_LOG.md
├── NEXT_STEPS.md
├── CHANGELOG.md
├── src/
│   └── atlas_ai/
│       ├── __init__.py
│       ├── main.py
│       ├── core/
│       ├── domain/
│       ├── risk/
│       ├── data/
│       ├── strategies/
│       ├── backtest/
│       ├── knowledge/
│       └── models/
│           ├── __init__.py
│           ├── exceptions.py
│           ├── config.py
│           ├── schemas.py
│           ├── labeling.py
│           ├── features.py
│           ├── classifiers.py
│           ├── validation.py
│           └── strategy.py
└── tests/
    ├── __init__.py
    ├── fixtures/
    │   └── synthetic_xauusd_h1.csv
    ├── test_foundation.py
    ├── test_domain_and_risk.py
    ├── test_market_data.py
    ├── test_features.py
    ├── test_strategies.py
    ├── test_smc_ict.py
    ├── test_multi_timeframe.py
    ├── test_news_filter.py
    ├── test_confluence.py
    ├── test_signal_generation.py
    ├── test_signal_lifecycle.py
    ├── test_backtest.py
    ├── test_knowledge.py
    └── test_models.py
```
