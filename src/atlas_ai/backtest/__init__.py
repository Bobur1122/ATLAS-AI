"""Historical Backtesting Engine package for ATLAS AI Phase 4."""

from atlas_ai.backtest.config import (
    BacktestConfig,
    IntrabarTieBreakPolicy,
    default_xauusd_instrument_spec,
)
from atlas_ai.backtest.engine import BacktestEngine
from atlas_ai.backtest.exceptions import (
    BacktestConfigurationError,
    BacktestError,
    BacktestExecutionError,
)
from atlas_ai.backtest.execution import EntryFillEvaluation, ExecutionSimulator
from atlas_ai.backtest.metrics import calculate_backtest_metrics
from atlas_ai.backtest.models import (
    BacktestMetrics,
    BacktestResult,
    BacktestTradeRecord,
    EquityPoint,
    ExitReason,
    RiskRejectionRecord,
    SimulatedPosition,
)
from atlas_ai.backtest.portfolio import PortfolioTracker

__all__ = [
    "BacktestConfig",
    "BacktestConfigurationError",
    "BacktestEngine",
    "BacktestError",
    "BacktestExecutionError",
    "BacktestMetrics",
    "BacktestResult",
    "BacktestTradeRecord",
    "EntryFillEvaluation",
    "EquityPoint",
    "ExecutionSimulator",
    "ExitReason",
    "IntrabarTieBreakPolicy",
    "PortfolioTracker",
    "RiskRejectionRecord",
    "SimulatedPosition",
    "calculate_backtest_metrics",
    "default_xauusd_instrument_spec",
]
