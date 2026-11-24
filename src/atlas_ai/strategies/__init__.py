"""Phase 3 Advanced Strategy Engine, SMC/ICT Analyzers, Confluence, and Signal Lifecycle package."""

from atlas_ai.strategies.base import BaseStrategy
from atlas_ai.strategies.config import (
    ConfluenceWeightsConfig,
    EmaTrendConfig,
    FVGConfig,
    KillZoneSessionConfig,
    LiquidityConfig,
    MarketStructureConfig,
    MultiTimeframeConfig,
    NewsFilterConfig,
    PremiumDiscountConfig,
    PreviousDayLevelsConfig,
    RegimeDetectorConfig,
    SignalGeneratorConfig,
    SignalLifecycleConfig,
)
from atlas_ai.strategies.confluence import ConfluenceEngine
from atlas_ai.strategies.engine import StrategyEngine
from atlas_ai.strategies.exceptions import (
    SignalLifecycleError,
    StrategyConfigurationError,
    StrategyError,
    StrategyEvaluationError,
)
from atlas_ai.strategies.fvg import FVGDetectionStrategy, FVGRetestStrategy, scan_historical_fvgs
from atlas_ai.strategies.liquidity import LiquiditySweepStrategy, detect_liquidity_pools
from atlas_ai.strategies.market_structure import MarketStructureStrategy, detect_confirmed_swings
from atlas_ai.strategies.models import (
    ConfluenceResult,
    EconomicEvent,
    FairValueGap,
    LifecycleEvent,
    LiquidityPool,
    ManagedSignalRecord,
    MarketRegime,
    MultiTimeframeEvaluation,
    NewsFilterResult,
    NewsFilterStatus,
    NewsImpactLevel,
    PremiumDiscountState,
    PreviousDayLevelsState,
    PriceZoneClassification,
    RiskEvaluationStatus,
    SignalLifecycleState,
    StrategyAnalysisReport,
    StrategyEvaluationResult,
    StructureEventType,
    SwingPoint,
    SwingType,
    TimeframeAnalysisSummary,
)
from atlas_ai.strategies.multi_timeframe import (
    MultiTimeframeAnalyzer,
    filter_closed_candles_as_of,
)
from atlas_ai.strategies.news_filter import NewsFilter
from atlas_ai.strategies.premium_discount import PremiumDiscountAnalyzer
from atlas_ai.strategies.sessions import (
    KillZoneSessionAnalyzer,
    PreviousDayLevelsAnalyzer,
    compute_trading_day_label,
)
from atlas_ai.strategies.signal_generator import SignalGenerator
from atlas_ai.strategies.signal_lifecycle import SignalLifecycleManager
from atlas_ai.strategies.trend import EmaTrendStrategy, MarketRegimeDetector

__all__ = [
    "BaseStrategy",
    "ConfluenceEngine",
    "ConfluenceResult",
    "ConfluenceWeightsConfig",
    "EconomicEvent",
    "EmaTrendConfig",
    "EmaTrendStrategy",
    "FVGConfig",
    "FVGDetectionStrategy",
    "FVGRetestStrategy",
    "FairValueGap",
    "KillZoneSessionAnalyzer",
    "KillZoneSessionConfig",
    "LifecycleEvent",
    "LiquidityConfig",
    "LiquidityPool",
    "LiquiditySweepStrategy",
    "ManagedSignalRecord",
    "MarketRegime",
    "MarketRegimeDetector",
    "MarketStructureConfig",
    "MarketStructureStrategy",
    "MultiTimeframeAnalyzer",
    "MultiTimeframeConfig",
    "MultiTimeframeEvaluation",
    "NewsFilter",
    "NewsFilterConfig",
    "NewsFilterResult",
    "NewsFilterStatus",
    "NewsImpactLevel",
    "PremiumDiscountAnalyzer",
    "PremiumDiscountConfig",
    "PremiumDiscountState",
    "PreviousDayLevelsAnalyzer",
    "PreviousDayLevelsConfig",
    "PreviousDayLevelsState",
    "PriceZoneClassification",
    "RegimeDetectorConfig",
    "RiskEvaluationStatus",
    "SignalGenerator",
    "SignalGeneratorConfig",
    "SignalLifecycleConfig",
    "SignalLifecycleError",
    "SignalLifecycleManager",
    "SignalLifecycleState",
    "StrategyAnalysisReport",
    "StrategyConfigurationError",
    "StrategyEngine",
    "StrategyError",
    "StrategyEvaluationError",
    "StrategyEvaluationResult",
    "StructureEventType",
    "SwingPoint",
    "SwingType",
    "TimeframeAnalysisSummary",
    "compute_trading_day_label",
    "detect_confirmed_swings",
    "detect_liquidity_pools",
    "filter_closed_candles_as_of",
    "scan_historical_fvgs",
]
