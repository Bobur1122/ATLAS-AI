"""Top-level StrategyEngine orchestrator for ATLAS AI Phase 3."""

from typing import TYPE_CHECKING, Mapping, Sequence

import pandas as pd

from atlas_ai.data.features import FeaturePipeline
from atlas_ai.data.integrity import get_timeframe_delta
from atlas_ai.domain.market import InstrumentSpec, Timeframe
from atlas_ai.domain.risk import AccountState, RiskDecision, RiskParameters
from atlas_ai.risk.engine import RiskEngine

if TYPE_CHECKING:
    from atlas_ai.knowledge.repository import KnowledgeRepository
    from atlas_ai.knowledge.strategy import KnowledgeBasedStrategy
    from atlas_ai.models.classifiers import BaseMLClassifier
    from atlas_ai.models.strategy import MLSignalStrategy
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
from atlas_ai.strategies.fvg import FVGDetectionStrategy, FVGRetestStrategy
from atlas_ai.strategies.liquidity import LiquiditySweepStrategy
from atlas_ai.strategies.market_structure import MarketStructureStrategy
from atlas_ai.strategies.models import (
    EconomicEvent,
    ManagedSignalRecord,
    MultiTimeframeEvaluation,
    RiskEvaluationStatus,
    StrategyAnalysisReport,
    StrategyEvaluationResult,
)
from atlas_ai.strategies.multi_timeframe import MultiTimeframeAnalyzer
from atlas_ai.strategies.news_filter import NewsFilter
from atlas_ai.strategies.premium_discount import PremiumDiscountAnalyzer
from atlas_ai.strategies.sessions import KillZoneSessionAnalyzer, PreviousDayLevelsAnalyzer
from atlas_ai.strategies.signal_generator import SignalGenerator
from atlas_ai.strategies.signal_lifecycle import SignalLifecycleManager
from atlas_ai.strategies.trend import EmaTrendStrategy, MarketRegimeDetector


class StrategyEngine:
    """Orchestrates the end-to-end Phase 3 trading intelligence pipeline:

    Validated OHLCV / Feature Data -> Market Regime Detection -> MTF Analysis ->
    Market Structure -> Liquidity Analysis -> FVG Analysis & Retest ->
    Premium/Discount -> PDH/PDL -> Kill Zone Analysis -> News Filter ->
    Confluence Engine -> Signal Generator -> Signal Lifecycle -> Risk Engine ->
    StrategyAnalysisReport.
    """

    def __init__(
        self,
        *,
        feature_pipeline: FeaturePipeline | None = None,
        regime_config: RegimeDetectorConfig | None = None,
        ema_config: EmaTrendConfig | None = None,
        structure_config: MarketStructureConfig | None = None,
        liquidity_config: LiquidityConfig | None = None,
        fvg_config: FVGConfig | None = None,
        pd_config: PremiumDiscountConfig | None = None,
        pdh_pdl_config: PreviousDayLevelsConfig | None = None,
        killzone_config: KillZoneSessionConfig | None = None,
        mtf_config: MultiTimeframeConfig | None = None,
        news_config: NewsFilterConfig | None = None,
        confluence_config: ConfluenceWeightsConfig | None = None,
        signal_generator_config: SignalGeneratorConfig | None = None,
        lifecycle_config: SignalLifecycleConfig | None = None,
        knowledge_strategy: "KnowledgeBasedStrategy | None" = None,
        knowledge_repository: "KnowledgeRepository | None" = None,
        ml_strategy: "MLSignalStrategy | None" = None,
        ml_classifier: "BaseMLClassifier | None" = None,
    ) -> None:
        self._feature_pipeline = feature_pipeline or FeaturePipeline()
        self._regime_detector = MarketRegimeDetector(regime_config)
        self._ema_trend = EmaTrendStrategy(ema_config)
        self._market_structure = MarketStructureStrategy(structure_config)
        self._liquidity = LiquiditySweepStrategy(liquidity_config)
        self._fvg_detector = FVGDetectionStrategy(fvg_config)
        self._fvg_retest = FVGRetestStrategy(fvg_config)
        self._pd_analyzer = PremiumDiscountAnalyzer(pd_config)
        self._pdh_pdl_analyzer = PreviousDayLevelsAnalyzer(pdh_pdl_config)
        self._killzone_analyzer = KillZoneSessionAnalyzer(killzone_config)
        self._mtf_analyzer = MultiTimeframeAnalyzer(mtf_config)
        self._news_filter = NewsFilter(news_config)

        if knowledge_strategy is not None:
            self._knowledge_strategy: "KnowledgeBasedStrategy | None" = knowledge_strategy
        elif knowledge_repository is not None:
            from atlas_ai.knowledge.strategy import KnowledgeBasedStrategy

            self._knowledge_strategy = KnowledgeBasedStrategy(knowledge_repository)
        else:
            self._knowledge_strategy = None

        if ml_strategy is not None:
            self._ml_strategy: "MLSignalStrategy | None" = ml_strategy
        elif ml_classifier is not None:
            from atlas_ai.models.strategy import MLSignalStrategy

            self._ml_strategy = MLSignalStrategy(ml_classifier)
        else:
            self._ml_strategy = None

        eff_conf_cfg = confluence_config or ConfluenceWeightsConfig()
        if confluence_config is None:
            updates: dict[str, object] = {}
            if self._knowledge_strategy is not None:
                updates["enable_knowledge_strategy"] = True
            if self._ml_strategy is not None:
                updates["enable_ml_strategy"] = True
            if updates:
                eff_conf_cfg = eff_conf_cfg.model_copy(update=updates)

        self._confluence_engine = ConfluenceEngine(eff_conf_cfg)
        self._signal_generator = SignalGenerator(signal_generator_config)
        self._lifecycle_manager = SignalLifecycleManager(lifecycle_config)

    @property
    def knowledge_strategy(self) -> "KnowledgeBasedStrategy | None":
        """Return the optional Phase 5 KnowledgeBasedStrategy instance."""
        return self._knowledge_strategy

    @property
    def ml_strategy(self) -> "MLSignalStrategy | None":
        """Return the optional Phase 6 MLSignalStrategy instance."""
        return self._ml_strategy

    @property
    def lifecycle_manager(self) -> SignalLifecycleManager:
        """Return the underlying SignalLifecycleManager."""
        return self._lifecycle_manager

    def _ensure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical features via FeaturePipeline if not already present."""
        required = ("tr", "atr_14", "ema_9", "ema_21", "ema_50", "sma_20", "sma_50")
        if all(col in df.columns for col in required):
            return df
        return self._feature_pipeline.transform(df)

    def analyze(
        self,
        df: pd.DataFrame,
        *,
        symbol: str = "XAUUSD",
        timeframe: Timeframe = Timeframe.H1,
        as_of_index: int | None = None,
        mtf_data: Mapping[Timeframe, pd.DataFrame] | None = None,
        economic_events: Sequence[EconomicEvent] | None = (),
        account_state: AccountState | None = None,
        risk_parameters: RiskParameters | None = None,
        instrument_spec: InstrumentSpec | None = None,
        register_in_lifecycle: bool = True,
    ) -> StrategyAnalysisReport:
        """Execute a complete causal analysis at `as_of_index` (defaults to the final row of `df`)."""
        cleaned_symbol = symbol.strip().upper()
        sliced_raw = df if as_of_index is None else df.iloc[: as_of_index + 1].copy()
        feat_df = self._ensure_features(sliced_raw)

        last_row = feat_df.iloc[-1]
        eval_ts = self._regime_detector.extract_timestamp(last_row)
        # Candle close timestamp for MTF synchronization
        eval_close_ts = eval_ts + get_timeframe_delta(timeframe)

        # 1. Market Regime Detection
        regime, regime_eval = self._regime_detector.detect_regime(feat_df)

        # 2. Primary Strategy & SMC/ICT Context Evaluations
        evaluations: dict[str, StrategyEvaluationResult] = {
            self._ema_trend.strategy_id: self._ema_trend.evaluate(feat_df),
            self._market_structure.strategy_id: self._market_structure.evaluate(feat_df),
            self._liquidity.strategy_id: self._liquidity.evaluate(feat_df),
            self._fvg_detector.strategy_id: self._fvg_detector.evaluate(feat_df),
            self._fvg_retest.strategy_id: self._fvg_retest.evaluate(feat_df),
            self._pd_analyzer.strategy_id: self._pd_analyzer.evaluate(feat_df),
            self._pdh_pdl_analyzer.strategy_id: self._pdh_pdl_analyzer.evaluate(feat_df),
            self._killzone_analyzer.strategy_id: self._killzone_analyzer.evaluate(feat_df),
        }

        if self._knowledge_strategy is not None:
            evaluations[self._knowledge_strategy.strategy_id] = (
                self._knowledge_strategy.evaluate_with_context(
                    feat_df,
                    symbol=cleaned_symbol,
                    timeframe=timeframe,
                )
            )

        if self._ml_strategy is not None:
            evaluations[self._ml_strategy.strategy_id] = self._ml_strategy.evaluate(feat_df)

        # 3. Multi-Timeframe Analysis
        mtf_eval: MultiTimeframeEvaluation | None = None
        if mtf_data is not None:
            prepared_mtf = {tf: self._ensure_features(t_df) for tf, t_df in mtf_data.items()}
            mtf_eval = self._mtf_analyzer.analyze_mtf(prepared_mtf, eval_close_ts)
            evaluations[self._mtf_analyzer.strategy_id] = StrategyEvaluationResult(
                strategy_id=self._mtf_analyzer.strategy_id,
                direction=mtf_eval.aligned_direction,
                timestamp=eval_ts,
                confidence=mtf_eval.alignment_score,
                rationale=mtf_eval.rationale,
                supporting_evidence=mtf_eval.model_dump(mode="json"),
            )
        else:
            evaluations[self._mtf_analyzer.strategy_id] = self._mtf_analyzer.evaluate(feat_df)

        # 4. News Filter Evaluation
        news_res = self._news_filter.evaluate(
            evaluation_timestamp=eval_ts,
            events=economic_events,
        )

        # 5. Confluence Engine
        confluence_res = self._confluence_engine.evaluate(
            evaluations=evaluations,
            mtf_evaluation=mtf_eval,
            news_result=news_res,
        )

        # 6. Signal Generation
        trade_signal = self._signal_generator.generate(
            symbol=cleaned_symbol,
            timeframe=timeframe,
            timestamp=eval_ts,
            current_bar=last_row,
            confluence=confluence_res,
        )

        # 7. Signal Lifecycle & Independent Risk Engine Integration
        lifecycle_record: ManagedSignalRecord | None = None
        risk_status = RiskEvaluationStatus.NOT_EVALUATED
        risk_decision: RiskDecision | None = None

        if trade_signal.is_executable:
            if register_in_lifecycle:
                # Avoid throwing duplicate error during repeated stateless scans if already tracked;
                # only register if not already tracked in active state
                existing_dup = self._lifecycle_manager._is_duplicate_active_setup(trade_signal)
                if existing_dup is not None:
                    lifecycle_record = existing_dup
                else:
                    lifecycle_record = self._lifecycle_manager.register_signal(
                        trade_signal,
                        supporting_evidence={
                            "regime": regime.value,
                            "buy_score": str(confluence_res.buy_score),
                            "sell_score": str(confluence_res.sell_score),
                        },
                    )

            # Submit to Phase 1 RiskEngine ONLY if all 3 risk inputs are supplied
            if (
                account_state is not None
                and risk_parameters is not None
                and instrument_spec is not None
            ):
                risk_engine = RiskEngine(
                    risk_parameters=risk_parameters,
                    instrument_specs=instrument_spec,
                )
                risk_decision = risk_engine.evaluate(
                    signal=trade_signal,
                    account=account_state,
                )
                risk_status = (
                    RiskEvaluationStatus.RISK_APPROVED
                    if risk_decision.approved
                    else RiskEvaluationStatus.RISK_REJECTED
                )
                if (
                    lifecycle_record is not None
                    and lifecycle_record.risk_status == RiskEvaluationStatus.NOT_EVALUATED
                ):
                    lifecycle_record = self._lifecycle_manager.apply_risk_decision(
                        lifecycle_record.signal_id,
                        risk_decision,
                        eval_ts,
                    )

        return StrategyAnalysisReport(
            symbol=cleaned_symbol,
            timeframe=timeframe,
            timestamp=eval_ts,
            regime=regime,
            regime_evaluation=regime_eval,
            mtf_evaluation=mtf_eval,
            confluence=confluence_res,
            signal=trade_signal,
            lifecycle_record=lifecycle_record,
            risk_status=risk_status,
            risk_decision=risk_decision,
        )
