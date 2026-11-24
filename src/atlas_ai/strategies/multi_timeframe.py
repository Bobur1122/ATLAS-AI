"""Multi-Timeframe (MTF) Analyzer with strict closed-candle synchronization for ATLAS AI."""

from datetime import datetime
from decimal import Decimal
from typing import Mapping

import pandas as pd

from atlas_ai.data.integrity import get_timeframe_delta
from atlas_ai.domain.market import Timeframe
from atlas_ai.domain.signals import SignalDirection
from atlas_ai.strategies.base import BaseStrategy
from atlas_ai.strategies.config import MultiTimeframeConfig
from atlas_ai.strategies.fvg import FVGRetestStrategy
from atlas_ai.strategies.liquidity import LiquiditySweepStrategy
from atlas_ai.strategies.market_structure import MarketStructureStrategy
from atlas_ai.strategies.models import (
    MarketRegime,
    MultiTimeframeEvaluation,
    PriceZoneClassification,
    StrategyEvaluationResult,
    StructureEventType,
    TimeframeAnalysisSummary,
)
from atlas_ai.strategies.premium_discount import PremiumDiscountAnalyzer
from atlas_ai.strategies.sessions import PreviousDayLevelsAnalyzer
from atlas_ai.strategies.trend import EmaTrendStrategy, MarketRegimeDetector


def filter_closed_candles_as_of(
    df: pd.DataFrame,
    timeframe: Timeframe,
    evaluation_timestamp: datetime,
) -> pd.DataFrame:
    """Return only candles in `df` that have FULLY CLOSED on or before `evaluation_timestamp`.

    CRITICAL ANTI-LOOKAHEAD RULE:
    In OHLCV datasets where `timestamp` represents the candle OPEN time:
    - An H1 candle with open `timestamp = 10:00` closes at `11:00` (`open_time + timedelta(hours=1)`).
    - An M15 candle with open `timestamp = 10:15` closes at `10:30`.
    If we are evaluating an M15 setup at `10:30` (when the `10:15` M15 candle closes), the `10:00` H1
    candle has NOT closed yet (it closes at `11:00`). Using the `10:00` H1 candle at `10:30` would leak
    30 minutes of future price action (`10:30` to `11:00`)!
    Therefore, a candle is closed at `evaluation_timestamp` if and only if:
        `candle.open_timestamp + get_timeframe_delta(timeframe) <= evaluation_timestamp`.
    """
    if df.empty:
        return df.copy()

    delta = get_timeframe_delta(timeframe)
    eval_ts = pd.Timestamp(evaluation_timestamp)
    if eval_ts.tzinfo is None:
        eval_ts = eval_ts.tz_localize("UTC")

    open_times = pd.to_datetime(df["timestamp"])
    if open_times.dt.tz is None:
        open_times = open_times.dt.tz_localize("UTC")
    else:
        open_times = open_times.dt.tz_convert(eval_ts.tzinfo)

    close_times = open_times + pd.Timedelta(delta)
    mask = close_times <= eval_ts
    return df.loc[mask].copy().reset_index(drop=True)


class MultiTimeframeAnalyzer(BaseStrategy):
    """Synchronizes and evaluates Higher (H1), Middle (M15), Lower (M5), and optional Precision (M1) timeframes.

    Decision Rules:
    1. Higher timeframe (H1) provides directional context (trend, regime, swing structure, Premium/Discount, PDH/PDL).
    2. Middle timeframe (M15) identifies the setup (BOS/CHoCH, liquidity sweep, FVG).
    3. Lower timeframe (M5) confirms entry (FVG retest, candle rejection, local structure).
    4. Optional M1 precision confirmation (not required by default).
    5. Only FULLY CLOSED candles (`open_ts + tf_delta <= eval_close_ts`) are used on every timeframe.
    6. Missing timeframe data is explicitly reported in `missing_timeframes` and never fabricated.
    7. Strong directional disagreement between H1 and M15/M5 sets `has_conflict=True` and returns `WAIT`.
    """

    def __init__(self, config: MultiTimeframeConfig | None = None) -> None:
        self._config = config or MultiTimeframeConfig()
        self._regime_detector = MarketRegimeDetector()
        self._ema_trend = EmaTrendStrategy()
        self._structure = MarketStructureStrategy()
        self._liquidity = LiquiditySweepStrategy()
        self._fvg_retest = FVGRetestStrategy()
        self._pd_analyzer = PremiumDiscountAnalyzer()
        self._pdh_pdl = PreviousDayLevelsAnalyzer()

    @property
    def strategy_id(self) -> str:
        return "multi_timeframe_analyzer"

    @property
    def name(self) -> str:
        return "Multi-Timeframe (H1 / M15 / M5 / M1) Synchronization & Alignment Analyzer"

    @property
    def required_features(self) -> tuple[str, ...]:
        return ("timestamp", "open", "high", "low", "close")

    def _summarize_single_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: Timeframe,
    ) -> TimeframeAnalysisSummary:
        """Evaluate a closed-candle DataFrame for a single timeframe."""
        last_row = df.iloc[-1]
        last_ts = self.extract_timestamp(last_row)

        # Regime (if all required feature columns exist)
        regime = MarketRegime.UNKNOWN
        if all(c in df.columns for c in self._regime_detector.required_features):
            regime, _ = self._regime_detector.detect_regime(df)

        # Trend
        trend_dir = SignalDirection.WAIT
        if all(c in df.columns for c in self._ema_trend.required_features):
            trend_res = self._ema_trend.evaluate(df)
            trend_dir = trend_res.direction

        # Structure
        struct_res = self._structure.evaluate(df)
        struct_event = StructureEventType(
            str(struct_res.supporting_evidence.get("structure_event", StructureEventType.NONE.value))
        )

        # Liquidity & FVG retest
        liq_res = self._liquidity.evaluate(df)
        fvg_res = self._fvg_retest.evaluate(df)

        # Premium/Discount & PDH/PDL
        pd_state, _ = self._pd_analyzer.analyze_state(df)
        _, pdh_res = self._pdh_pdl.analyze_levels(df)

        # Determine combined bias for this timeframe
        votes: list[SignalDirection] = []
        for d in (trend_dir, struct_res.direction, liq_res.direction, fvg_res.direction, pdh_res.direction):
            if d in (SignalDirection.BUY, SignalDirection.SELL):
                votes.append(d)

        if not votes and regime == MarketRegime.UPTREND:
            votes.append(SignalDirection.BUY)
        elif not votes and regime == MarketRegime.DOWNTREND:
            votes.append(SignalDirection.SELL)

        buy_v = sum(1 for v in votes if v == SignalDirection.BUY)
        sell_v = sum(1 for v in votes if v == SignalDirection.SELL)

        if buy_v > sell_v:
            bias = SignalDirection.BUY
            conf = Decimal("0.75")
        elif sell_v > buy_v:
            bias = SignalDirection.SELL
            conf = Decimal("0.75")
        else:
            bias = SignalDirection.WAIT
            conf = Decimal("0.00")

        rationale = (
            f"{timeframe.value} closed@{last_ts.isoformat()}: bias={bias.value}, "
            f"regime={regime.value}, structure={struct_event.value}, zone={pd_state.zone.value}."
        )
        return TimeframeAnalysisSummary(
            timeframe=timeframe,
            last_closed_timestamp=last_ts,
            direction_bias=bias,
            regime=regime,
            structure_event=struct_event,
            pd_zone=pd_state.zone,
            has_fvg_retest=fvg_res.direction != SignalDirection.WAIT,
            has_liquidity_sweep=liq_res.direction != SignalDirection.WAIT,
            confidence=conf,
            rationale=rationale,
        )

    def analyze_mtf(
        self,
        timeframe_data: Mapping[Timeframe, pd.DataFrame],
        evaluation_close_timestamp: datetime,
    ) -> MultiTimeframeEvaluation:
        """Synchronize closed candles across timeframes as of `evaluation_close_timestamp` and evaluate alignment."""
        cfg = self._config
        required_tfs = [cfg.htf_timeframe, cfg.mtf_timeframe, cfg.ltf_timeframe]
        if cfg.require_precision_timeframe and cfg.precision_timeframe is not None:
            required_tfs.append(cfg.precision_timeframe)

        missing_tfs: list[Timeframe] = []
        summaries: dict[Timeframe, TimeframeAnalysisSummary] = {}

        all_candidate_tfs = [cfg.htf_timeframe, cfg.mtf_timeframe, cfg.ltf_timeframe]
        if cfg.precision_timeframe is not None and cfg.precision_timeframe not in all_candidate_tfs:
            all_candidate_tfs.append(cfg.precision_timeframe)

        for tf in all_candidate_tfs:
            raw_df = timeframe_data.get(tf)
            if raw_df is None or raw_df.empty:
                if tf in required_tfs:
                    missing_tfs.append(tf)
                continue

            closed_df = filter_closed_candles_as_of(raw_df, tf, evaluation_close_timestamp)
            if closed_df.empty:
                if tf in required_tfs:
                    missing_tfs.append(tf)
                continue

            summaries[tf] = self._summarize_single_timeframe(closed_df, tf)

        if missing_tfs:
            return MultiTimeframeEvaluation(
                aligned_direction=SignalDirection.WAIT,
                alignment_score=Decimal("0"),
                has_conflict=False,
                missing_timeframes=tuple(missing_tfs),
                summaries=summaries,
                rationale=(
                    f"Missing closed-candle data for required timeframe(s): "
                    f"{[t.value for t in missing_tfs]} as of {evaluation_close_timestamp.isoformat()}."
                ),
            )

        htf_sum = summaries[cfg.htf_timeframe]
        mtf_sum = summaries[cfg.mtf_timeframe]
        ltf_sum = summaries[cfg.ltf_timeframe]

        active_dirs = [
            s.direction_bias
            for s in (htf_sum, mtf_sum, ltf_sum)
            if s.direction_bias in (SignalDirection.BUY, SignalDirection.SELL)
        ]

        # Check strong directional conflict between HTF and MTF/LTF
        if (
            htf_sum.direction_bias in (SignalDirection.BUY, SignalDirection.SELL)
            and mtf_sum.direction_bias in (SignalDirection.BUY, SignalDirection.SELL)
            and htf_sum.direction_bias != mtf_sum.direction_bias
        ) or (
            htf_sum.direction_bias in (SignalDirection.BUY, SignalDirection.SELL)
            and ltf_sum.direction_bias in (SignalDirection.BUY, SignalDirection.SELL)
            and htf_sum.direction_bias != ltf_sum.direction_bias
        ):
            return MultiTimeframeEvaluation(
                aligned_direction=SignalDirection.WAIT,
                alignment_score=Decimal("0"),
                has_conflict=True,
                missing_timeframes=(),
                summaries=summaries,
                rationale=(
                    f"Strong MTF directional conflict: {cfg.htf_timeframe.value}={htf_sum.direction_bias.value}, "
                    f"{cfg.mtf_timeframe.value}={mtf_sum.direction_bias.value}, "
                    f"{cfg.ltf_timeframe.value}={ltf_sum.direction_bias.value}. Returning WAIT."
                ),
            )

        if not active_dirs or len(set(active_dirs)) > 1:
            return MultiTimeframeEvaluation(
                aligned_direction=SignalDirection.WAIT,
                alignment_score=Decimal("0"),
                has_conflict=len(set(active_dirs)) > 1,
                missing_timeframes=(),
                summaries=summaries,
                rationale="Timeframes lack unanimous directional bias; returning WAIT.",
            )

        target_dir = active_dirs[0]
        # Score based on how many of HTF, MTF, LTF actively confirm `target_dir`
        score = Decimal("0")
        if htf_sum.direction_bias == target_dir:
            score += Decimal("0.40")
        if mtf_sum.direction_bias == target_dir:
            score += Decimal("0.35")
        if ltf_sum.direction_bias == target_dir:
            score += Decimal("0.25")

        return MultiTimeframeEvaluation(
            aligned_direction=target_dir,
            alignment_score=min(Decimal("1.00"), score),
            has_conflict=False,
            missing_timeframes=(),
            summaries=summaries,
            rationale=(
                f"MTF aligned {target_dir.value} (score={score:.2f}): "
                f"{cfg.htf_timeframe.value}={htf_sum.direction_bias.value}, "
                f"{cfg.mtf_timeframe.value}={mtf_sum.direction_bias.value}, "
                f"{cfg.ltf_timeframe.value}={ltf_sum.direction_bias.value}."
            ),
        )

    def evaluate(
        self,
        df: pd.DataFrame,
        as_of_index: int | None = None,
    ) -> StrategyEvaluationResult:
        """Single-DataFrame fallback evaluation when only the primary timeframe DataFrame is passed."""
        sliced = self.prepare_causal_slice(df, as_of_index)
        last_row = sliced.iloc[-1]
        ts = self.extract_timestamp(last_row)
        tf_str = str(last_row.get("timeframe", Timeframe.H1.value))
        tf = Timeframe(tf_str) if tf_str in Timeframe._value2member_map_ else Timeframe.H1
        summary = self._summarize_single_timeframe(sliced, tf)
        return StrategyEvaluationResult(
            strategy_id=self.strategy_id,
            direction=summary.direction_bias,
            timestamp=ts,
            confidence=summary.confidence,
            rationale=summary.rationale,
            supporting_evidence=summary.model_dump(mode="json"),
        )
