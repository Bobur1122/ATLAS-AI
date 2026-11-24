"""Market Regime Detection and EMA Trend Following Strategy for ATLAS AI."""

from decimal import Decimal, ROUND_DOWN

import pandas as pd

from atlas_ai.domain.signals import SignalDirection
from atlas_ai.strategies.base import BaseStrategy
from atlas_ai.strategies.config import EmaTrendConfig, RegimeDetectorConfig
from atlas_ai.strategies.models import MarketRegime, StrategyEvaluationResult


class MarketRegimeDetector(BaseStrategy):
    """Classifies market regime into UPTREND, DOWNTREND, RANGING, HIGH_VOLATILITY, LOW_VOLATILITY, or UNKNOWN.

    Uses EMA 9/21/50, SMA 20/50/100/200 (when available), ATR, ATR percentage, rolling highs/lows,
    and candle range without any lookahead.
    """

    def __init__(self, config: RegimeDetectorConfig | None = None) -> None:
        self._config = config or RegimeDetectorConfig()

    @property
    def strategy_id(self) -> str:
        return "market_regime_detector"

    @property
    def name(self) -> str:
        return "Market Regime Detector"

    @property
    def required_features(self) -> tuple[str, ...]:
        return (
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "ema_9",
            "ema_21",
            "ema_50",
            "sma_20",
            "sma_50",
            "atr_14",
            "atr_pct_14",
            "rolling_high_20",
            "rolling_low_20",
            "candle_range",
        )

    @property
    def is_contextual_filter(self) -> bool:
        return True

    def detect_regime(
        self,
        df: pd.DataFrame,
        as_of_index: int | None = None,
    ) -> tuple[MarketRegime, StrategyEvaluationResult]:
        """Classify the regime at `as_of_index` and return `(regime, StrategyEvaluationResult)`."""
        sliced = self.prepare_causal_slice(df, as_of_index)
        last_row = sliced.iloc[-1]
        ts = self.extract_timestamp(last_row)

        nan_cols = self.has_warmup_nan(
            last_row,
            ("ema_9", "ema_21", "ema_50", "sma_20", "sma_50", "atr_14", "atr_pct_14", "rolling_high_20", "rolling_low_20"),
        )
        if nan_cols:
            res = self.build_wait_result(
                timestamp=ts,
                rationale=f"Market regime is UNKNOWN due to indicator warm-up NaNs in {nan_cols}.",
                rejection_reasons=["INSUFFICIENT_WARMUP_DATA"],
                supporting_evidence={"regime": MarketRegime.UNKNOWN.value, "nan_columns": nan_cols},
            )
            return MarketRegime.UNKNOWN, res

        close_v = Decimal(str(last_row["close"]))
        ema9 = Decimal(str(last_row["ema_9"]))
        ema21 = Decimal(str(last_row["ema_21"]))
        ema50 = Decimal(str(last_row["ema_50"]))
        sma20 = Decimal(str(last_row["sma_20"]))
        sma50 = Decimal(str(last_row["sma_50"]))
        atr = Decimal(str(last_row["atr_14"]))
        atr_pct = Decimal(str(last_row["atr_pct_14"]))
        r_high = Decimal(str(last_row["rolling_high_20"]))
        r_low = Decimal(str(last_row["rolling_low_20"]))
        c_range = Decimal(str(last_row["candle_range"]))

        if atr <= Decimal("0"):
            res = self.build_wait_result(
                timestamp=ts,
                rationale="Market regime is UNKNOWN because ATR_14 is non-positive.",
                rejection_reasons=["NON_POSITIVE_ATR"],
                supporting_evidence={"regime": MarketRegime.UNKNOWN.value},
            )
            return MarketRegime.UNKNOWN, res

        ema_sep_ratio = abs(ema9 - ema50) / atr
        evidence: dict[str, object] = {
            "close": str(close_v),
            "ema_9": str(ema9.quantize(Decimal("0.01"))),
            "ema_21": str(ema21.quantize(Decimal("0.01"))),
            "ema_50": str(ema50.quantize(Decimal("0.01"))),
            "sma_20": str(sma20.quantize(Decimal("0.01"))),
            "sma_50": str(sma50.quantize(Decimal("0.01"))),
            "atr_14": str(atr.quantize(Decimal("0.01"))),
            "atr_pct_14": str(atr_pct.quantize(Decimal("0.0001"))),
            "ema_sep_atr_ratio": str(ema_sep_ratio.quantize(Decimal("0.0001"))),
            "rolling_high_20": str(r_high.quantize(Decimal("0.01"))),
            "rolling_low_20": str(r_low.quantize(Decimal("0.01"))),
            "candle_range": str(c_range.quantize(Decimal("0.01"))),
        }

        # Check optional longer SMAs if present and warmed up
        for opt_sma in ("sma_100", "sma_200"):
            if opt_sma in last_row and not pd.isna(last_row[opt_sma]):
                evidence[opt_sma] = str(Decimal(str(last_row[opt_sma])).quantize(Decimal("0.01")))

        # 1. Volatility extremes check
        if atr_pct >= self._config.high_volatility_atr_pct:
            evidence["regime"] = MarketRegime.HIGH_VOLATILITY.value
            res = self.build_wait_result(
                timestamp=ts,
                rationale=(
                    f"Regime HIGH_VOLATILITY: atr_pct_14 ({atr_pct:.3f}%) >= "
                    f"high_volatility_atr_pct ({self._config.high_volatility_atr_pct}%)."
                ),
                rejection_reasons=["HIGH_VOLATILITY_REGIME"],
                supporting_evidence=evidence,
            )
            return MarketRegime.HIGH_VOLATILITY, res

        if atr_pct <= self._config.low_volatility_atr_pct:
            evidence["regime"] = MarketRegime.LOW_VOLATILITY.value
            res = self.build_wait_result(
                timestamp=ts,
                rationale=(
                    f"Regime LOW_VOLATILITY: atr_pct_14 ({atr_pct:.3f}%) <= "
                    f"low_volatility_atr_pct ({self._config.low_volatility_atr_pct}%)."
                ),
                rejection_reasons=["LOW_VOLATILITY_REGIME"],
                supporting_evidence=evidence,
            )
            return MarketRegime.LOW_VOLATILITY, res

        # 2. Directional trend vs ranging check
        if (
            ema9 > ema21 > ema50
            and close_v > ema21
            and ema_sep_ratio >= self._config.min_ema_separation_atr_ratio
        ):
            evidence["regime"] = MarketRegime.UPTREND.value
            res = StrategyEvaluationResult(
                strategy_id=self.strategy_id,
                direction=SignalDirection.BUY,
                timestamp=ts,
                confidence=Decimal("0.80"),
                rationale=(
                    f"Regime UPTREND: EMA9 ({ema9:.2f}) > EMA21 ({ema21:.2f}) > EMA50 ({ema50:.2f}) "
                    f"with close ({close_v:.2f}) > EMA21."
                ),
                supporting_evidence=evidence,
                is_contextual_filter=True,
            )
            return MarketRegime.UPTREND, res

        if (
            ema9 < ema21 < ema50
            and close_v < ema21
            and ema_sep_ratio >= self._config.min_ema_separation_atr_ratio
        ):
            evidence["regime"] = MarketRegime.DOWNTREND.value
            res = StrategyEvaluationResult(
                strategy_id=self.strategy_id,
                direction=SignalDirection.SELL,
                timestamp=ts,
                confidence=Decimal("0.80"),
                rationale=(
                    f"Regime DOWNTREND: EMA9 ({ema9:.2f}) < EMA21 ({ema21:.2f}) < EMA50 ({ema50:.2f}) "
                    f"with close ({close_v:.2f}) < EMA21."
                ),
                supporting_evidence=evidence,
                is_contextual_filter=True,
            )
            return MarketRegime.DOWNTREND, res

        evidence["regime"] = MarketRegime.RANGING.value
        res = self.build_wait_result(
            timestamp=ts,
            rationale="Regime RANGING: Moving averages are compressed or intertwined without directional alignment.",
            rejection_reasons=["RANGING_REGIME"],
            supporting_evidence=evidence,
        )
        return MarketRegime.RANGING, res

    def evaluate(
        self,
        df: pd.DataFrame,
        as_of_index: int | None = None,
    ) -> StrategyEvaluationResult:
        """Standard BaseStrategy evaluation wrapper returning StrategyEvaluationResult."""
        _, res = self.detect_regime(df, as_of_index=as_of_index)
        return res


class EmaTrendStrategy(BaseStrategy):
    """EMA 9 / 21 / 50 Trend Following Strategy.

    Rules:
    - BUY: EMA 9 > EMA 21 > EMA 50 and price confirms bullish direction (`close > ema_9` and bullish candle).
    - SELL: EMA 9 < EMA 21 < EMA 50 and price confirms bearish direction (`close < ema_9` and bearish candle).
    - Otherwise: WAIT.
    """

    def __init__(self, config: EmaTrendConfig | None = None) -> None:
        self._config = config or EmaTrendConfig()

    @property
    def strategy_id(self) -> str:
        return "ema_trend_following"

    @property
    def name(self) -> str:
        return "EMA 9/21/50 Trend Following Strategy"

    @property
    def required_features(self) -> tuple[str, ...]:
        return (
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            self._config.fast_ema_col,
            self._config.mid_ema_col,
            self._config.slow_ema_col,
            self._config.atr_col,
        )

    def evaluate(
        self,
        df: pd.DataFrame,
        as_of_index: int | None = None,
    ) -> StrategyEvaluationResult:
        sliced = self.prepare_causal_slice(df, as_of_index)
        row = sliced.iloc[-1]
        ts = self.extract_timestamp(row)

        req_cols = (
            self._config.fast_ema_col,
            self._config.mid_ema_col,
            self._config.slow_ema_col,
            self._config.atr_col,
        )
        nan_cols = self.has_warmup_nan(row, req_cols)
        if nan_cols:
            return self.build_wait_result(
                timestamp=ts,
                rationale=f"EMA Trend strategy waiting for indicator warm-up ({nan_cols}).",
                rejection_reasons=["WARMUP_NAN"],
                supporting_evidence={"nan_columns": nan_cols},
            )

        open_p = Decimal(str(row["open"]))
        close_p = Decimal(str(row["close"]))
        fast_ema = Decimal(str(row[self._config.fast_ema_col]))
        mid_ema = Decimal(str(row[self._config.mid_ema_col]))
        slow_ema = Decimal(str(row[self._config.slow_ema_col]))
        atr = Decimal(str(row[self._config.atr_col]))

        if atr <= Decimal("0"):
            return self.build_wait_result(
                timestamp=ts,
                rationale="ATR is non-positive; cannot compute ATR stop distance.",
                rejection_reasons=["INVALID_ATR"],
            )

        evidence: dict[str, object] = {
            "open": str(open_p),
            "close": str(close_p),
            "ema_9": str(fast_ema.quantize(Decimal("0.01"))),
            "ema_21": str(mid_ema.quantize(Decimal("0.01"))),
            "ema_50": str(slow_ema.quantize(Decimal("0.01"))),
            "atr_14": str(atr.quantize(Decimal("0.01"))),
        }

        bullish_candle = close_p > open_p if self._config.require_candle_direction_confirmation else True
        bearish_candle = close_p < open_p if self._config.require_candle_direction_confirmation else True

        sl_dist = (atr * self._config.sl_atr_multiplier).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        if sl_dist <= Decimal("0"):
            sl_dist = Decimal("0.50")
        tp_dist = (sl_dist * self._config.rr_ratio).quantize(Decimal("0.01"), rounding=ROUND_DOWN)

        # Check Bullish alignment: EMA 9 > EMA 21 > EMA 50 and price confirmation
        if fast_ema > mid_ema > slow_ema:
            if close_p > fast_ema and bullish_candle:
                entry = close_p.quantize(Decimal("0.01"))
                sl = (entry - sl_dist).quantize(Decimal("0.01"))
                tp = (entry + tp_dist).quantize(Decimal("0.01"))
                return StrategyEvaluationResult(
                    strategy_id=self.strategy_id,
                    direction=SignalDirection.BUY,
                    timestamp=ts,
                    entry_price=entry,
                    stop_loss=sl,
                    take_profit=tp,
                    confidence=Decimal("0.75"),
                    rationale=(
                        f"Bullish EMA alignment (EMA9={fast_ema:.2f} > EMA21={mid_ema:.2f} > EMA50={slow_ema:.2f}) "
                        f"confirmed by bullish close ({close_p:.2f}) above EMA9."
                    ),
                    supporting_evidence=evidence,
                )
            return self.build_wait_result(
                timestamp=ts,
                rationale="Bullish EMA stack present, but price close did not confirm above EMA9 with bullish candle.",
                rejection_reasons=["PRICE_CONFIRMATION_MISSING"],
                supporting_evidence=evidence,
            )

        # Check Bearish alignment: EMA 9 < EMA 21 < EMA 50 and price confirmation
        if fast_ema < mid_ema < slow_ema:
            if close_p < fast_ema and bearish_candle:
                entry = close_p.quantize(Decimal("0.01"))
                sl = (entry + sl_dist).quantize(Decimal("0.01"))
                tp = (entry - tp_dist).quantize(Decimal("0.01"))
                return StrategyEvaluationResult(
                    strategy_id=self.strategy_id,
                    direction=SignalDirection.SELL,
                    timestamp=ts,
                    entry_price=entry,
                    stop_loss=sl,
                    take_profit=tp,
                    confidence=Decimal("0.75"),
                    rationale=(
                        f"Bearish EMA alignment (EMA9={fast_ema:.2f} < EMA21={mid_ema:.2f} < EMA50={slow_ema:.2f}) "
                        f"confirmed by bearish close ({close_p:.2f}) below EMA9."
                    ),
                    supporting_evidence=evidence,
                )
            return self.build_wait_result(
                timestamp=ts,
                rationale="Bearish EMA stack present, but price close did not confirm below EMA9 with bearish candle.",
                rejection_reasons=["PRICE_CONFIRMATION_MISSING"],
                supporting_evidence=evidence,
            )

        return self.build_wait_result(
            timestamp=ts,
            rationale="EMAs (9, 21, 50) are not aligned in a clean bullish or bearish stack.",
            rejection_reasons=["EMA_STACK_NOT_ALIGNED"],
            supporting_evidence=evidence,
        )
