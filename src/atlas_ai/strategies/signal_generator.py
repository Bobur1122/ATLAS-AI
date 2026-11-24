"""Signal Generator translating ConfluenceResult and structural/ATR levels into TradeSignal domain objects."""

from datetime import datetime
from decimal import Decimal, ROUND_DOWN

import pandas as pd

from atlas_ai.domain.market import Timeframe
from atlas_ai.domain.signals import SignalDirection, TradeSignal
from atlas_ai.strategies.config import SignalGeneratorConfig
from atlas_ai.strategies.models import ConfluenceResult


class SignalGenerator:
    """Constructs canonical `TradeSignal` domain objects from `ConfluenceResult` and market structure/ATR levels.

    Rules:
    - Never fabricates trade levels: if `confluence.direction == WAIT`, or if valid positive entry/stop-loss
      cannot be calculated from the current candle close, structural levels, or ATR, returns a `WAIT` `TradeSignal`.
    - For `BUY`: ensures `stop_loss < entry_price < take_profit`.
    - For `SELL`: ensures `take_profit < entry_price < stop_loss`.
    - Prioritizes structural stop-loss candidates from agreeing high-precision strategies (`fvg_retest`,
      `liquidity_sweep`, `smc_market_structure`, `ema_trend_following`), bounded by `[min_stop_atr, max_stop_atr]`.
    """

    def __init__(self, config: SignalGeneratorConfig | None = None) -> None:
        self._config = config or SignalGeneratorConfig()

    @property
    def config(self) -> SignalGeneratorConfig:
        return self._config

    def generate(
        self,
        *,
        symbol: str,
        timeframe: Timeframe,
        timestamp: datetime,
        current_bar: pd.Series,
        confluence: ConfluenceResult,
        strategy_id: str = "atlas_confluence_v1",
    ) -> TradeSignal:
        """Generate a validated `TradeSignal` from `ConfluenceResult` and the current closed candle."""
        cleaned_symbol = symbol.strip().upper()

        if confluence.direction == SignalDirection.WAIT:
            return TradeSignal(
                symbol=cleaned_symbol,
                direction=SignalDirection.WAIT,
                entry_price=None,
                stop_loss=None,
                take_profit=None,
                timeframe=timeframe,
                strategy_id=strategy_id,
                confidence=confluence.final_confidence,
                timestamp=timestamp,
                rationale=confluence.rationale,
            )

        raw_close = current_bar.get("close")
        if raw_close is None or pd.isna(raw_close) or float(raw_close) <= 0:
            return TradeSignal(
                symbol=cleaned_symbol,
                direction=SignalDirection.WAIT,
                timeframe=timeframe,
                strategy_id=strategy_id,
                confidence=Decimal("0"),
                timestamp=timestamp,
                rationale="Cannot generate BUY/SELL signal: current bar close price is missing or non-positive.",
            )

        entry = Decimal(str(raw_close)).quantize(Decimal("0.01"))
        raw_atr = current_bar.get("atr_14")
        atr: Decimal | None = None
        if raw_atr is not None and not pd.isna(raw_atr) and float(raw_atr) > 0:
            atr = Decimal(str(raw_atr)).quantize(Decimal("0.01"))

        # Collect structural stop-loss candidates from agreeing component strategies in priority order
        priority_order = (
            "fvg_retest",
            "liquidity_sweep",
            "smc_market_structure",
            "fvg_detection",
            "ema_trend_following",
            "knowledge_based_strategy",
            "ml_signal_strategy",
        )
        candidate_sl: Decimal | None = None
        for mod_id in priority_order:
            ev = confluence.component_evaluations.get(mod_id)
            if (
                ev is not None
                and ev.direction == confluence.direction
                and ev.stop_loss is not None
                and ev.stop_loss > Decimal("0")
            ):
                if confluence.direction == SignalDirection.BUY and ev.stop_loss < entry:
                    candidate_sl = ev.stop_loss.quantize(Decimal("0.01"))
                    break
                if confluence.direction == SignalDirection.SELL and ev.stop_loss > entry:
                    candidate_sl = ev.stop_loss.quantize(Decimal("0.01"))
                    break

        # Fallback to ATR-based stop-loss if no component provided a valid structural stop-loss
        if candidate_sl is None and atr is not None and atr > Decimal("0"):
            sl_dist = (atr * self._config.atr_sl_multiplier).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
            if sl_dist > Decimal("0"):
                if confluence.direction == SignalDirection.BUY:
                    candidate_sl = (entry - sl_dist).quantize(Decimal("0.01"))
                else:
                    candidate_sl = (entry + sl_dist).quantize(Decimal("0.01"))

        if candidate_sl is None or candidate_sl <= Decimal("0"):
            return TradeSignal(
                symbol=cleaned_symbol,
                direction=SignalDirection.WAIT,
                timeframe=timeframe,
                strategy_id=strategy_id,
                confidence=Decimal("0"),
                timestamp=timestamp,
                rationale="Cannot generate executable signal: neither structural stop-loss nor valid ATR available.",
            )

        # Clamp stop distance against ATR bounds if ATR is available
        stop_dist = abs(entry - candidate_sl)
        if atr is not None and atr > Decimal("0"):
            min_dist = (atr * self._config.min_stop_atr_multiplier).quantize(Decimal("0.01"))
            max_dist = (atr * self._config.max_stop_atr_multiplier).quantize(Decimal("0.01"))
            if stop_dist < min_dist and min_dist > Decimal("0"):
                stop_dist = min_dist
            elif stop_dist > max_dist and max_dist > Decimal("0"):
                stop_dist = max_dist

            if confluence.direction == SignalDirection.BUY:
                candidate_sl = (entry - stop_dist).quantize(Decimal("0.01"))
            else:
                candidate_sl = (entry + stop_dist).quantize(Decimal("0.01"))

        if stop_dist <= Decimal("0") or candidate_sl <= Decimal("0"):
            return TradeSignal(
                symbol=cleaned_symbol,
                direction=SignalDirection.WAIT,
                timeframe=timeframe,
                strategy_id=strategy_id,
                confidence=Decimal("0"),
                timestamp=timestamp,
                rationale="Calculated stop-loss distance is non-positive; returning WAIT.",
            )

        tp_dist = (stop_dist * self._config.default_rr_ratio).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        if confluence.direction == SignalDirection.BUY:
            tp = (entry + tp_dist).quantize(Decimal("0.01"))
        else:
            tp = (entry - tp_dist).quantize(Decimal("0.01"))

        if tp <= Decimal("0"):
            return TradeSignal(
                symbol=cleaned_symbol,
                direction=SignalDirection.WAIT,
                timeframe=timeframe,
                strategy_id=strategy_id,
                confidence=Decimal("0"),
                timestamp=timestamp,
                rationale="Calculated take-profit price is non-positive; returning WAIT.",
            )

        return TradeSignal(
            symbol=cleaned_symbol,
            direction=confluence.direction,
            entry_price=entry,
            stop_loss=candidate_sl,
            take_profit=tp,
            timeframe=timeframe,
            strategy_id=strategy_id,
            confidence=confluence.final_confidence,
            timestamp=timestamp,
            rationale=confluence.rationale,
        )
