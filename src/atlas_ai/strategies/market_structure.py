"""Causal SMC/ICT Market Structure Analyzer & Strategy (Swings, HH/HL/LH/LL, BOS, CHoCH)."""

from decimal import Decimal, ROUND_DOWN

import numpy as np
import pandas as pd

from atlas_ai.domain.signals import SignalDirection
from atlas_ai.strategies.base import BaseStrategy
from atlas_ai.strategies.config import MarketStructureConfig
from atlas_ai.strategies.models import (
    StrategyEvaluationResult,
    StructureEventType,
    SwingPoint,
    SwingType,
)


def detect_confirmed_swings(
    df: pd.DataFrame,
    left_bars: int = 2,
    right_bars: int = 2,
    equal_tolerance_atr: Decimal = Decimal("0.05"),
) -> tuple[list[SwingPoint], list[SwingPoint]]:
    """Detect causally confirmed swing highs and swing lows without lookahead bias.

    A bar at index `p` can ONLY be confirmed as a swing pivot once `p + right_bars < len(df)`.
    Therefore, for the last `right_bars` candles of `df`, no swing can be confirmed yet,
    preventing future candles from ever altering previously confirmed swings.

    Returns:
        Tuple of `(confirmed_swing_highs, confirmed_swing_lows)` in chronological order.
    """
    n = len(df)
    if n < left_bars + right_bars + 1:
        return [], []

    highs = df["high"].to_numpy(dtype="float64")
    lows = df["low"].to_numpy(dtype="float64")
    timestamps = [
        ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else pd.Timestamp(ts).to_pydatetime()
        for ts in df["timestamp"]
    ]
    atr_arr = (
        df["atr_14"].to_numpy(dtype="float64")
        if "atr_14" in df.columns
        else np.full(n, 1.0, dtype="float64")
    )

    swing_highs: list[SwingPoint] = []
    swing_lows: list[SwingPoint] = []

    max_pivot_idx = n - 1 - right_bars
    for p in range(left_bars, max_pivot_idx + 1):
        confirm_idx = p + right_bars
        atr_val = atr_arr[p]
        tol = float(equal_tolerance_atr) * (atr_val if not np.isnan(atr_val) and atr_val > 0 else 1.0)

        # Check Swing High at pivot index `p`:
        # Strictly higher than or equal to left window, and strictly higher than right window
        left_h = highs[p - left_bars : p]
        right_h = highs[p + 1 : p + right_bars + 1]
        if highs[p] >= np.max(left_h) and highs[p] > np.max(right_h):
            price_dec = Decimal(str(round(float(highs[p]), 4))).quantize(Decimal("0.01"))
            if not swing_highs:
                s_type = SwingType.HIGH
            else:
                prev_price = swing_highs[-1].price
                if abs(float(price_dec - prev_price)) <= tol:
                    s_type = SwingType.EH
                elif price_dec > prev_price:
                    s_type = SwingType.HH
                else:
                    s_type = SwingType.LH

            swing_highs.append(
                SwingPoint(
                    index=p,
                    confirmed_at_index=confirm_idx,
                    timestamp=timestamps[p],
                    confirmed_at_timestamp=timestamps[confirm_idx],
                    price=price_dec,
                    is_high=True,
                    swing_type=s_type,
                )
            )

        # Check Swing Low at pivot index `p`:
        left_l = lows[p - left_bars : p]
        right_l = lows[p + 1 : p + right_bars + 1]
        if lows[p] <= np.min(left_l) and lows[p] < np.min(right_l):
            price_dec = Decimal(str(round(float(lows[p]), 4))).quantize(Decimal("0.01"))
            if not swing_lows:
                s_type = SwingType.LOW
            else:
                prev_price = swing_lows[-1].price
                if abs(float(price_dec - prev_price)) <= tol:
                    s_type = SwingType.EL
                elif price_dec > prev_price:
                    s_type = SwingType.HL
                else:
                    s_type = SwingType.LL

            swing_lows.append(
                SwingPoint(
                    index=p,
                    confirmed_at_index=confirm_idx,
                    timestamp=timestamps[p],
                    confirmed_at_timestamp=timestamps[confirm_idx],
                    price=price_dec,
                    is_high=False,
                    swing_type=s_type,
                )
            )

    return swing_highs, swing_lows


class MarketStructureStrategy(BaseStrategy):
    """SMC/ICT Market Structure Analyzer & Strategy.

    Detects:
    - Confirmed Swing Highs (HH, LH, EH) and Swing Lows (HL, LL, EL)
    - Bullish BOS (Break of Structure in existing bullish trend)
    - Bearish BOS (Break of Structure in existing bearish trend)
    - Bullish CHoCH (Change of Character breaking a prior Lower High in a bearish trend)
    - Bearish CHoCH (Change of Character breaking a prior Higher Low in a bullish trend)
    """

    def __init__(self, config: MarketStructureConfig | None = None) -> None:
        self._config = config or MarketStructureConfig()

    @property
    def strategy_id(self) -> str:
        return "smc_market_structure"

    @property
    def name(self) -> str:
        return "SMC/ICT Market Structure (BOS / CHoCH)"

    @property
    def required_features(self) -> tuple[str, ...]:
        return ("timestamp", "open", "high", "low", "close")

    def evaluate(
        self,
        df: pd.DataFrame,
        as_of_index: int | None = None,
    ) -> StrategyEvaluationResult:
        sliced = self.prepare_causal_slice(df, as_of_index)
        last_row = sliced.iloc[-1]
        ts = self.extract_timestamp(last_row)

        swing_highs, swing_lows = detect_confirmed_swings(
            sliced,
            left_bars=self._config.left_bars,
            right_bars=self._config.right_bars,
            equal_tolerance_atr=self._config.equal_swing_tolerance_atr,
        )

        if not swing_highs or not swing_lows:
            return self.build_wait_result(
                timestamp=ts,
                rationale="Insufficient confirmed swing highs and lows to evaluate market structure.",
                rejection_reasons=["INSUFFICIENT_CONFIRMED_SWINGS"],
                supporting_evidence={
                    "confirmed_swing_highs_count": len(swing_highs),
                    "confirmed_swing_lows_count": len(swing_lows),
                    "structure_event": StructureEventType.NONE.value,
                },
            )

        last_sh = swing_highs[-1]
        last_sl = swing_lows[-1]
        close_p = Decimal(str(last_row["close"])).quantize(Decimal("0.01"))
        prev_close = (
            Decimal(str(sliced.iloc[-2]["close"])).quantize(Decimal("0.01"))
            if len(sliced) >= 2
            else close_p
        )

        raw_atr = last_row.get("atr_14")
        if raw_atr is not None and not pd.isna(raw_atr) and float(raw_atr) > 0:
            atr = Decimal(str(raw_atr)).quantize(Decimal("0.01"))
        else:
            atr = max(Decimal("1.00"), abs(last_sh.price - last_sl.price) * Decimal("0.10")).quantize(
                Decimal("0.01")
            )

        # Determine prior structural bias before the current candle's potential break
        # Bearish bias if last swing high was LH or last swing low was LL
        prior_bearish_structure = last_sh.swing_type == SwingType.LH or last_sl.swing_type == SwingType.LL
        prior_bullish_structure = last_sh.swing_type == SwingType.HH or last_sl.swing_type == SwingType.HL

        event = StructureEventType.NONE
        # Check fresh close above last confirmed swing high (or current close above un-broken swing high)
        broke_above_sh = close_p > last_sh.price and prev_close <= last_sh.price
        broke_below_sl = close_p < last_sl.price and prev_close >= last_sl.price

        if broke_above_sh:
            if prior_bearish_structure and last_sh.swing_type == SwingType.LH:
                event = StructureEventType.BULLISH_CHOCH
            else:
                event = StructureEventType.BULLISH_BOS
        elif broke_below_sl:
            if prior_bullish_structure and last_sl.swing_type == SwingType.HL:
                event = StructureEventType.BEARISH_CHOCH
            else:
                event = StructureEventType.BEARISH_BOS

        evidence: dict[str, object] = {
            "last_swing_high_price": str(last_sh.price),
            "last_swing_high_type": last_sh.swing_type.value,
            "last_swing_high_index": last_sh.index,
            "last_swing_low_price": str(last_sl.price),
            "last_swing_low_type": last_sl.swing_type.value,
            "last_swing_low_index": last_sl.index,
            "confirmed_swing_highs_count": len(swing_highs),
            "confirmed_swing_lows_count": len(swing_lows),
            "structure_event": event.value,
        }

        buffer = (atr * self._config.sl_buffer_atr).quantize(Decimal("0.01"), rounding=ROUND_DOWN)

        if event in (StructureEventType.BULLISH_BOS, StructureEventType.BULLISH_CHOCH):
            entry = close_p
            sl = (last_sl.price - buffer).quantize(Decimal("0.01"))
            if sl >= entry:
                sl = (entry - atr).quantize(Decimal("0.01"))
            risk_dist = entry - sl
            tp = (entry + risk_dist * self._config.rr_ratio).quantize(Decimal("0.01"))
            conf = Decimal("0.82") if event == StructureEventType.BULLISH_BOS else Decimal("0.78")
            return StrategyEvaluationResult(
                strategy_id=self.strategy_id,
                direction=SignalDirection.BUY,
                timestamp=ts,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                confidence=conf,
                rationale=(
                    f"{event.value} detected: candle closed at {close_p:.2f} above confirmed "
                    f"{last_sh.swing_type.value} swing high ({last_sh.price:.2f})."
                ),
                supporting_evidence=evidence,
            )

        if event in (StructureEventType.BEARISH_BOS, StructureEventType.BEARISH_CHOCH):
            entry = close_p
            sl = (last_sh.price + buffer).quantize(Decimal("0.01"))
            if sl <= entry:
                sl = (entry + atr).quantize(Decimal("0.01"))
            risk_dist = sl - entry
            tp = (entry - risk_dist * self._config.rr_ratio).quantize(Decimal("0.01"))
            conf = Decimal("0.82") if event == StructureEventType.BEARISH_BOS else Decimal("0.78")
            return StrategyEvaluationResult(
                strategy_id=self.strategy_id,
                direction=SignalDirection.SELL,
                timestamp=ts,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                confidence=conf,
                rationale=(
                    f"{event.value} detected: candle closed at {close_p:.2f} below confirmed "
                    f"{last_sl.swing_type.value} swing low ({last_sl.price:.2f})."
                ),
                supporting_evidence=evidence,
            )

        return self.build_wait_result(
            timestamp=ts,
            rationale=(
                f"Price ({close_p:.2f}) remains inside confirmed swing range "
                f"[{last_sl.price:.2f} ({last_sl.swing_type.value}), "
                f"{last_sh.price:.2f} ({last_sh.swing_type.value})]; no BOS or CHoCH on current bar."
            ),
            rejection_reasons=["NO_STRUCTURE_BREAK_ON_CURRENT_BAR"],
            supporting_evidence=evidence,
        )
