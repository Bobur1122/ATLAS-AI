"""Dealing Range Premium / Discount / Equilibrium Analyzer for ATLAS AI."""

from decimal import Decimal, ROUND_DOWN

import pandas as pd

from atlas_ai.domain.signals import SignalDirection
from atlas_ai.strategies.base import BaseStrategy
from atlas_ai.strategies.config import PremiumDiscountConfig
from atlas_ai.strategies.market_structure import detect_confirmed_swings
from atlas_ai.strategies.models import (
    PremiumDiscountState,
    PriceZoneClassification,
    StrategyEvaluationResult,
)


class PremiumDiscountAnalyzer(BaseStrategy):
    """Evaluates dealing range Premium (> 50%), Discount (< 50%), and Equilibrium (~50%) zones.

    Uses only causally confirmed swing high and swing low (never future swings).
    Acts as a contextual confluence filter (`is_contextual_filter=True`):
    - DISCOUNT favors BUY setups.
    - PREMIUM favors SELL setups.
    - EQUILIBRIUM is neutral (`WAIT`).
    """

    def __init__(self, config: PremiumDiscountConfig | None = None) -> None:
        self._config = config or PremiumDiscountConfig()

    @property
    def strategy_id(self) -> str:
        return "premium_discount_analyzer"

    @property
    def name(self) -> str:
        return "SMC/ICT Dealing Range Premium/Discount Analyzer"

    @property
    def required_features(self) -> tuple[str, ...]:
        return ("timestamp", "open", "high", "low", "close")

    @property
    def is_contextual_filter(self) -> bool:
        return True

    def analyze_state(
        self,
        df: pd.DataFrame,
        as_of_index: int | None = None,
    ) -> tuple[PremiumDiscountState, StrategyEvaluationResult]:
        """Compute `PremiumDiscountState` and contextual `StrategyEvaluationResult` at `as_of_index`."""
        sliced = self.prepare_causal_slice(df, as_of_index)
        last_row = sliced.iloc[-1]
        ts = self.extract_timestamp(last_row)

        swing_highs, swing_lows = detect_confirmed_swings(
            sliced,
            left_bars=self._config.left_bars,
            right_bars=self._config.right_bars,
        )

        if not swing_highs or not swing_lows:
            state = PremiumDiscountState(zone=PriceZoneClassification.UNKNOWN)
            res = self.build_wait_result(
                timestamp=ts,
                rationale="Insufficient confirmed swings to establish a dealing range.",
                rejection_reasons=["NO_CONFIRMED_DEALING_RANGE"],
                supporting_evidence=state.model_dump(mode="json"),
            )
            return state, res

        range_high = swing_highs[-1].price
        range_low = swing_lows[-1].price

        # If the most recent swing high is somehow <= the most recent swing low, use the recent confirmed swing window
        if range_high <= range_low:
            recent_highs = [s.price for s in swing_highs[-3:]]
            recent_lows = [s.price for s in swing_lows[-3:]]
            range_high = max(recent_highs)
            range_low = min(recent_lows)

        if range_high <= range_low:
            state = PremiumDiscountState(zone=PriceZoneClassification.UNKNOWN)
            res = self.build_wait_result(
                timestamp=ts,
                rationale="Dealing range high is not above dealing range low.",
                rejection_reasons=["DEGENERATE_DEALING_RANGE"],
                supporting_evidence=state.model_dump(mode="json"),
            )
            return state, res

        close_p = Decimal(str(last_row["close"])).quantize(Decimal("0.01"))
        equilibrium = ((range_high + range_low) / Decimal("2")).quantize(Decimal("0.01"))
        pos_in_range = ((close_p - range_low) / (range_high - range_low)).quantize(
            Decimal("0.0001"), rounding=ROUND_DOWN
        )

        tol = self._config.equilibrium_tolerance
        lower_eq = Decimal("0.50") - tol
        upper_eq = Decimal("0.50") + tol

        if lower_eq <= pos_in_range <= upper_eq:
            zone = PriceZoneClassification.EQUILIBRIUM
            favored_dir = SignalDirection.WAIT
            conf = Decimal("0")
            rationale = (
                f"Price ({close_p:.2f}) is in EQUILIBRIUM ({pos_in_range * 100:.1f}%) of dealing range "
                f"[{range_low:.2f}, {range_high:.2f}] (EQ={equilibrium:.2f})."
            )
        elif pos_in_range < lower_eq:
            zone = PriceZoneClassification.DISCOUNT
            favored_dir = SignalDirection.BUY
            conf = Decimal("0.70")
            rationale = (
                f"Price ({close_p:.2f}) is in DISCOUNT ({pos_in_range * 100:.1f}% < 50%) of dealing range "
                f"[{range_low:.2f}, {range_high:.2f}] (EQ={equilibrium:.2f}), favoring BUY setups."
            )
        else:
            zone = PriceZoneClassification.PREMIUM
            favored_dir = SignalDirection.SELL
            conf = Decimal("0.70")
            rationale = (
                f"Price ({close_p:.2f}) is in PREMIUM ({pos_in_range * 100:.1f}% > 50%) of dealing range "
                f"[{range_low:.2f}, {range_high:.2f}] (EQ={equilibrium:.2f}), favoring SELL setups."
            )

        state = PremiumDiscountState(
            range_high=range_high,
            range_low=range_low,
            equilibrium=equilibrium,
            current_price=close_p,
            position_in_range=pos_in_range,
            zone=zone,
        )
        eval_res = StrategyEvaluationResult(
            strategy_id=self.strategy_id,
            direction=favored_dir,
            timestamp=ts,
            confidence=conf,
            rationale=rationale,
            supporting_evidence=state.model_dump(mode="json"),
            rejection_reasons=("IN_EQUILIBRIUM",) if zone == PriceZoneClassification.EQUILIBRIUM else (),
            is_contextual_filter=True,
        )
        return state, eval_res

    def evaluate(
        self,
        df: pd.DataFrame,
        as_of_index: int | None = None,
    ) -> StrategyEvaluationResult:
        _, res = self.analyze_state(df, as_of_index=as_of_index)
        return res
