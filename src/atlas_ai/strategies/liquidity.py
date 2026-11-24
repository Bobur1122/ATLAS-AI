"""Price-action Liquidity Pool (EQH, EQL, BSL, SSL) and Liquidity Sweep + Rejection Strategy."""

from decimal import Decimal, ROUND_DOWN

import pandas as pd

from atlas_ai.domain.signals import SignalDirection
from atlas_ai.strategies.base import BaseStrategy
from atlas_ai.strategies.config import LiquidityConfig
from atlas_ai.strategies.market_structure import detect_confirmed_swings
from atlas_ai.strategies.models import LiquidityPool, StrategyEvaluationResult, SwingPoint


def detect_liquidity_pools(
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    atr: Decimal,
    eq_tolerance_atr: Decimal = Decimal("0.15"),
) -> list[LiquidityPool]:
    """Identify Buy-Side Liquidity (EQH / BSL) and Sell-Side Liquidity (EQL / SSL) from confirmed swings.

    Note: These are deterministic price-action liquidity proxies based on swing highs/lows and
    equal highs/lows, not claims of direct institutional order-book visibility.
    """
    tol = atr * eq_tolerance_atr
    pools: list[LiquidityPool] = []

    # Check consecutive confirmed swing highs for Equal Highs (EQH) or standard BSL
    for i, sh in enumerate(swing_highs):
        is_eqh = False
        first_ts = sh.timestamp
        second_ts = None
        level = sh.price
        if i > 0:
            prev_sh = swing_highs[i - 1]
            if abs(sh.price - prev_sh.price) <= tol:
                is_eqh = True
                first_ts = prev_sh.timestamp
                second_ts = sh.timestamp
                level = max(sh.price, prev_sh.price)

        pools.append(
            LiquidityPool(
                pool_type="EQH" if is_eqh else "BSL",
                is_buy_side=True,
                level_price=level,
                first_timestamp=first_ts,
                second_timestamp=second_ts,
            )
        )

    # Check consecutive confirmed swing lows for Equal Lows (EQL) or standard SSL
    for i, sl in enumerate(swing_lows):
        is_eql = False
        first_ts = sl.timestamp
        second_ts = None
        level = sl.price
        if i > 0:
            prev_sl = swing_lows[i - 1]
            if abs(sl.price - prev_sl.price) <= tol:
                is_eql = True
                first_ts = prev_sl.timestamp
                second_ts = sl.timestamp
                level = min(sl.price, prev_sl.price)

        pools.append(
            LiquidityPool(
                pool_type="EQL" if is_eql else "SSL",
                is_buy_side=False,
                level_price=level,
                first_timestamp=first_ts,
                second_timestamp=second_ts,
            )
        )

    return pools


class LiquiditySweepStrategy(BaseStrategy):
    """Detects Equal Highs (EQH), Equal Lows (EQL), Buy-Side/Sell-Side swing liquidity, and sweeps.

    BUY Setup (Sell-Side Liquidity Sweep + Rejection):
    - Current candle low sweeps below an unswept confirmed liquidity low (EQL or SSL) by at least
      `min_sweep_penetration_atr * ATR`.
    - Current candle closes back above the swept liquidity level (`close > level_price`).
    - Confirmation conditions satisfied (bullish candle close or rejection wick).

    SELL Setup (Buy-Side Liquidity Sweep + Rejection):
    - Current candle high sweeps above an unswept confirmed liquidity high (EQH or BSL) by at least
      `min_sweep_penetration_atr * ATR`.
    - Current candle closes back below the swept liquidity level (`close < level_price`).
    - Confirmation conditions satisfied (bearish candle close or rejection wick).
    """

    def __init__(self, config: LiquidityConfig | None = None) -> None:
        self._config = config or LiquidityConfig()

    @property
    def strategy_id(self) -> str:
        return "liquidity_sweep"

    @property
    def name(self) -> str:
        return "SMC/ICT Liquidity Sweep & Rejection Strategy"

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

        if len(sliced) < self._config.left_bars + self._config.right_bars + 2:
            return self.build_wait_result(
                timestamp=ts,
                rationale="Insufficient bars to confirm prior swing liquidity pools.",
                rejection_reasons=["INSUFFICIENT_HISTORY"],
            )

        # Detect swings confirmed strictly BEFORE the current bar (`sliced.iloc[:-1]`)
        # so the current bar can sweep a pre-existing confirmed liquidity level
        prior_df = sliced.iloc[:-1]
        swing_highs, swing_lows = detect_confirmed_swings(
            prior_df,
            left_bars=self._config.left_bars,
            right_bars=self._config.right_bars,
            equal_tolerance_atr=self._config.eq_tolerance_atr,
        )

        raw_atr = last_row.get("atr_14")
        if raw_atr is not None and not pd.isna(raw_atr) and float(raw_atr) > 0:
            atr = Decimal(str(raw_atr)).quantize(Decimal("0.01"))
        else:
            atr = Decimal("2.00")

        pools = detect_liquidity_pools(
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            atr=atr,
            eq_tolerance_atr=self._config.eq_tolerance_atr,
        )

        open_p = Decimal(str(last_row["open"])).quantize(Decimal("0.01"))
        high_p = Decimal(str(last_row["high"])).quantize(Decimal("0.01"))
        low_p = Decimal(str(last_row["low"])).quantize(Decimal("0.01"))
        close_p = Decimal(str(last_row["close"])).quantize(Decimal("0.01"))

        min_penetration = atr * self._config.min_sweep_penetration_atr
        sl_buffer = (atr * self._config.sl_buffer_atr).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        if sl_buffer <= Decimal("0"):
            sl_buffer = Decimal("0.50")

        eqh_levels = [str(p.level_price) for p in pools if p.pool_type == "EQH"]
        eql_levels = [str(p.level_price) for p in pools if p.pool_type == "EQL"]

        evidence: dict[str, object] = {
            "eqh_levels": eqh_levels,
            "eql_levels": eql_levels,
            "total_pools_detected": len(pools),
            "swept_pool_type": None,
            "swept_level": None,
        }

        # Filter to most recent unswept SSL/EQL and BSL/EQH pools
        # A pool at swing index `s_idx` is unswept before current bar if no bar between `s_idx + 1` and `len(prior_df) - 1` broke it
        prior_highs = prior_df["high"].to_numpy(dtype="float64")
        prior_lows = prior_df["low"].to_numpy(dtype="float64")

        # 1. Check Sell-Side Liquidity (SSL / EQL) sweep for BUY setup
        for sl_point in reversed(swing_lows):
            intervening_lows = prior_lows[sl_point.index + 1 :]
            if len(intervening_lows) > 0 and float(intervening_lows.min()) < float(sl_point.price):
                continue  # Already swept on an earlier bar

            level = sl_point.price
            is_eql = any(p.pool_type == "EQL" and p.level_price == level for p in pools)
            pool_label = "EQL" if is_eql else "SSL"

            if low_p <= level - min_penetration and close_p > level:
                dir_ok = (close_p >= open_p) if self._config.require_directional_close else True
                if dir_ok:
                    entry = close_p
                    sl = (low_p - sl_buffer).quantize(Decimal("0.01"))
                    risk_dist = entry - sl
                    tp = (entry + risk_dist * self._config.rr_ratio).quantize(Decimal("0.01"))
                    evidence["swept_pool_type"] = pool_label
                    evidence["swept_level"] = str(level)
                    conf = Decimal("0.85") if is_eql else Decimal("0.78")
                    return StrategyEvaluationResult(
                        strategy_id=self.strategy_id,
                        direction=SignalDirection.BUY,
                        timestamp=ts,
                        entry_price=entry,
                        stop_loss=sl,
                        take_profit=tp,
                        confidence=conf,
                        rationale=(
                            f"Bullish liquidity sweep & rejection of {pool_label} at {level:.2f} "
                            f"(wick low {low_p:.2f}, closed back above at {close_p:.2f})."
                        ),
                        supporting_evidence=evidence,
                    )

        # 2. Check Buy-Side Liquidity (BSL / EQH) sweep for SELL setup
        for sh_point in reversed(swing_highs):
            intervening_highs = prior_highs[sh_point.index + 1 :]
            if len(intervening_highs) > 0 and float(intervening_highs.max()) > float(sh_point.price):
                continue  # Already swept on an earlier bar

            level = sh_point.price
            is_eqh = any(p.pool_type == "EQH" and p.level_price == level for p in pools)
            pool_label = "EQH" if is_eqh else "BSL"

            if high_p >= level + min_penetration and close_p < level:
                dir_ok = (close_p <= open_p) if self._config.require_directional_close else True
                if dir_ok:
                    entry = close_p
                    sl = (high_p + sl_buffer).quantize(Decimal("0.01"))
                    risk_dist = sl - entry
                    tp = (entry - risk_dist * self._config.rr_ratio).quantize(Decimal("0.01"))
                    evidence["swept_pool_type"] = pool_label
                    evidence["swept_level"] = str(level)
                    conf = Decimal("0.85") if is_eqh else Decimal("0.78")
                    return StrategyEvaluationResult(
                        strategy_id=self.strategy_id,
                        direction=SignalDirection.SELL,
                        timestamp=ts,
                        entry_price=entry,
                        stop_loss=sl,
                        take_profit=tp,
                        confidence=conf,
                        rationale=(
                            f"Bearish liquidity sweep & rejection of {pool_label} at {level:.2f} "
                            f"(wick high {high_p:.2f}, closed back below at {close_p:.2f})."
                        ),
                        supporting_evidence=evidence,
                    )

        return self.build_wait_result(
            timestamp=ts,
            rationale="No confirmed liquidity sweep and rejection occurred on the current candle.",
            rejection_reasons=["NO_LIQUIDITY_SWEEP_ON_CURRENT_BAR"],
            supporting_evidence=evidence,
        )
