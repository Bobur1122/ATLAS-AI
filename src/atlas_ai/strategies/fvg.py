"""Fair Value Gap (FVG) Detection and FVG Retest Strategies for ATLAS AI."""

from decimal import Decimal, ROUND_DOWN

import numpy as np
import pandas as pd

from atlas_ai.domain.signals import SignalDirection
from atlas_ai.strategies.base import BaseStrategy
from atlas_ai.strategies.config import FVGConfig
from atlas_ai.strategies.models import FairValueGap, StrategyEvaluationResult


def scan_historical_fvgs(
    df: pd.DataFrame,
    min_gap_atr_ratio: Decimal = Decimal("0.10"),
    max_age_bars: int = 40,
) -> list[FairValueGap]:
    """Causally scan an OHLCV DataFrame up to its final row and track all Fair Value Gaps.

    Rules (zero lookahead):
    - Bullish FVG at bar `i` (`i >= 2`): `low[i] > high[i - 2]`.
      `lower_boundary = high[i - 2]`, `upper_boundary = low[i]`.
      Mitigated on any subsequent bar `j > i` if `low[j] <= lower_boundary`.
      Partially filled on bar `j > i` if `low[j] <= upper_boundary`.
    - Bearish FVG at bar `i` (`i >= 2`): `high[i] < low[i - 2]`.
      `upper_boundary = low[i - 2]`, `lower_boundary = high[i]`.
      Mitigated on any subsequent bar `j > i` if `high[j] >= upper_boundary`.
      Partially filled on bar `j > i` if `high[j] >= lower_boundary`.
    """
    n = len(df)
    if n < 3:
        return []

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

    last_idx = n - 1
    fvgs: list[FairValueGap] = []

    for i in range(2, n):
        atr_val = atr_arr[i]
        min_gap = float(min_gap_atr_ratio) * (
            atr_val if not np.isnan(atr_val) and atr_val > 0 else 0.0
        )

        # 1. Bullish FVG check: low[i] > high[i - 2]
        if lows[i] > highs[i - 2] and (lows[i] - highs[i - 2]) >= min_gap:
            lower = Decimal(str(round(float(highs[i - 2]), 4))).quantize(Decimal("0.01"))
            upper = Decimal(str(round(float(lows[i]), 4))).quantize(Decimal("0.01"))
            if upper > lower:
                mitigated = False
                partially_filled = False
                mit_ts = None
                for j in range(i + 1, n):
                    if lows[j] <= float(upper):
                        partially_filled = True
                    if lows[j] <= float(lower):
                        mitigated = True
                        mit_ts = timestamps[j]
                        break
                age = last_idx - i
                is_valid = (not mitigated) and (age <= max_age_bars)
                fvgs.append(
                    FairValueGap(
                        direction=SignalDirection.BUY,
                        upper_boundary=upper,
                        lower_boundary=lower,
                        formation_index=i,
                        formation_timestamp=timestamps[i],
                        gap_size=upper - lower,
                        mitigated=mitigated,
                        partially_filled=partially_filled,
                        mitigation_timestamp=mit_ts,
                        age_bars=age,
                        is_valid=is_valid,
                    )
                )

        # 2. Bearish FVG check: high[i] < low[i - 2]
        if highs[i] < lows[i - 2] and (lows[i - 2] - highs[i]) >= min_gap:
            upper = Decimal(str(round(float(lows[i - 2]), 4))).quantize(Decimal("0.01"))
            lower = Decimal(str(round(float(highs[i]), 4))).quantize(Decimal("0.01"))
            if upper > lower:
                mitigated = False
                partially_filled = False
                mit_ts = None
                for j in range(i + 1, n):
                    if highs[j] >= float(lower):
                        partially_filled = True
                    if highs[j] >= float(upper):
                        mitigated = True
                        mit_ts = timestamps[j]
                        break
                age = last_idx - i
                is_valid = (not mitigated) and (age <= max_age_bars)
                fvgs.append(
                    FairValueGap(
                        direction=SignalDirection.SELL,
                        upper_boundary=upper,
                        lower_boundary=lower,
                        formation_index=i,
                        formation_timestamp=timestamps[i],
                        gap_size=upper - lower,
                        mitigated=mitigated,
                        partially_filled=partially_filled,
                        mitigation_timestamp=mit_ts,
                        age_bars=age,
                        is_valid=is_valid,
                    )
                )

    return fvgs


class FVGDetectionStrategy(BaseStrategy):
    """Detects newly formed Bullish or Bearish Fair Value Gaps on the current bar and tracks active FVGs."""

    def __init__(self, config: FVGConfig | None = None) -> None:
        self._config = config or FVGConfig()

    @property
    def strategy_id(self) -> str:
        return "fvg_detection"

    @property
    def name(self) -> str:
        return "SMC/ICT Fair Value Gap (FVG) Detector"

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

        if len(sliced) < 3:
            return self.build_wait_result(
                timestamp=ts,
                rationale="At least 3 candles are required to detect a 3-bar Fair Value Gap.",
                rejection_reasons=["INSUFFICIENT_BARS_FOR_FVG"],
            )

        fvgs = scan_historical_fvgs(
            sliced,
            min_gap_atr_ratio=self._config.min_gap_atr_ratio,
            max_age_bars=self._config.max_age_bars,
        )
        valid_fvgs = [g for g in fvgs if g.is_valid]
        last_bar_idx = len(sliced) - 1
        newly_formed = [g for g in valid_fvgs if g.formation_index == last_bar_idx]

        evidence: dict[str, object] = {
            "total_fvgs_detected": len(fvgs),
            "active_unmitigated_fvgs": len(valid_fvgs),
            "newly_formed_on_current_bar": len(newly_formed) > 0,
        }

        raw_atr = last_row.get("atr_14")
        atr = (
            Decimal(str(raw_atr)).quantize(Decimal("0.01"))
            if raw_atr is not None and not pd.isna(raw_atr) and float(raw_atr) > 0
            else Decimal("2.00")
        )
        buffer = (atr * self._config.sl_buffer_atr).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        close_p = Decimal(str(last_row["close"])).quantize(Decimal("0.01"))

        if newly_formed:
            fvg = newly_formed[-1]
            evidence["latest_fvg"] = fvg.model_dump(mode="json")
            if fvg.direction == SignalDirection.BUY:
                entry = close_p
                sl = (fvg.lower_boundary - buffer).quantize(Decimal("0.01"))
                if sl >= entry:
                    sl = (entry - atr).quantize(Decimal("0.01"))
                tp = (entry + (entry - sl) * self._config.rr_ratio).quantize(Decimal("0.01"))
                return StrategyEvaluationResult(
                    strategy_id=self.strategy_id,
                    direction=SignalDirection.BUY,
                    timestamp=ts,
                    entry_price=entry,
                    stop_loss=sl,
                    take_profit=tp,
                    confidence=Decimal("0.70"),
                    rationale=(
                        f"Bullish FVG formed on current bar: zone [{fvg.lower_boundary:.2f}, "
                        f"{fvg.upper_boundary:.2f}] (size {fvg.gap_size:.2f})."
                    ),
                    supporting_evidence=evidence,
                )
            else:
                entry = close_p
                sl = (fvg.upper_boundary + buffer).quantize(Decimal("0.01"))
                if sl <= entry:
                    sl = (entry + atr).quantize(Decimal("0.01"))
                tp = (entry - (sl - entry) * self._config.rr_ratio).quantize(Decimal("0.01"))
                return StrategyEvaluationResult(
                    strategy_id=self.strategy_id,
                    direction=SignalDirection.SELL,
                    timestamp=ts,
                    entry_price=entry,
                    stop_loss=sl,
                    take_profit=tp,
                    confidence=Decimal("0.70"),
                    rationale=(
                        f"Bearish FVG formed on current bar: zone [{fvg.lower_boundary:.2f}, "
                        f"{fvg.upper_boundary:.2f}] (size {fvg.gap_size:.2f})."
                    ),
                    supporting_evidence=evidence,
                )

        return self.build_wait_result(
            timestamp=ts,
            rationale=f"No new FVG formed on current candle ({len(valid_fvgs)} active unmitigated FVGs tracked).",
            rejection_reasons=["NO_NEW_FVG_ON_CURRENT_BAR"],
            supporting_evidence=evidence,
        )


class FVGRetestStrategy(BaseStrategy):
    """Evaluates price revisiting an existing valid (unmitigated) Fair Value Gap with directional confirmation.

    BUY Retest Rules:
    - An unmitigated Bullish FVG formed on an earlier bar (`formation_index < current_bar_index`, `age <= max_age_bars`).
    - Current candle trades into the FVG zone (`low <= fvg.upper_boundary` and `low > fvg.lower_boundary`).
    - Price respects the lower boundary (`close >= fvg.lower_boundary`) and shows directional confirmation
      (bullish candle `close > open` or strong lower rejection wick).
    - Structural stop-loss is placed below `fvg.lower_boundary - buffer`.

    SELL Retest Rules:
    - An unmitigated Bearish FVG formed on an earlier bar (`formation_index < current_bar_index`, `age <= max_age_bars`).
    - Current candle trades into the FVG zone (`high >= fvg.lower_boundary` and `high < fvg.upper_boundary`).
    - Price respects the upper boundary (`close <= fvg.upper_boundary`) and shows directional confirmation
      (bearish candle `close < open` or strong upper rejection wick).
    - Structural stop-loss is placed above `fvg.upper_boundary + buffer`.
    """

    def __init__(self, config: FVGConfig | None = None) -> None:
        self._config = config or FVGConfig()

    @property
    def strategy_id(self) -> str:
        return "fvg_retest"

    @property
    def name(self) -> str:
        return "SMC/ICT Fair Value Gap (FVG) Retest Strategy"

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

        if len(sliced) < 4:
            return self.build_wait_result(
                timestamp=ts,
                rationale="At least 4 bars required (3 to form FVG, 1+ to retest).",
                rejection_reasons=["INSUFFICIENT_BARS_FOR_FVG_RETEST"],
            )

        # Scan FVGs that were active and unmitigated as of the PREVIOUS bar (`sliced.iloc[:-1]`)
        prior_fvgs = scan_historical_fvgs(
            sliced.iloc[:-1],
            min_gap_atr_ratio=self._config.min_gap_atr_ratio,
            max_age_bars=self._config.max_age_bars,
        )
        candidate_fvgs = [g for g in prior_fvgs if g.is_valid]

        open_p = Decimal(str(last_row["open"])).quantize(Decimal("0.01"))
        high_p = Decimal(str(last_row["high"])).quantize(Decimal("0.01"))
        low_p = Decimal(str(last_row["low"])).quantize(Decimal("0.01"))
        close_p = Decimal(str(last_row["close"])).quantize(Decimal("0.01"))

        raw_atr = last_row.get("atr_14")
        atr = (
            Decimal(str(raw_atr)).quantize(Decimal("0.01"))
            if raw_atr is not None and not pd.isna(raw_atr) and float(raw_atr) > 0
            else Decimal("2.00")
        )
        buffer = (atr * self._config.sl_buffer_atr).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        if buffer <= Decimal("0"):
            buffer = Decimal("0.50")

        evidence: dict[str, object] = {
            "prior_valid_fvgs_count": len(candidate_fvgs),
            "retested_fvg": None,
        }

        # Evaluate most recent valid FVGs first
        for fvg in reversed(candidate_fvgs):
            if fvg.direction == SignalDirection.BUY:
                # Bullish FVG retest: price dips into [lower_boundary, upper_boundary] without mitigating below lower_boundary
                if low_p <= fvg.upper_boundary and low_p > fvg.lower_boundary and close_p > fvg.lower_boundary:
                    # Directional confirmation: bullish candle close > open
                    if close_p > open_p:
                        entry = close_p
                        sl = (fvg.lower_boundary - buffer).quantize(Decimal("0.01"))
                        if sl >= entry:
                            continue
                        risk_dist = entry - sl
                        tp = (entry + risk_dist * self._config.rr_ratio).quantize(Decimal("0.01"))
                        evidence["retested_fvg"] = fvg.model_dump(mode="json")
                        return StrategyEvaluationResult(
                            strategy_id=self.strategy_id,
                            direction=SignalDirection.BUY,
                            timestamp=ts,
                            entry_price=entry,
                            stop_loss=sl,
                            take_profit=tp,
                            confidence=Decimal("0.84"),
                            rationale=(
                                f"Bullish FVG retest confirmed: low ({low_p:.2f}) tapped unmitigated FVG "
                                f"[{fvg.lower_boundary:.2f}, {fvg.upper_boundary:.2f}] and closed bullish at {close_p:.2f}."
                            ),
                            supporting_evidence=evidence,
                        )
                    else:
                        return self.build_wait_result(
                            timestamp=ts,
                            rationale=(
                                f"Price entered Bullish FVG [{fvg.lower_boundary:.2f}, {fvg.upper_boundary:.2f}], "
                                "but lacked bullish candle close confirmation."
                            ),
                            rejection_reasons=["FVG_RETEST_LACKED_DIRECTIONAL_CONFIRMATION"],
                            supporting_evidence=evidence,
                        )

            elif fvg.direction == SignalDirection.SELL:
                # Bearish FVG retest: price rallies into [lower_boundary, upper_boundary] without mitigating above upper_boundary
                if high_p >= fvg.lower_boundary and high_p < fvg.upper_boundary and close_p < fvg.upper_boundary:
                    if close_p < open_p:
                        entry = close_p
                        sl = (fvg.upper_boundary + buffer).quantize(Decimal("0.01"))
                        if sl <= entry:
                            continue
                        risk_dist = sl - entry
                        tp = (entry - risk_dist * self._config.rr_ratio).quantize(Decimal("0.01"))
                        evidence["retested_fvg"] = fvg.model_dump(mode="json")
                        return StrategyEvaluationResult(
                            strategy_id=self.strategy_id,
                            direction=SignalDirection.SELL,
                            timestamp=ts,
                            entry_price=entry,
                            stop_loss=sl,
                            take_profit=tp,
                            confidence=Decimal("0.84"),
                            rationale=(
                                f"Bearish FVG retest confirmed: high ({high_p:.2f}) tapped unmitigated FVG "
                                f"[{fvg.lower_boundary:.2f}, {fvg.upper_boundary:.2f}] and closed bearish at {close_p:.2f}."
                            ),
                            supporting_evidence=evidence,
                        )
                    else:
                        return self.build_wait_result(
                            timestamp=ts,
                            rationale=(
                                f"Price entered Bearish FVG [{fvg.lower_boundary:.2f}, {fvg.upper_boundary:.2f}], "
                                "but lacked bearish candle close confirmation."
                            ),
                            rejection_reasons=["FVG_RETEST_LACKED_DIRECTIONAL_CONFIRMATION"],
                            supporting_evidence=evidence,
                        )

        return self.build_wait_result(
            timestamp=ts,
            rationale="No valid unmitigated FVG was retested on the current candle.",
            rejection_reasons=["NO_FVG_RETEST"],
            supporting_evidence=evidence,
        )
