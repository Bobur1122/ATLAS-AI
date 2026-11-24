"""Deterministic, leakage-safe Triple-Barrier Labeling Engine for ATLAS AI Phase 6."""

from datetime import datetime
from decimal import Decimal, ROUND_DOWN

import numpy as np
import pandas as pd

from atlas_ai.domain.signals import SignalDirection
from atlas_ai.models.config import LabelingConfig
from atlas_ai.models.exceptions import ModelConfigurationError, ModelDataLeakageError
from atlas_ai.models.schemas import TripleBarrierLabel, direction_to_class_index

QUANTIZE_PRICE = Decimal("0.01")


class TripleBarrierLabeler:
    r"""Computes deterministic BUY, SELL, and WAIT labels over a bounded forward horizon `H`.

    Exact Deterministic Labeling Algorithm at Bar `t`:
    1. Let $C_t = \text{close}[t]$ and $\text{ATR}_t = \text{atr\_14}[t]$.
       Stop distance $d_{\text{SL}} = \text{sl\_atr\_multiplier} \times \text{ATR}_t$.
       Target distance $d_{\text{TP}} = \text{rr\_ratio} \times d_{\text{SL}}$.
    2. Forward Horizon Window:
       Only bars $k \in \{t + 1, \dots, \min(t + H, N - 1)\}$ are inspected.
       Labels NEVER inspect any bar $> t + H$.
       If $t + H \ge N$, the label is marked `is_complete_horizon = False` and set to `WAIT`
       so incomplete tail bars are never used as supervised training targets.
    3. Long (`BUY`) Barrier Simulation over $k = t + 1 \dots t + H$:
       - `long_sl = C_t - d_SL`, `long_tp = C_t + d_TP`.
       - At each bar $k$, apply the **Conservative SL-First Rule**:
         - If `low[k] <= long_sl`, Long hits Stop-Loss at bar $k$ (`long_succeeded = False`,
           `long_resolution_bar = k`) and terminates immediately (even if `high[k] >= long_tp`).
         - Else if `high[k] >= long_tp`, Long hits Take-Profit at bar $k$ (`long_succeeded = True`,
           `long_resolution_bar = k`) and terminates immediately.
    4. Short (`SELL`) Barrier Simulation over $k = t + 1 \dots t + H$:
       - `short_sl = C_t + d_SL`, `short_tp = C_t - d_TP`.
       - At each bar $k$, apply the **Conservative SL-First Rule**:
         - If `high[k] >= short_sl`, Short hits Stop-Loss at bar $k$ (`short_succeeded = False`,
           `short_resolution_bar = k`) and terminates immediately (even if `low[k] <= short_tp`).
         - Else if `low[k] <= short_tp`, Short hits Take-Profit at bar $k$ (`short_succeeded = True`,
           `short_resolution_bar = k`) and terminates immediately.
    5. Directional Outcome & Conflict Resolution:
       - If `long_succeeded` and not `short_succeeded` -> `BUY`.
       - If `short_succeeded` and not `long_succeeded` -> `SELL`.
       - If BOTH `long_succeeded` and `short_succeeded` within horizon $H$:
         - If `long_resolution_bar < short_resolution_bar` -> `BUY` (Long reached target strictly first).
         - If `short_resolution_bar < long_resolution_bar` -> `SELL` (Short reached target strictly first).
         - If `long_resolution_bar == short_resolution_bar` -> `WAIT` (ambiguous same-bar whip-saw).
       - If neither succeeded within $H$ bars -> `WAIT`.
    """

    def __init__(self, config: LabelingConfig | None = None) -> None:
        self._config = config or LabelingConfig()

    @property
    def config(self) -> LabelingConfig:
        return self._config

    @staticmethod
    def _extract_ts(val: object) -> datetime:
        if isinstance(val, pd.Timestamp):
            return val.to_pydatetime()
        if isinstance(val, datetime):
            return val
        return pd.Timestamp(str(val)).to_pydatetime()

    def label_dataframe(self, df: pd.DataFrame) -> tuple[TripleBarrierLabel, ...]:
        """Compute `TripleBarrierLabel` for every row in `df`."""
        if df.empty:
            return ()

        required_cols = ("timestamp", "open", "high", "low", "close", self._config.atr_column)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ModelConfigurationError(
                f"TripleBarrierLabeler missing required columns {missing}."
            )

        ts_series = pd.to_datetime(df["timestamp"])
        if not ts_series.is_monotonic_increasing or ts_series.duplicated().any():
            raise ModelDataLeakageError(
                "TripleBarrierLabeler requires strictly increasing, non-duplicate timestamps."
            )

        highs = df["high"].to_numpy(dtype="float64")
        lows = df["low"].to_numpy(dtype="float64")
        closes = df["close"].to_numpy(dtype="float64")
        atrs = df[self._config.atr_column].to_numpy(dtype="float64")
        timestamps = [self._extract_ts(t) for t in df["timestamp"]]

        n = len(df)
        h = self._config.horizon_bars
        sl_mult = self._config.sl_atr_multiplier
        rr = self._config.rr_ratio

        labels: list[TripleBarrierLabel] = []

        for t in range(n):
            c_val = closes[t]
            atr_val = atrs[t]
            ts = timestamps[t]
            entry_dec = Decimal(str(round(float(c_val), 4))).quantize(QUANTIZE_PRICE)

            # Warm-up NaN in ATR or incomplete forward horizon at the end of the dataset
            has_valid_atr = not np.isnan(atr_val) and float(atr_val) > 0.0
            is_complete = (t + h) < n

            if not has_valid_atr:
                fallback_dist = Decimal("1.00")
                labels.append(
                    TripleBarrierLabel(
                        bar_index=t,
                        timestamp=ts,
                        label=SignalDirection.WAIT,
                        class_index=direction_to_class_index(SignalDirection.WAIT),
                        entry_price=entry_dec,
                        stop_distance=fallback_dist,
                        target_distance=fallback_dist * rr,
                        long_tp=entry_dec + fallback_dist * rr,
                        long_sl=max(Decimal("0.01"), entry_dec - fallback_dist),
                        short_tp=max(Decimal("0.01"), entry_dec - fallback_dist * rr),
                        short_sl=entry_dec + fallback_dist,
                        long_succeeded=False,
                        short_succeeded=False,
                        long_resolution_bar=None,
                        short_resolution_bar=None,
                        max_future_bar_inspected=t,
                        is_complete_horizon=False,
                        rationale="ATR warm-up NaN; assigned WAIT (excluded from training).",
                    )
                )
                continue

            atr_dec = Decimal(str(round(float(atr_val), 4))).quantize(QUANTIZE_PRICE)
            d_sl = max(
                Decimal("0.05"),
                (atr_dec * sl_mult).quantize(QUANTIZE_PRICE, rounding=ROUND_DOWN),
            )
            d_tp = max(
                Decimal("0.05"),
                (d_sl * rr).quantize(QUANTIZE_PRICE, rounding=ROUND_DOWN),
            )

            long_sl = max(Decimal("0.01"), entry_dec - d_sl)
            long_tp = entry_dec + d_tp
            short_sl = entry_dec + d_sl
            short_tp = max(Decimal("0.01"), entry_dec - d_tp)

            if not is_complete:
                labels.append(
                    TripleBarrierLabel(
                        bar_index=t,
                        timestamp=ts,
                        label=SignalDirection.WAIT,
                        class_index=direction_to_class_index(SignalDirection.WAIT),
                        entry_price=entry_dec,
                        stop_distance=d_sl,
                        target_distance=d_tp,
                        long_tp=long_tp,
                        long_sl=long_sl,
                        short_tp=short_tp,
                        short_sl=short_sl,
                        long_succeeded=False,
                        short_succeeded=False,
                        long_resolution_bar=None,
                        short_resolution_bar=None,
                        max_future_bar_inspected=n - 1,
                        is_complete_horizon=False,
                        rationale=f"Incomplete forward horizon ({n - 1 - t} < {h} bars); assigned WAIT.",
                    )
                )
                continue

            long_sl_f = float(long_sl)
            long_tp_f = float(long_tp)
            short_sl_f = float(short_sl)
            short_tp_f = float(short_tp)

            long_done = False
            long_win = False
            long_res_bar: int | None = None

            short_done = False
            short_win = False
            short_res_bar: int | None = None

            max_k = t + h
            max_inspected = t

            for k in range(t + 1, max_k + 1):
                max_inspected = k
                bar_h = highs[k]
                bar_l = lows[k]

                # Long simulation (Conservative SL-First)
                if not long_done:
                    if bar_l <= long_sl_f:
                        long_done = True
                        long_win = False
                        long_res_bar = k
                    elif bar_h >= long_tp_f:
                        long_done = True
                        long_win = True
                        long_res_bar = k

                # Short simulation (Conservative SL-First)
                if not short_done:
                    if bar_h >= short_sl_f:
                        short_done = True
                        short_win = False
                        short_res_bar = k
                    elif bar_l <= short_tp_f:
                        short_done = True
                        short_win = True
                        short_res_bar = k

                if long_done and short_done:
                    break

            direction, reason = self.resolve_barrier_outcomes(
                long_succeeded=long_win,
                short_succeeded=short_win,
                long_resolution_bar=long_res_bar,
                short_resolution_bar=short_res_bar,
            )

            labels.append(
                TripleBarrierLabel(
                    bar_index=t,
                    timestamp=ts,
                    label=direction,
                    class_index=direction_to_class_index(direction),
                    entry_price=entry_dec,
                    stop_distance=d_sl,
                    target_distance=d_tp,
                    long_tp=long_tp,
                    long_sl=long_sl,
                    short_tp=short_tp,
                    short_sl=short_sl,
                    long_succeeded=long_win,
                    short_succeeded=short_win,
                    long_resolution_bar=long_res_bar,
                    short_resolution_bar=short_res_bar,
                    max_future_bar_inspected=max_inspected,
                    is_complete_horizon=True,
                    rationale=reason,
                )
            )

        return tuple(labels)

    @staticmethod
    def resolve_barrier_outcomes(
        *,
        long_succeeded: bool,
        short_succeeded: bool,
        long_resolution_bar: int | None,
        short_resolution_bar: int | None,
    ) -> tuple[SignalDirection, str]:
        """Deterministically resolve BUY, SELL, or WAIT from Long and Short barrier outcomes."""
        if long_succeeded and not short_succeeded:
            return (
                SignalDirection.BUY,
                f"Long TP reached at bar {long_resolution_bar} without touching Long SL.",
            )
        if short_succeeded and not long_succeeded:
            return (
                SignalDirection.SELL,
                f"Short TP reached at bar {short_resolution_bar} without touching Short SL.",
            )
        if long_succeeded and short_succeeded:
            assert long_resolution_bar is not None and short_resolution_bar is not None
            if long_resolution_bar < short_resolution_bar:
                return (
                    SignalDirection.BUY,
                    f"Both outcomes succeeded within horizon; Long TP won earlier "
                    f"(bar {long_resolution_bar} < bar {short_resolution_bar}).",
                )
            if short_resolution_bar < long_resolution_bar:
                return (
                    SignalDirection.SELL,
                    f"Both outcomes succeeded within horizon; Short TP won earlier "
                    f"(bar {short_resolution_bar} < bar {long_resolution_bar}).",
                )
            return (
                SignalDirection.WAIT,
                f"Ambiguous dual barrier success on the same bar ({long_resolution_bar}); assigned WAIT.",
            )
        return (
            SignalDirection.WAIT,
            "Neither Long TP nor Short TP succeeded within horizon.",
        )
