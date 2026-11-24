"""Previous Day High/Low/Close (PDH/PDL/PDC) and ICT Kill Zone Session Analyzers for ATLAS AI."""

from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pandas as pd

from atlas_ai.domain.signals import SignalDirection
from atlas_ai.strategies.base import BaseStrategy
from atlas_ai.strategies.config import KillZoneSessionConfig, PreviousDayLevelsConfig
from atlas_ai.strategies.models import (
    PreviousDayLevelsState,
    StrategyEvaluationResult,
)


def compute_trading_day_label(
    ts: datetime,
    tz: ZoneInfo,
    rollover_hour: int,
) -> date:
    """Assign a timestamp to its logical Forex/metals trading day based on `tz` and `rollover_hour`.

    For example, when `tz = ZoneInfo('America/New_York')` and `rollover_hour = 17` (5 PM NY close):
    - 2026-01-05 16:00 NY belongs to trading day `2026-01-05`.
    - 2026-01-05 18:00 NY belongs to trading day `2026-01-06`.
    When `rollover_hour = 0`, the trading day is simply the calendar date in `tz`.
    """
    local_ts = ts.astimezone(tz) if ts.tzinfo is not None else ts.replace(tzinfo=tz)
    if rollover_hour == 0:
        return local_ts.date()
    if local_ts.hour >= rollover_hour:
        return (local_ts + timedelta(days=1)).date()
    return local_ts.date()


def _hour_in_window(hour: int, start_hour: int, end_hour: int) -> bool:
    """Return True if `hour` falls in `[start_hour, end_hour)`, supporting midnight wrap-around."""
    if start_hour == end_hour:
        return False
    if start_hour < end_hour:
        return start_hour <= hour < end_hour
    # Wrap across midnight (e.g., 19:00 to 00:00 -> hours 19, 20, 21, 22, 23)
    return hour >= start_hour or hour < end_hour


class PreviousDayLevelsAnalyzer(BaseStrategy):
    """Computes Previous Day High (PDH), Previous Day Low (PDL), Previous Day Close (PDC),
    Current Day High/Low, and detects PDH/PDL liquidity sweeps without lookahead bias.
    """

    def __init__(self, config: PreviousDayLevelsConfig | None = None) -> None:
        self._config = config or PreviousDayLevelsConfig()
        self._tz = ZoneInfo(self._config.day_boundary_timezone)

    @property
    def strategy_id(self) -> str:
        return "pdh_pdl_analyzer"

    @property
    def name(self) -> str:
        return "Previous Day High / Low (PDH / PDL) Analyzer"

    @property
    def required_features(self) -> tuple[str, ...]:
        return ("timestamp", "open", "high", "low", "close")

    @property
    def is_contextual_filter(self) -> bool:
        return True

    def analyze_levels(
        self,
        df: pd.DataFrame,
        as_of_index: int | None = None,
    ) -> tuple[PreviousDayLevelsState, StrategyEvaluationResult]:
        """Causally compute PDH/PDL/PDC and current-day interaction up to `as_of_index`."""
        sliced = self.prepare_causal_slice(df, as_of_index)
        last_row = sliced.iloc[-1]
        ts = self.extract_timestamp(last_row)

        timestamps = [
            t.to_pydatetime() if isinstance(t, pd.Timestamp) else pd.Timestamp(t).to_pydatetime()
            for t in sliced["timestamp"]
        ]
        day_labels = [
            compute_trading_day_label(t, self._tz, self._config.day_rollover_hour)
            for t in timestamps
        ]

        current_day = day_labels[-1]
        # Identify distinct completed trading days strictly prior to `current_day`
        prior_days = sorted({d for d in day_labels if d < current_day})

        highs = sliced["high"].to_numpy(dtype="float64")
        lows = sliced["low"].to_numpy(dtype="float64")
        closes = sliced["close"].to_numpy(dtype="float64")

        curr_mask = [d == current_day for d in day_labels]
        curr_high = Decimal(str(round(float(highs[curr_mask].max()), 4))).quantize(Decimal("0.01"))
        curr_low = Decimal(str(round(float(lows[curr_mask].min()), 4))).quantize(Decimal("0.01"))

        if not prior_days:
            state = PreviousDayLevelsState(
                current_day_high=curr_high,
                current_day_low=curr_low,
            )
            res = self.build_wait_result(
                timestamp=ts,
                rationale="No completed previous trading day available in causal history for PDH/PDL.",
                rejection_reasons=["NO_PREVIOUS_TRADING_DAY"],
                supporting_evidence=state.model_dump(mode="json"),
            )
            return state, res

        prev_day = prior_days[-1]
        prev_indices = [i for i, d in enumerate(day_labels) if d == prev_day]
        pdh = Decimal(str(round(float(highs[prev_indices].max()), 4))).quantize(Decimal("0.01"))
        pdl = Decimal(str(round(float(lows[prev_indices].min()), 4))).quantize(Decimal("0.01"))
        pdc = Decimal(str(round(float(closes[prev_indices[-1]]), 4))).quantize(Decimal("0.01"))

        bar_high = Decimal(str(last_row["high"])).quantize(Decimal("0.01"))
        bar_low = Decimal(str(last_row["low"])).quantize(Decimal("0.01"))
        bar_close = Decimal(str(last_row["close"])).quantize(Decimal("0.01"))

        dist_pdh = (bar_close - pdh).quantize(Decimal("0.01"))
        dist_pdl = (bar_close - pdl).quantize(Decimal("0.01"))

        pdh_swept_rej = bar_high > pdh and bar_close < pdh
        pdl_swept_rej = bar_low < pdl and bar_close > pdl
        pdh_broken = bar_close > pdh
        pdl_broken = bar_close < pdl

        state = PreviousDayLevelsState(
            pdh=pdh,
            pdl=pdl,
            pdc=pdc,
            current_day_high=curr_high,
            current_day_low=curr_low,
            distance_to_pdh=dist_pdh,
            distance_to_pdl=dist_pdl,
            pdh_swept_and_rejected=pdh_swept_rej,
            pdl_swept_and_rejected=pdl_swept_rej,
            pdh_broken_above=pdh_broken,
            pdl_broken_below=pdl_broken,
        )

        evidence = state.model_dump(mode="json")
        evidence["previous_trading_day"] = prev_day.isoformat()
        evidence["current_trading_day"] = current_day.isoformat()

        if pdl_swept_rej:
            return state, StrategyEvaluationResult(
                strategy_id=self.strategy_id,
                direction=SignalDirection.BUY,
                timestamp=ts,
                confidence=Decimal("0.80"),
                rationale=(
                    f"Bullish PDL liquidity sweep: candle low ({bar_low:.2f}) swept below PDL "
                    f"({pdl:.2f}) and closed back above at {bar_close:.2f}."
                ),
                supporting_evidence=evidence,
                is_contextual_filter=True,
            )

        if pdh_swept_rej:
            return state, StrategyEvaluationResult(
                strategy_id=self.strategy_id,
                direction=SignalDirection.SELL,
                timestamp=ts,
                confidence=Decimal("0.80"),
                rationale=(
                    f"Bearish PDH liquidity sweep: candle high ({bar_high:.2f}) swept above PDH "
                    f"({pdh:.2f}) and closed back below at {bar_close:.2f}."
                ),
                supporting_evidence=evidence,
                is_contextual_filter=True,
            )

        return state, self.build_wait_result(
            timestamp=ts,
            rationale=(
                f"PDH={pdh:.2f}, PDL={pdl:.2f}, PDC={pdc:.2f}; current close={bar_close:.2f} "
                "(no PDH/PDL sweep-and-rejection on current candle)."
            ),
            rejection_reasons=["NO_PDH_PDL_SWEEP"],
            supporting_evidence=evidence,
        )

    def evaluate(
        self,
        df: pd.DataFrame,
        as_of_index: int | None = None,
    ) -> StrategyEvaluationResult:
        _, res = self.analyze_levels(df, as_of_index=as_of_index)
        return res


class KillZoneSessionAnalyzer(BaseStrategy):
    """DST-aware Session & ICT Kill Zone Analyzer.

    Identifies:
    - Asian Session, London Session, New York Session, London/NY Overlap
    - London Kill Zone and New York Kill Zone
    - Causal Session High / Session Low and prior-session liquidity sweeps (e.g., London sweeping Asian High/Low).
    """

    def __init__(self, config: KillZoneSessionConfig | None = None) -> None:
        self._config = config or KillZoneSessionConfig()
        self._tz = ZoneInfo(self._config.timezone)

    @property
    def strategy_id(self) -> str:
        return "killzone_session_analyzer"

    @property
    def name(self) -> str:
        return "ICT Kill Zone & Session Liquidity Analyzer"

    @property
    def required_features(self) -> tuple[str, ...]:
        return ("timestamp", "open", "high", "low", "close")

    @property
    def is_contextual_filter(self) -> bool:
        return True

    def classify_bar_time(self, ts: datetime) -> dict[str, object]:
        """Classify a timestamp in `self._tz` (with automatic IANA DST handling)."""
        local_ts = ts.astimezone(self._tz) if ts.tzinfo is not None else ts.replace(tzinfo=self._tz)
        h = local_ts.hour
        cfg = self._config

        in_asian = _hour_in_window(h, cfg.asian_start_hour, cfg.asian_end_hour)
        in_london = _hour_in_window(h, cfg.london_start_hour, cfg.london_end_hour)
        in_ny = _hour_in_window(h, cfg.ny_start_hour, cfg.ny_end_hour)
        in_overlap = _hour_in_window(h, cfg.overlap_start_hour, cfg.overlap_end_hour)
        in_london_kz = _hour_in_window(h, cfg.london_killzone_start_hour, cfg.london_killzone_end_hour)
        in_ny_kz = _hour_in_window(h, cfg.ny_killzone_start_hour, cfg.ny_killzone_end_hour)

        if in_overlap:
            primary_session = "LONDON_NY_OVERLAP"
        elif in_london:
            primary_session = "LONDON"
        elif in_ny:
            primary_session = "NEW_YORK"
        elif in_asian:
            primary_session = "ASIAN"
        else:
            primary_session = "OFF_HOURS"

        active_killzone: str | None = None
        if in_london_kz:
            active_killzone = "LONDON_KILL_ZONE"
        elif in_ny_kz:
            active_killzone = "NEW_YORK_KILL_ZONE"

        dst_offset = local_ts.dst()
        is_dst = bool(dst_offset and dst_offset.total_seconds() != 0)

        return {
            "local_timestamp": local_ts.isoformat(),
            "local_hour": h,
            "timezone": cfg.timezone,
            "is_dst": is_dst,
            "primary_session": primary_session,
            "in_asian_session": in_asian,
            "in_london_session": in_london,
            "in_ny_session": in_ny,
            "in_london_ny_overlap": in_overlap,
            "in_london_killzone": in_london_kz,
            "in_ny_killzone": in_ny_kz,
            "active_killzone": active_killzone,
        }

    def evaluate(
        self,
        df: pd.DataFrame,
        as_of_index: int | None = None,
    ) -> StrategyEvaluationResult:
        sliced = self.prepare_causal_slice(df, as_of_index)
        last_row = sliced.iloc[-1]
        ts = self.extract_timestamp(last_row)

        info = self.classify_bar_time(ts)
        primary_session = str(info["primary_session"])
        active_kz = info["active_killzone"]

        # Track causal current-session high/low and most recent completed prior-session high/low
        timestamps = [
            t.to_pydatetime() if isinstance(t, pd.Timestamp) else pd.Timestamp(t).to_pydatetime()
            for t in sliced["timestamp"]
        ]
        sessions = [str(self.classify_bar_time(t)["primary_session"]) for t in timestamps]
        highs = sliced["high"].to_numpy(dtype="float64")
        lows = sliced["low"].to_numpy(dtype="float64")

        # Current contiguous session block
        curr_start = len(sliced) - 1
        while curr_start > 0 and sessions[curr_start - 1] == primary_session:
            curr_start -= 1

        curr_sess_high = Decimal(str(round(float(highs[curr_start:].max()), 4))).quantize(Decimal("0.01"))
        curr_sess_low = Decimal(str(round(float(lows[curr_start:].min()), 4))).quantize(Decimal("0.01"))
        info["current_session_high"] = str(curr_sess_high)
        info["current_session_low"] = str(curr_sess_low)

        # Prior contiguous session block (if available)
        prior_sess_high: Decimal | None = None
        prior_sess_low: Decimal | None = None
        prior_sess_name: str | None = None
        if curr_start > 0:
            prior_end = curr_start
            prior_sess_name = sessions[prior_end - 1]
            prior_start = prior_end - 1
            while prior_start > 0 and sessions[prior_start - 1] == prior_sess_name:
                prior_start -= 1
            prior_sess_high = Decimal(
                str(round(float(highs[prior_start:prior_end].max()), 4))
            ).quantize(Decimal("0.01"))
            prior_sess_low = Decimal(
                str(round(float(lows[prior_start:prior_end].min()), 4))
            ).quantize(Decimal("0.01"))
            info["prior_session_name"] = prior_sess_name
            info["prior_session_high"] = str(prior_sess_high)
            info["prior_session_low"] = str(prior_sess_low)

        bar_high = Decimal(str(last_row["high"])).quantize(Decimal("0.01"))
        bar_low = Decimal(str(last_row["low"])).quantize(Decimal("0.01"))
        bar_close = Decimal(str(last_row["close"])).quantize(Decimal("0.01"))

        swept_prior_low = (
            prior_sess_low is not None and bar_low < prior_sess_low and bar_close > prior_sess_low
        )
        swept_prior_high = (
            prior_sess_high is not None and bar_high > prior_sess_high and bar_close < prior_sess_high
        )
        info["swept_prior_session_low"] = swept_prior_low
        info["swept_prior_session_high"] = swept_prior_high

        if self._config.require_killzone_for_entry and active_kz is None:
            return self.build_wait_result(
                timestamp=ts,
                rationale=(
                    f"Outside London and New York Kill Zones (session={primary_session}); "
                    "require_killzone_for_entry=True blocks entry."
                ),
                rejection_reasons=["OUTSIDE_REQUIRED_KILLZONE"],
                supporting_evidence=info,
            )

        if swept_prior_low and active_kz is not None:
            return StrategyEvaluationResult(
                strategy_id=self.strategy_id,
                direction=SignalDirection.BUY,
                timestamp=ts,
                confidence=Decimal("0.80"),
                rationale=(
                    f"Active {active_kz}: swept prior {prior_sess_name} session low ({prior_sess_low}) "
                    f"and closed back above at {bar_close:.2f}."
                ),
                supporting_evidence=info,
                is_contextual_filter=True,
            )

        if swept_prior_high and active_kz is not None:
            return StrategyEvaluationResult(
                strategy_id=self.strategy_id,
                direction=SignalDirection.SELL,
                timestamp=ts,
                confidence=Decimal("0.80"),
                rationale=(
                    f"Active {active_kz}: swept prior {prior_sess_name} session high ({prior_sess_high}) "
                    f"and closed back below at {bar_close:.2f}."
                ),
                supporting_evidence=info,
                is_contextual_filter=True,
            )

        return self.build_wait_result(
            timestamp=ts,
            rationale=f"Session={primary_session}, KillZone={active_kz or 'NONE'}.",
            rejection_reasons=() if active_kz is not None else ("NOT_IN_KILLZONE",),
            supporting_evidence=info,
        )
