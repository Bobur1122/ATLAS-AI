"""Time-series integrity and missing-candle detection for ATLAS AI."""

from datetime import datetime, timedelta

import pandas as pd

from atlas_ai.data.models import (
    DataValidationIssue,
    DataValidationIssueCode,
    MarketCalendarConfig,
    MissingBarGap,
    MissingBarPolicy,
)
from atlas_ai.domain.market import Timeframe

TIMEFRAME_DELTAS: dict[Timeframe, timedelta] = {
    Timeframe.M1: timedelta(minutes=1),
    Timeframe.M5: timedelta(minutes=5),
    Timeframe.M15: timedelta(minutes=15),
    Timeframe.M30: timedelta(minutes=30),
    Timeframe.H1: timedelta(hours=1),
    Timeframe.H4: timedelta(hours=4),
    Timeframe.D1: timedelta(days=1),
    Timeframe.W1: timedelta(weeks=1),
    Timeframe.MN1: timedelta(days=30),
}


def get_timeframe_delta(timeframe: Timeframe) -> timedelta:
    """Return the expected chronological timedelta for a given Timeframe."""
    return TIMEFRAME_DELTAS[timeframe]


def check_timeseries_integrity(
    df: pd.DataFrame,
    timeframe: Timeframe,
    calendar: MarketCalendarConfig | None = None,
    missing_bar_policy: MissingBarPolicy = MissingBarPolicy.STRICT,
) -> tuple[list[DataValidationIssue], list[MissingBarGap], int]:
    """Inspect a timestamp-indexed or `timestamp`-column DataFrame for chronological integrity.

    Detects:
    - Duplicate timestamps (`DUPLICATE_TIMESTAMP`)
    - Out-of-order / non-monotonic timestamps (`OUT_OF_ORDER_TIMESTAMP`)
    - Unexpected timeframe intervals not aligned to the bar step (`UNEXPECTED_TIMEFRAME_INTERVAL`)
    - Missing expected open-market bars (`MISSING_EXPECTED_BAR`) while respecting legitimate
      weekend closures, configured daily rollover breaks, and explicit market holidays.

    Args:
        df: DataFrame containing a timezone-aware `timestamp` series.
        timeframe: Expected `Timeframe` of the series.
        calendar: MarketCalendarConfig defining weekend/holiday/break closures.
        missing_bar_policy: STRICT (missing bars are errors), REPORT (missing bars recorded as
            non-fatal issues), or ALLOW (gaps recorded in `gaps` metadata without raising issues).

    Returns:
        Tuple of `(issues, gaps, total_missing_bars)`.
    """
    active_calendar = calendar or MarketCalendarConfig()
    issues: list[DataValidationIssue] = []
    gaps: list[MissingBarGap] = []
    total_missing_bars = 0

    if df.empty or len(df) < 2:
        return issues, gaps, total_missing_bars

    expected_delta = get_timeframe_delta(timeframe)
    expected_seconds = int(expected_delta.total_seconds())

    timestamps: list[datetime] = [ts.to_pydatetime() for ts in df["timestamp"]]

    seen_timestamps: set[datetime] = {timestamps[0]}

    for idx in range(1, len(timestamps)):
        prev_ts = timestamps[idx - 1]
        curr_ts = timestamps[idx]

        # 1. Duplicate check
        if curr_ts == prev_ts or curr_ts in seen_timestamps:
            issues.append(
                DataValidationIssue(
                    code=DataValidationIssueCode.DUPLICATE_TIMESTAMP,
                    message=f"Duplicate timestamp detected at row {idx}: {curr_ts.isoformat()}.",
                    row_index=idx,
                    timestamp=curr_ts,
                    is_error=True,
                )
            )
            continue

        seen_timestamps.add(curr_ts)

        # 2. Monotonic increasing check
        if curr_ts < prev_ts:
            issues.append(
                DataValidationIssue(
                    code=DataValidationIssueCode.OUT_OF_ORDER_TIMESTAMP,
                    message=(
                        f"Out-of-order timestamp at row {idx}: {curr_ts.isoformat()} "
                        f"is earlier than previous timestamp {prev_ts.isoformat()}."
                    ),
                    row_index=idx,
                    timestamp=curr_ts,
                    is_error=True,
                )
            )
            continue

        # Skip fixed-step gap walking for monthly bars
        if timeframe == Timeframe.MN1:
            continue

        diff = curr_ts - prev_ts
        diff_seconds = int(diff.total_seconds())

        # 3. Check if interval is an exact multiple of the timeframe delta
        if diff_seconds % expected_seconds != 0:
            issues.append(
                DataValidationIssue(
                    code=DataValidationIssueCode.UNEXPECTED_TIMEFRAME_INTERVAL,
                    message=(
                        f"Unexpected interval of {diff} between {prev_ts.isoformat()} and "
                        f"{curr_ts.isoformat()} (not a multiple of {timeframe.value} = {expected_delta})."
                    ),
                    row_index=idx,
                    timestamp=curr_ts,
                    is_error=True,
                )
            )
            continue

        # 4. If more than 1 step elapsed, count how many intermediate steps were open market hours
        steps = diff_seconds // expected_seconds
        if steps > 1:
            # Cap step walking at 10,000 steps to prevent DoS on multi-year gaps in M1 data
            max_walk_steps = min(steps - 1, 10000)
            missing_open_bars = 0
            first_missing_ts: datetime | None = None

            for step_idx in range(1, max_walk_steps + 1):
                candidate_ts = prev_ts + (expected_delta * step_idx)
                if active_calendar.is_market_open(candidate_ts):
                    missing_open_bars += 1
                    if first_missing_ts is None:
                        first_missing_ts = candidate_ts

            if missing_open_bars > 0 and first_missing_ts is not None:
                total_missing_bars += missing_open_bars
                gap = MissingBarGap(
                    previous_timestamp=prev_ts,
                    next_timestamp=curr_ts,
                    missing_bars_count=missing_open_bars,
                    first_missing_timestamp=first_missing_ts,
                )
                gaps.append(gap)

                if missing_bar_policy == MissingBarPolicy.STRICT:
                    issues.append(
                        DataValidationIssue(
                            code=DataValidationIssueCode.MISSING_EXPECTED_BAR,
                            message=(
                                f"Missing {missing_open_bars} expected {timeframe.value} bar(s) between "
                                f"{prev_ts.isoformat()} and {curr_ts.isoformat()} "
                                f"(first missing: {first_missing_ts.isoformat()})."
                            ),
                            row_index=idx,
                            timestamp=curr_ts,
                            is_error=True,
                        )
                    )
                elif missing_bar_policy == MissingBarPolicy.REPORT:
                    issues.append(
                        DataValidationIssue(
                            code=DataValidationIssueCode.MISSING_EXPECTED_BAR,
                            message=(
                                f"Reported gap of {missing_open_bars} expected {timeframe.value} bar(s) between "
                                f"{prev_ts.isoformat()} and {curr_ts.isoformat()} "
                                f"(first missing: {first_missing_ts.isoformat()})."
                            ),
                            row_index=idx,
                            timestamp=curr_ts,
                            is_error=False,
                        )
                    )
                # Under MissingBarPolicy.ALLOW, the gap is recorded in `gaps` without adding an issue.

    return issues, gaps, total_missing_bars
