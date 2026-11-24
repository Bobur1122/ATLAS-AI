"""Deterministic Economic Calendar News Filter Interface for ATLAS AI."""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Sequence
from zoneinfo import ZoneInfo

from atlas_ai.strategies.config import NewsFilterConfig
from atlas_ai.strategies.models import (
    EconomicEvent,
    NewsFilterResult,
    NewsFilterStatus,
    NewsImpactLevel,
)


class NewsFilter:
    """Evaluates supplied structured economic events against configurable blackout windows.

    Rules:
    1. Does not fetch or invent live external news data; operates deterministically on supplied `events`.
    2. If `events is None` (missing news feed), status is explicitly reported as `NewsFilterStatus.UNKNOWN`
       (never `CLEAR`). If `config.mandatory_news_data` is `True`, `UNKNOWN` blocks trade approval.
    3. If `events` is an empty sequence `[]` (explicitly verified no scheduled events), status is `CLEAR`.
    4. HIGH-impact events within `[-minutes_before_high_impact, +minutes_after_high_impact]` block new entries.
    5. MEDIUM-impact events within `[-minutes_before_medium_impact, +minutes_after_medium_impact]` either
       block entries (if `block_on_medium_impact=True`) or reduce confidence via `medium_impact_confidence_multiplier`.
    6. LOW-impact events are ignored by default.
    """

    def __init__(self, config: NewsFilterConfig | None = None) -> None:
        self._config = config or NewsFilterConfig()

    @property
    def config(self) -> NewsFilterConfig:
        return self._config

    def evaluate(
        self,
        evaluation_timestamp: datetime,
        events: Sequence[EconomicEvent] | None = None,
    ) -> NewsFilterResult:
        """Deterministically evaluate news blackout status at `evaluation_timestamp`."""
        cfg = self._config
        if not cfg.enabled:
            return NewsFilterResult(
                status=NewsFilterStatus.CLEAR,
                allow_new_entries=True,
                confidence_multiplier=Decimal("1.0"),
                active_events=(),
                rationale="NewsFilter is disabled by configuration.",
            )

        # Rule 4 & 5: Missing news data is UNKNOWN, never NO_NEWS/CLEAR
        if events is None:
            allow = not cfg.mandatory_news_data
            mult = Decimal("1.0") if allow else Decimal("0.0")
            return NewsFilterResult(
                status=NewsFilterStatus.UNKNOWN,
                allow_new_entries=allow,
                confidence_multiplier=mult,
                active_events=(),
                rationale=(
                    "News event data is missing (UNKNOWN). "
                    + (
                        "Trading blocked because mandatory_news_data=True."
                        if not allow
                        else "Trading permitted because mandatory_news_data=False."
                    )
                ),
            )

        eval_utc = (
            evaluation_timestamp.astimezone(ZoneInfo("UTC"))
            if evaluation_timestamp.tzinfo is not None
            else evaluation_timestamp.replace(tzinfo=ZoneInfo("UTC"))
        )

        active_high: list[EconomicEvent] = []
        active_medium: list[EconomicEvent] = []

        for ev in events:
            if ev.currency not in cfg.monitored_currencies:
                continue

            ev_tz = ZoneInfo(ev.timezone)
            ev_dt = (
                ev.scheduled_time.astimezone(ZoneInfo("UTC"))
                if ev.scheduled_time.tzinfo is not None
                else ev.scheduled_time.replace(tzinfo=ev_tz).astimezone(ZoneInfo("UTC"))
            )

            if ev.impact_level == NewsImpactLevel.HIGH:
                start_win = ev_dt - timedelta(minutes=cfg.minutes_before_high_impact)
                end_win = ev_dt + timedelta(minutes=cfg.minutes_after_high_impact)
                if start_win <= eval_utc <= end_win:
                    active_high.append(ev)

            elif ev.impact_level == NewsImpactLevel.MEDIUM:
                start_win = ev_dt - timedelta(minutes=cfg.minutes_before_medium_impact)
                end_win = ev_dt + timedelta(minutes=cfg.minutes_after_medium_impact)
                if start_win <= eval_utc <= end_win:
                    active_medium.append(ev)

        if active_high and cfg.block_on_high_impact:
            names = ", ".join(f"{e.currency} {e.event_name}" for e in active_high)
            return NewsFilterResult(
                status=NewsFilterStatus.BLOCKED_HIGH,
                allow_new_entries=False,
                confidence_multiplier=Decimal("0.0"),
                active_events=tuple(active_high + active_medium),
                rationale=f"HIGH-impact news blackout active for: {names}.",
            )

        if active_medium:
            names = ", ".join(f"{e.currency} {e.event_name}" for e in active_medium)
            if cfg.block_on_medium_impact:
                return NewsFilterResult(
                    status=NewsFilterStatus.RESTRICTED_MEDIUM,
                    allow_new_entries=False,
                    confidence_multiplier=Decimal("0.0"),
                    active_events=tuple(active_medium),
                    rationale=f"MEDIUM-impact news blackout blocking entries for: {names}.",
                )
            return NewsFilterResult(
                status=NewsFilterStatus.RESTRICTED_MEDIUM,
                allow_new_entries=True,
                confidence_multiplier=cfg.medium_impact_confidence_multiplier,
                active_events=tuple(active_medium),
                rationale=(
                    f"MEDIUM-impact news window active ({names}); confidence reduced by multiplier "
                    f"{cfg.medium_impact_confidence_multiplier}."
                ),
            )

        return NewsFilterResult(
            status=NewsFilterStatus.CLEAR,
            allow_new_entries=True,
            confidence_multiplier=Decimal("1.0"),
            active_events=(),
            rationale="No HIGH or MEDIUM impact news events within blackout window.",
        )
