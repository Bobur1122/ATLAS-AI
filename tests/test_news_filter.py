"""Phase 3 tests for deterministic NewsFilter evaluation."""

from datetime import datetime, timezone
from decimal import Decimal

from atlas_ai.strategies.config import NewsFilterConfig
from atlas_ai.strategies.models import (
    EconomicEvent,
    NewsFilterStatus,
    NewsImpactLevel,
)
from atlas_ai.strategies.news_filter import NewsFilter


def test_news_filter_high_medium_low_and_unknown_handling() -> None:
    """Verify HIGH impact blackout, MEDIUM impact confidence reduction, LOW impact ignore, and UNKNOWN status."""
    nfp_event = EconomicEvent(
        event_name="US Non-Farm Payrolls",
        currency="USD",
        impact_level=NewsImpactLevel.HIGH,
        scheduled_time=datetime(2026, 1, 9, 13, 30, tzinfo=timezone.utc),
    )
    medium_event = EconomicEvent(
        event_name="US ISM Services",
        currency="USD",
        impact_level=NewsImpactLevel.MEDIUM,
        scheduled_time=datetime(2026, 1, 9, 15, 0, tzinfo=timezone.utc),
    )
    low_event = EconomicEvent(
        event_name="3-Month Bill Auction",
        currency="USD",
        impact_level=NewsImpactLevel.LOW,
        scheduled_time=datetime(2026, 1, 9, 16, 30, tzinfo=timezone.utc),
    )

    nf = NewsFilter()

    # 1. 15 mins before HIGH impact NFP -> BLOCKED_HIGH
    res_high = nf.evaluate(
        datetime(2026, 1, 9, 13, 15, tzinfo=timezone.utc),
        events=[nfp_event, medium_event, low_event],
    )
    assert res_high.status == NewsFilterStatus.BLOCKED_HIGH
    assert res_high.allow_new_entries is False
    assert res_high.confidence_multiplier == Decimal("0.0")

    # 2. 10 mins before MEDIUM impact -> RESTRICTED_MEDIUM with multiplier 0.75
    res_med = nf.evaluate(
        datetime(2026, 1, 9, 14, 50, tzinfo=timezone.utc),
        events=[nfp_event, medium_event, low_event],
    )
    assert res_med.status == NewsFilterStatus.RESTRICTED_MEDIUM
    assert res_med.allow_new_entries is True
    assert res_med.confidence_multiplier == Decimal("0.75")

    # 3. During LOW impact event -> CLEAR
    res_low = nf.evaluate(
        datetime(2026, 1, 9, 16, 30, tzinfo=timezone.utc),
        events=[nfp_event, medium_event, low_event],
    )
    assert res_low.status == NewsFilterStatus.CLEAR
    assert res_low.allow_new_entries is True

    # 4. Missing news data (`events=None`) -> UNKNOWN (and blocks when mandatory_news_data=True)
    res_unknown_opt = nf.evaluate(datetime(2026, 1, 9, 10, 0, tzinfo=timezone.utc), events=None)
    assert res_unknown_opt.status == NewsFilterStatus.UNKNOWN
    assert res_unknown_opt.allow_new_entries is True

    mandatory_nf = NewsFilter(NewsFilterConfig(mandatory_news_data=True))
    res_unknown_mand = mandatory_nf.evaluate(datetime(2026, 1, 9, 10, 0, tzinfo=timezone.utc), events=None)
    assert res_unknown_mand.status == NewsFilterStatus.UNKNOWN
    assert res_unknown_mand.allow_new_entries is False
