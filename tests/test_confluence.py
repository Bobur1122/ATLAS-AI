"""Phase 3 tests for ConfluenceEngine weighted scoring, conflict detection, contextual filters, and news overrides."""

from datetime import datetime, timezone
from decimal import Decimal

from atlas_ai.domain.signals import SignalDirection
from atlas_ai.strategies.config import ConfluenceWeightsConfig
from atlas_ai.strategies.confluence import ConfluenceEngine
from atlas_ai.strategies.models import (
    NewsFilterResult,
    NewsFilterStatus,
    StrategyEvaluationResult,
)

FIXED_TS = datetime(2026, 1, 6, 10, 0, tzinfo=timezone.utc)


def _eval(strategy_id: str, direction: SignalDirection, conf: str = "0.80", contextual: bool = False) -> StrategyEvaluationResult:
    return StrategyEvaluationResult(
        strategy_id=strategy_id,
        direction=direction,
        timestamp=FIXED_TS,
        confidence=Decimal(conf),
        rationale=f"{strategy_id} voted {direction.value}",
        is_contextual_filter=contextual,
    )


def test_confluence_engine_approves_strong_aligned_evidence() -> None:
    """ConfluenceEngine must approve BUY when primary strategies agree and contextual filters align."""
    engine = ConfluenceEngine()
    evals = {
        "ema_trend_following": _eval("ema_trend_following", SignalDirection.BUY, "0.80"),
        "smc_market_structure": _eval("smc_market_structure", SignalDirection.BUY, "0.85"),
        "liquidity_sweep": _eval("liquidity_sweep", SignalDirection.BUY, "0.85"),
        "fvg_detection": _eval("fvg_detection", SignalDirection.BUY, "0.70"),
        "fvg_retest": _eval("fvg_retest", SignalDirection.BUY, "0.85"),
        "multi_timeframe_analyzer": _eval("multi_timeframe_analyzer", SignalDirection.BUY, "0.80"),
        "premium_discount_analyzer": _eval("premium_discount_analyzer", SignalDirection.BUY, "0.70", contextual=True),
        "pdh_pdl_analyzer": _eval("pdh_pdl_analyzer", SignalDirection.BUY, "0.80", contextual=True),
        "killzone_session_analyzer": _eval("killzone_session_analyzer", SignalDirection.BUY, "0.80", contextual=True),
    }
    res = engine.evaluate(evaluations=evals)
    assert res.direction == SignalDirection.BUY
    assert res.final_confidence >= Decimal("0.80")
    assert res.directional_agreement_count == 5  # fvg_detection deduplicated when fvg_retest also fires


def test_confluence_engine_returns_wait_on_contradictory_evidence() -> None:
    """ConfluenceEngine must return WAIT and record conflicts when BUY and SELL scores are balanced."""
    engine = ConfluenceEngine()
    evals = {
        "ema_trend_following": _eval("ema_trend_following", SignalDirection.BUY, "0.80"),
        "smc_market_structure": _eval("smc_market_structure", SignalDirection.BUY, "0.80"),
        "liquidity_sweep": _eval("liquidity_sweep", SignalDirection.SELL, "0.80"),
        "fvg_retest": _eval("fvg_retest", SignalDirection.SELL, "0.80"),
        "fvg_detection": _eval("fvg_detection", SignalDirection.WAIT, "0.0"),
        "multi_timeframe_analyzer": _eval("multi_timeframe_analyzer", SignalDirection.WAIT, "0.0"),
    }
    res = engine.evaluate(evaluations=evals)
    assert res.direction == SignalDirection.WAIT
    assert len(res.conflicts) >= 1
    assert "CONFLICTING_EVIDENCE_BELOW_EDGE_THRESHOLD" in res.rejection_reasons


def test_contextual_filters_alone_never_trigger_trade_and_news_blackout_overrides() -> None:
    """Contextual filters (Premium/Discount, PDH/PDL) cannot trigger a trade without primary directional signals,
    and HIGH-impact news blackout overrides strong primary confluence.
    """
    engine = ConfluenceEngine()

    # 1. Only contextual filters vote BUY while all primary modules vote WAIT -> must return WAIT
    ctx_only = {
        "ema_trend_following": _eval("ema_trend_following", SignalDirection.WAIT, "0.0"),
        "smc_market_structure": _eval("smc_market_structure", SignalDirection.WAIT, "0.0"),
        "liquidity_sweep": _eval("liquidity_sweep", SignalDirection.WAIT, "0.0"),
        "fvg_detection": _eval("fvg_detection", SignalDirection.WAIT, "0.0"),
        "fvg_retest": _eval("fvg_retest", SignalDirection.WAIT, "0.0"),
        "multi_timeframe_analyzer": _eval("multi_timeframe_analyzer", SignalDirection.WAIT, "0.0"),
        "premium_discount_analyzer": _eval("premium_discount_analyzer", SignalDirection.BUY, "0.90", contextual=True),
        "pdh_pdl_analyzer": _eval("pdh_pdl_analyzer", SignalDirection.BUY, "0.90", contextual=True),
    }
    res_ctx = engine.evaluate(evaluations=ctx_only)
    assert res_ctx.direction == SignalDirection.WAIT

    # 2. Strong primary BUY overridden by NewsFilter BLOCKED_HIGH
    strong_buy = {
        "ema_trend_following": _eval("ema_trend_following", SignalDirection.BUY, "0.90"),
        "smc_market_structure": _eval("smc_market_structure", SignalDirection.BUY, "0.90"),
        "liquidity_sweep": _eval("liquidity_sweep", SignalDirection.BUY, "0.90"),
        "fvg_retest": _eval("fvg_retest", SignalDirection.BUY, "0.90"),
        "multi_timeframe_analyzer": _eval("multi_timeframe_analyzer", SignalDirection.BUY, "0.90"),
    }
    blocked_news = NewsFilterResult(
        status=NewsFilterStatus.BLOCKED_HIGH,
        allow_new_entries=False,
        confidence_multiplier=Decimal("0.0"),
        rationale="Blocked by FOMC.",
    )
    res_blocked = engine.evaluate(evaluations=strong_buy, news_result=blocked_news)
    assert res_blocked.direction == SignalDirection.WAIT
    assert "NEWS_FILTER_BLOCKED:BLOCKED_HIGH" in res_blocked.rejection_reasons
