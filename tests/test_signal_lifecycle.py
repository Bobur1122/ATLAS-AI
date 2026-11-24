"""Phase 3 tests for SignalLifecycleManager state transitions, duplicate prevention, expiration, and SL/TP tracking."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from atlas_ai.domain.market import Timeframe
from atlas_ai.domain.risk import RiskDecision
from atlas_ai.domain.signals import SignalDirection, TradeSignal
from atlas_ai.strategies.exceptions import SignalLifecycleError
from atlas_ai.strategies.models import RiskEvaluationStatus, SignalLifecycleState
from atlas_ai.strategies.signal_lifecycle import SignalLifecycleManager

FIXED_TS = datetime(2026, 1, 6, 10, 0, tzinfo=timezone.utc)


def _sample_buy_signal(entry: str = "2650.00") -> TradeSignal:
    e = Decimal(entry)
    return TradeSignal(
        symbol="XAUUSD",
        direction=SignalDirection.BUY,
        entry_price=e,
        stop_loss=e - Decimal("10.00"),
        take_profit=e + Decimal("20.00"),
        timeframe=Timeframe.H1,
        strategy_id="test_strat",
        confidence=Decimal("0.82"),
        timestamp=FIXED_TS,
        rationale="Synthetic test BUY signal.",
    )


def test_signal_lifecycle_prevents_activation_without_risk_approval() -> None:
    """A GENERATED or CONFIRMED signal must never transition to ACTIVE without passing RISK_APPROVED."""
    mgr = SignalLifecycleManager()
    sig = _sample_buy_signal()
    rec = mgr.register_signal(sig)
    assert rec.state == SignalLifecycleState.GENERATED
    assert rec.risk_status == RiskEvaluationStatus.NOT_EVALUATED

    # Attempting to transition directly from GENERATED to ACTIVE must raise SignalLifecycleError
    with pytest.raises(SignalLifecycleError, match="Invalid signal state transition"):
        mgr.transition(rec.signal_id, SignalLifecycleState.ACTIVE, FIXED_TS, "Illegal jump to ACTIVE")

    # Apply RiskDecision(approved=True) -> transitions to RISK_APPROVED
    approved_decision = RiskDecision(
        approved=True,
        calculated_risk_amount=Decimal("100.00"),
        calculated_position_size=Decimal("0.10"),
        risk_reward_ratio=Decimal("2.0"),
        risk_percent=Decimal("1.0"),
    )
    rec = mgr.apply_risk_decision(rec.signal_id, approved_decision, FIXED_TS + timedelta(minutes=1))
    assert rec.state == SignalLifecycleState.RISK_APPROVED
    assert rec.risk_status == RiskEvaluationStatus.RISK_APPROVED

    # Now transition to ACTIVE succeeds
    rec = mgr.transition(rec.signal_id, SignalLifecycleState.ACTIVE, FIXED_TS + timedelta(minutes=2), "Order filled")
    assert rec.state == SignalLifecycleState.ACTIVE

    # Simulate price hitting Take Profit (2670.00) -> TARGET_HIT
    rec = mgr.evaluate_market_update(
        rec.signal_id,
        current_timestamp=FIXED_TS + timedelta(minutes=30),
        high_price=Decimal("2672.00"),
        low_price=Decimal("2655.00"),
    )
    assert rec.state == SignalLifecycleState.TARGET_HIT
    assert len(rec.history) == 4


def test_signal_lifecycle_duplicate_prevention_and_expiration() -> None:
    """SignalLifecycleManager must block duplicate active setups and expire stale non-active signals."""
    mgr = SignalLifecycleManager()
    sig1 = _sample_buy_signal("2650.00")
    rec1 = mgr.register_signal(sig1, expiration_minutes=60)

    # Registering another BUY signal on XAUUSD H1 within entry_zone_dedup_tolerance ($2.00) must fail
    sig_dup = _sample_buy_signal("2651.00")
    with pytest.raises(SignalLifecycleError, match="Duplicate active signal"):
        mgr.register_signal(sig_dup)

    # Advance time past expiration_minutes (60 mins) -> transitions to EXPIRED
    expired_rec = mgr.evaluate_market_update(
        rec1.signal_id,
        current_timestamp=FIXED_TS + timedelta(minutes=61),
        high_price=Decimal("2655.00"),
        low_price=Decimal("2645.00"),
    )
    assert expired_rec.state == SignalLifecycleState.EXPIRED

    # Once expired, a new signal at the same level is allowed
    rec2 = mgr.register_signal(sig_dup)
    assert rec2.state == SignalLifecycleState.GENERATED
