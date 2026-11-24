"""Phase 3 tests for SignalGenerator and StrategyEngine -> RiskEngine integration."""

from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd

from atlas_ai.domain.market import InstrumentSpec, Timeframe
from atlas_ai.domain.risk import AccountState, RiskParameters
from atlas_ai.domain.signals import SignalDirection
from atlas_ai.strategies.engine import StrategyEngine
from atlas_ai.strategies.models import (
    ConfluenceResult,
    RiskEvaluationStatus,
    StrategyEvaluationResult,
)
from atlas_ai.strategies.signal_generator import SignalGenerator
from tests.test_strategies import _make_trend_df

FIXED_TS = datetime(2026, 1, 6, 10, 0, tzinfo=timezone.utc)


def test_signal_generator_buy_sell_and_wait_fallback() -> None:
    """SignalGenerator must construct valid BUY/SELL TradeSignals and fall back to WAIT if stop/ATR is missing."""
    gen = SignalGenerator()
    bar = pd.Series({"close": 2650.00, "atr_14": 10.00})

    # 1. BUY with structural stop-loss from fvg_retest (2642.00) -> risk=8.00, TP=2666.00 (2.0 RR)
    buy_conf = ConfluenceResult(
        direction=SignalDirection.BUY,
        final_confidence=Decimal("0.82"),
        component_evaluations={
            "fvg_retest": StrategyEvaluationResult(
                strategy_id="fvg_retest",
                direction=SignalDirection.BUY,
                timestamp=FIXED_TS,
                entry_price=Decimal("2650.00"),
                stop_loss=Decimal("2642.00"),
                take_profit=Decimal("2666.00"),
                confidence=Decimal("0.84"),
                rationale="Bullish FVG retest.",
            )
        },
        rationale="Bullish confluence.",
    )
    buy_sig = gen.generate(
        symbol="XAUUSD",
        timeframe=Timeframe.H1,
        timestamp=FIXED_TS,
        current_bar=bar,
        confluence=buy_conf,
    )
    assert buy_sig.direction == SignalDirection.BUY
    assert buy_sig.entry_price == Decimal("2650.00")
    assert buy_sig.stop_loss == Decimal("2642.00")
    assert buy_sig.take_profit == Decimal("2666.00")

    # 2. Missing both structural SL and ATR -> must safely return WAIT instead of fabricating levels
    bar_no_atr = pd.Series({"close": 2650.00, "atr_14": float("nan")})
    conf_no_sl = ConfluenceResult(
        direction=SignalDirection.BUY,
        final_confidence=Decimal("0.80"),
        component_evaluations={},
        rationale="Bullish confluence without SL.",
    )
    wait_sig = gen.generate(
        symbol="XAUUSD",
        timeframe=Timeframe.H1,
        timestamp=FIXED_TS,
        current_bar=bar_no_atr,
        confluence=conf_no_sl,
    )
    assert wait_sig.direction == SignalDirection.WAIT


def test_strategy_engine_risk_integration_not_evaluated_approved_and_rejected() -> None:
    """Verify StrategyEngine sets NOT_EVALUATED when risk inputs are omitted, and RISK_APPROVED / RISK_REJECTED when supplied."""
    up_df = _make_trend_df("up", bars=80)
    engine = StrategyEngine()

    # 1. Without AccountState / RiskParameters / InstrumentSpec -> NOT_EVALUATED
    rep_no_risk = engine.analyze(up_df, symbol="XAUUSD", timeframe=Timeframe.H1, register_in_lifecycle=False)
    assert rep_no_risk.signal.direction == SignalDirection.BUY
    assert rep_no_risk.risk_status == RiskEvaluationStatus.NOT_EVALUATED
    assert rep_no_risk.risk_decision is None

    # 2. With valid AccountState, RiskParameters, and InstrumentSpec -> RISK_APPROVED
    spec = InstrumentSpec(
        symbol="XAUUSD",
        contract_size=Decimal("100"),
        tick_size=Decimal("0.01"),
        tick_value=Decimal("1.00"),
        volume_step=Decimal("0.01"),
        minimum_volume=Decimal("0.01"),
        maximum_volume=Decimal("50.0"),
    )
    account = AccountState(
        balance=Decimal("10000.00"),
        equity=Decimal("10000.00"),
        free_margin=Decimal("10000.00"),
        daily_starting_balance=Decimal("10000.00"),
    )
    params = RiskParameters()

    rep_approved = engine.analyze(
        up_df,
        symbol="XAUUSD",
        timeframe=Timeframe.H1,
        account_state=account,
        risk_parameters=params,
        instrument_spec=spec,
    )
    assert rep_approved.risk_status == RiskEvaluationStatus.RISK_APPROVED
    assert rep_approved.risk_decision is not None and rep_approved.risk_decision.approved is True
    assert rep_approved.lifecycle_record is not None
    assert rep_approved.lifecycle_record.risk_status == RiskEvaluationStatus.RISK_APPROVED

    # 3. With allow_trading=False in RiskParameters -> RISK_REJECTED
    disabled_params = params.model_copy(update={"allow_trading": False})
    engine_rej = StrategyEngine()
    rep_rejected = engine_rej.analyze(
        up_df,
        symbol="XAUUSD",
        timeframe=Timeframe.H1,
        account_state=account,
        risk_parameters=disabled_params,
        instrument_spec=spec,
    )
    assert rep_rejected.risk_status == RiskEvaluationStatus.RISK_REJECTED
    assert rep_rejected.risk_decision is not None and rep_rejected.risk_decision.approved is False
