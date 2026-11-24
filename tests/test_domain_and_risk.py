"""Comprehensive Phase 1 unit tests for domain models, position sizing, and Deterministic RiskEngine."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from atlas_ai.domain.market import Candle, InstrumentSpec, Timeframe
from atlas_ai.domain.risk import (
    AccountState,
    RiskParameters,
    RiskRejectionCode,
)
from atlas_ai.domain.signals import SignalDirection, TradeSignal
from atlas_ai.risk.engine import RiskEngine, calculate_position_size

FIXED_TS = datetime(2026, 11, 15, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def xauusd_spec() -> InstrumentSpec:
    """Standard broker specification for XAUUSD (100 oz contract, $0.01 tick = $1.00/lot)."""
    return InstrumentSpec(
        symbol="XAUUSD",
        contract_size=Decimal("100"),
        tick_size=Decimal("0.01"),
        tick_value=Decimal("1.00"),
        volume_step=Decimal("0.01"),
        minimum_volume=Decimal("0.01"),
        maximum_volume=Decimal("50.00"),
        margin_rate=Decimal("0.01"),  # 1:100 leverage
    )


@pytest.fixture
def healthy_account() -> AccountState:
    """Standard $10,000 funded account state with no open positions or losses."""
    return AccountState(
        balance=Decimal("10000.00"),
        equity=Decimal("10000.00"),
        free_margin=Decimal("10000.00"),
        currency="USD",
        open_positions_count=0,
        current_daily_pnl=Decimal("0.00"),
        daily_starting_balance=Decimal("10000.00"),
        peak_balance=Decimal("10000.00"),
    )


@pytest.fixture
def default_risk_params() -> RiskParameters:
    """Standard risk parameters: 1% risk/trade, 3% max daily loss, 10% max DD, min 1.5 RR."""
    return RiskParameters(
        risk_per_trade_percent=Decimal("1.0"),
        max_daily_loss_percent=Decimal("3.0"),
        max_total_drawdown_percent=Decimal("10.0"),
        max_open_positions=3,
        minimum_risk_reward_ratio=Decimal("1.5"),
        max_position_size=Decimal("10.0"),
        allow_trading=True,
    )


@pytest.fixture
def valid_buy_signal() -> TradeSignal:
    """Valid BUY signal on XAUUSD: entry=2650.00, SL=2640.00 ($10 risk), TP=2670.00 ($20 reward, RR=2.0)."""
    return TradeSignal(
        symbol="XAUUSD",
        direction=SignalDirection.BUY,
        entry_price=Decimal("2650.00"),
        stop_loss=Decimal("2640.00"),
        take_profit=Decimal("2670.00"),
        timeframe=Timeframe.H1,
        strategy_id="london_breakout_v1",
        confidence=Decimal("0.82"),
        timestamp=FIXED_TS,
        rationale="Bullish H1 structure break above session high with clean retest.",
    )


@pytest.fixture
def valid_sell_signal() -> TradeSignal:
    """Valid SELL signal on XAUUSD: entry=2650.00, SL=2660.00 ($10 risk), TP=2630.00 ($20 reward, RR=2.0)."""
    return TradeSignal(
        symbol="XAUUSD",
        direction=SignalDirection.SELL,
        entry_price=Decimal("2650.00"),
        stop_loss=Decimal("2660.00"),
        take_profit=Decimal("2630.00"),
        timeframe=Timeframe.M15,
        strategy_id="ny_reversal_v1",
        confidence=Decimal("0.75"),
        timestamp=FIXED_TS,
        rationale="Bearish liquidity sweep and displacement below M15 support.",
    )


# ============================================================================
# 1. CANDLE DOMAIN VALIDATION TESTS
# ============================================================================


def test_valid_candle_creation_and_normalization() -> None:
    """Valid OHLCV candle must normalize symbol and preserve exact Decimal prices."""
    candle = Candle(
        symbol=" xauusd ",
        timeframe=Timeframe.M15,
        timestamp=FIXED_TS,
        open=Decimal("2650.10"),
        high=Decimal("2655.50"),
        low=Decimal("2648.00"),
        close=Decimal("2653.25"),
        volume=Decimal("1420"),
    )
    assert candle.symbol == "XAUUSD"
    assert candle.timeframe == Timeframe.M15
    assert candle.high >= max(candle.open, candle.close)
    assert candle.low <= min(candle.open, candle.close)
    assert candle.volume == Decimal("1420")

    # Flat doji candle (open == high == low == close) with zero volume is also valid
    doji = Candle(
        symbol="XAUUSD",
        timeframe=Timeframe.M1,
        timestamp=FIXED_TS,
        open=Decimal("2650.00"),
        high=Decimal("2650.00"),
        low=Decimal("2650.00"),
        close=Decimal("2650.00"),
        volume=Decimal("0"),
    )
    assert doji.high == doji.low


@pytest.mark.parametrize(
    ("open_p", "high_p", "low_p", "close_p", "vol"),
    [
        # Negative or zero prices
        (Decimal("0"), Decimal("2655"), Decimal("2648"), Decimal("2650"), Decimal("100")),
        (Decimal("-2650"), Decimal("2655"), Decimal("2648"), Decimal("2650"), Decimal("100")),
        # High < Open
        (Decimal("2656.00"), Decimal("2655.00"), Decimal("2648.00"), Decimal("2650.00"), Decimal("100")),
        # High < Close
        (Decimal("2650.00"), Decimal("2655.00"), Decimal("2648.00"), Decimal("2656.00"), Decimal("100")),
        # Low > Open
        (Decimal("2647.00"), Decimal("2655.00"), Decimal("2648.00"), Decimal("2650.00"), Decimal("100")),
        # Low > Close
        (Decimal("2650.00"), Decimal("2655.00"), Decimal("2648.00"), Decimal("2647.00"), Decimal("100")),
        # High < Low
        (Decimal("2650.00"), Decimal("2645.00"), Decimal("2648.00"), Decimal("2650.00"), Decimal("100")),
        # Negative volume
        (Decimal("2650.00"), Decimal("2655.00"), Decimal("2648.00"), Decimal("2652.00"), Decimal("-1")),
    ],
)
def test_invalid_candle_rejected(
    open_p: Decimal,
    high_p: Decimal,
    low_p: Decimal,
    close_p: Decimal,
    vol: Decimal,
) -> None:
    """Candles violating positivity, OHLC ordering, or non-negative volume must be rejected."""
    with pytest.raises(ValidationError):
        Candle(
            symbol="XAUUSD",
            timeframe=Timeframe.H1,
            timestamp=FIXED_TS,
            open=open_p,
            high=high_p,
            low=low_p,
            close=close_p,
            volume=vol,
        )


# ============================================================================
# 2. TRADE SIGNAL DOMAIN VALIDATION TESTS
# ============================================================================


def test_valid_buy_sell_and_wait_signals(
    valid_buy_signal: TradeSignal,
    valid_sell_signal: TradeSignal,
) -> None:
    """BUY, SELL, and WAIT signals must validate geometry and expose helper properties."""
    assert valid_buy_signal.is_executable is True
    assert valid_buy_signal.stop_loss_distance == Decimal("10.00")
    assert valid_buy_signal.take_profit_distance == Decimal("20.00")

    assert valid_sell_signal.is_executable is True
    assert valid_sell_signal.stop_loss_distance == Decimal("10.00")
    assert valid_sell_signal.take_profit_distance == Decimal("20.00")

    wait_signal = TradeSignal(
        symbol="xauusd",
        direction=SignalDirection.WAIT,
        timeframe=Timeframe.H1,
        strategy_id="regime_filter",
        confidence=Decimal("0.40"),
        timestamp=FIXED_TS,
        rationale="Choppy Asian range; no high-probability setup.",
    )
    assert wait_signal.symbol == "XAUUSD"
    assert wait_signal.is_executable is False
    assert wait_signal.stop_loss_distance is None


@pytest.mark.parametrize(
    ("direction", "entry", "sl", "tp", "confidence"),
    [
        # BUY with stop_loss >= entry_price
        (SignalDirection.BUY, Decimal("2650"), Decimal("2650"), Decimal("2670"), Decimal("0.8")),
        (SignalDirection.BUY, Decimal("2650"), Decimal("2655"), Decimal("2670"), Decimal("0.8")),
        # BUY with take_profit <= entry_price
        (SignalDirection.BUY, Decimal("2650"), Decimal("2640"), Decimal("2650"), Decimal("0.8")),
        (SignalDirection.BUY, Decimal("2650"), Decimal("2640"), Decimal("2645"), Decimal("0.8")),
        # SELL with stop_loss <= entry_price
        (SignalDirection.SELL, Decimal("2650"), Decimal("2650"), Decimal("2630"), Decimal("0.8")),
        (SignalDirection.SELL, Decimal("2650"), Decimal("2645"), Decimal("2630"), Decimal("0.8")),
        # SELL with take_profit >= entry_price
        (SignalDirection.SELL, Decimal("2650"), Decimal("2660"), Decimal("2650"), Decimal("0.8")),
        (SignalDirection.SELL, Decimal("2650"), Decimal("2660"), Decimal("2655"), Decimal("0.8")),
        # Missing stop_loss or take_profit on BUY/SELL
        (SignalDirection.BUY, Decimal("2650"), None, Decimal("2670"), Decimal("0.8")),
        (SignalDirection.SELL, Decimal("2650"), Decimal("2660"), None, Decimal("0.8")),
        # Confidence out of [0, 1]
        (SignalDirection.BUY, Decimal("2650"), Decimal("2640"), Decimal("2670"), Decimal("-0.01")),
        (SignalDirection.BUY, Decimal("2650"), Decimal("2640"), Decimal("2670"), Decimal("1.01")),
    ],
)
def test_invalid_trade_signals_rejected(
    direction: SignalDirection,
    entry: Decimal | None,
    sl: Decimal | None,
    tp: Decimal | None,
    confidence: Decimal,
) -> None:
    """Signals with inverted SL/TP, missing prices, or out-of-range confidence must be rejected."""
    with pytest.raises(ValidationError):
        TradeSignal(
            symbol="XAUUSD",
            direction=direction,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            timeframe=Timeframe.H1,
            strategy_id="test_strat",
            confidence=confidence,
            timestamp=FIXED_TS,
            rationale="Test rationale",
        )


# ============================================================================
# 3. ACCOUNT STATE & RISK PARAMETERS VALIDATION TESTS
# ============================================================================


def test_invalid_account_states_rejected() -> None:
    """AccountState must reject non-positive balance/equity, negative margin/positions, or free_margin > equity."""
    # Zero balance
    with pytest.raises(ValidationError):
        AccountState(
            balance=Decimal("0"),
            equity=Decimal("10000"),
            free_margin=Decimal("5000"),
            daily_starting_balance=Decimal("10000"),
        )

    # Free margin greater than equity
    with pytest.raises(ValidationError, match="free_margin"):
        AccountState(
            balance=Decimal("10000"),
            equity=Decimal("9000"),
            free_margin=Decimal("9500"),
            daily_starting_balance=Decimal("10000"),
        )

    # Negative open positions
    with pytest.raises(ValidationError):
        AccountState(
            balance=Decimal("10000"),
            equity=Decimal("10000"),
            free_margin=Decimal("10000"),
            open_positions_count=-1,
            daily_starting_balance=Decimal("10000"),
        )

    # Invalid currency code
    with pytest.raises(ValidationError):
        AccountState(
            balance=Decimal("10000"),
            equity=Decimal("10000"),
            free_margin=Decimal("10000"),
            currency="USDT",
            daily_starting_balance=Decimal("10000"),
        )


def test_invalid_risk_parameters_rejected() -> None:
    """RiskParameters must enforce positive bounds and risk hierarchy (per_trade <= daily <= total_dd)."""
    # risk_per_trade_percent > max_daily_loss_percent
    with pytest.raises(ValidationError, match="risk_per_trade_percent"):
        RiskParameters(
            risk_per_trade_percent=Decimal("4.0"),
            max_daily_loss_percent=Decimal("3.0"),
            max_total_drawdown_percent=Decimal("10.0"),
        )

    # max_daily_loss_percent > max_total_drawdown_percent
    with pytest.raises(ValidationError, match="max_daily_loss_percent"):
        RiskParameters(
            risk_per_trade_percent=Decimal("1.0"),
            max_daily_loss_percent=Decimal("12.0"),
            max_total_drawdown_percent=Decimal("10.0"),
        )

    # Zero or negative minimum_risk_reward_ratio
    with pytest.raises(ValidationError):
        RiskParameters(minimum_risk_reward_ratio=Decimal("0"))


# ============================================================================
# 4. POSITION SIZING & INSTRUMENT SPECIFICATION TESTS
# ============================================================================


def test_position_sizing_exact_and_volume_rounding(xauusd_spec: InstrumentSpec) -> None:
    """Position sizing must floor to volume_step and never exceed allowed risk."""
    # Case 1: Exact division ($10,000 equity, 1% risk = $100 allowed; $10 stop = $1000/lot -> 0.10 lots)
    exact = calculate_position_size(
        equity=Decimal("10000.00"),
        risk_per_trade_percent=Decimal("1.0"),
        entry_price=Decimal("2650.00"),
        stop_loss=Decimal("2640.00"),
        instrument_spec=xauusd_spec,
    )
    assert exact.approved is True
    assert exact.rounded_position_size == Decimal("0.10")
    assert exact.actual_risk_amount == Decimal("100.00")
    assert exact.actual_risk_percent == Decimal("1.00")

    # Case 2: Non-exact division requiring downward floor rounding
    # $10,000 equity, 1% risk = $100 allowed; $7.00 stop = 700 ticks = $700/lot
    # Raw lots = 100 / 700 = 0.142857... -> Must floor to 0.14 lots (actual risk = $98.00 <= $100.00)
    floored = calculate_position_size(
        equity=Decimal("10000.00"),
        risk_per_trade_percent=Decimal("1.0"),
        entry_price=Decimal("2650.00"),
        stop_loss=Decimal("2643.00"),
        instrument_spec=xauusd_spec,
    )
    assert floored.approved is True
    assert floored.rounded_position_size == Decimal("0.14")
    assert floored.actual_risk_amount == Decimal("98.00")
    assert floored.actual_risk_amount <= floored.allowed_risk_amount


def test_position_sizing_different_broker_contract_spec() -> None:
    """Position sizing must adapt dynamically to non-standard broker XAUUSD specifications."""
    # Mini-Gold specification: 10 oz contract size, $0.01 tick size, $0.10 tick value
    mini_xau_spec = InstrumentSpec(
        symbol="XAUUSD",
        contract_size=Decimal("10"),
        tick_size=Decimal("0.01"),
        tick_value=Decimal("0.10"),
        volume_step=Decimal("0.01"),
        minimum_volume=Decimal("0.01"),
        maximum_volume=Decimal("100.0"),
    )
    res = calculate_position_size(
        equity=Decimal("10000.00"),
        risk_per_trade_percent=Decimal("1.0"),
        entry_price=Decimal("2650.00"),
        stop_loss=Decimal("2640.00"),
        instrument_spec=mini_xau_spec,
    )
    # $10 stop = 1000 ticks * $0.10 = $100/lot -> $100 risk / $100/lot = 1.00 lot
    assert res.approved is True
    assert res.rounded_position_size == Decimal("1.00")
    assert res.actual_risk_amount == Decimal("100.00")


def test_position_sizing_rejects_missing_spec_or_zero_stop(xauusd_spec: InstrumentSpec) -> None:
    """Position sizing must safely reject missing InstrumentSpec or zero stop distance."""
    missing = calculate_position_size(
        equity=Decimal("10000"),
        risk_per_trade_percent=Decimal("1.0"),
        entry_price=Decimal("2650"),
        stop_loss=Decimal("2640"),
        instrument_spec=None,
    )
    assert missing.approved is False
    assert RiskRejectionCode.MISSING_INSTRUMENT_SPEC.value in (missing.rejection_reason or "")

    zero_stop = calculate_position_size(
        equity=Decimal("10000"),
        risk_per_trade_percent=Decimal("1.0"),
        entry_price=Decimal("2650"),
        stop_loss=Decimal("2650"),
        instrument_spec=xauusd_spec,
    )
    assert zero_stop.approved is False
    assert RiskRejectionCode.INVALID_STOP_DISTANCE.value in (zero_stop.rejection_reason or "")


def test_position_sizing_rejects_below_min_or_above_max_volume(xauusd_spec: InstrumentSpec) -> None:
    """Position sizing must reject trades whose floored size is < minimum_volume or > max_position_size."""
    # Small account ($500) with 1% risk ($5 allowed) and $10 stop ($1000/lot) -> 0.005 lots -> floors to 0.00 < 0.01 min
    below_min = calculate_position_size(
        equity=Decimal("500.00"),
        risk_per_trade_percent=Decimal("1.0"),
        entry_price=Decimal("2650.00"),
        stop_loss=Decimal("2640.00"),
        instrument_spec=xauusd_spec,
    )
    assert below_min.approved is False
    assert RiskRejectionCode.POSITION_SIZE_BELOW_MINIMUM.value in (below_min.rejection_reason or "")

    # Large account ($1,000,000) with 1% risk ($10,000) and $5 stop ($500/lot) -> 20 lots > max_position_size (10.0)
    above_max = calculate_position_size(
        equity=Decimal("1000000.00"),
        risk_per_trade_percent=Decimal("1.0"),
        entry_price=Decimal("2650.00"),
        stop_loss=Decimal("2645.00"),
        instrument_spec=xauusd_spec,
        max_position_size=Decimal("10.0"),
    )
    assert above_max.approved is False
    assert RiskRejectionCode.POSITION_SIZE_EXCEEDS_MAXIMUM.value in (above_max.rejection_reason or "")


# ============================================================================
# 5. DETERMINISTIC RISK ENGINE EVALUATION TESTS
# ============================================================================


def test_risk_engine_approves_valid_buy_and_sell(
    xauusd_spec: InstrumentSpec,
    healthy_account: AccountState,
    default_risk_params: RiskParameters,
    valid_buy_signal: TradeSignal,
    valid_sell_signal: TradeSignal,
) -> None:
    """RiskEngine must approve valid BUY and SELL signals that satisfy all 10 safety gates."""
    engine = RiskEngine(risk_parameters=default_risk_params, instrument_specs=xauusd_spec)

    buy_decision = engine.evaluate(signal=valid_buy_signal, account=healthy_account)
    assert buy_decision.approved is True
    assert buy_decision.rejection_reason is None
    assert buy_decision.calculated_position_size == Decimal("0.10")
    assert buy_decision.calculated_risk_amount == Decimal("100.00")
    assert buy_decision.risk_reward_ratio == Decimal("2.0000")
    assert buy_decision.risk_percent == Decimal("1.0000")

    sell_decision = engine.evaluate(signal=valid_sell_signal, account=healthy_account)
    assert sell_decision.approved is True
    assert sell_decision.rejection_reason is None
    assert sell_decision.calculated_position_size == Decimal("0.10")
    assert sell_decision.calculated_risk_amount == Decimal("100.00")


def test_risk_engine_rejects_when_trading_disabled(
    xauusd_spec: InstrumentSpec,
    healthy_account: AccountState,
    default_risk_params: RiskParameters,
    valid_buy_signal: TradeSignal,
) -> None:
    """RiskEngine must immediately reject trades when allow_trading is False."""
    disabled_params = default_risk_params.model_copy(update={"allow_trading": False})
    engine = RiskEngine(risk_parameters=disabled_params, instrument_specs=xauusd_spec)

    decision = engine.evaluate(signal=valid_buy_signal, account=healthy_account)
    assert decision.approved is False
    assert RiskRejectionCode.TRADING_DISABLED.value in (decision.rejection_reason or "")
    assert decision.calculated_position_size == Decimal("0")


def test_risk_engine_rejects_wait_signal(
    xauusd_spec: InstrumentSpec,
    healthy_account: AccountState,
    default_risk_params: RiskParameters,
) -> None:
    """RiskEngine must always reject WAIT signals."""
    engine = RiskEngine(risk_parameters=default_risk_params, instrument_specs=xauusd_spec)
    wait_signal = TradeSignal(
        symbol="XAUUSD",
        direction=SignalDirection.WAIT,
        timeframe=Timeframe.H1,
        strategy_id="no_trade_filter",
        confidence=Decimal("0.90"),
        timestamp=FIXED_TS,
        rationale="Waiting for FOMC release.",
    )
    decision = engine.evaluate(signal=wait_signal, account=healthy_account)
    assert decision.approved is False
    assert RiskRejectionCode.WAIT_SIGNAL_NOT_EXECUTABLE.value in (decision.rejection_reason or "")


def test_risk_engine_rejects_missing_instrument_spec(
    healthy_account: AccountState,
    default_risk_params: RiskParameters,
    valid_buy_signal: TradeSignal,
) -> None:
    """RiskEngine must reject signals when InstrumentSpec is not configured for the symbol."""
    engine = RiskEngine(risk_parameters=default_risk_params, instrument_specs=None)
    decision = engine.evaluate(signal=valid_buy_signal, account=healthy_account)
    assert decision.approved is False
    assert RiskRejectionCode.MISSING_INSTRUMENT_SPEC.value in (decision.rejection_reason or "")


def test_risk_engine_rejects_insufficient_risk_reward(
    xauusd_spec: InstrumentSpec,
    healthy_account: AccountState,
    default_risk_params: RiskParameters,
) -> None:
    """RiskEngine must reject signals whose RR ratio is below minimum_risk_reward_ratio (1.5)."""
    engine = RiskEngine(risk_parameters=default_risk_params, instrument_specs=xauusd_spec)
    low_rr_signal = TradeSignal(
        symbol="XAUUSD",
        direction=SignalDirection.BUY,
        entry_price=Decimal("2650.00"),
        stop_loss=Decimal("2640.00"),  # $10 risk
        take_profit=Decimal("2662.00"),  # $12 reward -> RR = 1.2 < 1.5
        timeframe=Timeframe.H1,
        strategy_id="scalp_v1",
        confidence=Decimal("0.80"),
        timestamp=FIXED_TS,
        rationale="Low RR test signal.",
    )
    decision = engine.evaluate(signal=low_rr_signal, account=healthy_account)
    assert decision.approved is False
    assert RiskRejectionCode.INSUFFICIENT_RISK_REWARD_RATIO.value in (decision.rejection_reason or "")
    assert decision.risk_reward_ratio == Decimal("1.2000")


def test_risk_engine_rejects_max_open_positions(
    xauusd_spec: InstrumentSpec,
    healthy_account: AccountState,
    default_risk_params: RiskParameters,
    valid_buy_signal: TradeSignal,
) -> None:
    """RiskEngine must reject new trades when open_positions_count >= max_open_positions."""
    engine = RiskEngine(risk_parameters=default_risk_params, instrument_specs=xauusd_spec)
    full_account = healthy_account.model_copy(update={"open_positions_count": 3})

    decision = engine.evaluate(signal=valid_buy_signal, account=full_account)
    assert decision.approved is False
    assert RiskRejectionCode.MAX_OPEN_POSITIONS_REACHED.value in (decision.rejection_reason or "")


def test_risk_engine_rejects_daily_loss_limit_breach_and_projected_breach(
    xauusd_spec: InstrumentSpec,
    default_risk_params: RiskParameters,
    valid_buy_signal: TradeSignal,
) -> None:
    """RiskEngine must reject when daily loss limit (3% = $300) is already reached OR would be breached by the trade."""
    engine = RiskEngine(risk_parameters=default_risk_params, instrument_specs=xauusd_spec)

    # Case A: Already down $300 (-3.0%) on the day
    breached_account = AccountState(
        balance=Decimal("9700.00"),
        equity=Decimal("9700.00"),
        free_margin=Decimal("9700.00"),
        current_daily_pnl=Decimal("-300.00"),
        daily_starting_balance=Decimal("10000.00"),
    )
    res_a = engine.evaluate(signal=valid_buy_signal, account=breached_account)
    assert res_a.approved is False
    assert RiskRejectionCode.DAILY_LOSS_LIMIT_EXCEEDED.value in (res_a.rejection_reason or "")

    # Case B: Down $250 (-2.5%) on the day; new 1% trade risks ~$97.50 -> projected loss $340 > $300 max
    near_limit_account = AccountState(
        balance=Decimal("9750.00"),
        equity=Decimal("9750.00"),
        free_margin=Decimal("9750.00"),
        current_daily_pnl=Decimal("-250.00"),
        daily_starting_balance=Decimal("10000.00"),
    )
    res_b = engine.evaluate(signal=valid_buy_signal, account=near_limit_account)
    assert res_b.approved is False
    assert RiskRejectionCode.DAILY_LOSS_LIMIT_EXCEEDED.value in (res_b.rejection_reason or "")


def test_risk_engine_rejects_total_drawdown_limit_breach_and_projected_breach(
    xauusd_spec: InstrumentSpec,
    default_risk_params: RiskParameters,
    valid_buy_signal: TradeSignal,
) -> None:
    """RiskEngine must reject when total drawdown (10% of peak) is reached OR would be breached by the trade."""
    engine = RiskEngine(risk_parameters=default_risk_params, instrument_specs=xauusd_spec)

    # Case A: Peak was $10,000, current equity is $9,000 (10% total drawdown reached, even if flat today)
    dd_breached = AccountState(
        balance=Decimal("9000.00"),
        equity=Decimal("9000.00"),
        free_margin=Decimal("9000.00"),
        current_daily_pnl=Decimal("0.00"),
        daily_starting_balance=Decimal("9000.00"),
        peak_balance=Decimal("10000.00"),
    )
    res_a = engine.evaluate(signal=valid_buy_signal, account=dd_breached)
    assert res_a.approved is False
    assert RiskRejectionCode.TOTAL_DRAWDOWN_LIMIT_EXCEEDED.value in (res_a.rejection_reason or "")

    # Case B: Peak was $10,000, current equity is $9,050 ($950 DD); new trade risks $90 -> projected DD $1,040 > $1,000
    dd_near_limit = AccountState(
        balance=Decimal("9050.00"),
        equity=Decimal("9050.00"),
        free_margin=Decimal("9050.00"),
        current_daily_pnl=Decimal("0.00"),
        daily_starting_balance=Decimal("9050.00"),
        peak_balance=Decimal("10000.00"),
    )
    res_b = engine.evaluate(signal=valid_buy_signal, account=dd_near_limit)
    assert res_b.approved is False
    assert RiskRejectionCode.TOTAL_DRAWDOWN_LIMIT_EXCEEDED.value in (res_b.rejection_reason or "")


def test_risk_engine_rejects_insufficient_free_margin(
    xauusd_spec: InstrumentSpec,
    default_risk_params: RiskParameters,
    valid_buy_signal: TradeSignal,
) -> None:
    """RiskEngine must reject trades when required initial margin exceeds available free_margin."""
    engine = RiskEngine(risk_parameters=default_risk_params, instrument_specs=xauusd_spec)
    # 0.10 lots at 2650.00 * 100 oz * 0.01 margin_rate = $265.00 required margin
    low_margin_account = AccountState(
        balance=Decimal("10000.00"),
        equity=Decimal("10000.00"),
        free_margin=Decimal("200.00"),  # Less than $265.00 required
        daily_starting_balance=Decimal("10000.00"),
    )
    decision = engine.evaluate(signal=valid_buy_signal, account=low_margin_account)
    assert decision.approved is False
    assert RiskRejectionCode.INSUFFICIENT_FREE_MARGIN.value in (decision.rejection_reason or "")


def test_risk_engine_determinism(
    xauusd_spec: InstrumentSpec,
    healthy_account: AccountState,
    default_risk_params: RiskParameters,
    valid_buy_signal: TradeSignal,
) -> None:
    """Repeated evaluations with identical inputs must produce bit-for-bit identical RiskDecisions."""
    engine = RiskEngine(risk_parameters=default_risk_params, instrument_specs=xauusd_spec)
    first_decision = engine.evaluate(signal=valid_buy_signal, account=healthy_account)

    for _ in range(25):
        repeat_decision = engine.evaluate(signal=valid_buy_signal, account=healthy_account)
        assert repeat_decision == first_decision
        assert repeat_decision.model_dump() == first_decision.model_dump()
