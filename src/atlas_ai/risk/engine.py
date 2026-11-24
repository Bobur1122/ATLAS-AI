"""Deterministic, LLM-independent Risk Engine and Position Sizer for ATLAS AI."""

from decimal import Decimal, ROUND_DOWN

from atlas_ai.core.logger import get_logger
from atlas_ai.domain.market import InstrumentSpec
from atlas_ai.domain.risk import (
    AccountState,
    PositionSizeResult,
    RiskDecision,
    RiskParameters,
    RiskRejectionCode,
)
from atlas_ai.domain.signals import SignalDirection, TradeSignal

logger = get_logger("risk.engine")

HUNDRED = Decimal("100")
QUANTIZE_MONEY = Decimal("0.01")
QUANTIZE_RATIO = Decimal("0.0001")


def calculate_position_size(
    *,
    equity: Decimal,
    risk_per_trade_percent: Decimal,
    entry_price: Decimal,
    stop_loss: Decimal,
    instrument_spec: InstrumentSpec | None,
    max_position_size: Decimal | None = None,
) -> PositionSizeResult:
    """Deterministically calculate position size in lots from equity, risk %, and stop distance.

    Formula:
        Allowed Risk Amount = Equity * (risk_per_trade_percent / 100)
        Stop Loss Distance  = |entry_price - stop_loss|
        Ticks At Risk       = Stop Loss Distance / instrument_spec.tick_size
        Loss Per 1.0 Lot    = Ticks At Risk * instrument_spec.tick_value
        Raw Lots            = Allowed Risk Amount / Loss Per 1.0 Lot
        Rounded Lots        = floor(Raw Lots / volume_step) * volume_step

    Volume is strictly rounded DOWN (floored) to `instrument_spec.volume_step` so that
    actual monetary risk never exceeds `Allowed Risk Amount`.
    """
    if instrument_spec is None:
        return PositionSizeResult(
            approved=False,
            rejection_reason=(
                f"{RiskRejectionCode.MISSING_INSTRUMENT_SPEC.value}: "
                "Instrument contract specification is required for position sizing."
            ),
        )

    if equity <= Decimal("0") or risk_per_trade_percent <= Decimal("0"):
        return PositionSizeResult(
            approved=False,
            rejection_reason=(
                f"{RiskRejectionCode.INVALID_ACCOUNT_EQUITY.value}: "
                "Account equity and risk_per_trade_percent must be strictly positive."
            ),
        )

    stop_distance = abs(entry_price - stop_loss)
    if entry_price <= Decimal("0") or stop_loss <= Decimal("0") or stop_distance <= Decimal("0"):
        return PositionSizeResult(
            approved=False,
            rejection_reason=(
                f"{RiskRejectionCode.INVALID_STOP_DISTANCE.value}: "
                f"Stop-loss distance must be strictly positive (got {stop_distance})."
            ),
            stop_loss_distance= max(stop_distance, Decimal("0")),
        )

    allowed_risk_amount = equity * (risk_per_trade_percent / HUNDRED)
    ticks_at_risk = stop_distance / instrument_spec.tick_size
    loss_per_lot = ticks_at_risk * instrument_spec.tick_value

    if loss_per_lot <= Decimal("0"):
        return PositionSizeResult(
            approved=False,
            rejection_reason=(
                f"{RiskRejectionCode.INVALID_STOP_DISTANCE.value}: "
                "Calculated monetary loss per lot must be strictly positive."
            ),
            stop_loss_distance=stop_distance,
        )

    raw_lots = allowed_risk_amount / loss_per_lot

    # Check if raw lots exceed risk parameter max_position_size or broker maximum_volume
    effective_max_volume = instrument_spec.maximum_volume
    if max_position_size is not None:
        effective_max_volume = min(effective_max_volume, max_position_size)

    # Floor to volume_step so we NEVER exceed allowed_risk_amount due to rounding
    steps_count = (raw_lots / instrument_spec.volume_step).to_integral_value(rounding=ROUND_DOWN)
    rounded_lots = (steps_count * instrument_spec.volume_step).quantize(
        instrument_spec.volume_step, rounding=ROUND_DOWN
    )

    if rounded_lots < instrument_spec.minimum_volume:
        return PositionSizeResult(
            approved=False,
            rejection_reason=(
                f"{RiskRejectionCode.POSITION_SIZE_BELOW_MINIMUM.value}: "
                f"Calculated volume ({rounded_lots} lots) is below minimum_volume "
                f"({instrument_spec.minimum_volume} lots) required to stay within "
                f"{risk_per_trade_percent}% risk ({allowed_risk_amount:.2f})."
            ),
            raw_position_size=raw_lots,
            rounded_position_size=rounded_lots,
            allowed_risk_amount=allowed_risk_amount,
            stop_loss_distance=stop_distance,
            loss_per_lot=loss_per_lot,
        )

    if rounded_lots > effective_max_volume:
        return PositionSizeResult(
            approved=False,
            rejection_reason=(
                f"{RiskRejectionCode.POSITION_SIZE_EXCEEDS_MAXIMUM.value}: "
                f"Calculated volume ({rounded_lots} lots) exceeds maximum allowed position size "
                f"({effective_max_volume} lots)."
            ),
            raw_position_size=raw_lots,
            rounded_position_size=rounded_lots,
            allowed_risk_amount=allowed_risk_amount,
            stop_loss_distance=stop_distance,
            loss_per_lot=loss_per_lot,
        )

    actual_risk_amount = rounded_lots * loss_per_lot
    actual_risk_percent = (actual_risk_amount / equity) * HUNDRED

    # Final invariant check: rounding must never cause actual risk to exceed allowed risk
    if actual_risk_amount > allowed_risk_amount:
        return PositionSizeResult(
            approved=False,
            rejection_reason=(
                f"{RiskRejectionCode.RISK_PER_TRADE_EXCEEDED.value}: "
                f"Actual risk ({actual_risk_amount}) exceeds allowed risk ({allowed_risk_amount})."
            ),
            raw_position_size=raw_lots,
            rounded_position_size=rounded_lots,
            allowed_risk_amount=allowed_risk_amount,
            actual_risk_amount=actual_risk_amount,
            actual_risk_percent=actual_risk_percent,
            stop_loss_distance=stop_distance,
            loss_per_lot=loss_per_lot,
        )

    return PositionSizeResult(
        approved=True,
        rejection_reason=None,
        raw_position_size=raw_lots,
        rounded_position_size=rounded_lots,
        allowed_risk_amount=allowed_risk_amount,
        actual_risk_amount=actual_risk_amount,
        actual_risk_percent=actual_risk_percent,
        stop_loss_distance=stop_distance,
        loss_per_lot=loss_per_lot,
    )


class RiskEngine:
    """Standalone deterministic risk validation engine.

    Operates completely independently of any LLM, AI model, or external API.
    Evaluates every TradeSignal against AccountState, RiskParameters, and InstrumentSpec.
    """

    def __init__(
        self,
        risk_parameters: RiskParameters,
        instrument_specs: dict[str, InstrumentSpec] | InstrumentSpec | None = None,
    ) -> None:
        """Initialize the RiskEngine with deterministic parameters and instrument specifications.

        Args:
            risk_parameters: Validated RiskParameters instance.
            instrument_specs: Either a single InstrumentSpec or a mapping of symbol -> InstrumentSpec.
        """
        self._risk_parameters = risk_parameters
        self._instrument_specs: dict[str, InstrumentSpec] = {}
        if isinstance(instrument_specs, InstrumentSpec):
            self._instrument_specs[instrument_specs.symbol] = instrument_specs
        elif isinstance(instrument_specs, dict):
            for sym, spec in instrument_specs.items():
                self._instrument_specs[sym.strip().upper()] = spec

    @property
    def risk_parameters(self) -> RiskParameters:
        """Return configured RiskParameters."""
        return self._risk_parameters

    def get_instrument_spec(self, symbol: str) -> InstrumentSpec | None:
        """Look up the InstrumentSpec for a given symbol."""
        return self._instrument_specs.get(symbol.strip().upper())

    def evaluate(
        self,
        signal: TradeSignal,
        account: AccountState,
        risk_parameters: RiskParameters | None = None,
        instrument_spec: InstrumentSpec | None = None,
    ) -> RiskDecision:
        """Deterministically evaluate a TradeSignal against account state and risk limits.

        Checks performed in strict deterministic order:
        1. Trading enabled/disabled (`allow_trading`).
        2. WAIT signal rejection (`SignalDirection.WAIT`).
        3. Instrument specification availability and symbol match.
        4. Valid entry, stop-loss, and take-profit prices & directional geometry.
        5. Account equity and free margin availability.
        6. Maximum open positions limit (`open_positions_count < max_open_positions`).
        7. Minimum risk/reward ratio (`reward / risk >= minimum_risk_reward_ratio`).
        8. Maximum daily loss limit (current daily loss + proposed trade risk <= max_daily_loss_percent).
        9. Maximum total drawdown limit (current drawdown + proposed trade risk <= max_total_drawdown_percent).
        10. Deterministic position sizing, volume step floor rounding, min/max volume caps, and margin check.
        """
        params = risk_parameters or self._risk_parameters
        spec = instrument_spec or self.get_instrument_spec(signal.symbol)

        details: dict[str, object] = {
            "symbol": signal.symbol,
            "direction": signal.direction.value,
            "strategy_id": signal.strategy_id,
            "allow_trading": params.allow_trading,
            "open_positions_count": account.open_positions_count,
            "max_open_positions": params.max_open_positions,
        }

        # 1. Trading enabled/disabled check
        if not params.allow_trading:
            return self._reject(
                code=RiskRejectionCode.TRADING_DISABLED,
                message="Trading is disabled by RiskParameters (allow_trading=False).",
                details=details,
            )

        # 2. WAIT signal rejection
        if signal.direction == SignalDirection.WAIT or not signal.is_executable:
            return self._reject(
                code=RiskRejectionCode.WAIT_SIGNAL_NOT_EXECUTABLE,
                message="Signal direction is WAIT; non-executable signals cannot open trades.",
                details=details,
            )

        # 3. Instrument specification check
        if spec is None:
            return self._reject(
                code=RiskRejectionCode.MISSING_INSTRUMENT_SPEC,
                message=f"Missing InstrumentSpec for symbol '{signal.symbol}'.",
                details=details,
            )

        if spec.symbol != signal.symbol:
            return self._reject(
                code=RiskRejectionCode.SYMBOL_MISMATCH,
                message=(
                    f"InstrumentSpec symbol ('{spec.symbol}') does not match "
                    f"TradeSignal symbol ('{signal.symbol}')."
                ),
                details=details,
            )

        # 4. Entry, Stop-Loss, and Take-Profit validation (defensive check even if bypassed via construct)
        entry = signal.entry_price
        sl = signal.stop_loss
        tp = signal.take_profit
        if entry is None or sl is None or tp is None or entry <= Decimal("0") or sl <= Decimal("0") or tp <= Decimal("0"):
            return self._reject(
                code=RiskRejectionCode.INVALID_SIGNAL_PRICES,
                message="Entry, stop-loss, and take-profit prices must all be present and > 0.",
                details=details,
            )

        if signal.direction == SignalDirection.BUY and not (sl < entry < tp):
            return self._reject(
                code=RiskRejectionCode.INVALID_SIGNAL_PRICES,
                message=f"BUY signal requires stop_loss ({sl}) < entry_price ({entry}) < take_profit ({tp}).",
                details=details,
            )

        if signal.direction == SignalDirection.SELL and not (tp < entry < sl):
            return self._reject(
                code=RiskRejectionCode.INVALID_SIGNAL_PRICES,
                message=f"SELL signal requires take_profit ({tp}) < entry_price ({entry}) < stop_loss ({sl}).",
                details=details,
            )

        stop_distance = abs(entry - sl)
        reward_distance = abs(tp - entry)
        if stop_distance <= Decimal("0"):
            return self._reject(
                code=RiskRejectionCode.INVALID_STOP_DISTANCE,
                message="Stop-loss distance must be strictly positive.",
                details=details,
            )

        # 5. Account equity and basic free margin check
        if account.equity <= Decimal("0") or account.balance <= Decimal("0"):
            return self._reject(
                code=RiskRejectionCode.INVALID_ACCOUNT_EQUITY,
                message="Account equity and balance must be strictly positive.",
                details=details,
            )

        if account.free_margin <= Decimal("0"):
            return self._reject(
                code=RiskRejectionCode.INSUFFICIENT_FREE_MARGIN,
                message=f"Account free_margin ({account.free_margin}) must be > 0 to open a trade.",
                details=details,
            )

        # 6. Maximum open positions check
        if account.open_positions_count >= params.max_open_positions:
            return self._reject(
                code=RiskRejectionCode.MAX_OPEN_POSITIONS_REACHED,
                message=(
                    f"Open positions count ({account.open_positions_count}) has reached "
                    f"or exceeded max_open_positions ({params.max_open_positions})."
                ),
                details=details,
            )

        # 7. Minimum Risk/Reward Ratio check
        rr_ratio = reward_distance / stop_distance
        quantized_rr = rr_ratio.quantize(QUANTIZE_RATIO, rounding=ROUND_DOWN)
        details["risk_reward_ratio"] = str(quantized_rr)
        details["minimum_risk_reward_ratio"] = str(params.minimum_risk_reward_ratio)

        if rr_ratio < params.minimum_risk_reward_ratio:
            return self._reject(
                code=RiskRejectionCode.INSUFFICIENT_RISK_REWARD_RATIO,
                message=(
                    f"Risk/Reward ratio ({quantized_rr}) is below minimum_risk_reward_ratio "
                    f"({params.minimum_risk_reward_ratio})."
                ),
                details=details,
                risk_reward_ratio=quantized_rr,
            )

        # 8. Daily Loss Limit check (both current breach and post-trade projected breach)
        max_daily_loss_amount = account.daily_starting_balance * (
            params.max_daily_loss_percent / HUNDRED
        )
        # current_daily_loss is positive when the account is down on the day
        loss_from_pnl = max(Decimal("0"), -account.current_daily_pnl)
        loss_from_equity = max(Decimal("0"), account.daily_starting_balance - account.equity)
        current_daily_loss = max(loss_from_pnl, loss_from_equity)
        current_daily_loss_pct = (current_daily_loss / account.daily_starting_balance) * HUNDRED

        details["current_daily_loss_amount"] = str(current_daily_loss.quantize(QUANTIZE_MONEY))
        details["max_daily_loss_amount"] = str(max_daily_loss_amount.quantize(QUANTIZE_MONEY))

        if current_daily_loss >= max_daily_loss_amount:
            return self._reject(
                code=RiskRejectionCode.DAILY_LOSS_LIMIT_EXCEEDED,
                message=(
                    f"Current daily loss ({current_daily_loss:.2f} / {current_daily_loss_pct:.2f}%) "
                    f"has reached or exceeded max_daily_loss_percent ({params.max_daily_loss_percent}% / "
                    f"{max_daily_loss_amount:.2f})."
                ),
                details=details,
                risk_reward_ratio=quantized_rr,
            )

        # 9. Total Drawdown Limit check (both current breach and post-trade projected breach)
        reference_peak = account.effective_peak_balance
        max_total_dd_amount = reference_peak * (params.max_total_drawdown_percent / HUNDRED)
        current_dd_amount = max(Decimal("0"), reference_peak - account.equity)
        current_dd_pct = (current_dd_amount / reference_peak) * HUNDRED

        details["current_drawdown_amount"] = str(current_dd_amount.quantize(QUANTIZE_MONEY))
        details["max_total_drawdown_amount"] = str(max_total_dd_amount.quantize(QUANTIZE_MONEY))

        if current_dd_amount >= max_total_dd_amount:
            return self._reject(
                code=RiskRejectionCode.TOTAL_DRAWDOWN_LIMIT_EXCEEDED,
                message=(
                    f"Current total drawdown ({current_dd_amount:.2f} / {current_dd_pct:.2f}%) "
                    f"has reached or exceeded max_total_drawdown_percent "
                    f"({params.max_total_drawdown_percent}% / {max_total_dd_amount:.2f})."
                ),
                details=details,
                risk_reward_ratio=quantized_rr,
            )

        # 10. Deterministic Position Sizing
        sizing = calculate_position_size(
            equity=account.equity,
            risk_per_trade_percent=params.risk_per_trade_percent,
            entry_price=entry,
            stop_loss=sl,
            instrument_spec=spec,
            max_position_size=params.max_position_size,
        )
        details["raw_position_size"] = str(sizing.raw_position_size)
        details["rounded_position_size"] = str(sizing.rounded_position_size)
        details["allowed_risk_amount"] = str(sizing.allowed_risk_amount.quantize(QUANTIZE_MONEY))
        details["actual_risk_amount"] = str(sizing.actual_risk_amount.quantize(QUANTIZE_MONEY))

        if not sizing.approved:
            return RiskDecision(
                approved=False,
                rejection_reason=sizing.rejection_reason,
                calculated_risk_amount=Decimal("0"),
                calculated_position_size=Decimal("0"),
                risk_reward_ratio=quantized_rr,
                risk_percent=Decimal("0"),
                validation_details=details,
            )

        # Verify that adding this trade's actual risk does not breach remaining daily loss or total drawdown buffer
        projected_daily_loss = current_daily_loss + sizing.actual_risk_amount
        if projected_daily_loss > max_daily_loss_amount:
            return self._reject(
                code=RiskRejectionCode.DAILY_LOSS_LIMIT_EXCEEDED,
                message=(
                    f"Projected daily loss including new trade risk ({projected_daily_loss:.2f}) "
                    f"would exceed max_daily_loss_percent limit ({max_daily_loss_amount:.2f})."
                ),
                details=details,
                risk_reward_ratio=quantized_rr,
            )

        projected_total_dd = current_dd_amount + sizing.actual_risk_amount
        if projected_total_dd > max_total_dd_amount:
            return self._reject(
                code=RiskRejectionCode.TOTAL_DRAWDOWN_LIMIT_EXCEEDED,
                message=(
                    f"Projected total drawdown including new trade risk ({projected_total_dd:.2f}) "
                    f"would exceed max_total_drawdown_percent limit ({max_total_dd_amount:.2f})."
                ),
                details=details,
                risk_reward_ratio=quantized_rr,
            )

        # Verify sufficient free margin for required initial margin of the sized position
        notional_value = sizing.rounded_position_size * spec.contract_size * entry
        required_margin = notional_value * spec.margin_rate
        details["required_margin"] = str(required_margin.quantize(QUANTIZE_MONEY))
        details["free_margin"] = str(account.free_margin.quantize(QUANTIZE_MONEY))

        if required_margin > account.free_margin:
            return self._reject(
                code=RiskRejectionCode.INSUFFICIENT_FREE_MARGIN,
                message=(
                    f"Required margin ({required_margin:.2f}) for {sizing.rounded_position_size} lots "
                    f"exceeds available free_margin ({account.free_margin:.2f})."
                ),
                details=details,
                risk_reward_ratio=quantized_rr,
            )

        quantized_risk_amount = sizing.actual_risk_amount.quantize(QUANTIZE_MONEY, rounding=ROUND_DOWN)
        quantized_risk_pct = sizing.actual_risk_percent.quantize(QUANTIZE_RATIO, rounding=ROUND_DOWN)

        logger.info(
            "RiskEngine APPROVED %s %s: size=%s lots, risk=%s (%s%%), RR=%s",
            signal.direction.value,
            signal.symbol,
            sizing.rounded_position_size,
            quantized_risk_amount,
            quantized_risk_pct,
            quantized_rr,
        )

        return RiskDecision(
            approved=True,
            rejection_reason=None,
            calculated_risk_amount=quantized_risk_amount,
            calculated_position_size=sizing.rounded_position_size,
            risk_reward_ratio=quantized_rr,
            risk_percent=quantized_risk_pct,
            validation_details=details,
        )

    @staticmethod
    def _reject(
        *,
        code: RiskRejectionCode,
        message: str,
        details: dict[str, object],
        risk_reward_ratio: Decimal = Decimal("0"),
    ) -> RiskDecision:
        """Build a standardized rejected RiskDecision and log the rejection."""
        reason = f"{code.value}: {message}"
        details["rejection_code"] = code.value
        logger.warning("RiskEngine REJECTED trade — %s", reason)
        return RiskDecision(
            approved=False,
            rejection_reason=reason,
            calculated_risk_amount=Decimal("0"),
            calculated_position_size=Decimal("0"),
            risk_reward_ratio=risk_reward_ratio,
            risk_percent=Decimal("0"),
            validation_details=details,
        )
