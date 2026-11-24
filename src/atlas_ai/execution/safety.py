"""Deterministic Pre-Trade Safety Gate for ATLAS AI Phase 7.4."""

from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN

from pydantic import BaseModel, ConfigDict, Field

from atlas_ai.core.config import ExecutionMode
from atlas_ai.core.logger import get_logger
from atlas_ai.domain.market import InstrumentSpec
from atlas_ai.domain.risk import AccountState, RiskDecision, RiskParameters
from atlas_ai.domain.signals import SignalDirection, TradeSignal
from atlas_ai.execution.account_state import PersistentAccountRiskTracker
from atlas_ai.execution.config import ExecutionSafetyConfig
from atlas_ai.execution.connection import (
    MT5ConnectionAdapter,
    resolve_supported_filling_mode,
)
from atlas_ai.execution.exceptions import (
    DuplicateOrderError,
    LiveTradingBlockedError,
    OrderValidationError,
)
from atlas_ai.execution.ledger import (
    PersistentExecutionLedger,
    extract_comment_identifiers,
)
from atlas_ai.execution.market_data import MT5MarketDataAdapter
from atlas_ai.execution.models import (
    MT5AccountInfo,
    MT5OrderCheckResult,
    MT5OrderFillingMode,
    MT5OrderRequest,
    MT5OrderTimeMode,
    MT5OrderType,
    MT5Retcode,
    MT5SymbolInfo,
    MT5SymbolTradeMode,
    MT5TradeAction,
    SymbolTick,
)
from atlas_ai.risk.engine import RiskEngine
from atlas_ai.strategies.exceptions import SignalLifecycleError
from atlas_ai.strategies.models import SignalLifecycleState
from atlas_ai.strategies.signal_lifecycle import SignalLifecycleManager

logger = get_logger("execution.safety")


class PreTradeSafetyValidationResult(BaseModel):
    """Immutable audit bundle produced when all 14 PreTradeSafetyGate checks succeed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    client_order_id: str = Field(..., min_length=1)
    signal_id: str = Field(..., min_length=1)
    signal: TradeSignal
    live_signal: TradeSignal
    account_info: MT5AccountInfo
    account_state: AccountState
    symbol_info: MT5SymbolInfo
    instrument_spec: InstrumentSpec
    latest_tick: SymbolTick
    spread_points: Decimal
    price_drift_points: Decimal
    filling_mode: MT5OrderFillingMode
    approved_volume: Decimal = Field(..., gt=Decimal("0"))
    risk_decision: RiskDecision
    order_request: MT5OrderRequest
    order_check_result: MT5OrderCheckResult
    validated_at_utc: datetime


class PreTradeSafetyGate:
    """Deterministic, LLM-independent pre-trade safety gate enforcing all 14 execution checks.

    Checks performed in strict fail-closed order:
    1. ExecutionSafetyConfig DEMO-only verification (`execution_mode == DEMO`, `live_trading_enabled == False`).
    2. Terminal connection health, API permissions, and DEMO account verification (`trade_mode == 0`).
    3. Execution ledger health (`RECOVERY_REQUIRED_LOCKED` block) and unresolved intent block.
    4. Signal executability (`BUY`/`SELL` only, valid prices) and `SignalLifecycleManager` state/expiration check.
    5. Duplicate or conflicting order check across execution ledger and live broker positions.
    6. Symbol availability, visibility, and contract metadata (`InstrumentSpec`).
    7. Real-time tick freshness and stale/future market data rejection (`max_tick_age_seconds`).
    8. Live Bid-Ask spread limit (`max_allowed_spread_points`).
    9. Live execution price drift (`max_entry_deviation_points`) and live directional SL/TP geometry.
    10. Broker stop-level (`trade_stops_level`) and freeze-level (`trade_freeze_level`) restrictions.
    11. Persistent AccountState reconstruction and Phase 1 `RiskEngine.evaluate()` (risk-per-trade,
        daily loss limit, total drawdown limit, max open positions, minimum RR ratio, free margin).
    12. Volume minimum, maximum, step multiple, and RiskEngine cap validation.
    13. Execution-mode-aware filling policy resolution (`resolve_supported_filling_mode`).
    14. Pre-flight MT5 `order_check()` margin and server acceptance verification.
    """

    def __init__(
        self,
        *,
        connection: MT5ConnectionAdapter,
        market_data: MT5MarketDataAdapter,
        account_tracker: PersistentAccountRiskTracker,
        ledger: PersistentExecutionLedger,
        risk_engine: RiskEngine,
        safety_config: ExecutionSafetyConfig | None = None,
        lifecycle_manager: SignalLifecycleManager | None = None,
    ) -> None:
        self._connection = connection
        self._market_data = market_data
        self._account_tracker = account_tracker
        self._ledger = ledger
        self._risk_engine = risk_engine
        self._safety_config = safety_config or ExecutionSafetyConfig()
        self._lifecycle_manager = lifecycle_manager

    @property
    def connection(self) -> MT5ConnectionAdapter:
        return self._connection

    @property
    def market_data(self) -> MT5MarketDataAdapter:
        return self._market_data

    @property
    def account_tracker(self) -> PersistentAccountRiskTracker:
        return self._account_tracker

    @property
    def ledger(self) -> PersistentExecutionLedger:
        return self._ledger

    @property
    def risk_engine(self) -> RiskEngine:
        return self._risk_engine

    @property
    def safety_config(self) -> ExecutionSafetyConfig:
        return self._safety_config

    @property
    def lifecycle_manager(self) -> SignalLifecycleManager | None:
        return self._lifecycle_manager

    def validate_and_build_order(
        self,
        signal: TradeSignal,
        *,
        signal_id: str,
        client_order_id: str | None = None,
        requested_volume: Decimal | None = None,
        risk_parameters: RiskParameters | None = None,
        reference_now_utc: datetime | None = None,
    ) -> PreTradeSafetyValidationResult:
        """Execute all 14 deterministic pre-trade safety checks and return a validated order bundle."""
        now_utc = (
            reference_now_utc.astimezone(timezone.utc)
            if (reference_now_utc is not None and reference_now_utc.tzinfo is not None)
            else (
                reference_now_utc.replace(tzinfo=timezone.utc)
                if reference_now_utc is not None
                else datetime.now(timezone.utc)
            )
        )
        cleaned_sig_id = signal_id.strip()
        if not cleaned_sig_id:
            raise OrderValidationError("signal_id must be a non-empty string.")
        cleaned_cid = (client_order_id or f"ord-{cleaned_sig_id}").strip()
        if not cleaned_cid:
            raise OrderValidationError("client_order_id must be a non-empty string.")
        if len(cleaned_cid) > 64:
            raise OrderValidationError(
                f"client_order_id '{cleaned_cid}' exceeds MT5 64-character comment limit."
            )

        # 1. ExecutionSafetyConfig DEMO-only check
        if (
            self._safety_config.live_trading_enabled
            or self._safety_config.execution_mode != ExecutionMode.DEMO
        ):
            self._connection.disconnect()
            raise LiveTradingBlockedError(
                "SAFETY VIOLATION: PreTradeSafetyGate blocked execution because ExecutionSafetyConfig "
                "is not in DEMO-only mode."
            )

        # 2. Terminal connection, permissions, and mandatory DEMO account verification
        self._connection.ensure_connected()
        term_info, account_info = self._connection.check_health()
        if not account_info.is_demo:
            self._connection.disconnect()
            raise LiveTradingBlockedError(
                f"SAFETY VIOLATION: Connected account {account_info.login}@{account_info.server} "
                f"has trade_mode={account_info.trade_mode} (not DEMO=0). Terminal shut down."
            )
        if not term_info.trade_allowed or term_info.tradeapi_disabled:
            raise OrderValidationError(
                "MT5 terminal algorithmic trading is disabled (trade_allowed=False or tradeapi_disabled=True)."
            )
        if not account_info.trade_allowed or not account_info.trade_expert:
            raise OrderValidationError(
                f"MT5 account {account_info.login} has trading or Expert Advisor execution disabled."
            )

        # 3. Signal executability & SignalLifecycleManager state verification
        if signal.direction == SignalDirection.WAIT or not signal.is_executable:
            raise OrderValidationError(
                "WAIT signals are non-executable and cannot pass PreTradeSafetyGate."
            )
        if (
            signal.entry_price is None
            or signal.stop_loss is None
            or signal.take_profit is None
            or signal.entry_price <= Decimal("0")
            or signal.stop_loss <= Decimal("0")
            or signal.take_profit <= Decimal("0")
        ):
            raise OrderValidationError(
                "Signal entry_price, stop_loss, and take_profit must all be present and > 0."
            )

        # 4. Execution ledger health, unresolved intent block, and duplicate/conflict check
        self._ledger.ensure_ready_for_new_submission(
            account_login=account_info.login,
            server=account_info.server,
            client_order_id=cleaned_cid,
            signal_id=cleaned_sig_id,
            symbol=signal.symbol,
            direction=signal.direction,
        )

        # 5. Live broker position & working-order duplicate and opposite-direction conflict check
        symbol_positions = self._connection.backend.positions_get(signal.symbol)
        if symbol_positions is None:
            raise OrderValidationError(
                f"MT5 positions_get('{signal.symbol}') returned None during pre-trade safety verification."
            )
        for pos in symbol_positions:
            pos_tokens = extract_comment_identifiers(pos.comment)
            if cleaned_cid in pos_tokens or cleaned_sig_id in pos_tokens:
                raise DuplicateOrderError(
                    f"Duplicate order on broker: open position ticket={pos.ticket} already carries "
                    f"comment='{pos.comment}' matching '{cleaned_cid}' / '{cleaned_sig_id}'."
                )
            pos_dir = SignalDirection.BUY if pos.position_type == 0 else SignalDirection.SELL
            if pos_dir != signal.direction:
                raise DuplicateOrderError(
                    f"Conflicting open position on broker (ticket={pos.ticket}, symbol={pos.symbol}, "
                    f"magic={pos.magic}, direction={pos_dir.value}) opposes requested direction "
                    f"({signal.direction.value})."
                )

        symbol_orders = self._connection.backend.orders_get(signal.symbol)
        if symbol_orders is None:
            raise OrderValidationError(
                f"MT5 orders_get('{signal.symbol}') returned None during pre-trade safety verification."
            )
        for ord_rec in symbol_orders:
            ord_tokens = extract_comment_identifiers(ord_rec.comment)
            if cleaned_cid in ord_tokens or cleaned_sig_id in ord_tokens:
                raise DuplicateOrderError(
                    f"Duplicate working order on broker: order ticket={ord_rec.ticket} already carries "
                    f"comment='{ord_rec.comment}' matching '{cleaned_cid}' / '{cleaned_sig_id}'."
                )
            ord_dir = SignalDirection.BUY if (ord_rec.order_type % 2 == 0) else SignalDirection.SELL
            if ord_dir != signal.direction:
                raise DuplicateOrderError(
                    f"Conflicting working order on broker (ticket={ord_rec.ticket}, symbol={ord_rec.symbol}, "
                    f"magic={ord_rec.magic}, direction={ord_dir.value}) opposes requested direction "
                    f"({signal.direction.value})."
                )
            if ord_rec.magic == self._safety_config.magic_number:
                raise DuplicateOrderError(
                    f"Existing ATLAS working order on broker (ticket={ord_rec.ticket}, symbol={ord_rec.symbol}, "
                    f"comment='{ord_rec.comment}') blocks new order submission on '{signal.symbol}'."
                )

        # Check pre-registered signal in SignalLifecycleManager if already present
        if self._lifecycle_manager is not None:
            try:
                managed_rec = self._lifecycle_manager.get_record(cleaned_sig_id)
            except SignalLifecycleError:
                managed_rec = None

            if managed_rec is not None:
                if (
                    managed_rec.state != SignalLifecycleState.ACTIVE
                    and managed_rec.expires_at is not None
                    and now_utc >= managed_rec.expires_at
                ):
                    if managed_rec.state not in (
                        SignalLifecycleState.EXPIRED,
                        SignalLifecycleState.CANCELLED,
                        SignalLifecycleState.RISK_REJECTED,
                    ):
                        self._lifecycle_manager.transition(
                            cleaned_sig_id,
                            SignalLifecycleState.EXPIRED,
                            now_utc,
                            f"Signal expired at {managed_rec.expires_at.isoformat()} prior to order execution.",
                        )
                    raise OrderValidationError(
                        f"Signal '{cleaned_sig_id}' has expired at {managed_rec.expires_at.isoformat()}."
                    )

                if managed_rec.state == SignalLifecycleState.ACTIVE:
                    raise DuplicateOrderError(
                        f"Signal '{cleaned_sig_id}' is already ACTIVE in SignalLifecycleManager."
                    )
                if managed_rec.state == SignalLifecycleState.RISK_APPROVED:
                    raise DuplicateOrderError(
                        f"Signal '{cleaned_sig_id}' is already in RISK_APPROVED state in SignalLifecycleManager."
                    )
                if managed_rec.state == SignalLifecycleState.PENDING_CONFIRMATION:
                    raise OrderValidationError(
                        f"Signal '{cleaned_sig_id}' is in PENDING_CONFIRMATION state and has not been CONFIRMED."
                    )
                if managed_rec.state in (
                    SignalLifecycleState.RISK_REJECTED,
                    SignalLifecycleState.CANCELLED,
                    SignalLifecycleState.EXPIRED,
                    SignalLifecycleState.TARGET_HIT,
                    SignalLifecycleState.STOP_LOSS_HIT,
                ):
                    raise OrderValidationError(
                        f"Signal '{cleaned_sig_id}' is in terminal lifecycle state {managed_rec.state.value}."
                    )

        # 6. Symbol availability, ENUM_SYMBOL_TRADE_MODE permissions & InstrumentSpec contract metadata
        symbol_info = self._connection.get_symbol_info(signal.symbol)
        if symbol_info.trade_mode == MT5SymbolTradeMode.SYMBOL_TRADE_MODE_DISABLED:
            raise OrderValidationError(
                f"Trading is disabled for symbol '{signal.symbol}' (SYMBOL_TRADE_MODE_DISABLED=0)."
            )
        if symbol_info.trade_mode == MT5SymbolTradeMode.SYMBOL_TRADE_MODE_CLOSEONLY:
            raise OrderValidationError(
                f"Symbol '{signal.symbol}' is in CLOSEONLY mode (SYMBOL_TRADE_MODE_CLOSEONLY=3); "
                "opening new positions is forbidden."
            )
        if (
            symbol_info.trade_mode == MT5SymbolTradeMode.SYMBOL_TRADE_MODE_LONGONLY
            and signal.direction != SignalDirection.BUY
        ):
            raise OrderValidationError(
                f"Symbol '{signal.symbol}' is restricted to LONGONLY (SYMBOL_TRADE_MODE_LONGONLY=1); "
                f"{signal.direction.value} orders are forbidden."
            )
        if (
            symbol_info.trade_mode == MT5SymbolTradeMode.SYMBOL_TRADE_MODE_SHORTONLY
            and signal.direction != SignalDirection.SELL
        ):
            raise OrderValidationError(
                f"Symbol '{signal.symbol}' is restricted to SHORTONLY (SYMBOL_TRADE_MODE_SHORTONLY=2); "
                f"{signal.direction.value} orders are forbidden."
            )
        instrument_spec = self._connection.get_instrument_spec(signal.symbol)

        # 7. Real-time tick freshness & stale market data check
        tick = self._market_data.fetch_latest_tick(
            signal.symbol,
            reference_now_utc=now_utc,
        )

        # 8. Bid-Ask spread limit check (measured consistently in symbol_info.point units)
        spread_points = tick.spread_points(symbol_info.point)
        if spread_points > self._safety_config.max_allowed_spread_points:
            raise OrderValidationError(
                f"Live Bid-Ask spread for '{signal.symbol}' ({spread_points} points) exceeds "
                f"max_allowed_spread_points ({self._safety_config.max_allowed_spread_points})."
            )

        # 9. Live execution price drift & directional SL/TP geometry at live market price
        live_price = tick.ask if signal.direction == SignalDirection.BUY else tick.bid
        price_drift_points = abs(live_price - signal.entry_price) / symbol_info.point
        if price_drift_points > Decimal(self._safety_config.max_entry_deviation_points):
            raise OrderValidationError(
                f"Live execution price ({live_price}) for '{signal.symbol}' drifted "
                f"{price_drift_points:.1f} points from signal.entry_price ({signal.entry_price}), "
                f"exceeding max_entry_deviation_points ({self._safety_config.max_entry_deviation_points})."
            )

        if signal.direction == SignalDirection.BUY:
            if not (signal.stop_loss < live_price < signal.take_profit):
                raise OrderValidationError(
                    f"BUY signal SL/TP invalid at live Ask ({live_price}): require "
                    f"stop_loss ({signal.stop_loss}) < Ask ({live_price}) < take_profit ({signal.take_profit})."
                )
        else:
            if not (signal.take_profit < live_price < signal.stop_loss):
                raise OrderValidationError(
                    f"SELL signal SL/TP invalid at live Bid ({live_price}): require "
                    f"take_profit ({signal.take_profit}) < Bid ({live_price}) < stop_loss ({signal.stop_loss})."
                )

        # 10. Broker stop-level (trade_stops_level) & freeze-level (trade_freeze_level) check
        min_level_points = max(symbol_info.trade_stops_level, symbol_info.trade_freeze_level)
        min_stop_distance = Decimal(min_level_points) * symbol_info.point
        sl_distance = abs(live_price - signal.stop_loss)
        tp_distance = abs(signal.take_profit - live_price)

        if sl_distance < min_stop_distance:
            raise OrderValidationError(
                f"Stop-loss distance ({sl_distance}) for '{signal.symbol}' is below broker "
                f"stop/freeze level minimum distance ({min_stop_distance}, {min_level_points} points)."
            )
        if tp_distance < min_stop_distance:
            raise OrderValidationError(
                f"Take-profit distance ({tp_distance}) for '{signal.symbol}' is below broker "
                f"stop/freeze level minimum distance ({min_stop_distance}, {min_level_points} points)."
            )

        # 11. Reconstruct verified AccountState & evaluate Phase 1 deterministic RiskEngine
        account_state = self._account_tracker.reconstruct_account_state(
            reference_now_utc=now_utc,
            symbol=signal.symbol,
        )
        live_signal = signal.model_copy(update={"entry_price": live_price})
        active_risk_params = risk_parameters or self._risk_engine.risk_parameters

        risk_decision = self._risk_engine.evaluate(
            signal=live_signal,
            account=account_state,
            risk_parameters=active_risk_params,
            instrument_spec=instrument_spec,
        )

        if not risk_decision.approved:
            if self._lifecycle_manager is not None:
                try:
                    rec_now = self._lifecycle_manager.get_record(cleaned_sig_id)
                except SignalLifecycleError:
                    try:
                        rec_now = self._lifecycle_manager.register_signal(
                            signal,
                            signal_id=cleaned_sig_id,
                        )
                    except SignalLifecycleError as dup_exc:
                        if "Duplicate" in str(dup_exc):
                            raise DuplicateOrderError(str(dup_exc)) from dup_exc
                        raise OrderValidationError(str(dup_exc)) from dup_exc

                if rec_now.state in (
                    SignalLifecycleState.GENERATED,
                    SignalLifecycleState.CONFIRMED,
                ):
                    self._lifecycle_manager.apply_risk_decision(
                        cleaned_sig_id,
                        risk_decision,
                        now_utc,
                    )
            raise OrderValidationError(
                f"RiskEngine rejected signal '{cleaned_sig_id}': {risk_decision.rejection_reason}"
            )

        try:
            # 12. Volume minimum, maximum, step, and RiskEngine upper-bound verification
            if requested_volume is not None:
                if requested_volume <= Decimal("0"):
                    raise OrderValidationError(
                        f"Requested volume ({requested_volume}) must be > 0."
                    )
                if (
                    requested_volume < symbol_info.volume_min
                    or requested_volume < instrument_spec.minimum_volume
                ):
                    raise OrderValidationError(
                        f"Requested volume ({requested_volume}) is below minimum allowed volume "
                        f"({symbol_info.volume_min})."
                    )
                max_allowed_vol = min(
                    symbol_info.volume_max,
                    instrument_spec.maximum_volume,
                    active_risk_params.max_position_size,
                )
                if requested_volume > max_allowed_vol:
                    raise OrderValidationError(
                        f"Requested volume ({requested_volume}) exceeds maximum allowed volume ({max_allowed_vol})."
                    )
                if (requested_volume % symbol_info.volume_step) != Decimal("0"):
                    raise OrderValidationError(
                        f"Requested volume ({requested_volume}) is not an exact multiple of "
                        f"volume_step ({symbol_info.volume_step})."
                    )
                if requested_volume > risk_decision.calculated_position_size:
                    raise OrderValidationError(
                        f"Requested volume ({requested_volume}) exceeds RiskEngine-approved position size "
                        f"({risk_decision.calculated_position_size})."
                    )
                approved_volume = requested_volume
                if approved_volume != risk_decision.calculated_position_size:
                    stop_distance = abs(live_price - signal.stop_loss)
                    ticks_at_risk = stop_distance / instrument_spec.tick_size
                    loss_per_lot = ticks_at_risk * instrument_spec.tick_value
                    raw_risk_amount = approved_volume * loss_per_lot
                    actual_risk_amount = raw_risk_amount.quantize(
                        Decimal("0.01"), rounding=ROUND_DOWN
                    )
                    actual_risk_pct = (
                        (raw_risk_amount / account_state.equity) * Decimal("100")
                    ).quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
                    updated_details = dict(risk_decision.validation_details)
                    updated_details["requested_volume_override"] = str(approved_volume)
                    updated_details["rounded_position_size"] = str(approved_volume)
                    updated_details["actual_risk_amount"] = str(actual_risk_amount)
                    risk_decision = risk_decision.model_copy(
                        update={
                            "calculated_position_size": approved_volume,
                            "calculated_risk_amount": actual_risk_amount,
                            "risk_percent": actual_risk_pct,
                            "validation_details": updated_details,
                        }
                    )
            else:
                approved_volume = risk_decision.calculated_position_size
                if (
                    approved_volume < symbol_info.volume_min
                    or approved_volume > symbol_info.volume_max
                    or (approved_volume % symbol_info.volume_step) != Decimal("0")
                ):
                    raise OrderValidationError(
                        f"Calculated volume ({approved_volume}) violates broker volume constraints "
                        f"(min={symbol_info.volume_min}, max={symbol_info.volume_max}, step={symbol_info.volume_step})."
                    )

            # 13. Execution-mode-aware filling policy resolution
            filling_mode = resolve_supported_filling_mode(symbol_info)

            # 14. Build MT5OrderRequest and verify via pre-flight MT5 order_check()
            order_type = (
                MT5OrderType.ORDER_TYPE_BUY
                if signal.direction == SignalDirection.BUY
                else MT5OrderType.ORDER_TYPE_SELL
            )
            order_request = MT5OrderRequest(
                action=MT5TradeAction.TRADE_ACTION_DEAL,
                symbol=signal.symbol,
                volume=approved_volume,
                order_type=order_type,
                price=live_price,
                sl=signal.stop_loss,
                tp=signal.take_profit,
                deviation=self._safety_config.max_entry_deviation_points,
                magic=self._safety_config.magic_number,
                comment=cleaned_cid,
                type_time=MT5OrderTimeMode.ORDER_TIME_GTC,
                type_filling=filling_mode,
            )

            order_check_result = self._connection.backend.order_check(order_request)
            if order_check_result is None:
                raise OrderValidationError(
                    "MT5 order_check() returned None; refusing to submit order."
                )
            if (
                order_check_result.retcode not in (0, int(MT5Retcode.TRADE_RETCODE_DONE))
                or order_check_result.margin_free < Decimal("0")
            ):
                raise OrderValidationError(
                    f"MT5 order_check() rejected order request: retcode={order_check_result.retcode}, "
                    f"comment='{order_check_result.comment}', margin_free={order_check_result.margin_free}."
                )

            if self._lifecycle_manager is not None:
                try:
                    rec_now = self._lifecycle_manager.get_record(cleaned_sig_id)
                except SignalLifecycleError:
                    try:
                        rec_now = self._lifecycle_manager.register_signal(
                            signal,
                            signal_id=cleaned_sig_id,
                        )
                    except SignalLifecycleError as dup_exc:
                        if "Duplicate" in str(dup_exc):
                            raise DuplicateOrderError(str(dup_exc)) from dup_exc
                        raise OrderValidationError(str(dup_exc)) from dup_exc

                if rec_now.state in (
                    SignalLifecycleState.GENERATED,
                    SignalLifecycleState.CONFIRMED,
                ):
                    self._lifecycle_manager.apply_risk_decision(
                        cleaned_sig_id,
                        risk_decision,
                        now_utc,
                    )
        except Exception as post_risk_exc:
            if self._lifecycle_manager is not None:
                try:
                    cur_rec = self._lifecycle_manager.get_record(cleaned_sig_id)
                    if cur_rec.state in (
                        SignalLifecycleState.GENERATED,
                        SignalLifecycleState.CONFIRMED,
                        SignalLifecycleState.RISK_APPROVED,
                    ):
                        self._lifecycle_manager.transition(
                            cleaned_sig_id,
                            SignalLifecycleState.CANCELLED,
                            now_utc,
                            f"Pre-flight check failed after risk evaluation: {post_risk_exc}",
                        )
                except Exception:
                    pass
            raise

        return PreTradeSafetyValidationResult(
            client_order_id=cleaned_cid,
            signal_id=cleaned_sig_id,
            signal=signal,
            live_signal=live_signal,
            account_info=account_info,
            account_state=account_state,
            symbol_info=symbol_info,
            instrument_spec=instrument_spec,
            latest_tick=tick,
            spread_points=spread_points,
            price_drift_points=price_drift_points,
            filling_mode=filling_mode,
            approved_volume=approved_volume,
            risk_decision=risk_decision,
            order_request=order_request,
            order_check_result=order_check_result,
            validated_at_utc=now_utc,
        )
