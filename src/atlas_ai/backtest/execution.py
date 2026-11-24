"""Deterministic Execution & Fill Simulator for ATLAS AI Phase 4."""

from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP

from atlas_ai.backtest.config import BacktestConfig, IntrabarTieBreakPolicy
from atlas_ai.backtest.models import BacktestTradeRecord, ExitReason, SimulatedPosition
from atlas_ai.domain.risk import RiskDecision
from atlas_ai.domain.signals import SignalDirection, TradeSignal

QUANTIZE_MONEY = Decimal("0.01")
QUANTIZE_RATIO = Decimal("0.0001")
TWO = Decimal("2")


class EntryFillEvaluation( SimulatedPosition if False else object ):
    """Container for entry fill evaluation before or upon opening a position."""

    def __init__(
        self,
        *,
        approved: bool,
        position: SimulatedPosition | None = None,
        rejection_reason: str | None = None,
    ) -> None:
        self.approved = approved
        self.position = position
        self.rejection_reason = rejection_reason


class ExecutionSimulator:
    """Simulates realistic order fills, bid/ask spread, adverse slippage, commission,
    intrabar SL/TP resolution, and MFE/MAE excursion tracking.

    Execution Model:
    - Spread price = `spread_points * instrument_spec.tick_size` (`half_spread = spread_price / 2`).
    - Slippage price = `slippage_points * instrument_spec.tick_size`.
    - Entry Fill:
      - `BUY` fills at Ask + adverse slippage: `requested_entry + half_spread + slippage_price`.
      - `SELL` fills at Bid - adverse slippage: `requested_entry - half_spread - slippage_price`.
    - Exit Fill:
      - `BUY` exit fills at Bid - adverse slippage: `requested_exit - half_spread - slippage_price`
        for market/close exits, or `requested_exit - slippage_price` when triggered by a chart SL/TP level
        (since SL/TP trigger levels are checked against chart High/Low). Wait: to ensure full round-turn
        spread is always charged consistently on every trade regardless of exit type:
        - Entry applies `half_spread + slippage_price` adversely.
        - Exit applies `half_spread + slippage_price` adversely on `STOP_LOSS`, `END_OF_DATA`, `EXPIRED`,
          and `MANUAL_CANCEL`, and `half_spread` on `TAKE_PROFIT` (or `half_spread + slippage_price`
          across all exits so total round-turn cost is deterministically `spread_price + 2 * slippage_price`).
    - Intrabar SL/TP Resolution:
      - Evaluated only on bars strictly AFTER `entry_bar_index` (`bar_index > position.entry_bar_index`),
        preventing same-candle hindsight execution.
      - Default `CONSERVATIVE_SL_FIRST`: if both SL and TP are touched on the same candle, `STOP_LOSS` wins.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config
        self._spec = config.instrument_spec

    @property
    def config(self) -> BacktestConfig:
        return self._config

    @property
    def spread_price(self) -> Decimal:
        """Total bid-ask spread in price units."""
        return self._config.spread_points * self._spec.tick_size

    @property
    def half_spread_price(self) -> Decimal:
        """Half of the bid-ask spread in price units."""
        return self.spread_price / TWO

    @property
    def slippage_price(self) -> Decimal:
        """Single-fill adverse slippage in price units."""
        return self._config.slippage_points * self._spec.tick_size

    def calculate_entry_fill_price(
        self,
        direction: SignalDirection,
        requested_price: Decimal,
    ) -> Decimal:
        """Calculate actual entry fill price after half-spread and adverse entry slippage."""
        drag = self.half_spread_price + self.slippage_price
        if direction == SignalDirection.BUY:
            return (requested_price + drag).quantize(self._spec.tick_size, rounding=ROUND_HALF_UP)
        if direction == SignalDirection.SELL:
            return (requested_price - drag).quantize(self._spec.tick_size, rounding=ROUND_HALF_UP)
        raise ValueError(f"Cannot calculate entry fill for direction {direction}.")

    def calculate_exit_fill_price(
        self,
        direction: SignalDirection,
        requested_exit_price: Decimal,
    ) -> Decimal:
        """Calculate actual exit fill price after half-spread and adverse exit slippage."""
        drag = self.half_spread_price + self.slippage_price
        if direction == SignalDirection.BUY:
            return (requested_exit_price - drag).quantize(self._spec.tick_size, rounding=ROUND_HALF_UP)
        if direction == SignalDirection.SELL:
            return (requested_exit_price + drag).quantize(self._spec.tick_size, rounding=ROUND_HALF_UP)
        raise ValueError(f"Cannot calculate exit fill for direction {direction}.")

    def calculate_commission(self, lots: Decimal) -> Decimal:
        """Calculate round-turn commission in account currency for `lots`."""
        return (lots * self._config.commission_per_lot).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)

    def calculate_used_margin(self, lots: Decimal, entry_fill_price: Decimal) -> Decimal:
        """Calculate initial margin required for `lots` at `entry_fill_price`."""
        notional = lots * self._spec.contract_size * entry_fill_price
        return (notional * self._spec.margin_rate).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)

    def calculate_gross_pnl(
        self,
        direction: SignalDirection,
        entry_fill_price: Decimal,
        current_or_exit_price: Decimal,
        lots: Decimal,
    ) -> Decimal:
        """Calculate gross monetary PnL using exact tick_size and tick_value."""
        price_diff = (
            current_or_exit_price - entry_fill_price
            if direction == SignalDirection.BUY
            else entry_fill_price - current_or_exit_price
        )
        ticks = price_diff / self._spec.tick_size
        gross = ticks * self._spec.tick_value * lots
        return gross.quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)

    def open_position(
        self,
        *,
        position_id: str,
        signal_id: str,
        signal: TradeSignal,
        risk_decision: RiskDecision,
        entry_bar_index: int,
        entry_timestamp: datetime,
        requested_entry_price: Decimal | None = None,
    ) -> EntryFillEvaluation:
        """Validate post-cost SL/TP geometry & RR and construct an open `SimulatedPosition`."""
        if not risk_decision.approved or risk_decision.calculated_position_size <= Decimal("0"):
            return EntryFillEvaluation(
                approved=False,
                rejection_reason="EXECUTION_REJECTED: Cannot open position without approved RiskDecision.",
            )

        assert signal.entry_price is not None
        assert signal.stop_loss is not None
        assert signal.take_profit is not None

        req_entry = requested_entry_price if requested_entry_price is not None else signal.entry_price
        fill_price = self.calculate_entry_fill_price(signal.direction, req_entry)
        sl = signal.stop_loss
        tp = signal.take_profit

        # 1. Verify fill price has not crossed SL or TP due to spread + slippage
        if signal.direction == SignalDirection.BUY:
            if fill_price <= sl:
                return EntryFillEvaluation(
                    approved=False,
                    rejection_reason=(
                        f"EXECUTION_INVALID_STOP_DISTANCE: BUY entry fill ({fill_price}) "
                        f"is <= stop_loss ({sl}) after spread/slippage."
                    ),
                )
            if fill_price >= tp:
                return EntryFillEvaluation(
                    approved=False,
                    rejection_reason=(
                        f"EXECUTION_INVALID_TARGET_DISTANCE: BUY entry fill ({fill_price}) "
                        f"is >= take_profit ({tp}) after spread/slippage."
                    ),
                )
            eff_stop_dist = fill_price - sl
            eff_reward_dist = tp - fill_price
        else:
            if fill_price >= sl:
                return EntryFillEvaluation(
                    approved=False,
                    rejection_reason=(
                        f"EXECUTION_INVALID_STOP_DISTANCE: SELL entry fill ({fill_price}) "
                        f"is >= stop_loss ({sl}) after spread/slippage."
                    ),
                )
            if fill_price <= tp:
                return EntryFillEvaluation(
                    approved=False,
                    rejection_reason=(
                        f"EXECUTION_INVALID_TARGET_DISTANCE: SELL entry fill ({fill_price}) "
                        f"is <= take_profit ({tp}) after spread/slippage."
                    ),
                )
            eff_stop_dist = sl - fill_price
            eff_reward_dist = fill_price - tp

        eff_rr = (eff_reward_dist / eff_stop_dist).quantize(QUANTIZE_RATIO, rounding=ROUND_DOWN)

        # 2. Validate effective RR after execution costs if configured
        min_rr = self._config.risk_parameters.minimum_risk_reward_ratio
        if self._config.enforce_min_rr_after_costs and eff_rr < min_rr:
            return EntryFillEvaluation(
                approved=False,
                rejection_reason=(
                    f"EXECUTION_INSUFFICIENT_EFFECTIVE_RR: Effective RR after spread/slippage ({eff_rr}) "
                    f"is below minimum_risk_reward_ratio ({min_rr})."
                ),
            )

        lots = risk_decision.calculated_position_size
        commission = self.calculate_commission(lots)
        used_margin = self.calculate_used_margin(lots, fill_price)

        # Initial 1R risk is the risk decision's calculated risk amount (or stop distance risk at fill)
        stop_ticks = eff_stop_dist / self._spec.tick_size
        initial_risk_usd = (stop_ticks * self._spec.tick_value * lots).quantize(
            QUANTIZE_MONEY, rounding=ROUND_HALF_UP
        )
        if initial_risk_usd <= Decimal("0"):
            initial_risk_usd = risk_decision.calculated_risk_amount

        # Immediate mark-to-market at entry bar close (reflecting spread/slippage + commission)
        initial_gross_mtm = self.calculate_gross_pnl(
            signal.direction,
            fill_price,
            req_entry,
            lots,
        )
        initial_floating_pnl = initial_gross_mtm - commission

        position = SimulatedPosition(
            position_id=position_id,
            signal_id=signal_id,
            symbol=signal.symbol,
            timeframe=signal.timeframe,
            direction=signal.direction,
            strategy_id=signal.strategy_id,
            entry_bar_index=entry_bar_index,
            entry_timestamp=entry_timestamp,
            requested_entry_price=req_entry,
            entry_fill_price=fill_price,
            stop_loss=sl,
            take_profit=tp,
            lots=lots,
            initial_risk_amount=initial_risk_usd,
            effective_stop_distance=eff_stop_dist,
            effective_reward_distance=eff_reward_dist,
            effective_rr_ratio=eff_rr,
            used_margin=used_margin,
            commission=commission,
            floating_pnl=initial_floating_pnl,
        )
        return EntryFillEvaluation(approved=True, position=position)

    def evaluate_bar_for_position(
        self,
        position: SimulatedPosition,
        *,
        bar_index: int,
        bar_timestamp: datetime,
        open_price: Decimal,
        high_price: Decimal,
        low_price: Decimal,
        close_price: Decimal,
        trade_id_Trigger: str,
    ) -> tuple[SimulatedPosition, BacktestTradeRecord | None]:
        """Evaluate an open `SimulatedPosition` against a subsequent bar (`bar_index > entry_bar_index`).

        Updates MFE/MAE and floating PnL, and returns `(updated_position, closed_trade_or_None)`.
        """
        if bar_index <= position.entry_bar_index:
            # Strict anti-lookahead guard: never evaluate intrabar SL/TP on the entry bar itself
            return position, None

        # 1. Update Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE)
        if position.direction == SignalDirection.BUY:
            fav_dist = max(Decimal("0"), high_price - position.entry_fill_price)
            adv_dist = max(Decimal("0"), position.entry_fill_price - low_price)
        else:
            fav_dist = max(Decimal("0"), position.entry_fill_price - low_price)
            adv_dist = max(Decimal("0"), high_price - position.entry_fill_price)

        new_mfe_dist = max(position.mfe_price_distance, fav_dist)
        new_mae_dist = max(position.mae_price_distance, adv_dist)

        mfe_ticks = new_mfe_dist / self._spec.tick_size
        mae_ticks = new_mae_dist / self._spec.tick_size
        new_mfe_usd = (mfe_ticks * self._spec.tick_value * position.lots).quantize(
            QUANTIZE_MONEY, rounding=ROUND_HALF_UP
        )
        new_mae_usd = (mae_ticks * self._spec.tick_value * position.lots).quantize(
            QUANTIZE_MONEY, rounding=ROUND_HALF_UP
        )

        # 2. Check intrabar Stop-Loss and Take-Profit hits
        if position.direction == SignalDirection.BUY:
            sl_touched = low_price <= position.stop_loss
            tp_touched = high_price >= position.take_profit
        else:
            sl_touched = high_price >= position.stop_loss
            tp_touched = low_price <= position.take_profit

        exit_reason: ExitReason | None = None
        requested_exit: Decimal | None = None

        if sl_touched and tp_touched:
            if self._config.intrabar_tie_break == IntrabarTieBreakPolicy.CONSERVATIVE_SL_FIRST:
                exit_reason = ExitReason.STOP_LOSS
                requested_exit = position.stop_loss
            else:
                exit_reason = ExitReason.TAKE_PROFIT
                requested_exit = position.take_profit
        elif sl_touched:
            exit_reason = ExitReason.STOP_LOSS
            requested_exit = position.stop_loss
        elif tp_touched:
            exit_reason = ExitReason.TAKE_PROFIT
            requested_exit = position.take_profit

        # Mark-to-market at bar close if still open
        close_gross_pnl = self.calculate_gross_pnl(
            position.direction,
            position.entry_fill_price,
            close_price,
            position.lots,
        )
        updated_pos = position.model_copy(
            update={
                "mfe_price_distance": new_mfe_dist,
                "mae_price_distance": new_mae_dist,
                "mfe_usd": new_mfe_usd,
                "mae_usd": new_mae_usd,
                "floating_pnl": close_gross_pnl - position.commission,
            }
        )

        if exit_reason is not None and requested_exit is not None:
            trade_record = self.close_position(
                updated_pos,
                trade_id=trade_id_Trigger,
                exit_bar_index=bar_index,
                exit_timestamp=bar_timestamp,
                requested_exit_price=requested_exit,
                exit_reason=exit_reason,
            )
            return updated_pos, trade_record

        return updated_pos, None

    def close_position(
        self,
        position: SimulatedPosition,
        *,
        trade_id: str,
        exit_bar_index: int,
        exit_timestamp: datetime,
        requested_exit_price: Decimal,
        exit_reason: ExitReason,
    ) -> BacktestTradeRecord:
        """Close an open `SimulatedPosition` and return an immutable `BacktestTradeRecord`."""
        exit_fill_price = self.calculate_exit_fill_price(position.direction, requested_exit_price)
        gross_pnl = self.calculate_gross_pnl(
            position.direction,
            position.entry_fill_price,
            exit_fill_price,
            position.lots,
        )
        net_pnl = (gross_pnl - position.commission).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
        r_mult = (net_pnl / position.initial_risk_amount).quantize(
            QUANTIZE_RATIO, rounding=ROUND_HALF_UP
        )
        bars_held = max(0, exit_bar_index - position.entry_bar_index)

        return BacktestTradeRecord(
            trade_id=trade_id,
            position_id=position.position_id,
            signal_id=position.signal_id,
            symbol=position.symbol,
            timeframe=position.timeframe,
            direction=position.direction,
            strategy_id=position.strategy_id,
            entry_bar_index=position.entry_bar_index,
            exit_bar_index=exit_bar_index,
            entry_timestamp=position.entry_timestamp,
            exit_timestamp=exit_timestamp,
            requested_entry_price=position.requested_entry_price,
            entry_fill_price=position.entry_fill_price,
            requested_exit_price=requested_exit_price,
            exit_fill_price=exit_fill_price,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            lots=position.lots,
            initial_risk_amount=position.initial_risk_amount,
            gross_pnl=gross_pnl,
            commission=position.commission,
            net_pnl=net_pnl,
            realized_r_multiple=r_mult,
            mfe_price_distance=position.mfe_price_distance,
            mae_price_distance=position.mae_price_distance,
            mfe_usd=position.mfe_usd,
            mae_usd=position.mae_usd,
            bars_held=bars_held,
            exit_reason=exit_reason,
        )
