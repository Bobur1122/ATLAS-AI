"""Domain models for the ATLAS AI Phase 4 Historical Backtesting Engine."""

from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atlas_ai.backtest.config import BacktestConfig
from atlas_ai.domain.market import Timeframe
from atlas_ai.domain.risk import AccountState, RiskDecision
from atlas_ai.domain.signals import SignalDirection
from atlas_ai.strategies.models import ManagedSignalRecord


class ExitReason(StrEnum):
    """Standardized reasons for closing a simulated position during a backtest."""

    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    EXPIRED = "EXPIRED"
    END_OF_DATA = "END_OF_DATA"
    MANUAL_CANCEL = "MANUAL_CANCEL"


class SimulatedPosition(BaseModel):
    """Immutable snapshot of an open simulated position during backtesting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    position_id: str = Field(..., min_length=1, description="Unique deterministic position identifier.")
    signal_id: str = Field(..., min_length=1, description="Lifecycle signal_id that originated this position.")
    symbol: str = Field(..., min_length=3, max_length=16)
    timeframe: Timeframe
    direction: SignalDirection = Field(..., description="BUY or SELL (never WAIT).")
    strategy_id: str = Field(..., min_length=1)
    entry_bar_index: int = Field(..., ge=0, description="Bar index at whose close the position was opened.")
    entry_timestamp: datetime = Field(..., description="Timestamp when the position was opened.")
    requested_entry_price: Decimal = Field(..., gt=Decimal("0"))
    entry_fill_price: Decimal = Field(..., gt=Decimal("0"), description="Actual entry fill price after spread and slippage.")
    stop_loss: Decimal = Field(..., gt=Decimal("0"))
    take_profit: Decimal = Field(..., gt=Decimal("0"))
    lots: Decimal = Field(..., gt=Decimal("0"))
    initial_risk_amount: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Initial 1R monetary risk in account currency at entry.",
    )
    effective_stop_distance: Decimal = Field(..., gt=Decimal("0"))
    effective_reward_distance: Decimal = Field(..., gt=Decimal("0"))
    effective_rr_ratio: Decimal = Field(..., gt=Decimal("0"))
    used_margin: Decimal = Field(..., ge=Decimal("0"))
    commission: Decimal = Field(..., ge=Decimal("0"), description="Round-turn commission for this position.")
    mfe_price_distance: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Maximum Favorable Excursion in price distance from entry_fill_price.",
    )
    mae_price_distance: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Maximum Adverse Excursion in price distance from entry_fill_price.",
    )
    mfe_usd: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Maximum Favorable Excursion in account currency (gross).",
    )
    mae_usd: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Maximum Adverse Excursion in account currency (gross).",
    )
    floating_pnl: Decimal = Field(
        default=Decimal("0"),
        description="Current marked-to-market net unrealized PnL (including commission).",
    )

    @model_validator(mode="after")
    def validate_open_position_direction(self) -> "SimulatedPosition":
        """Ensure only BUY or SELL positions can exist and fill price stays within SL/TP bounds."""
        if self.direction not in (SignalDirection.BUY, SignalDirection.SELL):
            raise ValueError(f"SimulatedPosition direction must be BUY or SELL, got {self.direction}.")
        if self.direction == SignalDirection.BUY and not (self.stop_loss < self.entry_fill_price < self.take_profit):
            raise ValueError(
                f"BUY SimulatedPosition requires stop_loss ({self.stop_loss}) "
                f"< entry_fill_price ({self.entry_fill_price}) < take_profit ({self.take_profit})."
            )
        if self.direction == SignalDirection.SELL and not (self.take_profit < self.entry_fill_price < self.stop_loss):
            raise ValueError(
                f"SELL SimulatedPosition requires take_profit ({self.take_profit}) "
                f"< entry_fill_price ({self.entry_fill_price}) < stop_loss ({self.stop_loss})."
            )
        return self


class BacktestTradeRecord(BaseModel):
    """Immutable audit record of a closed trade in the backtest ledger."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trade_id: str = Field(..., min_length=1)
    position_id: str = Field(..., min_length=1)
    signal_id: str = Field(..., min_length=1)
    symbol: str = Field(..., min_length=3, max_length=16)
    timeframe: Timeframe
    direction: SignalDirection
    strategy_id: str = Field(..., min_length=1)
    entry_bar_index: int = Field(..., ge=0)
    exit_bar_index: int = Field(..., ge=0)
    entry_timestamp: datetime
    exit_timestamp: datetime
    requested_entry_price: Decimal = Field(..., gt=Decimal("0"))
    entry_fill_price: Decimal = Field(..., gt=Decimal("0"))
    requested_exit_price: Decimal = Field(..., gt=Decimal("0"))
    exit_fill_price: Decimal = Field(..., gt=Decimal("0"))
    stop_loss: Decimal = Field(..., gt=Decimal("0"))
    take_profit: Decimal = Field(..., gt=Decimal("0"))
    lots: Decimal = Field(..., gt=Decimal("0"))
    initial_risk_amount: Decimal = Field(..., gt=Decimal("0"), description="Initial 1R monetary risk in USD.")
    gross_pnl: Decimal = Field(..., description="Realized PnL before commission.")
    commission: Decimal = Field(..., ge=Decimal("0"), description="Total round-turn commission charged.")
    net_pnl: Decimal = Field(..., description="Realized PnL after subtracting commission (gross_pnl - commission).")
    realized_r_multiple: Decimal = Field(..., description="net_pnl / initial_risk_amount.")
    mfe_price_distance: Decimal = Field(..., ge=Decimal("0"))
    mae_price_distance: Decimal = Field(..., ge=Decimal("0"))
    mfe_usd: Decimal = Field(..., ge=Decimal("0"))
    mae_usd: Decimal = Field(..., ge=Decimal("0"))
    bars_held: int = Field(..., ge=0)
    exit_reason: ExitReason


class EquityPoint(BaseModel):
    """Immutable bar-end snapshot of account balance, equity, margin, and drawdown."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bar_index: int = Field(..., ge=0)
    timestamp: datetime
    trading_day: str = Field(..., min_length=1, description="Logical trading day ISO date string (YYYY-MM-DD).")
    balance: Decimal = Field(..., gt=Decimal("0"))
    floating_pnl: Decimal
    equity: Decimal = Field(..., gt=Decimal("0"))
    used_margin: Decimal = Field(..., ge=Decimal("0"))
    free_margin: Decimal = Field(..., ge=Decimal("0"))
    open_positions_count: int = Field(..., ge=0)
    current_daily_pnl: Decimal
    daily_starting_balance: Decimal = Field(..., gt=Decimal("0"))
    peak_balance: Decimal = Field(..., gt=Decimal("0"))
    drawdown_usd: Decimal = Field(..., ge=Decimal("0"))
    drawdown_pct: Decimal = Field(..., ge=Decimal("0"))


class RiskRejectionRecord(BaseModel):
    """Immutable audit record of a signal or execution candidate rejected during backtesting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bar_index: int = Field(..., ge=0)
    timestamp: datetime
    signal_id: str | None = Field(default=None)
    symbol: str
    direction: SignalDirection
    strategy_id: str
    rejection_reason: str
    account_state: AccountState
    risk_decision: RiskDecision | None = Field(default=None)


class BacktestMetrics(BaseModel):
    """Comprehensive, deterministic statistical performance summary of a backtest run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    initial_balance: Decimal
    final_balance: Decimal
    final_equity: Decimal
    total_trades: int = Field(..., ge=0)
    winning_trades: int = Field(..., ge=0)
    losing_trades: int = Field(..., ge=0)
    breakeven_trades: int = Field(..., ge=0)
    win_rate_pct: Decimal = Field(..., ge=Decimal("0"), le=Decimal("100"))
    gross_profit: Decimal = Field(..., ge=Decimal("0"))
    gross_loss: Decimal = Field(..., ge=Decimal("0"), description="Absolute magnitude of total gross losses (>= 0).")
    total_commission: Decimal = Field(..., ge=Decimal("0"))
    net_profit_usd: Decimal
    net_profit_pct: Decimal
    profit_factor: Decimal = Field(..., ge=Decimal("0"))
    expectancy_usd: Decimal
    expectancy_r: Decimal
    average_win_usd: Decimal = Field(..., ge=Decimal("0"))
    average_loss_usd: Decimal = Field(..., ge=Decimal("0"), description="Absolute magnitude of average losing trade (>= 0).")
    payoff_ratio: Decimal = Field(..., ge=Decimal("0"))
    average_r_multiple: Decimal
    max_drawdown_usd: Decimal = Field(..., ge=Decimal("0"))
    max_drawdown_pct: Decimal = Field(..., ge=Decimal("0"))
    sharpe_ratio: Decimal
    sortino_ratio: Decimal


class BacktestResult(BaseModel):
    """Immutable top-level result returned by BacktestEngine.run()."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    config: BacktestConfig
    start_timestamp: datetime | None = Field(default=None)
    end_timestamp: datetime | None = Field(default=None)
    total_bars: int = Field(..., ge=0)
    trades: tuple[BacktestTradeRecord, ...] = Field(default_factory=tuple)
    open_positions: tuple[SimulatedPosition, ...] = Field(default_factory=tuple)
    equity_curve: tuple[EquityPoint, ...] = Field(default_factory=tuple)
    risk_rejections: tuple[RiskRejectionRecord, ...] = Field(default_factory=tuple)
    lifecycle_records: tuple[ManagedSignalRecord, ...] = Field(default_factory=tuple)
    metrics: BacktestMetrics
