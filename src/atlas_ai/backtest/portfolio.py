"""Deterministic Portfolio & AccountState Tracker for ATLAS AI Phase 4."""

from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from zoneinfo import ZoneInfo

from atlas_ai.backtest.config import BacktestConfig
from atlas_ai.backtest.exceptions import BacktestExecutionError
from atlas_ai.backtest.models import (
    BacktestTradeRecord,
    EquityPoint,
    SimulatedPosition,
)
from atlas_ai.domain.risk import AccountState
from atlas_ai.strategies.sessions import compute_trading_day_label

QUANTIZE_MONEY = Decimal("0.01")
QUANTIZE_PCT = Decimal("0.0001")
HUNDRED = Decimal("100")


class PortfolioTracker:
    """Tracks balance, floating PnL, equity, margin reservation/release, trading-day rollover,
    and peak balance/equity high-water marks using exact Decimal arithmetic.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config
        self._tz = ZoneInfo(config.day_boundary_timezone)
        self._balance: Decimal = config.initial_balance.quantize(QUANTIZE_MONEY)
        self._daily_starting_balance: Decimal = self._balance
        self._peak_balance: Decimal = self._balance
        self._current_trading_day: date | None = None
        self._open_positions: dict[str, SimulatedPosition] = {}
        self._closed_trades: list[BacktestTradeRecord] = []
        self._equity_curve: list[EquityPoint] = []

    @property
    def balance(self) -> Decimal:
        """Current closed-trade account balance."""
        return self._balance

    @property
    def floating_pnl(self) -> Decimal:
        """Total unrealized net PnL across all open positions."""
        total = sum((pos.floating_pnl for pos in self._open_positions.values()), Decimal("0.00"))
        return total.quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)

    @property
    def equity(self) -> Decimal:
        """Current floating account equity (`balance + floating_pnl`)."""
        return (self._balance + self.floating_pnl).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)

    @property
    def used_margin(self) -> Decimal:
        """Total initial margin currently reserved by open positions."""
        total = sum((pos.used_margin for pos in self._open_positions.values()), Decimal("0.00"))
        return total.quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)

    @property
    def free_margin(self) -> Decimal:
        """Available free margin (`max(0, equity - used_margin)`)."""
        avail = self.equity - self.used_margin
        return max(Decimal("0.00"), avail).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)

    @property
    def open_positions_count(self) -> int:
        """Number of currently open positions."""
        return len(self._open_positions)

    @property
    def open_positions(self) -> tuple[SimulatedPosition, ...]:
        """Tuple of currently open positions in insertion order."""
        return tuple(self._open_positions.values())

    @property
    def closed_trades(self) -> tuple[BacktestTradeRecord, ...]:
        """Tuple of all closed trade records."""
        return tuple(self._closed_trades)

    @property
    def equity_curve(self) -> tuple[EquityPoint, ...]:
        """Tuple of recorded bar-by-bar equity curve points."""
        return tuple(self._equity_curve)

    @property
    def daily_starting_balance(self) -> Decimal:
        """Reference starting balance/equity for the current trading day."""
        return self._daily_starting_balance

    @property
    def current_daily_pnl(self) -> Decimal:
        """Net realized + unrealized PnL for the current trading day (`equity - daily_starting_balance`)."""
        return (self.equity - self._daily_starting_balance).quantize(
            QUANTIZE_MONEY, rounding=ROUND_HALF_UP
        )

    @property
    def peak_balance(self) -> Decimal:
        """High-water mark across historical balance and equity."""
        return self._peak_balance

    @property
    def current_trading_day(self) -> date | None:
        """Current logical trading day label."""
        return self._current_trading_day

    def update_trading_day_rollover(self, timestamp: datetime) -> bool:
        """Check if `timestamp` crosses into a new trading day (default 5 PM NY rollover).

        When a new trading day begins:
        - `daily_starting_balance` is reset to the current `equity` (or `balance` if no open positions).
        - `current_daily_pnl` becomes `0.00` at the moment of rollover.

        Returns:
            True if a trading-day rollover occurred on this bar, False otherwise.
        """
        day_label = compute_trading_day_label(
            timestamp,
            self._tz,
            self._config.day_rollover_hour,
        )
        if self._current_trading_day is None:
            self._current_trading_day = day_label
            self._daily_starting_balance = self._balance
            return False

        if day_label > self._current_trading_day:
            self._current_trading_day = day_label
            # Reset daily starting reference to current equity (or balance if flat)
            ref = self.equity if self.open_positions_count > 0 else self._balance
            if ref > Decimal("0"):
                self._daily_starting_balance = ref
            self._update_peak()
            return True

        return False

    def _update_peak(self) -> None:
        """Update high-water mark from current balance and equity."""
        self._peak_balance = max(self._peak_balance, self._balance, self.equity)

    def add_open_position(self, position: SimulatedPosition) -> None:
        """Register a newly opened `SimulatedPosition` and reserve its margin."""
        if position.position_id in self._open_positions:
            raise BacktestExecutionError(
                f"Duplicate position_id '{position.position_id}' in PortfolioTracker."
            )
        if position.used_margin > self.free_margin:
            raise BacktestExecutionError(
                f"Insufficient free margin ({self.free_margin}) to reserve "
                f"{position.used_margin} for position '{position.position_id}'."
            )
        self._open_positions[position.position_id] = position
        self._update_peak()

    def update_open_position(self, position: SimulatedPosition) -> None:
        """Update an existing open position's mark-to-market state."""
        if position.position_id not in self._open_positions:
            raise BacktestExecutionError(
                f"Cannot update unknown open position '{position.position_id}'."
            )
        self._open_positions[position.position_id] = position
        self._update_peak()

    def record_closed_trade(self, trade: BacktestTradeRecord) -> None:
        """Remove closed position, release its margin, and settle `net_pnl` into `balance`."""
        if trade.position_id not in self._open_positions:
            raise BacktestExecutionError(
                f"Cannot close unknown open position '{trade.position_id}'."
            )
        del self._open_positions[trade.position_id]
        self._balance = (self._balance + trade.net_pnl).quantize(
            QUANTIZE_MONEY, rounding=ROUND_HALF_UP
        )
        self._closed_trades.append(trade)
        self._update_peak()

    def build_account_state(self) -> AccountState:
        """Construct an immutable `AccountState` snapshot for `RiskEngine` evaluation."""
        self._update_peak()
        safe_balance = max(Decimal("0.01"), self._balance)
        safe_equity = max(Decimal("0.01"), self.equity)
        safe_free_margin = min(safe_equity, max(Decimal("0.00"), safe_equity - self.used_margin))
        return AccountState(
            balance=safe_balance,
            equity=safe_equity,
            free_margin=safe_free_margin,
            currency=self._config.currency,
            open_positions_count=self.open_positions_count,
            current_daily_pnl=self.current_daily_pnl,
            daily_starting_balance=max(Decimal("0.01"), self._daily_starting_balance),
            peak_balance=max(Decimal("0.01"), self._peak_balance),
        )

    def record_equity_point(self, bar_index: int, timestamp: datetime) -> EquityPoint:
        """Record and return the bar-end `EquityPoint`."""
        if self._current_trading_day is None:
            self.update_trading_day_rollover(timestamp)
        assert self._current_trading_day is not None

        self._update_peak()
        eq = max(Decimal("0.01"), self.equity)
        bal = max(Decimal("0.01"), self._balance)
        peak = max(Decimal("0.01"), self._peak_balance)
        dd_usd = max(Decimal("0.00"), peak - eq).quantize(QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
        dd_pct = ((dd_usd / peak) * HUNDRED).quantize(QUANTIZE_PCT, rounding=ROUND_HALF_UP)

        point = EquityPoint(
            bar_index=bar_index,
            timestamp=timestamp,
            trading_day=self._current_trading_day.isoformat(),
            balance=bal,
            floating_pnl=self.floating_pnl,
            equity=eq,
            used_margin=self.used_margin,
            free_margin=min(eq, self.free_margin),
            open_positions_count=self.open_positions_count,
            current_daily_pnl=self.current_daily_pnl,
            daily_starting_balance=max(Decimal("0.01"), self._daily_starting_balance),
            peak_balance=peak,
            drawdown_usd=dd_usd,
            drawdown_pct=dd_pct,
        )
        self._equity_curve.append(point)
        return point
