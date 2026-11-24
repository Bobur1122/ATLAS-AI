"""Pluggable MT5TerminalBackend Protocol, lazy RealMetaTrader5Backend, and offline MockMT5TerminalBackend."""

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import importlib
from typing import Any, Protocol, Sequence, runtime_checkable

from atlas_ai.domain.market import Timeframe
from atlas_ai.execution.exceptions import (
    LiveTradingBlockedError,
    MT5BackendNotAvailableError,
)
from atlas_ai.execution.models import (
    MT5AccountInfo,
    MT5DealEntry,
    MT5DealRecord,
    MT5DealType,
    MT5OrderCheckResult,
    MT5OrderFillingMode,
    MT5OrderRecord,
    MT5OrderRequest,
    MT5OrderSendResult,
    MT5OrderType,
    MT5PositionRecord,
    MT5RateBar,
    MT5Retcode,
    MT5SymbolFillingFlag,
    MT5SymbolInfo,
    MT5SymbolTradeMode,
    MT5TerminalInfo,
    MT5TradeExecutionMode,
    MT5TradeMode,
    SymbolTick,
)

# Standard MT5 integer timeframe constants
MT5_TIMEFRAME_CONSTANTS: dict[Timeframe, int] = {
    Timeframe.M1: 1,
    Timeframe.M5: 5,
    Timeframe.M15: 15,
    Timeframe.M30: 30,
    Timeframe.H1: 16385,
    Timeframe.H4: 16388,
    Timeframe.D1: 16408,
    Timeframe.W1: 32769,
    Timeframe.MN1: 49153,
}


@runtime_checkable
class MT5TerminalBackend(Protocol):
    """Protocol abstracting the MetaTrader 5 terminal interface for offline testability."""

    def initialize(
        self,
        *,
        path: str | None = None,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        timeout: int = 10000,
        portable: bool = False,
    ) -> bool:
        """Initialize connection to the MetaTrader 5 terminal."""

    def login(
        self,
        login: int,
        *,
        password: str | None = None,
        server: str | None = None,
        timeout: int = 10000,
    ) -> bool:
        """Authenticate with a specific trading account."""

    def shutdown(self) -> None:
        """Close the MetaTrader 5 terminal connection cleanly."""

    def last_error(self) -> tuple[int, str]:
        """Return `(error_code, error_description)` from the last terminal call."""

    def terminal_info(self) -> MT5TerminalInfo | None:
        """Return current terminal connectivity and permission metadata."""

    def account_info(self) -> MT5AccountInfo | None:
        """Return current trading account balances, trade_mode, and server metadata."""

    def symbol_select(self, symbol: str, enable: bool = True) -> bool:
        """Select or deselect a symbol in the terminal's MarketWatch window."""

    def symbol_info(self, symbol: str) -> MT5SymbolInfo | None:
        """Return contract, stop-level, trade_exemode, and filling_mode metadata for `symbol`."""

    def symbol_info_tick(self, symbol: str) -> SymbolTick | None:
        """Return the latest Bid/Ask tick quote for `symbol`."""

    def copy_rates_from_pos(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_pos: int,
        count: int,
    ) -> tuple[MT5RateBar, ...] | None:
        """Return `count` historical rate bars starting at `start_pos` (0 = current forming bar, 1 = last closed bar)."""

    def history_deals_get(
        self,
        date_from: datetime,
        date_to: datetime,
        *,
        symbol: str | None = None,
    ) -> tuple[MT5DealRecord, ...] | None:
        """Return historical deal records in `[date_from, date_to]`."""

    def orders_get(self, symbol: str | None = None) -> tuple[MT5OrderRecord, ...] | None:
        """Return currently active working orders, optionally filtered by `symbol`."""

    def positions_get(self, symbol: str | None = None) -> tuple[MT5PositionRecord, ...] | None:
        """Return currently open positions, optionally filtered by `symbol`."""

    def order_check(self, request: MT5OrderRequest) -> MT5OrderCheckResult | None:
        """Validate margin and trade request parameters without submitting the order."""

    def order_send(self, request: MT5OrderRequest) -> MT5OrderSendResult | None:
        """Submit a validated demo trade request to the terminal."""


class RealMetaTrader5Backend:
    """Lazy-loading adapter for the official `MetaTrader5` Python package.

    CRITICAL SAFETY & OFFLINE GUARANTEES:
    1. Never imports `MetaTrader5` at module import time; importing `atlas_ai` or running
       offline tests never requires the `MetaTrader5` package to be installed.
    2. `order_send()` performs an unconditional hardware-level check on `mt5.account_info().trade_mode`
       immediately before order transmission and raises `LiveTradingBlockedError` if `trade_mode != 0 (DEMO)`.
    """

    def __init__(self) -> None:
        self._mt5: Any = None

    def _get_mt5(self) -> Any:
        if self._mt5 is None:
            try:
                self._mt5 = importlib.import_module("MetaTrader5")
            except ImportError as exc:
                raise MT5BackendNotAvailableError(
                    "The 'MetaTrader5' Python package is not installed in this environment. "
                    "Use MockMT5TerminalBackend for offline testing or install MetaTrader5 on a Windows terminal host."
                ) from exc
        return self._mt5

    def initialize(
        self,
        *,
        path: str | None = None,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        timeout: int = 10000,
        portable: bool = False,
    ) -> bool:
        mt5 = self._get_mt5()
        kwargs: dict[str, Any] = {"timeout": timeout, "portable": portable}
        if login is not None:
            kwargs["login"] = login
        if password is not None:
            kwargs["password"] = password
        if server is not None:
            kwargs["server"] = server
        if path is not None:
            return bool(mt5.initialize(path, **kwargs))
        return bool(mt5.initialize(**kwargs))

    def login(
        self,
        login: int,
        *,
        password: str | None = None,
        server: str | None = None,
        timeout: int = 10000,
    ) -> bool:
        mt5 = self._get_mt5()
        kwargs: dict[str, Any] = {"timeout": timeout}
        if password is not None:
            kwargs["password"] = password
        if server is not None:
            kwargs["server"] = server
        return bool(mt5.login(login, **kwargs))

    def shutdown(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()

    def last_error(self) -> tuple[int, str]:
        mt5 = self._get_mt5()
        res = mt5.last_error()
        if isinstance(res, tuple) and len(res) >= 2:
            return (int(res[0]), str(res[1]))
        return (-1, "Unknown MT5 error")

    def terminal_info(self) -> MT5TerminalInfo | None:
        mt5 = self._get_mt5()
        raw = mt5.terminal_info()
        if raw is None:
            return None
        return MT5TerminalInfo(
            connected=bool(getattr(raw, "connected", False)),
            trade_allowed=bool(getattr(raw, "trade_allowed", False)),
            tradeapi_disabled=bool(getattr(raw, "tradeapi_disabled", False)),
            community_account=bool(getattr(raw, "community_account", False)),
            name=str(getattr(raw, "name", "MetaTrader 5")),
            company=str(getattr(raw, "company", "MetaQuotes")),
            path=str(getattr(raw, "path", "")),
            build=int(getattr(raw, "build", 0)),
        )

    def account_info(self) -> MT5AccountInfo | None:
        mt5 = self._get_mt5()
        raw = mt5.account_info()
        if raw is None:
            return None
        return MT5AccountInfo(
            login=int(raw.login),
            trade_mode=int(raw.trade_mode),
            leverage=int(getattr(raw, "leverage", 100)),
            limit_orders=int(getattr(raw, "limit_orders", 200)),
            trade_allowed=bool(getattr(raw, "trade_allowed", True)),
            trade_expert=bool(getattr(raw, "trade_expert", True)),
            balance=Decimal(str(raw.balance)),
            credit=Decimal(str(getattr(raw, "credit", 0.0))),
            profit=Decimal(str(getattr(raw, "profit", 0.0))),
            equity=Decimal(str(raw.equity)),
            margin=Decimal(str(getattr(raw, "margin", 0.0))),
            margin_free=Decimal(str(raw.margin_free)),
            currency=str(getattr(raw, "currency", "USD")),
            currency_digits=int(getattr(raw, "currency_digits", 2)),
            server=str(raw.server),
            name=str(getattr(raw, "name", "Trader")),
            company=str(getattr(raw, "company", "Broker")),
        )

    def symbol_select(self, symbol: str, enable: bool = True) -> bool:
        mt5 = self._get_mt5()
        return bool(mt5.symbol_select(symbol, enable))

    def symbol_info(self, symbol: str) -> MT5SymbolInfo | None:
        mt5 = self._get_mt5()
        raw = mt5.symbol_info(symbol)
        if raw is None:
            return None
        return MT5SymbolInfo(
            name=str(raw.name),
            visible=bool(getattr(raw, "visible", True)),
            select=bool(getattr(raw, "select", True)),
            digits=int(getattr(raw, "digits", 2)),
            point=Decimal(str(raw.point)),
            trade_contract_size=Decimal(str(raw.trade_contract_size)),
            trade_tick_size=Decimal(str(raw.trade_tick_size)),
            trade_tick_value=Decimal(str(raw.trade_tick_value)),
            volume_min=Decimal(str(raw.volume_min)),
            volume_max=Decimal(str(raw.volume_max)),
            volume_step=Decimal(str(raw.volume_step)),
            trade_stops_level=int(getattr(raw, "trade_stops_level", 0)),
            trade_freeze_level=int(getattr(raw, "trade_freeze_level", 0)),
            trade_mode=MT5SymbolTradeMode(int(getattr(raw, "trade_mode", 4))),
            trade_exemode=MT5TradeExecutionMode(int(getattr(raw, "trade_exemode", 2))),
            filling_mode=int(getattr(raw, "filling_mode", 1)),
        )

    def symbol_info_tick(self, symbol: str) -> SymbolTick | None:
        mt5 = self._get_mt5()
        raw = mt5.symbol_info_tick(symbol)
        if raw is None:
            return None
        ts = datetime.fromtimestamp(int(raw.time), tz=timezone.utc)
        return SymbolTick(
            symbol=symbol,
            timestamp=ts,
            bid=Decimal(str(raw.bid)),
            ask=Decimal(str(raw.ask)),
            last=Decimal(str(getattr(raw, "last", 0.0))),
            volume=Decimal(str(getattr(raw, "volume", 0.0))),
        )

    def copy_rates_from_pos(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_pos: int,
        count: int,
    ) -> tuple[MT5RateBar, ...] | None:
        mt5 = self._get_mt5()
        tf_code = MT5_TIMEFRAME_CONSTANTS[timeframe]
        raw_rates = mt5.copy_rates_from_pos(symbol, tf_code, start_pos, count)
        if raw_rates is None:
            return None
        bars: list[MT5RateBar] = []
        for r in raw_rates:
            bars.append(
                MT5RateBar(
                    timestamp=datetime.fromtimestamp(int(r["time"]), tz=timezone.utc),
                    open=Decimal(str(r["open"])),
                    high=Decimal(str(r["high"])),
                    low=Decimal(str(r["low"])),
                    close=Decimal(str(r["close"])),
                    tick_volume=Decimal(str(r["tick_volume"])),
                    spread=int(r["spread"]),
                    real_volume=Decimal(str(r["real_volume"])),
                )
            )
        return tuple(bars)

    def history_deals_get(
        self,
        date_from: datetime,
        date_to: datetime,
        *,
        symbol: str | None = None,
    ) -> tuple[MT5DealRecord, ...] | None:
        mt5 = self._get_mt5()
        raw_deals = (
            mt5.history_deals_get(date_from, date_to, group=f"*{symbol}*")
            if symbol
            else mt5.history_deals_get(date_from, date_to)
        )
        if raw_deals is None:
            return None
        records: list[MT5DealRecord] = []
        for d in raw_deals:
            records.append(
                MT5DealRecord(
                    ticket=int(d.ticket),
                    order=int(getattr(d, "order", 0)),
                    time_utc=datetime.fromtimestamp(int(d.time), tz=timezone.utc),
                    deal_type=int(d.type),
                    entry=int(getattr(d, "entry", 0)),
                    magic=int(getattr(d, "magic", 0)),
                    position_id=int(getattr(d, "position_id", 0)),
                    symbol=str(getattr(d, "symbol", "")),
                    volume=Decimal(str(getattr(d, "volume", 0.0))),
                    price=Decimal(str(getattr(d, "price", 0.0))),
                    profit=Decimal(str(getattr(d, "profit", 0.0))),
                    commission=Decimal(str(getattr(d, "commission", 0.0))),
                    swap=Decimal(str(getattr(d, "swap", 0.0))),
                    fee=Decimal(str(getattr(d, "fee", 0.0))),
                    comment=str(getattr(d, "comment", "")),
                )
            )
        return tuple(records)

    def orders_get(self, symbol: str | None = None) -> tuple[MT5OrderRecord, ...] | None:
        mt5 = self._get_mt5()
        raw_orders = mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()
        if raw_orders is None:
            return None
        records: list[MT5OrderRecord] = []
        for o in raw_orders:
            records.append(
                MT5OrderRecord(
                    ticket=int(o.ticket),
                    time_setup_utc=datetime.fromtimestamp(int(o.time_setup), tz=timezone.utc),
                    order_type=int(o.type),
                    state=int(getattr(o, "state", 1)),
                    magic=int(getattr(o, "magic", 0)),
                    position_id=int(getattr(o, "position_id", 0)),
                    symbol=str(o.symbol),
                    volume_initial=Decimal(str(o.volume_initial)),
                    volume_current=Decimal(str(o.volume_current)),
                    price_open=Decimal(str(o.price_open)),
                    sl=Decimal(str(getattr(o, "sl", 0.0))),
                    tp=Decimal(str(getattr(o, "tp", 0.0))),
                    comment=str(getattr(o, "comment", "")),
                )
            )
        return tuple(records)

    def positions_get(self, symbol: str | None = None) -> tuple[MT5PositionRecord, ...] | None:
        mt5 = self._get_mt5()
        raw_positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if raw_positions is None:
            return None
        records: list[MT5PositionRecord] = []
        for p in raw_positions:
            records.append(
                MT5PositionRecord(
                    ticket=int(p.ticket),
                    time_utc=datetime.fromtimestamp(int(p.time), tz=timezone.utc),
                    position_type=int(p.type),
                    magic=int(getattr(p, "magic", 0)),
                    identifier=int(getattr(p, "identifier", p.ticket)),
                    symbol=str(p.symbol),
                    volume=Decimal(str(p.volume)),
                    price_open=Decimal(str(p.price_open)),
                    sl=Decimal(str(getattr(p, "sl", 0.0))),
                    tp=Decimal(str(getattr(p, "tp", 0.0))),
                    price_current=Decimal(str(p.price_current)),
                    profit=Decimal(str(getattr(p, "profit", 0.0))),
                    swap=Decimal(str(getattr(p, "swap", 0.0))),
                    comment=str(getattr(p, "comment", "")),
                )
            )
        return tuple(records)

    @staticmethod
    def _request_to_mt5_dict(request: MT5OrderRequest) -> dict[str, Any]:
        return {
            "action": int(request.action),
            "symbol": request.symbol,
            "volume": float(request.volume),
            "type": int(request.order_type),
            "price": float(request.price),
            "sl": float(request.sl),
            "tp": float(request.tp),
            "deviation": int(request.deviation),
            "magic": int(request.magic),
            "comment": request.comment,
            "type_time": int(request.type_time),
            "type_filling": int(request.type_filling),
            "position": int(request.position),
        }

    def order_check(self, request: MT5OrderRequest) -> MT5OrderCheckResult | None:
        mt5 = self._get_mt5()
        acct = self.account_info()
        if acct is None or not acct.is_demo:
            raise LiveTradingBlockedError(
                "SAFETY VIOLATION: order_check blocked because active MT5 account is not verified as DEMO (trade_mode=0)."
            )
        raw = mt5.order_check(self._request_to_mt5_dict(request))
        if raw is None:
            return None
        return MT5OrderCheckResult(
            retcode=int(raw.retcode),
            balance=Decimal(str(getattr(raw, "balance", 0.0))),
            equity=Decimal(str(getattr(raw, "equity", 0.0))),
            profit=Decimal(str(getattr(raw, "profit", 0.0))),
            margin=Decimal(str(getattr(raw, "margin", 0.0))),
            margin_free=Decimal(str(getattr(raw, "margin_free", 0.0))),
            margin_level=Decimal(str(getattr(raw, "margin_level", 0.0))),
            comment=str(getattr(raw, "comment", "")),
        )

    def order_send(self, request: MT5OrderRequest) -> MT5OrderSendResult | None:
        mt5 = self._get_mt5()
        # Hardware-level demo-account verification immediately prior to order transmission
        acct = self.account_info()
        if acct is None or not acct.is_demo:
            self.shutdown()
            raise LiveTradingBlockedError(
                "SAFETY VIOLATION: order_send blocked and terminal shut down because active MT5 account "
                f"is not a DEMO account (trade_mode={acct.trade_mode if acct else 'None'})."
            )
        raw = mt5.order_send(self._request_to_mt5_dict(request))
        if raw is None:
            return None
        return MT5OrderSendResult(
            retcode=int(raw.retcode),
            deal=int(getattr(raw, "deal", 0)),
            order=int(getattr(raw, "order", 0)),
            volume=Decimal(str(getattr(raw, "volume", 0.0))),
            price=Decimal(str(getattr(raw, "price", 0.0))),
            bid=Decimal(str(getattr(raw, "bid", 0.0))),
            ask=Decimal(str(getattr(raw, "ask", 0.0))),
            comment=str(getattr(raw, "comment", "")),
            request_id=int(getattr(raw, "request_id", 0)),
        )


class MockMT5TerminalBackend:
    """Deterministic in-memory MetaTrader 5 terminal simulator for 100% offline test execution."""

    def __init__(
        self,
        *,
        account: MT5AccountInfo | None = None,
        terminal: MT5TerminalInfo | None = None,
        symbols: Sequence[MT5SymbolInfo] | None = None,
    ) -> None:
        self._initialized: bool = False
        self._last_error: tuple[int, str] = (0, "Success")
        self._fail_next_initialize: bool = False
        self._fail_next_login: bool = False

        self._terminal = terminal or MT5TerminalInfo(
            connected=True,
            trade_allowed=True,
            tradeapi_disabled=False,
            name="Mock MetaTrader 5 Terminal",
            company="MetaQuotes Ltd.",
            build=4200,
        )
        self._account = account or MT5AccountInfo(
            login=100200300,
            trade_mode=int(MT5TradeMode.ACCOUNT_TRADE_MODE_DEMO),
            leverage=100,
            balance=Decimal("10000.00"),
            equity=Decimal("10000.00"),
            margin=Decimal("0.00"),
            margin_free=Decimal("10000.00"),
            currency="USD",
            currency_digits=2,
            server="AtlasBroker-Demo",
            name="ATLAS AI Demo Account",
        )

        default_xau = MT5SymbolInfo(
            name="XAUUSD",
            visible=True,
            select=True,
            digits=2,
            point=Decimal("0.01"),
            trade_contract_size=Decimal("100"),
            trade_tick_size=Decimal("0.01"),
            trade_tick_value=Decimal("1.00"),
            volume_min=Decimal("0.01"),
            volume_max=Decimal("50.00"),
            volume_step=Decimal("0.01"),
            trade_stops_level=20,
            trade_exemode=MT5TradeExecutionMode.SYMBOL_TRADE_EXECUTION_MARKET,
            filling_mode=int(MT5SymbolFillingFlag.SYMBOL_FILLING_FOK | MT5SymbolFillingFlag.SYMBOL_FILLING_IOC),
            margin_initial=Decimal("0.01"),
        )
        self._symbols: dict[str, MT5SymbolInfo] = {default_xau.name: default_xau}
        if symbols:
            for s in symbols:
                self._symbols[s.name] = s

        default_ts = datetime(2026, 11, 20, 14, 0, 0, tzinfo=timezone.utc)
        self._ticks: dict[str, SymbolTick] = {
            "XAUUSD": SymbolTick(
                symbol="XAUUSD",
                timestamp=default_ts,
                bid=Decimal("2650.00"),
                ask=Decimal("2650.20"),
                last=Decimal("2650.10"),
                volume=Decimal("150"),
            )
        }
        self._rates: dict[tuple[str, Timeframe], tuple[MT5RateBar, ...]] = {}
        self._deals: list[MT5DealRecord] = [
            MT5DealRecord(
                ticket=1001,
                order=0,
                time_utc=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                deal_type=int(MT5DealType.DEAL_TYPE_BALANCE),
                entry=int(MT5DealEntry.DEAL_ENTRY_IN),
                profit=Decimal("10000.00"),
                comment="Initial Demo Deposit",
            )
        ]
        self._orders: list[MT5OrderRecord] = []
        self._positions: list[MT5PositionRecord] = []
        self._next_ticket: int = 50001
        self._order_send_hook: Callable[[MT5OrderRequest], MT5OrderSendResult | None] | None = None

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def set_account(self, account: MT5AccountInfo | None) -> None:
        self._account = account

    def set_terminal(self, terminal: MT5TerminalInfo | None) -> None:
        self._terminal = terminal

    def set_symbol(self, symbol_info: MT5SymbolInfo) -> None:
        self._symbols[symbol_info.name] = symbol_info

    def set_tick(self, tick: SymbolTick) -> None:
        self._ticks[tick.symbol] = tick

    def set_rates(self, symbol: str, timeframe: Timeframe, bars: Sequence[MT5RateBar]) -> None:
        self._rates[(symbol.strip().upper(), timeframe)] = tuple(bars)

    def set_deals(self, deals: Sequence[MT5DealRecord]) -> None:
        self._deals = list(deals)

    def add_deal(self, deal: MT5DealRecord) -> None:
        self._deals.append(deal)

    def set_positions(self, positions: Sequence[MT5PositionRecord]) -> None:
        self._positions = list(positions)

    def set_orders(self, orders: Sequence[MT5OrderRecord]) -> None:
        self._orders = list(orders)

    def set_order_send_hook(
        self, hook: Callable[[MT5OrderRequest], MT5OrderSendResult | None] | None
    ) -> None:
        """Install an optional hook to simulate timeouts, partial fills, or network exceptions on order_send."""
        self._order_send_hook = hook

    def set_fail_next_initialize(self, fail: bool = True) -> None:
        self._fail_next_initialize = fail

    def set_fail_next_login(self, fail: bool = True) -> None:
        self._fail_next_login = fail

    def initialize(
        self,
        *,
        path: str | None = None,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        timeout: int = 10000,
        portable: bool = False,
    ) -> bool:
        if self._fail_next_initialize:
            self._fail_next_initialize = False
            self._last_error = (-10001, "Mock initialize failure")
            self._initialized = False
            return False
        self._initialized = True
        if login is not None:
            return self.login(login, password=password, server=server, timeout=timeout)
        self._last_error = (0, "Success")
        return True

    def login(
        self,
        login: int,
        *,
        password: str | None = None,
        server: str | None = None,
        timeout: int = 10000,
    ) -> bool:
        if self._fail_next_login or self._account is None:
            self._fail_next_login = False
            self._last_error = (-10002, "Mock login authentication failure")
            return False
        if self._account.login != login:
            self._last_error = (-10003, f"Account login mismatch: expected {self._account.login}, got {login}")
            return False
        if server is not None and self._account.server.lower() != server.strip().lower():
            self._last_error = (-10004, f"Server mismatch: expected {self._account.server}, got {server}")
            return False
        self._last_error = (0, "Success")
        return True

    def shutdown(self) -> None:
        self._initialized = False

    def last_error(self) -> tuple[int, str]:
        return self._last_error

    def terminal_info(self) -> MT5TerminalInfo | None:
        if not self._initialized:
            return None
        return self._terminal

    def account_info(self) -> MT5AccountInfo | None:
        if not self._initialized:
            return None
        return self._account

    def symbol_select(self, symbol: str, enable: bool = True) -> bool:
        if not self._initialized:
            return False
        key = symbol.strip().upper()
        if key not in self._symbols:
            self._last_error = (-10005, f"Unknown symbol '{key}'")
            return False
        sym = self._symbols[key]
        self._symbols[key] = sym.model_copy(update={"visible": enable, "select": enable})
        return True

    def symbol_info(self, symbol: str) -> MT5SymbolInfo | None:
        if not self._initialized:
            return None
        return self._symbols.get(symbol.strip().upper())

    def symbol_info_tick(self, symbol: str) -> SymbolTick | None:
        if not self._initialized:
            return None
        return self._ticks.get(symbol.strip().upper())

    def copy_rates_from_pos(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_pos: int,
        count: int,
    ) -> tuple[MT5RateBar, ...] | None:
        if not self._initialized:
            return None
        key = (symbol.strip().upper(), timeframe)
        if key not in self._rates:
            # Generate deterministic synthetic closed bars if none explicitly seeded
            base_ts = datetime(2026, 11, 20, 12, 0, 0, tzinfo=timezone.utc)
            generated: list[MT5RateBar] = []
            total_needed = start_pos + count
            for i in range(total_needed):
                bar_ts = base_ts - timedelta(hours=total_needed - i)
                p = Decimal("2640.00") + Decimal(str(i % 15)) * Decimal("0.50")
                generated.append(
                    MT5RateBar(
                        timestamp=bar_ts,
                        open=p,
                        high=p + Decimal("1.20"),
                        low=p - Decimal("0.80"),
                        close=p + Decimal("0.40"),
                        tick_volume=Decimal("100"),
                        spread=20,
                    )
                )
            self._rates[key] = tuple(generated)

        all_bars = self._rates[key]
        n = len(all_bars)
        if start_pos < 0 or count <= 0 or start_pos >= n:
            return ()
        # In MT5 copy_rates_from_pos, position 0 is the latest bar (at the end of the chronological array)
        end_idx = n - start_pos
        start_idx = max(0, end_idx - count)
        return tuple(all_bars[start_idx:end_idx])

    def history_deals_get(
        self,
        date_from: datetime,
        date_to: datetime,
        *,
        symbol: str | None = None,
    ) -> tuple[MT5DealRecord, ...] | None:
        if not self._initialized:
            return None
        d_from = date_from.astimezone(timezone.utc) if date_from.tzinfo else date_from.replace(tzinfo=timezone.utc)
        d_to = date_to.astimezone(timezone.utc) if date_to.tzinfo else date_to.replace(tzinfo=timezone.utc)
        sym_filter = symbol.strip().upper() if symbol else None

        matched = [
            d
            for d in self._deals
            if d_from <= d.time_utc <= d_to
            and (sym_filter is None or d.symbol.upper() == sym_filter)
        ]
        return tuple(matched)

    def orders_get(self, symbol: str | None = None) -> tuple[MT5OrderRecord, ...] | None:
        if not self._initialized:
            return None
        if symbol is None:
            return tuple(self._orders)
        sym = symbol.strip().upper()
        return tuple(o for o in self._orders if o.symbol == sym)

    def positions_get(self, symbol: str | None = None) -> tuple[MT5PositionRecord, ...] | None:
        if not self._initialized:
            return None
        if symbol is None:
            return tuple(self._positions)
        sym = symbol.strip().upper()
        return tuple(p for p in self._positions if p.symbol == sym)

    def order_check(self, request: MT5OrderRequest) -> MT5OrderCheckResult | None:
        if not self._initialized or self._account is None:
            return None
        if not self._account.is_demo:
            self.shutdown()
            raise LiveTradingBlockedError(
                f"SAFETY VIOLATION: Mock order_check blocked on non-demo account (trade_mode={self._account.trade_mode})."
            )
        sym = self._symbols.get(request.symbol)
        if sym is None:
            return MT5OrderCheckResult(
                retcode=int(MT5Retcode.TRADE_RETCODE_INVALID),
                comment="Unknown symbol",
            )
        req_margin = (request.volume * sym.trade_contract_size * request.price * sym.margin_initial).quantize(
            Decimal("0.01")
        )
        if req_margin > self._account.margin_free:
            return MT5OrderCheckResult(
                retcode=int(MT5Retcode.TRADE_RETCODE_NO_MONEY),
                balance=self._account.balance,
                equity=self._account.equity,
                margin=req_margin,
                margin_free=self._account.margin_free,
                comment="Insufficient free margin",
            )
        return MT5OrderCheckResult(
            retcode=0,
            balance=self._account.balance,
            equity=self._account.equity,
            margin=req_margin,
            margin_free=self._account.margin_free - req_margin,
            comment="Done",
        )

    def order_send(self, request: MT5OrderRequest) -> MT5OrderSendResult | None:
        if not self._initialized or self._account is None:
            return None
        if not self._account.is_demo:
            self.shutdown()
            raise LiveTradingBlockedError(
                f"SAFETY VIOLATION: Mock order_send blocked on non-demo account (trade_mode={self._account.trade_mode})."
            )
        if self._order_send_hook is not None:
            return self._order_send_hook(request)

        sym = self._symbols.get(request.symbol)
        if sym is None:
            return MT5OrderSendResult(
                retcode=int(MT5Retcode.TRADE_RETCODE_INVALID),
                comment="Unknown symbol",
            )

        # Validate filling mode compatibility against trade_exemode
        if (
            sym.trade_exemode == MT5TradeExecutionMode.SYMBOL_TRADE_EXECUTION_MARKET
            and request.type_filling == MT5OrderFillingMode.ORDER_FILLING_RETURN
        ):
            return MT5OrderSendResult(
                retcode=int(MT5Retcode.TRADE_RETCODE_INVALID_FILL),
                comment="ORDER_FILLING_RETURN is not supported under Market Execution",
            )

        order_ticket = self._next_ticket
        deal_ticket = self._next_ticket + 1
        pos_ticket = self._next_ticket + 2
        self._next_ticket += 3

        tick = self._ticks.get(request.symbol)
        now_ts = tick.timestamp if tick is not None else datetime(2026, 11, 20, 14, 0, 0, tzinfo=timezone.utc)
        fill_price = (
            (tick.ask if request.order_type == MT5OrderType.ORDER_TYPE_BUY else tick.bid)
            if tick is not None
            else request.price
        )

        # Record open position and entry deal in the mock terminal
        self._positions.append(
            MT5PositionRecord(
                ticket=pos_ticket,
                time_utc=now_ts,
                position_type=int(request.order_type),
                magic=request.magic,
                identifier=pos_ticket,
                symbol=request.symbol,
                volume=request.volume,
                price_open=fill_price,
                sl=request.sl,
                tp=request.tp,
                price_current=fill_price,
                profit=Decimal("0.00"),
                comment=request.comment,
            )
        )
        self._deals.append(
            MT5DealRecord(
                ticket=deal_ticket,
                order=order_ticket,
                time_utc=now_ts,
                deal_type=(
                    int(MT5DealType.DEAL_TYPE_BUY)
                    if request.order_type == MT5OrderType.ORDER_TYPE_BUY
                    else int(MT5DealType.DEAL_TYPE_SELL)
                ),
                entry=int(MT5DealEntry.DEAL_ENTRY_IN),
                magic=request.magic,
                position_id=pos_ticket,
                symbol=request.symbol,
                volume=request.volume,
                price=fill_price,
                profit=Decimal("0.00"),
                comment=request.comment,
            )
        )

        return MT5OrderSendResult(
            retcode=int(MT5Retcode.TRADE_RETCODE_DONE),
            deal=deal_ticket,
            order=order_ticket,
            volume=request.volume,
            price=fill_price,
            bid=tick.bid if tick else fill_price,
            ask=tick.ask if tick else fill_price,
            comment="Request executed",
            request_id=1,
        )
