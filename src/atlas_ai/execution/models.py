"""Immutable Pydantic domain models and official MQL5/MT5 enums for ATLAS AI Phase 7."""

from datetime import datetime, timezone
from decimal import Decimal
from enum import IntEnum, StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atlas_ai.domain.market import Timeframe
from atlas_ai.domain.signals import SignalDirection


class MT5TradeMode(IntEnum):
    """Official MQL5 ENUM_ACCOUNT_TRADE_MODE constants."""

    ACCOUNT_TRADE_MODE_DEMO = 0
    ACCOUNT_TRADE_MODE_CONTEST = 1
    ACCOUNT_TRADE_MODE_REAL = 2


class MT5SymbolTradeMode(IntEnum):
    """Official MQL5 ENUM_SYMBOL_TRADE_MODE constants."""

    SYMBOL_TRADE_MODE_DISABLED = 0
    SYMBOL_TRADE_MODE_LONGONLY = 1
    SYMBOL_TRADE_MODE_SHORTONLY = 2
    SYMBOL_TRADE_MODE_CLOSEONLY = 3
    SYMBOL_TRADE_MODE_FULL = 4


class MT5TradeExecutionMode(IntEnum):
    """Official MQL5 ENUM_SYMBOL_TRADE_EXECUTION constants."""

    SYMBOL_TRADE_EXECUTION_REQUEST = 0
    SYMBOL_TRADE_EXECUTION_INSTANT = 1
    SYMBOL_TRADE_EXECUTION_MARKET = 2
    SYMBOL_TRADE_EXECUTION_EXCHANGE = 3


class MT5SymbolFillingFlag(IntEnum):
    """Bitmask flags on symbol_info().filling_mode."""

    SYMBOL_FILLING_FOK = 1
    SYMBOL_FILLING_IOC = 2


class MT5OrderFillingMode(IntEnum):
    """Official MQL5 ENUM_ORDER_TYPE_FILLING constants for order requests."""

    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2


class MT5OrderType(IntEnum):
    """Official MQL5 ENUM_ORDER_TYPE constants for market execution."""

    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1


class MT5TradeAction(IntEnum):
    """Official MQL5 ENUM_TRADE_REQUEST_ACTIONS constants."""

    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_PENDING = 5
    TRADE_ACTION_SLTP = 6
    TRADE_ACTION_MODIFY = 7
    TRADE_ACTION_REMOVE = 8
    TRADE_ACTION_CLOSE_BY = 10


class MT5OrderTimeMode(IntEnum):
    """Official MQL5 ENUM_ORDER_TYPE_TIME constants."""

    ORDER_TIME_GTC = 0
    ORDER_TIME_DAY = 1
    ORDER_TIME_SPECIFIED = 2
    ORDER_TIME_SPECIFIED_DAY = 3


class MT5DealType(IntEnum):
    """Official MQL5 ENUM_DEAL_TYPE constants and numeric values (0..17)."""

    DEAL_TYPE_BUY = 0
    DEAL_TYPE_SELL = 1
    DEAL_TYPE_BALANCE = 2
    DEAL_TYPE_CREDIT = 3
    DEAL_TYPE_CHARGE = 4
    DEAL_TYPE_CORRECTION = 5
    DEAL_TYPE_BONUS = 6
    DEAL_TYPE_COMMISSION = 7
    DEAL_TYPE_COMMISSION_DAILY = 8
    DEAL_TYPE_COMMISSION_MONTHLY = 9
    DEAL_TYPE_COMMISSION_AGENT_DAILY = 10
    DEAL_TYPE_COMMISSION_AGENT_MONTHLY = 11
    DEAL_TYPE_INTEREST = 12
    DEAL_TYPE_BUY_CANCELED = 13
    DEAL_TYPE_SELL_CANCELED = 14
    DEAL_DIVIDEND = 15
    DEAL_DIVIDEND_FRANKED = 16
    DEAL_TAX = 17


class MT5DealEntry(IntEnum):
    """Official MQL5 ENUM_DEAL_ENTRY constants."""

    DEAL_ENTRY_IN = 0
    DEAL_ENTRY_OUT = 1
    DEAL_ENTRY_INOUT = 2
    DEAL_ENTRY_OUT_BY = 3


class MT5Retcode(IntEnum):
    """Official MQL5 trade server return codes."""

    TRADE_RETCODE_REQUOTE = 10004
    TRADE_RETCODE_REJECT = 10006
    TRADE_RETCODE_CANCEL = 10007
    TRADE_RETCODE_PLACED = 10008
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_DONE_PARTIAL = 10010
    TRADE_RETCODE_ERROR = 10011
    TRADE_RETCODE_TIMEOUT = 10012
    TRADE_RETCODE_INVALID = 10013
    TRADE_RETCODE_INVALID_VOLUME = 10014
    TRADE_RETCODE_INVALID_PRICE = 10015
    TRADE_RETCODE_INVALID_STOPS = 10016
    TRADE_RETCODE_TRADE_DISABLED = 10017
    TRADE_RETCODE_MARKET_CLOSED = 10018
    TRADE_RETCODE_NO_MONEY = 10019
    TRADE_RETCODE_PRICE_CHANGED = 10020
    TRADE_RETCODE_PRICE_OFF = 10021
    TRADE_RETCODE_INVALID_EXPIRATION = 10022
    TRADE_RETCODE_ORDER_CHANGED = 10023
    TRADE_RETCODE_TOO_MANY_REQUESTS = 10024
    TRADE_RETCODE_NO_CHANGES = 10025
    TRADE_RETCODE_SERVER_DISABLES_AT = 10026
    TRADE_RETCODE_CLIENT_DISABLES_AT = 10027
    TRADE_RETCODE_LOCKED = 10028
    TRADE_RETCODE_FROZEN = 10029
    TRADE_RETCODE_INVALID_FILL = 10030
    TRADE_RETCODE_CONNECTION = 10031
    TRADE_RETCODE_ONLY_REAL = 10032
    TRADE_RETCODE_LIMIT_ORDERS = 10033
    TRADE_RETCODE_LIMIT_VOLUME = 10034
    TRADE_RETCODE_INVALID_ORDER = 10035
    TRADE_RETCODE_POSITION_CLOSED = 10036
    TRADE_RETCODE_INVALID_CLOSE_VOLUME = 10038
    TRADE_RETCODE_CLOSE_ORDER_EXIST = 10039
    TRADE_RETCODE_LIMIT_POSITIONS = 10040
    TRADE_RETCODE_REJECT_CANCEL = 10041
    TRADE_RETCODE_LONG_ONLY = 10042
    TRADE_RETCODE_SHORT_ONLY = 10043
    TRADE_RETCODE_CLOSE_ONLY = 10044
    TRADE_RETCODE_FIFO_CLOSE = 10045


AMBIGUOUS_SUBMISSION_RETCODES: frozenset[int] = frozenset(
    {
        int(MT5Retcode.TRADE_RETCODE_TIMEOUT),
        int(MT5Retcode.TRADE_RETCODE_CONNECTION),
    }
)

DETERMINISTIC_REJECTION_RETCODES: frozenset[int] = frozenset(
    {
        int(MT5Retcode.TRADE_RETCODE_REQUOTE),
        int(MT5Retcode.TRADE_RETCODE_REJECT),
        int(MT5Retcode.TRADE_RETCODE_CANCEL),
        int(MT5Retcode.TRADE_RETCODE_INVALID),
        int(MT5Retcode.TRADE_RETCODE_INVALID_VOLUME),
        int(MT5Retcode.TRADE_RETCODE_INVALID_PRICE),
        int(MT5Retcode.TRADE_RETCODE_INVALID_STOPS),
        int(MT5Retcode.TRADE_RETCODE_TRADE_DISABLED),
        int(MT5Retcode.TRADE_RETCODE_MARKET_CLOSED),
        int(MT5Retcode.TRADE_RETCODE_NO_MONEY),
        int(MT5Retcode.TRADE_RETCODE_PRICE_CHANGED),
        int(MT5Retcode.TRADE_RETCODE_PRICE_OFF),
        int(MT5Retcode.TRADE_RETCODE_INVALID_EXPIRATION),
        int(MT5Retcode.TRADE_RETCODE_TOO_MANY_REQUESTS),
        int(MT5Retcode.TRADE_RETCODE_NO_CHANGES),
        int(MT5Retcode.TRADE_RETCODE_SERVER_DISABLES_AT),
        int(MT5Retcode.TRADE_RETCODE_CLIENT_DISABLES_AT),
        int(MT5Retcode.TRADE_RETCODE_LOCKED),
        int(MT5Retcode.TRADE_RETCODE_FROZEN),
        int(MT5Retcode.TRADE_RETCODE_INVALID_FILL),
        int(MT5Retcode.TRADE_RETCODE_LIMIT_ORDERS),
        int(MT5Retcode.TRADE_RETCODE_LIMIT_VOLUME),
        int(MT5Retcode.TRADE_RETCODE_INVALID_ORDER),
        int(MT5Retcode.TRADE_RETCODE_POSITION_CLOSED),
        int(MT5Retcode.TRADE_RETCODE_INVALID_CLOSE_VOLUME),
        int(MT5Retcode.TRADE_RETCODE_CLOSE_ORDER_EXIST),
        int(MT5Retcode.TRADE_RETCODE_LIMIT_POSITIONS),
        int(MT5Retcode.TRADE_RETCODE_REJECT_CANCEL),
        int(MT5Retcode.TRADE_RETCODE_LONG_ONLY),
        int(MT5Retcode.TRADE_RETCODE_SHORT_ONLY),
        int(MT5Retcode.TRADE_RETCODE_CLOSE_ONLY),
        int(MT5Retcode.TRADE_RETCODE_FIFO_CLOSE),
    }
)


class OrderLedgerStatus(StrEnum):
    """Persistent execution ledger status for an order submission lifecycle."""

    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    ACTIVE = "ACTIVE"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"
    UNKNOWN_PENDING_RECONCILIATION = "UNKNOWN_PENDING_RECONCILIATION"
    FAILED = "FAILED"


class ReconciliationStatus(StrEnum):
    """Broker reconciliation status for an execution ledger entry."""

    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING = "PENDING"
    RECONCILED_EXECUTED = "RECONCILED_EXECUTED"
    RECONCILED_NOT_EXECUTED = "RECONCILED_NOT_EXECUTED"


class LedgerHealthStatus(StrEnum):
    """Operational health status of PersistentExecutionLedger."""

    HEALTHY = "HEALTHY"
    RECOVERY_REQUIRED_LOCKED = "RECOVERY_REQUIRED_LOCKED"


class MT5TerminalInfo(BaseModel):
    """Immutable snapshot of MT5 terminal connection and permission status."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    connected: bool = Field(..., description="True if terminal is connected to the broker server.")
    trade_allowed: bool = Field(..., description="True if algorithmic trading is enabled in the terminal.")
    tradeapi_disabled: bool = Field(default=False, description="True if external Python API trading is disabled.")
    community_account: bool = Field(default=False)
    name: str = Field(default="MetaTrader 5")
    company: str = Field(default="MetaQuotes")
    path: str = Field(default="")
    build: int = Field(default=0, ge=0)


class MT5AccountInfo(BaseModel):
    """Immutable snapshot of MT5 account information."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    login: int = Field(..., gt=0)
    trade_mode: int = Field(..., description="Raw trade_mode integer (0=DEMO, 1=CONTEST, 2=REAL).")
    leverage: int = Field(default=100, gt=0)
    limit_orders: int = Field(default=200, ge=0)
    trade_allowed: bool = Field(default=True)
    trade_expert: bool = Field(default=True)
    balance: Decimal = Field(..., ge=Decimal("0"))
    credit: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    profit: Decimal = Field(default=Decimal("0"))
    equity: Decimal = Field(..., ge=Decimal("0"))
    margin: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    margin_free: Decimal = Field(..., ge=Decimal("0"))
    currency: str = Field(default="USD", min_length=3, max_length=3)
    currency_digits: int = Field(default=2, ge=0, le=8)
    server: str = Field(..., min_length=1)
    name: str = Field(default="Demo Trader")
    company: str = Field(default="Broker")

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if len(cleaned) != 3 or not cleaned.isalpha():
            raise ValueError(f"Currency must be a 3-letter alphabetic code, got '{value}'.")
        return cleaned

    @property
    def is_demo(self) -> bool:
        """Return True ONLY when trade_mode is strictly ACCOUNT_TRADE_MODE_DEMO (0)."""
        return self.trade_mode == int(MT5TradeMode.ACCOUNT_TRADE_MODE_DEMO)

    def reconciliation_tolerance(self, tolerance_units: int = 5) -> Decimal:
        """Compute account-currency-aware balance reconciliation tolerance (e.g., 5 * 10^-currency_digits)."""
        unit = Decimal(10) ** (-self.currency_digits)
        return Decimal(tolerance_units) * unit


class MT5SymbolInfo(BaseModel):
    """Immutable snapshot of MT5 symbol contract and execution specifications."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=3, max_length=16)
    visible: bool = Field(default=True)
    select: bool = Field(default=True)
    digits: int = Field(default=2, ge=0, le=10)
    point: Decimal = Field(..., gt=Decimal("0"))
    trade_contract_size: Decimal = Field(..., gt=Decimal("0"))
    trade_tick_size: Decimal = Field(..., gt=Decimal("0"))
    trade_tick_value: Decimal = Field(..., gt=Decimal("0"))
    volume_min: Decimal = Field(..., gt=Decimal("0"))
    volume_max: Decimal = Field(..., gt=Decimal("0"))
    volume_step: Decimal = Field(..., gt=Decimal("0"))
    trade_stops_level: int = Field(default=0, ge=0, description="Minimum stop distance in points.")
    trade_freeze_level: int = Field(default=0, ge=0)
    trade_mode: MT5SymbolTradeMode = Field(
        default=MT5SymbolTradeMode.SYMBOL_TRADE_MODE_FULL,
        description="Official MQL5 ENUM_SYMBOL_TRADE_MODE permission (0=DISABLED, 1=LONGONLY, 2=SHORTONLY, 3=CLOSEONLY, 4=FULL).",
    )
    trade_exemode: MT5TradeExecutionMode = Field(
        default=MT5TradeExecutionMode.SYMBOL_TRADE_EXECUTION_MARKET,
        description="Official MQL5 ENUM_SYMBOL_TRADE_EXECUTION mode.",
    )
    filling_mode: int = Field(
        default=int(MT5SymbolFillingFlag.SYMBOL_FILLING_FOK),
        ge=0,
        description="Bitmask of supported filling flags (1=FOK, 2=IOC, 0=default/exchange).",
    )
    margin_initial: Decimal = Field(default=Decimal("0.01"), gt=Decimal("0"), le=Decimal("1"))

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned.isalnum():
            raise ValueError(f"Symbol name must be alphanumeric, got '{value}'.")
        return cleaned


class SymbolTick(BaseModel):
    """Immutable real-time Bid/Ask tick quote normalized to UTC."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(..., min_length=3, max_length=16)
    timestamp: datetime = Field(..., description="Timezone-aware UTC tick timestamp.")
    bid: Decimal = Field(..., gt=Decimal("0"))
    ask: Decimal = Field(..., gt=Decimal("0"))
    last: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    volume: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned.isalnum():
            raise ValueError(f"Tick symbol must be alphanumeric, got '{value}'.")
        return cleaned

    @field_validator("timestamp")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("SymbolTick timestamp must be timezone-aware.")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_bid_ask(self) -> "SymbolTick":
        if self.ask < self.bid:
            raise ValueError(f"Invalid SymbolTick: ask ({self.ask}) cannot be less than bid ({self.bid}).")
        return self

    @property
    def spread_price(self) -> Decimal:
        """Return exact Bid-Ask spread in price units (`ask - bid`)."""
        return self.ask - self.bid

    def spread_points(self, tick_size: Decimal) -> Decimal:
        """Return Bid-Ask spread in instrument `tick_size` points."""
        if tick_size <= Decimal("0"):
            raise ValueError("tick_size must be > 0.")
        return self.spread_price / tick_size

    def age_seconds(self, reference_now_utc: datetime) -> float:
        """Return elapsed seconds between tick timestamp and `reference_now_utc`."""
        ref = (
            reference_now_utc.astimezone(timezone.utc)
            if reference_now_utc.tzinfo is not None
            else reference_now_utc.replace(tzinfo=timezone.utc)
        )
        return (ref - self.timestamp).total_seconds()


class MT5RateBar(BaseModel):
    """Immutable OHLCV rate bar returned by MT5 terminal copy_rates calls, normalized to UTC."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime = Field(..., description="Candle open timestamp in UTC.")
    open: Decimal = Field(..., gt=Decimal("0"))
    high: Decimal = Field(..., gt=Decimal("0"))
    low: Decimal = Field(..., gt=Decimal("0"))
    close: Decimal = Field(..., gt=Decimal("0"))
    tick_volume: Decimal = Field(..., ge=Decimal("0"))
    spread: int = Field(default=0, ge=0)
    real_volume: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))

    @field_validator("timestamp")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("MT5RateBar timestamp must be timezone-aware.")
        return value.astimezone(timezone.utc)


class MT5DealRecord(BaseModel):
    """Immutable broker deal record from MT5 history_deals_get()."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticket: int = Field(..., gt=0, description="Unique broker deal ticket.")
    order: int = Field(default=0, ge=0, description="Associated order ticket (0 for account-level deals).")
    time_utc: datetime = Field(..., description="Deal execution timestamp in UTC.")
    deal_type: int = Field(..., description="Raw integer deal type (mapped against official MT5DealType 0..17).")
    entry: int = Field(default=int(MT5DealEntry.DEAL_ENTRY_IN), ge=0, le=3)
    magic: int = Field(default=0, ge=0)
    position_id: int = Field(default=0, ge=0)
    symbol: str = Field(default="")
    volume: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    price: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    profit: Decimal = Field(default=Decimal("0"))
    commission: Decimal = Field(default=Decimal("0"))
    swap: Decimal = Field(default=Decimal("0"))
    fee: Decimal = Field(default=Decimal("0"))
    comment: str = Field(default="")

    @field_validator("time_utc")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("MT5DealRecord time_utc must be timezone-aware.")
        return value.astimezone(timezone.utc)

    @property
    def net_cash_impact(self) -> Decimal:
        """Total monetary impact of this deal (`profit + commission + fee + swap`)."""
        return self.profit + self.commission + self.fee + self.swap


class MT5OrderRecord(BaseModel):
    """Immutable active order record from MT5 orders_get()."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticket: int = Field(..., gt=0)
    time_setup_utc: datetime
    order_type: int
    state: int = Field(default=1)
    magic: int = Field(default=0, ge=0)
    position_id: int = Field(default=0, ge=0)
    symbol: str = Field(..., min_length=3, max_length=16)
    volume_initial: Decimal = Field(..., gt=Decimal("0"))
    volume_current: Decimal = Field(..., ge=Decimal("0"))
    price_open: Decimal = Field(..., gt=Decimal("0"))
    sl: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    tp: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    comment: str = Field(default="")

    @field_validator("time_setup_utc")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("MT5OrderRecord time_setup_utc must be timezone-aware.")
        return value.astimezone(timezone.utc)


class MT5PositionRecord(BaseModel):
    """Immutable open position record from MT5 positions_get()."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticket: int = Field(..., gt=0)
    time_utc: datetime
    position_type: int = Field(..., ge=0, le=1, description="0=POSITION_TYPE_BUY, 1=POSITION_TYPE_SELL.")
    magic: int = Field(default=0, ge=0)
    identifier: int = Field(..., gt=0, description="Persistent position_id in MT5.")
    symbol: str = Field(..., min_length=3, max_length=16)
    volume: Decimal = Field(..., gt=Decimal("0"))
    price_open: Decimal = Field(..., gt=Decimal("0"))
    sl: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    tp: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    price_current: Decimal = Field(..., gt=Decimal("0"))
    profit: Decimal = Field(default=Decimal("0"))
    swap: Decimal = Field(default=Decimal("0"))
    comment: str = Field(default="")

    @field_validator("time_utc")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("MT5PositionRecord time_utc must be timezone-aware.")
        return value.astimezone(timezone.utc)


class MT5OrderRequest(BaseModel):
    """Validated MT5 trade request payload for order_check() and order_send()."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: MT5TradeAction = Field(default=MT5TradeAction.TRADE_ACTION_DEAL)
    symbol: str = Field(..., min_length=3, max_length=16)
    volume: Decimal = Field(..., gt=Decimal("0"))
    order_type: MT5OrderType
    price: Decimal = Field(..., gt=Decimal("0"))
    sl: Decimal = Field(..., gt=Decimal("0"))
    tp: Decimal = Field(..., gt=Decimal("0"))
    deviation: int = Field(default=20, ge=0, le=500)
    magic: int = Field(..., gt=0)
    comment: str = Field(..., min_length=1, max_length=64)
    type_time: MT5OrderTimeMode = Field(default=MT5OrderTimeMode.ORDER_TIME_GTC)
    type_filling: MT5OrderFillingMode = Field(default=MT5OrderFillingMode.ORDER_FILLING_FOK)
    position: int = Field(default=0, ge=0)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned.isalnum():
            raise ValueError(f"Order symbol must be alphanumeric, got '{value}'.")
        return cleaned

    @model_validator(mode="after")
    def validate_directional_stops(self) -> "MT5OrderRequest":
        """Enforce stop_loss < price < take_profit for BUY and take_profit < price < stop_loss for SELL."""
        if self.order_type == MT5OrderType.ORDER_TYPE_BUY:
            if not (self.sl < self.price < self.tp):
                raise ValueError(
                    f"BUY MT5OrderRequest requires sl ({self.sl}) < price ({self.price}) < tp ({self.tp})."
                )
        elif self.order_type == MT5OrderType.ORDER_TYPE_SELL:
            if not (self.tp < self.price < self.sl):
                raise ValueError(
                    f"SELL MT5OrderRequest requires tp ({self.tp}) < price ({self.price}) < sl ({self.sl})."
                )
        return self


class MT5OrderCheckResult(BaseModel):
    """Immutable response from MT5 order_check()."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    retcode: int
    balance: Decimal = Field(default=Decimal("0"))
    equity: Decimal = Field(default=Decimal("0"))
    profit: Decimal = Field(default=Decimal("0"))
    margin: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    margin_free: Decimal = Field(default=Decimal("0"))
    margin_level: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    comment: str = Field(default="")


class MT5OrderSendResult(BaseModel):
    """Immutable response from MT5 order_send()."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    retcode: int
    deal: int = Field(default=0, ge=0)
    order: int = Field(default=0, ge=0)
    volume: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    price: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    bid: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    ask: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    comment: str = Field(default="")
    request_id: int = Field(default=0, ge=0)


class AccountRiskPersistenceSnapshot(BaseModel):
    """Persisted account risk and trusted rollover balance anchor snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    account_login: int = Field(..., gt=0)
    server: str = Field(..., min_length=1)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    currency_digits: int = Field(default=2, ge=0, le=8)
    trading_day_label: str = Field(..., min_length=10, max_length=10, description="YYYY-MM-DD trading day.")
    anchor_timestamp_utc: datetime
    anchor_balance: Decimal = Field(..., gt=Decimal("0"))
    daily_starting_balance: Decimal = Field(..., gt=Decimal("0"))
    peak_balance: Decimal = Field(..., gt=Decimal("0"))
    last_verified_balance: Decimal = Field(..., gt=Decimal("0"))
    last_updated_utc: datetime
    initial_deposit_ticket: int | None = Field(
        default=None,
        gt=0,
        description="Optional broker deal ticket of a same-day account-opening deposit already folded into daily_starting_balance.",
    )

    @field_validator("anchor_timestamp_utc", "last_updated_utc")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("Snapshot timestamp must be timezone-aware.")
        return value.astimezone(timezone.utc)


class LedgerTransitionRecord(BaseModel):
    """Immutable audit record of a single state transition on an ExecutionLedgerEntry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    from_status: OrderLedgerStatus | None = Field(default=None)
    to_status: OrderLedgerStatus
    timestamp_utc: datetime
    reason: str = Field(..., min_length=1)
    retcode: int | None = Field(default=None)

    @field_validator("timestamp_utc")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("LedgerTransitionRecord timestamp_utc must be timezone-aware.")
        return value.astimezone(timezone.utc)


class ExecutionLedgerEntry(BaseModel):
    """Immutable persistent execution ledger entry guaranteeing cross-restart idempotency."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    client_order_id: str = Field(..., min_length=1)
    signal_id: str = Field(..., min_length=1)
    symbol: str = Field(..., min_length=3, max_length=16)
    timeframe: Timeframe
    direction: SignalDirection
    magic: int = Field(..., gt=0)
    requested_volume: Decimal = Field(..., gt=Decimal("0"))
    executed_volume: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    remaining_volume: Decimal = Field(..., ge=Decimal("0"))
    requested_price: Decimal = Field(..., gt=Decimal("0"))
    fill_price: Decimal | None = Field(default=None)
    stop_loss: Decimal = Field(..., gt=Decimal("0"))
    take_profit: Decimal = Field(..., gt=Decimal("0"))
    filling_mode: MT5OrderFillingMode
    order_ticket: int | None = Field(default=None)
    deal_ticket: int | None = Field(default=None)
    position_ticket: int | None = Field(default=None)
    retcode: int | None = Field(default=None)
    status: OrderLedgerStatus
    reconciliation_status: ReconciliationStatus = Field(default=ReconciliationStatus.NOT_REQUIRED)
    created_at_utc: datetime
    updated_at_utc: datetime
    status_reason: str = Field(..., min_length=1)
    attempt_count: int = Field(default=1, ge=1)
    history: tuple[LedgerTransitionRecord, ...] = Field(default=())

    @field_validator("created_at_utc", "updated_at_utc")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("Ledger timestamp must be timezone-aware.")
        return value.astimezone(timezone.utc)
