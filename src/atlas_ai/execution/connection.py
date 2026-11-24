"""MT5 Connection Adapter with mandatory DEMO verification, health checks, and filling policy resolution."""

from atlas_ai.core.logger import get_logger
from atlas_ai.domain.market import InstrumentSpec
from atlas_ai.execution.backend import MT5TerminalBackend
from atlas_ai.execution.config import MT5ConnectionConfig
from atlas_ai.execution.exceptions import (
    LiveTradingBlockedError,
    MT5AuthenticationError,
    MT5ConnectionError,
    OrderValidationError,
)
from atlas_ai.execution.models import (
    MT5AccountInfo,
    MT5OrderFillingMode,
    MT5SymbolFillingFlag,
    MT5SymbolInfo,
    MT5TerminalInfo,
    MT5TradeExecutionMode,
    MT5TradeMode,
)

logger = get_logger("execution.connection")


def resolve_supported_filling_mode(symbol_info: MT5SymbolInfo) -> MT5OrderFillingMode:
    """Deterministically resolve a supported MT5 order filling policy from `trade_exemode` and `filling_mode`.

    Rules (per official MQL5 reference):
    1. Under Market Execution (`SYMBOL_TRADE_EXECUTION_MARKET = 2`):
       - `ORDER_FILLING_RETURN` is NEVER permitted.
       - If `filling_mode & SYMBOL_FILLING_FOK (1)` -> `ORDER_FILLING_FOK`.
       - Else if `filling_mode & SYMBOL_FILLING_IOC (2)` -> `ORDER_FILLING_IOC`.
       - Else if `filling_mode == 0` (broker bridge reports 0 for default FOK/IOC) -> `ORDER_FILLING_FOK`.
       - Else -> raise `OrderValidationError`.
    2. Under Request (`0`), Instant (`1`), or Exchange (`3`) Execution:
       - `ORDER_FILLING_RETURN` IS permitted.
       - If `filling_mode & SYMBOL_FILLING_FOK (1)` -> `ORDER_FILLING_FOK`.
       - Else if `filling_mode & SYMBOL_FILLING_IOC (2)` -> `ORDER_FILLING_IOC`.
       - Else (including when `filling_mode == 0`) -> `ORDER_FILLING_RETURN`.
    """
    fmode = symbol_info.filling_mode
    exemode = symbol_info.trade_exemode

    supports_fok = bool(fmode & int(MT5SymbolFillingFlag.SYMBOL_FILLING_FOK))
    supports_ioc = bool(fmode & int(MT5SymbolFillingFlag.SYMBOL_FILLING_IOC))

    if exemode == MT5TradeExecutionMode.SYMBOL_TRADE_EXECUTION_MARKET:
        if supports_fok:
            return MT5OrderFillingMode.ORDER_FILLING_FOK
        if supports_ioc:
            return MT5OrderFillingMode.ORDER_FILLING_IOC
        if fmode == 0:
            return MT5OrderFillingMode.ORDER_FILLING_FOK
        raise OrderValidationError(
            f"Symbol '{symbol_info.name}' under Market Execution has unsupported filling_mode bitmask ({fmode}); "
            "neither FOK (1) nor IOC (2) is set and ORDER_FILLING_RETURN is forbidden."
        )

    # Request, Instant, or Exchange Execution
    if supports_fok:
        return MT5OrderFillingMode.ORDER_FILLING_FOK
    if supports_ioc:
        return MT5OrderFillingMode.ORDER_FILLING_IOC
    return MT5OrderFillingMode.ORDER_FILLING_RETURN


class MT5ConnectionAdapter:
    """Manages MT5 terminal connection lifecycle, health checks, bounded reconnection,
    mandatory DEMO-only account verification, and InstrumentSpec contract mapping.
    """

    def __init__(
        self,
        backend: MT5TerminalBackend,
        config: MT5ConnectionConfig | None = None,
    ) -> None:
        self._backend = backend
        self._config = config or MT5ConnectionConfig()
        self._connected: bool = False

    @property
    def backend(self) -> MT5TerminalBackend:
        return self._backend

    @property
    def config(self) -> MT5ConnectionConfig:
        return self._config

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> MT5AccountInfo:
        """Initialize terminal connection, authenticate, and verify DEMO account status.

        Raises:
            MT5ConnectionError: If terminal initialization or health check fails.
            LiveTradingBlockedError: If connected account is REAL (2) or CONTEST (1).
            MT5AuthenticationError: If login or server does not match expected configuration.
        """
        cfg = self._config
        logger.info("Connecting to MT5 terminal with config: %s", cfg.redacted_dict())

        pw_raw = cfg.password.get_secret_value() if cfg.password is not None else None
        path_str = str(cfg.terminal_path) if cfg.terminal_path is not None else None

        ok = self._backend.initialize(
            path=path_str,
            login=cfg.login,
            password=pw_raw,
            server=cfg.server,
            timeout=cfg.timeout_ms,
            portable=cfg.portable,
        )
        if not ok:
            err_code, err_msg = self._backend.last_error()
            self._connected = False
            self._backend.shutdown()
            raise MT5ConnectionError(
                f"MT5 terminal initialize() failed (code={err_code}, message='{err_msg}')."
            )

        # Immediately verify DEMO-only trade_mode, expected login/server, and terminal health
        _, account = self.check_health()
        self._connected = True
        logger.info(
            "MT5 DEMO connection verified: login=%d, server='%s', trade_mode=%d (DEMO), balance=%s %s",
            account.login,
            account.server,
            account.trade_mode,
            account.balance,
            account.currency,
        )
        return account

    def disconnect(self) -> None:
        """Shut down the MT5 terminal connection cleanly."""
        self._connected = False
        self._backend.shutdown()
        logger.info("MT5 terminal connection shut down cleanly.")

    def verify_demo_account_only(self) -> MT5AccountInfo:
        """Verify that the active MT5 account has `trade_mode == 0 (DEMO)` and matches expected login/server.

        CRITICAL SAFETY INVARIANT:
        - Never relies on server-name heuristics (e.g. 'Demo' in server string).
        - If `trade_mode != 0` (e.g. REAL=2 or CONTEST=1), immediately calls `shutdown()`
          and raises `LiveTradingBlockedError`.
        """
        acct = self._backend.account_info()
        if acct is None:
            self._connected = False
            self._backend.shutdown()
            raise MT5ConnectionError("MT5 account_info() returned None; terminal not authenticated.")

        # 1. Primary mandatory DEMO trade_mode check (0 = ACCOUNT_TRADE_MODE_DEMO)
        if not acct.is_demo:
            self._connected = False
            self._backend.shutdown()
            raise LiveTradingBlockedError(
                f"SAFETY VIOLATION: Connected MT5 account (login={acct.login}, server='{acct.server}') "
                f"has trade_mode={acct.trade_mode} (expected 0=ACCOUNT_TRADE_MODE_DEMO). "
                "Terminal connection has been immediately shut down."
            )

        # 2. Expected login verification
        expected_login = self._config.expected_login or self._config.login
        if expected_login is not None and acct.login != expected_login:
            self._connected = False
            self._backend.shutdown()
            raise MT5AuthenticationError(
                f"SAFETY VIOLATION: Connected MT5 account login ({acct.login}) does not match "
                f"expected_login ({expected_login}). Terminal shut down."
            )

        # 3. Expected server verification
        expected_server = self._config.expected_server or self._config.server
        if (
            expected_server is not None
            and acct.server.strip().lower() != expected_server.strip().lower()
        ):
            self._connected = False
            self._backend.shutdown()
            raise MT5AuthenticationError(
                f"SAFETY VIOLATION: Connected MT5 server ('{acct.server}') does not match "
                f"expected_server ('{expected_server}'). Terminal shut down."
            )

        return acct

    def check_health(self) -> tuple[MT5TerminalInfo, MT5AccountInfo]:
        """Verify terminal connectivity, API permissions, and DEMO account status."""
        term = self._backend.terminal_info()
        if term is None or not term.connected:
            self._connected = False
            raise MT5ConnectionError("MT5 terminal_info() reports disconnected or unavailable.")

        if term.tradeapi_disabled:
            self._connected = False
            raise MT5ConnectionError("MT5 terminal has external Python API trading disabled (tradeapi_disabled=True).")

        acct = self.verify_demo_account_only()
        self._connected = True
        return term, acct

    def ensure_connected(self) -> MT5AccountInfo:
        """Check connection health and perform bounded reconnection with DEMO re-verification if disconnected."""
        try:
            _, acct = self.check_health()
            return acct
        except (LiveTradingBlockedError, MT5AuthenticationError):
            # Never retry safety or authentication violations
            raise
        except MT5ConnectionError as initial_err:
            max_attempts = self._config.max_reconnect_attempts
            last_err: Exception = initial_err
            for attempt in range(1, max_attempts + 1):
                logger.warning(
                    "MT5 connection unhealthy (%s). Reconnect attempt %d/%d...",
                    last_err,
                    attempt,
                    max_attempts,
                )
                try:
                    self._backend.shutdown()
                    return self.connect()
                except (LiveTradingBlockedError, MT5AuthenticationError):
                    raise
                except MT5ConnectionError as rec_err:
                    last_err = rec_err

            self._connected = False
            self._backend.shutdown()
            raise MT5ConnectionError(
                f"Failed to restore MT5 connection after {max_attempts} reconnect attempts: {last_err}"
            ) from last_err

    def get_symbol_info(self, symbol: str = "XAUUSD") -> MT5SymbolInfo:
        """Select `symbol` in MarketWatch and return validated `MT5SymbolInfo`."""
        self.ensure_connected()
        cleaned = symbol.strip().upper()
        if not cleaned or not cleaned.isalnum():
            raise OrderValidationError(f"Invalid symbol identifier '{symbol}'.")

        if not self._backend.symbol_select(cleaned, True):
            err_code, err_msg = self._backend.last_error()
            raise OrderValidationError(
                f"Symbol '{cleaned}' could not be selected in MT5 MarketWatch "
                f"(code={err_code}, message='{err_msg}')."
            )

        sym_info = self._backend.symbol_info(cleaned)
        if sym_info is None or not sym_info.visible:
            raise OrderValidationError(
                f"Symbol '{cleaned}' metadata is unavailable or not visible in MT5."
            )
        return sym_info

    def get_instrument_spec(self, symbol: str = "XAUUSD") -> InstrumentSpec:
        """Map MT5 symbol contract metadata into a validated Phase 1 `InstrumentSpec`."""
        sym_info = self.get_symbol_info(symbol)
        return InstrumentSpec(
            symbol=sym_info.name,
            contract_size=sym_info.trade_contract_size,
            tick_size=sym_info.trade_tick_size,
            tick_value=sym_info.trade_tick_value,
            volume_step=sym_info.volume_step,
            minimum_volume=sym_info.volume_min,
            maximum_volume=sym_info.volume_max,
            margin_rate=sym_info.margin_initial,
        )
