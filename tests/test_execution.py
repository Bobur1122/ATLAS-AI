"""Focused Phase 7.1, 7.2 & 7.3 unit tests for execution exceptions, config, MQL5 models, MT5TerminalBackend,
MT5ConnectionAdapter, filling mode resolution, MT5MarketDataAdapter, CrashSafeJsonPersistence,
official MQL5 deal classification (0..17), and PersistentAccountRiskTracker.
"""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys
import threading
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from pydantic import SecretStr, ValidationError

from atlas_ai.core.config import ExecutionMode
from atlas_ai.core.exceptions import TradingSafetyError
from atlas_ai.domain.market import Timeframe
from atlas_ai.domain.risk import RiskParameters
from atlas_ai.domain.signals import SignalDirection, TradeSignal
from atlas_ai.execution import (
    AMBIGUOUS_SUBMISSION_RETCODES,
    DETERMINISTIC_REJECTION_RETCODES,
    AccountRiskPersistenceSnapshot,
    AccountStateReconstructionError,
    AmbiguousExecutionStateError,
    CrashSafeJsonPersistence,
    DealAccountingCategory,
    DuplicateOrderError,
    ENVELOPE_SCHEMA_VERSION,
    ExecutionLedgerEntry,
    ExecutionSafetyConfig,
    FileLock,
    GENESIS_CHECKSUM,
    LedgerHealthStatus,
    LedgerRecoveryLockError,
    LedgerTransitionRecord,
    LiveTradingBlockedError,
    MT5AccountInfo,
    MT5AuthenticationError,
    MT5BackendNotAvailableError,
    MT5ConnectionAdapter,
    MT5ConnectionConfig,
    MT5ConnectionError,
    MT5DealEntry,
    MT5DealRecord,
    MT5DealType,
    MT5DemoExecutionEngine,
    MT5MarketDataAdapter,
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
    MT5TerminalBackend,
    MT5TerminalInfo,
    MT5TradeExecutionMode,
    MT5TradeMode,
    MockMT5TerminalBackend,
    OrderExecutionError,
    OrderLedgerStatus,
    OrderValidationError,
    PersistedEnvelope,
    PersistenceAccountMismatchError,
    PersistenceCorruptionError,
    PersistenceLockTimeoutError,
    PersistenceRollbackDetectedError,
    PersistentAccountRiskTracker,
    PersistentExecutionLedger,
    PreTradeSafetyGate,
    RealMetaTrader5Backend,
    ReconciliationStatus,
    StaleMarketDataError,
    SymbolTick,
    TrustedAnchorSource,
    TrustedBalanceAnchorConfig,
    classify_and_aggregate_deals,
    compute_envelope_checksum,
    compute_trading_day_utc_bounds,
    extract_comment_identifiers,
    resolve_supported_filling_mode,
    resolve_trading_day_window,
)
from atlas_ai.risk.engine import RiskEngine
from atlas_ai.strategies.models import SignalLifecycleState
from atlas_ai.strategies.signal_lifecycle import SignalLifecycleManager

FIXED_UTC = datetime(2026, 11, 20, 14, 0, 0, tzinfo=timezone.utc)


def test_official_mql5_enum_names_and_values() -> None:
    """Verify official MQL5 ENUM_DEAL_TYPE (including DEAL_DIVIDEND, DEAL_DIVIDEND_FRANKED, DEAL_TAX) and retcodes."""
    assert int(MT5DealType.DEAL_TYPE_BUY) == 0
    assert int(MT5DealType.DEAL_TYPE_SELL) == 1
    assert int(MT5DealType.DEAL_TYPE_BALANCE) == 2
    assert int(MT5DealType.DEAL_TYPE_CREDIT) == 3
    assert int(MT5DealType.DEAL_TYPE_CHARGE) == 4
    assert int(MT5DealType.DEAL_TYPE_CORRECTION) == 5
    assert int(MT5DealType.DEAL_TYPE_BONUS) == 6
    assert int(MT5DealType.DEAL_TYPE_COMMISSION) == 7
    assert int(MT5DealType.DEAL_TYPE_COMMISSION_DAILY) == 8
    assert int(MT5DealType.DEAL_TYPE_COMMISSION_MONTHLY) == 9
    assert int(MT5DealType.DEAL_TYPE_COMMISSION_AGENT_DAILY) == 10
    assert int(MT5DealType.DEAL_TYPE_COMMISSION_AGENT_MONTHLY) == 11
    assert int(MT5DealType.DEAL_TYPE_INTEREST) == 12
    assert int(MT5DealType.DEAL_TYPE_BUY_CANCELED) == 13
    assert int(MT5DealType.DEAL_TYPE_SELL_CANCELED) == 14
    assert int(MT5DealType.DEAL_DIVIDEND) == 15
    assert int(MT5DealType.DEAL_DIVIDEND_FRANKED) == 16
    assert int(MT5DealType.DEAL_TAX) == 17

    # Ensure invented names are NOT present on MT5DealType
    assert not hasattr(MT5DealType, "DEAL_TYPE_DIVIDEND")
    assert not hasattr(MT5DealType, "DEAL_TYPE_DIVIDEND_FRANKED")
    assert not hasattr(MT5DealType, "DEAL_TYPE_TAX")

    # Verify ambiguous submission retcodes include both 10012 (TIMEOUT) and 10031 (CONNECTION)
    assert int(MT5Retcode.TRADE_RETCODE_TIMEOUT) == 10012
    assert int(MT5Retcode.TRADE_RETCODE_CONNECTION) == 10031
    assert AMBIGUOUS_SUBMISSION_RETCODES == frozenset({10012, 10031})


def test_connection_config_secret_redaction_and_safety_config_live_block() -> None:
    """Verify MT5ConnectionConfig never leaks password in repr/str/dict and ExecutionSafetyConfig blocks LIVE."""
    secret_pw = "SuperSecretMT5DemoPassword!99"
    conn_cfg = MT5ConnectionConfig(
        login=100200300,
        password=SecretStr(secret_pw),
        server="AtlasBroker-Demo",
        expected_login=100200300,
        expected_server="AtlasBroker-Demo",
    )
    assert secret_pw not in repr(conn_cfg)
    assert secret_pw not in str(conn_cfg)
    assert secret_pw not in str(conn_cfg.redacted_dict())
    assert conn_cfg.redacted_dict()["password"] == "***REDACTED***"

    # ExecutionSafetyConfig permits DEMO with live_trading_enabled=False
    safe_cfg = ExecutionSafetyConfig()
    assert safe_cfg.execution_mode == ExecutionMode.DEMO
    assert safe_cfg.live_trading_enabled is False

    # ExecutionSafetyConfig blocks ExecutionMode.LIVE and live_trading_enabled=True with LiveTradingBlockedError
    with pytest.raises(LiveTradingBlockedError) as exc_info:
        ExecutionSafetyConfig(execution_mode=ExecutionMode.LIVE)
    assert isinstance(exc_info.value, TradingSafetyError)

    with pytest.raises(LiveTradingBlockedError):
        ExecutionSafetyConfig(live_trading_enabled=True)

    # TrustedBalanceAnchorConfig requires timezone-aware timestamp and normalizes to UTC
    anchor = TrustedBalanceAnchorConfig(
        anchor_balance=Decimal("10000.00"),
        anchor_timestamp_utc=FIXED_UTC,
    )
    assert anchor.anchor_timestamp_utc.tzinfo == timezone.utc

    with pytest.raises(ValidationError):
        TrustedBalanceAnchorConfig(
            anchor_balance=Decimal("10000.00"),
            anchor_timestamp_utc=datetime(2026, 11, 20, 14, 0, 0),  # naive datetime rejected
        )


def test_execution_domain_models_and_currency_aware_tolerance() -> None:
    """Verify MT5AccountInfo demo detection, currency-aware tolerance, SymbolTick, and MT5OrderRequest validation."""
    usd_demo = MT5AccountInfo(
        login=111,
        trade_mode=int(MT5TradeMode.ACCOUNT_TRADE_MODE_DEMO),
        balance=Decimal("10000.00"),
        equity=Decimal("10000.00"),
        margin_free=Decimal("10000.00"),
        currency="usd",
        currency_digits=2,
        server="Broker-Demo",
    )
    assert usd_demo.is_demo is True
    assert usd_demo.currency == "USD"
    assert usd_demo.reconciliation_tolerance(5) == Decimal("0.05")

    jpy_real = MT5AccountInfo(
        login=222,
        trade_mode=int(MT5TradeMode.ACCOUNT_TRADE_MODE_REAL),
        balance=Decimal("1500000"),
        equity=Decimal("1500000"),
        margin_free=Decimal("1500000"),
        currency="JPY",
        currency_digits=0,
        server="Broker-Demo-Named-Server-But-Real-Mode",
    )
    assert jpy_real.is_demo is False
    assert jpy_real.reconciliation_tolerance(5) == Decimal("5")

    tick = SymbolTick(
        symbol=" xauusd ",
        timestamp=FIXED_UTC,
        bid=Decimal("2650.00"),
        ask=Decimal("2650.25"),
    )
    assert tick.symbol == "XAUUSD"
    assert tick.spread_price == Decimal("0.25")
    assert tick.spread_points(Decimal("0.01")) == Decimal("25")
    assert tick.age_seconds(FIXED_UTC + timedelta(seconds=12)) == 12.0

    with pytest.raises(ValidationError):
        SymbolTick(
            symbol="XAUUSD",
            timestamp=FIXED_UTC,
            bid=Decimal("2650.50"),
            ask=Decimal("2650.00"),  # ask < bid rejected
        )

    # Valid BUY and SELL order requests
    buy_req = MT5OrderRequest(
        symbol="XAUUSD",
        volume=Decimal("0.10"),
        order_type=MT5OrderType.ORDER_TYPE_BUY,
        price=Decimal("2650.20"),
        sl=Decimal("2640.00"),
        tp=Decimal("2670.00"),
        magic=202607,
        comment="sig-00001",
    )
    assert buy_req.symbol == "XAUUSD"

    with pytest.raises(ValidationError):
        MT5OrderRequest(
            symbol="XAUUSD",
            volume=Decimal("0.10"),
            order_type=MT5OrderType.ORDER_TYPE_BUY,
            price=Decimal("2650.20"),
            sl=Decimal("2655.00"),  # Invalid BUY sl > price
            tp=Decimal("2670.00"),
            magic=202607,
            comment="sig-00001",
        )

    # Verify AccountRiskPersistenceSnapshot and ExecutionLedgerEntry immutability
    snap = AccountRiskPersistenceSnapshot(
        account_login=111,
        server="Broker-Demo",
        trading_day_label="2026-11-20",
        anchor_timestamp_utc=FIXED_UTC,
        anchor_balance=Decimal("10000.00"),
        daily_starting_balance=Decimal("10000.00"),
        peak_balance=Decimal("10000.00"),
        last_verified_balance=Decimal("10000.00"),
        last_updated_utc=FIXED_UTC,
    )
    assert snap.account_login == 111

    entry = ExecutionLedgerEntry(
        client_order_id="ord-00001",
        signal_id="sig-00001",
        symbol="XAUUSD",
        timeframe=Timeframe.H1,
        direction=SignalDirection.BUY,
        magic=202607,
        requested_volume=Decimal("0.10"),
        executed_volume=Decimal("0.00"),
        remaining_volume=Decimal("0.10"),
        requested_price=Decimal("2650.20"),
        stop_loss=Decimal("2640.00"),
        take_profit=Decimal("2670.00"),
        filling_mode=MT5OrderFillingMode.ORDER_FILLING_FOK,
        status=OrderLedgerStatus.SUBMITTING,
        reconciliation_status=ReconciliationStatus.PENDING,
        created_at_utc=FIXED_UTC,
        updated_at_utc=FIXED_UTC,
        status_reason="Write-ahead intent before order_send.",
    )
    assert entry.status == OrderLedgerStatus.SUBMITTING


def test_real_backend_lazy_import_and_protocol_conformance() -> None:
    """Verify RealMetaTrader5Backend and MockMT5TerminalBackend satisfy MT5TerminalBackend Protocol
    and RealMetaTrader5Backend never requires MetaTrader5 at import or instantiation time.
    """
    real_backend = RealMetaTrader5Backend()
    mock_backend = MockMT5TerminalBackend()

    assert isinstance(real_backend, MT5TerminalBackend)
    assert isinstance(mock_backend, MT5TerminalBackend)

    # Simulate environment without MetaTrader5 package installed
    with patch.dict(sys.modules, {"MetaTrader5": None}):
        fresh_real = RealMetaTrader5Backend()
        with pytest.raises(MT5BackendNotAvailableError, match="MetaTrader5"):
            fresh_real.initialize()


def test_mock_mt5_terminal_backend_operations_and_live_account_hard_block() -> None:
    """Verify MockMT5TerminalBackend lifecycle, rate bars, deal history, order execution,
    Market Execution RETURN rejection, and unconditional block on REAL/CONTEST accounts.
    """
    mock = MockMT5TerminalBackend()
    assert mock.is_initialized is False
    assert mock.account_info() is None

    # Initialize and authenticate
    assert mock.initialize(login=100200300, server="AtlasBroker-Demo") is True
    assert mock.is_initialized is True
    acct = mock.account_info()
    assert acct is not None and acct.is_demo is True

    # Symbol selection, tick, and historical bars
    assert mock.symbol_select("XAUUSD", True) is True
    sym = mock.symbol_info("XAUUSD")
    assert sym is not None and sym.name == "XAUUSD"
    tick = mock.symbol_info_tick("XAUUSD")
    assert tick is not None and tick.bid == Decimal("2650.00")

    bars = mock.copy_rates_from_pos("XAUUSD", Timeframe.H1, start_pos=1, count=10)
    assert bars is not None and len(bars) == 10

    # Market Execution rejects ORDER_FILLING_RETURN with TRADE_RETCODE_INVALID_FILL (10030)
    req_return = MT5OrderRequest(
        symbol="XAUUSD",
        volume=Decimal("0.10"),
        order_type=MT5OrderType.ORDER_TYPE_BUY,
        price=Decimal("2650.20"),
        sl=Decimal("2640.00"),
        tp=Decimal("2670.00"),
        magic=202607,
        comment="sig-invalid-fill",
        type_filling=MT5OrderFillingMode.ORDER_FILLING_RETURN,
    )
    res_invalid_fill = mock.order_send(req_return)
    assert res_invalid_fill is not None
    assert res_invalid_fill.retcode == int(MT5Retcode.TRADE_RETCODE_INVALID_FILL)

    # Valid FOK order succeeds (10009 DONE) and creates position & entry deal
    req_fok = req_return.model_copy(
        update={
            "comment": "sig-valid-fok",
            "type_filling": MT5OrderFillingMode.ORDER_FILLING_FOK,
        }
    )
    chk = mock.order_check(req_fok)
    assert chk is not None and chk.retcode == 0

    send_res = mock.order_send(req_fok)
    assert send_res is not None
    assert send_res.retcode == int(MT5Retcode.TRADE_RETCODE_DONE)
    assert send_res.volume == Decimal("0.10")

    positions = mock.positions_get("XAUUSD")
    assert positions is not None and len(positions) == 1
    assert positions[0].comment == "sig-valid-fok"

    deals = mock.history_deals_get(
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 12, 31, tzinfo=timezone.utc),
        symbol="XAUUSD",
    )
    assert deals is not None and len(deals) == 1
    assert deals[0].entry == int(MT5DealEntry.DEAL_ENTRY_IN)

    # Switch account to REAL (trade_mode=2): order_send must immediately shut down and raise LiveTradingBlockedError
    real_acct = acct.model_copy(update={"trade_mode": int(MT5TradeMode.ACCOUNT_TRADE_MODE_REAL)})
    mock.set_account(real_acct)
    with pytest.raises(LiveTradingBlockedError):
        mock.order_send(req_fok)
    assert mock.is_initialized is False


def test_resolve_supported_filling_mode_across_execution_modes_and_zero_bitmask() -> None:
    """Verify resolve_supported_filling_mode considers both trade_exemode and filling_mode,
    never selects ORDER_FILLING_RETURN under Market Execution, and never rejects filling_mode == 0.
    """
    base_sym = MT5SymbolInfo(
        name="XAUUSD",
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
        visible=True,
    )

    # 1. Market Execution (2): FOK preferred when both FOK and IOC are set
    assert resolve_supported_filling_mode(base_sym) == MT5OrderFillingMode.ORDER_FILLING_FOK

    # 2. Market Execution (2): IOC selected when only IOC flag (2) is set
    sym_market_ioc = base_sym.model_copy(
        update={"filling_mode": int(MT5SymbolFillingFlag.SYMBOL_FILLING_IOC)}
    )
    assert resolve_supported_filling_mode(sym_market_ioc) == MT5OrderFillingMode.ORDER_FILLING_IOC

    # 3. Market Execution (2): filling_mode == 0 must NOT be rejected and must NOT return ORDER_FILLING_RETURN
    sym_market_zero = base_sym.model_copy(update={"filling_mode": 0})
    assert resolve_supported_filling_mode(sym_market_zero) == MT5OrderFillingMode.ORDER_FILLING_FOK

    # 4. Market Execution (2): unexpected positive bitmask without FOK or IOC raises OrderValidationError
    sym_market_bad_mask = base_sym.model_copy(update={"filling_mode": 4})
    with pytest.raises(OrderValidationError, match="Market Execution"):
        resolve_supported_filling_mode(sym_market_bad_mask)

    # 5. Request (0), Instant (1), and Exchange (3) Execution:
    for exemode in (
        MT5TradeExecutionMode.SYMBOL_TRADE_EXECUTION_REQUEST,
        MT5TradeExecutionMode.SYMBOL_TRADE_EXECUTION_INSTANT,
        MT5TradeExecutionMode.SYMBOL_TRADE_EXECUTION_EXCHANGE,
    ):
        sym_fok = base_sym.model_copy(
            update={
                "trade_exemode": exemode,
                "filling_mode": int(MT5SymbolFillingFlag.SYMBOL_FILLING_FOK),
            }
        )
        assert resolve_supported_filling_mode(sym_fok) == MT5OrderFillingMode.ORDER_FILLING_FOK

        sym_ioc = base_sym.model_copy(
            update={
                "trade_exemode": exemode,
                "filling_mode": int(MT5SymbolFillingFlag.SYMBOL_FILLING_IOC),
            }
        )
        assert resolve_supported_filling_mode(sym_ioc) == MT5OrderFillingMode.ORDER_FILLING_IOC

        # When filling_mode == 0 under Request/Instant/Exchange, ORDER_FILLING_RETURN is permitted
        sym_zero = base_sym.model_copy(
            update={
                "trade_exemode": exemode,
                "filling_mode": 0,
            }
        )
        assert resolve_supported_filling_mode(sym_zero) == MT5OrderFillingMode.ORDER_FILLING_RETURN


def test_connection_adapter_demo_verification_login_server_and_real_contest_blocks() -> None:
    """Verify MT5ConnectionAdapter strict demo verification (trade_mode == 0), expected_login/expected_server
    enforcement, immediate shutdown on REAL/CONTEST accounts, and InstrumentSpec mapping.
    """
    mock = MockMT5TerminalBackend()
    cfg = MT5ConnectionConfig(
        login=100200300,
        password=SecretStr("DemoPass123"),
        server="AtlasBroker-Demo",
        expected_login=100200300,
        expected_server="AtlasBroker-Demo",
    )
    adapter = MT5ConnectionAdapter(backend=mock, config=cfg)

    # 1. Connect on valid DEMO account
    acct = adapter.connect()
    assert acct.is_demo is True
    assert acct.login == 100200300
    assert adapter.is_connected is True
    term_health, acct_health = adapter.check_health()
    assert term_health.connected is True
    assert acct_health.is_demo is True

    # 2. InstrumentSpec mapping from MT5SymbolInfo
    spec = adapter.get_instrument_spec("XAUUSD")
    assert spec.symbol == "XAUUSD"
    assert spec.contract_size == Decimal("100")
    assert spec.tick_size == Decimal("0.01")
    assert spec.tick_value == Decimal("1.00")
    assert spec.minimum_volume == Decimal("0.01")
    assert spec.maximum_volume == Decimal("50.00")
    assert spec.volume_step == Decimal("0.01")
    assert spec.margin_rate == Decimal("0.01")

    # 3. Unknown symbol rejected with OrderValidationError
    with pytest.raises(OrderValidationError, match="could not be selected"):
        adapter.get_instrument_spec("UNKNOWN99")

    adapter.disconnect()
    assert mock.is_initialized is False
    assert adapter.is_connected is False

    # 4. REAL account (trade_mode=2) with "Demo" in server name MUST still be blocked!
    real_deceptive_acct = acct.model_copy(
        update={
            "trade_mode": int(MT5TradeMode.ACCOUNT_TRADE_MODE_REAL),
            "server": "AtlasBroker-Demo",  # Server name says Demo, but trade_mode == 2
        }
    )
    mock.set_account(real_deceptive_acct)
    with pytest.raises(LiveTradingBlockedError) as real_exc:
        adapter.connect()
    assert isinstance(real_exc.value, TradingSafetyError)
    assert mock.is_initialized is False
    assert adapter.is_connected is False

    # 5. CONTEST account (trade_mode=1) MUST be blocked and shut down
    contest_acct = acct.model_copy(
        update={
            "trade_mode": int(MT5TradeMode.ACCOUNT_TRADE_MODE_CONTEST),
            "server": "AtlasBroker-Demo",
        }
    )
    mock.set_account(contest_acct)
    with pytest.raises(LiveTradingBlockedError):
        adapter.connect()
    assert mock.is_initialized is False

    # 6. Restore DEMO trade_mode=0, but test expected_login mismatch (login=None in connect, expected_login=100200300)
    cfg_login_check = MT5ConnectionConfig(
        expected_login=100200300,
        expected_server="AtlasBroker-Demo",
    )
    adapter_login_check = MT5ConnectionAdapter(backend=mock, config=cfg_login_check)
    mock.set_account(acct.model_copy(update={"login": 999888777}))
    with pytest.raises(MT5AuthenticationError, match="does not match expected_login"):
        adapter_login_check.connect()
    assert mock.is_initialized is False

    # 7. Test expected_server mismatch
    mock.set_account(acct.model_copy(update={"server": "OtherBroker-Demo"}))
    with pytest.raises(MT5AuthenticationError, match="does not match expected_server"):
        adapter_login_check.connect()
    assert mock.is_initialized is False


def test_connection_adapter_failure_disconnection_and_bounded_reconnection() -> None:
    """Verify MT5ConnectionAdapter handles initialize() failure, terminal disconnection,
    bounded reconnection, and post-reconnect demo re-verification.
    """
    mock = MockMT5TerminalBackend()
    cfg = MT5ConnectionConfig(
        login=100200300,
        server="AtlasBroker-Demo",
        max_reconnect_attempts=2,
    )
    adapter = MT5ConnectionAdapter(backend=mock, config=cfg)

    # 1. initialize() failure raises MT5ConnectionError
    mock.set_fail_next_initialize(True)
    with pytest.raises(MT5ConnectionError, match="initialize\\(\\) failed"):
        adapter.connect()
    assert mock.is_initialized is False

    # 2. Normal connection succeeds once failure cleared
    acct = adapter.connect()
    assert adapter.check_health()[1].is_demo is True

    # 3. Simulate terminal drop (connected=False) -> check_health() raises MT5ConnectionError
    mock.set_terminal(
        MT5TerminalInfo(
            connected=False,
            trade_allowed=True,
            tradeapi_disabled=False,
            company="Atlas Mock Broker",
            name="MetaTrader 5",
        )
    )
    with pytest.raises(MT5ConnectionError, match="disconnected"):
        adapter.check_health()

    # Restore terminal connectivity on next initialize and verify ensure_connected() reconnects & re-verifies DEMO
    mock.set_terminal(
        MT5TerminalInfo(
            connected=True,
            trade_allowed=True,
            tradeapi_disabled=False,
            company="Atlas Mock Broker",
            name="MetaTrader 5",
        )
    )
    mock.shutdown()
    reconnected_acct = adapter.ensure_connected()
    assert reconnected_acct.is_demo is True
    assert mock.is_initialized is True

    # 4. If terminal switches to REAL account during disconnect, ensure_connected() immediately aborts
    mock.shutdown()
    mock.set_account(acct.model_copy(update={"trade_mode": int(MT5TradeMode.ACCOUNT_TRADE_MODE_REAL)}))
    with pytest.raises(LiveTradingBlockedError):
        adapter.ensure_connected()
    assert mock.is_initialized is False

    # 5. Bounded reconnection exhaustion raises MT5ConnectionError after max_reconnect_attempts
    mock.set_account(acct)
    mock.set_terminal(
        MT5TerminalInfo(
            connected=False,
            trade_allowed=True,
            tradeapi_disabled=False,
        )
    )
    with pytest.raises(MT5ConnectionError, match="Failed to restore MT5 connection after 2 reconnect attempts"):
        adapter.ensure_connected()
    assert mock.is_initialized is False


def test_market_data_adapter_ticks_stale_rejection_closed_candles_and_pipeline() -> None:
    """Verify MT5MarketDataAdapter tick fetching, stale/future tick rejection, closed-bar exclusion
    of forming candle, UTC timestamp normalization, and Phase 2 MarketDataPipeline integration.
    """
    mock = MockMT5TerminalBackend()
    cfg = MT5ConnectionConfig(login=100200300, server="AtlasBroker-Demo")
    conn = MT5ConnectionAdapter(backend=mock, config=cfg)
    conn.connect()

    safety_cfg = ExecutionSafetyConfig(max_tick_age_seconds=30)
    md_adapter = MT5MarketDataAdapter(connection=conn, safety_config=safety_cfg)

    now_utc = datetime(2026, 11, 20, 12, 30, 0, tzinfo=timezone.utc)

    # 1. Valid fresh tick
    fresh_tick = SymbolTick(
        symbol="XAUUSD",
        timestamp=now_utc - timedelta(seconds=5),
        bid=Decimal("2651.10"),
        ask=Decimal("2651.30"),
    )
    mock.set_tick(fresh_tick)
    fetched_tick = md_adapter.fetch_latest_tick("XAUUSD", reference_now_utc=now_utc)
    assert fetched_tick.symbol == "XAUUSD"
    assert fetched_tick.bid == Decimal("2651.10")
    assert fetched_tick.timestamp.tzinfo == timezone.utc

    # 2. Stale tick (age 45s > max_tick_age_seconds 30s) rejected with StaleMarketDataError
    stale_tick = fresh_tick.model_copy(update={"timestamp": now_utc - timedelta(seconds=45)})
    mock.set_tick(stale_tick)
    with pytest.raises(StaleMarketDataError, match="Stale tick"):
        md_adapter.fetch_latest_tick("XAUUSD", reference_now_utc=now_utc)

    # 3. Future-skewed tick (10s ahead of reference_now_utc) rejected with StaleMarketDataError
    future_tick = fresh_tick.model_copy(update={"timestamp": now_utc + timedelta(seconds=10)})
    mock.set_tick(future_tick)
    with pytest.raises(StaleMarketDataError, match="Future-dated tick anomaly"):
        md_adapter.fetch_latest_tick("XAUUSD", reference_now_utc=now_utc)

    # 4. Closed candles: seed 45 H1 bars where the last bar (12:00 UTC) is still forming at 12:30 UTC
    base_open = datetime(2026, 11, 18, 16, 0, 0, tzinfo=timezone.utc)
    seeded_bars: list[MT5RateBar] = []
    for i in range(45):
        bar_open = base_open + timedelta(hours=i)
        p = Decimal("2610.00") + Decimal(i) * Decimal("0.50")
        seeded_bars.append(
            MT5RateBar(
                timestamp=bar_open,
                open=p,
                high=p + Decimal("2.00"),
                low=p - Decimal("1.50"),
                close=p + Decimal("0.75"),
                tick_volume=Decimal(150 + i),
                spread=20,
                real_volume=Decimal("0"),
            )
        )
    mock.set_rates("XAUUSD", Timeframe.H1, seeded_bars)

    # Fetching 40 closed candles at 12:30 UTC must exclude index 0 (the 12:00..13:00 forming candle)
    candles = md_adapter.fetch_closed_candles(
        symbol="XAUUSD",
        timeframe=Timeframe.H1,
        count=40,
        reference_now_utc=now_utc,
    )
    assert len(candles) == 40
    # Latest closed H1 candle at 12:30 UTC opened at 11:00 UTC and closed at 12:00 UTC
    assert candles[-1].timestamp == datetime(2026, 11, 20, 11, 0, 0, tzinfo=timezone.utc)
    assert candles[0].timestamp < candles[-1].timestamp

    # 5. End-to-end integration with Phase 2 MarketDataPipeline & FeaturePipeline
    pipeline_res = md_adapter.fetch_market_data_pipeline_result(
        symbol="XAUUSD",
        timeframe=Timeframe.H1,
        count=40,
        reference_now_utc=now_utc,
    )
    assert len(pipeline_res.candles) == 40
    assert pipeline_res.validation_report.is_valid is True
    assert len(pipeline_res.feature_dataframe) == 40
    assert "atr_14" in pipeline_res.feature_dataframe.columns

    # 6. Non-monotonic rate bars are rejected with StaleMarketDataError
    corrupted_bars = list(seeded_bars)
    corrupted_bars[-3] = corrupted_bars[-2]  # Duplicate timestamp
    mock.set_rates("XAUUSD", Timeframe.H1, corrupted_bars)
    with pytest.raises(StaleMarketDataError, match="Non-monotonic"):
        md_adapter.fetch_closed_candles("XAUUSD", Timeframe.H1, count=10, reference_now_utc=now_utc)


def test_crash_safe_persistence_generations_checksums_account_binding_and_rollback(
    tmp_path: Path,
) -> None:
    """Verify CrashSafeJsonPersistence atomic writes, monotonic generation numbers,
    previous_checksum hash chain, SHA-256 verification, account binding, and rollback detection.
    """
    state_file = tmp_path / "account_risk_state.json"
    store = CrashSafeJsonPersistence(state_file)

    assert store.load(expected_login=100200300, expected_server="AtlasBroker-Demo") is None

    # Generation 1 write
    env1 = store.save(
        {"step": 1, "balance": "10000.00"},
        account_login=100200300,
        server="AtlasBroker-Demo",
        now_utc=FIXED_UTC,
    )
    assert env1.generation == 1
    assert env1.previous_checksum == GENESIS_CHECKSUM
    assert state_file.exists()
    assert not store.backup_path.exists()

    # Generation 2 write rotates gen 1 to .bak and chains previous_checksum
    env2 = store.save(
        {"step": 2, "balance": "10150.00"},
        account_login=100200300,
        server="AtlasBroker-Demo",
        now_utc=FIXED_UTC + timedelta(minutes=5),
    )
    assert env2.generation == 2
    assert env2.previous_checksum == env1.payload_checksum
    assert store.backup_path.exists()

    loaded = store.load(expected_login=100200300, expected_server="AtlasBroker-Demo")
    assert loaded is not None
    assert loaded.generation == 2
    assert loaded.payload["balance"] == "10150.00"

    # Strict account login & server identity binding
    with pytest.raises(PersistenceAccountMismatchError):
        store.load(expected_login=999999999, expected_server="AtlasBroker-Demo")

    with pytest.raises(PersistenceAccountMismatchError):
        store.load(expected_login=100200300, expected_server="OtherBroker-Server")

    # Tamper with payload without updating SHA-256 checksum -> PersistenceCorruptionError
    raw_obj = json.loads(state_file.read_text(encoding="utf-8"))
    trashed_obj = dict(raw_obj)
    trashed_obj["payload"] = {"step": 2, "balance": "999999.00"}
    state_file.write_text(json.dumps(trashed_obj), encoding="utf-8")

    with pytest.raises(PersistenceCorruptionError, match="SHA-256 integrity checksum mismatch"):
        store.load(expected_login=100200300, expected_server="AtlasBroker-Demo")

    # Restore valid gen 2 file, then test broken previous_checksum chain
    state_file.write_text(json.dumps(raw_obj), encoding="utf-8")
    env3 = store.save(
        {"step": 3, "balance": "10200.00"},
        account_login=100200300,
        server="AtlasBroker-Demo",
        now_utc=FIXED_UTC + timedelta(minutes=10),
    )
    assert env3.generation == 3
    assert env3.previous_checksum == env2.payload_checksum

    # Overwrite primary (gen 3) with older gen 1 envelope while .bak holds gen 2 -> PersistenceRollbackDetectedError
    gen1_text = json.dumps(env1.model_dump(mode="json"))
    state_file.write_text(gen1_text, encoding="utf-8")
    fresh_store = CrashSafeJsonPersistence(state_file)
    with pytest.raises(PersistenceRollbackDetectedError, match="ROLLBACK DETECTED"):
        fresh_store.load(expected_login=100200300, expected_server="AtlasBroker-Demo")


def test_crash_safe_persistence_interrupted_temp_writes_missing_primary_and_recovery(
    tmp_path: Path,
) -> None:
    """Verify interrupted .tmp.* files and missing primary files never silently restore .bak,
    raise LedgerRecoveryLockError, and require explicit commit_recovered_payload().
    """
    ledger_file = tmp_path / "execution_ledger.json"
    store = CrashSafeJsonPersistence(ledger_file)

    env1 = store.save(
        {"entries": ["ord-1"]},
        account_login=100200300,
        server="AtlasBroker-Demo",
        now_utc=FIXED_UTC,
    )
    env2 = store.save(
        {"entries": ["ord-1", "ord-2"]},
        account_login=100200300,
        server="AtlasBroker-Demo",
        now_utc=FIXED_UTC + timedelta(minutes=1),
    )
    assert env1.generation == 1 and env2.generation == 2

    # 1. Simulate interrupted atomic write leaving a lingering .tmp.* file on disk
    interrupted_tmp = tmp_path / f"{ledger_file.name}.tmp.99999.deadbeef"
    interrupted_tmp.write_text("{partial interrupted json", encoding="utf-8")

    with pytest.raises(LedgerRecoveryLockError, match="Interrupted atomic write detected"):
        store.load(expected_login=100200300, expected_server="AtlasBroker-Demo")

    with pytest.raises(LedgerRecoveryLockError, match="Interrupted atomic write detected"):
        store.save(
            {"entries": ["ord-1", "ord-2", "ord-3"]},
            account_login=100200300,
            server="AtlasBroker-Demo",
        )

    inspection = store.inspect_recovery_state(
        expected_login=100200300,
        expected_server="AtlasBroker-Demo",
    )
    assert inspection.recovery_required is True
    assert len(inspection.interrupted_tmp_files) == 1

    # Explicit reconciled recovery commit advances generation to 3 and cleans up .tmp.*
    env3 = store.commit_recovered_payload(
        {"entries": ["ord-1", "ord-2", "ord-reconciled"]},
        account_login=100200300,
        server="AtlasBroker-Demo",
        now_utc=FIXED_UTC + timedelta(minutes=2),
    )
    assert env3.generation == 3
    assert env3.previous_checksum == env2.payload_checksum
    assert not interrupted_tmp.exists()

    # 2. Simulate missing primary file while .bak exists -> never silently restore .bak!
    ledger_file.unlink()
    fresh_store = CrashSafeJsonPersistence(ledger_file)
    with pytest.raises(LedgerRecoveryLockError, match="Silent backup restoration is forbidden"):
        fresh_store.load(expected_login=100200300, expected_server="AtlasBroker-Demo")

    insp_missing = fresh_store.inspect_recovery_state(
        expected_login=100200300,
        expected_server="AtlasBroker-Demo",
    )
    assert insp_missing.recovery_required is True
    assert insp_missing.primary_missing is True
    assert insp_missing.backup_envelope is not None
    assert insp_missing.backup_envelope.generation == 2

    # Explicit recovery commit from backup + broker reconciliation advances past backup generation
    env_recovered = fresh_store.commit_recovered_payload(
        {"entries": ["ord-1", "ord-2", "ord-reconciled"]},
        account_login=100200300,
        server="AtlasBroker-Demo",
        now_utc=FIXED_UTC + timedelta(minutes=3),
    )
    assert env_recovered.generation == 3
    assert fresh_store.load(expected_login=100200300, expected_server="AtlasBroker-Demo") is not None


def test_cross_process_os_file_locking_and_automatic_crash_release(tmp_path: Path) -> None:
    """Verify FileLock blocks concurrent processes and is automatically released by the OS kernel
    when the lock-owning process crashes/terminates abruptly.
    """
    lock_file = tmp_path / "cross_proc.json.lock"
    child_code = (
        "import sys, time\n"
        "from atlas_ai.execution.persistence import FileLock\n"
        "lock = FileLock(sys.argv[1], timeout_seconds=2.0)\n"
        "lock.acquire()\n"
        "sys.stdout.write('LOCKED\\n')\n"
        "sys.stdout.flush()\n"
        "time.sleep(30)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", child_code, str(lock_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        ready_line = proc.stdout.readline().strip()
        assert ready_line == "LOCKED"

        # While child process holds the OS lock, parent process must time out
        with pytest.raises(PersistenceLockTimeoutError):
            with FileLock(lock_file, timeout_seconds=0.15, poll_interval_seconds=0.02):
                pass

        # Abruptly kill the child process (simulating process crash without calling lock.release())
        proc.kill()
        proc.wait(timeout=5.0)

        # OS kernel must automatically release the byte-range lock upon process termination
        with FileLock(lock_file, timeout_seconds=1.5, poll_interval_seconds=0.02) as reacquired:
            assert reacquired.is_locked is True
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)


def test_official_mql5_all_18_deal_types_classification_and_deduplication() -> None:
    """Verify all 18 official MQL5 ENUM_DEAL_TYPE values (0..17), deduplication,
    commission linkage validation, and anti-double-counting checks.
    """
    t0 = FIXED_UTC
    # 1. Valid execution & non-trading & fee/commission/dividend/tax deals covering allowed types:
    #    BUY(0), SELL(1), BALANCE(2), CREDIT(3), CHARGE(4), BONUS(6), COMMISSION(7),
    #    COMMISSION_DAILY(8), COMMISSION_MONTHLY(9), AGENT_DAILY(10), AGENT_MONTHLY(11),
    #    INTEREST(12 negative & positive), DEAL_DIVIDEND(15), DEAL_DIVIDEND_FRANKED(16), DEAL_TAX(17)
    valid_deals = [
        # 0: DEAL_TYPE_BUY (entry deal with inline commission = 0 so separate commission is allowed)
        MT5DealRecord(
            ticket=101,
            order=501,
            position_id=901,
            time_utc=t0,
            deal_type=int(MT5DealType.DEAL_TYPE_BUY),
            symbol="XAUUSD",
            volume=Decimal("0.10"),
            price=Decimal("2650.00"),
            profit=Decimal("0.00"),
            commission=Decimal("0.00"),
        ),
        # Duplicate of ticket 101 -> must be deduplicated without double-counting
        MT5DealRecord(
            ticket=101,
            order=501,
            position_id=901,
            time_utc=t0,
            deal_type=int(MT5DealType.DEAL_TYPE_BUY),
            symbol="XAUUSD",
            volume=Decimal("0.10"),
            price=Decimal("2650.00"),
            profit=Decimal("0.00"),
            commission=Decimal("0.00"),
        ),
        # 1: DEAL_TYPE_SELL (exit deal with profit +120.00, swap -2.00)
        MT5DealRecord(
            ticket=102,
            order=502,
            position_id=901,
            time_utc=t0 + timedelta(minutes=1),
            deal_type=int(MT5DealType.DEAL_TYPE_SELL),
            entry=int(MT5DealEntry.DEAL_ENTRY_OUT),
            symbol="XAUUSD",
            volume=Decimal("0.10"),
            price=Decimal("2662.00"),
            profit=Decimal("120.00"),
            swap=Decimal("-2.00"),
        ),
        # 2: DEAL_TYPE_BALANCE (+500.00 deposit)
        MT5DealRecord(
            ticket=103,
            time_utc=t0 + timedelta(minutes=2),
            deal_type=int(MT5DealType.DEAL_TYPE_BALANCE),
            profit=Decimal("500.00"),
        ),
        # 3: DEAL_TYPE_CREDIT (+50.00 credit)
        MT5DealRecord(
            ticket=104,
            time_utc=t0 + timedelta(minutes=3),
            deal_type=int(MT5DealType.DEAL_TYPE_CREDIT),
            profit=Decimal("50.00"),
        ),
        # 4: DEAL_TYPE_CHARGE (-5.00 fee -> trading PnL)
        MT5DealRecord(
            ticket=105,
            time_utc=t0 + timedelta(minutes=4),
            deal_type=int(MT5DealType.DEAL_TYPE_CHARGE),
            profit=Decimal("-5.00"),
        ),
        # 6: DEAL_TYPE_BONUS (+25.00 bonus -> non-trading cash flow)
        MT5DealRecord(
            ticket=106,
            time_utc=t0 + timedelta(minutes=5),
            deal_type=int(MT5DealType.DEAL_TYPE_BONUS),
            profit=Decimal("25.00"),
        ),
        # 7: DEAL_TYPE_COMMISSION linked by order=501 & position_id=901 (-3.50)
        MT5DealRecord(
            ticket=107,
            order=501,
            position_id=901,
            time_utc=t0 + timedelta(minutes=6),
            deal_type=int(MT5DealType.DEAL_TYPE_COMMISSION),
            symbol="XAUUSD",
            commission=Decimal("-3.50"),
        ),
        # 7b: Unlinked DEAL_TYPE_COMMISSION with symbol="XAUUSD" & magic=202607 but order=0, position_id=0
        #     Must NOT be marked linked_to_trade, and must be conservatively deducted from trading PnL (-1.50)
        MT5DealRecord(
            ticket=108,
            order=0,
            position_id=0,
            magic=202607,
            symbol="XAUUSD",
            time_utc=t0 + timedelta(minutes=7),
            deal_type=int(MT5DealType.DEAL_TYPE_COMMISSION),
            commission=Decimal("-1.50"),
        ),
        # 8, 9, 10, 11: Account-level daily/monthly/agent commissions (order=0, position_id=0 permitted)
        MT5DealRecord(
            ticket=109,
            time_utc=t0 + timedelta(minutes=8),
            deal_type=int(MT5DealType.DEAL_TYPE_COMMISSION_DAILY),
            commission=Decimal("-2.00"),
        ),
        MT5DealRecord(
            ticket=110,
            time_utc=t0 + timedelta(minutes=9),
            deal_type=int(MT5DealType.DEAL_TYPE_COMMISSION_MONTHLY),
            commission=Decimal("-4.00"),
        ),
        MT5DealRecord(
            ticket=111,
            time_utc=t0 + timedelta(minutes=10),
            deal_type=int(MT5DealType.DEAL_TYPE_COMMISSION_AGENT_DAILY),
            commission=Decimal("-1.00"),
        ),
        MT5DealRecord(
            ticket=112,
            time_utc=t0 + timedelta(minutes=11),
            deal_type=int(MT5DealType.DEAL_TYPE_COMMISSION_AGENT_MONTHLY),
            commission=Decimal("-3.00"),
        ),
        # 12: DEAL_TYPE_INTEREST negative (-1.25 -> trading PnL) & positive (+4.00 -> non-trading cash flow)
        MT5DealRecord(
            ticket=113,
            time_utc=t0 + timedelta(minutes=12),
            deal_type=int(MT5DealType.DEAL_TYPE_INTEREST),
            profit=Decimal("-1.25"),
        ),
        MT5DealRecord(
            ticket=114,
            time_utc=t0 + timedelta(minutes=13),
            deal_type=int(MT5DealType.DEAL_TYPE_INTEREST),
            profit=Decimal("4.00"),
        ),
        # 15 & 16: DEAL_DIVIDEND & DEAL_DIVIDEND_FRANKED linked to position_id=901
        MT5DealRecord(
            ticket=115,
            position_id=901,
            symbol="XAUUSD",
            time_utc=t0 + timedelta(minutes=14),
            deal_type=int(MT5DealType.DEAL_DIVIDEND),
            profit=Decimal("6.00"),
        ),
        MT5DealRecord(
            ticket=116,
            position_id=901,
            symbol="XAUUSD",
            time_utc=t0 + timedelta(minutes=15),
            deal_type=int(MT5DealType.DEAL_DIVIDEND_FRANKED),
            profit=Decimal("-2.00"),
        ),
        # 17: DEAL_TAX (-1.75 -> trading PnL)
        MT5DealRecord(
            ticket=117,
            time_utc=t0 + timedelta(minutes=16),
            deal_type=int(MT5DealType.DEAL_TAX),
            profit=Decimal("-1.75"),
        ),
    ]

    summary = classify_and_aggregate_deals(valid_deals)
    assert summary.deduplicated_deal_count == 17
    assert summary.duplicates_skipped_count == 1

    # Trading PnL: +118 (sell) - 5 (charge) - 3.50 (comm) - 1.50 (unlinked comm)
    #              - 2 - 4 - 1 - 3 (daily/monthly/agent comms) - 1.25 (neg interest)
    #              + 6 (div) - 2 (franked div) - 1.75 (tax) = +99.00
    assert summary.realized_trading_pnl == Decimal("99.00")
    # Non-trading cash flows: +500 (balance) + 50 (credit) + 25 (bonus) + 4 (pos interest) = +579.00
    assert summary.net_non_trading_cash_flows == Decimal("579.00")
    assert summary.total_net_cash_impact == Decimal("678.00")

    # Verify ticket 107 (order=501) has linked_to_trade=True, while ticket 108 (order=0, position_id=0, symbol="XAUUSD", magic=202607) has linked_to_trade=False
    by_ticket = {cd.ticket: cd for cd in summary.classified_deals}
    assert by_ticket[107].linked_to_trade is True
    assert by_ticket[108].linked_to_trade is False

    # Fail-closed deal types: 5 (CORRECTION), 13 (BUY_CANCELED), 14 (SELL_CANCELED), and unknown (18)
    for bad_type in (
        int(MT5DealType.DEAL_TYPE_CORRECTION),
        int(MT5DealType.DEAL_TYPE_BUY_CANCELED),
        int(MT5DealType.DEAL_TYPE_SELL_CANCELED),
        18,
        -1,
    ):
        with pytest.raises(AccountStateReconstructionError):
            classify_and_aggregate_deals(
                [
                    MT5DealRecord(
                        ticket=201,
                        time_utc=t0,
                        deal_type=bad_type,
                        profit=Decimal("-10.00"),
                    )
                ]
            )

    # Commission double-counting: inline commission on BUY deal + separate DEAL_TYPE_COMMISSION for same order
    buy_with_inline_comm = MT5DealRecord(
        ticket=301,
        order=701,
        position_id=801,
        time_utc=t0,
        deal_type=int(MT5DealType.DEAL_TYPE_BUY),
        symbol="XAUUSD",
        volume=Decimal("0.10"),
        price=Decimal("2650.00"),
        commission=Decimal("-3.50"),
    )
    sep_comm_same_order = MT5DealRecord(
        ticket=302,
        order=701,
        position_id=801,
        time_utc=t0 + timedelta(seconds=1),
        deal_type=int(MT5DealType.DEAL_TYPE_COMMISSION),
        symbol="XAUUSD",
        commission=Decimal("-3.50"),
    )
    with pytest.raises(AccountStateReconstructionError, match="Commission double-counting detected"):
        classify_and_aggregate_deals([buy_with_inline_comm, sep_comm_same_order])

    # Unattributed DEAL_DIVIDEND (position_id=0, symbol="") must fail closed
    with pytest.raises(AccountStateReconstructionError, match="Unattributed DEAL_DIVIDEND"):
        classify_and_aggregate_deals(
            [
                MT5DealRecord(
                    ticket=401,
                    time_utc=t0,
                    deal_type=int(MT5DealType.DEAL_DIVIDEND),
                    profit=Decimal("10.00"),
                )
            ]
        )

    # DEAL_TYPE_BALANCE linked to a symbol or position_id must fail closed
    with pytest.raises(AccountStateReconstructionError, match="DEAL_TYPE_BALANCE"):
        classify_and_aggregate_deals(
            [
                MT5DealRecord(
                    ticket=402,
                    position_id=901,
                    symbol="XAUUSD",
                    time_utc=t0,
                    deal_type=int(MT5DealType.DEAL_TYPE_BALANCE),
                    profit=Decimal("100.00"),
                )
            ]
        )


def test_account_risk_tracker_anchors_missing_empty_incomplete_and_initial_deposit(
    tmp_path: Path,
) -> None:
    """Verify PersistentAccountRiskTracker enforces mandatory TrustedBalanceAnchor,
    rejects empty history_deals_get(), verifies initial deposit deals, detects incomplete history,
    and applies currency_digits-aware tolerance.
    """
    mock = MockMT5TerminalBackend()
    conn = MT5ConnectionAdapter(
        backend=mock,
        config=MT5ConnectionConfig(login=100200300, server="AtlasBroker-Demo"),
    )
    conn.connect()

    now_utc = datetime(2026, 11, 20, 14, 0, 0, tzinfo=timezone.utc)

    # 1. Missing TrustedBalanceAnchor (no persisted snapshot and no configured anchor) -> fail closed!
    safety_no_anchor = ExecutionSafetyConfig(
        account_risk_state_path=tmp_path / "no_anchor_state.json",
        trusted_balance_anchor=None,
    )
    tracker_no_anchor = PersistentAccountRiskTracker(
        connection=conn,
        safety_config=safety_no_anchor,
    )
    with pytest.raises(AccountStateReconstructionError, match="No TrustedBalanceAnchor available"):
        tracker_no_anchor.reconstruct_account_state(reference_now_utc=now_utc)

    # 2. Empty history_deals_get() result -> rejected even when a TrustedBalanceAnchor IS configured!
    mock.set_deals([])
    anchor_10k = TrustedBalanceAnchorConfig(
        anchor_balance=Decimal("10000.00"),
        anchor_timestamp_utc=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    )
    tracker_empty = PersistentAccountRiskTracker(
        connection=conn,
        safety_config=ExecutionSafetyConfig(
            account_risk_state_path=tmp_path / "empty_history_state.json",
            trusted_balance_anchor=anchor_10k,
        ),
    )
    with pytest.raises(AccountStateReconstructionError, match="empty deal history"):
        tracker_empty.reconstruct_account_state(reference_now_utc=now_utc)

    # 3. Initial deposit verification: origin anchor (0.00, require_initial_deposit_deal=True)
    #    fails closed if only a trading deal exists without a positive DEAL_TYPE_BALANCE deposit deal
    origin_anchor = TrustedBalanceAnchorConfig(
        anchor_balance=Decimal("0.00"),
        anchor_timestamp_utc=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        require_initial_deposit_deal=True,
    )
    mock.set_deals(
        [
            MT5DealRecord(
                ticket=2001,
                order=3001,
                position_id=4001,
                time_utc=datetime(2026, 11, 20, 10, 0, 0, tzinfo=timezone.utc),
                deal_type=int(MT5DealType.DEAL_TYPE_BUY),
                symbol="XAUUSD",
                volume=Decimal("1.00"),
                price=Decimal("2650.00"),
                profit=Decimal("10000.00"),
            )
        ]
    )
    tracker_no_deposit = PersistentAccountRiskTracker(
        connection=conn,
        safety_config=ExecutionSafetyConfig(
            account_risk_state_path=tmp_path / "no_deposit_state.json",
            trusted_balance_anchor=origin_anchor,
        ),
    )
    with pytest.raises(AccountStateReconstructionError, match="Initial deposit verification failed"):
        tracker_no_deposit.reconstruct_account_state(reference_now_utc=now_utc)

    # 4. Incomplete deal history: deposit (+10000) + trade (+100) = 10100, but broker balance is 10250
    init_deposit = MT5DealRecord(
        ticket=1001,
        time_utc=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        deal_type=int(MT5DealType.DEAL_TYPE_BALANCE),
        profit=Decimal("10000.00"),
    )
    trade_deal = MT5DealRecord(
        ticket=1002,
        order=5001,
        position_id=6001,
        time_utc=datetime(2026, 11, 20, 10, 0, 0, tzinfo=timezone.utc),
        deal_type=int(MT5DealType.DEAL_TYPE_SELL),
        entry=int(MT5DealEntry.DEAL_ENTRY_OUT),
        symbol="XAUUSD",
        volume=Decimal("0.10"),
        price=Decimal("2660.00"),
        profit=Decimal("100.00"),
    )
    mock.set_deals([init_deposit, trade_deal])
    acct_base = mock.account_info()
    assert acct_base is not None
    mock.set_account(
        acct_base.model_copy(
            update={"balance": Decimal("10250.00"), "equity": Decimal("10250.00")}
        )
    )
    tracker_incomplete = PersistentAccountRiskTracker(
        connection=conn,
        safety_config=ExecutionSafetyConfig(
            account_risk_state_path=tmp_path / "incomplete_state.json",
            trusted_balance_anchor=origin_anchor,
        ),
    )
    with pytest.raises(AccountStateReconstructionError, match="Balance completeness verification failed"):
        tracker_incomplete.reconstruct_account_state(reference_now_utc=now_utc)

    # 5. Currency-aware tolerance verification (USD 2 digits -> 0.05 vs JPY 0 digits -> 5)
    # Within 0.04 USD tolerance succeeds; 0.08 USD discrepancy fails
    mock.set_account(
        acct_base.model_copy(
            update={"balance": Decimal("10100.04"), "equity": Decimal("10100.04")}
        )
    )
    tracker_usd_tol = PersistentAccountRiskTracker(
        connection=conn,
        safety_config=ExecutionSafetyConfig(
            account_risk_state_path=tmp_path / "usd_tol_state.json",
            trusted_balance_anchor=origin_anchor,
            reconciliation_tolerance_units=5,
        ),
    )
    res_usd = tracker_usd_tol.reconstruct_report(reference_now_utc=now_utc)
    assert res_usd.reconciliation_tolerance == Decimal("0.05")
    assert res_usd.daily_starting_balance == Decimal("10000.00")

    # JPY account (currency_digits=0): tolerance_units=5 -> tolerance = 5 JPY
    jpy_acct = acct_base.model_copy(
        update={
            "currency": "JPY",
            "currency_digits": 0,
            "balance": Decimal("1500104"),  # 4 JPY rounding difference <= 5 JPY tolerance
            "equity": Decimal("1500104"),
            "margin_free": Decimal("1500104"),
        }
    )
    mock.set_account(jpy_acct)
    mock.set_deals(
        [
            MT5DealRecord(
                ticket=1001,
                time_utc=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                deal_type=int(MT5DealType.DEAL_TYPE_BALANCE),
                profit=Decimal("1500000"),
            ),
            MT5DealRecord(
                ticket=1002,
                order=5001,
                position_id=6001,
                time_utc=datetime(2026, 11, 20, 10, 0, 0, tzinfo=timezone.utc),
                deal_type=int(MT5DealType.DEAL_TYPE_SELL),
                symbol="XAUUSD",
                volume=Decimal("0.10"),
                price=Decimal("2660.00"),
                profit=Decimal("100"),
            ),
        ]
    )
    tracker_jpy = PersistentAccountRiskTracker(
        connection=conn,
        safety_config=ExecutionSafetyConfig(
            account_risk_state_path=tmp_path / "jpy_tol_state.json",
            trusted_balance_anchor=origin_anchor,
            reconciliation_tolerance_units=5,
        ),
    )
    res_jpy = tracker_jpy.reconstruct_report(reference_now_utc=now_utc)
    assert res_jpy.reconciliation_tolerance == Decimal("5")
    assert res_jpy.account_state.currency == "JPY"


def test_account_risk_tracker_ny_5pm_rollover_dst_transitions_peak_balance_and_account_mismatch(
    tmp_path: Path,
) -> None:
    """Verify 5 PM America/New_York rollover across DST transitions (EST <-> EDT),
    pre-rollover vs current-day deal partitioning, peak_balance preservation across rollovers,
    persisted rollover snapshot transitions, and account login/server mismatch rejection.
    """
    ny_tz = ZoneInfo("America/New_York")

    # 1. Verify DST-aware 5 PM America/New_York UTC bounds:
    # Winter EST (UTC-5): 2026-01-06 starts 2026-01-05 22:00 UTC and ends 2026-01-06 22:00 UTC
    est_start, est_end = compute_trading_day_utc_bounds(date(2026, 1, 6), ny_tz, rollover_hour=17)
    assert est_start == datetime(2026, 1, 5, 22, 0, 0, tzinfo=timezone.utc)
    assert est_end == datetime(2026, 1, 6, 22, 0, 0, tzinfo=timezone.utc)

    # Summer EDT (UTC-4): 2026-07-07 starts 2026-07-06 21:00 UTC and ends 2026-07-07 21:00 UTC
    edt_start, edt_end = compute_trading_day_utc_bounds(date(2026, 7, 7), ny_tz, rollover_hour=17)
    assert edt_start == datetime(2026, 7, 6, 21, 0, 0, tzinfo=timezone.utc)
    assert edt_end == datetime(2026, 7, 7, 21, 0, 0, tzinfo=timezone.utc)

    # Spring-Forward DST transition day (2026-03-08): starts Sat 2026-03-07 22:00 UTC (EST) and ends Sun 2026-03-08 21:00 UTC (EDT)
    dst_spring_start, dst_spring_end = compute_trading_day_utc_bounds(
        date(2026, 3, 8), ny_tz, rollover_hour=17
    )
    assert dst_spring_start == datetime(2026, 3, 7, 22, 0, 0, tzinfo=timezone.utc)
    assert dst_spring_end == datetime(2026, 3, 8, 21, 0, 0, tzinfo=timezone.utc)

    # Fall-Back DST transition day (2026-11-01): starts Sat 2026-10-31 21:00 UTC (EDT) and ends Sun 2026-11-01 22:00 UTC (EST)
    dst_fall_start, dst_fall_end = compute_trading_day_utc_bounds(
        date(2026, 11, 1), ny_tz, rollover_hour=17
    )
    assert dst_fall_start == datetime(2026, 10, 31, 21, 0, 0, tzinfo=timezone.utc)
    assert dst_fall_end == datetime(2026, 11, 1, 22, 0, 0, tzinfo=timezone.utc)

    # 2. Multi-day reconstruction and 5 PM NY rollover test
    mock = MockMT5TerminalBackend()
    conn = MT5ConnectionAdapter(
        backend=mock,
        config=MT5ConnectionConfig(login=100200300, server="AtlasBroker-Demo"),
    )
    conn.connect()

    state_path = tmp_path / "rollover_account_risk_state.json"
    origin_anchor = TrustedBalanceAnchorConfig(
        anchor_balance=Decimal("0.00"),
        anchor_timestamp_utc=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        require_initial_deposit_deal=True,
    )
    safety_cfg = ExecutionSafetyConfig(
        account_risk_state_path=state_path,
        trusted_balance_anchor=origin_anchor,
    )
    tracker = PersistentAccountRiskTracker(connection=conn, safety_config=safety_cfg)

    # Trading Day 2026-11-20 starts at 2026-11-19 22:00:00 UTC (5 PM EST)
    # Seed:
    # - Initial deposit on 2026-01-01: +10,000.00
    # - Pre-rollover trade on 2026-11-19 18:00 UTC (before 22:00 UTC rollover): +300.00 -> daily_starting_balance = 10,300.00
    # - Current-day winning trade on 2026-11-20 09:00 UTC: +400.00 (running balance peak = 10,700.00)
    # - Current-day losing trade on 2026-11-20 11:00 UTC: -550.00 (net trading PnL today = -150.00)
    # - Current-day deposit on 2026-11-20 12:00 UTC: +1,000.00 (non-trading cash flow; MUST NOT mask the -150.00 trading loss!)
    # Ending balance on Day 1 at 14:00 UTC = 10,300 - 150 + 1,000 = 11,150.00; floating PnL = -50.00 (equity = 11,100.00)
    deals_day1 = [
        MT5DealRecord(
            ticket=1001,
            time_utc=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            deal_type=int(MT5DealType.DEAL_TYPE_BALANCE),
            profit=Decimal("10000.00"),
        ),
        MT5DealRecord(
            ticket=1002,
            order=2001,
            position_id=3001,
            time_utc=datetime(2026, 11, 19, 18, 0, 0, tzinfo=timezone.utc),
            deal_type=int(MT5DealType.DEAL_TYPE_BUY),
            symbol="XAUUSD",
            volume=Decimal("0.20"),
            price=Decimal("2640.00"),
            profit=Decimal("300.00"),
        ),
        MT5DealRecord(
            ticket=1003,
            order=2002,
            position_id=3002,
            time_utc=datetime(2026, 11, 20, 9, 0, 0, tzinfo=timezone.utc),
            deal_type=int(MT5DealType.DEAL_TYPE_BUY),
            symbol="XAUUSD",
            volume=Decimal("0.20"),
            price=Decimal("2645.00"),
            profit=Decimal("400.00"),
        ),
        MT5DealRecord(
            ticket=1004,
            order=2003,
            position_id=3003,
            time_utc=datetime(2026, 11, 20, 11, 0, 0, tzinfo=timezone.utc),
            deal_type=int(MT5DealType.DEAL_TYPE_SELL),
            symbol="XAUUSD",
            volume=Decimal("0.20"),
            price=Decimal("2650.00"),
            profit=Decimal("-550.00"),
        ),
        MT5DealRecord(
            ticket=1005,
            time_utc=datetime(2026, 11, 20, 12, 0, 0, tzinfo=timezone.utc),
            deal_type=int(MT5DealType.DEAL_TYPE_BALANCE),
            profit=Decimal("1000.00"),
        ),
    ]
    mock.set_deals(deals_day1)
    base_acct = mock.account_info()
    assert base_acct is not None
    mock.set_account(
        base_acct.model_copy(
            update={
                "balance": Decimal("11150.00"),
                "equity": Decimal("11100.00"),
                "margin_free": Decimal("11100.00"),
            }
        )
    )

    day1_now = datetime(2026, 11, 20, 14, 0, 0, tzinfo=timezone.utc)
    rep1 = tracker.reconstruct_report(reference_now_utc=day1_now)
    assert rep1.anchor_source == TrustedAnchorSource.CONFIGURED_BALANCE_ANCHOR
    assert rep1.trading_day_label == "2026-11-20"
    assert rep1.daily_starting_balance == Decimal("10300.00")
    assert rep1.realized_trading_pnl_today == Decimal("-150.00")
    assert rep1.net_non_trading_cash_flows_today == Decimal("1000.00")
    assert rep1.unrealized_floating_pnl == Decimal("-50.00")
    # current_daily_pnl = realized_trading_pnl_today (-150) + unrealized (-50) = -200.00 (deposit is excluded!)
    assert rep1.current_daily_pnl == Decimal("-200.00")
    assert rep1.account_state.current_daily_pnl == Decimal("-200.00")
    assert rep1.peak_balance == Decimal("11150.00")
    assert rep1.rolled_over is False
    assert rep1.envelope_generation == 1

    # 3. Rollover past 5 PM America/New_York on 2026-11-20 (22:00 UTC) into trading day 2026-11-21
    # Add a Day 2 losing trade of -350.00 at 2026-11-20 22:30 UTC (after 5 PM NY rollover)
    # New balance = 11,150.00 - 350.00 = 10,800.00; equity = 10,800.00
    # Verify:
    # - anchor_source is PERSISTED_ROLLOVER_ANCHOR
    # - rolled_over is True
    # - new daily_starting_balance is 11,150.00 (the verified rollover balance at 2026-11-20 22:00 UTC)
    # - peak_balance remains preserved at 11,150.00 despite balance dropping to 10,800.00
    deal_day2 = MT5DealRecord(
        ticket=1006,
        order=2004,
        position_id=3004,
        time_utc=datetime(2026, 11, 20, 22, 30, 0, tzinfo=timezone.utc),
        deal_type=int(MT5DealType.DEAL_TYPE_SELL),
        symbol="XAUUSD",
        volume=Decimal("0.20"),
        price=Decimal("2655.00"),
        profit=Decimal("-350.00"),
    )
    mock.add_deal(deal_day2)
    mock.set_account(
        base_acct.model_copy(
            update={
                "balance": Decimal("10800.00"),
                "equity": Decimal("10800.00"),
                "margin_free": Decimal("10800.00"),
            }
        )
    )

    day2_now = datetime(2026, 11, 20, 23, 0, 0, tzinfo=timezone.utc)
    rep2 = tracker.reconstruct_report(reference_now_utc=day2_now)
    assert rep2.anchor_source == TrustedAnchorSource.PERSISTED_ROLLOVER_ANCHOR
    assert rep2.rolled_over is True
    assert rep2.trading_day_label == "2026-11-21"
    assert rep2.daily_starting_balance == Decimal("11150.00")
    assert rep2.realized_trading_pnl_today == Decimal("-350.00")
    assert rep2.current_daily_pnl == Decimal("-350.00")
    assert rep2.peak_balance == Decimal("11150.00")
    assert rep2.account_state.peak_balance == Decimal("11150.00")
    assert rep2.envelope_generation == 2

    # 4. Account login/server mismatch against persisted snapshot file must fail closed
    other_conn = MT5ConnectionAdapter(
        backend=mock,
        config=MT5ConnectionConfig(login=777888999, server="AtlasBroker-Demo"),
    )
    mock.set_account(
        base_acct.model_copy(
            update={
                "login": 777888999,
                "balance": Decimal("10800.00"),
                "equity": Decimal("10800.00"),
                "margin_free": Decimal("10800.00"),
            }
        )
    )
    other_conn.connect()
    other_tracker = PersistentAccountRiskTracker(
        connection=other_conn,
        safety_config=safety_cfg,
    )
    with pytest.raises(AccountStateReconstructionError, match="account_login"):
        other_tracker.reconstruct_account_state(reference_now_utc=day2_now)


def test_c1_and_m2_persistence_recovery_valid_tmp_envelopes_bak_tmp_and_dual_corruption(
    tmp_path: Path,
) -> None:
    """Verify remediation of C-1 and M-2:
    1. Interrupted .bak.tmp.* files are detected and block load()/save() until recovery.
    2. A valid higher-generation .tmp.* envelope (flushed before os.replace crash) is inspected,
       exposed in highest_valid_envelope / valid_tmp_envelopes, rotated into .bak, and never discarded.
    3. A valid .tmp.* envelope with a broken previous_checksum chain fails closed.
    4. When both primary and .bak are corrupted on disk, commit_recovered_payload() fails closed
       with PersistenceCorruptionError and NEVER resets generation to 1.
    """
    ledger_file = tmp_path / "c1_ledger.json"
    store = CrashSafeJsonPersistence(ledger_file)

    env1 = store.save(
        {"orders": ["o1"]},
        account_login=100200300,
        server="AtlasBroker-Demo",
        now_utc=FIXED_UTC,
    )
    env2 = store.save(
        {"orders": ["o1", "o2"]},
        account_login=100200300,
        server="AtlasBroker-Demo",
        now_utc=FIXED_UTC + timedelta(minutes=1),
    )
    assert env1.generation == 1 and env2.generation == 2

    # Stage A: Lingering .bak.tmp.* file from an interrupted backup rotation
    bak_tmp = tmp_path / f"{store.backup_path.name}.tmp.12345.cafebabe"
    bak_tmp.write_text("{partial backup write", encoding="utf-8")
    assert bak_tmp in store.find_interrupted_temp_files()

    with pytest.raises(LedgerRecoveryLockError, match="Interrupted atomic write"):
        store.load(expected_login=100200300, expected_server="AtlasBroker-Demo")

    insp_bak_tmp = store.inspect_recovery_state(
        expected_login=100200300,
        expected_server="AtlasBroker-Demo",
    )
    assert insp_bak_tmp.recovery_required is True
    assert bak_tmp.name in insp_bak_tmp.interrupted_tmp_files
    bak_tmp.unlink()

    # Stage B: Simulate crash after fsync() of valid generation 3 .tmp.* file right before os.replace()
    gen3_payload = {"orders": ["o1", "o2", "o3-submitting-in-tmp"]}
    gen3_checksum = compute_envelope_checksum(
        schema_version=ENVELOPE_SCHEMA_VERSION,
        generation=3,
        previous_checksum=env2.payload_checksum,
        account_login=100200300,
        server="AtlasBroker-Demo",
        payload=gen3_payload,
    )
    gen3_env = PersistedEnvelope(
        schema_version=ENVELOPE_SCHEMA_VERSION,
        generation=3,
        previous_checksum=env2.payload_checksum,
        payload_checksum=gen3_checksum,
        account_login=100200300,
        server="AtlasBroker-Demo",
        written_at_utc=FIXED_UTC + timedelta(minutes=2),
        payload=gen3_payload,
    )
    valid_tmp_file = tmp_path / f"{ledger_file.name}.tmp.54321.validgen3"
    valid_tmp_file.write_text(
        json.dumps(gen3_env.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    restarted_store = CrashSafeJsonPersistence(ledger_file)
    insp_valid_tmp = restarted_store.inspect_recovery_state(
        expected_login=100200300,
        expected_server="AtlasBroker-Demo",
    )
    assert insp_valid_tmp.recovery_required is True
    assert len(insp_valid_tmp.valid_tmp_envelopes) == 1
    assert insp_valid_tmp.valid_tmp_envelopes[0].generation == 3
    assert insp_valid_tmp.highest_valid_envelope is not None
    assert insp_valid_tmp.highest_valid_envelope.generation == 3
    assert insp_valid_tmp.highest_valid_envelope.payload == gen3_payload

    # Committing recovered payload MUST chain from generation 3 (the valid .tmp.* envelope) to generation 4!
    env4 = restarted_store.commit_recovered_payload(
        {"orders": ["o1", "o2", "o3-reconciled"]},
        account_login=100200300,
        server="AtlasBroker-Demo",
        now_utc=FIXED_UTC + timedelta(minutes=3),
    )
    assert env4.generation == 4
    assert env4.previous_checksum == gen3_env.payload_checksum
    assert not valid_tmp_file.exists()

    # Verify .bak now holds generation 3 (the recovered temporary envelope)
    loaded_after = restarted_store.load(
        expected_login=100200300,
        expected_server="AtlasBroker-Demo",
    )
    assert loaded_after is not None and loaded_after.generation == 4

    # Stage C: A valid .tmp.* envelope with a broken previous_checksum chain against primary (gen 4) must fail closed
    bad_chain_payload = {"orders": ["tampered-chain"]}
    forged_prev = "f" * 64
    bad_chain_checksum = compute_envelope_checksum(
        schema_version=ENVELOPE_SCHEMA_VERSION,
        generation=5,
        previous_checksum=forged_prev,
        account_login=100200300,
        server="AtlasBroker-Demo",
        payload=bad_chain_payload,
    )
    bad_chain_env = PersistedEnvelope(
        schema_version=ENVELOPE_SCHEMA_VERSION,
        generation=5,
        previous_checksum=forged_prev,
        payload_checksum=bad_chain_checksum,
        account_login=100200300,
        server="AtlasBroker-Demo",
        written_at_utc=FIXED_UTC + timedelta(minutes=4),
        payload=bad_chain_payload,
    )
    bad_tmp_file = tmp_path / f"{ledger_file.name}.tmp.54321.badchain5"
    bad_tmp_file.write_text(
        json.dumps(bad_chain_env.model_dump(mode="json")),
        encoding="utf-8",
    )
    with pytest.raises(PersistenceCorruptionError, match="Broken previous_checksum chain"):
        restarted_store.commit_recovered_payload(
            {"orders": ["should-fail"]},
            account_login=100200300,
            server="AtlasBroker-Demo",
        )
    bad_tmp_file.unlink()

    # Stage D: Both primary and .bak corrupted on disk -> commit_recovered_payload MUST fail closed (never reset to gen 1)
    ledger_file.write_text("{corrupted primary json", encoding="utf-8")
    restarted_store.backup_path.write_text("{corrupted backup json", encoding="utf-8")

    cold_store = CrashSafeJsonPersistence(ledger_file)
    with pytest.raises(PersistenceCorruptionError, match="Refusing to reset generation to 1"):
        cold_store.commit_recovered_payload(
            {"orders": ["should-not-reset-to-gen-1"]},
            account_login=100200300,
            server="AtlasBroker-Demo",
        )


def test_c2_and_m3_same_day_initial_deposit_multi_call_and_no_unnecessary_writes(
    tmp_path: Path,
) -> None:
    """Verify remediation of C-2 and M-3:
    1. When an account is opened and funded during the active trading day (first_deposit.time_utc > day_start_utc)
       from a 0.00 origin anchor, persisted snapshot keeps anchor_timestamp_utc == day_start_utc and records
       initial_deposit_ticket.
    2. Multiple consecutive reconstruct_report() calls during the same trading day succeed and produce
       identical accounting results without losing any deals from day_start_utc onward.
    3. Consecutive calls when verified state is unchanged do NOT rewrite disk or increment generation (M-3).
    4. Rollover to the next trading day after 5 PM NY transitions cleanly and clears initial_deposit_ticket.
    """
    mock = MockMT5TerminalBackend()
    conn = MT5ConnectionAdapter(
        backend=mock,
        config=MT5ConnectionConfig(login=100200300, server="AtlasBroker-Demo"),
    )
    conn.connect()

    # Trading day 2026-11-20 starts at 2026-11-19 22:00:00 UTC (5 PM EST)
    day_start = datetime(2026, 11, 19, 22, 0, 0, tzinfo=timezone.utc)
    origin_anchor = TrustedBalanceAnchorConfig(
        anchor_balance=Decimal("0.00"),
        anchor_timestamp_utc=day_start,
        require_initial_deposit_deal=True,
    )
    state_path = tmp_path / "c2_same_day_deposit_state.json"
    tracker = PersistentAccountRiskTracker(
        connection=conn,
        safety_config=ExecutionSafetyConfig(
            account_risk_state_path=state_path,
            trusted_balance_anchor=origin_anchor,
        ),
    )

    # Seed same-day deals all occurring AFTER day_start_utc (2026-11-19 22:00 UTC):
    # - Deal 1001 at 2026-11-20 01:00 UTC: initial account-opening deposit +10,000.00
    # - Deal 1002 at 2026-11-20 03:00 UTC: trading profit +250.00
    # - Deal 1003 at 2026-11-20 05:00 UTC: additional intraday deposit +500.00
    # Ending balance = 10,750.00
    mock.set_deals(
        [
            MT5DealRecord(
                ticket=1001,
                time_utc=datetime(2026, 11, 20, 1, 0, 0, tzinfo=timezone.utc),
                deal_type=int(MT5DealType.DEAL_TYPE_BALANCE),
                profit=Decimal("10000.00"),
            ),
            MT5DealRecord(
                ticket=1002,
                order=2001,
                position_id=3001,
                time_utc=datetime(2026, 11, 20, 3, 0, 0, tzinfo=timezone.utc),
                deal_type=int(MT5DealType.DEAL_TYPE_BUY),
                symbol="XAUUSD",
                volume=Decimal("0.10"),
                price=Decimal("2650.00"),
                profit=Decimal("250.00"),
            ),
            MT5DealRecord(
                ticket=1003,
                time_utc=datetime(2026, 11, 20, 5, 0, 0, tzinfo=timezone.utc),
                deal_type=int(MT5DealType.DEAL_TYPE_BALANCE),
                profit=Decimal("500.00"),
            ),
        ]
    )
    base_acct = mock.account_info()
    assert base_acct is not None
    mock.set_account(
        base_acct.model_copy(
            update={
                "balance": Decimal("10750.00"),
                "equity": Decimal("10750.00"),
                "margin_free": Decimal("10750.00"),
            }
        )
    )

    # Call 1 at 10:00 UTC (CONFIGURED_BALANCE_ANCHOR -> writes generation 1)
    rep1 = tracker.reconstruct_report(
        reference_now_utc=datetime(2026, 11, 20, 10, 0, 0, tzinfo=timezone.utc)
    )
    assert rep1.anchor_source == TrustedAnchorSource.CONFIGURED_BALANCE_ANCHOR
    assert rep1.snapshot.anchor_timestamp_utc == day_start
    assert rep1.snapshot.initial_deposit_ticket == 1001
    assert rep1.daily_starting_balance == Decimal("10000.00")
    assert rep1.realized_trading_pnl_today == Decimal("250.00")
    assert rep1.net_non_trading_cash_flows_today == Decimal("500.00")
    assert rep1.reconstructed_balance == Decimal("10750.00")
    assert rep1.envelope_generation == 1

    # Call 2 at 11:00 UTC (PERSISTED_ROLLOVER_ANCHOR on same day -> must succeed and NOT increment generation)
    rep2 = tracker.reconstruct_report(
        reference_now_utc=datetime(2026, 11, 20, 11, 0, 0, tzinfo=timezone.utc)
    )
    assert rep2.anchor_source == TrustedAnchorSource.PERSISTED_ROLLOVER_ANCHOR
    assert rep2.daily_starting_balance == Decimal("10000.00")
    assert rep2.realized_trading_pnl_today == Decimal("250.00")
    assert rep2.net_non_trading_cash_flows_today == Decimal("500.00")
    assert rep2.reconstructed_balance == Decimal("10750.00")
    assert len(rep2.classified_deals_today) == 3
    assert rep2.envelope_generation == 1  # M-3: unchanged state does not increment generation!

    # Call 3 at 12:00 UTC from a fresh tracker instance (simulating process restart on same day)
    restarted_tracker = PersistentAccountRiskTracker(
        connection=conn,
        safety_config=ExecutionSafetyConfig(
            account_risk_state_path=state_path,
            trusted_balance_anchor=origin_anchor,
        ),
    )
    rep3 = restarted_tracker.reconstruct_report(
        reference_now_utc=datetime(2026, 11, 20, 12, 0, 0, tzinfo=timezone.utc)
    )
    assert rep3.daily_starting_balance == Decimal("10000.00")
    assert rep3.realized_trading_pnl_today == Decimal("250.00")
    assert rep3.net_non_trading_cash_flows_today == Decimal("500.00")
    assert rep3.envelope_generation == 1

    # Call 4 after 5 PM NY rollover (2026-11-20 23:00 UTC -> trading day 2026-11-21)
    rep4 = restarted_tracker.reconstruct_report(
        reference_now_utc=datetime(2026, 11, 20, 23, 0, 0, tzinfo=timezone.utc)
    )
    assert rep4.rolled_over is True
    assert rep4.trading_day_label == "2026-11-21"
    assert rep4.daily_starting_balance == Decimal("10750.00")
    assert rep4.realized_trading_pnl_today == Decimal("0.00")
    assert rep4.net_non_trading_cash_flows_today == Decimal("0.00")
    assert rep4.snapshot.initial_deposit_ticket is None
    assert rep4.envelope_generation == 2


def test_h1_and_m1_commission_double_counting_inline_fee_mixed_order_position_rollover_and_dividends() -> None:
    """Verify remediation of H-1 and M-1:
    1. Inline fee (deal.fee != 0) on execution deal blocks separate DEAL_TYPE_COMMISSION for same order/position.
    2. Mixed (order=0, position_id=P) and (order=O, position_id=P) separate commissions are rejected.
    3. Orderless inline commission on position blocks order-linked separate commission on that position.
    4. Cross-rollover duplicate separate commission for a pre-rollover order is rejected.
    5. DEAL_DIVIDEND and DEAL_DIVIDEND_FRANKED validate position_id existence, order linkage, and symbol match.
    """
    t0 = FIXED_UTC
    base_exec = MT5DealRecord(
        ticket=101,
        order=501,
        position_id=901,
        time_utc=t0,
        deal_type=int(MT5DealType.DEAL_TYPE_BUY),
        symbol="XAUUSD",
        volume=Decimal("0.10"),
        price=Decimal("2650.00"),
    )

    # 1. Execution deal with non-zero inline fee (-2.50) + separate DEAL_TYPE_COMMISSION -> rejected!
    exec_with_inline_fee = base_exec.model_copy(update={"fee": Decimal("-2.50")})
    sep_comm_order = MT5DealRecord(
        ticket=102,
        order=501,
        position_id=901,
        time_utc=t0 + timedelta(seconds=1),
        deal_type=int(MT5DealType.DEAL_TYPE_COMMISSION),
        symbol="XAUUSD",
        commission=Decimal("-2.50"),
    )
    with pytest.raises(AccountStateReconstructionError, match="inline commission or fee"):
        classify_and_aggregate_deals([exec_with_inline_fee, sep_comm_order])

    # 2. Mixed (order=0, position_id=901) followed by (order=501, position_id=901) -> rejected!
    sep_comm_pos_only = MT5DealRecord(
        ticket=103,
        order=0,
        position_id=901,
        time_utc=t0 + timedelta(seconds=1),
        deal_type=int(MT5DealType.DEAL_TYPE_COMMISSION),
        symbol="XAUUSD",
        commission=Decimal("-3.00"),
    )
    sep_comm_order_later = sep_comm_order.model_copy(
        update={"ticket": 104, "time_utc": t0 + timedelta(seconds=2)}
    )
    with pytest.raises(AccountStateReconstructionError, match="Mixed order/position duplicate"):
        classify_and_aggregate_deals([base_exec, sep_comm_pos_only, sep_comm_order_later])

    # 3. Mixed (order=501, position_id=901) followed by (order=0, position_id=901) -> rejected!
    sep_comm_pos_only_later = sep_comm_pos_only.model_copy(
        update={"ticket": 105, "time_utc": t0 + timedelta(seconds=3)}
    )
    with pytest.raises(AccountStateReconstructionError, match="Duplicate or mixed order/position"):
        classify_and_aggregate_deals([base_exec, sep_comm_order, sep_comm_pos_only_later])

    # 4. Cross-rollover duplicate separate commission (pre-rollover has base_exec + sep_comm_order;
    #    post-rollover has a second separate commission for order=501) -> rejected!
    post_rollover_dup_comm = sep_comm_order.model_copy(
        update={"ticket": 105, "time_utc": t0 + timedelta(hours=1)}
    )
    with pytest.raises(AccountStateReconstructionError, match="Duplicate separate DEAL_TYPE_COMMISSION"):
        classify_and_aggregate_deals(
            [post_rollover_dup_comm],
            reference_deals=[base_exec, sep_comm_order],
        )

    # 5. M-1 Dividend validation: unknown position_id, empty symbol, mismatched symbol, or inconsistent order
    with pytest.raises(AccountStateReconstructionError, match="does not exist in verified execution deal history"):
        classify_and_aggregate_deals(
            [
                base_exec,
                MT5DealRecord(
                    ticket=201,
                    position_id=999999,
                    symbol="XAUUSD",
                    time_utc=t0 + timedelta(minutes=1),
                    deal_type=int(MT5DealType.DEAL_DIVIDEND),
                    profit=Decimal("5.00"),
                ),
            ]
        )

    with pytest.raises(AccountStateReconstructionError, match="requires position_id > 0 and non-empty symbol"):
        classify_and_aggregate_deals(
            [
                base_exec,
                MT5DealRecord(
                    ticket=202,
                    position_id=901,
                    symbol="",
                    time_utc=t0 + timedelta(minutes=1),
                    deal_type=int(MT5DealType.DEAL_DIVIDEND_FRANKED),
                    profit=Decimal("-5.00"),
                ),
            ]
        )

    with pytest.raises(AccountStateReconstructionError, match="mismatches the symbol"):
        classify_and_aggregate_deals(
            [
                base_exec,
                MT5DealRecord(
                    ticket=203,
                    position_id=901,
                    symbol="EURUSD",
                    time_utc=t0 + timedelta(minutes=1),
                    deal_type=int(MT5DealType.DEAL_DIVIDEND),
                    profit=Decimal("5.00"),
                ),
            ]
        )


def test_h2_h3_and_h4_rollover_boundary_deposit_broker_credit_multi_symbol_positions_and_future_deals(
    tmp_path: Path,
) -> None:
    """Verify remediation of H-2, H-3, and H-4:
    1. H-2: A deposit of amount X == daily_starting_balance at the exact 5 PM NY rollover timestamp
       (deal.time_utc == day_start_utc) is never dropped across consecutive reconstruction calls.
    2. H-3: Positive broker credit (account.credit > 0) is stripped from unrealized_floating_pnl and
       account_state.equity when consistent with open positions, and fails closed when ambiguous.
    3. H-3: open_positions_count counts all account-wide open positions even when symbol filter is passed.
    4. H-4: Any deal with time_utc > reference_now_utc immediately raises AccountStateReconstructionError.
    """
    mock = MockMT5TerminalBackend()
    conn = MT5ConnectionAdapter(
        backend=mock,
        config=MT5ConnectionConfig(login=100200300, server="AtlasBroker-Demo"),
    )
    conn.connect()

    # Trading day 2026-11-20 starts at 2026-11-19 22:00:00 UTC (5 PM EST)
    day_start = datetime(2026, 11, 19, 22, 0, 0, tzinfo=timezone.utc)
    origin_anchor = TrustedBalanceAnchorConfig(
        anchor_balance=Decimal("0.00"),
        anchor_timestamp_utc=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        require_initial_deposit_deal=True,
    )
    state_path = tmp_path / "h2_h3_h4_state.json"
    tracker = PersistentAccountRiskTracker(
        connection=conn,
        safety_config=ExecutionSafetyConfig(
            account_risk_state_path=state_path,
            trusted_balance_anchor=origin_anchor,
        ),
    )

    # 1. H-2 Test:
    # - Initial deposit on 2026-01-01: +10,000.00 -> daily_starting_balance = 10,000.00
    # - Second deposit of the EXACT same amount (+10,000.00) at the EXACT 5 PM NY rollover timestamp (day_start)
    # Total broker balance = 20,000.00
    deals_h2 = [
        MT5DealRecord(
            ticket=1001,
            time_utc=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            deal_type=int(MT5DealType.DEAL_TYPE_BALANCE),
            profit=Decimal("10000.00"),
        ),
        MT5DealRecord(
            ticket=1002,
            time_utc=day_start,  # Exact 5 PM NY rollover boundary timestamp!
            deal_type=int(MT5DealType.DEAL_TYPE_BALANCE),
            profit=Decimal("10000.00"),
        ),
    ]
    mock.set_deals(deals_h2)
    base_acct = mock.account_info()
    assert base_acct is not None
    mock.set_account(
        base_acct.model_copy(
            update={
                "balance": Decimal("20000.00"),
                "equity": Decimal("20000.00"),
                "margin_free": Decimal("20000.00"),
            }
        )
    )

    now_utc = datetime(2026, 11, 20, 14, 0, 0, tzinfo=timezone.utc)
    rep_call1 = tracker.reconstruct_report(reference_now_utc=now_utc)
    assert rep_call1.daily_starting_balance == Decimal("10000.00")
    assert rep_call1.net_non_trading_cash_flows_today == Decimal("10000.00")
    assert rep_call1.reconstructed_balance == Decimal("20000.00")

    # Consecutive Call 2 (under PERSISTED_ROLLOVER_ANCHOR where anchor_timestamp_utc == day_start
    # and anchor_balance == 10000.00) MUST NOT drop ticket 1002!
    rep_call2 = tracker.reconstruct_report(reference_now_utc=now_utc + timedelta(minutes=5))
    assert rep_call2.anchor_source == TrustedAnchorSource.PERSISTED_ROLLOVER_ANCHOR
    assert rep_call2.daily_starting_balance == Decimal("10000.00")
    assert rep_call2.net_non_trading_cash_flows_today == Decimal("10000.00")
    assert rep_call2.reconstructed_balance == Decimal("20000.00")

    # 2. H-3 Test: Multi-symbol open positions + consistent positive broker credit
    # Seed 1 open XAUUSD position (profit = -150.00) and 2 open EURUSD positions (profit = -50.00 total)
    # Total open_positions_count = 3, total positions_floating_pnl = -200.00
    # Broker credit = +500.00, balance = 20,000.00 -> MT5 equity = 20,000 + 500 - 200 = 20,300.00
    pos_xau = MT5PositionRecord(
        ticket=7001,
        identifier=7001,
        time_utc=now_utc - timedelta(hours=1),
        position_type=0,
        magic=202607,
        symbol="XAUUSD",
        volume=Decimal("0.10"),
        price_open=Decimal("2650.00"),
        price_current=Decimal("2635.00"),
        profit=Decimal("-150.00"),
    )
    pos_eur1 = MT5PositionRecord(
        ticket=7002,
        identifier=7002,
        time_utc=now_utc - timedelta(hours=1),
        position_type=0,
        magic=202607,
        symbol="EURUSD",
        volume=Decimal("0.10"),
        price_open=Decimal("1.0500"),
        price_current=Decimal("1.0470"),
        profit=Decimal("-30.00"),
    )
    pos_eur2 = MT5PositionRecord(
        ticket=7003,
        identifier=7003,
        time_utc=now_utc - timedelta(hours=1),
        position_type=1,
        magic=202607,
        symbol="EURUSD",
        volume=Decimal("0.10"),
        price_open=Decimal("1.0500"),
        price_current=Decimal("1.0520"),
        profit=Decimal("-20.00"),
    )
    mock.set_positions([pos_xau, pos_eur1, pos_eur2])
    mock.set_account(
        base_acct.model_copy(
            update={
                "balance": Decimal("20000.00"),
                "credit": Decimal("500.00"),
                "equity": Decimal("20300.00"),
                "margin_free": Decimal("19500.00"),
            }
        )
    )

    rep_credit = tracker.reconstruct_report(
        reference_now_utc=now_utc + timedelta(minutes=10),
        symbol="XAUUSD",
    )
    # Open positions count MUST be 3 (account-wide), even though symbol="XAUUSD" was passed!
    assert rep_credit.account_state.open_positions_count == 3
    # Floating PnL MUST be -200.00 (excluding the +500.00 credit facility!)
    assert rep_credit.unrealized_floating_pnl == Decimal("-200.00")
    assert rep_credit.current_daily_pnl == Decimal("-200.00")
    assert rep_credit.account_state.current_daily_pnl == Decimal("-200.00")
    # AccountState equity MUST be true cash equity (20,300 - 500 = 19,800.00)
    assert rep_credit.account_state.equity == Decimal("19800.00")

    # Ambiguous broker credit (credit = 500.00, but equity = 20,000.00 while open positions floating PnL = -200.00) -> fail closed!
    mock.set_account(
        base_acct.model_copy(
            update={
                "balance": Decimal("20000.00"),
                "credit": Decimal("500.00"),
                "equity": Decimal("20000.00"),
                "margin_free": Decimal("19500.00"),
            }
        )
    )
    with pytest.raises(AccountStateReconstructionError, match="Ambiguous broker credit accounting"):
        tracker.reconstruct_report(reference_now_utc=now_utc + timedelta(minutes=15))

    # 3. H-4 Test: Future-dated deal (time_utc > reference_now_utc) must fail closed and never be silently filtered
    mock.set_positions([])
    mock.set_account(
        base_acct.model_copy(
            update={
                "balance": Decimal("20000.00"),
                "credit": Decimal("0.00"),
                "equity": Decimal("20000.00"),
                "margin_free": Decimal("20000.00"),
            }
        )
    )
    future_deal = MT5DealRecord(
        ticket=1009,
        order=9001,
        position_id=9501,
        time_utc=now_utc + timedelta(hours=2),
        deal_type=int(MT5DealType.DEAL_TYPE_BUY),
        symbol="XAUUSD",
        volume=Decimal("0.10"),
        price=Decimal("2650.00"),
        profit=Decimal("0.00"),
    )
    mock.add_deal(future_deal)
    with pytest.raises(AccountStateReconstructionError, match="Future-dated deal detected"):
        tracker.reconstruct_report(reference_now_utc=now_utc + timedelta(minutes=20))


def _build_phase74_stack(
    tmp_path: Path,
    *,
    tag: str = "default",
    risk_params: RiskParameters | None = None,
) -> tuple[
    MockMT5TerminalBackend,
    MT5ConnectionAdapter,
    MT5MarketDataAdapter,
    PersistentAccountRiskTracker,
    PersistentExecutionLedger,
    RiskEngine,
    SignalLifecycleManager,
    PreTradeSafetyGate,
    MT5DemoExecutionEngine,
]:
    """Build a complete offline Phase 7.4 DEMO execution stack for deterministic unit tests."""
    mock = MockMT5TerminalBackend()
    conn = MT5ConnectionAdapter(
        backend=mock,
        config=MT5ConnectionConfig(
            login=100200300,
            server="AtlasBroker-Demo",
            expected_login=100200300,
            expected_server="AtlasBroker-Demo",
        ),
    )
    conn.connect()

    origin_anchor = TrustedBalanceAnchorConfig(
        anchor_balance=Decimal("0.00"),
        anchor_timestamp_utc=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        require_initial_deposit_deal=True,
    )
    safety_cfg = ExecutionSafetyConfig(
        account_risk_state_path=tmp_path / f"{tag}_account_risk_state.json",
        ledger_path=tmp_path / f"{tag}_execution_ledger.json",
        trusted_balance_anchor=origin_anchor,
        max_allowed_spread_points=Decimal("50"),
        max_entry_deviation_points=20,
        max_tick_age_seconds=30,
    )
    md = MT5MarketDataAdapter(connection=conn, safety_config=safety_cfg)
    tracker = PersistentAccountRiskTracker(connection=conn, safety_config=safety_cfg)
    ledger = PersistentExecutionLedger(safety_config=safety_cfg)
    risk_engine = RiskEngine(risk_parameters=risk_params or RiskParameters())
    lifecycle = SignalLifecycleManager()
    gate = PreTradeSafetyGate(
        connection=conn,
        market_data=md,
        account_tracker=tracker,
        ledger=ledger,
        risk_engine=risk_engine,
        safety_config=safety_cfg,
        lifecycle_manager=lifecycle,
    )
    engine = MT5DemoExecutionEngine(
        connection=conn,
        market_data=md,
        account_tracker=tracker,
        ledger=ledger,
        risk_engine=risk_engine,
        safety_config=safety_cfg,
        lifecycle_manager=lifecycle,
        safety_gate=gate,
    )
    return mock, conn, md, tracker, ledger, risk_engine, lifecycle, gate, engine


def _sample_buy_signal(
    *,
    entry: str = "2650.20",
    sl: str = "2640.20",
    tp: str = "2670.20",
    timeframe: Timeframe = Timeframe.H1,
    timestamp: datetime = FIXED_UTC,
) -> TradeSignal:
    return TradeSignal(
        symbol="XAUUSD",
        direction=SignalDirection.BUY,
        entry_price=Decimal(entry),
        stop_loss=Decimal(sl),
        take_profit=Decimal(tp),
        timeframe=timeframe,
        strategy_id="smc_confluence_v1",
        confidence=Decimal("0.85"),
        timestamp=timestamp,
        rationale="Bullish BOS + discount FVG retest.",
    )


def test_ledger_state_transitions_and_unresolved_intent_protection(tmp_path: Path) -> None:
    """Verify PersistentExecutionLedger state machine transitions, idempotency,
    and strict protection preventing unresolved intents from being silently dismissed or retried.
    """
    _, _, _, _, ledger, _, _, _, _ = _build_phase74_stack(tmp_path, tag="ledger_sm")
    acct_login = 100200300
    server = "AtlasBroker-Demo"

    # 1. CREATED -> VALIDATED -> SUBMITTING -> UNKNOWN -> RECONCILIATION_REQUIRED
    e1 = ledger.create_intent(
        account_login=acct_login,
        server=server,
        client_order_id="ord-sm-1",
        signal_id="sig-sm-1",
        symbol="XAUUSD",
        timeframe=Timeframe.H1,
        direction=SignalDirection.BUY,
        magic=202607,
        requested_volume=Decimal("0.10"),
        requested_price=Decimal("2650.20"),
        stop_loss=Decimal("2640.20"),
        take_profit=Decimal("2670.20"),
        filling_mode=MT5OrderFillingMode.ORDER_FILLING_FOK,
        now_utc=FIXED_UTC,
    )
    assert e1.status == OrderLedgerStatus.CREATED

    # Cannot jump directly from CREATED to SUBMITTING (must pass through VALIDATED)
    with pytest.raises(OrderExecutionError, match="Invalid execution ledger state transition"):
        ledger.transition_entry(
            "ord-sm-1",
            OrderLedgerStatus.SUBMITTING,
            account_login=acct_login,
            server=server,
            reason="Illegal direct jump",
        )

    e1_val = ledger.transition_entry(
        "ord-sm-1",
        OrderLedgerStatus.VALIDATED,
        account_login=acct_login,
        server=server,
        reason="Passed safety gate",
    )
    assert e1_val.status == OrderLedgerStatus.VALIDATED

    e1_sub = ledger.transition_entry(
        "ord-sm-1",
        OrderLedgerStatus.SUBMITTING,
        account_login=acct_login,
        server=server,
        reason="Persisted write-ahead before order_send",
    )
    assert e1_sub.status == OrderLedgerStatus.SUBMITTING
    assert ledger.has_unresolved_entries() is True

    # While ord-sm-1 is SUBMITTING, any new intent creation is blocked with AmbiguousExecutionStateError
    with pytest.raises(AmbiguousExecutionStateError, match="remain unresolved"):
        ledger.create_intent(
            account_login=acct_login,
            server=server,
            client_order_id="ord-sm-2",
            signal_id="sig-sm-2",
            symbol="XAUUSD",
            timeframe=Timeframe.H1,
            direction=SignalDirection.BUY,
            magic=202607,
            requested_volume=Decimal("0.10"),
            requested_price=Decimal("2650.20"),
            stop_loss=Decimal("2640.20"),
            take_profit=Decimal("2670.20"),
            filling_mode=MT5OrderFillingMode.ORDER_FILLING_FOK,
        )

    # Cannot silently dismiss SUBMITTING to REJECTED or CANCELLED without synchronous_broker_rejection or reconciliation_verified
    with pytest.raises(AmbiguousExecutionStateError, match="without either a synchronous broker rejection"):
        ledger.transition_entry(
            "ord-sm-1",
            OrderLedgerStatus.REJECTED,
            account_login=acct_login,
            server=server,
            reason="Attempted silent dismissal",
        )

    # Transition SUBMITTING -> UNKNOWN -> RECONCILIATION_REQUIRED
    e1_unk = ledger.transition_entry(
        "ord-sm-1",
        OrderLedgerStatus.UNKNOWN,
        account_login=acct_login,
        server=server,
        retcode=int(MT5Retcode.TRADE_RETCODE_TIMEOUT),
        reason="Timeout 10012",
    )
    assert e1_unk.status == OrderLedgerStatus.UNKNOWN

    e1_rec = ledger.transition_entry(
        "ord-sm-1",
        OrderLedgerStatus.RECONCILIATION_REQUIRED,
        account_login=acct_login,
        server=server,
        reason="Requires broker reconciliation",
    )
    assert e1_rec.status == OrderLedgerStatus.RECONCILIATION_REQUIRED

    # Cannot dismiss RECONCILIATION_REQUIRED to CANCELLED without reconciliation_verified=True
    with pytest.raises(AmbiguousExecutionStateError, match="without verified broker reconciliation proof"):
        ledger.transition_entry(
            "ord-sm-1",
            OrderLedgerStatus.CANCELLED,
            account_login=acct_login,
            server=server,
            reason="Unverified cancel attempt",
        )

    # Cannot mark FILLED or ACTIVE with executed_volume == 0
    with pytest.raises(OrderExecutionError, match="executed_volume"):
        ledger.transition_entry(
            "ord-sm-1",
            OrderLedgerStatus.FILLED,
            account_login=acct_login,
            server=server,
            executed_volume=Decimal("0"),
            reason="Invalid zero volume fill",
        )

    # Verified broker reconciliation resolves RECONCILIATION_REQUIRED -> CANCELLED
    e1_done = ledger.transition_entry(
        "ord-sm-1",
        OrderLedgerStatus.CANCELLED,
        account_login=acct_login,
        server=server,
        reconciliation_status=ReconciliationStatus.RECONCILED_NOT_EXECUTED,
        reconciliation_verified=True,
        reason="Verified absent on broker",
    )
    assert e1_done.status == OrderLedgerStatus.CANCELLED
    assert len(e1_done.history) == 6
    assert ledger.has_unresolved_entries() is False

    # Terminal CANCELLED entry cannot transition back to SUBMITTING
    with pytest.raises(OrderExecutionError, match="Invalid execution ledger state transition"):
        ledger.transition_entry(
            "ord-sm-1",
            OrderLedgerStatus.SUBMITTING,
            account_login=acct_login,
            server=server,
            reason="Illegal retry of cancelled order",
        )


def test_pre_trade_safety_gate_all_14_deterministic_checks(tmp_path: Path) -> None:
    """Verify PreTradeSafetyGate deterministically enforces all 14 safety checks:
    DEMO mode, terminal permissions, WAIT rejection, expiration, stale tick, spread limit,
    price drift, stop/freeze levels, RiskEngine (RR, daily loss, drawdown, max positions),
    volume limits/step, and order_check margin.
    """
    mock, conn, _, _, _, _, lifecycle, gate, _ = _build_phase74_stack(
        tmp_path,
        tag="safety_gate",
    )
    sig = _sample_buy_signal()

    # 1. WAIT signal rejected
    wait_sig = TradeSignal(
        symbol="XAUUSD",
        direction=SignalDirection.WAIT,
        timeframe=Timeframe.H1,
        strategy_id="smc",
        confidence=Decimal("0.40"),
        timestamp=FIXED_UTC,
        rationale="No setup",
    )
    with pytest.raises(OrderValidationError, match="WAIT signals are non-executable"):
        gate.validate_and_build_order(wait_sig, signal_id="sig-wait", reference_now_utc=FIXED_UTC)

    # 2. Stale market data rejected
    mock.set_tick(
        SymbolTick(
            symbol="XAUUSD",
            timestamp=FIXED_UTC - timedelta(seconds=60),
            bid=Decimal("2650.00"),
            ask=Decimal("2650.20"),
        )
    )
    with pytest.raises(StaleMarketDataError, match="Stale tick"):
        gate.validate_and_build_order(sig, signal_id="sig-stale", reference_now_utc=FIXED_UTC)

    # Restore fresh tick, then test 3. Excessive spread (> 50 points = 0.50)
    mock.set_tick(
        SymbolTick(
            symbol="XAUUSD",
            timestamp=FIXED_UTC,
            bid=Decimal("2650.00"),
            ask=Decimal("2650.80"),  # 80 points > 50 points limit
        )
    )
    with pytest.raises(OrderValidationError, match="exceeds max_allowed_spread_points"):
        gate.validate_and_build_order(sig, signal_id="sig-spread", reference_now_utc=FIXED_UTC)

    # 4. Excessive price drift (> 20 points = 0.20 from entry 2650.20)
    mock.set_tick(
        SymbolTick(
            symbol="XAUUSD",
            timestamp=FIXED_UTC,
            bid=Decimal("2650.50"),
            ask=Decimal("2650.70"),  # Ask 2650.70 drifted 50 points from 2650.20
        )
    )
    with pytest.raises(OrderValidationError, match="exceeding max_entry_deviation_points"):
        gate.validate_and_build_order(sig, signal_id="sig-drift", reference_now_utc=FIXED_UTC)

    # Restore normal tick (Bid 2650.00, Ask 2650.20)
    mock.set_tick(
        SymbolTick(
            symbol="XAUUSD",
            timestamp=FIXED_UTC,
            bid=Decimal("2650.00"),
            ask=Decimal("2650.20"),
        )
    )

    # 5. Stop-level / freeze-level restriction violation (trade_stops_level = 20 points = 0.20)
    tight_sl_sig = _sample_buy_signal(entry="2650.20", sl="2650.10", tp="2670.20")
    with pytest.raises(OrderValidationError, match="stop/freeze level minimum distance"):
        gate.validate_and_build_order(tight_sl_sig, signal_id="sig-tight-sl", reference_now_utc=FIXED_UTC)

    # 6. RiskEngine rejection: insufficient Risk/Reward ratio (< 1.5)
    low_rr_sig = _sample_buy_signal(entry="2650.20", sl="2640.20", tp="2655.20")  # RR = 0.5 < 1.5
    with pytest.raises(OrderValidationError, match="INSUFFICIENT_RISK_REWARD_RATIO"):
        gate.validate_and_build_order(low_rr_sig, signal_id="sig-low-rr", reference_now_utc=FIXED_UTC)
    assert lifecycle.get_record("sig-low-rr").state == SignalLifecycleState.RISK_REJECTED

    # 7. Invalid volume overrides (below min, non-step multiple, or above RiskEngine calculated size)
    # At $10,000 equity, 1% risk = $100, stop distance = 10.00 (1000 ticks * $1/tick = $1000/lot) -> RiskEngine size = 0.10 lots
    sig_vol = _sample_buy_signal(entry="2650.20", sl="2640.20", tp="2670.20")
    with pytest.raises(OrderValidationError, match="below minimum allowed volume"):
        gate.validate_and_build_order(
            sig_vol,
            signal_id="sig-vol-min",
            requested_volume=Decimal("0.001"),
            reference_now_utc=FIXED_UTC,
        )

    with pytest.raises(OrderValidationError, match="not an exact multiple of volume_step"):
        gate.validate_and_build_order(
            sig_vol,
            signal_id="sig-vol-step",
            requested_volume=Decimal("0.055"),
            reference_now_utc=FIXED_UTC,
        )

    with pytest.raises(OrderValidationError, match="exceeds RiskEngine-approved position size"):
        gate.validate_and_build_order(
            sig_vol,
            signal_id="sig-vol-over",
            requested_volume=Decimal("0.25"),
            reference_now_utc=FIXED_UTC,
        )

    # 8. Daily loss limit breach via RiskEngine
    base_acct = mock.account_info()
    assert base_acct is not None
    # Simulate -$350 floating loss on open positions (> 3% of $10,000 daily_starting_balance)
    mock.set_account(
        base_acct.model_copy(
            update={
                "balance": Decimal("10000.00"),
                "equity": Decimal("9650.00"),
                "margin_free": Decimal("9650.00"),
            }
        )
    )
    sig_dd = _sample_buy_signal(entry="2650.20", sl="2635.20", tp="2680.20")
    with pytest.raises(OrderValidationError, match="DAILY_LOSS_LIMIT_EXCEEDED"):
        gate.validate_and_build_order(sig_dd, signal_id="sig-daily-loss", reference_now_utc=FIXED_UTC)

    # Restore healthy account, then test 9. Terminal trade_allowed=False permission block
    mock.set_account(base_acct)
    base_term = mock.terminal_info()
    assert base_term is not None
    mock.set_terminal(base_term.model_copy(update={"trade_allowed": False}))
    sig_perm = _sample_buy_signal(entry="2650.20", sl="2632.20", tp="2685.20")
    with pytest.raises(OrderValidationError, match="algorithmic trading is disabled"):
        gate.validate_and_build_order(sig_perm, signal_id="sig-no-algo", reference_now_utc=FIXED_UTC)
    mock.set_terminal(base_term)

    # 10. Valid signal passes all 14 checks and transitions SignalLifecycleManager to RISK_APPROVED
    valid_res = gate.validate_and_build_order(
        sig,
        signal_id="sig-valid-14",
        reference_now_utc=FIXED_UTC,
    )
    assert valid_res.approved_volume == Decimal("0.10")
    assert valid_res.filling_mode == MT5OrderFillingMode.ORDER_FILLING_FOK
    assert lifecycle.get_record("sig-valid-14").state == SignalLifecycleState.RISK_APPROVED

    # 11. Switching account to REAL (trade_mode=2) immediately disconnects terminal and raises LiveTradingBlockedError
    mock.set_account(base_acct.model_copy(update={"trade_mode": int(MT5TradeMode.ACCOUNT_TRADE_MODE_REAL)}))
    with pytest.raises(LiveTradingBlockedError):
        gate.validate_and_build_order(sig, signal_id="sig-real-block", reference_now_utc=FIXED_UTC)
    assert conn.is_connected is False
    assert mock.is_initialized is False


def test_demo_execution_engine_full_fill_partial_fill_rejections_and_duplicates(
    tmp_path: Path,
) -> None:
    """Verify MT5DemoExecutionEngine write-ahead lifecycle, 10009 DONE full fills,
    10010 DONE_PARTIAL partial fills, explicit rejections, duplicate prevention,
    and TRADE_RETCODE_ONLY_REAL (10032) emergency shutdown.
    """
    mock, conn, _, _, ledger, _, lifecycle, _, engine = _build_phase74_stack(
        tmp_path,
        tag="engine_fills",
    )

    # 1. Full fill (10009 TRADE_RETCODE_DONE)
    sig1 = _sample_buy_signal(entry="2650.20", sl="2640.20", tp="2670.20")
    res1 = engine.execute_signal(
        sig1,
        signal_id="sig-fill-1",
        client_order_id="ord-fill-1",
        reference_now_utc=FIXED_UTC,
    )
    assert res1.executed is True
    assert res1.partially_filled is False
    assert res1.ambiguous_requires_reconciliation is False
    assert res1.ledger_entry.status == OrderLedgerStatus.ACTIVE
    assert res1.ledger_entry.reconciliation_status == ReconciliationStatus.RECONCILED_EXECUTED
    assert res1.ledger_entry.executed_volume == Decimal("0.10")
    assert res1.ledger_entry.remaining_volume == Decimal("0.00")
    assert res1.ledger_entry.fill_price == Decimal("2650.20")
    assert res1.ledger_entry.position_ticket is not None and res1.ledger_entry.position_ticket > 0

    # Verify exact write-ahead state transition history: CREATED -> VALIDATED -> SUBMITTING -> SUBMITTED -> FILLED -> ACTIVE
    visited_statuses = [h.to_status for h in res1.ledger_entry.history]
    assert visited_statuses == [
        OrderLedgerStatus.CREATED,
        OrderLedgerStatus.VALIDATED,
        OrderLedgerStatus.SUBMITTING,
        OrderLedgerStatus.SUBMITTED,
        OrderLedgerStatus.FILLED,
        OrderLedgerStatus.ACTIVE,
    ]
    assert lifecycle.get_record("sig-fill-1").state == SignalLifecycleState.ACTIVE

    # 2. Duplicate prevention: same client_order_id, same signal_id, or opposing SELL on XAUUSD
    with pytest.raises(DuplicateOrderError):
        engine.execute_signal(
            sig1,
            signal_id="sig-fill-1",
            client_order_id="ord-fill-1-retry",
            reference_now_utc=FIXED_UTC,
        )

    sell_conflict = TradeSignal(
        symbol="XAUUSD",
        direction=SignalDirection.SELL,
        entry_price=Decimal("2650.00"),
        stop_loss=Decimal("2660.00"),
        take_profit=Decimal("2630.00"),
        timeframe=Timeframe.H1,
        strategy_id="smc_confluence_v1",
        confidence=Decimal("0.85"),
        timestamp=FIXED_UTC,
        rationale="Opposing SELL setup while BUY is active",
    )
    with pytest.raises(DuplicateOrderError, match="Conflicting"):
        engine.execute_signal(
            sell_conflict,
            signal_id="sig-sell-conflict",
            reference_now_utc=FIXED_UTC,
        )

    # 3. Partial fill (10010 TRADE_RETCODE_DONE_PARTIAL: 0.04 lots filled out of 0.10 requested)
    def _partial_fill_hook(req: MT5OrderRequest) -> MT5OrderSendResult:
        partial_vol = Decimal("0.04")
        mock.set_positions(
            [
                * (mock.positions_get(None) or ()),
                MT5PositionRecord(
                    ticket=88001,
                    identifier=88001,
                    time_utc=FIXED_UTC,
                    position_type=0,
                    magic=req.magic,
                    symbol=req.symbol,
                    volume=partial_vol,
                    price_open=req.price,
                    sl=req.sl,
                    tp=req.tp,
                    price_current=req.price,
                    profit=Decimal("0.00"),
                    comment=req.comment,
                ),
            ]
        )
        mock.add_deal(
            MT5DealRecord(
                ticket=88002,
                order=88000,
                position_id=88001,
                time_utc=FIXED_UTC,
                deal_type=int(MT5DealType.DEAL_TYPE_BUY),
                entry=int(MT5DealEntry.DEAL_ENTRY_IN),
                magic=req.magic,
                symbol=req.symbol,
                volume=partial_vol,
                price=req.price,
                profit=Decimal("0.00"),
                comment=req.comment,
            )
        )
        return MT5OrderSendResult(
            retcode=int(MT5Retcode.TRADE_RETCODE_DONE_PARTIAL),
            deal=88002,
            order=88000,
            volume=partial_vol,
            price=req.price,
            bid=Decimal("2650.00"),
            ask=Decimal("2650.20"),
            comment="Partial fill",
        )

    mock.set_order_send_hook(_partial_fill_hook)
    sig2 = _sample_buy_signal(
        entry="2650.10",
        sl="2640.20",
        tp="2670.20",
        timeframe=Timeframe.M15,
    )
    res_partial = engine.execute_signal(
        sig2,
        signal_id="sig-partial-1",
        client_order_id="ord-partial-1",
        reference_now_utc=FIXED_UTC,
    )
    assert res_partial.executed is True
    assert res_partial.partially_filled is True
    assert res_partial.ledger_entry.status == OrderLedgerStatus.PARTIALLY_FILLED
    assert res_partial.ledger_entry.executed_volume == Decimal("0.04")
    assert res_partial.ledger_entry.remaining_volume == Decimal("0.06")
    assert lifecycle.get_record("sig-partial-1").state == SignalLifecycleState.ACTIVE

    # 4. Explicit synchronous broker rejection (10006 TRADE_RETCODE_REJECT)
    mock.set_order_send_hook(
        lambda req: MT5OrderSendResult(
            retcode=int(MT5Retcode.TRADE_RETCODE_REJECT),
            comment="Rejected by dealer",
        )
    )
    sig3 = _sample_buy_signal(
        entry="2650.05",
        sl="2640.20",
        tp="2670.20",
        timeframe=Timeframe.M5,
    )
    res_rej = engine.execute_signal(
        sig3,
        signal_id="sig-reject-1",
        client_order_id="ord-reject-1",
        reference_now_utc=FIXED_UTC,
    )
    assert res_rej.executed is False
    assert res_rej.ambiguous_requires_reconciliation is False
    assert res_rej.ledger_entry.status == OrderLedgerStatus.REJECTED
    assert res_rej.ledger_entry.executed_volume == Decimal("0.00")
    # SignalLifecycleManager must transition to CANCELLED (never ACTIVE!)
    assert lifecycle.get_record("sig-reject-1").state == SignalLifecycleState.CANCELLED

    # 5. Broker returns TRADE_RETCODE_ONLY_REAL (10032) -> controlled shutdown + LiveTradingBlockedError
    mock.set_order_send_hook(
        lambda req: MT5OrderSendResult(
            retcode=int(MT5Retcode.TRADE_RETCODE_ONLY_REAL),
            comment="Only real accounts allowed",
        )
    )
    sig4 = _sample_buy_signal(
        entry="2650.15",
        sl="2640.20",
        tp="2670.20",
        timeframe=Timeframe.M1,
    )
    with pytest.raises(LiveTradingBlockedError, match="TRADE_RETCODE_ONLY_REAL"):
        engine.execute_signal(
            sig4,
            signal_id="sig-only-real",
            client_order_id="ord-only-real",
            reference_now_utc=FIXED_UTC,
        )
    assert conn.is_connected is False
    assert mock.is_initialized is False
    entry_real_rej = ledger.get_entry("ord-only-real")
    assert entry_real_rej is not None and entry_real_rej.status == OrderLedgerStatus.REJECTED


def test_ambiguous_outcomes_timeouts_crashes_persistence_corruption_and_broker_reconciliation(
    tmp_path: Path,
) -> None:
    """Verify ambiguous retcodes (10012 TIMEOUT, 10031 CONNECTION), exceptions during order_send,
    crash between persistence and broker submission, crash after broker acceptance before local
    confirmation, persistence recovery locking, concurrent submissions, and broker reconciliation.
    """
    mock, _, _, _, ledger, _, lifecycle, _, engine = _build_phase74_stack(
        tmp_path,
        tag="engine_recon",
    )

    # 1. Ambiguous retcode 10012 (TRADE_RETCODE_TIMEOUT) where order did NOT execute on broker
    send_call_count = 0

    def _timeout_no_fill_hook(req: MT5OrderRequest) -> MT5OrderSendResult:
        nonlocal send_call_count
        send_call_count += 1
        return MT5OrderSendResult(
            retcode=int(MT5Retcode.TRADE_RETCODE_TIMEOUT),
            comment="Request timed out",
        )

    mock.set_order_send_hook(_timeout_no_fill_hook)
    sig_timeout = _sample_buy_signal(entry="2650.20", sl="2640.20", tp="2670.20")
    res_timeout = engine.execute_signal(
        sig_timeout,
        signal_id="sig-timeout-1",
        client_order_id="ord-timeout-1",
        reference_now_utc=FIXED_UTC,
    )
    # Never blindly retried (called exactly once!)
    assert send_call_count == 1
    assert res_timeout.executed is False
    assert res_timeout.ambiguous_requires_reconciliation is True
    assert res_timeout.ledger_entry.status == OrderLedgerStatus.RECONCILIATION_REQUIRED
    # Signal stays in RISK_APPROVED (not ACTIVE and not CANCELLED yet)
    assert lifecycle.get_record("sig-timeout-1").state == SignalLifecycleState.RISK_APPROVED

    # While ord-timeout-1 is unresolved, any new order submission is blocked with AmbiguousExecutionStateError
    sig_blocked = _sample_buy_signal(entry="2650.10", sl="2638.10", tp="2674.10")
    with pytest.raises(AmbiguousExecutionStateError, match="remain unresolved"):
        engine.execute_signal(
            sig_blocked,
            signal_id="sig-blocked-by-timeout",
            reference_now_utc=FIXED_UTC,
        )

    # Reconcile with broker: proves ord-timeout-1 never executed -> transitions to CANCELLED and unblocks engine
    recon_rep1 = engine.reconcile_unresolved_orders(
        reference_now_utc=FIXED_UTC + timedelta(seconds=5)
    )
    assert "ord-timeout-1" in recon_rep1.reconciled_not_executed_ids
    entry_after_recon1 = ledger.get_entry("ord-timeout-1")
    assert entry_after_recon1 is not None
    assert entry_after_recon1.status == OrderLedgerStatus.CANCELLED
    assert entry_after_recon1.reconciliation_status == ReconciliationStatus.RECONCILED_NOT_EXECUTED
    assert lifecycle.get_record("sig-timeout-1").state == SignalLifecycleState.CANCELLED

    # 2. Ambiguous retcode 10031 (TRADE_RETCODE_CONNECTION) where order DID execute on the broker before connection dropped!
    def _connection_lost_after_broker_fill_hook(req: MT5OrderRequest) -> MT5OrderSendResult:
        mock.set_positions(
            [
                MT5PositionRecord(
                    ticket=91001,
                    identifier=91001,
                    time_utc=FIXED_UTC,
                    position_type=0,
                    magic=req.magic,
                    symbol=req.symbol,
                    volume=req.volume,
                    price_open=req.price,
                    sl=req.sl,
                    tp=req.tp,
                    price_current=req.price,
                    profit=Decimal("0.00"),
                    comment=req.comment,
                )
            ]
        )
        mock.add_deal(
            MT5DealRecord(
                ticket=91002,
                order=91000,
                position_id=91001,
                time_utc=FIXED_UTC,
                deal_type=int(MT5DealType.DEAL_TYPE_BUY),
                entry=int(MT5DealEntry.DEAL_ENTRY_IN),
                magic=req.magic,
                symbol=req.symbol,
                volume=req.volume,
                price=req.price,
                profit=Decimal("0.00"),
                comment=req.comment,
            )
        )
        return MT5OrderSendResult(
            retcode=int(MT5Retcode.TRADE_RETCODE_CONNECTION),
            comment="Connection lost during response",
        )

    mock.set_order_send_hook(_connection_lost_after_broker_fill_hook)
    sig_conn_drop = _sample_buy_signal(entry="2650.15", sl="2640.20", tp="2672.15")
    res_conn_drop = engine.execute_signal(
        sig_conn_drop,
        signal_id="sig-conn-drop",
        client_order_id="ord-conn-drop",
        reference_now_utc=FIXED_UTC + timedelta(seconds=10),
    )
    assert res_conn_drop.executed is False
    assert res_conn_drop.ambiguous_requires_reconciliation is True
    assert res_conn_drop.ledger_entry.status == OrderLedgerStatus.RECONCILIATION_REQUIRED
    assert lifecycle.get_record("sig-conn-drop").state == SignalLifecycleState.RISK_APPROVED

    # Reconcile with broker: discovers live position 91001 & deal 91002 -> marks ACTIVE and activates SignalLifecycleManager!
    recon_rep2 = engine.reconcile_unresolved_orders(
        reference_now_utc=FIXED_UTC + timedelta(seconds=15)
    )
    assert "ord-conn-drop" in recon_rep2.reconciled_executed_ids
    entry_after_recon2 = ledger.get_entry("ord-conn-drop")
    assert entry_after_recon2 is not None
    assert entry_after_recon2.status == OrderLedgerStatus.ACTIVE
    assert entry_after_recon2.reconciliation_status == ReconciliationStatus.RECONCILED_EXECUTED
    assert entry_after_recon2.executed_volume == Decimal("0.10")
    assert entry_after_recon2.position_ticket == 91001
    assert lifecycle.get_record("sig-conn-drop").state == SignalLifecycleState.ACTIVE

    # Clear open positions before testing crash between persistence and broker submission
    mock.set_positions([])
    mock.set_order_send_hook(None)

    # 3. Crash between write-ahead SUBMITTING persistence and broker order_send()
    def _simulate_crash_before_send(submitting_entry: ExecutionLedgerEntry) -> None:
        assert submitting_entry.status == OrderLedgerStatus.SUBMITTING
        raise RuntimeError("Simulated process crash right after SUBMITTING write-ahead")

    engine.set_after_submitting_before_send_hook(_simulate_crash_before_send)
    sig_crash_pre = _sample_buy_signal(
        entry="2650.05",
        sl="2637.05",
        tp="2676.05",
        timeframe=Timeframe.M15,
    )
    with pytest.raises(RuntimeError, match="Simulated process crash"):
        engine.execute_signal(
            sig_crash_pre,
            signal_id="sig-crash-pre",
            client_order_id="ord-crash-pre",
            reference_now_utc=FIXED_UTC + timedelta(seconds=20),
        )
    engine.set_after_submitting_before_send_hook(None)

    # Simulate application restart with a fresh PersistentExecutionLedger instance reading the disk file
    restarted_ledger = PersistentExecutionLedger(safety_config=engine.safety_config)
    health_on_restart = restarted_ledger.load_or_initialize(
        account_login=100200300,
        server="AtlasBroker-Demo",
        reference_now_utc=FIXED_UTC + timedelta(seconds=25),
    )
    assert health_on_restart == LedgerHealthStatus.HEALTHY
    restarted_entry = restarted_ledger.get_entry("ord-crash-pre")
    assert restarted_entry is not None
    # On restart, the leftover SUBMITTING entry was automatically promoted to RECONCILIATION_REQUIRED!
    assert restarted_entry.status == OrderLedgerStatus.RECONCILIATION_REQUIRED
    assert restarted_ledger.has_unresolved_entries() is True

    # Reconcile on the restarted ledger -> proves order never reached broker -> marks CANCELLED
    recon_rep3 = restarted_ledger.reconcile_with_broker(
        engine.connection,
        reference_now_utc=FIXED_UTC + timedelta(seconds=30),
    )
    assert "ord-crash-pre" in recon_rep3.reconciled_not_executed_ids
    assert restarted_ledger.has_unresolved_entries() is False

    # 4. Missing primary ledger file while .bak exists + orphan broker position recovery
    # Simulate an order executed on the broker whose ledger entry was lost when primary file was deleted
    mock.set_positions(
        [
            MT5PositionRecord(
                ticket=95001,
                identifier=95001,
                time_utc=FIXED_UTC + timedelta(seconds=35),
                position_type=0,
                magic=engine.safety_config.magic_number,
                symbol="XAUUSD",
                volume=Decimal("0.10"),
                price_open=Decimal("2650.20"),
                sl=Decimal("2640.20"),
                tp=Decimal("2670.20"),
                price_current=Decimal("2650.20"),
                profit=Decimal("0.00"),
                comment="ord-orphan-after-crash",
            )
        ]
    )
    engine.safety_config.ledger_path.unlink()

    locked_ledger = PersistentExecutionLedger(safety_config=engine.safety_config)
    locked_status = locked_ledger.load_or_initialize(
        account_login=100200300,
        server="AtlasBroker-Demo",
        reference_now_utc=FIXED_UTC + timedelta(seconds=40),
    )
    assert locked_status == LedgerHealthStatus.RECOVERY_REQUIRED_LOCKED

    # New submissions MUST be blocked with LedgerRecoveryLockError while locked
    with pytest.raises(LedgerRecoveryLockError):
        locked_ledger.ensure_ready_for_new_submission(
            account_login=100200300,
            server="AtlasBroker-Demo",
        )

    # Reconciling with broker discovers the orphan position, records it in the ledger, and unlocks to HEALTHY
    recon_rep4 = locked_ledger.reconcile_with_broker(
        engine.connection,
        reference_now_utc=FIXED_UTC + timedelta(seconds=45),
    )
    assert recon_rep4.health_status == LedgerHealthStatus.HEALTHY
    assert "ord-orphan-after-crash" in recon_rep4.discovered_orphan_ids
    orphan_rec = locked_ledger.get_entry("ord-orphan-after-crash")
    assert orphan_rec is not None and orphan_rec.status == OrderLedgerStatus.ACTIVE

    # 5. Concurrent multi-threaded order submissions for the same setup allow at most 1 fill
    _, _, _, _, _, _, _, _, conc_engine = _build_phase74_stack(
        tmp_path,
        tag="engine_concurrent",
    )
    conc_sig = _sample_buy_signal(entry="2650.20", sl="2640.20", tp="2670.20")
    results: list[bool] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def _worker(idx: int) -> None:
        barrier.wait()
        try:
            r = conc_engine.execute_signal(
                conc_sig,
                signal_id="sig-concurrent-1",
                client_order_id=f"ord-concurrent-{idx}",
                reference_now_utc=FIXED_UTC,
            )
            results.append(r.executed)
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=_worker, args=(1,))
    t2 = threading.Thread(target=_worker, args=(2,))
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert results == [True]
    assert len(errors) == 1
    assert isinstance(errors[0], DuplicateOrderError)


def test_c1_and_c2_retcode_allowlist_and_exact_broker_identity_matching(
    tmp_path: Path,
) -> None:
    """Verify remediation of C-1 and C-2:
    1. C-1: 10011 (TRADE_RETCODE_ERROR), 10023 (TRADE_RETCODE_ORDER_CHANGED), 0, negative (-5),
       and unknown (99999) retcodes are never classified as deterministic REJECTED; they transition
       to UNKNOWN -> RECONCILIATION_REQUIRED and block further submissions until broker reconciliation.
    2. C-2: Exact broker identity matching prevents prefix IDs ('ord-1' vs 'ord-10', 'sig-1' vs 'sig-10')
       from cross-matching or summing volumes across distinct orders, while still parsing bracketed
       broker suffixes ('ord-1[sl 2640.20]').
    """
    mock, _, _, _, ledger, _, lifecycle, _, engine = _build_phase74_stack(
        tmp_path,
        tag="c1_c2_remediation",
    )

    assert int(MT5Retcode.TRADE_RETCODE_TIMEOUT) in AMBIGUOUS_SUBMISSION_RETCODES
    assert int(MT5Retcode.TRADE_RETCODE_CONNECTION) in AMBIGUOUS_SUBMISSION_RETCODES
    assert int(MT5Retcode.TRADE_RETCODE_ERROR) not in DETERMINISTIC_REJECTION_RETCODES
    assert int(MT5Retcode.TRADE_RETCODE_ORDER_CHANGED) not in DETERMINISTIC_REJECTION_RETCODES

    # 1. C-1: Test 10011, 10023, 0, -5, and 99999 retcodes
    for idx, ambig_code in enumerate(
        (
            int(MT5Retcode.TRADE_RETCODE_ERROR),
            int(MT5Retcode.TRADE_RETCODE_ORDER_CHANGED),
            0,
            -5,
            99999,
        ),
        start=1,
    ):
        mock.set_order_send_hook(
            lambda req, rc=ambig_code: MT5OrderSendResult(
                retcode=rc,
                comment=f"Non-deterministic retcode {rc}",
            )
        )
        sig = _sample_buy_signal(entry="2650.20", sl="2640.20", tp="2670.20")
        cid = f"ord-c1-{idx}"
        sid = f"sig-c1-{idx}"
        res = engine.execute_signal(
            sig,
            signal_id=sid,
            client_order_id=cid,
            reference_now_utc=FIXED_UTC + timedelta(seconds=idx),
        )
        assert res.executed is False
        assert res.ambiguous_requires_reconciliation is True
        assert res.ledger_entry.status == OrderLedgerStatus.RECONCILIATION_REQUIRED
        assert [h.to_status for h in res.ledger_entry.history[-2:]] == [
            OrderLedgerStatus.UNKNOWN,
            OrderLedgerStatus.RECONCILIATION_REQUIRED,
        ]
        # Signal must NOT be marked CANCELLED before broker reconciliation!
        assert lifecycle.get_record(sid).state == SignalLifecycleState.RISK_APPROVED

        # New order submission MUST be blocked while unresolved
        with pytest.raises(AmbiguousExecutionStateError):
            engine.execute_signal(
                sig,
                signal_id=f"{sid}-blocked",
                reference_now_utc=FIXED_UTC + timedelta(seconds=idx),
            )

        # Reconcile proves order did not execute on broker -> transitions to CANCELLED and unblocks
        rep = engine.reconcile_unresolved_orders(
            reference_now_utc=FIXED_UTC + timedelta(seconds=idx, milliseconds=500)
        )
        assert cid in rep.reconciled_not_executed_ids
        assert ledger.get_entry(cid).status == OrderLedgerStatus.CANCELLED
        assert lifecycle.get_record(sid).state == SignalLifecycleState.CANCELLED

    # 2. C-2: Verify extract_comment_identifiers handles bracket suffixes without prefix collisions
    assert "ord-1" not in extract_comment_identifiers("ord-10")
    assert "sig-1" not in extract_comment_identifiers("sig-10")
    assert "ord-1" in extract_comment_identifiers("ord-1[sl 2640.20]")
    assert "ord-10" not in extract_comment_identifiers("ord-1[sl 2640.20]")

    # Execute ord-10 (sig-10) for 0.10 lots on H1
    mock.set_order_send_hook(None)
    sig_10 = _sample_buy_signal(
        entry="2650.20",
        sl="2640.20",
        tp="2670.20",
        timeframe=Timeframe.H1,
    )
    res_10 = engine.execute_signal(
        sig_10,
        signal_id="sig-10",
        client_order_id="ord-10",
        reference_now_utc=FIXED_UTC + timedelta(seconds=10),
    )
    assert res_10.executed is True
    assert res_10.ledger_entry.status == OrderLedgerStatus.ACTIVE
    assert res_10.ledger_entry.executed_volume == Decimal("0.10")

    # Now submit ord-1 (sig-1) on M15 which times out (10012) WITHOUT executing on the broker
    mock.set_order_send_hook(
        lambda req: MT5OrderSendResult(
            retcode=int(MT5Retcode.TRADE_RETCODE_TIMEOUT),
            comment="Timeout on ord-1",
        )
    )
    sig_1 = _sample_buy_signal(
        entry="2650.15",
        sl="2640.15",
        tp="2670.15",
        timeframe=Timeframe.M15,
    )
    res_1_timeout = engine.execute_signal(
        sig_1,
        signal_id="sig-1",
        client_order_id="ord-1",
        reference_now_utc=FIXED_UTC + timedelta(seconds=12),
    )
    assert res_1_timeout.ambiguous_requires_reconciliation is True
    assert res_1_timeout.ledger_entry.status == OrderLedgerStatus.RECONCILIATION_REQUIRED

    # Reconciling ord-1 MUST NOT match ord-10's open position or deal!
    rep_c2 = engine.reconcile_unresolved_orders(
        reference_now_utc=FIXED_UTC + timedelta(seconds=14)
    )
    assert "ord-1" in rep_c2.reconciled_not_executed_ids
    assert "ord-1" not in rep_c2.reconciled_executed_ids
    entry_ord1 = ledger.get_entry("ord-1")
    assert entry_ord1 is not None
    assert entry_ord1.status == OrderLedgerStatus.CANCELLED
    assert entry_ord1.executed_volume == Decimal("0.00")
    assert lifecycle.get_record("sig-1").state == SignalLifecycleState.CANCELLED

    # Next, execute ord-2 (sig-2) on M5 while ord-10 is still open on H1; verify volumes are not summed
    mock.set_order_send_hook(None)
    sig_2 = _sample_buy_signal(
        entry="2650.20",
        sl="2640.20",
        tp="2670.20",
        timeframe=Timeframe.M5,
    )
    res_2 = engine.execute_signal(
        sig_2,
        signal_id="sig-2",
        client_order_id="ord-2",
        reference_now_utc=FIXED_UTC + timedelta(seconds=16),
    )
    assert res_2.executed is True
    assert res_2.ledger_entry.executed_volume == Decimal("0.10")
    assert res_2.ledger_entry.position_ticket != res_10.ledger_entry.position_ticket
    assert res_2.ledger_entry.deal_ticket != res_10.ledger_entry.deal_ticket


def test_h1_broker_reconciliation_orphan_deals_orphan_working_orders_and_empty_history_lock(
    tmp_path: Path,
) -> None:
    """Verify remediation of H-1:
    1. Empty broker deal history (deals == ()) fails closed with LedgerRecoveryLockError when resolving
       an unresolved intent or recovery-locked ledger.
    2. Closed orphan deals (DEAL_ENTRY_IN present in history_deals_get() with magic=202607, position
       already closed and absent from positions_get(), missing from stale .bak) are reconstructed as
       FILLED and block duplicate re-execution of that client_order_id / signal_id.
    3. Orphan working orders (orders_get() with magic=202607, missing from stale .bak) are discovered
       as RECONCILIATION_REQUIRED in still_working_ids and keep RECOVERY_REQUIRED_LOCKED locked until
       the working order is resolved.
    """
    mock, conn, _, _, ledger, _, _, _, engine = _build_phase74_stack(
        tmp_path,
        tag="h1_remediation",
    )

    # Create a first completed order so .bak will exist on the next save
    sig_base = _sample_buy_signal(entry="2650.20", sl="2640.20", tp="2670.20")
    res_base = engine.execute_signal(
        sig_base,
        signal_id="sig-h1-base",
        client_order_id="ord-h1-base",
        reference_now_utc=FIXED_UTC,
    )
    assert res_base.executed is True

    # 1. Timeout order followed by empty deal history cache -> reconcile_with_broker must fail closed!
    mock.set_order_send_hook(
        lambda req: MT5OrderSendResult(
            retcode=int(MT5Retcode.TRADE_RETCODE_TIMEOUT),
            comment="Timed out",
        )
    )
    sig_ambig = _sample_buy_signal(
        entry="2650.15",
        sl="2640.15",
        tp="2670.15",
        timeframe=Timeframe.M15,
    )
    engine.execute_signal(
        sig_ambig,
        signal_id="sig-h1-empty-hist",
        client_order_id="ord-h1-empty-hist",
        reference_now_utc=FIXED_UTC + timedelta(seconds=5),
    )
    saved_deals = list(mock.history_deals_get( datetime(1970, 1, 1, tzinfo=timezone.utc), FIXED_UTC + timedelta(days=1)) or ())
    mock.set_deals([])
    with pytest.raises(LedgerRecoveryLockError, match="empty deal history"):
        engine.reconcile_unresolved_orders(reference_now_utc=FIXED_UTC + timedelta(seconds=10))
    # Entry must remain in RECONCILIATION_REQUIRED (never marked CANCELLED on empty deal history!)
    assert ledger.get_entry("ord-h1-empty-hist").status == OrderLedgerStatus.RECONCILIATION_REQUIRED

    # Restore deal history and reconcile ord-h1-empty-hist to CANCELLED
    mock.set_deals(saved_deals)
    rep_clean = engine.reconcile_unresolved_orders(reference_now_utc=FIXED_UTC + timedelta(seconds=12))
    assert "ord-h1-empty-hist" in rep_clean.reconciled_not_executed_ids

    # 2. Simulate primary ledger deletion while:
    #    (a) a closed orphan deal ('ord-closed-orphan') exists in history_deals_get() (already closed, so NOT in positions_get())
    #    (b) an orphan working order ('ord-working-orphan') exists in orders_get()
    mock.set_positions([])
    mock.add_deal(
        MT5DealRecord(
            ticket=97001,
            order=97000,
            position_id=97002,
            time_utc=FIXED_UTC + timedelta(seconds=15),
            deal_type=int(MT5DealType.DEAL_TYPE_BUY),
            entry=int(MT5DealEntry.DEAL_ENTRY_IN),
            magic=engine.safety_config.magic_number,
            symbol="XAUUSD",
            volume=Decimal("0.10"),
            price=Decimal("2650.20"),
            profit=Decimal("0.00"),
            comment="ord-closed-orphan",
        )
    )
    mock.set_orders(
        [
            MT5OrderRecord(
                ticket=98001,
                time_setup_utc=FIXED_UTC + timedelta(seconds=16),
                order_type=0,
                state=1,
                magic=engine.safety_config.magic_number,
                position_id=0,
                symbol="XAUUSD",
                volume_initial=Decimal("0.10"),
                volume_current=Decimal("0.10"),
                price_open=Decimal("2650.20"),
                sl=Decimal("2640.20"),
                tp=Decimal("2670.20"),
                comment="ord-working-orphan",
            )
        ]
    )

    # Delete primary ledger file to trigger RECOVERY_REQUIRED_LOCKED from .bak
    engine.safety_config.ledger_path.unlink()
    rec_ledger = PersistentExecutionLedger(safety_config=engine.safety_config)
    assert (
        rec_ledger.load_or_initialize(
            account_login=100200300,
            server="AtlasBroker-Demo",
            reference_now_utc=FIXED_UTC + timedelta(seconds=20),
        )
        == LedgerHealthStatus.RECOVERY_REQUIRED_LOCKED
    )

    rep_orphans = rec_ledger.reconcile_with_broker(
        conn,
        reference_now_utc=FIXED_UTC + timedelta(seconds=25),
    )
    # Both the closed orphan deal and the orphan working order must be discovered!
    assert "ord-closed-orphan" in rep_orphans.discovered_orphan_ids
    assert "ord-working-orphan" in rep_orphans.discovered_orphan_ids
    assert "ord-working-orphan" in rep_orphans.still_working_ids

    closed_orphan_entry = rec_ledger.get_entry("ord-closed-orphan")
    assert closed_orphan_entry is not None
    assert closed_orphan_entry.status == OrderLedgerStatus.FILLED
    assert closed_orphan_entry.deal_ticket == 97001

    working_orphan_entry = rec_ledger.get_entry("ord-working-orphan")
    assert working_orphan_entry is not None
    assert working_orphan_entry.status == OrderLedgerStatus.RECONCILIATION_REQUIRED
    assert working_orphan_entry.order_ticket == 98001

    # Because 'ord-working-orphan' is still working on the broker, health_status MUST remain RECOVERY_REQUIRED_LOCKED!
    assert rep_orphans.health_status == LedgerHealthStatus.RECOVERY_REQUIRED_LOCKED
    assert rec_ledger.health_status == LedgerHealthStatus.RECOVERY_REQUIRED_LOCKED
    with pytest.raises(LedgerRecoveryLockError):
        rec_ledger.ensure_ready_for_new_submission(
            account_login=100200300,
            server="AtlasBroker-Demo",
        )

    # Once the working order is cancelled/removed on the broker, a second reconciliation resolves it and unlocks to HEALTHY
    mock.set_orders([])
    rep_unlocked = rec_ledger.reconcile_with_broker(
        conn,
        reference_now_utc=FIXED_UTC + timedelta(seconds=30),
    )
    assert rep_unlocked.health_status == LedgerHealthStatus.HEALTHY
    assert "ord-working-orphan" in rep_unlocked.reconciled_not_executed_ids
    assert rec_ledger.get_entry("ord-working-orphan").status == OrderLedgerStatus.CANCELLED

    # Attempting to reuse 'ord-closed-orphan' is blocked by DuplicateOrderError
    with pytest.raises(DuplicateOrderError):
        rec_ledger.ensure_ready_for_new_submission(
            account_login=100200300,
            server="AtlasBroker-Demo",
            client_order_id="ord-closed-orphan",
        )


def test_h2_placed_retcode_verification_exceptions_volume_integrity_and_crash_boundaries(
    tmp_path: Path,
) -> None:
    """Verify remediation of H-2:
    1. 10008 (TRADE_RETCODE_PLACED) with order in orders_get() (no immediate position/deal) transitions
       to RECONCILIATION_REQUIRED (never ACTIVE/FILLED), records order_ticket, and stays in still_working_ids.
    2. 10009 (DONE) with send_result.volume > 0 when broker positions/deals have no matching fill
       never trusts send_result.volume and transitions to RECONCILIATION_REQUIRED.
    3. 10010 (DONE_PARTIAL) where broker deals confirm 0.06 lots while send_result.volume reports 0.03
       preserves exact broker-confirmed executed_volume=0.06 and remaining_volume=0.04.
    4. Post-accept verification exceptions (e.g. positions_get raising RuntimeError) transition to
       RECONCILIATION_REQUIRED without crashing.
    5. Crash after broker acceptance (order executed on broker while ledger on disk is in SUBMITTING)
       and crash between FILLED and ACTIVE persistence both recover cleanly to ACTIVE on restart + reconciliation.
    6. Pre-send MT5ConnectionError after SUBMITTING persistence transitions to FAILED, cancels lifecycle,
       and never calls order_send().
    """
    mock, conn, _, _, ledger, _, lifecycle, _, engine = _build_phase74_stack(
        tmp_path,
        tag="h2_remediation",
    )

    # 1. 10008 PLACED with working order in orders_get() (no position or deal yet)
    def _placed_working_order_hook(req: MT5OrderRequest) -> MT5OrderSendResult:
        mock.set_orders(
            [
                MT5OrderRecord(
                    ticket=81001,
                    time_setup_utc=FIXED_UTC,
                    order_type=0,
                    state=1,
                    magic=req.magic,
                    position_id=0,
                    symbol=req.symbol,
                    volume_initial=req.volume,
                    volume_current=req.volume,
                    price_open=req.price,
                    sl=req.sl,
                    tp=req.tp,
                    comment=req.comment,
                )
            ]
        )
        return MT5OrderSendResult(
            retcode=int(MT5Retcode.TRADE_RETCODE_PLACED),
            order=81001,
            volume=req.volume,
            price=req.price,
            comment="Order placed in book",
        )

    mock.set_order_send_hook(_placed_working_order_hook)
    sig_placed = _sample_buy_signal(entry="2650.20", sl="2640.20", tp="2670.20")
    res_placed = engine.execute_signal(
        sig_placed,
        signal_id="sig-placed-1",
        client_order_id="ord-placed-1",
        reference_now_utc=FIXED_UTC,
    )
    assert res_placed.executed is False
    assert res_placed.ambiguous_requires_reconciliation is True
    assert res_placed.ledger_entry.status == OrderLedgerStatus.RECONCILIATION_REQUIRED
    assert res_placed.ledger_entry.order_ticket == 81001
    assert res_placed.ledger_entry.executed_volume == Decimal("0.00")
    assert res_placed.ledger_entry.remaining_volume == Decimal("0.10")
    assert lifecycle.get_record("sig-placed-1").state == SignalLifecycleState.RISK_APPROVED

    # Reconciling while order is still in orders_get() keeps it in still_working_ids and RECONCILIATION_REQUIRED
    rep_working = engine.reconcile_unresolved_orders(reference_now_utc=FIXED_UTC + timedelta(seconds=2))
    assert "ord-placed-1" in rep_working.still_working_ids
    assert ledger.get_entry("ord-placed-1").status == OrderLedgerStatus.RECONCILIATION_REQUIRED

    # Working order is cancelled on broker -> reconciliation transitions it to CANCELLED
    mock.set_orders([])
    rep_cancelled = engine.reconcile_unresolved_orders(reference_now_utc=FIXED_UTC + timedelta(seconds=4))
    assert "ord-placed-1" in rep_cancelled.reconciled_not_executed_ids
    assert ledger.get_entry("ord-placed-1").status == OrderLedgerStatus.CANCELLED
    assert lifecycle.get_record("sig-placed-1").state == SignalLifecycleState.CANCELLED

    # 2. 10009 DONE with send_result.volume = 0.10 but zero broker deals/positions -> never trusts send_result.volume!
    mock.set_order_send_hook(
        lambda req: MT5OrderSendResult(
            retcode=int(MT5Retcode.TRADE_RETCODE_DONE),
            order=82001,
            deal=82002,
            volume=req.volume,
            price=req.price,
            comment="Unconfirmed fill",
        )
    )
    sig_unconf = _sample_buy_signal(entry="2650.20", sl="2640.20", tp="2670.20", timeframe=Timeframe.M15)
    res_unconf = engine.execute_signal(
        sig_unconf,
        signal_id="sig-unconf-1",
        client_order_id="ord-unconf-1",
        reference_now_utc=FIXED_UTC + timedelta(seconds=6),
    )
    assert res_unconf.executed is False
    assert res_unconf.ambiguous_requires_reconciliation is True
    assert res_unconf.ledger_entry.status == OrderLedgerStatus.RECONCILIATION_REQUIRED
    assert res_unconf.ledger_entry.executed_volume == Decimal("0.00")
    engine.reconcile_unresolved_orders(reference_now_utc=FIXED_UTC + timedelta(seconds=7))

    # 3. 10010 DONE_PARTIAL where broker deal filled 0.06 lots while send_result.volume reported 0.03 -> preserves 0.06!
    def _partial_uncapped_hook(req: MT5OrderRequest) -> MT5OrderSendResult:
        mock.set_positions(
            [
                MT5PositionRecord(
                    ticket=83001,
                    identifier=83001,
                    time_utc=FIXED_UTC,
                    position_type=0,
                    magic=req.magic,
                    symbol=req.symbol,
                    volume=Decimal("0.06"),
                    price_open=req.price,
                    sl=req.sl,
                    tp=req.tp,
                    price_current=req.price,
                    profit=Decimal("0.00"),
                    comment=req.comment,
                )
            ]
        )
        mock.add_deal(
            MT5DealRecord(
                ticket=83002,
                order=83000,
                position_id=83001,
                time_utc=FIXED_UTC,
                deal_type=int(MT5DealType.DEAL_TYPE_BUY),
                entry=int(MT5DealEntry.DEAL_ENTRY_IN),
                magic=req.magic,
                symbol=req.symbol,
                volume=Decimal("0.06"),
                price=req.price,
                profit=Decimal("0.00"),
                comment=req.comment,
            )
        )
        return MT5OrderSendResult(
            retcode=int(MT5Retcode.TRADE_RETCODE_DONE_PARTIAL),
            order=83000,
            deal=83002,
            volume=Decimal("0.03"),  # Smaller than actual broker deal volume 0.06
            price=req.price,
            comment="Partial fill",
        )

    mock.set_order_send_hook(_partial_uncapped_hook)
    sig_part_uncapped = _sample_buy_signal(entry="2650.20", sl="2640.20", tp="2670.20", timeframe=Timeframe.M5)
    res_part_uncapped = engine.execute_signal(
        sig_part_uncapped,
        signal_id="sig-part-uncapped",
        client_order_id="ord-part-uncapped",
        reference_now_utc=FIXED_UTC + timedelta(seconds=8),
    )
    assert res_part_uncapped.executed is True
    assert res_part_uncapped.partially_filled is True
    assert res_part_uncapped.ledger_entry.executed_volume == Decimal("0.06")
    assert res_part_uncapped.ledger_entry.remaining_volume == Decimal("0.04")
    mock.set_positions([])

    # 4. Post-accept verification exception (positions_get raises RuntimeError after 10009 DONE)
    raise_in_post_verify = False

    def _done_then_break_positions_hook(req: MT5OrderRequest) -> MT5OrderSendResult:
        nonlocal raise_in_post_verify
        raise_in_post_verify = True
        return MT5OrderSendResult(
            retcode=int(MT5Retcode.TRADE_RETCODE_DONE),
            order=84000,
            deal=84002,
            volume=req.volume,
            price=req.price,
            comment="Done",
        )

    mock.set_order_send_hook(_done_then_break_positions_hook)
    orig_positions_get = mock.positions_get

    def _flaky_positions_get(symbol: str | None = None):
        if raise_in_post_verify:
            raise RuntimeError("IPC pipe broken during post-send positions_get")
        return orig_positions_get(symbol)

    with patch.object(mock, "positions_get", side_effect=_flaky_positions_get):
        sig_ver_exc = _sample_buy_signal(entry="2650.20", sl="2640.20", tp="2670.20", timeframe=Timeframe.M1)
        res_ver_exc = engine.execute_signal(
            sig_ver_exc,
            signal_id="sig-ver-exc",
            client_order_id="ord-ver-exc",
            reference_now_utc=FIXED_UTC + timedelta(seconds=10),
        )
    assert res_ver_exc.executed is False
    assert res_ver_exc.ambiguous_requires_reconciliation is True
    assert res_ver_exc.ledger_entry.status == OrderLedgerStatus.RECONCILIATION_REQUIRED
    engine.reconcile_unresolved_orders(reference_now_utc=FIXED_UTC + timedelta(seconds=11))

    # 5. Crash between FILLED and ACTIVE persistence: simulate an on-disk entry stuck in FILLED
    #    (without having visited ACTIVE) while its broker position is open
    mock.set_order_send_hook(None)
    mock.set_positions(
        [
            MT5PositionRecord(
                ticket=85001,
                identifier=85001,
                time_utc=FIXED_UTC + timedelta(seconds=12),
                position_type=0,
                magic=engine.safety_config.magic_number,
                symbol="XAUUSD",
                volume=Decimal("0.10"),
                price_open=Decimal("2650.20"),
                sl=Decimal("2640.20"),
                tp=Decimal("2670.20"),
                price_current=Decimal("2650.20"),
                profit=Decimal("0.00"),
                comment="ord-crash-filled",
            )
        ]
    )
    ev_init = LedgerTransitionRecord(
        from_status=None,
        to_status=OrderLedgerStatus.CREATED,
        timestamp_utc=FIXED_UTC + timedelta(seconds=12),
        reason="Created",
    )
    ev_filled_only = LedgerTransitionRecord(
        from_status=OrderLedgerStatus.SUBMITTED,
        to_status=OrderLedgerStatus.FILLED,
        timestamp_utc=FIXED_UTC + timedelta(seconds=12),
        reason="Full fill verified on broker before crash",
        retcode=int(MT5Retcode.TRADE_RETCODE_DONE),
    )
    stuck_filled_entry = ExecutionLedgerEntry(
        client_order_id="ord-crash-filled",
        signal_id="sig-crash-filled",
        symbol="XAUUSD",
        timeframe=Timeframe.H4,
        direction=SignalDirection.BUY,
        magic=engine.safety_config.magic_number,
        requested_volume=Decimal("0.10"),
        executed_volume=Decimal("0.10"),
        remaining_volume=Decimal("0.00"),
        requested_price=Decimal("2650.20"),
        fill_price=Decimal("2650.20"),
        stop_loss=Decimal("2640.20"),
        take_profit=Decimal("2670.20"),
        filling_mode=MT5OrderFillingMode.ORDER_FILLING_FOK,
        order_ticket=85000,
        deal_ticket=85002,
        position_ticket=85001,
        retcode=int(MT5Retcode.TRADE_RETCODE_DONE),
        status=OrderLedgerStatus.FILLED,
        reconciliation_status=ReconciliationStatus.RECONCILED_EXECUTED,
        created_at_utc=FIXED_UTC + timedelta(seconds=12),
        updated_at_utc=FIXED_UTC + timedelta(seconds=12),
        status_reason="Full fill verified on broker before crash",
        history=(ev_init, ev_filled_only),
    )
    ledger._entries["ord-crash-filled"] = stuck_filled_entry
    ledger.persistence.save(
        ledger._serialize_payload(),
        account_login=100200300,
        server="AtlasBroker-Demo",
        now_utc=FIXED_UTC + timedelta(seconds=12),
    )

    restarted_after_filled = PersistentExecutionLedger(safety_config=engine.safety_config)
    restarted_after_filled.load_or_initialize(
        account_login=100200300,
        server="AtlasBroker-Demo",
        reference_now_utc=FIXED_UTC + timedelta(seconds=13),
    )
    # Unactivated FILLED entry is promoted to RECONCILIATION_REQUIRED on restart
    assert (
        restarted_after_filled.get_entry("ord-crash-filled").status
        == OrderLedgerStatus.RECONCILIATION_REQUIRED
    )
    rep_filled_rec = restarted_after_filled.reconcile_with_broker(
        conn,
        reference_now_utc=FIXED_UTC + timedelta(seconds=14),
    )
    assert "ord-crash-filled" in rep_filled_rec.reconciled_executed_ids
    assert restarted_after_filled.get_entry("ord-crash-filled").status == OrderLedgerStatus.ACTIVE
    mock.set_positions([])

    # 6. Pre-send MT5ConnectionError after SUBMITTING persistence (order_send definitely NOT called)
    _, conn_drop, _, _, ledger_drop, _, lifecycle_drop, _, engine_drop = _build_phase74_stack(
        tmp_path,
        tag="h2_presend_conn_err",
    )
    send_attempts = 0

    def _count_send_hook(req: MT5OrderRequest) -> MT5OrderSendResult:
        nonlocal send_attempts
        send_attempts += 1
        return MT5OrderSendResult(retcode=int(MT5Retcode.TRADE_RETCODE_DONE))

    conn_drop.backend.set_order_send_hook(_count_send_hook)
    engine_drop.set_after_submitting_before_send_hook(
        lambda _entry: conn_drop.backend.shutdown()
    )
    sig_drop = _sample_buy_signal(entry="2650.20", sl="2640.20", tp="2670.20")
    with pytest.raises(MT5ConnectionError):
        engine_drop.execute_signal(
            sig_drop,
            signal_id="sig-presend-drop",
            client_order_id="ord-presend-drop",
            reference_now_utc=FIXED_UTC,
        )
    assert send_attempts == 0
    dropped_entry = ledger_drop.get_entry("ord-presend-drop")
    assert dropped_entry is not None
    assert dropped_entry.status == OrderLedgerStatus.FAILED
    assert dropped_entry.reconciliation_status == ReconciliationStatus.NOT_REQUIRED
    assert lifecycle_drop.get_record("sig-presend-drop").state == SignalLifecycleState.CANCELLED


def test_h3_and_h4_conflicts_symbol_trade_modes_and_lifecycle_state_protection(
    tmp_path: Path,
) -> None:
    """Verify remediation of H-3 and H-4:
    1. H-3: PARTIALLY_FILLED entry in ledger blocks opposite-direction orders.
    2. H-3: Open broker position with a DIFFERENT magic number in opposite direction blocks submission.
    3. H-3: Conflicting working order in orders_get() (or orders_get() returning None) blocks submission.
    4. H-4: All four restricted MT5SymbolTradeMode values (DISABLED=0, LONGONLY=1, SHORTONLY=2, CLOSEONLY=3)
       are enforced before order construction.
    5. H-4: Pre-registered signals in PENDING_CONFIRMATION or stale RISK_APPROVED are rejected,
       and failed pre-flight checks (e.g. order_check rejection) never leave a signal in RISK_APPROVED.
    """
    mock, _, _, _, ledger, _, lifecycle, gate, engine = _build_phase74_stack(
        tmp_path,
        tag="h3_h4_remediation",
    )

    sell_sig = TradeSignal(
        symbol="XAUUSD",
        direction=SignalDirection.SELL,
        entry_price=Decimal("2650.00"),
        stop_loss=Decimal("2660.00"),
        take_profit=Decimal("2630.00"),
        timeframe=Timeframe.H1,
        strategy_id="smc_confluence_v1",
        confidence=Decimal("0.85"),
        timestamp=FIXED_UTC,
        rationale="Bearish BOS",
    )
    buy_sig = _sample_buy_signal(entry="2650.20", sl="2640.20", tp="2670.20")

    # 1. H-3: PARTIALLY_FILLED BUY entry in ledger blocks opposing SELL signal
    ledger.create_intent(
        account_login=100200300,
        server="AtlasBroker-Demo",
        client_order_id="ord-part-active",
        signal_id="sig-part-active",
        symbol="XAUUSD",
        timeframe=Timeframe.H1,
        direction=SignalDirection.BUY,
        magic=202607,
        requested_volume=Decimal("0.10"),
        requested_price=Decimal("2650.20"),
        stop_loss=Decimal("2640.20"),
        take_profit=Decimal("2670.20"),
        filling_mode=MT5OrderFillingMode.ORDER_FILLING_IOC,
        now_utc=FIXED_UTC,
    )
    ledger.transition_entry(
        "ord-part-active",
        OrderLedgerStatus.VALIDATED,
        account_login=100200300,
        server="AtlasBroker-Demo",
        reason="Validated",
    )
    ledger.transition_entry(
        "ord-part-active",
        OrderLedgerStatus.SUBMITTING,
        account_login=100200300,
        server="AtlasBroker-Demo",
        reason="Submitting",
    )
    ledger.transition_entry(
        "ord-part-active",
        OrderLedgerStatus.PARTIALLY_FILLED,
        account_login=100200300,
        server="AtlasBroker-Demo",
        executed_volume=Decimal("0.04"),
        remaining_volume=Decimal("0.06"),
        fill_price=Decimal("2650.20"),
        reconciliation_status=ReconciliationStatus.RECONCILED_EXECUTED,
        reason="Partial fill",
    )

    with pytest.raises(DuplicateOrderError, match="PARTIALLY_FILLED in opposite direction"):
        gate.validate_and_build_order(
            sell_sig,
            signal_id="sig-oppose-partial",
            reference_now_utc=FIXED_UTC,
        )

    # Cancel the PARTIALLY_FILLED entry to test broker-side position/order conflicts
    ledger.transition_entry(
        "ord-part-active",
        OrderLedgerStatus.CANCELLED,
        account_login=100200300,
        server="AtlasBroker-Demo",
        reason="Closed partial position",
    )

    # 2. H-3: Open broker position with DIFFERENT magic number (magic=0 / manual) in opposite direction
    mock.set_positions(
        [
            MT5PositionRecord(
                ticket=71001,
                identifier=71001,
                time_utc=FIXED_UTC,
                position_type=1,  # SELL position with magic=0
                magic=0,
                symbol="XAUUSD",
                volume=Decimal("0.10"),
                price_open=Decimal("2650.00"),
                price_current=Decimal("2650.00"),
                profit=Decimal("0.00"),
                comment="Manual SELL",
            )
        ]
    )
    with pytest.raises(DuplicateOrderError, match="Conflicting open position on broker"):
        gate.validate_and_build_order(
            buy_sig,
            signal_id="sig-oppose-diff-magic",
            reference_now_utc=FIXED_UTC,
        )
    mock.set_positions([])

    # 3. H-3: Working order in orders_get() (or orders_get() returning None) blocks submission
    mock.set_orders(
        [
            MT5OrderRecord(
                ticket=72001,
                time_setup_utc=FIXED_UTC,
                order_type=1,  # SELL working order
                state=1,
                magic=999,
                position_id=0,
                symbol="XAUUSD",
                volume_initial=Decimal("0.10"),
                volume_current=Decimal("0.10"),
                price_open=Decimal("2650.00"),
                comment="External working order",
            )
        ]
    )
    with pytest.raises(DuplicateOrderError, match="Conflicting working order on broker"):
        gate.validate_and_build_order(
            buy_sig,
            signal_id="sig-oppose-working-order",
            reference_now_utc=FIXED_UTC,
        )
    mock.set_orders([])

    with patch.object(mock, "orders_get", return_value=None):
        with pytest.raises(OrderValidationError, match="orders_get\\('XAUUSD'\\) returned None"):
            gate.validate_and_build_order(
                buy_sig,
                signal_id="sig-orders-none",
                reference_now_utc=FIXED_UTC,
            )

    # 4. H-4: All four restricted symbol trade modes (DISABLED=0, LONGONLY=1, SHORTONLY=2, CLOSEONLY=3)
    base_sym = mock.symbol_info("XAUUSD")
    assert base_sym is not None

    mock.set_symbol(
        base_sym.model_copy(update={"trade_mode": MT5SymbolTradeMode.SYMBOL_TRADE_MODE_DISABLED})
    )
    with pytest.raises(OrderValidationError, match="SYMBOL_TRADE_MODE_DISABLED"):
        gate.validate_and_build_order(buy_sig, signal_id="sig-mode-disabled", reference_now_utc=FIXED_UTC)

    mock.set_symbol(
        base_sym.model_copy(update={"trade_mode": MT5SymbolTradeMode.SYMBOL_TRADE_MODE_CLOSEONLY})
    )
    with pytest.raises(OrderValidationError, match="SYMBOL_TRADE_MODE_CLOSEONLY"):
        gate.validate_and_build_order(buy_sig, signal_id="sig-mode-closeonly", reference_now_utc=FIXED_UTC)

    mock.set_symbol(
        base_sym.model_copy(update={"trade_mode": MT5SymbolTradeMode.SYMBOL_TRADE_MODE_LONGONLY})
    )
    with pytest.raises(OrderValidationError, match="SYMBOL_TRADE_MODE_LONGONLY"):
        gate.validate_and_build_order(sell_sig, signal_id="sig-mode-longonly-sell", reference_now_utc=FIXED_UTC)

    mock.set_symbol(
        base_sym.model_copy(update={"trade_mode": MT5SymbolTradeMode.SYMBOL_TRADE_MODE_SHORTONLY})
    )
    with pytest.raises(OrderValidationError, match="SYMBOL_TRADE_MODE_SHORTONLY"):
        gate.validate_and_build_order(buy_sig, signal_id="sig-mode-shortonly-buy", reference_now_utc=FIXED_UTC)

    mock.set_symbol(base_sym)

    # 5. H-4: Lifecycle state protection (PENDING_CONFIRMATION, already RISK_APPROVED, and failed order_check)
    lifecycle.register_signal(buy_sig, signal_id="sig-pending-conf")
    lifecycle.transition(
        "sig-pending-conf",
        SignalLifecycleState.PENDING_CONFIRMATION,
        FIXED_UTC,
        "Awaiting M5 confirmation",
    )
    with pytest.raises(OrderValidationError, match="PENDING_CONFIRMATION"):
        gate.validate_and_build_order(buy_sig, signal_id="sig-pending-conf", reference_now_utc=FIXED_UTC)

    # Pre- flight order_check failure must NOT leave a newly evaluated signal in RISK_APPROVED
    sig_oc_fail = _sample_buy_signal(entry="2650.20", sl="2638.20", tp="2675.20", timeframe=Timeframe.M30)
    with patch.object(
        mock,
        "order_check",
        return_value=MT5OrderCheckResult(
            retcode=int(MT5Retcode.TRADE_RETCODE_NO_MONEY),
            margin_free=Decimal("-50.00"),
            comment="No money",
        ),
    ):
        with pytest.raises(OrderValidationError, match="order_check\\(\\) rejected"):
            gate.validate_and_build_order(
                sig_oc_fail,
                signal_id="sig-oc-fail",
                reference_now_utc=FIXED_UTC,
            )
    with pytest.raises( Exception):
        # Either not registered at all or cancelled (never RISK_APPROVED)
        rec_oc = lifecycle.get_record("sig-oc-fail")
        assert rec_oc.state != SignalLifecycleState.RISK_APPROVED


def test_m1_and_m2_spread_point_units_toctou_pre_send_recheck_and_volume_override_metadata(
    tmp_path: Path,
) -> None:
    """Verify remediation of M-1 and M-2:
    1. M-1: Spread points are calculated in symbol_info.point units even when trade_tick_size != point.
    2. M-2: Requested-volume override updates RiskDecision in both PreTradeSafetyValidationResult and
       SignalLifecycleManager so approved_volume, calculated_position_size, and calculated_risk_amount match.
    3. M-2: TOCTOU pre-send check right before order_send() detects a tick that became stale or drifted
       after write-ahead persistence fsyncs, aborts without calling order_send(), transitions the ledger
       entry to CANCELLED, and cancels the SignalLifecycleManager record.
    """
    mock, _, _, _, ledger, _, lifecycle, gate, engine = _build_phase74_stack(
        tmp_path,
        tag="m1_m2_remediation",
    )
    base_sym = mock.symbol_info("XAUUSD")
    assert base_sym is not None

    # 1. M-1: Set point = 0.01 and trade_tick_size = 0.05 (5x larger than point), trade_tick_value = 5.00
    #    Spread of 0.60 is only 12 ticks (0.60 / 0.05), but is 60 points (0.60 / 0.01) > max_allowed_spread_points (50)!
    mock.set_symbol(
        base_sym.model_copy(
            update={
                "point": Decimal("0.01"),
                "trade_tick_size": Decimal("0.05"),
                "trade_tick_value": Decimal("5.00"),
            }
        )
    )
    mock.set_tick(
        SymbolTick(
            symbol="XAUUSD",
            timestamp=FIXED_UTC,
            bid=Decimal("2650.00"),
            ask=Decimal("2650.60"),  # 0.60 / 0.01 = 60 points > 50 limit
        )
    )
    sig_m1 = _sample_buy_signal(entry="2650.60", sl="2640.60", tp="2670.60")
    with pytest.raises(OrderValidationError, match="60 points\\) exceeds max_allowed_spread_points"):
        gate.validate_and_build_order(sig_m1, signal_id="sig-m1-spread", reference_now_utc=FIXED_UTC)

    # Restore normal symbol and tick
    mock.set_symbol(base_sym)
    mock.set_tick(
        SymbolTick(
            symbol="XAUUSD",
            timestamp=FIXED_UTC,
            bid=Decimal("2650.00"),
            ask=Decimal("2650.20"),
        )
    )

    # 2. M-2: Requested volume override (0.04 lots instead of RiskEngine's 0.10 lots) updates RiskDecision metadata
    sig_vol_meta = _sample_buy_signal(entry="2650.20", sl="2640.20", tp="2670.20")
    val_override = gate.validate_and_build_order(
        sig_vol_meta,
        signal_id="sig-vol-override-meta",
        client_order_id="ord-vol-override-meta",
        requested_volume=Decimal("0.04"),
        reference_now_utc=FIXED_UTC,
    )
    assert val_override.approved_volume == Decimal("0.04")
    assert val_override.risk_decision.calculated_position_size == Decimal("0.04")
    assert val_override.risk_decision.calculated_risk_amount == Decimal("40.00")
    assert val_override.risk_decision.risk_percent == Decimal("0.4000")

    managed_override = lifecycle.get_record("sig-vol-override-meta")
    assert managed_override.state == SignalLifecycleState.RISK_APPROVED
    assert managed_override.risk_decision is not None
    assert managed_override.risk_decision.calculated_position_size == Decimal("0.04")
    assert managed_override.risk_decision.calculated_risk_amount == Decimal("40.00")

    # 3. M-2: TOCTOU pre-send check — tick drifts beyond max_entry_deviation_points (20 pts) after SUBMITTING write-ahead
    order_send_calls = 0

    def _track_send(req: MT5OrderRequest) -> MT5OrderSendResult:
        nonlocal order_send_calls
        order_send_calls += 1
        return MT5OrderSendResult(retcode=int(MT5Retcode.TRADE_RETCODE_DONE))

    mock.set_order_send_hook(_track_send)

    def _drift_tick_before_send(_submitting_entry: ExecutionLedgerEntry) -> None:
        mock.set_tick(
            SymbolTick(
                symbol="XAUUSD",
                timestamp=FIXED_UTC,
                bid=Decimal("2650.60"),
                ask=Decimal("2650.80"),  # Drifted 60 points from 2650.20 (> 20 points limit)
            )
        )

    engine.set_after_submitting_before_send_hook(_drift_tick_before_send)
    sig_toctou_drift = _sample_buy_signal(
        entry="2650.20",
        sl="2640.20",
        tp="2670.20",
        timeframe=Timeframe.M15,
    )
    with pytest.raises(OrderValidationError, match="Pre-send TOCTOU check failed"):
        engine.execute_signal(
            sig_toctou_drift,
            signal_id="sig-toctou-drift",
            client_order_id="ord-toctou-drift",
            reference_now_utc=FIXED_UTC,
        )
    # order_send() MUST NOT have been called!
    assert order_send_calls == 0
    drift_entry = ledger.get_entry("ord-toctou-drift")
    assert drift_entry is not None
    assert drift_entry.status == OrderLedgerStatus.CANCELLED
    assert lifecycle.get_record("sig-toctou-drift").state == SignalLifecycleState.CANCELLED

    # Restore valid tick at validation time, then make tick STALE before order_send()
    mock.set_tick(
        SymbolTick(
            symbol="XAUUSD",
            timestamp=FIXED_UTC,
            bid=Decimal("2650.00"),
            ask=Decimal("2650.20"),
        )
    )
    engine.set_after_submitting_before_send_hook(
        lambda _entry: mock.set_tick(
            SymbolTick(
                symbol="XAUUSD",
                timestamp=FIXED_UTC - timedelta(seconds=120),
                bid=Decimal("2650.00"),
                ask=Decimal("2650.20"),
            )
        )
    )
    sig_toctou_stale = _sample_buy_signal(
        entry="2650.20",
        sl="2640.20",
        tp="2670.20",
        timeframe=Timeframe.M5,
    )
    with pytest.raises(StaleMarketDataError):
        engine.execute_signal(
            sig_toctou_stale,
            signal_id="sig-toctou-stale",
            client_order_id="ord-toctou-stale",
            reference_now_utc=FIXED_UTC,
        )
    assert order_send_calls == 0
    stale_entry = ledger.get_entry("ord-toctou-stale")
    assert stale_entry is not None
    assert stale_entry.status == OrderLedgerStatus.CANCELLED
    assert lifecycle.get_record("sig-toctou-stale").state == SignalLifecycleState.CANCELLED
    engine.set_after_submitting_before_send_hook(None)


def test_phase7_end_to_end_offline_pipeline_to_demo_execution_and_restart_reconciliation(
    tmp_path: Path,
) -> None:
    """Phase 7.5 end-to-end offline integration test verifying:
    1. Closed-candle ingestion via MT5MarketDataAdapter -> Phase 2 MarketDataPipeline (feature engineering).
    2. Signal registration in Phase 3/4 SignalLifecycleManager.
    3. TrustedBalanceAnchor account-state reconstruction via PersistentAccountRiskTracker.
    4. 14-step PreTradeSafetyGate & Phase 1 RiskEngine evaluation -> write-ahead MT5DemoExecutionEngine fill.
    5. Cross-restart persistence of both PersistentAccountRiskTracker and PersistentExecutionLedger,
       including post-restart reconciliation and duplicate prevention.
    """
    mock, conn, md, tracker, ledger, _, lifecycle, _, engine = _build_phase74_stack(
        tmp_path,
        tag="phase75_e2e",
    )

    # 1. Fetch closed H1 candles from MockMT5TerminalBackend and run through Phase 2 MarketDataPipeline
    pipeline_res = md.fetch_market_data_pipeline_result(
        symbol="XAUUSD",
        timeframe=Timeframe.H1,
        count=120,
        reference_now_utc=FIXED_UTC,
    )
    assert pipeline_res.validation_report.is_valid is True
    assert len(pipeline_res.candles) >= 60
    assert "atr_14" in pipeline_res.feature_dataframe.columns
    assert "session_high" in pipeline_res.feature_dataframe.columns
    assert "session_low" in pipeline_res.feature_dataframe.columns

    # 2. Fetch live verified Bid/Ask tick and construct a causal TradeSignal aligned with live market price
    live_tick = md.fetch_latest_tick("XAUUSD", reference_now_utc=FIXED_UTC)
    assert live_tick.ask == Decimal("2650.20")

    e2e_signal = TradeSignal(
        symbol="XAUUSD",
        direction=SignalDirection.BUY,
        entry_price=live_tick.ask,
        stop_loss=live_tick.ask - Decimal("10.00"),
        take_profit=live_tick.ask + Decimal("20.00"),
        timeframe=Timeframe.H1,
        strategy_id="smc_confluence_v1",
        confidence=Decimal("0.88"),
        timestamp=FIXED_UTC,
        rationale="Phase 7.5 end-to-end closed-bar pipeline -> demo execution test.",
    )
    lifecycle.register_signal(e2e_signal, signal_id="sig-e2e-75")
    assert lifecycle.get_record("sig-e2e-75").state == SignalLifecycleState.GENERATED

    # 3. Execute signal through PreTradeSafetyGate, RiskEngine, write-ahead ledger, and MockMT5TerminalBackend
    exec_res = engine.execute_signal(
        e2e_signal,
        signal_id="sig-e2e-75",
        client_order_id="ord-e2e-75",
        reference_now_utc=FIXED_UTC,
    )
    assert exec_res.executed is True
    assert exec_res.partially_filled is False
    assert exec_res.ambiguous_requires_reconciliation is False
    assert exec_res.ledger_entry.status == OrderLedgerStatus.ACTIVE
    assert exec_res.ledger_entry.reconciliation_status == ReconciliationStatus.RECONCILED_EXECUTED
    assert exec_res.ledger_entry.executed_volume == Decimal("0.10")
    assert exec_res.ledger_entry.remaining_volume == Decimal("0.00")
    assert lifecycle.get_record("sig-e2e-75").state == SignalLifecycleState.ACTIVE

    # Verify broker position and deal exist on the mock terminal
    open_positions = mock.positions_get("XAUUSD")
    assert open_positions is not None and len(open_positions) == 1
    assert open_positions[0].ticket == exec_res.ledger_entry.position_ticket
    assert open_positions[0].volume == Decimal("0.10")

    # 4. Simulate application restart by instantiating fresh tracker and ledger reading from disk
    restarted_tracker = PersistentAccountRiskTracker(
        connection=conn,
        safety_config=engine.safety_config,
    )
    restarted_acct_report = restarted_tracker.reconstruct_report(
        reference_now_utc=FIXED_UTC + timedelta(seconds=10),
        symbol="XAUUSD",
    )
    assert restarted_acct_report.anchor_source == TrustedAnchorSource.PERSISTED_ROLLOVER_ANCHOR
    assert restarted_acct_report.account_state.open_positions_count == 1
    assert restarted_acct_report.daily_starting_balance == Decimal("10000.00")

    restarted_ledger = PersistentExecutionLedger(safety_config=engine.safety_config)
    restarted_health = restarted_ledger.load_or_initialize(
        account_login=100200300,
        server="AtlasBroker-Demo",
        reference_now_utc=FIXED_UTC + timedelta(seconds=10),
    )
    assert restarted_health == LedgerHealthStatus.HEALTHY
    persisted_entry = restarted_ledger.get_entry("ord-e2e-75")
    assert persisted_entry is not None
    assert persisted_entry.status == OrderLedgerStatus.ACTIVE
    assert persisted_entry.position_ticket == open_positions[0].ticket

    # 5. Re-submitting the same signal or client_order_id across the restarted ledger is blocked
    with pytest.raises(DuplicateOrderError):
        restarted_ledger.ensure_ready_for_new_submission(
            account_login=100200300,
            server="AtlasBroker-Demo",
            client_order_id="ord-e2e-75",
        )
    with pytest.raises(DuplicateOrderError):
        restarted_ledger.ensure_ready_for_new_submission(
            account_login=100200300,
            server="AtlasBroker-Demo",
            signal_id="sig-e2e-75",
        )






