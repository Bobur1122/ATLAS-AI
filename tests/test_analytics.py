"""Focused Phase 8.1 offline unit and integration tests for Post-Trade Analytics:
DemoTradeReconstructor, PersistentTradeJournal, EntryContextSnapshot, PartialExitProgressRecord,
CompletedTradeRecord, DEAL_ENTRY_INOUT / DEAL_ENTRY_OUT_BY fail-closed enforcement,
and inline vs. separate commission/fee/swap cash-flow accounting.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from atlas_ai.analytics import (
    CompletedTradeRecord,
    ContextAttributionStatus,
    DemoTradeReconstructor,
    EntryContextSnapshot,
    KillZoneSession,
    PersistentTradeJournal,
    PostTradeAnalyticsConfig,
    TradeReconstructionError,
    TradeSource,
    build_entry_context_snapshot,
)
from atlas_ai.backtest.models import BacktestTradeRecord, ExitReason
from atlas_ai.domain.market import Timeframe
from atlas_ai.domain.risk import RiskParameters
from atlas_ai.domain.signals import SignalDirection, TradeSignal
from atlas_ai.execution import (
    ExecutionSafetyConfig,
    MT5ConnectionAdapter,
    MT5ConnectionConfig,
    MT5DealEntry,
    MT5DealRecord,
    MT5DealType,
    MT5DemoExecutionEngine,
    MT5MarketDataAdapter,
    MT5PositionRecord,
    MockMT5TerminalBackend,
    OrderLedgerStatus,
    PersistenceAccountMismatchError,
    PersistenceCorruptionError,
    PersistentAccountRiskTracker,
    PersistentExecutionLedger,
    PreTradeSafetyGate,
    ReconciliationStatus,
    TrustedBalanceAnchorConfig,
)
from atlas_ai.risk.engine import RiskEngine
from atlas_ai.strategies.models import (
    ConfluenceResult,
    MarketRegime,
    RiskEvaluationStatus,
    SignalLifecycleState,
    StrategyAnalysisReport,
    StrategyEvaluationResult,
)
from atlas_ai.strategies.signal_lifecycle import SignalLifecycleManager

FIXED_UTC = datetime(2026, 11, 20, 14, 0, 0, tzinfo=timezone.utc)


def _build_phase81_stack(
    tmp_path: Path,
    *,
    tag: str = "p81",
) -> tuple[
    MockMT5TerminalBackend,
    MT5ConnectionAdapter,
    PersistentExecutionLedger,
    SignalLifecycleManager,
    MT5DemoExecutionEngine,
    PersistentTradeJournal,
    DemoTradeReconstructor,
]:
    """Build a complete 100% offline Phase 7 + Phase 8.1 DEMO execution & post-trade stack."""
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
        account_risk_state_path=tmp_path / f"{tag}_account_risk.json",
        ledger_path=tmp_path / f"{tag}_execution_ledger.json",
        trusted_balance_anchor=origin_anchor,
    )
    analytics_cfg = PostTradeAnalyticsConfig(
        journal_path=tmp_path / f"{tag}_trade_journal.json",
    )

    md = MT5MarketDataAdapter(connection=conn, safety_config=safety_cfg)
    tracker = PersistentAccountRiskTracker(connection=conn, safety_config=safety_cfg)
    ledger = PersistentExecutionLedger(safety_config=safety_cfg)
    risk_engine = RiskEngine(risk_parameters=RiskParameters())
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
    journal = PersistentTradeJournal(config=analytics_cfg)
    reconstructor = DemoTradeReconstructor(
        connection=conn,
        ledger=ledger,
        journal=journal,
        lifecycle_manager=lifecycle,
        config=analytics_cfg,
        execution_engine=engine,
    )
    return mock, conn, ledger, lifecycle, engine, journal, reconstructor


def _sample_buy_signal(
    *,
    entry: str = "2650.20",
    sl: str = "2640.20",
    tp: str = "2670.20",
    timeframe: Timeframe = Timeframe.H1,
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
        timestamp=FIXED_UTC,
        rationale="Bullish BOS + discount FVG retest.",
    )


def test_cash_flow_identity_and_inline_vs_separate_commission_fee_swap_attribution(
    tmp_path: Path,
) -> None:
    """Verify:
    1. Consistent cash-flow identity: net_pnl == gross_pnl - commission - fee + swap == sum(d.net_cash_impact)
       without ever subtracting commissions or fees twice.
    2. Separate DEAL_TYPE_COMMISSION deals linked by order and position_id are attributed accurately.
    3. Inline (commission or fee) + separate DEAL_TYPE_COMMISSION double-counting, duplicate separate
       commissions, mixed order/position separate commissions, and conflicting duplicate deal tickets
       fail closed with TradeReconstructionError.
    """
    mock, _, ledger, lifecycle, engine, journal, reconstructor = _build_phase81_stack(
        tmp_path,
        tag="comm_accounting",
    )

    sig = _sample_buy_signal()
    exec_res = engine.execute_signal(
        sig,
        signal_id="sig-comm-1",
        client_order_id="ord-comm-1",
        reference_now_utc=FIXED_UTC,
    )
    assert exec_res.executed is True
    pos_tkt = exec_res.ledger_entry.position_ticket
    ord_tkt = exec_res.ledger_entry.order_ticket
    in_deal_tkt = exec_res.ledger_entry.deal_ticket
    assert pos_tkt is not None and ord_tkt is not None and in_deal_tkt is not None

    # Close position on broker with:
    # - Separate entry commission deal (order=ord_tkt, position_id=pos_tkt): -1.75
    # - Exit DEAL_ENTRY_OUT deal (order=60001, position_id=pos_tkt) at 2670.20 (TP):
    #   profit = +200.00, swap = -1.20, inline commission = 0, fee = 0
    # - Separate exit commission deal (order=60001, position_id=pos_tkt): -1.75
    # Expected: gross_pnl = 200.00, commission = 3.50, fee = 0.00, swap = -1.20 -> net_pnl = 195.30
    sep_comm_in = MT5DealRecord(
        ticket=60002,
        order=ord_tkt,
        position_id=pos_tkt,
        time_utc=FIXED_UTC + timedelta(seconds=1),
        deal_type=int(MT5DealType.DEAL_TYPE_COMMISSION),
        magic=202607,
        symbol="XAUUSD",
        commission=Decimal("-1.75"),
    )
    out_deal = MT5DealRecord(
        ticket=60003,
        order=60001,
        position_id=pos_tkt,
        time_utc=FIXED_UTC + timedelta(minutes=30),
        deal_type=int(MT5DealType.DEAL_TYPE_SELL),
        entry=int(MT5DealEntry.DEAL_ENTRY_OUT),
        magic=202607,
        symbol="XAUUSD",
        volume=Decimal("0.10"),
        price=Decimal("2670.20"),
        profit=Decimal("200.00"),
        commission=Decimal("0.00"),
        fee=Decimal("0.00"),
        swap=Decimal("-1.20"),
        comment="ord-comm-1[tp]",
    )
    sep_comm_out = MT5DealRecord(
        ticket=60004,
        order=60001,
        position_id=pos_tkt,
        time_utc=FIXED_UTC + timedelta(minutes=30, seconds=1),
        deal_type=int(MT5DealType.DEAL_TYPE_COMMISSION),
        magic=202607,
        symbol="XAUUSD",
        commission=Decimal("-1.75"),
    )

    # First test anti-double-counting: if out_deal ALSO has inline fee = -0.50 while sep_comm_out exists for order=60001 -> fail closed!
    base_deals = list(
        mock.history_deals_get(
            datetime(1970, 1, 1, tzinfo=timezone.utc),
            FIXED_UTC + timedelta(days=1),
        )
        or ()
    )
    mock.set_positions([])
    out_deal_with_inline_fee = out_deal.model_copy(update={"fee": Decimal("-0.50")})
    mock.set_deals([*base_deals, sep_comm_in, out_deal_with_inline_fee, sep_comm_out])

    with pytest.raises(TradeReconstructionError, match="Commission double-counting detected"):
        reconstructor.sync_closed_demo_trades(
            reference_now_utc=FIXED_UTC + timedelta(minutes=31)
        )

    # Test duplicate separate commission for the same order=60001 -> fail closed!
    dup_sep_comm = sep_comm_out.model_copy(update={"ticket": 60005})
    mock.set_deals([*base_deals, sep_comm_in, out_deal, sep_comm_out, dup_sep_comm])
    with pytest.raises(TradeReconstructionError, match="Duplicate separate DEAL_TYPE_COMMISSION"):
        reconstructor.sync_closed_demo_trades(
            reference_now_utc=FIXED_UTC + timedelta(minutes=31)
        )

    # Test mixed order-linked (sep_comm_in, order=ord_tkt) + orderless (order=0, position_id=pos_tkt) -> fail closed!
    orderless_sep_comm = sep_comm_out.model_copy(update={"ticket": 60006, "order": 0})
    mock.set_deals([*base_deals, sep_comm_in, out_deal, orderless_sep_comm])
    with pytest.raises(TradeReconstructionError, match="Duplicate or mixed order/position"):
        reconstructor.sync_closed_demo_trades(
            reference_now_utc=FIXED_UTC + timedelta(minutes=31)
        )

    # Now provide valid deals (plus an exact duplicate deal record simulating duplicate MT5 history retrieval after restart)
    mock.set_deals([*base_deals, sep_comm_in, out_deal, sep_comm_out, out_deal])
    report = reconstructor.sync_closed_demo_trades(
        reference_now_utc=FIXED_UTC + timedelta(minutes=31)
    )
    assert len(report.newly_completed_trades) == 1
    tr = report.newly_completed_trades[0]
    assert tr.source == TradeSource.MT5_DEMO
    assert tr.client_order_id == "ord-comm-1"
    assert tr.position_ticket == pos_tkt
    assert tr.gross_pnl == Decimal("200.00")
    assert tr.commission == Decimal("3.50")
    assert tr.fee == Decimal("0.00")
    assert tr.swap == Decimal("-1.20")
    assert tr.net_pnl == Decimal("195.30")
    assert tr.net_pnl == tr.gross_pnl - tr.commission - tr.fee + tr.swap
    assert tr.initial_risk_amount == Decimal("100.00")
    assert tr.realized_r_multiple == Decimal("1.9530")
    assert tr.exit_reason == ExitReason.TAKE_PROFIT
    assert tr.commission_deal_tickets == (60002, 60004)

    # Ledger entry transitioned from ACTIVE -> FILLED, and SignalLifecycleManager transitioned to TARGET_HIT
    updated_ledger_entry = ledger.get_entry("ord-comm-1")
    assert updated_ledger_entry is not None
    assert updated_ledger_entry.status == OrderLedgerStatus.FILLED
    assert updated_ledger_entry.remaining_volume == Decimal("0.00")
    assert lifecycle.get_record("sig-comm-1").state == SignalLifecycleState.TARGET_HIT


def test_deal_entry_inout_and_out_by_fail_closed_policy(tmp_path: Path) -> None:
    """Verify DEAL_ENTRY_INOUT (2) and DEAL_ENTRY_OUT_BY (3) strictly fail closed
    with TradeReconstructionError without mutating the ledger, lifecycle, or journal.
    """
    mock, _, ledger, lifecycle, engine, journal, reconstructor = _build_phase81_stack(
        tmp_path,
        tag="inout_block",
    )

    sig = _sample_buy_signal()
    exec_res = engine.execute_signal(
        sig,
        signal_id="sig-inout-1",
        client_order_id="ord-inout-1",
        reference_now_utc=FIXED_UTC,
    )
    pos_tkt = exec_res.ledger_entry.position_ticket
    assert pos_tkt is not None

    base_deals = list(
        mock.history_deals_get(
            datetime(1970, 1, 1, tzinfo=timezone.utc),
            FIXED_UTC + timedelta(days=1),
        )
        or ()
    )

    for forbidden_entry in (
        int(MT5DealEntry.DEAL_ENTRY_INOUT),
        int(MT5DealEntry.DEAL_ENTRY_OUT_BY),
    ):
        bad_deal = MT5DealRecord(
            ticket=70010 + forbidden_entry,
            order=70000 + forbidden_entry,
            position_id=pos_tkt,
            time_utc=FIXED_UTC + timedelta(minutes=5),
            deal_type=int(MT5DealType.DEAL_TYPE_SELL),
            entry=forbidden_entry,
            magic=202607,
            symbol="XAUUSD",
            volume=Decimal("0.20"),
            price=Decimal("2645.00"),
            profit=Decimal("-52.00"),
            comment="Illegal reversal or close-by",
        )
        mock.set_deals([*base_deals, bad_deal])
        with pytest.raises(TradeReconstructionError, match="Forbidden deal entry type"):
            reconstructor.sync_closed_demo_trades(
                reference_now_utc=FIXED_UTC + timedelta(minutes=6)
            )

        # Verify zero state mutation occurred
        assert ledger.get_entry("ord-inout-1").status == OrderLedgerStatus.ACTIVE
        assert lifecycle.get_record("sig-inout-1").state == SignalLifecycleState.ACTIVE
        assert len(journal.list_completed_trades()) == 0


def test_partial_position_closure_progress_vs_full_closure_lifecycle_and_volume_checks(
    tmp_path: Path,
) -> None:
    """Verify:
    1. Partial position exit (0.04 lots closed out of 0.10, with 0.06 lots still open on broker)
       records PartialExitProgressRecord in PersistentTradeJournal, keeps ledger ACTIVE, and
       NEVER marks SignalLifecycleManager as TARGET_HIT or STOP_LOSS_HIT prematurely.
    2. Volume mismatches (position missing while v_out < v_in, v_open != v_in - v_out, v_out > v_in,
       or position still present when v_out == v_in) fail closed with TradeReconstructionError.
    3. Final exit deal closing the remaining 0.06 lots transitions ledger to FILLED, transitions
       SignalLifecycleManager to TARGET_HIT, removes PartialExitProgressRecord, and writes CompletedTradeRecord.
    """
    mock, _, ledger, lifecycle, engine, journal, reconstructor = _build_phase81_stack(
        tmp_path,
        tag="partial_close",
    )

    sig = _sample_buy_signal()
    exec_res = engine.execute_signal(
        sig,
        signal_id="sig-pclose-1",
        client_order_id="ord-pclose-1",
        reference_now_utc=FIXED_UTC,
    )
    pos_tkt = exec_res.ledger_entry.position_ticket
    assert pos_tkt is not None

    base_deals = list(
        mock.history_deals_get(
            datetime(1970, 1, 1, tzinfo=timezone.utc),
            FIXED_UTC + timedelta(days=1),
        )
        or ()
    )
    open_pos_list = list(mock.positions_get(None) or ())
    assert len(open_pos_list) == 1
    orig_pos = open_pos_list[0]

    # Step 1: Add first partial exit deal of 0.04 lots at 2660.20 (+40.00 gross, -1.00 inline commission)
    partial_out_1 = MT5DealRecord(
        ticket=75001,
        order=75000,
        position_id=pos_tkt,
        time_utc=FIXED_UTC + timedelta(minutes=10),
        deal_type=int(MT5DealType.DEAL_TYPE_SELL),
        entry=int(MT5DealEntry.DEAL_ENTRY_OUT),
        magic=202607,
        symbol="XAUUSD",
        volume=Decimal("0.04"),
        price=Decimal("2660.20"),
        profit=Decimal("40.00"),
        commission=Decimal("-1.00"),
        comment="Partial close 1",
    )
    mock.set_deals([*base_deals, partial_out_1])

    # If broker still reports 0.10 lots open instead of 0.06 lots -> fail closed!
    with pytest.raises(TradeReconstructionError, match="broker open volume 0.10 != expected remaining volume 0.06"):
        reconstructor.sync_closed_demo_trades(
            reference_now_utc=FIXED_UTC + timedelta(minutes=11)
        )

    # If broker reports position completely missing while only 0.04/0.10 exited -> fail closed!
    mock.set_positions([])
    with pytest.raises(TradeReconstructionError, match="missing from positions_get\\(None\\) while remaining volume"):
        reconstructor.sync_closed_demo_trades(
            reference_now_utc=FIXED_UTC + timedelta(minutes=11)
        )

    # Set valid remaining open position volume = 0.06 lots on broker
    mock.set_positions([orig_pos.model_copy(update={"volume": Decimal("0.06")})])
    rep_partial = reconstructor.sync_closed_demo_trades(
        reference_now_utc=FIXED_UTC + timedelta(minutes=11)
    )
    assert len(rep_partial.newly_completed_trades) == 0
    assert len(rep_partial.partial_exit_records) == 1
    assert "ord-pclose-1" in rep_partial.still_open_client_order_ids

    pe = rep_partial.partial_exit_records[0]
    assert pe.initial_entry_volume == Decimal("0.10")
    assert pe.closed_volume_so_far == Decimal("0.04")
    assert pe.remaining_open_volume == Decimal("0.06")
    assert pe.realized_gross_pnl_so_far == Decimal("40.00")
    assert pe.realized_commission_so_far == Decimal("1.00")
    assert pe.realized_net_pnl_so_far == Decimal("39.00")

    # CRITICAL: SignalLifecycleManager and PersistentExecutionLedger MUST remain ACTIVE!
    assert lifecycle.get_record("sig-pclose-1").state == SignalLifecycleState.ACTIVE
    assert ledger.get_entry("ord-pclose-1").status == OrderLedgerStatus.ACTIVE
    assert journal.get_partial_exit_progress("ord-pclose-1") is not None

    # Step 2: Add second exit deal closing the remaining 0.06 lots at 2670.20 (+120.00 gross, -1.50 inline commission)
    partial_out_2 = MT5DealRecord(
        ticket=75003,
        order=75002,
        position_id=pos_tkt,
        time_utc=FIXED_UTC + timedelta(minutes=25),
        deal_type=int(MT5DealType.DEAL_TYPE_SELL),
        entry=int(MT5DealEntry.DEAL_ENTRY_OUT),
        magic=202607,
        symbol="XAUUSD",
        volume=Decimal("0.06"),
        price=Decimal("2670.20"),
        profit=Decimal("120.00"),
        commission=Decimal("-1.50"),
        comment="Final close[tp]",
    )
    mock.set_deals([*base_deals, partial_out_1, partial_out_2])

    # If position still lingers in positions_get() after 100% volume exited -> fail closed!
    with pytest.raises(TradeReconstructionError, match="still appears as an open position"):
        reconstructor.sync_closed_demo_trades(
            reference_now_utc=FIXED_UTC + timedelta(minutes=26)
        )

    # Remove closed position from positions_get() and reconcile full closure
    mock.set_positions([])
    rep_full = reconstructor.sync_closed_demo_trades(
        reference_now_utc=FIXED_UTC + timedelta(minutes=26)
    )
    assert len(rep_full.newly_completed_trades) == 1
    assert len(rep_full.partial_exit_records) == 0
    assert journal.get_partial_exit_progress("ord-pclose-1") is None

    done_tr = rep_full.newly_completed_trades[0]
    assert done_tr.volume == Decimal("0.10")
    # Volume-weighted average exit price: (0.04 * 2660.20 + 0.06 * 2670.20) / 0.10 = 2666.20
    assert done_tr.exit_fill_price == Decimal("2666.20000")
    assert done_tr.gross_pnl == Decimal("160.00")
    assert done_tr.commission == Decimal("2.50")
    assert done_tr.net_pnl == Decimal("157.50")
    assert done_tr.partial_exit_count == 2
    assert done_tr.exit_reason == ExitReason.TAKE_PROFIT

    # Now that full closure is verified, ledger is FILLED and SignalLifecycleManager is TARGET_HIT
    assert ledger.get_entry("ord-pclose-1").status == OrderLedgerStatus.FILLED
    assert lifecycle.get_record("sig-pclose-1").state == SignalLifecycleState.TARGET_HIT


def test_persistent_trade_journal_entry_context_missing_context_recovery_and_crash_safety(
    tmp_path: Path,
) -> None:
    """Verify:
    1. EntryContextSnapshot persistence, cross-restart reload, account binding, and joining with CompletedTradeRecord.
    2. Missing-context recovery (UNATTRIBUTED_MISSING_CONTEXT) when an order executes and closes
       without a persisted EntryContextSnapshot.
    3. Cross-restart idempotency and deal-ticket uniqueness enforcement.
    4. Phase 4 BacktestTradeRecord conversion to CompletedTradeRecord.
    """
    mock, conn, ledger, lifecycle, engine, journal, reconstructor = _build_phase81_stack(
        tmp_path,
        tag="journal_ctx",
    )

    # 1. Trade 1 WITH persisted EntryContextSnapshot
    sig1 = _sample_buy_signal(entry="2650.20", sl="2640.20", tp="2670.20")
    comp_eval = StrategyEvaluationResult(
        strategy_id="market_structure_smc",
        direction=SignalDirection.BUY,
        timestamp=FIXED_UTC,
        confidence=Decimal("0.90"),
        rationale="Bullish BOS",
    )
    conf_res = ConfluenceResult(
        direction=SignalDirection.BUY,
        buy_score=Decimal("0.78"),
        sell_score=Decimal("0.10"),
        wait_weight_ratio=Decimal("0.12"),
        net_directional_edge=Decimal("0.68"),
        final_confidence=Decimal("0.78"),
        directional_agreement_count=3,
        component_evaluations={"market_structure_smc": comp_eval},
        rationale="Strong bullish confluence",
    )
    analysis_rep = StrategyAnalysisReport(
        symbol="XAUUSD",
        timeframe=Timeframe.H1,
        timestamp=FIXED_UTC,
        regime=MarketRegime.UPTREND,
        regime_evaluation=StrategyEvaluationResult(
            strategy_id="regime_detector",
            direction=SignalDirection.BUY,
            timestamp=FIXED_UTC,
            confidence=Decimal("0.85"),
            rationale="Trending bullish",
        ),
        mtf_evaluation=None,
        confluence=conf_res,
        signal=sig1,
        lifecycle_record=None,
        risk_status=RiskEvaluationStatus.RISK_APPROVED,
        risk_decision=None,
    )
    ctx1 = build_entry_context_snapshot(
        analysis_rep,
        client_order_id="ord-ctx-1",
        signal_id="sig-ctx-1",
        approved_volume=Decimal("0.10"),
        initial_risk_amount=Decimal("100.00"),
        expected_spread_points=Decimal("20"),
        captured_at_utc=FIXED_UTC,
    )
    ctx1 = ctx1.model_copy(
        update={
            "killzone_session": KillZoneSession.LONDON_NY_OVERLAP,
            "contributing_knowledge_rule_ids": ("rule-ict-bos-01",),
            "ml_model_id": "logistic_v1",
            "ml_predicted_direction": SignalDirection.BUY,
            "ml_buy_probability": Decimal("0.72"),
            "ml_sell_probability": Decimal("0.13"),
            "ml_wait_probability": Decimal("0.15"),
        }
    )
    journal.record_entry_context(
        ctx1,
        account_login=100200300,
        server="AtlasBroker-Demo",
        now_utc=FIXED_UTC,
    )
    assert journal.last_generation == 1

    # Conflicting context for the same client_order_id must raise TradeReconstructionError
    with pytest.raises(TradeReconstructionError, match="Conflicting EntryContextSnapshot"):
        journal.record_entry_context(
            ctx1.model_copy(update={"initial_risk_amount": Decimal("250.00")}),
            account_login=100200300,
            server="AtlasBroker-Demo",
        )

    exec1 = engine.execute_signal(
        sig1,
        signal_id="sig-ctx-1",
        client_order_id="ord-ctx-1",
        reference_now_utc=FIXED_UTC,
    )
    pos1 = exec1.ledger_entry.position_ticket
    assert pos1 is not None

    # 2. Trade 2 WITHOUT any persisted EntryContextSnapshot (simulating pre-Phase-8 order or crash recovery orphan)
    journal.unbind_execution_engine()
    sig2 = _sample_buy_signal(
        entry="2650.20",
        sl="2640.20",
        tp="2670.20",
        timeframe=Timeframe.M15,
    )
    exec2 = engine.execute_signal(
        sig2,
        signal_id="sig-noctx-2",
        client_order_id="ord-noctx-2",
        reference_now_utc=FIXED_UTC + timedelta(seconds=5),
    )
    pos2 = exec2.ledger_entry.position_ticket
    assert pos2 is not None
    journal.bind_execution_engine(engine)

    # Close Trade 1 at TP (+200.00) and Trade 2 at SL (-102.00 with 20 points adverse exit slippage at 2640.00)
    mock.set_positions([])
    mock.add_deal(
        MT5DealRecord(
            ticket=80001,
            order=80000,
            position_id=pos1,
            time_utc=FIXED_UTC + timedelta(minutes=15),
            deal_type=int(MT5DealType.DEAL_TYPE_SELL),
            entry=int(MT5DealEntry.DEAL_ENTRY_OUT),
            magic=202607,
            symbol="XAUUSD",
            volume=Decimal("0.10"),
            price=Decimal("2670.20"),
            profit=Decimal("200.00"),
            comment="ord-ctx-1[tp]",
        )
    )
    mock.add_deal(
        MT5DealRecord(
            ticket=80003,
            order=80002,
            position_id=pos2,
            time_utc=FIXED_UTC + timedelta(minutes=20),
            deal_type=int(MT5DealType.DEAL_TYPE_SELL),
            entry=int(MT5DealEntry.DEAL_ENTRY_OUT),
            magic=202607,
            symbol="XAUUSD",
            volume=Decimal("0.10"),
            price=Decimal("2640.00"),  # 20 points below stop_loss 2640.20
            profit=Decimal("-102.00"),
            comment="ord-noctx-2[sl]",
        )
    )

    rep = reconstructor.sync_closed_demo_trades(
        reference_now_utc=FIXED_UTC + timedelta(minutes=25)
    )
    assert len(rep.newly_completed_trades) == 2

    tr1 = journal.get_completed_trade_by_client_order_id("ord-ctx-1")
    assert tr1 is not None
    assert tr1.context_attribution_status == ContextAttributionStatus.VERIFIED_ENTRY_CONTEXT
    assert tr1.entry_context is not None
    assert tr1.entry_context.regime == MarketRegime.UPTREND
    assert tr1.entry_context.killzone_session == KillZoneSession.LONDON_NY_OVERLAP
    assert tr1.entry_context.contributing_knowledge_rule_ids == ("rule-ict-bos-01",)

    tr2 = journal.get_completed_trade_by_client_order_id("ord-noctx-2")
    assert tr2 is not None
    assert tr2.context_attribution_status == ContextAttributionStatus.UNATTRIBUTED_MISSING_CONTEXT
    assert tr2.entry_context is None
    assert tr2.net_pnl == Decimal("-102.00")
    assert tr2.initial_risk_amount == Decimal("100.00")
    assert tr2.realized_r_multiple == Decimal("-1.0200")
    assert tr2.exit_reason == ExitReason.STOP_LOSS
    assert tr2.exit_slippage_points == Decimal("20.0000")
    assert lifecycle.get_record("sig-noctx-2").state == SignalLifecycleState.STOP_LOSS_HIT

    # 3. Simulate restart and verify idempotency + account binding
    gen_before = journal.last_generation
    rep_again = reconstructor.sync_closed_demo_trades(
        reference_now_utc=FIXED_UTC + timedelta(minutes=26)
    )
    assert len(rep_again.newly_completed_trades) == 0
    assert set(rep_again.already_journaled_client_order_ids) == {"ord-ctx-1", "ord-noctx-2"}
    assert journal.last_generation == gen_before

    restarted_journal = PersistentTradeJournal(config=journal.config)
    loaded_gen = restarted_journal.load_or_initialize(
        account_login=100200300,
        server="AtlasBroker-Demo",
    )
    assert loaded_gen == gen_before
    assert len(restarted_journal.list_completed_trades()) == 2

    # Account login mismatch must fail closed with PersistenceAccountMismatchError
    wrong_acct_journal = PersistentTradeJournal(config=journal.config)
    with pytest.raises(PersistenceAccountMismatchError):
        wrong_acct_journal.load_or_initialize(
            account_login=999999999,
            server="AtlasBroker-Demo",
        )

    # 4. Verify Phase 4 BacktestTradeRecord conversion to CompletedTradeRecord
    bt_record = BacktestTradeRecord(
        trade_id="bt-tr-1",
        position_id="bt-pos-1",
        signal_id="bt-sig-1",
        symbol="XAUUSD",
        timeframe=Timeframe.H1,
        direction=SignalDirection.BUY,
        strategy_id="smc_confluence_v1",
        entry_bar_index=10,
        exit_bar_index=15,
        entry_timestamp=FIXED_UTC,
        exit_timestamp=FIXED_UTC + timedelta(hours=5),
        requested_entry_price=Decimal("2650.00"),
        entry_fill_price=Decimal("2650.25"),
        requested_exit_price=Decimal("2670.00"),
        exit_fill_price=Decimal("2669.90"),
        stop_loss=Decimal("2640.00"),
        take_profit=Decimal("2670.00"),
        lots=Decimal("0.10"),
        initial_risk_amount=Decimal("100.00"),
        gross_pnl=Decimal("196.50"),
        commission=Decimal("0.70"),
        net_pnl=Decimal("195.80"),
        realized_r_multiple=Decimal("1.9580"),
        mfe_price_distance=Decimal("20.00"),
        mae_price_distance=Decimal("2.00"),
        mfe_usd=Decimal("200.00"),
        mae_usd=Decimal("20.00"),
        bars_held=5,
        exit_reason=ExitReason.TAKE_PROFIT,
    )
    converted_bt = DemoTradeReconstructor.convert_backtest_trade(
        bt_record,
        symbol_point=Decimal("0.01"),
    )
    assert converted_bt.source == TradeSource.BACKTEST
    assert converted_bt.entry_slippage_points == Decimal("25.0000")
    assert converted_bt.exit_slippage_points == Decimal("10.0000")
    assert converted_bt.net_pnl == Decimal("195.80")


def test_phase811_mandatory_pre_send_entry_context_persistence_and_failure_blocking(
    tmp_path: Path,
) -> None:
    """Verify Phase 8.1.1 Mandatory Entry Context Persistence Remediation:
    1. Journal persistence occurs strictly BEFORE mocked backend.order_send() executes
       (verified inside order_send_hook by reloading the journal from disk).
    2. Persistence failure (e.g., corrupted journal file on disk) results in ZERO order_send() calls,
       transitions ledger from SUBMITTING -> FAILED (with has_unresolved_entries() == False),
       and transitions SignalLifecycleManager from RISK_APPROVED -> CANCELLED.
    3. Exact approved volume and initial monetary risk (1R) from PreTradeSafetyGate / RiskDecision
       are persisted (overriding any stale or unapproved pre-staged sizing values).
    4. Persisted EntryContextSnapshot survives full journal reload from disk across restart.
    5. Direct calls to engine.execute_signal() cannot bypass mandatory context persistence
       (both auto-synthesized/rebound mode and strict require_staged_context=True mode).
    6. Existing missing-context recovery (UNATTRIBUTED_MISSING_CONTEXT) remains functional.
    """
    mock, _, ledger, lifecycle, engine, journal, reconstructor = _build_phase81_stack(
        tmp_path,
        tag="p811_presend",
    )

    sig = _sample_buy_signal(entry="2650.20", sl="2640.20", tp="2670.20")
    comp_eval = StrategyEvaluationResult(
        strategy_id="market_structure_smc",
        direction=SignalDirection.BUY,
        timestamp=FIXED_UTC,
        confidence=Decimal("0.92"),
        rationale="Bullish BOS + OB",
    )
    conf_res = ConfluenceResult(
        direction=SignalDirection.BUY,
        buy_score=Decimal("0.81"),
        sell_score=Decimal("0.08"),
        wait_weight_ratio=Decimal("0.11"),
        net_directional_edge=Decimal("0.73"),
        final_confidence=Decimal("0.81"),
        directional_agreement_count=3,
        component_evaluations={"market_structure_smc": comp_eval},
        rationale="High-conviction bullish confluence",
    )
    analysis_rep = StrategyAnalysisReport(
        symbol="XAUUSD",
        timeframe=Timeframe.H1,
        timestamp=FIXED_UTC,
        regime=MarketRegime.UPTREND,
        regime_evaluation=StrategyEvaluationResult(
            strategy_id="regime_detector",
            direction=SignalDirection.BUY,
            timestamp=FIXED_UTC,
            confidence=Decimal("0.88"),
            rationale="Bullish trend regime",
        ),
        mtf_evaluation=None,
        confluence=conf_res,
        signal=sig,
        lifecycle_record=None,
        risk_status=RiskEvaluationStatus.RISK_APPROVED,
        risk_decision=None,
    )

    # Pre-stage an EntryContextSnapshot with STALE / UNAPPROVED sizing (0.50 lots, $500.00 risk)
    # and pass requested_volume=Decimal("0.04") to execute_signal() (which RiskEngine approves at 0.04 lots, $40.00 risk).
    stale_ctx = build_entry_context_snapshot(
        analysis_rep,
        client_order_id="ord-p811-1",
        signal_id="sig-p811-1",
        approved_volume=Decimal("0.50"),
        initial_risk_amount=Decimal("500.00"),
        expected_spread_points=Decimal("20"),
        captured_at_utc=FIXED_UTC,
    ).model_copy(
        update={
            "killzone_session": KillZoneSession.NEW_YORK,
            "contributing_knowledge_rule_ids": ("rule-fvg-02",),
            "ml_model_id": "xgb_v2",
            "ml_predicted_direction": SignalDirection.BUY,
            "ml_buy_probability": Decimal("0.76"),
            "ml_sell_probability": Decimal("0.12"),
            "ml_wait_probability": Decimal("0.12"),
        }
    )
    journal.stage_entry_context(stale_ctx)

    # Install an order_send_hook on MockMT5TerminalBackend to prove the journal is already
    # durably persisted on disk BEFORE order_send() processes the request, and track total order_send() invocations.
    order_send_invocations: int = 0
    orig_order_send = mock.order_send
    disk_context_verified_during_order_send: list[EntryContextSnapshot] = []

    def _tracked_order_send_with_disk_check(req):
        nonlocal order_send_invocations
        order_send_invocations += 1
        if req.comment == "ord-p811-1":
            disk_reader = PersistentTradeJournal(config=journal.config)
            disk_reader.load_or_initialize(
                account_login=100200300,
                server="AtlasBroker-Demo",
            )
            persisted = disk_reader.get_entry_context("ord-p811-1")
            assert persisted is not None, "EntryContextSnapshot MUST be persisted on disk before order_send() executes!"
            disk_context_verified_during_order_send.append(persisted)
        mock.set_order_send_hook(None)
        try:
            return orig_order_send(req)
        finally:
            mock.set_order_send_hook(_tracked_order_send_with_disk_check)

    mock.set_order_send_hook(_tracked_order_send_with_disk_check)

    exec_res_1 = engine.execute_signal(
        sig,
        signal_id="sig-p811-1",
        client_order_id="ord-p811-1",
        requested_volume=Decimal("0.04"),
        reference_now_utc=FIXED_UTC,
    )

    assert exec_res_1.executed is True
    assert order_send_invocations == 1
    assert len(disk_context_verified_during_order_send) == 1

    # Verify (1), (3), and (4): Persisted before order_send, exact approved volume (0.04) and 1R ($40.00),
    # and survives journal reload from disk.
    reloaded_journal = PersistentTradeJournal(config=journal.config)
    reloaded_journal.load_or_initialize(
        account_login=100200300,
        server="AtlasBroker-Demo",
    )
    persisted_1 = reloaded_journal.get_entry_context("ord-p811-1")
    assert persisted_1 is not None
    assert persisted_1 == disk_context_verified_during_order_send[0]
    assert persisted_1.client_order_id == "ord-p811-1"
    assert persisted_1.signal_id == "sig-p811-1"
    assert persisted_1.approved_volume == Decimal("0.04")
    assert persisted_1.initial_risk_amount == Decimal("40.00")
    assert persisted_1.regime == MarketRegime.UPTREND
    assert persisted_1.killzone_session == KillZoneSession.NEW_YORK
    assert persisted_1.contributing_knowledge_rule_ids == ("rule-fvg-02",)
    assert persisted_1.ml_model_id == "xgb_v2"

    # Verify (5): Direct execution of engine.execute_signal() without prior manual record_entry_context()
    # or staging STILL persists a verified EntryContextSnapshot before order_send()!
    sig_direct = _sample_buy_signal(entry="2650.20", sl="2640.20", tp="2670.20", timeframe=Timeframe.M15)
    exec_res_2 = engine.execute_signal(
        sig_direct,
        signal_id="sig-p811-direct",
        client_order_id="ord-p811-direct",
        requested_volume=Decimal("0.05"),
        reference_now_utc=FIXED_UTC + timedelta(seconds=10),
    )
    assert exec_res_2.executed is True
    assert order_send_invocations == 2
    persisted_direct = journal.get_entry_context("ord-p811-direct")
    assert persisted_direct is not None
    assert persisted_direct.client_order_id == "ord-p811-direct"
    assert persisted_direct.signal_id == "sig-p811-direct"
    assert persisted_direct.approved_volume == Decimal("0.05")
    assert persisted_direct.initial_risk_amount == Decimal("50.00")

    # Also verify staging via stage_analysis_report() with auto client_order_id/signal_id generation
    sig_m30 = _sample_buy_signal(entry="2650.20", sl="2640.20", tp="2670.20", timeframe=Timeframe.M30)
    analysis_rep_m30 = analysis_rep.model_copy(
        update={
            "timeframe": Timeframe.M30,
            "signal": sig_m30,
        }
    )
    journal.stage_analysis_report(
        "sig-p811-m30",
        analysis_rep_m30,
        killzone_session=KillZoneSession.LONDON,
        contributing_knowledge_rule_ids=("rule-london-judas",),
    )
    exec_res_3 = engine.execute_signal(
        sig_m30,
        signal_id="sig-p811-m30",
        requested_volume=Decimal("0.02"),
        reference_now_utc=FIXED_UTC + timedelta(seconds=20),
    )
    assert exec_res_3.executed is True
    assert order_send_invocations == 3
    auto_coid = exec_res_3.ledger_entry.client_order_id
    auto_sid = exec_res_3.ledger_entry.signal_id
    persisted_staged_rep = journal.get_entry_context(auto_coid)
    assert persisted_staged_rep is not None
    assert persisted_staged_rep.signal_id == auto_sid
    assert persisted_staged_rep.approved_volume == Decimal("0.02")
    assert persisted_staged_rep.initial_risk_amount == Decimal("20.00")
    assert persisted_staged_rep.killzone_session == KillZoneSession.LONDON
    assert persisted_staged_rep.contributing_knowledge_rule_ids == ("rule-london-judas",)

    # Close position 3 with a valid DEAL_ENTRY_OUT deal (+40.00) and update mock account balance so max_open_positions=3 allows testing the next signals
    pos3_tkt = exec_res_3.ledger_entry.position_ticket
    assert pos3_tkt is not None
    mock.set_positions([p for p in (mock.positions_get(None) or ()) if p.ticket != pos3_tkt])
    mock.add_deal(
        MT5DealRecord(
            ticket=90999,
            order=90998,
            position_id=pos3_tkt,
            time_utc=FIXED_UTC + timedelta(seconds=22),
            deal_type=int(MT5DealType.DEAL_TYPE_SELL),
            entry=int(MT5DealEntry.DEAL_ENTRY_OUT),
            magic=202607,
            symbol="XAUUSD",
            volume=Decimal("0.02"),
            price=Decimal("2670.20"),
            profit=Decimal("40.00"),
            comment=f"{auto_coid}[tp]",
        )
    )
    acct_now = mock.account_info()
    assert acct_now is not None
    mock.set_account(
        acct_now.model_copy(
            update={
                "balance": Decimal("10040.00"),
                "equity": Decimal("10040.00"),
                "margin_free": Decimal("10040.00"),
            }
        )
    )

    # Verify strict require_staged_context=True mode blocks unstaged direct execute_signal() BEFORE order_send()
    calls_before_strict = order_send_invocations
    journal.bind_execution_engine(engine, require_staged_context=True)
    sig_h4 = _sample_buy_signal(entry="2650.20", sl="2640.20", tp="2670.20", timeframe=Timeframe.H4)
    with pytest.raises(TradeReconstructionError, match="require_staged_context=True"):
        engine.execute_signal(
            sig_h4,
            signal_id="sig-p811-unstaged-blocked",
            client_order_id="ord-p811-unstaged-blocked",
            reference_now_utc=FIXED_UTC + timedelta(seconds=25),
        )
    # Zero additional order_send() calls occurred!
    assert order_send_invocations == calls_before_strict
    blocked_entry = ledger.get_entry("ord-p811-unstaged-blocked")
    assert blocked_entry is not None
    assert blocked_entry.status == OrderLedgerStatus.FAILED
    assert blocked_entry.reconciliation_status == ReconciliationStatus.NOT_REQUIRED
    assert blocked_entry.executed_volume == Decimal("0")
    assert ledger.has_unresolved_entries() is False
    assert lifecycle.get_record("sig-p811-unstaged-blocked").state == SignalLifecycleState.CANCELLED

    # Restore default binding (require_staged_context=False) and test (2):
    # Journal persistence failure (corrupted journal file on disk) MUST block order_send() (0 calls),
    # transition ledger from SUBMITTING -> FAILED, and transition lifecycle from RISK_APPROVED -> CANCELLED.
    journal.bind_execution_engine(engine, require_staged_context=False)
    saved_journal_bytes = journal.config.journal_path.read_bytes()
    journal.config.journal_path.write_text("{corrupted_json_before_order_send", encoding="utf-8")

    sig_m5 = _sample_buy_signal(entry="2650.20", sl="2640.20", tp="2670.20", timeframe=Timeframe.M5)
    calls_before_corrupt = order_send_invocations
    with pytest.raises(PersistenceCorruptionError):
        engine.execute_signal(
            sig_m5,
            signal_id="sig-p811-disk-fail",
            client_order_id="ord-p811-disk-fail",
            reference_now_utc=FIXED_UTC + timedelta(seconds=28),
        )
    # CRITICAL: backend.order_send() was NEVER called!
    assert order_send_invocations == calls_before_corrupt
    failed_entry = ledger.get_entry("ord-p811-disk-fail")
    assert failed_entry is not None
    assert failed_entry.status == OrderLedgerStatus.FAILED
    assert failed_entry.reconciliation_status == ReconciliationStatus.NOT_REQUIRED
    assert failed_entry.executed_volume == Decimal("0")
    assert ledger.has_unresolved_entries() is False
    assert lifecycle.get_record("sig-p811-disk-fail").state == SignalLifecycleState.CANCELLED

    # Restore valid journal bytes and verify normal close reconciliation joins the persisted contexts
    journal.config.journal_path.write_bytes(saved_journal_bytes)
    pos1_tkt = exec_res_1.ledger_entry.position_ticket
    assert pos1_tkt is not None
    open_positions = [p for p in (mock.positions_get(None) or ()) if p.ticket != pos1_tkt]
    mock.set_positions(open_positions)
    mock.add_deal(
        MT5DealRecord(
            ticket=91001,
            order=91000,
            position_id=pos1_tkt,
            time_utc=FIXED_UTC + timedelta(minutes=15),
            deal_type=int(MT5DealType.DEAL_TYPE_SELL),
            entry=int(MT5DealEntry.DEAL_ENTRY_OUT),
            magic=202607,
            symbol="XAUUSD",
            volume=Decimal("0.04"),
            price=Decimal("2670.20"),
            profit=Decimal("80.00"),
            comment="ord-p811-1[tp]",
        )
    )
    sync_rep = reconstructor.sync_closed_demo_trades(
        reference_now_utc=FIXED_UTC + timedelta(minutes=16)
    )
    assert len(sync_rep.newly_completed_trades) == 2
    completed_1 = journal.get_completed_trade_by_client_order_id("ord-p811-1")
    assert completed_1 is not None
    assert completed_1.client_order_id == "ord-p811-1"
    assert completed_1.context_attribution_status == ContextAttributionStatus.VERIFIED_ENTRY_CONTEXT
    assert completed_1.entry_context is not None
    assert completed_1.entry_context.approved_volume == Decimal("0.04")
    assert completed_1.entry_context.initial_risk_amount == Decimal("40.00")
    assert completed_1.initial_risk_amount == Decimal("40.00")
    assert completed_1.realized_r_multiple == Decimal("2.0000")

    completed_3 = journal.get_completed_trade_by_client_order_id(auto_coid)
    assert completed_3 is not None
    assert completed_3.context_attribution_status == ContextAttributionStatus.VERIFIED_ENTRY_CONTEXT
    assert completed_3.entry_context is not None
    assert completed_3.entry_context.approved_volume == Decimal("0.02")
    assert completed_3.entry_context.initial_risk_amount == Decimal("20.00")
    assert completed_3.realized_r_multiple == Decimal("2.0000")

