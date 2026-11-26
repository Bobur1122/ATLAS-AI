"""Focused Phase 8.1 offline unit and integration tests for Post-Trade Analytics:
DemoTradeReconstructor, PersistentTradeJournal, EntryContextSnapshot, PartialExitProgressRecord,
CompletedTradeRecord, DEAL_ENTRY_INOUT / DEAL_ENTRY_OUT_BY fail-closed enforcement,
and inline vs. separate commission/fee/swap cash-flow accounting.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError
import pytest

from atlas_ai.analytics import (
    AnalyticsError,
    CompletedTradeRecord,
    ContextAttributionStatus,
    DemoTradeReconstructor,
    DriftAndHealthReport,
    DriftDetectionEngine,
    DriftStatus,
    EntryContextSnapshot,
    ExecutionQualityReport,
    FeedbackLeakageError,
    GovernancePromotionDecision,
    GovernancePromotionEvaluator,
    GovernancePromotionRejectedError,
    KillZoneSession,
    MISSING_CONTEXT_SEGMENT_KEY,
    MULTI_LABEL_DIMENSIONS,
    NO_KNOWLEDGE_RULE_SEGMENT_KEY,
    PerformanceAttributionEngine,
    PerformanceAttributionReport,
    PersistentTradeJournal,
    PostTradeAnalyticsConfig,
    SINGLE_LABEL_DIMENSIONS,
    SegmentPerformanceBreakdown,
    TradeReconstructionError,
    TradeSource,
    build_entry_context_snapshot,
    compute_execution_quality_report,
    compute_max_consecutive_losses,
    compute_overall_backtest_metrics,
    compute_post_trade_calibration_metrics,
    compute_r_drawdown,
    compute_r_sharpe_and_sortino,
    compute_segment_breakdown,
    validate_completed_trade_record,
    validate_risk_parameters_no_relaxation,
    validate_walk_forward_leakage_safety,
)
from atlas_ai.backtest.metrics import calculate_backtest_metrics
from atlas_ai.backtest.models import BacktestTradeRecord, EquityPoint, ExitReason
from atlas_ai.domain.market import Timeframe
from atlas_ai.domain.risk import RiskParameters
from atlas_ai.domain.signals import SignalDirection, TradeSignal
from atlas_ai.models.config import LabelingConfig, WalkForwardConfig
from atlas_ai.models.schemas import (
    ChronologicalSplitIndices,
    ClassClassificationMetrics,
    ModelEvaluationReport,
    ModelType,
    WalkForwardFoldResult,
    WalkForwardReport,
)
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


def _make_phase82_trade(
    *,
    trade_id: str,
    source: TradeSource = TradeSource.MT5_DEMO,
    direction: SignalDirection = SignalDirection.BUY,
    timeframe: Timeframe = Timeframe.H1,
    strategy_id: str = "smc_confluence_v1",
    entry_offset_minutes: int = 0,
    exit_offset_minutes: int = 30,
    requested_entry_price: str = "2650.00",
    entry_fill_price: str = "2650.00",
    requested_exit_price: str = "2670.00",
    exit_fill_price: str = "2670.00",
    stop_loss: str = "2640.00",
    take_profit: str = "2670.00",
    volume: str = "0.10",
    initial_risk_amount: str = "100.00",
    gross_pnl: str = "200.00",
    commission: str = "2.00",
    fee: str = "0.50",
    swap: str = "-0.50",
    partial_exit_count: int = 1,
    exit_reason: ExitReason = ExitReason.TAKE_PROFIT,
    with_context: bool = True,
    regime: MarketRegime = MarketRegime.UPTREND,
    killzone_session: KillZoneSession = KillZoneSession.LONDON_NY_OVERLAP,
    contributing_strategy_ids: tuple[str, ...] = ("market_structure_smc", "fvg_retest"),
    contributing_knowledge_rule_ids: tuple[str, ...] = ("rule-ict-bos-01",),
    symbol_point: Decimal = Decimal("0.01"),
) -> CompletedTradeRecord:
    """Helper to construct a valid Phase 8.2 CompletedTradeRecord with exact slippage and cash-flow identity."""
    req_in = Decimal(requested_entry_price)
    fill_in = Decimal(entry_fill_price)
    req_out = Decimal(requested_exit_price)
    fill_out = Decimal(exit_fill_price)
    sl_dec = Decimal(stop_loss)
    tp_dec = Decimal(take_profit)
    vol_dec = Decimal(volume)
    risk_dec = Decimal(initial_risk_amount)
    gross_dec = Decimal(gross_pnl)
    comm_dec = Decimal(commission)
    fee_dec = Decimal(fee)
    swap_dec = Decimal(swap)
    net_dec = gross_dec - comm_dec - fee_dec + swap_dec
    r_mult = (net_dec / risk_dec).quantize(Decimal("0.0001"))

    if direction == SignalDirection.BUY:
        entry_slip = ((fill_in - req_in) / symbol_point).quantize(Decimal("0.0001"))
        exit_slip = ((req_out - fill_out) / symbol_point).quantize(Decimal("0.0001"))
    else:
        entry_slip = ((req_in - fill_in) / symbol_point).quantize(Decimal("0.0001"))
        exit_slip = ((fill_out - req_out) / symbol_point).quantize(Decimal("0.0001"))

    entry_ts = FIXED_UTC + timedelta(minutes=entry_offset_minutes)
    exit_ts = FIXED_UTC + timedelta(minutes=exit_offset_minutes)
    coid = f"ord-{trade_id}"
    sid = f"sig-{trade_id}"

    if with_context:
        ctx = EntryContextSnapshot(
            client_order_id=coid,
            signal_id=sid,
            symbol="XAUUSD",
            timeframe=timeframe,
            direction=direction,
            strategy_id=strategy_id,
            captured_at_utc=entry_ts,
            regime=regime,
            killzone_session=killzone_session,
            mtf_aligned=True,
            buy_confluence_score=Decimal("0.80") if direction == SignalDirection.BUY else Decimal("0.15"),
            sell_confluence_score=Decimal("0.15") if direction == SignalDirection.BUY else Decimal("0.80"),
            signal_confidence=Decimal("0.82"),
            contributing_strategy_ids=contributing_strategy_ids,
            contributing_knowledge_rule_ids=contributing_knowledge_rule_ids,
            requested_entry_price=req_in,
            stop_loss=sl_dec,
            take_profit=tp_dec,
            expected_spread_points=Decimal("20"),
            approved_volume=vol_dec,
            initial_risk_amount=risk_dec,
        )
        attr_status = ContextAttributionStatus.VERIFIED_ENTRY_CONTEXT
    else:
        ctx = None
        attr_status = ContextAttributionStatus.UNATTRIBUTED_MISSING_CONTEXT

    return CompletedTradeRecord(
        trade_id=trade_id,
        source=source,
        client_order_id=coid,
        signal_id=sid,
        position_ticket=None,
        symbol="XAUUSD",
        timeframe=timeframe,
        direction=direction,
        strategy_id=strategy_id,
        entry_timestamp_utc=entry_ts,
        exit_timestamp_utc=exit_ts,
        requested_entry_price=req_in,
        entry_fill_price=fill_in,
        requested_exit_price=req_out,
        exit_fill_price=fill_out,
        stop_loss=sl_dec,
        take_profit=tp_dec,
        volume=vol_dec,
        entry_slippage_points=entry_slip,
        exit_slippage_points=exit_slip,
        initial_risk_amount=risk_dec,
        gross_pnl=gross_dec,
        commission=comm_dec,
        fee=fee_dec,
        swap=swap_dec,
        net_pnl=net_dec,
        realized_r_multiple=r_mult,
        mfe_usd=max(Decimal("0.00"), gross_dec),
        mae_usd=max(Decimal("0.00"), -gross_dec),
        exit_reason=exit_reason,
        partial_exit_count=partial_exit_count,
        context_attribution_status=attr_status,
        entry_context=ctx,
    )


def test_phase82_slippage_sign_conventions_and_execution_quality_cost_drag() -> None:
    """Verify Phase 8.2 Gate 1 & ExecutionQualityReport:
    1. Exact BUY and SELL entry/exit slippage sign and magnitude conventions
       (positive = adverse slippage, negative = favorable price improvement, zero = exact fill).
    2. Exact partition: adverse + favorable + zero == total_evaluated_trades.
    3. Exact cost drag: total_cost_drag_usd == commission + fee - swap and cost_drag_r_per_trade.
    4. Any sign or magnitude discrepancy on BUY or SELL entry/exit slippage fails closed with AnalyticsError.
    """
    # Empty sequence -> zeroed ExecutionQualityReport
    empty_eq = compute_execution_quality_report(())
    assert empty_eq.total_evaluated_trades == 0
    assert empty_eq.mean_entry_slippage_points == Decimal("0.0000")
    assert empty_eq.max_adverse_entry_slippage_points == Decimal("0.0000")
    assert empty_eq.total_cost_drag_usd == Decimal("0.00")
    assert empty_eq.cost_drag_r_per_trade == Decimal("0.0000")

    # Trade 1 (BUY): requested_entry=2650.00, fill=2650.25 -> +25.0000 pts adverse entry slippage
    #                requested_exit=2670.00, fill=2669.90  -> +10.0000 pts adverse exit slippage
    #                comm=3.00, fee=0.50, swap=-1.50 -> cost_drag = 3.00 + 0.50 - (-1.50) = 5.00 USD (0.0500 R on 100 R)
    t1_buy_adverse = _make_phase82_trade(
        trade_id="tr-eq-1",
        direction=SignalDirection.BUY,
        requested_entry_price="2650.00",
        entry_fill_price="2650.25",
        requested_exit_price="2670.00",
        exit_fill_price="2669.90",
        initial_risk_amount="100.00",
        gross_pnl="196.50",
        commission="3.00",
        fee="0.50",
        swap="-1.50",
        partial_exit_count=2,
    )
    assert t1_buy_adverse.entry_slippage_points == Decimal("25.0000")
    assert t1_buy_adverse.exit_slippage_points == Decimal("10.0000")

    # Trade 2 (SELL): requested_entry=2650.00, fill=2650.10 -> -10.0000 pts FAVORABLE entry slippage (sold higher!)
    #                 requested_exit=2630.00, fill=2630.15  -> +15.0000 pts ADVERSE exit slippage (covered higher!)
    #                 comm=2.00, fee=0.00, swap=+1.00 -> cost_drag = 2.00 + 0.00 - 1.00 = 1.00 USD (0.0200 R on 50 R)
    t2_sell_fav_entry = _make_phase82_trade(
        trade_id="tr-eq-2",
        direction=SignalDirection.SELL,
        entry_offset_minutes=40,
        exit_offset_minutes=80,
        requested_entry_price="2650.00",
        entry_fill_price="2650.10",
        requested_exit_price="2630.00",
        exit_fill_price="2630.15",
        stop_loss="2660.00",
        take_profit="2630.00",
        initial_risk_amount="50.00",
        gross_pnl="99.75",
        commission="2.00",
        fee="0.00",
        swap="1.00",
        partial_exit_count=1,
    )
    assert t2_sell_fav_entry.entry_slippage_points == Decimal("-10.0000")
    assert t2_sell_fav_entry.exit_slippage_points == Decimal("15.0000")

    # Trade 3 (SELL): requested_entry=2650.00, fill=2650.00 -> 0.0000 pts zero entry slippage
    #                 requested_exit=2660.00, fill=2659.95  -> -5.0000 pts favorable exit slippage
    #                 comm=1.50, fee=0.50, swap=-1.00 -> cost_drag = 1.50 + 0.50 - (-1.00) = 3.00 USD (0.0300 R on 100 R)
    t3_sell_zero_entry = _make_phase82_trade(
        trade_id="tr-eq-3",
        direction=SignalDirection.SELL,
        entry_offset_minutes=90,
        exit_offset_minutes=120,
        requested_entry_price="2650.00",
        entry_fill_price="2650.00",
        requested_exit_price="2660.00",
        exit_fill_price="2659.95",
        stop_loss="2660.00",
        take_profit="2630.00",
        initial_risk_amount="100.00",
        gross_pnl="-99.50",
        commission="1.50",
        fee="0.50",
        swap="-1.00",
        exit_reason=ExitReason.STOP_LOSS,
    )
    assert t3_sell_zero_entry.entry_slippage_points == Decimal("0.0000")
    assert t3_sell_zero_entry.exit_slippage_points == Decimal("-5.0000")

    eq = compute_execution_quality_report((t1_buy_adverse, t2_sell_fav_entry, t3_sell_zero_entry))
    assert eq.total_evaluated_trades == 3
    assert eq.adverse_entry_slippage_count == 1
    assert eq.favorable_entry_slippage_count == 1
    assert eq.zero_entry_slippage_count == 1
    # Mean entry slippage: (25.0000 - 10.0000 + 0.0000) / 3 = 5.0000
    assert eq.mean_entry_slippage_points == Decimal("5.0000")
    assert eq.max_adverse_entry_slippage_points == Decimal("25.0000")
    # Mean exit slippage: (10.0000 + 15.0000 - 5.0000) / 3 = 6.6667
    assert eq.mean_exit_slippage_points == Decimal("6.6667")
    assert eq.partial_exit_trade_count == 1
    # Gross profit sum of positive gross_pnl trades: 196.50 + 99.75 = 296.25
    assert eq.total_gross_profit_usd == Decimal("296.25")
    assert eq.total_commission_usd == Decimal("6.50")
    assert eq.total_fee_usd == Decimal("1.00")
    assert eq.total_swap_usd == Decimal("-1.50")
    assert eq.total_cost_drag_usd == Decimal("9.00")
    # Mean cost drag in R: (0.0500 + 0.0200 + 0.0300) / 3 = 0.0333
    assert eq.cost_drag_r_per_trade == Decimal("0.0333")

    # Verify sign or magnitude tampering on entry_slippage_points or exit_slippage_points fails closed!
    bad_buy_sign = t1_buy_adverse.model_copy(update={"entry_slippage_points": Decimal("-25.0000")})
    with pytest.raises(AnalyticsError, match="Entry slippage violation on BUY"):
        validate_completed_trade_record(bad_buy_sign)

    bad_sell_mag = t2_sell_fav_entry.model_copy(update={"exit_slippage_points": Decimal("12.0000")})
    with pytest.raises(AnalyticsError, match="Exit slippage violation on SELL"):
        validate_completed_trade_record(bad_sell_mag)


def test_phase82_pre_aggregation_cash_flow_and_risk_amount_validation_fails_closed() -> None:
    """Verify Phase 8.2 Gate 2 & Issue 2 & Issue 6:
    1. Zero or negative initial_risk_amount (constructed via model_construct) fails closed with AnalyticsError.
    2. Cash-flow identity violation (even 1 cent or unquantized sub-cent residual) fails closed with AnalyticsError.
    3. R-multiple mismatch against quantize(net_pnl / initial_risk_amount, 0.0001) fails closed with AnalyticsError.
    4. Duplicate trade_id in input sequence fails closed with AnalyticsError.
    """
    valid_tr = _make_phase82_trade(trade_id="tr-valid-1")
    validate_completed_trade_record(valid_tr)

    # 1. Zero or negative initial_risk_amount via model_construct
    for bad_risk in (Decimal("0.00"), Decimal("-100.00")):
        bad_risk_tr = CompletedTradeRecord.model_construct(
            **{**valid_tr.__dict__, "initial_risk_amount": bad_risk}
        )
        with pytest.raises(AnalyticsError, match="Invalid non-positive initial_risk_amount"):
            compute_execution_quality_report((bad_risk_tr,))

    # 2. Unquantized sub-cent monetary field (no arbitrary tolerances permitted)
    subcent_tr = CompletedTradeRecord.model_construct(
        **{
            **valid_tr.__dict__,
            "gross_pnl": Decimal("200.004"),
            "net_pnl": Decimal("197.004"),
        }
    )
    with pytest.raises(AnalyticsError, match="Monetary precision violation"):
        validate_completed_trade_record(subcent_tr)

    # 3. 1-cent cash-flow identity discrepancy via model_construct
    bad_cf_tr = CompletedTradeRecord.model_construct(
        **{**valid_tr.__dict__, "net_pnl": valid_tr.net_pnl + Decimal("0.01")}
    )
    with pytest.raises(AnalyticsError, match="Cash-flow identity violation"):
        validate_completed_trade_record(bad_cf_tr)

    # 4. R-multiple mismatch (0.0001 difference)
    bad_r_tr = valid_tr.model_copy(
        update={"realized_r_multiple": valid_tr.realized_r_multiple + Decimal("0.0001")}
    )
    with pytest.raises(AnalyticsError, match="R-multiple consistency violation"):
        validate_completed_trade_record(bad_r_tr)

    # 5. Duplicate trade_id in sequence
    with pytest.raises(AnalyticsError, match="Duplicate trade_id 'tr-valid-1'"):
        compute_execution_quality_report((valid_tr, valid_tr))


def test_phase82_single_label_vs_multi_label_attribution_invariants_and_missing_context() -> None:
    """Verify Phase 8.2 Issue 4 & Multi-Dimensional Attribution:
    1. Single-label dimensions (source, timeframe, direction, strategy_id, regime, killzone_session)
       partition total_trades, winning/losing/breakeven counts, and net_profit_usd additively.
    2. Multi-label dimensions (contributing_strategy_id, knowledge_rule_id) deduplicate tags per trade,
       satisfy 1 <= segment.total_trades <= N, and sum(segment.total_trades) > N when trades carry multiple tags.
    3. Missing-context trades (entry_context=None) are isolated under UNATTRIBUTED_MISSING_CONTEXT,
       and verified-context trades with empty rules are isolated under NO_KNOWLEDGE_RULE.
    """
    # Trade 1: Verified context, UPTREND, LONDON_NY_OVERLAP, 2 contributing strategies + 2 knowledge rules
    t1 = _make_phase82_trade(
        trade_id="tr-attr-1",
        source=TradeSource.MT5_DEMO,
        direction=SignalDirection.BUY,
        timeframe=Timeframe.H1,
        strategy_id="smc_confluence_v1",
        entry_offset_minutes=0,
        exit_offset_minutes=20,
        gross_pnl="200.00",
        commission="2.00",
        fee="0.00",
        swap="0.00",
        regime=MarketRegime.UPTREND,
        killzone_session=KillZoneSession.LONDON_NY_OVERLAP,
        contributing_strategy_ids=("market_structure_smc", "fvg_retest", "market_structure_smc"),  # duplicate tag to verify per-trade dedup
        contributing_knowledge_rule_ids=("rule-bos-01", "rule-fvg-02"),
    )
    # Trade 2: Verified context, RANGING, NEW_YORK, 0 knowledge rules (NO_KNOWLEDGE_RULE)
    t2 = _make_phase82_trade(
        trade_id="tr-attr-2",
        source=TradeSource.MT5_DEMO,
        direction=SignalDirection.SELL,
        timeframe=Timeframe.M15,
        strategy_id="smc_confluence_v1",
        entry_offset_minutes=30,
        exit_offset_minutes=50,
        requested_entry_price="2650.00",
        entry_fill_price="2650.00",
        requested_exit_price="2660.00",
        exit_fill_price="2660.00",
        stop_loss="2660.00",
        take_profit="2630.00",
        gross_pnl="-100.00",
        commission="2.00",
        fee="0.00",
        swap="0.00",
        exit_reason=ExitReason.STOP_LOSS,
        regime=MarketRegime.RANGING,
        killzone_session=KillZoneSession.NEW_YORK,
        contributing_strategy_ids=("liquidity_sweep",),
        contributing_knowledge_rule_ids=(),
    )
    # Trade 3: Missing entry_context (UNATTRIBUTED_MISSING_CONTEXT), BACKTEST source
    t3 = _make_phase82_trade(
        trade_id="tr-attr-3",
        source=TradeSource.BACKTEST,
        direction=SignalDirection.BUY,
        timeframe=Timeframe.H1,
        strategy_id="ema_trend_v1",
        entry_offset_minutes=60,
        exit_offset_minutes=90,
        gross_pnl="150.00",
        commission="1.00",
        fee="0.00",
        swap="0.00",
        with_context=False,
    )

    cfg = PostTradeAnalyticsConfig(min_trades_for_drift_evaluation=2)
    engine = PerformanceAttributionEngine(config=cfg)
    report = engine.analyze_trades((t1, t2, t3), initial_balance=Decimal("10000.00"))

    assert report.overall_metrics.total_trades == 3
    assert report.verified_context_trades == 2
    assert report.unattributed_missing_context_trades == 1
    assert report.overall_metrics.net_profit_usd == Decimal("245.00")  # 198.00 - 102.00 + 149.00

    # Verify single-label additive partition invariant across all 6 single-label dimensions
    for dim_name in SINGLE_LABEL_DIMENSIONS:
        dim_slices = report.list_dimension(dim_name)
        assert sum(s.total_trades for s in dim_slices) == 3
        assert sum((s.net_profit_usd for s in dim_slices), Decimal("0.00")) == Decimal("245.00")

    # Check specific single-label slices
    h1_slice = report.get_segment("timeframe", "H1")
    assert h1_slice is not None
    assert h1_slice.total_trades == 2
    assert h1_slice.insufficient_sample is False  # 2 >= min_trades_for_drift_evaluation (2)

    m15_slice = report.get_segment("timeframe", "M15")
    assert m15_slice is not None
    assert m15_slice.total_trades == 1
    assert m15_slice.insufficient_sample is True  # 1 < 2

    unattr_regime = report.get_segment("regime", MISSING_CONTEXT_SEGMENT_KEY)
    assert unattr_regime is not None
    assert unattr_regime.total_trades == 1
    assert unattr_regime.net_profit_usd == Decimal("149.00")

    # Verify multi-label dimensions: sum of segment trade counts > 3 because t1 has 2 strategies and 2 rules
    for multi_dim in MULTI_LABEL_DIMENSIONS:
        multi_slices = report.list_dimension(multi_dim)
        assert sum(s.total_trades for s in multi_slices) == 4  # 2 tags on t1 + 1 on t2 + 1 on t3
        for s in multi_slices:
            assert 1 <= s.total_trades <= 3

    # Verify duplicate tag "market_structure_smc" on t1 was deduplicated to 1 count
    smc_contrib = report.get_segment("contributing_strategy_id", "market_structure_smc")
    assert smc_contrib is not None
    assert smc_contrib.total_trades == 1

    no_rule_slice = report.get_segment("knowledge_rule_id", NO_KNOWLEDGE_RULE_SEGMENT_KEY)
    assert no_rule_slice is not None
    assert no_rule_slice.total_trades == 1
    assert no_rule_slice.net_profit_usd == Decimal("-102.00")


def test_phase82_phase4_backtest_metrics_exact_compatibility_and_edge_cases() -> None:
    """Verify Phase 8.2 Gate 3, Gate 4 & Issue 5:
    1. Exact 24-field compatibility between Phase 4 calculate_backtest_metrics() and Phase 8.2
       compute_overall_backtest_metrics() when given converted BacktestTradeRecord items and equity_curve.
    2. Statistical edge cases: 0 trades, 1 trade, all winners (gross_loss == 0, downside_dev == 0),
       all losers (gross_profit == 0), and constant identical R-multiples (zero variance s == 0).
    3. Separation of account-return Sharpe/Sortino vs. trade-based R-multiple Sharpe/Sortino.
    """
    bt_trades = (
        BacktestTradeRecord(
            trade_id="bt-compat-1",
            position_id="pos-1",
            signal_id="sig-1",
            symbol="XAUUSD",
            timeframe=Timeframe.H1,
            direction=SignalDirection.BUY,
            strategy_id="smc_v1",
            entry_bar_index=1,
            exit_bar_index=5,
            entry_timestamp=FIXED_UTC,
            exit_timestamp=FIXED_UTC + timedelta(hours=4),
            requested_entry_price=Decimal("2650.00"),
            entry_fill_price=Decimal("2650.10"),
            requested_exit_price=Decimal("2670.00"),
            exit_fill_price=Decimal("2670.00"),
            stop_loss=Decimal("2640.00"),
            take_profit=Decimal("2670.00"),
            lots=Decimal("0.10"),
            initial_risk_amount=Decimal("100.00"),
            gross_pnl=Decimal("199.00"),
            commission=Decimal("1.00"),
            net_pnl=Decimal("198.00"),
            realized_r_multiple=Decimal("1.9800"),
            mfe_price_distance=Decimal("20.00"),
            mae_price_distance=Decimal("1.00"),
            mfe_usd=Decimal("199.00"),
            mae_usd=Decimal("10.00"),
            bars_held=4,
            exit_reason=ExitReason.TAKE_PROFIT,
        ),
        BacktestTradeRecord(
            trade_id="bt-compat-2",
            position_id="pos-2",
            signal_id="sig-2",
            symbol="XAUUSD",
            timeframe=Timeframe.H1,
            direction=SignalDirection.SELL,
            strategy_id="smc_v1",
            entry_bar_index=6,
            exit_bar_index=9,
            entry_timestamp=FIXED_UTC + timedelta(hours=5),
            exit_timestamp=FIXED_UTC + timedelta(hours=8),
            requested_entry_price=Decimal("2670.00"),
            entry_fill_price=Decimal("2669.90"),
            requested_exit_price=Decimal("2680.00"),
            exit_fill_price=Decimal("2680.10"),
            stop_loss=Decimal("2680.00"),
            take_profit=Decimal("2650.00"),
            lots=Decimal("0.10"),
            initial_risk_amount=Decimal("100.00"),
            gross_pnl=Decimal("-102.00"),
            commission=Decimal("1.00"),
            net_pnl=Decimal("-103.00"),
            realized_r_multiple=Decimal("-1.0300"),
            mfe_price_distance=Decimal("2.00"),
            mae_price_distance=Decimal("10.20"),
            mfe_usd=Decimal("20.00"),
            mae_usd=Decimal("102.00"),
            bars_held=3,
            exit_reason=ExitReason.STOP_LOSS,
        ),
        BacktestTradeRecord(
            trade_id="bt-compat-3",
            position_id="pos-3",
            signal_id="sig-3",
            symbol="XAUUSD",
            timeframe=Timeframe.H1,
            direction=SignalDirection.BUY,
            strategy_id="smc_v1",
            entry_bar_index=10,
            exit_bar_index=14,
            entry_timestamp=FIXED_UTC + timedelta(hours=9),
            exit_timestamp=FIXED_UTC + timedelta(hours=13),
            requested_entry_price=Decimal("2660.00"),
            entry_fill_price=Decimal("2660.00"),
            requested_exit_price=Decimal("2675.00"),
            exit_fill_price=Decimal("2675.00"),
            stop_loss=Decimal("2650.00"),
            take_profit=Decimal("2675.00"),
            lots=Decimal("0.10"),
            initial_risk_amount=Decimal("100.00"),
            gross_pnl=Decimal("150.00"),
            commission=Decimal("1.00"),
            net_pnl=Decimal("149.00"),
            realized_r_multiple=Decimal("1.4900"),
            mfe_price_distance=Decimal("15.00"),
            mae_price_distance=Decimal("0.50"),
            mfe_usd=Decimal("150.00"),
            mae_usd=Decimal("5.00"),
            bars_held=4,
            exit_reason=ExitReason.TAKE_PROFIT,
        ),
    )

    eq_curve = (
        EquityPoint(
            bar_index=5,
            timestamp=FIXED_UTC + timedelta(hours=4),
            trading_day="2026-11-20",
            balance=Decimal("10198.00"),
            floating_pnl=Decimal("0.00"),
            equity=Decimal("10198.00"),
            used_margin=Decimal("0.00"),
            free_margin=Decimal("10198.00"),
            open_positions_count=0,
            current_daily_pnl=Decimal("198.00"),
            daily_starting_balance=Decimal("10000.00"),
            peak_balance=Decimal("10198.00"),
            drawdown_usd=Decimal("0.00"),
            drawdown_pct=Decimal("0.0000"),
        ),
        EquityPoint(
            bar_index=9,
            timestamp=FIXED_UTC + timedelta(hours=8),
            trading_day="2026-11-20",
            balance=Decimal("10095.00"),
            floating_pnl=Decimal("0.00"),
            equity=Decimal("10095.00"),
            used_margin=Decimal("0.00"),
            free_margin=Decimal("10095.00"),
            open_positions_count=0,
            current_daily_pnl=Decimal("95.00"),
            daily_starting_balance=Decimal("10000.00"),
            peak_balance=Decimal("10198.00"),
            drawdown_usd=Decimal("103.00"),
            drawdown_pct=Decimal("1.0100"),
        ),
        EquityPoint(
            bar_index=14,
            timestamp=FIXED_UTC + timedelta(hours=13),
            trading_day="2026-11-20",
            balance=Decimal("10244.00"),
            floating_pnl=Decimal("0.00"),
            equity=Decimal("10244.00"),
            used_margin=Decimal("0.00"),
            free_margin=Decimal("10244.00"),
            open_positions_count=0,
            current_daily_pnl=Decimal("244.00"),
            daily_starting_balance=Decimal("10000.00"),
            peak_balance=Decimal("10244.00"),
            drawdown_usd=Decimal("0.00"),
            drawdown_pct=Decimal("0.0000"),
        ),
    )

    p4_metrics = calculate_backtest_metrics(
        initial_balance=Decimal("10000.00"),
        final_balance=Decimal("10244.00"),
        final_equity=Decimal("10244.00"),
        trades=bt_trades,
        equity_curve=eq_curve,
    )

    converted_trades = tuple(
        DemoTradeReconstructor.convert_backtest_trade(bt, symbol_point=Decimal("0.01"))
        for bt in bt_trades
    )
    p82_metrics, max_dd_r = compute_overall_backtest_metrics(
        converted_trades,
        initial_balance=Decimal("10000.00"),
        final_balance=Decimal("10244.00"),
        final_equity=Decimal("10244.00"),
        equity_curve=eq_curve,
    )

    # All 24 BacktestMetrics fields MUST be 100% bit-for-bit identical!
    assert p82_metrics == p4_metrics
    assert max_dd_r == Decimal("1.0300")

    # Also verify trade-only mode (when equity_curve is empty in Phase 4 vs omitted in Phase 8.2)
    p4_no_eq = calculate_backtest_metrics(
        initial_balance=Decimal("10000.00"),
        final_balance=Decimal("10244.00"),
        final_equity=Decimal("10244.00"),
        trades=bt_trades,
        equity_curve=(),
    )
    p82_no_eq, _ = compute_overall_backtest_metrics(
        converted_trades,
        initial_balance=Decimal("10000.00"),
        equity_curve=(),
    )
    assert p82_no_eq == p4_no_eq

    # Edge cases:
    # A) 0 trades
    m_empty, dd_empty = compute_overall_backtest_metrics(())
    assert m_empty.total_trades == 0
    assert m_empty.profit_factor == Decimal("0.0000")
    assert m_empty.sharpe_ratio == Decimal("0.0000")
    assert m_empty.sortino_ratio == Decimal("0.0000")
    assert dd_empty == Decimal("0.0000")

    # B) 1 trade (N < 2 -> Sharpe & Sortino == 0.0000; gross_loss == 0 -> profit_factor == gross_profit)
    m_one, _ = compute_overall_backtest_metrics((converted_trades[0],))
    assert m_one.total_trades == 1
    assert m_one.profit_factor == Decimal("199.0000")
    assert m_one.sharpe_ratio == Decimal("0.0000")
    assert m_one.sortino_ratio == Decimal("0.0000")

    # C) Constant identical R-multiples across 2 winning trades (zero variance s == 0, downside_dev == 0)
    const_1 = _make_phase82_trade(
        trade_id="tr-const-1",
        entry_offset_minutes=0,
        exit_offset_minutes=10,
        gross_pnl="100.00",
        commission="0.00",
        fee="0.00",
        swap="0.00",
    )
    const_2 = _make_phase82_trade(
        trade_id="tr-const-2",
        entry_offset_minutes=20,
        exit_offset_minutes=30,
        gross_pnl="100.00",
        commission="0.00",
        fee="0.00",
        swap="0.00",
    )
    seg_const = compute_segment_breakdown("strategy_id", "smc_confluence_v1", (const_1, const_2))
    assert seg_const.sharpe_ratio == Decimal("0.0000")
    assert seg_const.sortino_ratio == Decimal("0.0000")
    assert seg_const.profit_factor == Decimal("200.0000")


def test_phase82_end_to_end_journal_to_attribution_pipeline(tmp_path: Path) -> None:
    """Verify end-to-end integration from MT5DemoExecutionEngine (with Phase 8.1.1 pre-send hook)
    -> DemoTradeReconstructor.sync_closed_demo_trades() -> PerformanceAttributionEngine.analyze_journal().
    """
    mock, _, _, _, engine, journal, reconstructor = _build_phase81_stack(
        tmp_path,
        tag="p82_e2e",
    )

    sig_h1 = _sample_buy_signal(entry="2650.20", sl="2640.20", tp="2670.20", timeframe=Timeframe.H1)
    comp_eval = StrategyEvaluationResult(
        strategy_id="market_structure_smc",
        direction=SignalDirection.BUY,
        timestamp=FIXED_UTC,
        confidence=Decimal("0.90"),
        rationale="Bullish BOS",
    )
    conf_res = ConfluenceResult(
        direction=SignalDirection.BUY,
        buy_score=Decimal("0.82"),
        sell_score=Decimal("0.08"),
        wait_weight_ratio=Decimal("0.10"),
        net_directional_edge=Decimal("0.74"),
        final_confidence=Decimal("0.82"),
        directional_agreement_count=3,
        component_evaluations={"market_structure_smc": comp_eval},
        rationale="Strong bullish confluence",
    )
    rep_h1 = StrategyAnalysisReport(
        symbol="XAUUSD",
        timeframe=Timeframe.H1,
        timestamp=FIXED_UTC,
        regime=MarketRegime.UPTREND,
        regime_evaluation=StrategyEvaluationResult(
            strategy_id="regime_detector",
            direction=SignalDirection.BUY,
            timestamp=FIXED_UTC,
            confidence=Decimal("0.88"),
            rationale="Bullish trend",
        ),
        mtf_evaluation=None,
        confluence=conf_res,
        signal=sig_h1,
        lifecycle_record=None,
        risk_status=RiskEvaluationStatus.RISK_APPROVED,
        risk_decision=None,
    )
    journal.stage_analysis_report(
        "sig-e2e-1",
        rep_h1,
        killzone_session=KillZoneSession.LONDON_NY_OVERLAP,
        contributing_knowledge_rule_ids=("rule-bos-01", "rule-ob-02"),
    )
    exec_1 = engine.execute_signal(
        sig_h1,
        signal_id="sig-e2e-1",
        client_order_id="ord-e2e-1",
        requested_volume=Decimal("0.10"),
        reference_now_utc=FIXED_UTC,
    )
    assert exec_1.executed is True
    pos_1 = exec_1.ledger_entry.position_ticket
    assert pos_1 is not None

    # Close position 1 on broker at TP (+200.00 gross, -2.00 inline commission)
    mock.set_positions([])
    mock.add_deal(
        MT5DealRecord(
            ticket=95001,
            order=95000,
            position_id=pos_1,
            time_utc=FIXED_UTC + timedelta(minutes=20),
            deal_type=int(MT5DealType.DEAL_TYPE_SELL),
            entry=int(MT5DealEntry.DEAL_ENTRY_OUT),
            magic=202607,
            symbol="XAUUSD",
            volume=Decimal("0.10"),
            price=Decimal("2670.20"),
            profit=Decimal("200.00"),
            commission=Decimal("-2.00"),
            comment="ord-e2e-1[tp]",
        )
    )
    sync_rep = reconstructor.sync_closed_demo_trades(
        reference_now_utc=FIXED_UTC + timedelta(minutes=21)
    )
    assert len(sync_rep.newly_completed_trades) == 1

    attr_engine = PerformanceAttributionEngine(journal=journal)
    attr_report = attr_engine.analyze_journal(
        initial_balance=Decimal("10000.00"),
        source_filter=TradeSource.MT5_DEMO,
    )
    assert attr_report.overall_metrics.total_trades == 1
    assert attr_report.verified_context_trades == 1
    assert attr_report.unattributed_missing_context_trades == 0
    assert attr_report.overall_metrics.net_profit_usd == Decimal("198.00")
    assert attr_report.execution_quality.total_commission_usd == Decimal("2.00")
    assert attr_report.execution_quality.total_cost_drag_usd == Decimal("2.00")
    assert attr_report.execution_quality.cost_drag_r_per_trade == Decimal("0.0200")

    uptrend_seg = attr_report.get_segment("regime", "UPTREND")
    assert uptrend_seg is not None
    assert uptrend_seg.total_trades == 1
    assert uptrend_seg.expectancy_r == Decimal("1.9800")

    bos_rule_seg = attr_report.get_segment("knowledge_rule_id", "rule-bos-01")
    assert bos_rule_seg is not None
    assert bos_rule_seg.total_trades == 1
    assert bos_rule_seg.net_profit_usd == Decimal("198.00")


def _make_mock_model_eval_report(
    *,
    split_name: str = "OOS_TEST",
    total_samples: int = 80,
    active_signals: int = 20,
    brier: str = "0.2200",
    ece: str = "0.0500",
    exp_r: str = "0.4500",
    max_dd_r: str = "2.5000",
) -> ModelEvaluationReport:
    cls_m = ClassClassificationMetrics(
        direction=SignalDirection.BUY,
        support=25,
        predicted_count=20,
        true_positives=14,
        false_positives=6,
        false_negatives=11,
        precision=Decimal("0.7000"),
        recall=Decimal("0.5600"),
        f1_score=Decimal("0.6222"),
    )
    return ModelEvaluationReport(
        split_name=split_name,
        total_samples=total_samples,
        class_distribution={"BUY": 25, "SELL": 25, "WAIT": 30},
        class_imbalance_ratio=Decimal("1.2000"),
        buy_metrics=cls_m,
        sell_metrics=cls_m.model_copy(update={"direction": SignalDirection.SELL}),
        wait_metrics=cls_m.model_copy(update={"direction": SignalDirection.WAIT}),
        macro_f1=Decimal("0.6222"),
        active_directional_signals=active_signals,
        active_directional_precision=Decimal("0.7000"),
        multiclass_brier_score=Decimal(brier),
        mean_winning_probability=Decimal("0.7200"),
        mean_losing_probability=Decimal("0.5400"),
        calibration_error_ece=Decimal(ece),
        expected_r_per_signal=Decimal(exp_r),
        simulated_profit_factor=Decimal("1.8500"),
        simulated_max_drawdown_r=Decimal(max_dd_r),
        simulated_sharpe_ratio=Decimal("2.1000"),
    )


def test_phase83_expectancy_streak_and_execution_slippage_drift_detection() -> None:
    """Verify Phase 8.3 Capability 1 & 2 (Expectancy, Consecutive Loss Streak & Execution Slippage Drift):
    1. observation_window_size <= 0 or bool immediately fails closed with AnalyticsError (preventing [-0:] bug).
    2. INSUFFICIENT_SAMPLE when total_trades < min_trades_for_drift_evaluation (0 trades and 3 < 5 trades)
       suppresses all false-positive drift warnings and rule quarantines.
    3. HEALTHY when total_trades >= min_trades, expectancy_r >= threshold, consecutive_losses < alert,
       and mean_entry_slippage_points <= max_acceptable.
    4. Consecutive loss streak resets to 0 on both winning trades (net_pnl > 0) and breakeven trades (net_pnl == 0).
    5. WARNING_DRIFT_DETECTED when expectancy_r < threshold OR max_consecutive_losses >= alert (individually).
    6. QUARANTINE_RECOMMENDED on severe portfolio degradation when BOTH negative expectancy AND
       consecutive loss alert occur simultaneously.
    7. EXECUTION_DEGRADED when mean_entry_slippage_points > max_acceptable_mean_slippage_points.
    """
    cfg = PostTradeAnalyticsConfig(
        min_trades_for_drift_evaluation=4,
        max_acceptable_mean_slippage_points=Decimal("10.0"),
        min_expectancy_r_threshold=Decimal("0.0000"),
        max_consecutive_losses_alert=3,
    )
    engine = DriftDetectionEngine(config=cfg)

    # 1. Invalid observation_window_size must fail closed with AnalyticsError
    for invalid_win in (0, -1, -5, True, False):
        with pytest.raises(AnalyticsError, match="observation_window_size must be a positive integer"):
            engine.evaluate_trades((), observation_window_size=invalid_win)  # type: ignore[arg-type]

    # 2. Empty trades (0 < 4) -> INSUFFICIENT_SAMPLE
    rep_empty = engine.evaluate_trades(())
    assert rep_empty.overall_status == DriftStatus.INSUFFICIENT_SAMPLE
    assert rep_empty.insufficient_sample is True
    assert rep_empty.total_trades == 0
    assert rep_empty.realized_expectancy_r == Decimal("0.0000")
    assert rep_empty.max_consecutive_losses == 0
    assert rep_empty.warnings == ()
    assert rep_empty.quarantine_recommended_rule_ids == ()

    # 3 trades (< 4) with terrible slippage and all losses -> STILL INSUFFICIENT_SAMPLE (no false positives!)
    small_bad_trades = tuple(
        _make_phase82_trade(
            trade_id=f"tr-small-{i}",
            entry_offset_minutes=i * 10,
            exit_offset_minutes=i * 10 + 5,
            requested_entry_price="2650.00",
            entry_fill_price="2650.50",  # +50 pts slippage
            requested_exit_price="2640.00",
            exit_fill_price="2640.00",
            gross_pnl="-100.00",
            commission="2.00",
            fee="0.00",
            swap="0.00",
            exit_reason=ExitReason.STOP_LOSS,
        )
        for i in range(3)
    )
    rep_small = engine.evaluate_trades(small_bad_trades)
    assert rep_small.overall_status == DriftStatus.INSUFFICIENT_SAMPLE
    assert rep_small.insufficient_sample is True
    assert rep_small.total_trades == 3
    assert rep_small.max_consecutive_losses == 3
    assert rep_small.warnings == ()
    assert rep_small.quarantine_recommended_rule_ids == ()

    # 3. Healthy cohort of 4 trades (3 winners + 1 loser -> positive expectancy, streak=1, slippage=5.0 pts <= 10.0)
    healthy_trades = (
        _make_phase82_trade(
            trade_id="tr-hlth-1",
            entry_offset_minutes=0,
            exit_offset_minutes=10,
            requested_entry_price="2650.00",
            entry_fill_price="2650.05",  # +5 pts
            gross_pnl="200.00",
            commission="2.00",
            fee="0.00",
            swap="0.00",
        ),
        _make_phase82_trade(
            trade_id="tr-hlth-2",
            entry_offset_minutes=20,
            exit_offset_minutes=30,
            requested_entry_price="2650.00",
            entry_fill_price="2650.05",
            requested_exit_price="2640.00",
            exit_fill_price="2640.00",
            gross_pnl="-100.00",
            commission="2.00",
            fee="0.00",
            swap="0.00",
            exit_reason=ExitReason.STOP_LOSS,
        ),
        _make_phase82_trade(
            trade_id="tr-hlth-3",
            entry_offset_minutes=40,
            exit_offset_minutes=50,
            requested_entry_price="2650.00",
            entry_fill_price="2650.05",
            gross_pnl="150.00",
            commission="2.00",
            fee="0.00",
            swap="0.00",
        ),
        _make_phase82_trade(
            trade_id="tr-hlth-4",
            entry_offset_minutes=60,
            exit_offset_minutes=70,
            requested_entry_price="2650.00",
            entry_fill_price="2650.05",
            gross_pnl="150.00",
            commission="2.00",
            fee="0.00",
            swap="0.00",
        ),
    )
    rep_healthy = engine.evaluate_trades(healthy_trades)
    assert rep_healthy.overall_status == DriftStatus.HEALTHY
    assert rep_healthy.insufficient_sample is False
    assert rep_healthy.total_trades == 4
    assert rep_healthy.max_consecutive_losses == 1
    assert rep_healthy.mean_entry_slippage_points == Decimal("5.0000")
    assert rep_healthy.warnings == ()

    # 4. Verify breakeven trade (net_pnl == 0.00) resets consecutive loss streak
    # Sequence: Loss, Loss, Breakeven (net_pnl=0), Loss, Loss -> max_consecutive_losses == 2 (< 3 alert!)
    be_reset_trades = (
        _make_phase82_trade(
            trade_id="tr-be-1",
            entry_offset_minutes=0,
            exit_offset_minutes=5,
            gross_pnl="-50.00",
            commission="0.00",
            fee="0.00",
            swap="0.00",
        ),
        _make_phase82_trade(
            trade_id="tr-be-2",
            entry_offset_minutes=10,
            exit_offset_minutes=15,
            gross_pnl="-50.00",
            commission="0.00",
            fee="0.00",
            swap="0.00",
        ),
        _make_phase82_trade(
            trade_id="tr-be-3",
            entry_offset_minutes=20,
            exit_offset_minutes=25,
            gross_pnl="2.00",
            commission="2.00",
            fee="0.00",
            swap="0.00",  # net_pnl = 0.00 (breakeven!)
        ),
        _make_phase82_trade(
            trade_id="tr-be-4",
            entry_offset_minutes=30,
            exit_offset_minutes=35,
            gross_pnl="-50.00",
            commission="0.00",
            fee="0.00",
            swap="0.00",
        ),
        _make_phase82_trade(
            trade_id="tr-be-5",
            entry_offset_minutes=40,
            exit_offset_minutes=45,
            gross_pnl="300.00",
            commission="0.00",
            fee="0.00",
            swap="0.00",
        ),
    )
    assert compute_max_consecutive_losses(be_reset_trades) == 2

    # 5. Consecutive loss alert (3 losses + 1 large winner -> expectancy > 0, but streak == 3 >= 3)
    # -> WARNING_DRIFT_DETECTED
    streak_warn_trades = (
        _make_phase82_trade(
            trade_id="tr-sw-1",
            entry_offset_minutes=0,
            exit_offset_minutes=5,
            gross_pnl="-50.00",
            commission="0.00",
            fee="0.00",
            swap="0.00",
        ),
        _make_phase82_trade(
            trade_id="tr-sw-2",
            entry_offset_minutes=10,
            exit_offset_minutes=15,
            gross_pnl="-50.00",
            commission="0.00",
            fee="0.00",
            swap="0.00",
        ),
        _make_phase82_trade(
            trade_id="tr-sw-3",
            entry_offset_minutes=20,
            exit_offset_minutes=25,
            gross_pnl="-50.00",
            commission="0.00",
            fee="0.00",
            swap="0.00",
        ),
        _make_phase82_trade(
            trade_id="tr-sw-4",
            entry_offset_minutes=30,
            exit_offset_minutes=35,
            gross_pnl="300.00",
            commission="0.00",
            fee="0.00",
            swap="0.00",
        ),
    )
    rep_streak_warn = engine.evaluate_trades(streak_warn_trades)
    assert rep_streak_warn.overall_status == DriftStatus.WARNING_DRIFT_DETECTED
    assert rep_streak_warn.realized_expectancy_r == Decimal("0.3750")
    assert rep_streak_warn.max_consecutive_losses == 3
    assert any("Consecutive loss alert" in w for w in rep_streak_warn.warnings)

    # 6. Observation window slicing: passing observation_window_size=4 on a 5-trade sequence
    # where the first trade had huge slippage (+100 pts) but the most recent 4 are healthy_trades -> HEALTHY!
    old_outlier = _make_phase82_trade(
        trade_id="tr-old-0",
        entry_offset_minutes=-60,
        exit_offset_minutes=-30,
        requested_entry_price="2650.00",
        entry_fill_price="2651.00",  # +100 pts slippage
        gross_pnl="100.00",
        commission="0.00",
        fee="0.00",
        swap="0.00",
    )
    rep_windowed = engine.evaluate_trades(
        (old_outlier, *healthy_trades),
        observation_window_size=4,
    )
    assert rep_windowed.overall_status == DriftStatus.HEALTHY
    assert rep_windowed.total_trades == 4
    assert rep_windowed.mean_entry_slippage_points == Decimal("5.0000")

    # Without observation_window_size, mean slippage is (100 + 5*4)/5 = 24.0000 > 10.0 -> EXECUTION_DEGRADED!
    rep_exec_deg = engine.evaluate_trades((old_outlier, *healthy_trades))
    assert rep_exec_deg.overall_status == DriftStatus.EXECUTION_DEGRADED
    assert rep_exec_deg.mean_entry_slippage_points == Decimal("24.0000")
    assert any("Execution quality degraded" in w for w in rep_exec_deg.warnings)


def test_phase83_post_trade_ml_calibration_brier_and_ece_drift_detection() -> None:
    """Verify Phase 8.3 Capability 3 & Gate 4 (ML Calibration Brier Score & ECE Drift):
    1. Exact multiclass Brier score and 5-bin ECE computation on ML-attributed closed trades.
    2. Small-sample ML gate (0 < N_ML < min_trades_for_drift_evaluation): realized_brier_score and
       realized_ece are computed, while brier_drift_from_baseline and ece_drift_from_baseline
       are clamped to Decimal("0.0000") with zero false-positive calibration warnings.
    3. Calibration drift warning triggers when N_ML >= min_trades_for_drift_evaluation and
       ece_drift_from_baseline > max_acceptable_ece_drift or brier_drift_from_baseline > max_acceptable_brier_drift.
    4. Partially populated ML probabilities or probabilities not summing to 1.0 +/- 0.0010 fail closed with AnalyticsError.
    """
    cfg = PostTradeAnalyticsConfig(
        min_trades_for_drift_evaluation=4,
        max_acceptable_ece_drift=Decimal("0.1000"),
        max_acceptable_brier_drift=Decimal("0.1500"),
    )
    engine = DriftDetectionEngine(config=cfg)
    baseline_eval = _make_mock_model_eval_report(brier="0.1800", ece="0.0400", exp_r="0.3000")

    def _with_ml_probs(
        tr: CompletedTradeRecord,
        *,
        p_sell: str | None,
        p_wait: str | None,
        p_buy: str | None,
    ) -> CompletedTradeRecord:
        assert tr.entry_context is not None
        updated_ctx = tr.entry_context.model_copy(
            update={
                "ml_model_id": "logistic_v1",
                "ml_predicted_direction": tr.direction,
                "ml_sell_probability": Decimal(p_sell) if p_sell is not None else None,
                "ml_wait_probability": Decimal(p_wait) if p_wait is not None else None,
                "ml_buy_probability": Decimal(p_buy) if p_buy is not None else None,
            }
        )
        return tr.model_copy(update={"entry_context": updated_ctx})

    # Build 4 trades: 3 overconfident losing BUY trades (p_buy=0.90, p_wait=0.05, p_sell=0.05, net_pnl=-50.00)
    # and 1 large winning BUY trade (p_buy=0.90, p_wait=0.05, p_sell=0.05, net_pnl=+300.00 so portfolio expectancy > 0)
    # All 4 fall in bin 5 [0.8, 1.0]: mean conf = 0.9000, mean acc = 1/4 = 0.2500 -> ECE = |0.2500 - 0.9000| = 0.6500!
    # Brier per trade:
    # - Winner (true_cls=2): (0.05-0)^2 + (0.05-0)^2 + (0.90-1)^2 = 0.0025 + 0.0025 + 0.0100 = 0.0150
    # - Loser  (true_cls=1): (0.05-0)^2 + (0.05-1)^2 + (0.90-0)^2 = 0.0025 + 0.9025 + 0.8100 = 1.7150
    # Mean Brier over 4 trades = (0.0150 + 3 * 1.7150) / 4 = 5.1600 / 4 = 1.2900!
    ml_trades: list[CompletedTradeRecord] = []
    for i in range(4):
        is_winner = i == 3
        base_tr = _make_phase82_trade(
            trade_id=f"tr-ml-{i}",
            direction=SignalDirection.BUY,
            entry_offset_minutes=i * 15,
            exit_offset_minutes=i * 15 + 10,
            gross_pnl="300.00" if is_winner else "-50.00",
            commission="0.00",
            fee="0.00",
            swap="0.00",
            exit_reason=ExitReason.TAKE_PROFIT if is_winner else ExitReason.STOP_LOSS,
        )
        ml_trades.append(_with_ml_probs(base_tr, p_sell="0.05", p_wait="0.05", p_buy="0.90"))

    brier_val, ece_val, n_ml = compute_post_trade_calibration_metrics(ml_trades)
    assert n_ml == 4
    assert brier_val == Decimal("1.2900")
    assert ece_val == Decimal("0.6500")

    # Evaluate with N_ML == 4 >= min_trades (4) -> triggers both ECE and Brier calibration drift warnings!
    rep_cal_drift = engine.evaluate_trades(ml_trades, baseline_evaluation=baseline_eval)
    assert rep_cal_drift.overall_status == DriftStatus.WARNING_DRIFT_DETECTED
    assert rep_cal_drift.realized_brier_score == Decimal("1.2900")
    assert rep_cal_drift.realized_ece == Decimal("0.6500")
    assert rep_cal_drift.brier_drift_from_baseline == Decimal("1.1100")  # 1.2900 - 0.1800
    assert rep_cal_drift.ece_drift_from_baseline == Decimal("0.6100")    # 0.6500 - 0.0400
    assert any("ML ECE calibration drift detected" in w for w in rep_cal_drift.warnings)
    assert any("ML Brier calibration drift detected" in w for w in rep_cal_drift.warnings)

    # Test small-sample ML gate (N_total = 4 >= 4, but only N_ML = 2 < 4 have ML probabilities):
    # brier_drift_from_baseline and ece_drift_from_baseline must clamp to 0.0000 with zero calibration alerts!
    non_ml_2 = ml_trades[2].model_copy(
        update={
            "gross_pnl": Decimal("100.00"),
            "net_pnl": Decimal("100.00"),
            "realized_r_multiple": Decimal("1.0000"),
            "entry_context": ml_trades[2].entry_context.model_copy(  # type: ignore[union-attr]
                update={
                    "ml_model_id": None,
                    "ml_predicted_direction": None,
                    "ml_sell_probability": None,
                    "ml_wait_probability": None,
                    "ml_buy_probability": None,
                }
            ),
        }
    )
    non_ml_3 = ml_trades[3].model_copy(
        update={
            "entry_context": ml_trades[3].entry_context.model_copy(  # type: ignore[union-attr]
                update={
                    "ml_model_id": None,
                    "ml_predicted_direction": None,
                    "ml_sell_probability": None,
                    "ml_wait_probability": None,
                    "ml_buy_probability": None,
                }
            )
        }
    )
    rep_small_ml = engine.evaluate_trades(
        (ml_trades[0], ml_trades[1], non_ml_2, non_ml_3),
        baseline_evaluation=baseline_eval,
    )
    assert rep_small_ml.overall_status == DriftStatus.HEALTHY
    assert rep_small_ml.brier_drift_from_baseline == Decimal("0.0000")
    assert rep_small_ml.ece_drift_from_baseline == Decimal("0.0000")
    assert rep_small_ml.warnings == ()

    # Verify partially populated ML probabilities or non-normalized probabilities fail closed!
    partial_prob_tr = _with_ml_probs(ml_trades[0], p_sell="0.10", p_wait=None, p_buy="0.90")
    with pytest.raises(AnalyticsError, match="Partially populated ML probabilities"):
        compute_post_trade_calibration_metrics((partial_prob_tr,))

    unnormalized_prob_tr = _with_ml_probs(ml_trades[0], p_sell="0.20", p_wait="0.20", p_buy="0.80")
    with pytest.raises(AnalyticsError, match="sum to 1.20"):
        compute_post_trade_calibration_metrics((unnormalized_prob_tr,))


def test_phase83_knowledge_rule_quarantine_recommendations_are_advisory_only() -> None:
    """Verify Phase 8.3 Capability 4:
    1. Only knowledge rules with total_trades >= min_trades_for_drift_evaluation AND negative expectancy
       are added to quarantine_recommended_rule_ids (sorted lexicographically).
    2. Rules with total_trades < min_trades_for_drift_evaluation are protected from small-sample quarantine.
    3. Synthetic keys (UNATTRIBUTED_MISSING_CONTEXT and NO_KNOWLEDGE_RULE) are never quarantined.
    4. Quarantine is strictly an advisory recommendation in DriftAndHealthReport.
    """
    cfg = PostTradeAnalyticsConfig(
        min_trades_for_drift_evaluation=3,
        min_expectancy_r_threshold=Decimal("0.0000"),
        max_consecutive_losses_alert=10,
    )
    engine = DriftDetectionEngine(config=cfg)

    # 3 winning trades on "rule-good-01" (+200 USD each)
    good_rule_trades = [
        _make_phase82_trade(
            trade_id=f"tr-rgood-{i}",
            entry_offset_minutes=i * 10,
            exit_offset_minutes=i * 10 + 5,
            gross_pnl="200.00",
            commission="0.00",
            fee="0.00",
            swap="0.00",
            contributing_knowledge_rule_ids=("rule-good-01",),
        )
        for i in range(3)
    ]
    # 3 losing trades on "rule-bad-02" (-50 USD each -> expectancy_r = -0.5000 < 0, N=3 >= 3 -> QUARANTINE!)
    # Also attach "rule-small-03" to only 2 of them (N=2 < 3 -> protected by min_trades gate!)
    bad_rule_trades = [
        _make_phase82_trade(
            trade_id=f"tr-rbad-{i}",
            entry_offset_minutes=40 + i * 10,
            exit_offset_minutes=45 + i * 10,
            gross_pnl="-50.00",
            commission="0.00",
            fee="0.00",
            swap="0.00",
            exit_reason=ExitReason.STOP_LOSS,
            contributing_knowledge_rule_ids=(
                ("rule-bad-02", "rule-small-03") if i < 2 else ("rule-bad-02",)
            ),
        )
        for i in range(3)
    ]
    # 3 losing trades with NO_KNOWLEDGE_RULE (contributing_knowledge_rule_ids=()) -> synthetic key never quarantined
    no_rule_trades = [
        _make_phase82_trade(
            trade_id=f"tr-norule-{i}",
            entry_offset_minutes=80 + i * 10,
            exit_offset_minutes=85 + i * 10,
            gross_pnl="-20.00",
            commission="0.00",
            fee="0.00",
            swap="0.00",
            exit_reason=ExitReason.STOP_LOSS,
            contributing_knowledge_rule_ids=(),
        )
        for i in range(3)
    ]

    all_trades = (*good_rule_trades, *bad_rule_trades, *no_rule_trades)
    report = engine.evaluate_trades(all_trades)

    assert report.overall_status == DriftStatus.QUARANTINE_RECOMMENDED
    assert report.quarantine_recommended_rule_ids == ("rule-bad-02",)
    assert "rule-good-01" not in report.quarantine_recommended_rule_ids
    assert "rule-small-03" not in report.quarantine_recommended_rule_ids
    assert NO_KNOWLEDGE_RULE_SEGMENT_KEY not in report.quarantine_recommended_rule_ids
    assert MISSING_CONTEXT_SEGMENT_KEY not in report.quarantine_recommended_rule_ids


def test_phase83_walk_forward_governance_promotion_leakage_and_risk_ceiling_enforcement() -> None:
    """Verify Phase 8.3 Capability 5, Gate 1 & Gate 2 (Human-Gated Walk-Forward Promotion Governance):
    1. Empty WalkForwardReport folds (len(folds) == 0), purge_bars < horizon_bars, or overlapping
       train/purge/test indices fail closed with FeedbackLeakageError.
    2. Relaxing ANY of the 7 Phase 1 RiskParameters fields (risk_per_trade_percent, max_daily_loss_percent,
       max_total_drawdown_percent, max_open_positions, minimum_risk_reward_ratio, max_position_size,
       allow_trading False -> True) fails closed with GovernancePromotionRejectedError.
    3. Candidate failing out-of-sample expectancy, ECE, Brier, sample count, or drawdown gates returns
       approved_for_human_review=False (or raises GovernancePromotionRejectedError when strict_raise=True).
    4. Candidate passing all gates returns approved_for_human_review=True with requires_human_approval=True.
    5. requires_human_approval=False is impossible to construct (rejected by Pydantic ValidationError).
    """
    cfg = PostTradeAnalyticsConfig(
        min_trades_for_drift_evaluation=10,
        max_acceptable_ece_drift=Decimal("0.0500"),
        max_acceptable_brier_drift=Decimal("0.1000"),
        min_expectancy_r_threshold=Decimal("0.1000"),
    )
    evaluator = GovernancePromotionEvaluator(config=cfg)

    wf_cfg = WalkForwardConfig(
        purge_bars=12,
        labeling=LabelingConfig(horizon_bars=12),
    )
    base_oos = _make_mock_model_eval_report(
        active_signals=20,
        brier="0.2400",
        ece="0.0600",
        exp_r="0.3500",
        max_dd_r="3.0000",
    )
    valid_fold = WalkForwardFoldResult(
        fold_index=0,
        train_start_index=0,
        train_end_index=119,
        purge_start_index=120,
        purge_end_index=131,
        test_start_index=132,
        test_end_index=171,
        model_id="fold-0",
        evaluation=base_oos,
    )
    baseline_wf = WalkForwardReport(
        model_type=ModelType.REGULARIZED_LOGISTIC,
        config=wf_cfg,
        folds=(valid_fold,),
        aggregated_out_of_sample=base_oos,
    )

    # 1. Anti-leakage checks:
    # A) Empty folds sequence -> FeedbackLeakageError
    empty_folds_wf = baseline_wf.model_copy(update={"folds": ()})
    with pytest.raises(FeedbackLeakageError, match="at least one walk-forward fold"):
        validate_walk_forward_leakage_safety(empty_folds_wf)

    # B) Insufficient purge gap in fold (test_start_index - train_end_index - 1 < purge_bars)
    leaky_fold = valid_fold.model_copy(
        update={
            "purge_end_index": 125,
            "test_start_index": 126,  # gap is only 6 bars (< 12 purge_bars)
        }
    )
    leaky_wf = baseline_wf.model_copy(update={"folds": (leaky_fold,)})
    with pytest.raises(FeedbackLeakageError, match="purged gap"):
        evaluator.evaluate_candidate(
            candidate_id="cand-leaky",
            baseline_walk_forward=baseline_wf,
            candidate_walk_forward=leaky_wf,
        )

    # C) Standalone ModelEvaluationReport without split_indices -> FeedbackLeakageError
    with pytest.raises(FeedbackLeakageError, match="requires verified ChronologicalSplitIndices"):
        evaluator.evaluate_candidate(
            candidate_id="cand-no-split",
            baseline_walk_forward=base_oos,
            candidate_walk_forward=base_oos,
            split_indices=None,
        )

    # 2. All 7 Phase 1 RiskParameters ceiling checks:
    base_risk = RiskParameters(
        risk_per_trade_percent=Decimal("1.0"),
        max_daily_loss_percent=Decimal("3.0"),
        max_total_drawdown_percent=Decimal("10.0"),
        max_open_positions=3,
        minimum_risk_reward_ratio=Decimal("1.5"),
        max_position_size=Decimal("5.0"),
        allow_trading=False,
    )
    relaxed_cases = (
        (base_risk.model_copy(update={"risk_per_trade_percent": Decimal("1.5")}), "risk_per_trade_percent"),
        (base_risk.model_copy(update={"max_daily_loss_percent": Decimal("4.0")}), "max_daily_loss_percent"),
        (base_risk.model_copy(update={"max_total_drawdown_percent": Decimal("12.0")}), "max_total_drawdown_percent"),
        (base_risk.model_copy(update={"max_open_positions": 4}), "max_open_positions"),
        (base_risk.model_copy(update={"minimum_risk_reward_ratio": Decimal("1.2")}), "minimum_risk_reward_ratio"),
        (base_risk.model_copy(update={"max_position_size": Decimal("6.0")}), "max_position_size"),
        (base_risk.model_copy(update={"allow_trading": True}), "allow_trading"),
    )
    for bad_cand_risk, expected_field in relaxed_cases:
        with pytest.raises(GovernancePromotionRejectedError, match=expected_field):
            validate_risk_parameters_no_relaxation(
                baseline=base_risk,
                candidate=bad_cand_risk,
            )

    # 3. Candidate failing out-of-sample gates (lower expectancy, worse ECE, higher drawdown)
    inferior_oos = _make_mock_model_eval_report(
        active_signals=8,      # < 10
        brier="0.4000",        # > 0.2400 + 0.1000
        ece="0.1500",          # > 0.0600 + 0.0500
        exp_r="0.2000",        # <= 0.3500
        max_dd_r="4.5000",     # > 3.0000
    )
    inferior_wf = baseline_wf.model_copy(
        update={
            "folds": (valid_fold.model_copy(update={"evaluation": inferior_oos}),),
            "aggregated_out_of_sample": inferior_oos,
        }
    )
    rejected_decision = evaluator.evaluate_candidate(
        candidate_id="cand-inferior-v2",
        baseline_walk_forward=baseline_wf,
        candidate_walk_forward=inferior_wf,
        strict_raise=False,
    )
    assert rejected_decision.approved_for_human_review is False
    assert rejected_decision.requires_human_approval is True
    assert len(rejected_decision.rejection_reasons) == 5

    with pytest.raises(GovernancePromotionRejectedError, match="rejected by out-of-sample"):
        evaluator.evaluate_candidate(
            candidate_id="cand-inferior-v2",
            baseline_walk_forward=baseline_wf,
            candidate_walk_forward=inferior_wf,
            strict_raise=True,
        )

    # 4. Superior candidate passing all 5 gates and tightening risk parameters -> approved_for_human_review=True
    superior_oos = _make_mock_model_eval_report(
        active_signals=24,
        brier="0.2100",
        ece="0.0450",
        exp_r="0.5200",
        max_dd_r="2.2000",
    )
    superior_wf = baseline_wf.model_copy(
        update={
            "folds": (valid_fold.model_copy(update={"evaluation": superior_oos}),),
            "aggregated_out_of_sample": superior_oos,
        }
    )
    tighter_risk = base_risk.model_copy(update={"risk_per_trade_percent": Decimal("0.75")})
    approved_decision = evaluator.evaluate_candidate(
        candidate_id="cand-gbdt-v3",
        baseline_walk_forward=baseline_wf,
        candidate_walk_forward=superior_wf,
        baseline_risk_parameters=base_risk,
        candidate_risk_parameters=tighter_risk,
        strict_raise=True,
    )
    assert approved_decision.approved_for_human_review is True
    assert approved_decision.requires_human_approval is True
    assert approved_decision.rejection_reasons == ()
    assert approved_decision.purge_bars == 12
    assert approved_decision.horizon_bars == 12

    # 5. Attempting to construct GovernancePromotionDecision with requires_human_approval=False is blocked by schema
    with pytest.raises(ValidationError):
        GovernancePromotionDecision(
            candidate_id="cand-illegal-auto",
            approved_for_human_review=True,
            requires_human_approval=False,  # type: ignore[arg-type]
            baseline_expectancy_r=Decimal("0.3500"),
            candidate_expectancy_r=Decimal("0.5200"),
            baseline_ece=Decimal("0.0600"),
            candidate_ece=Decimal("0.0450"),
            candidate_max_drawdown_r=Decimal("2.2000"),
            purge_bars=12,
            horizon_bars=12,
            rejection_reasons=(),
            rationale="Illegal auto-promotion attempt",
        )


def test_phase83_remediation_m1_m2_m3_l1_l2_and_t1_t2_t3_coverage(tmp_path: Path) -> None:
    """Verify Phase 8.3 Audit Remediation Findings (M-1, M-2, M-3, L-1, L-2) and Test Coverage Gaps (T-1, T-2, T-3):
    1. T-1 & L-1: SELL trade ML calibration (win -> class 0, loss/breakeven -> class 1), multi-bin ECE distribution
       across bins [0.4, 0.6), [0.6, 0.8), [0.8, 1.0], and rejection of ml_predicted_direction='WAIT',
       opposing ml_predicted_direction, or ml_predicted_direction set when probabilities are None.
    2. T-2: Combined negative expectancy + consecutive-loss streak degradation producing QUARANTINE_RECOMMENDED
       even when quarantine_recommended_rule_ids is empty, QUARANTINE_RECOMMENDED severity precedence over
       EXECUTION_DEGRADED, and DriftDetectionEngine.evaluate_journal().
    3. M-1, M-2, M-3 & T-3:
       - M-1: Passing only baseline_risk_parameters or only candidate_risk_parameters raises AnalyticsError.
       - M-2 & T-3: Two-fold WalkForwardReport with overlapping test windows (fold 1 test_start_index <= fold 0 test_end_index)
         or non-advancing train_end_index / regressing train_start_index raises FeedbackLeakageError.
       - M-3: Standalone baseline ModelEvaluationReport without split_indices raises FeedbackLeakageError;
         conflicting split_indices.(purge_bars, horizon_bars) vs WalkForwardReport raises FeedbackLeakageError.
       - L-2 & T-3: Non-finite Decimal (NaN / Infinity) or invalid status/warning combinations on
         DriftAndHealthReport and GovernancePromotionDecision raise Pydantic ValidationError.
    """
    # --- 1. T-1 & L-1: SELL calibration + Multi-Bin ECE + ml_predicted_direction consistency ---
    # 4 trades spanning 3 distinct ECE bins:
    # - Trade A (SELL win, net_pnl=+100):  p_sell=0.50, p_wait=0.30, p_buy=0.20 -> bin 3 [0.4, 0.6), conf=0.50, acc=1.0,
    #   true_cls=0 -> Brier = (0.50-1)^2 + (0.30-0)^2 + (0.20-0)^2 = 0.25 + 0.09 + 0.04 = 0.3800
    # - Trade B (SELL loss, net_pnl=-50):  p_sell=0.70, p_wait=0.20, p_buy=0.10 -> bin 4 [0.6, 0.8), conf=0.70, acc=0.0,
    #   true_cls=1 -> Brier = (0.70-0)^2 + (0.20-1)^2 + (0.10-0)^2 = 0.49 + 0.64 + 0.01 = 1.1400
    # - Trade C (BUY win, net_pnl=+150):   p_sell=0.10, p_wait=0.20, p_buy=0.70 -> bin 4 [0.6, 0.8), conf=0.70, acc=1.0,
    #   true_cls=2 -> Brier = (0.10-0)^2 + (0.20-0)^2 + (0.70-1)^2 = 0.01 + 0.04 + 0.09 = 0.1400
    # - Trade D (SELL breakeven, net_pnl=0): p_sell=0.85, p_wait=0.10, p_buy=0.05 -> bin 5 [0.8, 1.0], conf=0.85, acc=0.0,
    #   true_cls=1 -> Brier = (0.85-0)^2 + (0.10-1)^2 + (0.05-0)^2 = 0.7225 + 0.8100 + 0.0025 = 1.5350
    # Expected Mean Brier = (0.3800 + 1.1400 + 0.1400 + 1.5350) / 4 = 3.1950 / 4 = 0.7988 (0.79875 -> 0.7988 ROUND_HALF_UP)
    # Expected Multi-Bin ECE:
    # - Bin 3 [0.4, 0.6): 1 trade (weight 0.25), |acc(1.0) - conf(0.50)| = 0.50 -> 0.25 * 0.50 = 0.1250
    # - Bin 4 [0.6, 0.8): 2 trades (weight 0.50), mean_acc=0.50, mean_conf=0.70, |0.50 - 0.70| = 0.20 -> 0.50 * 0.20 = 0.1000
    # - Bin 5 [0.8, 1.0]: 1 trade (weight 0.25), |acc(0.0) - conf(0.85)| = 0.85 -> 0.25 * 0.85 = 0.2125
    # Total ECE = 0.1250 + 0.1000 + 0.2125 = 0.4375!
    def _attach_ml(
        tr: CompletedTradeRecord,
        *,
        pred_dir: SignalDirection | None,
        p_sell: str | None,
        p_wait: str | None,
        p_buy: str | None,
    ) -> CompletedTradeRecord:
        assert tr.entry_context is not None
        ctx = tr.entry_context.model_copy(
            update={
                "ml_model_id": "gbdt_v1",
                "ml_predicted_direction": pred_dir,
                "ml_sell_probability": Decimal(p_sell) if p_sell is not None else None,
                "ml_wait_probability": Decimal(p_wait) if p_wait is not None else None,
                "ml_buy_probability": Decimal(p_buy) if p_buy is not None else None,
            }
        )
        return tr.model_copy(update={"entry_context": ctx})

    tr_a = _attach_ml(
        _make_phase82_trade(
            trade_id="tr-mbin-a",
            direction=SignalDirection.SELL,
            entry_offset_minutes=0,
            exit_offset_minutes=10,
            requested_entry_price="2650.00",
            entry_fill_price="2650.00",
            requested_exit_price="2640.00",
            exit_fill_price="2640.00",
            stop_loss="2660.00",
            take_profit="2640.00",
            gross_pnl="100.00",
            commission="0.00",
            fee="0.00",
            swap="0.00",
        ),
        pred_dir=SignalDirection.SELL,
        p_sell="0.50",
        p_wait="0.30",
        p_buy="0.20",
    )
    tr_b = _attach_ml(
        _make_phase82_trade(
            trade_id="tr-mbin-b",
            direction=SignalDirection.SELL,
            entry_offset_minutes=15,
            exit_offset_minutes=25,
            requested_entry_price="2650.00",
            entry_fill_price="2650.00",
            requested_exit_price="2660.00",
            exit_fill_price="2660.00",
            stop_loss="2660.00",
            take_profit="2640.00",
            gross_pnl="-50.00",
            commission="0.00",
            fee="0.00",
            swap="0.00",
            exit_reason=ExitReason.STOP_LOSS,
        ),
        pred_dir=SignalDirection.SELL,
        p_sell="0.70",
        p_wait="0.20",
        p_buy="0.10",
    )
    tr_c = _attach_ml(
        _make_phase82_trade(
            trade_id="tr-mbin-c",
            direction=SignalDirection.BUY,
            entry_offset_minutes=30,
            exit_offset_minutes=40,
            gross_pnl="150.00",
            commission="0.00",
            fee="0.00",
            swap="0.00",
        ),
        pred_dir=SignalDirection.BUY,
        p_sell="0.10",
        p_wait="0.20",
        p_buy="0.70",
    )
    tr_d = _attach_ml(
        _make_phase82_trade(
            trade_id="tr-mbin-d",
            direction=SignalDirection.SELL,
            entry_offset_minutes=45,
            exit_offset_minutes=55,
            requested_entry_price="2650.00",
            entry_fill_price="2650.00",
            requested_exit_price="2650.00",
            exit_fill_price="2650.00",
            stop_loss="2660.00",
            take_profit="2640.00",
            gross_pnl="2.00",
            commission="2.00",
            fee="0.00",
            swap="0.00",  # net_pnl == 0.00 (breakeven!)
        ),
        pred_dir=SignalDirection.SELL,
        p_sell="0.85",
        p_wait="0.10",
        p_buy="0.05",
    )

    brier_mbin, ece_mbin, n_mbin = compute_post_trade_calibration_metrics((tr_a, tr_b, tr_c, tr_d))
    assert n_mbin == 4
    assert brier_mbin == Decimal("0.7988")
    assert ece_mbin == Decimal("0.4375")

    # L-1 edge cases:
    # A) ml_predicted_direction='WAIT' on an executed trade -> AnalyticsError
    wait_pred_tr = _attach_ml(tr_a, pred_dir=SignalDirection.WAIT, p_sell="0.50", p_wait="0.30", p_buy="0.20")
    with pytest.raises(AnalyticsError, match="Invalid ml_predicted_direction='WAIT'"):
        compute_post_trade_calibration_metrics((wait_pred_tr,))

    # B) ml_predicted_direction='BUY' on a SELL trade -> AnalyticsError
    opposing_pred_tr = _attach_ml(tr_a, pred_dir=SignalDirection.BUY, p_sell="0.50", p_wait="0.30", p_buy="0.20")
    with pytest.raises(AnalyticsError, match="Inconsistent ml_predicted_direction='BUY' on SELL trade"):
        compute_post_trade_calibration_metrics((opposing_pred_tr,))

    # C) ml_predicted_direction set while probabilities are None -> AnalyticsError
    missing_probs_with_dir = _attach_ml(tr_a, pred_dir=SignalDirection.SELL, p_sell=None, p_wait=None, p_buy=None)
    with pytest.raises(AnalyticsError, match="ml_predicted_direction=.* is set but ML probabilities are None"):
        compute_post_trade_calibration_metrics((missing_probs_with_dir,))

    # --- 2. T-2: Combined Expectancy + Streak Quarantine, Precedence over Execution Degraded, & evaluate_journal() ---
    cfg_t2 = PostTradeAnalyticsConfig(
        journal_path=tmp_path / "t2_journal.json",
        min_trades_for_drift_evaluation=3,
        max_acceptable_mean_slippage_points=Decimal("10.0"),
        min_expectancy_r_threshold=Decimal("0.0000"),
        max_consecutive_losses_alert=3,
    )
    journal_t2 = PersistentTradeJournal(config=cfg_t2)
    engine_t2 = DriftDetectionEngine(config=cfg_t2, journal=journal_t2)

    # 3 consecutive losing trades with NO_KNOWLEDGE_RULE (contributing_knowledge_rule_ids=())
    # AND high entry slippage (+25 pts > 10.0 pts):
    # - quarantine_recommended_rule_ids is () because NO_KNOWLEDGE_RULE is synthetic
    # - BOTH expectancy_degraded (expectancy_r = -0.5000 < 0) AND streak_degraded (3 >= 3) are True
    # - execution_degraded is ALSO True (25.0000 > 10.0)
    # -> Must classify as DriftStatus.QUARANTINE_RECOMMENDED (taking precedence over EXECUTION_DEGRADED)!
    severe_no_rule_trades = tuple(
        _make_phase82_trade(
            trade_id=f"tr-sev-{i}",
            entry_offset_minutes=i * 10,
            exit_offset_minutes=i * 10 + 5,
            requested_entry_price="2650.00",
            entry_fill_price="2650.25",  # +25.0000 pts adverse slippage
            requested_exit_price="2645.00",
            exit_fill_price="2645.00",
            gross_pnl="-50.00",
            commission="0.00",
            fee="0.00",
            swap="0.00",
            exit_reason=ExitReason.STOP_LOSS,
            contributing_knowledge_rule_ids=(),
        )
        for i in range(3)
    )
    for idx, tr in enumerate(severe_no_rule_trades, start=1):
        journal_t2.record_completed_trade(
            tr.model_copy(
                update={
                    "entry_deal_tickets": (99000 + idx * 2,),
                    "exit_deal_tickets": (99001 + idx * 2,),
                }
            ),
            account_login=100200300,
            server="AtlasBroker-Demo",
        )

    rep_journal = engine_t2.evaluate_journal(source_filter=TradeSource.MT5_DEMO)
    assert rep_journal.overall_status == DriftStatus.QUARANTINE_RECOMMENDED
    assert rep_journal.quarantine_recommended_rule_ids == ()
    assert rep_journal.realized_expectancy_r == Decimal("-0.5000")
    assert rep_journal.max_consecutive_losses == 3
    assert rep_journal.mean_entry_slippage_points == Decimal("25.0000")
    assert len(rep_journal.warnings) == 3

    # --- 3. M-1, M-2, M-3, L-2 & T-3: Governance & Model Validator Edge Cases ---
    evaluator = GovernancePromotionEvaluator(config=cfg_t2)
    wf_cfg = WalkForwardConfig(purge_bars=12, labeling=LabelingConfig(horizon_bars=12))
    oos_eval = _make_mock_model_eval_report(active_signals=15, exp_r="0.4000", ece="0.0400", brier="0.2000")
    fold_0 = WalkForwardFoldResult(
        fold_index=0,
        train_start_index=0,
        train_end_index=119,
        purge_start_index=120,
        purge_end_index=131,
        test_start_index=132,
        test_end_index=171,
        model_id="fold-0",
        evaluation=oos_eval,
    )
    fold_1_valid = WalkForwardFoldResult(
        fold_index=1,
        train_start_index=0,
        train_end_index=159,
        purge_start_index=160,
        purge_end_index=171,
        test_start_index=172,
        test_end_index=211,
        model_id="fold-1",
        evaluation=oos_eval,
    )
    valid_2fold_wf = WalkForwardReport(
        model_type=ModelType.REGULARIZED_LOGISTIC,
        config=wf_cfg,
        folds=(fold_0, fold_1_valid),
        aggregated_out_of_sample=oos_eval,
    )
    assert validate_walk_forward_leakage_safety(valid_2fold_wf) == (12, 12)

    # M-2 & T-3: Fold 1 test_start_index (160) <= Fold 0 test_end_index (171) -> overlapping OOS test windows!
    fold_1_overlapping_test = WalkForwardFoldResult(
        fold_index=1,
        train_start_index=0,
        train_end_index=147,
        purge_start_index=148,
        purge_end_index=159,
        test_start_index=160,  # <= 171!
        test_end_index=199,
        model_id="fold-1-overlap",
        evaluation=oos_eval,
    )
    overlap_wf = valid_2fold_wf.model_copy(update={"folds": (fold_0, fold_1_overlapping_test)})
    with pytest.raises(FeedbackLeakageError, match="overlapping out-of-sample test windows across folds"):
        validate_walk_forward_leakage_safety(overlap_wf)

    # M-1: Passing only baseline_risk_parameters or only candidate_risk_parameters raises AnalyticsError
    with pytest.raises(AnalyticsError, match="must both be provided together"):
        evaluator.evaluate_candidate(
            candidate_id="cand-partial-risk-1",
            baseline_walk_forward=valid_2fold_wf,
            candidate_walk_forward=valid_2fold_wf,
            baseline_risk_parameters=RiskParameters(),
            candidate_risk_parameters=None,
        )
    with pytest.raises(AnalyticsError, match="must both be provided together"):
        evaluator.evaluate_candidate(
            candidate_id="cand-partial-risk-2",
            baseline_walk_forward=valid_2fold_wf,
            candidate_walk_forward=valid_2fold_wf,
            baseline_risk_parameters=None,
            candidate_risk_parameters=RiskParameters(),
        )

    # M-3: Standalone baseline ModelEvaluationReport without split_indices raises FeedbackLeakageError
    with pytest.raises(FeedbackLeakageError, match="standalone baseline ModelEvaluationReport"):
        evaluator.evaluate_candidate(
            candidate_id="cand-base-no-split",
            baseline_walk_forward=oos_eval,
            candidate_walk_forward=valid_2fold_wf,
            split_indices=None,
        )

    # M-3: Conflicting split_indices.(purge_bars, horizon_bars) alongside WalkForwardReport raises FeedbackLeakageError
    conflicting_split = ChronologicalSplitIndices(
        train_start=0,
        train_end=99,
        purge_1_start=100,
        purge_1_end=114,
        val_start=115,
        val_end=144,
        purge_2_start=145,
        purge_2_end=159,
        test_start=160,
        test_end=199,
        horizon_bars=15,
        purge_bars=15,
    )
    with pytest.raises(FeedbackLeakageError, match="conflicts with baseline WalkForwardReport"):
        evaluator.evaluate_candidate(
            candidate_id="cand-conflict-split",
            baseline_walk_forward=valid_2fold_wf,
            candidate_walk_forward=valid_2fold_wf,
            split_indices=conflicting_split,
        )

    # L-2 & T-3: Direct Pydantic model validation failures for non-finite Decimals and invalid report states
    with pytest.raises(ValidationError, match="finite"):
        DriftAndHealthReport(
            overall_status=DriftStatus.HEALTHY,
            total_trades=3,
            insufficient_sample=False,
            realized_expectancy_r=Decimal("NaN"),
            max_consecutive_losses=0,
            overall_metrics=rep_journal.overall_metrics,
        )
    with pytest.raises(ValidationError, match="HEALTHY DriftAndHealthReport cannot contain warnings"):
        DriftAndHealthReport(
            overall_status=DriftStatus.HEALTHY,
            total_trades=3,
            insufficient_sample=False,
            realized_expectancy_r=Decimal("0.5000"),
            max_consecutive_losses=0,
            warnings=("Unexpected warning on healthy report",),
            overall_metrics=rep_journal.overall_metrics,
        )
    with pytest.raises(ValidationError, match="WARNING_DRIFT_DETECTED requires at least one diagnostic warning"):
        DriftAndHealthReport(
            overall_status=DriftStatus.WARNING_DRIFT_DETECTED,
            total_trades=3,
            insufficient_sample=False,
            realized_expectancy_r=Decimal("-0.5000"),
            max_consecutive_losses=3,
            warnings=(),
            overall_metrics=rep_journal.overall_metrics,
        )
    with pytest.raises(ValidationError, match="finite"):
        GovernancePromotionDecision(
            candidate_id="cand-inf",
            approved_for_human_review=True,
            requires_human_approval=True,
            baseline_expectancy_r=Decimal("Infinity"),
            candidate_expectancy_r=Decimal("0.5000"),
            baseline_ece=Decimal("0.0400"),
            candidate_ece=Decimal("0.0400"),
            candidate_max_drawdown_r=Decimal("1.5000"),
            purge_bars=12,
            horizon_bars=12,
            rejection_reasons=(),
            rationale="Non-finite baseline expectancy",
        )




