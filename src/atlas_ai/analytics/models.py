"""Immutable Pydantic domain models for ATLAS AI Phase 8 Post-Trade Analytics & Continuous Feedback."""

from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atlas_ai.backtest.models import BacktestMetrics, ExitReason
from atlas_ai.domain.market import Timeframe
from atlas_ai.domain.signals import SignalDirection
from atlas_ai.strategies.models import MarketRegime


class KillZoneSession(StrEnum):
    """Canonical ICT trading session and killzone classifications for post-trade attribution."""

    ASIA = "ASIA"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    LONDON_NY_OVERLAP = "LONDON_NY_OVERLAP"
    OFF_SESSION = "OFF_SESSION"


class TradeSource(StrEnum):
    """Origin of a completed trade record."""

    BACKTEST = "BACKTEST"
    MT5_DEMO = "MT5_DEMO"


class ContextAttributionStatus(StrEnum):
    """Indicates whether a closed trade was joined with a verified entry-time causal context snapshot."""

    VERIFIED_ENTRY_CONTEXT = "VERIFIED_ENTRY_CONTEXT"
    UNATTRIBUTED_MISSING_CONTEXT = "UNATTRIBUTED_MISSING_CONTEXT"


class DriftStatus(StrEnum):
    """Deterministic health and drift classification for a strategy, segment, or model."""

    HEALTHY = "HEALTHY"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
    WARNING_DRIFT_DETECTED = "WARNING_DRIFT_DETECTED"
    EXECUTION_DEGRADED = "EXECUTION_DEGRADED"
    QUARANTINE_RECOMMENDED = "QUARANTINE_RECOMMENDED"


class EntryContextSnapshot(BaseModel):
    """Immutable snapshot of the causal market regime, session, confluence, and ML prediction
    state captured at order validation/entry time and bound to `(client_order_id, signal_id)`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    client_order_id: str = Field(..., min_length=1)
    signal_id: str = Field(..., min_length=1)
    symbol: str = Field(..., min_length=3, max_length=16)
    timeframe: Timeframe
    direction: SignalDirection
    strategy_id: str = Field(..., min_length=1)
    captured_at_utc: datetime
    regime: MarketRegime = Field(default=MarketRegime.RANGING)
    killzone_session: KillZoneSession = Field(default=KillZoneSession.OFF_SESSION)
    mtf_aligned: bool = Field(default=False)
    buy_confluence_score: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), le=Decimal("1"))
    sell_confluence_score: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), le=Decimal("1"))
    signal_confidence: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), le=Decimal("1"))
    contributing_strategy_ids: tuple[str, ...] = Field(default=())
    contributing_knowledge_rule_ids: tuple[str, ...] = Field(default=())
    ml_model_id: str | None = Field(default=None)
    ml_predicted_direction: SignalDirection | None = Field(default=None)
    ml_buy_probability: Decimal | None = Field(default=None, ge=Decimal("0"), le=Decimal("1"))
    ml_sell_probability: Decimal | None = Field(default=None, ge=Decimal("0"), le=Decimal("1"))
    ml_wait_probability: Decimal | None = Field(default=None, ge=Decimal("0"), le=Decimal("1"))
    requested_entry_price: Decimal = Field(..., gt=Decimal("0"))
    stop_loss: Decimal = Field(..., gt=Decimal("0"))
    take_profit: Decimal = Field(..., gt=Decimal("0"))
    expected_spread_points: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    approved_volume: Decimal = Field(..., gt=Decimal("0"))
    initial_risk_amount: Decimal = Field(..., gt=Decimal("0"))

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned.isalnum():
            raise ValueError(f"Symbol must be alphanumeric, got '{value}'.")
        return cleaned

    @field_validator("captured_at_utc")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("EntryContextSnapshot captured_at_utc must be timezone-aware.")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_direction(self) -> "EntryContextSnapshot":
        if self.direction not in (SignalDirection.BUY, SignalDirection.SELL):
            raise ValueError("EntryContextSnapshot direction must be BUY or SELL.")
        return self


class PartialExitProgressRecord(BaseModel):
    """Immutable tracking snapshot for an open position that has experienced partial exit deal(s)
    ($0 < V_{\\text{out}} < V_{\\text{in}}$) while remaining active on the broker ($V_{\\text{open}} = V_{\\text{in}} - V_{\\text{out}} > 0$).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    client_order_id: str = Field(..., min_length=1)
    signal_id: str = Field(..., min_length=1)
    position_ticket: int = Field(..., gt=0)
    symbol: str = Field(..., min_length=3, max_length=16)
    direction: SignalDirection
    initial_entry_volume: Decimal = Field(..., gt=Decimal("0"))
    closed_volume_so_far: Decimal = Field(..., gt=Decimal("0"))
    remaining_open_volume: Decimal = Field(..., gt=Decimal("0"))
    average_entry_price: Decimal = Field(..., gt=Decimal("0"))
    average_exit_price_so_far: Decimal = Field(..., gt=Decimal("0"))
    realized_gross_pnl_so_far: Decimal
    realized_commission_so_far: Decimal = Field(..., ge=Decimal("0"))
    realized_fee_so_far: Decimal = Field(..., ge=Decimal("0"))
    realized_swap_so_far: Decimal
    realized_net_pnl_so_far: Decimal
    entry_deal_tickets: tuple[int, ...] = Field(..., min_length=1)
    exit_deal_tickets_so_far: tuple[int, ...] = Field(..., min_length=1)
    commission_deal_tickets_so_far: tuple[int, ...] = Field(default=())
    last_exit_time_utc: datetime

    @field_validator("last_exit_time_utc")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("PartialExitProgressRecord last_exit_time_utc must be timezone-aware.")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_volumes_and_cash_flow(self) -> "PartialExitProgressRecord":
        if self.closed_volume_so_far + self.remaining_open_volume != self.initial_entry_volume:
            raise ValueError(
                f"Inconsistent partial exit volumes: closed ({self.closed_volume_so_far}) + "
                f"remaining ({self.remaining_open_volume}) != initial ({self.initial_entry_volume})."
            )
        expected_net = (
            self.realized_gross_pnl_so_far
            - self.realized_commission_so_far
            - self.realized_fee_so_far
            + self.realized_swap_so_far
        )
        if self.realized_net_pnl_so_far != expected_net:
            raise ValueError(
                f"Cash-flow identity violation on PartialExitProgressRecord: "
                f"realized_net_pnl_so_far ({self.realized_net_pnl_so_far}) != "
                f"gross - commission - fee + swap ({expected_net})."
            )
        return self


class CompletedTradeRecord(BaseModel):
    """Canonical immutable audit record of a fully closed round-turn trade from BACKTEST or MT5_DEMO.

    Enforces the exact cash-flow identity:
        `net_pnl == gross_pnl - commission - fee + swap`
    where `commission >= 0` and `fee >= 0` are non-negative cost magnitudes and `swap` is signed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trade_id: str = Field(..., min_length=1)
    source: TradeSource
    client_order_id: str = Field(..., min_length=1)
    signal_id: str = Field(..., min_length=1)
    position_ticket: int | None = Field(default=None, gt=0)
    symbol: str = Field(..., min_length=3, max_length=16)
    timeframe: Timeframe
    direction: SignalDirection
    strategy_id: str = Field(..., min_length=1)
    entry_timestamp_utc: datetime
    exit_timestamp_utc: datetime
    requested_entry_price: Decimal = Field(..., gt=Decimal("0"))
    entry_fill_price: Decimal = Field(..., gt=Decimal("0"))
    requested_exit_price: Decimal = Field(..., gt=Decimal("0"))
    exit_fill_price: Decimal = Field(..., gt=Decimal("0"))
    stop_loss: Decimal = Field(..., gt=Decimal("0"))
    take_profit: Decimal = Field(..., gt=Decimal("0"))
    volume: Decimal = Field(..., gt=Decimal("0"))
    entry_slippage_points: Decimal = Field(
        default=Decimal("0"),
        description="Signed entry slippage in symbol_info.point units (positive = adverse slippage, negative = favorable).",
    )
    exit_slippage_points: Decimal = Field(
        default=Decimal("0"),
        description="Signed exit slippage in symbol_info.point units relative to requested_exit_price (positive = adverse).",
    )
    initial_risk_amount: Decimal = Field(..., gt=Decimal("0"))
    gross_pnl: Decimal
    commission: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0"),
        description="Total non-negative round-turn commission magnitude (inline + separate DEAL_TYPE_COMMISSION).",
    )
    fee: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0"),
        description="Total non-negative round-turn exchange/broker fee magnitude.",
    )
    swap: Decimal = Field(
        default=Decimal("0.00"),
        description="Signed total overnight swap/rollover impact.",
    )
    net_pnl: Decimal
    realized_r_multiple: Decimal
    mfe_usd: Decimal = Field(default=Decimal("0.00"), ge=Decimal("0"))
    mae_usd: Decimal = Field(default=Decimal("0.00"), ge=Decimal("0"))
    exit_reason: ExitReason
    entry_deal_tickets: tuple[int, ...] = Field(default=())
    exit_deal_tickets: tuple[int, ...] = Field(default=())
    commission_deal_tickets: tuple[int, ...] = Field(default=())
    partial_exit_count: int = Field(default=1, ge=1)
    context_attribution_status: ContextAttributionStatus = Field(
        default=ContextAttributionStatus.UNATTRIBUTED_MISSING_CONTEXT
    )
    entry_context: EntryContextSnapshot | None = Field(default=None)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned.isalnum():
            raise ValueError(f"Symbol must be alphanumeric, got '{value}'.")
        return cleaned

    @field_validator("entry_timestamp_utc", "exit_timestamp_utc")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("CompletedTradeRecord timestamps must be timezone-aware.")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_invariants(self) -> "CompletedTradeRecord":
        if self.direction not in (SignalDirection.BUY, SignalDirection.SELL):
            raise ValueError("CompletedTradeRecord direction must be BUY or SELL.")
        if self.exit_timestamp_utc < self.entry_timestamp_utc:
            raise ValueError(
                f"exit_timestamp_utc ({self.exit_timestamp_utc.isoformat()}) cannot be earlier than "
                f"entry_timestamp_utc ({self.entry_timestamp_utc.isoformat()})."
            )
        expected_net = self.gross_pnl - self.commission - self.fee + self.swap
        if self.net_pnl != expected_net:
            raise ValueError(
                f"Cash-flow identity violation on CompletedTradeRecord '{self.trade_id}': "
                f"net_pnl ({self.net_pnl}) != gross_pnl ({self.gross_pnl}) - commission ({self.commission}) "
                f"- fee ({self.fee}) + swap ({self.swap}) = {expected_net}."
            )
        if self.entry_context is not None:
            if self.context_attribution_status != ContextAttributionStatus.VERIFIED_ENTRY_CONTEXT:
                raise ValueError(
                    "context_attribution_status must be VERIFIED_ENTRY_CONTEXT when entry_context is present."
                )
        else:
            if self.context_attribution_status != ContextAttributionStatus.UNATTRIBUTED_MISSING_CONTEXT:
                raise ValueError(
                    "context_attribution_status must be UNATTRIBUTED_MISSING_CONTEXT when entry_context is None."
                )
        return self


class TradeReconstructionReport(BaseModel):
    """Structured report produced by `DemoTradeReconstructor.sync_closed_demo_trades()`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    newly_completed_trades: tuple[CompletedTradeRecord, ...] = Field(default=())
    partial_exit_records: tuple[PartialExitProgressRecord, ...] = Field(default=())
    still_open_client_order_ids: tuple[str, ...] = Field(default=())
    already_journaled_client_order_ids: tuple[str, ...] = Field(default=())
    journal_generation: int = Field(..., ge=0)


class ExecutionQualityReport(BaseModel):
    """Quantifies fill quality, slippage, and cost drag across completed trades."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_evaluated_trades: int = Field(..., ge=0)
    mean_entry_slippage_points: Decimal = Field(default=Decimal("0.0000"))
    max_adverse_entry_slippage_points: Decimal = Field(default=Decimal("0.0000"), ge=Decimal("0"))
    adverse_entry_slippage_count: int = Field(default=0, ge=0)
    favorable_entry_slippage_count: int = Field(default=0, ge=0)
    zero_entry_slippage_count: int = Field(default=0, ge=0)
    mean_exit_slippage_points: Decimal = Field(default=Decimal("0.0000"))
    partial_exit_trade_count: int = Field(default=0, ge=0)
    total_gross_profit_usd: Decimal = Field(default=Decimal("0.00"), ge=Decimal("0"))
    total_commission_usd: Decimal = Field(default=Decimal("0.00"), ge=Decimal("0"))
    total_fee_usd: Decimal = Field(default=Decimal("0.00"), ge=Decimal("0"))
    total_swap_usd: Decimal = Field(default=Decimal("0.00"))
    total_cost_drag_usd: Decimal = Field(
        default=Decimal("0.00"),
        description="commission + fee - swap (positive means net cost drag).",
    )
    cost_drag_r_per_trade: Decimal = Field(default=Decimal("0.0000"))


class SegmentPerformanceBreakdown(BaseModel):
    """Deterministic performance breakdown for a single slice (Regime, Session, Timeframe, Strategy, Rule)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dimension_name: str = Field(..., min_length=1)
    segment_key: str = Field(..., min_length=1)
    total_trades: int = Field(..., ge=0)
    winning_trades: int = Field(..., ge=0)
    losing_trades: int = Field(..., ge=0)
    breakeven_trades: int = Field(..., ge=0)
    win_rate_pct: Decimal = Field(..., ge=Decimal("0"), le=Decimal("100"))
    gross_profit_usd: Decimal = Field(..., ge=Decimal("0"))
    gross_loss_usd: Decimal = Field(..., ge=Decimal("0"))
    net_profit_usd: Decimal
    profit_factor: Decimal = Field(..., ge=Decimal("0"))
    expectancy_usd: Decimal
    expectancy_r: Decimal
    max_drawdown_r: Decimal = Field(..., ge=Decimal("0"))
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    mean_entry_slippage_points: Decimal = Field(default=Decimal("0.0000"))
    insufficient_sample: bool = Field(
        ...,
        description="True when total_trades < config.min_trades_for_drift_evaluation.",
    )


class DriftAndHealthReport(BaseModel):
    """Deterministic diagnostic report evaluating strategy expectancy, ML calibration, and execution drift."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    overall_status: DriftStatus
    total_trades: int = Field(..., ge=0)
    insufficient_sample: bool
    realized_expectancy_r: Decimal
    max_consecutive_losses: int = Field(..., ge=0)
    realized_brier_score: Decimal = Field(default=Decimal("0.0000"), ge=Decimal("0"))
    realized_ece: Decimal = Field(default=Decimal("0.0000"), ge=Decimal("0"), le=Decimal("1"))
    brier_drift_from_baseline: Decimal = Field(default=Decimal("0.0000"))
    ece_drift_from_baseline: Decimal = Field(default=Decimal("0.0000"))
    mean_entry_slippage_points: Decimal = Field(default=Decimal("0.0000"))
    quarantine_recommended_rule_ids: tuple[str, ...] = Field(default=())
    warnings: tuple[str, ...] = Field(default=())
    overall_metrics: BacktestMetrics


class GovernancePromotionDecision(BaseModel):
    """Advisory, human-gated evaluation record for a candidate Phase 6 classifier or rule quarantine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(..., min_length=1)
    approved_for_human_review: bool
    requires_human_approval: Literal[True] = Field(
        default=True,
        description="Invariantly True: no model or strategy update can ever be auto-activated without human approval.",
    )
    baseline_expectancy_r: Decimal
    candidate_expectancy_r: Decimal
    baseline_ece: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    candidate_ece: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    candidate_max_drawdown_r: Decimal = Field(..., ge=Decimal("0"))
    purge_bars: int = Field(..., ge=1)
    horizon_bars: int = Field(..., ge=1)
    rejection_reasons: tuple[str, ...] = Field(default=())
    rationale: str = Field(..., min_length=1)
