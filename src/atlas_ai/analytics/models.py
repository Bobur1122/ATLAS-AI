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

    @model_validator(mode="after")
    def validate_slippage_counts(self) -> "ExecutionQualityReport":
        expected_total = (
            self.adverse_entry_slippage_count
            + self.favorable_entry_slippage_count
            + self.zero_entry_slippage_count
        )
        if expected_total != self.total_evaluated_trades:
            raise ValueError(
                f"Slippage count partition violation: adverse ({self.adverse_entry_slippage_count}) + "
                f"favorable ({self.favorable_entry_slippage_count}) + zero ({self.zero_entry_slippage_count}) "
                f"!= total_evaluated_trades ({self.total_evaluated_trades})."
            )
        if self.partial_exit_trade_count > self.total_evaluated_trades:
            raise ValueError(
                f"partial_exit_trade_count ({self.partial_exit_trade_count}) cannot exceed "
                f"total_evaluated_trades ({self.total_evaluated_trades})."
            )
        expected_drag = self.total_commission_usd + self.total_fee_usd - self.total_swap_usd
        if self.total_cost_drag_usd != expected_drag:
            raise ValueError(
                f"Cost-drag identity violation: total_cost_drag_usd ({self.total_cost_drag_usd}) != "
                f"commission ({self.total_commission_usd}) + fee ({self.total_fee_usd}) "
                f"- swap ({self.total_swap_usd}) = {expected_drag}."
            )
        return self


SINGLE_LABEL_DIMENSIONS: tuple[str, ...] = (
    "source",
    "timeframe",
    "direction",
    "strategy_id",
    "regime",
    "killzone_session",
)

MULTI_LABEL_DIMENSIONS: tuple[str, ...] = (
    "contributing_strategy_id",
    "knowledge_rule_id",
)


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

    @model_validator(mode="after")
    def validate_trade_counts(self) -> "SegmentPerformanceBreakdown":
        if self.winning_trades + self.losing_trades + self.breakeven_trades != self.total_trades:
            raise ValueError(
                f"Segment ({self.dimension_name}={self.segment_key}) count mismatch: "
                f"wins ({self.winning_trades}) + losses ({self.losing_trades}) + "
                f"breakevens ({self.breakeven_trades}) != total_trades ({self.total_trades})."
            )
        return self


class PerformanceAttributionReport(BaseModel):
    """Immutable top-level Phase 8.2 report combining portfolio BacktestMetrics, R-curve drawdown,
    ExecutionQualityReport, and multi-dimensional SegmentPerformanceBreakdown slices.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    overall_metrics: BacktestMetrics
    overall_max_drawdown_r: Decimal = Field(default=Decimal("0.0000"), ge=Decimal("0"))
    verified_context_trades: int = Field(..., ge=0)
    unattributed_missing_context_trades: int = Field(..., ge=0)
    execution_quality: ExecutionQualityReport
    segment_breakdowns: tuple[SegmentPerformanceBreakdown, ...] = Field(default=())

    @model_validator(mode="after")
    def validate_portfolio_and_dimension_invariants(self) -> "PerformanceAttributionReport":
        total_n = self.overall_metrics.total_trades
        if self.verified_context_trades + self.unattributed_missing_context_trades != total_n:
            raise ValueError(
                f"Context attribution count mismatch: verified ({self.verified_context_trades}) + "
                f"unattributed ({self.unattributed_missing_context_trades}) != total_trades ({total_n})."
            )
        if self.execution_quality.total_evaluated_trades != total_n:
            raise ValueError(
                f"execution_quality.total_evaluated_trades ({self.execution_quality.total_evaluated_trades}) "
                f"!= overall_metrics.total_trades ({total_n})."
            )

        by_dim: dict[str, list[SegmentPerformanceBreakdown]] = {}
        seen_keys: set[tuple[str, str]] = set()
        for seg in self.segment_breakdowns:
            pair = (seg.dimension_name, seg.segment_key)
            if pair in seen_keys:
                raise ValueError(
                    f"Duplicate SegmentPerformanceBreakdown for dimension='{seg.dimension_name}', "
                    f"segment_key='{seg.segment_key}'."
                )
            seen_keys.add(pair)
            by_dim.setdefault(seg.dimension_name, []).append(seg)

        # Single-label dimensions must partition total_trades and net_profit_usd exactly
        for dim_name in SINGLE_LABEL_DIMENSIONS:
            segs = by_dim.get(dim_name, [])
            if total_n == 0:
                if segs:
                    raise ValueError(
                        f"Single-label dimension '{dim_name}' must be empty when total_trades == 0."
                    )
                continue
            if not segs:
                continue
            sum_trades = sum(s.total_trades for s in segs)
            sum_wins = sum(s.winning_trades for s in segs)
            sum_losses = sum(s.losing_trades for s in segs)
            sum_be = sum(s.breakeven_trades for s in segs)
            sum_net = sum((s.net_profit_usd for s in segs), Decimal("0.00"))
            sum_gp = sum((s.gross_profit_usd for s in segs), Decimal("0.00"))
            sum_gl = sum((s.gross_loss_usd for s in segs), Decimal("0.00"))
            if (
                sum_trades != total_n
                or sum_wins != self.overall_metrics.winning_trades
                or sum_losses != self.overall_metrics.losing_trades
                or sum_be != self.overall_metrics.breakeven_trades
                or sum_net != self.overall_metrics.net_profit_usd
                or sum_gp != self.overall_metrics.gross_profit
                or sum_gl != self.overall_metrics.gross_loss
            ):
                raise ValueError(
                    f"Single-label partition invariant violated for dimension '{dim_name}': "
                    f"sum_trades={sum_trades} (expected {total_n}), sum_net={sum_net} "
                    f"(expected {self.overall_metrics.net_profit_usd})."
                )

        # Multi-label dimensions: each segment has 1 <= total_trades <= total_n, and sum >= total_n
        for dim_name in MULTI_LABEL_DIMENSIONS:
            segs = by_dim.get(dim_name, [])
            if total_n == 0:
                if segs:
                    raise ValueError(
                        f"Multi-label dimension '{dim_name}' must be empty when total_trades == 0."
                    )
                continue
            if not segs:
                continue
            for s in segs:
                if not (1 <= s.total_trades <= total_n):
                    raise ValueError(
                        f"Multi-label segment ({dim_name}={s.segment_key}) has invalid "
                        f"total_trades={s.total_trades} (portfolio total_trades={total_n})."
                    )
            sum_multi_trades = sum(s.total_trades for s in segs)
            if sum_multi_trades < total_n:
                raise ValueError(
                    f"Multi-label coverage violation for dimension '{dim_name}': "
                    f"sum(total_trades)={sum_multi_trades} < portfolio total_trades={total_n}."
                )

        return self

    def get_segment(
        self,
        dimension_name: str,
        segment_key: str,
    ) -> SegmentPerformanceBreakdown | None:
        """Look up a specific `(dimension_name, segment_key)` breakdown."""
        dim_clean = dimension_name.strip()
        key_clean = segment_key.strip()
        for seg in self.segment_breakdowns:
            if seg.dimension_name == dim_clean and seg.segment_key == key_clean:
                return seg
        return None

    def list_dimension(
        self,
        dimension_name: str,
    ) -> tuple[SegmentPerformanceBreakdown, ...]:
        """Return all `SegmentPerformanceBreakdown` slices for `dimension_name`."""
        dim_clean = dimension_name.strip()
        return tuple(seg for seg in self.segment_breakdowns if seg.dimension_name == dim_clean)


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

    @model_validator(mode="after")
    def validate_drift_report_invariants(self) -> "DriftAndHealthReport":
        for field_name, val in (
            ("realized_expectancy_r", self.realized_expectancy_r),
            ("realized_brier_score", self.realized_brier_score),
            ("realized_ece", self.realized_ece),
            ("brier_drift_from_baseline", self.brier_drift_from_baseline),
            ("ece_drift_from_baseline", self.ece_drift_from_baseline),
            ("mean_entry_slippage_points", self.mean_entry_slippage_points),
        ):
            if not isinstance(val, Decimal) or not val.is_finite():
                raise ValueError(
                    f"DriftAndHealthReport.{field_name} must be a finite Decimal, got {val!r}."
                )

        if self.total_trades != self.overall_metrics.total_trades:
            raise ValueError(
                f"DriftAndHealthReport.total_trades ({self.total_trades}) != "
                f"overall_metrics.total_trades ({self.overall_metrics.total_trades})."
            )
        if self.insufficient_sample:
            if self.overall_status != DriftStatus.INSUFFICIENT_SAMPLE:
                raise ValueError(
                    f"overall_status must be INSUFFICIENT_SAMPLE when insufficient_sample=True "
                    f"(got {self.overall_status})."
                )
            if self.quarantine_recommended_rule_ids or self.warnings:
                raise ValueError(
                    "quarantine_recommended_rule_ids and warnings must be empty when insufficient_sample=True."
                )
        else:
            if self.overall_status == DriftStatus.INSUFFICIENT_SAMPLE:
                raise ValueError(
                    "overall_status cannot be INSUFFICIENT_SAMPLE when insufficient_sample=False."
                )
            if self.overall_status == DriftStatus.HEALTHY:
                if self.warnings or self.quarantine_recommended_rule_ids:
                    raise ValueError(
                        "HEALTHY DriftAndHealthReport cannot contain warnings or quarantine_recommended_rule_ids."
                    )
            elif self.overall_status in (
                DriftStatus.WARNING_DRIFT_DETECTED,
                DriftStatus.EXECUTION_DEGRADED,
            ):
                if not self.warnings:
                    raise ValueError(
                        f"{self.overall_status.value} requires at least one diagnostic warning."
                    )
                if self.quarantine_recommended_rule_ids:
                    raise ValueError(
                        f"overall_status must be QUARANTINE_RECOMMENDED when quarantine_recommended_rule_ids "
                        f"is non-empty (got {self.overall_status.value})."
                    )
            elif self.overall_status == DriftStatus.QUARANTINE_RECOMMENDED:
                if not (self.quarantine_recommended_rule_ids or self.warnings):
                    raise ValueError(
                        "QUARANTINE_RECOMMENDED requires non-empty quarantine_recommended_rule_ids or warnings."
                    )
        return self


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

    @model_validator(mode="after")
    def validate_governance_invariants(self) -> "GovernancePromotionDecision":
        for field_name, val in (
            ("baseline_expectancy_r", self.baseline_expectancy_r),
            ("candidate_expectancy_r", self.candidate_expectancy_r),
            ("baseline_ece", self.baseline_ece),
            ("candidate_ece", self.candidate_ece),
            ("candidate_max_drawdown_r", self.candidate_max_drawdown_r),
        ):
            if not isinstance(val, Decimal) or not val.is_finite():
                raise ValueError(
                    f"GovernancePromotionDecision.{field_name} must be a finite Decimal, got {val!r}."
                )

        if self.purge_bars < self.horizon_bars:
            raise ValueError(
                f"LEAKAGE VIOLATION on GovernancePromotionDecision: purge_bars ({self.purge_bars}) "
                f"must be >= horizon_bars ({self.horizon_bars})."
            )
        if self.requires_human_approval is not True:
            raise ValueError("requires_human_approval must invariantly be True.")
        if self.approved_for_human_review and self.rejection_reasons:
            raise ValueError(
                "approved_for_human_review=True requires rejection_reasons to be empty."
            )
        if not self.approved_for_human_review and not self.rejection_reasons:
            raise ValueError(
                "approved_for_human_review=False requires at least one rejection reason."
            )
        return self
