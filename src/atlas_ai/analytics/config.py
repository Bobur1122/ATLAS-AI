"""Immutable configuration schema for ATLAS AI Phase 8 Post-Trade Analytics & Continuous Feedback."""

from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class PostTradeAnalyticsConfig(BaseModel):
    """Deterministic configuration for post-trade reconstruction, persistent journaling,
    multi-dimensional attribution, and drift/governance thresholds.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    journal_path: Path = Field(
        default=Path("data/processed/trade_journal.json"),
        description="File path for the crash-safe persistent post-trade journal.",
    )
    lock_timeout_seconds: float = Field(
        default=5.0,
        gt=0.0,
        le=60.0,
        description="Timeout in seconds for acquiring an exclusive OS file lock on the trade journal.",
    )
    exit_level_tolerance_points: Decimal = Field(
        default=Decimal("30"),
        ge=Decimal("0"),
        description="Tolerance in instrument points when matching an exit fill price to stop_loss or take_profit.",
    )
    min_trades_for_drift_evaluation: int = Field(
        default=10,
        ge=1,
        le=10000,
        description="Minimum closed trades required before statistical drift or quarantine recommendations activate.",
    )
    max_acceptable_mean_slippage_points: Decimal = Field(
        default=Decimal("10.0"),
        ge=Decimal("0"),
        description="Maximum acceptable mean adverse entry slippage in symbol_info.point units before execution drift warning.",
    )
    max_acceptable_ece_drift: Decimal = Field(
        default=Decimal("0.1000"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Maximum acceptable Expected Calibration Error (ECE) increase over walk-forward baseline.",
    )
    max_acceptable_brier_drift: Decimal = Field(
        default=Decimal("0.1500"),
        ge=Decimal("0"),
        le=Decimal("2"),
        description="Maximum acceptable multiclass Brier score increase over walk-forward baseline.",
    )
    min_expectancy_r_threshold: Decimal = Field(
        default=Decimal("0.0000"),
        description="Minimum realized R-multiple expectancy required before negative expectancy drift is flagged.",
    )
    max_consecutive_losses_alert: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Consecutive losing trade count threshold triggering a drift warning.",
    )
