"""Base strategy interface and causal warm-up/validation helpers for ATLAS AI."""

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from atlas_ai.domain.signals import SignalDirection
from atlas_ai.strategies.exceptions import StrategyEvaluationError
from atlas_ai.strategies.models import StrategyEvaluationResult


class BaseStrategy(ABC):
    """Abstract base class for all ATLAS AI trading strategies and market analyzers."""

    @property
    @abstractmethod
    def strategy_id(self) -> str:
        """Unique machine-readable strategy identifier."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @property
    def version(self) -> str:
        """Strategy semantic version."""
        return "1.0.0"

    @property
    @abstractmethod
    def required_features(self) -> tuple[str, ...]:
        """Tuple of DataFrame column names required by this strategy."""

    @property
    def is_contextual_filter(self) -> bool:
        """True if this module provides contextual confluence rather than standalone triggers."""
        return False

    @abstractmethod
    def evaluate(
        self,
        df: pd.DataFrame,
        as_of_index: int | None = None,
    ) -> StrategyEvaluationResult:
        """Deterministically evaluate the dataset up to `as_of_index` (inclusive).

        CRITICAL: Implementations must slice `df.iloc[: as_of_index + 1]` and never inspect
        rows after `as_of_index`.
        """

    def prepare_causal_slice(
        self,
        df: pd.DataFrame,
        as_of_index: int | None = None,
    ) -> pd.DataFrame:
        """Validate required columns, chronological order, and return the causal slice `df.iloc[:end]`."""
        if df.empty:
            raise StrategyEvaluationError(f"{self.strategy_id}: Cannot evaluate an empty DataFrame.")

        missing = [col for col in self.required_features if col not in df.columns]
        if missing:
            raise StrategyEvaluationError(
                f"{self.strategy_id}: Missing required columns {missing}. Available: {list(df.columns)}."
            )

        if as_of_index is None:
            sliced = df
        else:
            if as_of_index < 0 or as_of_index >= len(df):
                raise StrategyEvaluationError(
                    f"{self.strategy_id}: as_of_index ({as_of_index}) out of bounds for DataFrame of length {len(df)}."
                )
            sliced = df.iloc[: as_of_index + 1]

        ts_series = pd.to_datetime(sliced["timestamp"])
        if not ts_series.is_monotonic_increasing or ts_series.duplicated().any():
            raise StrategyEvaluationError(
                f"{self.strategy_id}: Input DataFrame timestamps must be strictly increasing without duplicates."
            )

        return sliced

    @staticmethod
    def extract_timestamp(row: pd.Series) -> datetime:
        """Extract a standard Python datetime from a DataFrame row's `timestamp` field."""
        ts_val: Any = row["timestamp"]
        if isinstance(ts_val, pd.Timestamp):
            return ts_val.to_pydatetime()
        if isinstance(ts_val, datetime):
            return ts_val
        return pd.Timestamp(ts_val).to_pydatetime()

    @staticmethod
    def has_warmup_nan(row: pd.Series, columns: tuple[str, ...]) -> list[str]:
        """Return a list of columns that contain NaN/null in the current row."""
        nan_cols: list[str] = []
        for col in columns:
            val = row.get(col)
            if val is None or pd.isna(val) or (isinstance(val, float) and np.isnan(val)):
                nan_cols.append(col)
        return nan_cols

    @staticmethod
    def to_decimal(value: float | int | str | Decimal, places: str = "0.01") -> Decimal:
        """Convert a numeric value to quantized Decimal."""
        return Decimal(str(value)).quantize(Decimal(places))

    def build_wait_result(
        self,
        *,
        timestamp: datetime,
        rationale: str,
        rejection_reasons: list[str] | tuple[str, ...],
        supporting_evidence: dict[str, object] | None = None,
    ) -> StrategyEvaluationResult:
        """Construct a standardized WAIT evaluation result."""
        return StrategyEvaluationResult(
            strategy_id=self.strategy_id,
            direction=SignalDirection.WAIT,
            timestamp=timestamp,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            confidence=Decimal("0"),
            rationale=rationale,
            supporting_evidence=supporting_evidence or {},
            rejection_reasons=tuple(rejection_reasons),
            is_contextual_filter=self.is_contextual_filter,
        )
