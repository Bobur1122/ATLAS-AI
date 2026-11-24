"""OHLCV DataFrame and time-series validator for ATLAS AI."""

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from atlas_ai.data.exceptions import DataValidationError
from atlas_ai.data.integrity import check_timeseries_integrity
from atlas_ai.data.models import (
    DataValidationIssue,
    DataValidationIssueCode,
    DataValidationReport,
    MarketCalendarConfig,
    MissingBarPolicy,
)
from atlas_ai.domain.market import Timeframe

REQUIRED_OHLCV_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


class MarketDataValidator:
    """Comprehensive OHLCV and time-series integrity validator.

    Validates:
    - Non-empty dataset and presence of canonical columns (`timestamp`, `open`, `high`, `low`, `close`, `volume`).
    - Valid timestamps (no `NaT`).
    - Numeric OHLCV values with no `NaN` or `±Inf`.
    - Positive OHLC prices (`> 0`) and non-negative `volume` (`>= 0`).
    - Valid OHLC invariants (`high >= max(open, close)`, `low <= min(open, close)`, `high >= low`).
    - Time-series integrity (duplicates, monotonic order, timeframe alignment, and missing bars
      respecting weekend/holiday/session closures).
    """

    def __init__(
        self,
        calendar: MarketCalendarConfig | None = None,
        missing_bar_policy: MissingBarPolicy = MissingBarPolicy.STRICT,
    ) -> None:
        """Initialize the validator with a market calendar and missing-bar policy."""
        self._calendar = calendar or MarketCalendarConfig()
        self._missing_bar_policy = missing_bar_policy

    @property
    def calendar(self) -> MarketCalendarConfig:
        """Return configured MarketCalendarConfig."""
        return self._calendar

    @property
    def missing_bar_policy(self) -> MissingBarPolicy:
        """Return configured MissingBarPolicy."""
        return self._missing_bar_policy

    def validate(
        self,
        df: pd.DataFrame,
        symbol: str = "XAUUSD",
        timeframe: Timeframe = Timeframe.H1,
        missing_bar_policy: MissingBarPolicy | None = None,
        raise_on_error: bool = False,
    ) -> DataValidationReport:
        """Validate an OHLCV DataFrame and return a structured DataValidationReport.

        Args:
            df: Normalized OHLCV DataFrame to validate.
            symbol: Instrument symbol.
            timeframe: Expected bar Timeframe.
            missing_bar_policy: Optional override for the validator's default MissingBarPolicy.
            raise_on_error: If True, raise DataValidationError when `report.is_valid` is False.

        Returns:
            DataValidationReport containing `is_valid`, `issues`, `gaps`, and summary counts.
        """
        policy = missing_bar_policy or self._missing_bar_policy
        normalized_symbol = symbol.strip().upper()
        issues: list[DataValidationIssue] = []

        # 1. Check required columns
        missing_cols = [col for col in REQUIRED_OHLCV_COLUMNS if col not in df.columns]
        if missing_cols:
            issues.append(
                DataValidationIssue(
                    code=DataValidationIssueCode.MISSING_REQUIRED_COLUMNS,
                    message=f"Dataset is missing required OHLCV columns: {missing_cols}.",
                    is_error=True,
                )
            )
            return self._finalize_report(
                symbol=normalized_symbol,
                timeframe=timeframe,
                total_rows=len(df),
                policy=policy,
                issues=issues,
                gaps=[],
                missing_bars_total=0,
                raise_on_error=raise_on_error,
            )

        # 2. Check empty dataset
        if df.empty:
            issues.append(
                DataValidationIssue(
                    code=DataValidationIssueCode.EMPTY_DATASET,
                    message="Dataset is empty (0 rows).",
                    is_error=True,
                )
            )
            return self._finalize_report(
                symbol=normalized_symbol,
                timeframe=timeframe,
                total_rows=0,
                policy=policy,
                issues=issues,
                gaps=[],
                missing_bars_total=0,
                raise_on_error=raise_on_error,
            )

        # 3. Row-level timestamp and numeric OHLCV validation
        has_row_errors = False
        for idx, row in enumerate(df.itertuples(index=False)):
            ts_val: Any = getattr(row, "timestamp", None)
            if ts_val is None or pd.isna(ts_val) or not isinstance(ts_val, (datetime, pd.Timestamp)):
                issues.append(
                    DataValidationIssue(
                        code=DataValidationIssueCode.INVALID_TIMESTAMP,
                        message=f"Invalid or missing timestamp at row {idx}: {ts_val!r}.",
                        row_index=idx,
                        is_error=True,
                    )
                )
                has_row_errors = True
                continue

            row_dt: datetime = ts_val.to_pydatetime() if isinstance(ts_val, pd.Timestamp) else ts_val

            numeric_vals: dict[str, float] = {}
            row_numeric_valid = True
            for col_name in ("open", "high", "low", "close", "volume"):
                raw_val = getattr(row, col_name)
                if raw_val is None or isinstance(raw_val, bool):
                    issues.append(
                        DataValidationIssue(
                            code=DataValidationIssueCode.NON_NUMERIC_VALUE,
                            message=f"Non-numeric value in column '{col_name}' at row {idx}: {raw_val!r}.",
                            row_index=idx,
                            timestamp=row_dt,
                            is_error=True,
                        )
                    )
                    row_numeric_valid = False
                    has_row_errors = True
                    break

                try:
                    f_val = float(raw_val)
                except (TypeError, ValueError):
                    issues.append(
                        DataValidationIssue(
                            code=DataValidationIssueCode.NON_NUMERIC_VALUE,
                            message=f"Non-numeric value in column '{col_name}' at row {idx}: {raw_val!r}.",
                            row_index=idx,
                            timestamp=row_dt,
                            is_error=True,
                        )
                    )
                    row_numeric_valid = False
                    has_row_errors = True
                    break

                if math.isnan(f_val) or math.isinf(f_val) or not np.isfinite(f_val):
                    issues.append(
                        DataValidationIssue(
                            code=DataValidationIssueCode.NAN_OR_INFINITE_VALUE,
                            message=f"NaN or infinite value in column '{col_name}' at row {idx}.",
                            row_index=idx,
                            timestamp=row_dt,
                            is_error=True,
                        )
                    )
                    row_numeric_valid = False
                    has_row_errors = True
                    break

                numeric_vals[col_name] = f_val

            if not row_numeric_valid:
                continue

            o_val = numeric_vals["open"]
            h_val = numeric_vals["high"]
            l_val = numeric_vals["low"]
            c_val = numeric_vals["close"]
            v_val = numeric_vals["volume"]

            # Price positivity check
            if o_val <= 0.0 or h_val <= 0.0 or l_val <= 0.0 or c_val <= 0.0:
                issues.append(
                    DataValidationIssue(
                        code=DataValidationIssueCode.NON_POSITIVE_PRICE,
                        message=(
                            f"Non-positive OHLC price at row {idx} "
                            f"(open={o_val}, high={h_val}, low={l_val}, close={c_val})."
                        ),
                        row_index=idx,
                        timestamp=row_dt,
                        is_error=True,
                    )
                )
                has_row_errors = True

            # Non-negative volume check
            if v_val < 0.0:
                issues.append(
                    DataValidationIssue(
                        code=DataValidationIssueCode.NEGATIVE_VOLUME,
                        message=f"Negative volume at row {idx} (volume={v_val}).",
                        row_index=idx,
                        timestamp=row_dt,
                        is_error=True,
                    )
                )
                has_row_errors = True

            # OHLC ordering relationship check
            max_oc = max(o_val, c_val)
            min_oc = min(o_val, c_val)
            if h_val < l_val or h_val < max_oc or l_val > min_oc:
                issues.append(
                    DataValidationIssue(
                        code=DataValidationIssueCode.INVALID_OHLC_RELATIONSHIP,
                        message=(
                            f"Invalid OHLC relationship at row {idx}: "
                            f"open={o_val}, high={h_val}, low={l_val}, close={c_val}."
                        ),
                        row_index=idx,
                        timestamp=row_dt,
                        is_error=True,
                    )
                )
                has_row_errors = True

        # 4. Time-series integrity checks (only if all timestamps are valid datetime objects)
        gaps = []
        missing_bars_total = 0
        ts_issues_exist = any(
            i.code == DataValidationIssueCode.INVALID_TIMESTAMP for i in issues
        )
        if not ts_issues_exist:
            ts_issues, gaps, missing_bars_total = check_timeseries_integrity(
                df=df,
                timeframe=timeframe,
                calendar=self._calendar,
                missing_bar_policy=policy,
            )
            issues.extend(ts_issues)

        return self._finalize_report(
            symbol=normalized_symbol,
            timeframe=timeframe,
            total_rows=len(df),
            policy=policy,
            issues=issues,
            gaps=gaps,
            missing_bars_total=missing_bars_total,
            raise_on_error=raise_on_error,
        )

    @staticmethod
    def _finalize_report(
        *,
        symbol: str,
        timeframe: Timeframe,
        total_rows: int,
        policy: MissingBarPolicy,
        issues: list[DataValidationIssue],
        gaps: list[Any],
        missing_bars_total: int,
        raise_on_error: bool,
    ) -> DataValidationReport:
        is_valid = not any(issue.is_error for issue in issues)
        report = DataValidationReport(
            is_valid=is_valid,
            symbol=symbol,
            timeframe=timeframe.value,
            total_rows=total_rows,
            missing_bar_policy=policy,
            missing_bars_total=missing_bars_total,
            gaps=tuple(gaps),
            issues=tuple(issues),
        )
        if raise_on_error and not report.is_valid:
            first_err = next(issue for issue in issues if issue.is_error)
            raise DataValidationError(
                f"Market data validation failed for {symbol} ({timeframe.value}): "
                f"[{first_err.code.value}] {first_err.message}"
            )
        return report
