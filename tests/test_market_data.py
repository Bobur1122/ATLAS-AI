"""Phase 2 tests for historical CSV ingestion, OHLCV validation, time-series integrity, and Candle conversion.

NOTE: All market data used in these tests (`tests/fixtures/synthetic_xauusd_h1.csv` and inline
fixtures) is strictly SYNTHETIC test data and is never presented as real historical market prices.
"""

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atlas_ai.data.exceptions import DataLoadError, DataValidationError
from atlas_ai.data.loader import (
    HistoricalCSVLoader,
    candles_to_dataframe,
    dataframe_to_candles,
    normalize_ohlcv_columns,
)
from atlas_ai.data.models import (
    DataValidationIssueCode,
    MarketCalendarConfig,
    MissingBarPolicy,
)
from atlas_ai.data.pipeline import MarketDataPipeline
from atlas_ai.data.validator import MarketDataValidator
from atlas_ai.domain.market import Candle, Timeframe

SYNTHETIC_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_xauusd_h1.csv"


def test_valid_synthetic_csv_ingestion_and_candle_roundtrip() -> None:
    """HistoricalCSVLoader must load the 220-bar synthetic XAUUSD CSV and roundtrip to Candle objects."""
    loader = HistoricalCSVLoader(source_timezone="UTC", target_timezone="UTC")
    df = loader.load_csv(SYNTHETIC_FIXTURE_PATH, symbol="xauusd", timeframe=Timeframe.H1)

    assert len(df) == 220
    assert df["symbol"].iloc[0] == "XAUUSD"
    assert df["timeframe"].iloc[0] == "H1"
    assert str(df["timestamp"].dtype) == "datetime64[ns, UTC]"

    candles = dataframe_to_candles(df, symbol="XAUUSD", timeframe=Timeframe.H1)
    assert len(candles) == 220
    assert isinstance(candles[0], Candle)
    assert candles[0].open == Decimal("2650.00")
    assert candles[0].close == Decimal("2648.50")

    roundtrip_df = candles_to_dataframe(candles, target_timezone="UTC")
    assert len(roundtrip_df) == 220
    assert list(roundtrip_df.columns) == [
        "symbol",
        "timeframe",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert roundtrip_df["open"].iloc[0] == pytest.approx(2650.00)


def test_column_normalization_variations(tmp_path: Path) -> None:
    """Loader must support common column naming aliases (Date/Time/Datetime, O/H/L/C, Tick_Volume)."""
    csv_file = tmp_path / "aliases.csv"
    csv_file.write_text(
        "DateTime,O,H,L,C,Tick_Volume\n"
        "2026-01-05 00:00:00,2650.0,2655.0,2648.0,2652.0,1100\n"
        "2026-01-05 01:00:00,2652.0,2658.0,2651.0,2656.0,1250\n",
        encoding="utf-8",
    )
    loader = HistoricalCSVLoader()
    df = loader.load_csv(csv_file, symbol="XAUUSD", timeframe=Timeframe.H1)
    assert len(df) == 2
    assert df["open"].iloc[0] == pytest.approx(2650.0)
    assert df["volume"].iloc[1] == pytest.approx(1250.0)


def test_invalid_csv_rejection_never_silently_discards_rows(tmp_path: Path) -> None:
    """Loader must reject missing files, empty CSVs, missing columns, bad timestamps, or non-numeric prices."""
    loader = HistoricalCSVLoader()

    # 1. Non-existent file
    with pytest.raises(DataLoadError, match="not found"):
        loader.load_csv(tmp_path / "does_not_exist.csv")

    # 2. Missing required column (`close` omitted)
    missing_col_csv = tmp_path / "missing_col.csv"
    missing_col_csv.write_text(
        "timestamp,open,high,low,volume\n2026-01-05 00:00:00,2650,2655,2648,100\n",
        encoding="utf-8",
    )
    with pytest.raises(DataLoadError, match="missing required OHLCV columns"):
        loader.load_csv(missing_col_csv)

    # 3. Empty data rows
    empty_rows_csv = tmp_path / "empty_rows.csv"
    empty_rows_csv.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")
    with pytest.raises(DataLoadError, match="0 data rows"):
        loader.load_csv(empty_rows_csv)

    # 4. Malformed timestamp on row 1
    bad_ts_csv = tmp_path / "bad_ts.csv"
    bad_ts_csv.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-01-05 00:00:00,2650,2655,2648,2652,100\n"
        "NOT_A_DATE,2652,2656,2650,2654,100\n",
        encoding="utf-8",
    )
    with pytest.raises(DataLoadError, match="Malformed timestamp at row 1"):
        loader.load_csv(bad_ts_csv)

    # 5. Non-numeric price on row 1 (never silently coerced to NaN or dropped)
    bad_num_csv = tmp_path / "bad_num.csv"
    bad_num_csv.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-01-05 00:00:00,2650,2655,2648,2652,100\n"
        "2026-01-05 01:00:00,2652,CORRUPT,2650,2654,100\n",
        encoding="utf-8",
    )
    with pytest.raises(DataLoadError, match="Non-numeric value in column 'high' at row 1"):
        loader.load_csv(bad_num_csv)

    # 6. Duplicate column aliases
    dup_col_df = pd.DataFrame(
        {"timestamp": ["2026-01-05T00:00:00Z"], "date": ["2026-01-05"], "open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]}
    )
    with pytest.raises(DataLoadError, match="Ambiguous duplicate column alias"):
        normalize_ohlcv_columns(dup_col_df)


def test_explicit_timezone_handling(tmp_path: Path) -> None:
    """Loader must localize naive timestamps using source_timezone and convert to target_timezone."""
    ny_csv = tmp_path / "ny_naive.csv"
    # 2026-01-05 09:00:00 in America/New_York (EST, UTC-5) == 2026-01-05 14:00:00 UTC
    ny_csv.write_text(
        "time,open,high,low,close,volume\n"
        "2026-01-05 09:00:00,2650.0,2655.0,2648.0,2652.0,500\n",
        encoding="utf-8",
    )
    loader = HistoricalCSVLoader(
        source_timezone="America/New_York",
        target_timezone="UTC",
    )
    df = loader.load_csv(ny_csv, symbol="XAUUSD", timeframe=Timeframe.H1)
    ts = df["timestamp"].iloc[0]
    assert ts == pd.Timestamp("2026-01-05 14:00:00", tz="UTC")


def test_validator_detects_invalid_ohlcv_values_and_empty_dataframe() -> None:
    """MarketDataValidator must detect empty datasets, NaN/Inf, non-positive prices, negative volume, and inverted OHLC."""
    validator = MarketDataValidator()

    # 1. Empty DataFrame
    empty_df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    empty_rep = validator.validate(empty_df)
    assert empty_rep.is_valid is False
    assert DataValidationIssueCode.EMPTY_DATASET in empty_rep.error_codes

    # 2. Invalid OHLC relationships, negative price, negative volume, NaN
    corrupt_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-05 00:00:00", periods=4, freq="1h", tz="UTC"),
            "open": [2650.0, -10.0, 2650.0, 2650.0],
            "high": [2645.0, 2655.0, 2655.0, np.nan],  # Row 0: high < open; Row 3: NaN
            "low": [2640.0, 2648.0, 2648.0, 2648.0],
            "close": [2648.0, 2652.0, 2652.0, 2652.0],
            "volume": [100.0, 100.0, -5.0, 100.0],  # Row 2: negative volume
        }
    )
    rep = validator.validate(corrupt_df)
    assert rep.is_valid is False
    assert DataValidationIssueCode.INVALID_OHLC_RELATIONSHIP in rep.error_codes
    assert DataValidationIssueCode.NON_POSITIVE_PRICE in rep.error_codes
    assert DataValidationIssueCode.NEGATIVE_VOLUME in rep.error_codes
    assert DataValidationIssueCode.NAN_OR_INFINITE_VALUE in rep.error_codes


def test_validator_detects_duplicate_and_out_of_order_timestamps() -> None:
    """MarketDataValidator must detect duplicate and non-monotonic timestamps."""
    validator = MarketDataValidator()
    ts_list = [
        pd.Timestamp("2026-01-05 00:00:00", tz="UTC"),
        pd.Timestamp("2026-01-05 02:00:00", tz="UTC"),
        pd.Timestamp("2026-01-05 02:00:00", tz="UTC"),  # Duplicate
        pd.Timestamp("2026-01-05 01:00:00", tz="UTC"),  # Out of order
    ]
    df = pd.DataFrame(
        {
            "timestamp": ts_list,
            "open": [2650.0] * 4,
            "high": [2655.0] * 4,
            "low": [2648.0] * 4,
            "close": [2652.0] * 4,
            "volume": [100.0] * 4,
        }
    )
    rep = validator.validate(df, timeframe=Timeframe.H1)
    assert rep.is_valid is False
    assert DataValidationIssueCode.DUPLICATE_TIMESTAMP in rep.error_codes
    assert DataValidationIssueCode.OUT_OF_ORDER_TIMESTAMP in rep.error_codes


def test_weekend_and_holiday_closures_are_not_flagged_as_missing_bars() -> None:
    """Weekend market closures (Friday 22:00 UTC -> Sunday 22:00 UTC) and configured holidays must pass STRICT validation."""
    loader = HistoricalCSVLoader()
    df = loader.load_csv(SYNTHETIC_FIXTURE_PATH, symbol="XAUUSD", timeframe=Timeframe.H1)

    # The 220-bar synthetic fixture spans across the weekend of 2026-01-09 21:00 -> 2026-01-11 22:00 UTC
    validator = MarketDataValidator(missing_bar_policy=MissingBarPolicy.STRICT)
    report = validator.validate(df, symbol="XAUUSD", timeframe=Timeframe.H1)
    assert report.is_valid is True
    assert report.missing_bars_total == 0
    assert len(report.gaps) == 0

    # Also verify explicit holiday closure: if Tuesday 2026-01-06 is marked as a holiday in MarketCalendarConfig,
    # dropping all bars on 2026-01-06 must NOT be flagged as missing bars.
    holiday_cal = MarketCalendarConfig(closed_dates=frozenset({date(2026, 1, 6)}))
    df_no_tuesday = df[df["timestamp"].dt.date != date(2026, 1, 6)].reset_index(drop=True)
    holiday_validator = MarketDataValidator(
        calendar=holiday_cal,
        missing_bar_policy=MissingBarPolicy.STRICT,
    )
    holiday_rep = holiday_validator.validate(df_no_tuesday, symbol="XAUUSD", timeframe=Timeframe.H1)
    assert holiday_rep.is_valid is True
    assert holiday_rep.missing_bars_total == 0


def test_missing_bar_policies_strict_report_and_allow() -> None:
    """MissingBarPolicy must reject in STRICT mode, warn in REPORT mode, and accept silently in ALLOW mode."""
    loader = HistoricalCSVLoader()
    df = loader.load_csv(SYNTHETIC_FIXTURE_PATH, symbol="XAUUSD", timeframe=Timeframe.H1)

    # Drop 2 open-market Monday hours (rows 5 and 6: 05:00 and 06:00 UTC)
    gapped_df = df.drop(index=[5, 6]).reset_index(drop=True)

    validator = MarketDataValidator()

    # 1. STRICT mode: is_valid=False, raises DataValidationError when raise_on_error=True
    strict_rep = validator.validate(
        gapped_df,
        timeframe=Timeframe.H1,
        missing_bar_policy=MissingBarPolicy.STRICT,
    )
    assert strict_rep.is_valid is False
    assert strict_rep.missing_bars_total == 2
    assert len(strict_rep.gaps) == 1
    assert strict_rep.gaps[0].missing_bars_count == 2
    assert DataValidationIssueCode.MISSING_EXPECTED_BAR in strict_rep.error_codes

    with pytest.raises(DataValidationError, match="MISSING_EXPECTED_BAR"):
        validator.validate(
            gapped_df,
            timeframe=Timeframe.H1,
            missing_bar_policy=MissingBarPolicy.STRICT,
            raise_on_error=True,
        )

    # 2. REPORT mode: is_valid=True, but issue is recorded in all_issue_codes with is_error=False
    report_rep = validator.validate(
        gapped_df,
        timeframe=Timeframe.H1,
        missing_bar_policy=MissingBarPolicy.REPORT,
    )
    assert report_rep.is_valid is True
    assert report_rep.missing_bars_total == 2
    assert len(report_rep.error_codes) == 0
    assert DataValidationIssueCode.MISSING_EXPECTED_BAR in report_rep.all_issue_codes

    # 3. ALLOW mode: is_valid=True, no validation issues emitted
    allow_rep = validator.validate(
        gapped_df,
        timeframe=Timeframe.H1,
        missing_bar_policy=MissingBarPolicy.ALLOW,
    )
    assert allow_rep.is_valid is True
    assert allow_rep.missing_bars_total == 2
    assert len(allow_rep.all_issue_codes) == 0


def test_end_to_end_market_data_pipeline() -> None:
    """MarketDataPipeline must run from CSV through validation, Candle conversion, and feature engineering."""
    pipeline = MarketDataPipeline(missing_bar_policy=MissingBarPolicy.STRICT)
    result = pipeline.run_from_csv(
        SYNTHETIC_FIXTURE_PATH,
        symbol="XAUUSD",
        timeframe=Timeframe.H1,
    )

    assert result.symbol == "XAUUSD"
    assert result.timeframe == Timeframe.H1
    assert result.validation_report.is_valid is True
    assert len(result.candles) == 220
    assert len(result.validated_dataframe) == 220
    assert len(result.feature_dataframe) == 220
    assert "sma_200" in result.feature_dataframe.columns
    assert "atr_14" in result.feature_dataframe.columns
    assert "session_high" in result.feature_dataframe.columns
