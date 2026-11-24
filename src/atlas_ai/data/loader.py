"""Historical CSV OHLCV loader and Candle domain converter for ATLAS AI."""

from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
from pydantic import ValidationError

from atlas_ai.data.exceptions import DataLoadError, DataValidationError
from atlas_ai.data.validator import REQUIRED_OHLCV_COLUMNS
from atlas_ai.domain.market import Candle, Timeframe

COLUMN_ALIASES: dict[str, str] = {
    "timestamp": "timestamp",
    "datetime": "timestamp",
    "time": "timestamp",
    "date": "timestamp",
    "open": "open",
    "o": "open",
    "high": "high",
    "h": "high",
    "low": "low",
    "l": "low",
    "close": "close",
    "c": "close",
    "volume": "volume",
    "vol": "volume",
    "tick_volume": "volume",
    "tickvol": "volume",
}


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common OHLCV column name variations to canonical lowercase names.

    Supported aliases:
    - `timestamp` / `datetime` / `time` / `date` -> `timestamp`
    - `open` / `o` -> `open`
    - `high` / `h` -> `high`
    - `low` / `l` -> `low`
    - `close` / `c` -> `close`
    - `volume` / `vol` / `tick_volume` / `tickvol` -> `volume`

    Raises:
        DataLoadError: If required canonical columns are missing or ambiguous duplicate aliases exist.
    """
    renamed_map: dict[str, str] = {}
    canonical_targets: set[str] = set()

    for raw_col in df.columns:
        cleaned = str(raw_col).strip().lower()
        if cleaned in COLUMN_ALIASES:
            target = COLUMN_ALIASES[cleaned]
            if target in canonical_targets:
                raise DataLoadError(
                    f"Ambiguous duplicate column alias mapping to '{target}' in columns {list(df.columns)}."
                )
            renamed_map[raw_col] = target
            canonical_targets.add(target)

    normalized_df = df.rename(columns=renamed_map).copy()
    missing = [col for col in REQUIRED_OHLCV_COLUMNS if col not in normalized_df.columns]
    if missing:
        raise DataLoadError(
            f"CSV dataset is missing required OHLCV columns {missing}. "
            f"Found columns: {list(df.columns)}."
        )

    return normalized_df


def parse_and_localize_timestamps(
    series: pd.Series,
    source_timezone: str = "UTC",
    target_timezone: str = "UTC",
) -> pd.Series:
    """Parse timestamps without silently dropping malformed values, applying explicit timezone rules.

    - If parsed timestamps are timezone-naive, they are localized to `source_timezone`.
    - All timestamps are then converted to `target_timezone`.
    - Any unparseable timestamp raises `DataLoadError`.
    """
    try:
        src_tz = ZoneInfo(source_timezone.strip())
        tgt_tz = ZoneInfo(target_timezone.strip())
    except (ZoneInfoNotFoundError, KeyError, ValueError) as exc:
        raise DataLoadError(
            f"Invalid timezone configuration (source='{source_timezone}', target='{target_timezone}')."
        ) from exc

    if series.isna().any():
        bad_idx = int(series.isna().idxmax())
        raise DataLoadError(f"Missing or null timestamp at row {bad_idx}.")

    parsed_list = []
    for idx, raw_val in enumerate(series):
        if isinstance(raw_val, str) and not raw_val.strip():
            raise DataLoadError(f"Empty timestamp string at row {idx}.")
        try:
            ts = pd.Timestamp(raw_val)
        except Exception as exc:
            raise DataLoadError(
                f"Malformed timestamp at row {idx}: {raw_val!r}."
            ) from exc

        if pd.isna(ts):
            raise DataLoadError(f"Malformed timestamp (NaT) at row {idx}: {raw_val!r}.")

        if ts.tzinfo is None:
            ts = ts.tz_localize(src_tz)
        else:
            ts = ts.tz_convert(src_tz)

        ts = ts.tz_convert(tgt_tz)
        parsed_list.append(ts)

    return pd.Series(parsed_list, index=series.index, dtype=f"datetime64[ns, {target_timezone.strip()}]")


class HistoricalCSVLoader:
    """Loads historical OHLCV CSV files into normalized DataFrames and canonical Candle domain objects."""

    def __init__(
        self,
        source_timezone: str = "UTC",
        target_timezone: str = "UTC",
    ) -> None:
        """Initialize the CSV loader with explicit source and target timezones."""
        # Validate timezones upfront
        try:
            ZoneInfo(source_timezone.strip())
            ZoneInfo(target_timezone.strip())
        except (ZoneInfoNotFoundError, KeyError, ValueError) as exc:
            raise DataLoadError(
                f"Invalid timezone name: source='{source_timezone}', target='{target_timezone}'."
            ) from exc

        self._source_timezone = source_timezone.strip()
        self._target_timezone = target_timezone.strip()

    @property
    def source_timezone(self) -> str:
        """Timezone assumed for naive timestamps in the input CSV."""
        return self._source_timezone

    @property
    def target_timezone(self) -> str:
        """Timezone to which all loaded timestamps are converted."""
        return self._target_timezone

    def load_csv(
        self,
        file_path: str | Path,
        symbol: str = "XAUUSD",
        timeframe: Timeframe = Timeframe.H1,
    ) -> pd.DataFrame:
        """Load and normalize an OHLCV CSV file without silently discarding malformed rows.

        Args:
            file_path: Path to the CSV file.
            symbol: Instrument symbol to attach to the dataset.
            timeframe: Timeframe enum to attach to the dataset.

        Returns:
            DataFrame with canonical columns: `symbol`, `timeframe`, `timestamp`,
            `open`, `high`, `low`, `close`, `volume`, plus `_raw_*` string columns for exact Decimal conversion.

        Raises:
            DataLoadError: If the file does not exist, cannot be parsed, is empty,
                lacks required columns, or contains unparseable timestamps/numbers.
        """
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise DataLoadError(f"CSV file not found: {path}")

        try:
            # Read as strings first so we never silently coerce malformed text or lose Decimal precision
            raw_df = pd.read_csv(path, dtype=str, keep_default_na=False)
        except Exception as exc:
            raise DataLoadError(f"Failed to read CSV file '{path}': {exc}") from exc

        if raw_df.empty and len(raw_df.columns) == 0:
            raise DataLoadError(f"CSV file '{path}' contains no columns.")

        df = normalize_ohlcv_columns(raw_df)
        if df.empty:
            raise DataLoadError(f"CSV file '{path}' contains 0 data rows.")

        df["timestamp"] = parse_and_localize_timestamps(
            df["timestamp"],
            source_timezone=self._source_timezone,
            target_timezone=self._target_timezone,
        )

        # Convert numeric columns safely without silently turning bad strings into NaN
        for col in ("open", "high", "low", "close", "volume"):
            raw_series = df[col].astype(str).str.strip()
            df[f"_raw_{col}"] = raw_series
            numeric_values: list[float] = []
            for idx, val_str in enumerate(raw_series):
                if not val_str or val_str.lower() in ("nan", "null", "none", "inf", "-inf", "+inf"):
                    raise DataLoadError(
                        f"Invalid or missing numeric value in column '{col}' at row {idx}: {val_str!r}."
                    )
                try:
                    numeric_values.append(float(val_str))
                except ValueError as exc:
                    raise DataLoadError(
                        f"Non-numeric value in column '{col}' at row {idx}: {val_str!r}."
                    ) from exc
            df[col] = pd.Series(numeric_values, index=df.index, dtype="float64")

        cleaned_symbol = symbol.strip().upper()
        df["symbol"] = cleaned_symbol
        df["timeframe"] = timeframe.value

        ordered_cols = [
            "symbol",
            "timeframe",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "_raw_open",
            "_raw_high",
            "_raw_low",
            "_raw_close",
            "_raw_volume",
        ]
        remaining_cols = [c for c in df.columns if c not in ordered_cols]
        return df[ordered_cols + remaining_cols].reset_index(drop=True)


def dataframe_to_candles(
    df: pd.DataFrame,
    symbol: str = "XAUUSD",
    timeframe: Timeframe = Timeframe.H1,
) -> list[Candle]:
    """Convert a normalized OHLCV DataFrame into a list of validated `Candle` domain objects.

    Uses exact `_raw_*` string representations when available (or `str(val)`) to construct
    `Decimal` values and validates every row through `atlas_ai.domain.market.Candle`.
    """
    missing = [col for col in REQUIRED_OHLCV_COLUMNS if col not in df.columns]
    if missing:
        raise DataValidationError(
            f"Cannot convert DataFrame to Candle objects; missing columns: {missing}."
        )

    candles: list[Candle] = []
    default_symbol = symbol.strip().upper()

    for idx, row in enumerate(df.to_dict(orient="records")):
        row_symbol = str(row.get("symbol", default_symbol)).strip().upper()
        row_tf_raw = row.get("timeframe", timeframe)
        row_tf = Timeframe(row_tf_raw) if not isinstance(row_tf_raw, Timeframe) else row_tf_raw

        ts_val = row["timestamp"]
        if isinstance(ts_val, pd.Timestamp):
            ts_dt = ts_val.to_pydatetime()
        else:
            ts_dt = ts_val

        try:
            open_dec = Decimal(str(row.get("_raw_open", row["open"])))
            high_dec = Decimal(str(row.get("_raw_high", row["high"])))
            low_dec = Decimal(str(row.get("_raw_low", row["low"])))
            close_dec = Decimal(str(row.get("_raw_close", row["close"])))
            vol_dec = Decimal(str(row.get("_raw_volume", row["volume"])))

            candle = Candle(
                symbol=row_symbol,
                timeframe=row_tf,
                timestamp=ts_dt,
                open=open_dec,
                high=high_dec,
                low=low_dec,
                close=close_dec,
                volume=vol_dec,
            )
            candles.append(candle)
        except (ValidationError, InvalidOperation, ValueError, TypeError) as exc:
            raise DataValidationError(
                f"Row {idx} failed Candle domain validation: {exc}"
            ) from exc

    return candles


def candles_to_dataframe(
    candles: Sequence[Candle],
    target_timezone: str | None = None,
) -> pd.DataFrame:
    """Convert a sequence of canonical `Candle` domain objects back into a clean OHLCV DataFrame."""
    columns = ["symbol", "timeframe", "timestamp", "open", "high", "low", "close", "volume"]
    if not candles:
        return pd.DataFrame(columns=columns)

    records = []
    tz_obj = ZoneInfo(target_timezone.strip()) if target_timezone else None

    for c in candles:
        ts = c.timestamp
        if tz_obj is not None:
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=tz_obj)
            else:
                ts = ts.astimezone(tz_obj)

        records.append(
            {
                "symbol": c.symbol,
                "timeframe": c.timeframe.value,
                "timestamp": pd.Timestamp(ts),
                "open": float(c.open),
                "high": float(c.high),
                "low": float(c.low),
                "close": float(c.close),
                "volume": float(c.volume),
            }
        )

    return pd.DataFrame.from_records(records, columns=columns)
