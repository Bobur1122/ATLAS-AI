"""End-to-end Historical Market Data & Feature Pipeline for ATLAS AI."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from atlas_ai.core.logger import get_logger
from atlas_ai.data.exceptions import DataValidationError
from atlas_ai.data.features import FeaturePipeline
from atlas_ai.data.loader import (
    HistoricalCSVLoader,
    candles_to_dataframe,
    dataframe_to_candles,
)
from atlas_ai.data.models import (
    DataValidationReport,
    FeatureConfig,
    MarketCalendarConfig,
    MissingBarPolicy,
)
from atlas_ai.data.validator import MarketDataValidator
from atlas_ai.domain.market import Candle, Timeframe

logger = get_logger("data.pipeline")


@dataclass(frozen=True)
class MarketDataPipelineResult:
    """Structured output of the end-to-end MarketDataPipeline.

    Preserves clear separation between:
    - `raw_dataframe`: Normalized OHLCV DataFrame loaded from the source.
    - `validation_report`: Comprehensive OHLCV and time-series integrity report.
    - `candles`: Canonical validated `Candle` domain objects.
    - `validated_dataframe`: Clean OHLCV DataFrame reconstructed from validated `Candle` objects.
    - `feature_dataframe`: Final chronological dataset with all engineered technical features.
    """

    symbol: str
    timeframe: Timeframe
    raw_dataframe: pd.DataFrame
    validation_report: DataValidationReport
    candles: tuple[Candle, ...]
    validated_dataframe: pd.DataFrame
    feature_dataframe: pd.DataFrame


class MarketDataPipeline:
    """Reusable end-to-end pipeline:

    Historical CSV -> Data Loader -> Timestamp & Column Normalization ->
    Data Integrity Validation -> Candle Domain Conversion ->
    Feature Engineering -> Validated Feature Dataset.
    """

    def __init__(
        self,
        *,
        source_timezone: str = "UTC",
        target_timezone: str = "UTC",
        calendar: MarketCalendarConfig | None = None,
        missing_bar_policy: MissingBarPolicy = MissingBarPolicy.STRICT,
        feature_config: FeatureConfig | None = None,
    ) -> None:
        """Configure the MarketDataPipeline components."""
        active_calendar = calendar or (
            feature_config.calendar if feature_config is not None else MarketCalendarConfig()
        )
        active_feature_cfg = feature_config or FeatureConfig(calendar=active_calendar)

        self._loader = HistoricalCSVLoader(
            source_timezone=source_timezone,
            target_timezone=target_timezone,
        )
        self._validator = MarketDataValidator(
            calendar=active_calendar,
            missing_bar_policy=missing_bar_policy,
        )
        self._feature_pipeline = FeaturePipeline(config=active_feature_cfg)
        self._target_timezone = target_timezone

    @property
    def loader(self) -> HistoricalCSVLoader:
        """Return the underlying HistoricalCSVLoader."""
        return self._loader

    @property
    def validator(self) -> MarketDataValidator:
        """Return the underlying MarketDataValidator."""
        return self._validator

    @property
    def feature_pipeline(self) -> FeaturePipeline:
        """Return the underlying FeaturePipeline."""
        return self._feature_pipeline

    def run_from_csv(
        self,
        file_path: str | Path,
        symbol: str = "XAUUSD",
        timeframe: Timeframe = Timeframe.H1,
        missing_bar_policy: MissingBarPolicy | None = None,
    ) -> MarketDataPipelineResult:
        """Execute the full ingestion, validation, domain conversion, and feature pipeline on a CSV file.

        Raises:
            DataLoadError: If CSV loading, column normalization, or timestamp/numeric parsing fails.
            DataValidationError: If OHLCV or time-series validation fails under the active policy.
            FeatureCalculationError: If feature computation fails.
        """
        cleaned_symbol = symbol.strip().upper()
        logger.info(
            "Starting MarketDataPipeline for %s (%s) from '%s'",
            cleaned_symbol,
            timeframe.value,
            file_path,
        )

        # Stage 1 & 2: Load CSV + normalize columns and timestamps
        loaded_df = self._loader.load_csv(
            file_path=file_path,
            symbol=cleaned_symbol,
            timeframe=timeframe,
        )

        # Stage 3: Validate OHLCV and time-series integrity
        report = self._validator.validate(
            df=loaded_df,
            symbol=cleaned_symbol,
            timeframe=timeframe,
            missing_bar_policy=missing_bar_policy,
            raise_on_error=True,
        )

        # Stage 4: Convert into canonical Candle domain objects and back into validated OHLCV DataFrame
        candles = dataframe_to_candles(
            df=loaded_df,
            symbol=cleaned_symbol,
            timeframe=timeframe,
        )
        validated_df = candles_to_dataframe(
            candles=candles,
            target_timezone=self._target_timezone,
        )

        # Stage 5: Compute deterministic technical features
        feature_df = self._feature_pipeline.transform(validated_df)

        # Strip internal `_raw_*` columns from raw_dataframe before exposing
        public_raw_cols = [c for c in loaded_df.columns if not str(c).startswith("_raw_")]
        clean_raw_df = loaded_df[public_raw_cols].copy()

        logger.info(
            "MarketDataPipeline completed for %s (%s): %d candles validated, %d columns in feature dataset (missing_bars=%d).",
            cleaned_symbol,
            timeframe.value,
            len(candles),
            len(feature_df.columns),
            report.missing_bars_total,
        )

        return MarketDataPipelineResult(
            symbol=cleaned_symbol,
            timeframe=timeframe,
            raw_dataframe=clean_raw_df,
            validation_report=report,
            candles=tuple(candles),
            validated_dataframe=validated_df,
            feature_dataframe=feature_df,
        )

    def run_from_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str = "XAUUSD",
        timeframe: Timeframe = Timeframe.H1,
        missing_bar_policy: MissingBarPolicy | None = None,
    ) -> MarketDataPipelineResult:
        """Execute validation, domain conversion, and feature engineering on an in-memory DataFrame."""
        cleaned_symbol = symbol.strip().upper()
        report = self._validator.validate(
            df=df,
            symbol=cleaned_symbol,
            timeframe=timeframe,
            missing_bar_policy=missing_bar_policy,
            raise_on_error=True,
        )
        if not report.is_valid:
            raise DataValidationError(f"DataFrame validation failed for {cleaned_symbol}.")

        candles = dataframe_to_candles(df=df, symbol=cleaned_symbol, timeframe=timeframe)
        validated_df = candles_to_dataframe(candles=candles, target_timezone=self._target_timezone)
        feature_df = self._feature_pipeline.transform(validated_df)

        public_raw_cols = [c for c in df.columns if not str(c).startswith("_raw_")]
        return MarketDataPipelineResult(
            symbol=cleaned_symbol,
            timeframe=timeframe,
            raw_dataframe=df[public_raw_cols].copy(),
            validation_report=report,
            candles=tuple(candles),
            validated_dataframe=validated_df,
            feature_dataframe=feature_df,
        )
