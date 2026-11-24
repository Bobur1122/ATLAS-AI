"""MT5 Market Data Adapter for real-time Bid/Ask ticks, closed candles, and Phase 2 pipeline integration."""

from datetime import datetime, timezone
from decimal import Decimal

from atlas_ai.data.integrity import get_timeframe_delta
from atlas_ai.data.loader import candles_to_dataframe
from atlas_ai.data.models import MissingBarPolicy
from atlas_ai.data.pipeline import MarketDataPipeline, MarketDataPipelineResult
from atlas_ai.domain.market import Candle, Timeframe
from atlas_ai.execution.config import ExecutionSafetyConfig
from atlas_ai.execution.connection import MT5ConnectionAdapter
from atlas_ai.execution.exceptions import StaleMarketDataError
from atlas_ai.execution.models import SymbolTick


class MT5MarketDataAdapter:
    """Retrieves and validates XAUUSD Bid/Ask ticks and closed historical candles from MT5.

    Key Guarantees:
    1. Re-verifies connection health and DEMO account status via `MT5ConnectionAdapter.ensure_connected()`
       before every market data fetch.
    2. Normalizes all timestamps to timezone-aware UTC.
    3. Rejects missing, malformed, or stale ticks (`age > max_tick_age_seconds`).
    4. Fetches historical candles starting at `start_pos=1` (excluding the currently forming bar `0`)
       and verifies `bar_open_utc + timeframe_delta <= reference_now_utc` when a reference clock is supplied.
    5. Converts all bars into validated Phase 1 `Candle` domain models and integrates directly with
       Phase 2 `MarketDataPipeline`.
    """

    def __init__(
        self,
        connection: MT5ConnectionAdapter,
        *,
        safety_config: ExecutionSafetyConfig | None = None,
        market_data_pipeline: MarketDataPipeline | None = None,
    ) -> None:
        self._connection = connection
        self._safety_config = safety_config or ExecutionSafetyConfig()
        self._pipeline = market_data_pipeline or MarketDataPipeline(
            source_timezone="UTC",
            target_timezone="UTC",
            missing_bar_policy=MissingBarPolicy.ALLOW,
        )

    @property
    def connection(self) -> MT5ConnectionAdapter:
        return self._connection

    @property
    def safety_config(self) -> ExecutionSafetyConfig:
        return self._safety_config

    @property
    def pipeline(self) -> MarketDataPipeline:
        return self._pipeline

    def fetch_latest_tick(
        self,
        symbol: str = "XAUUSD",
        *,
        reference_now_utc: datetime | None = None,
    ) -> SymbolTick:
        """Fetch and validate the latest Bid/Ask tick for `symbol`.

        Raises:
            StaleMarketDataError: If tick is None, prices are invalid, or tick age exceeds `max_tick_age_seconds`.
        """
        self._connection.get_symbol_info(symbol)
        cleaned = symbol.strip().upper()
        tick = self._connection.backend.symbol_info_tick(cleaned)
        if tick is None:
            raise StaleMarketDataError(f"MT5 symbol_info_tick('{cleaned}') returned None.")

        if tick.bid <= Decimal("0") or tick.ask <= Decimal("0") or tick.ask < tick.bid:
            raise StaleMarketDataError(
                f"Invalid tick prices for '{cleaned}': bid={tick.bid}, ask={tick.ask}."
            )

        if reference_now_utc is not None:
            ref_utc = (
                reference_now_utc.astimezone(timezone.utc)
                if reference_now_utc.tzinfo is not None
                else reference_now_utc.replace(tzinfo=timezone.utc)
            )
            age = tick.age_seconds(ref_utc)
            if age > float(self._safety_config.max_tick_age_seconds):
                raise StaleMarketDataError(
                    f"Stale tick for '{cleaned}': tick timestamp {tick.timestamp.isoformat()} is "
                    f"{age:.1f}s old (max_tick_age_seconds={self._safety_config.max_tick_age_seconds}s)."
                )
            if age < -5.0:
                raise StaleMarketDataError(
                    f"Future-dated tick anomaly for '{cleaned}': tick timestamp {tick.timestamp.isoformat()} "
                    f"is {-age:.1f}s ahead of reference clock {ref_utc.isoformat()}."
                )

        return tick

    def fetch_closed_candles(
        self,
        symbol: str = "XAUUSD",
        timeframe: Timeframe = Timeframe.H1,
        count: int = 250,
        *,
        reference_now_utc: datetime | None = None,
    ) -> tuple[Candle, ...]:
        """Fetch `count` closed historical candles for `(symbol, timeframe)` as validated Phase 1 `Candle` objects.

        CRITICAL ANTI-LOOKAHEAD RULE:
        - Requests `start_pos=1` from `copy_rates_from_pos`, which excludes position `0` (the currently forming candle).
        - If `reference_now_utc` is provided, also filters out any bar where `bar.timestamp + timeframe_delta > reference_now_utc`.
        """
        if count < 1:
            raise ValueError(f"Candle count must be >= 1, got {count}.")

        self._connection.get_symbol_info(symbol)
        cleaned = symbol.strip().upper()

        # start_pos=1 excludes the currently forming candle at index 0
        raw_bars = self._connection.backend.copy_rates_from_pos(
            cleaned,
            timeframe,
            start_pos=1,
            count=count,
        )
        if raw_bars is None or len(raw_bars) == 0:
            raise StaleMarketDataError(
                f"MT5 copy_rates_from_pos('{cleaned}', {timeframe.value}, start_pos=1, count={count}) "
                "returned no closed bars."
            )

        tf_delta = get_timeframe_delta(timeframe)
        ref_utc: datetime | None = None
        if reference_now_utc is not None:
            ref_utc = (
                reference_now_utc.astimezone(timezone.utc)
                if reference_now_utc.tzinfo is not None
                else reference_now_utc.replace(tzinfo=timezone.utc)
            )

        candles: list[Candle] = []
        prev_ts: datetime | None = None

        for idx, bar in enumerate(raw_bars):
            bar_open_utc = bar.timestamp.astimezone(timezone.utc)
            # Exclude any bar that has not yet fully closed as of ref_utc
            if ref_utc is not None and (bar_open_utc + tf_delta) > ref_utc:
                continue

            if prev_ts is not None and bar_open_utc <= prev_ts:
                raise StaleMarketDataError(
                    f"Non-monotonic or duplicate MT5 rate bar timestamp at index {idx}: "
                    f"{bar_open_utc.isoformat()} <= {prev_ts.isoformat()}."
                )
            prev_ts = bar_open_utc

            vol = bar.tick_volume if bar.tick_volume > Decimal("0") else bar.real_volume
            try:
                candle = Candle(
                    symbol=cleaned,
                    timeframe=timeframe,
                    timestamp=bar_open_utc,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=vol,
                )
            except Exception as exc:
                raise StaleMarketDataError(
                    f"MT5 rate bar at index {idx} ({bar_open_utc.isoformat()}) failed Candle validation: {exc}"
                ) from exc
            candles.append(candle)

        if not candles:
            raise StaleMarketDataError(
                f"Zero closed candles remained for '{cleaned}' ({timeframe.value}) after closed-bar filtering."
            )

        return tuple(candles)

    def fetch_market_data_pipeline_result(
        self,
        symbol: str = "XAUUSD",
        timeframe: Timeframe = Timeframe.H1,
        count: int = 250,
        *,
        missing_bar_policy: MissingBarPolicy = MissingBarPolicy.ALLOW,
        reference_now_utc: datetime | None = None,
    ) -> MarketDataPipelineResult:
        """Fetch closed candles from MT5 and run them through Phase 2 `MarketDataPipeline`."""
        candles = self.fetch_closed_candles(
            symbol=symbol,
            timeframe=timeframe,
            count=count,
            reference_now_utc=reference_now_utc,
        )
        df = candles_to_dataframe(candles, target_timezone="UTC")
        return self._pipeline.run_from_dataframe(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            missing_bar_policy=missing_bar_policy,
        )
