"""Chronological Event-Driven Historical Backtesting Engine for ATLAS AI Phase 4."""

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from decimal import Decimal

import pandas as pd

from atlas_ai.backtest.config import BacktestConfig
from atlas_ai.backtest.exceptions import BacktestExecutionError
from atlas_ai.backtest.execution import ExecutionSimulator
from atlas_ai.backtest.metrics import calculate_backtest_metrics
from atlas_ai.backtest.models import (
    BacktestResult,
    ExitReason,
    RiskRejectionRecord,
)
from atlas_ai.backtest.portfolio import PortfolioTracker
from atlas_ai.core.logger import get_logger
from atlas_ai.data.features import FeaturePipeline
from atlas_ai.data.integrity import get_timeframe_delta
from atlas_ai.domain.market import Timeframe
from atlas_ai.domain.risk import AccountState, RiskRejectionCode
from atlas_ai.domain.signals import SignalDirection, TradeSignal
from atlas_ai.risk.engine import RiskEngine
from atlas_ai.strategies.engine import StrategyEngine
from atlas_ai.strategies.models import (
    EconomicEvent,
    SignalLifecycleState,
)
from atlas_ai.strategies.multi_timeframe import filter_closed_candles_as_of
from atlas_ai.strategies.signal_lifecycle import SignalLifecycleManager

logger = get_logger("backtest.engine")

SignalProviderCallable = Callable[[pd.DataFrame, int, AccountState], TradeSignal]


class BacktestEngine:
    """Deterministic, chronological bar-by-bar historical backtesting engine.

    Architectural Invariants:
    1. Zero Lookahead Bias:
       - At bar `i`, strategy analysis only accesses `df.iloc[: i + 1]` and MTF candles
         whose close timestamp `<= bar_close_timestamp`.
       - A signal generated at bar `i` close executes at bar `i` close (adjusted for
         spread and adverse slippage) and is NEVER evaluated against bar `i`'s High/Low.
       - Intrabar SL/TP evaluation begins strictly at bar `i + 1`.
    2. Independent Deterministic Risk Management:
       - Every executable signal is routed through Phase 1 `RiskEngine` against the
         live `AccountState` before any position can be opened.
    3. Auditable Signal Lifecycle:
       - Every executable signal is tracked in `SignalLifecycleManager` from `GENERATED`
         through `RISK_APPROVED` -> `ACTIVE` -> `STOP_LOSS_HIT` / `TARGET_HIT` / `CANCELLED`.
    """

    def __init__(
        self,
        config: BacktestConfig | None = None,
        *,
        strategy_engine: StrategyEngine | None = None,
        feature_pipeline: FeaturePipeline | None = None,
    ) -> None:
        self._config = config or BacktestConfig()
        self._feature_pipeline = feature_pipeline or FeaturePipeline()
        self._strategy_engine = strategy_engine or StrategyEngine(
            feature_pipeline=self._feature_pipeline
        )
        self._risk_engine = RiskEngine(
            risk_parameters=self._config.risk_parameters,
            instrument_specs=self._config.instrument_spec,
        )

    @property
    def config(self) -> BacktestConfig:
        return self._config

    @property
    def strategy_engine(self) -> StrategyEngine:
        return self._strategy_engine

    @property
    def risk_engine(self) -> RiskEngine:
        return self._risk_engine

    def _prepare_causal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate chronological ordering and compute causal technical features."""
        if df.empty:
            return df.copy()
        required_ohlc = ("timestamp", "open", "high", "low", "close")
        for col in required_ohlc:
            if col not in df.columns:
                raise BacktestExecutionError(f"Backtest input DataFrame is missing required column '{col}'.")

        ts_series = pd.to_datetime(df["timestamp"])
        if not ts_series.is_monotonic_increasing:
            raise BacktestExecutionError("Backtest input DataFrame timestamps must be monotonically increasing.")
        if ts_series.duplicated().any():
            raise BacktestExecutionError("Backtest input DataFrame contains duplicate timestamps.")

        feature_cols = ("tr", "atr_14", "ema_9", "ema_21", "ema_50", "sma_20", "sma_50")
        if all(col in df.columns for col in feature_cols):
            return df.reset_index(drop=True).copy()
        return self._feature_pipeline.transform(df)

    @staticmethod
    def _extract_bar_timestamp(row: pd.Series) -> datetime:
        ts = row["timestamp"]
        if isinstance(ts, pd.Timestamp):
            return ts.to_pydatetime()
        if isinstance(ts, datetime):
            return ts
        return pd.Timestamp(ts).to_pydatetime()

    def run(
        self,
        df: pd.DataFrame,
        *,
        mtf_data: Mapping[Timeframe, pd.DataFrame] | None = None,
        economic_events: Sequence[EconomicEvent] | None = (),
        signal_provider: SignalProviderCallable | None = None,
    ) -> BacktestResult:
        """Execute a chronological bar-by-bar backtest on `df` and return an immutable `BacktestResult`."""
        cfg = self._config
        execution_sim = ExecutionSimulator(cfg)
        portfolio = PortfolioTracker(cfg)
        lifecycle_manager = SignalLifecycleManager(
            self._strategy_engine.lifecycle_manager.config
        )
        risk_rejections: list[RiskRejectionRecord] = []

        if df.empty:
            metrics = calculate_backtest_metrics(
                initial_balance=cfg.initial_balance,
                final_balance=portfolio.balance,
                final_equity=portfolio.equity,
                trades=(),
                equity_curve=(),
                annualization_periods=cfg.annualization_periods,
            )
            return BacktestResult(
                config=cfg,
                start_timestamp=None,
                end_timestamp=None,
                total_bars=0,
                trades=(),
                open_positions=(),
                equity_curve=(),
                risk_rejections=(),
                lifecycle_records=(),
                metrics=metrics,
            )

        feat_df = self._prepare_causal_features(df)
        prepared_mtf: dict[Timeframe, pd.DataFrame] | None = None
        if mtf_data is not None:
            prepared_mtf = {
                tf: self._prepare_causal_features(t_df) for tf, t_df in mtf_data.items()
            }

        n_bars = len(feat_df)
        tf_delta = get_timeframe_delta(cfg.timeframe)
        pos_counter = 0
        trade_counter = 0

        for i in range(n_bars):
            row = feat_df.iloc[i]
            bar_ts = self._extract_bar_timestamp(row)
            open_p = Decimal(str(row["open"]))
            high_p = Decimal(str(row["high"]))
            low_p = Decimal(str(row["low"]))
            close_p = Decimal(str(row["close"]))

            # Step A: Update trading-day rollover (default 5 PM New York time)
            portfolio.update_trading_day_rollover(bar_ts)

            # Step B & C: Evaluate existing open positions against current OHLC data
            # and process SL/TP exits + MFE/MAE updates
            for pos in list(portfolio.open_positions):
                next_trade_id = f"trd-{trade_counter + 1:05d}"
                updated_pos, closed_trade = execution_sim.evaluate_bar_for_position(
                    pos,
                    bar_index=i,
                    bar_timestamp=bar_ts,
                    open_price=open_p,
                    high_price=high_p,
                    low_price=low_p,
                    close_price=close_p,
                    trade_id_Trigger=next_trade_id,
                )
                if closed_trade is not None:
                    trade_counter += 1
                    portfolio.record_closed_trade(closed_trade)
                    target_lifecycle_state = (
                        SignalLifecycleState.STOP_LOSS_HIT
                        if closed_trade.exit_reason == ExitReason.STOP_LOSS
                        else SignalLifecycleState.TARGET_HIT
                    )
                    lifecycle_manager.transition(
                        pos.signal_id,
                        target_lifecycle_state,
                        bar_ts,
                        f"Position '{pos.position_id}' exited via {closed_trade.exit_reason.value} "
                        f"at {closed_trade.exit_fill_price} (net_pnl={closed_trade.net_pnl}).",
                    )
                else:
                    portfolio.update_open_position(updated_pos)

            # Synchronize any non-ACTIVE pending signals in SignalLifecycleManager
            for rec in lifecycle_manager.list_records():
                if rec.state not in (
                    SignalLifecycleState.ACTIVE,
                    SignalLifecycleState.RISK_REJECTED,
                    SignalLifecycleState.CANCELLED,
                    SignalLifecycleState.EXPIRED,
                    SignalLifecycleState.TARGET_HIT,
                    SignalLifecycleState.STOP_LOSS_HIT,
                ):
                    lifecycle_manager.evaluate_market_update(
                        rec.signal_id,
                        current_timestamp=bar_ts,
                        high_price=high_p,
                        low_price=low_p,
                    )

            # Step D & E: Build current immutable AccountState after mark-to-market
            account_state = portfolio.build_account_state()

            # Determine whether new entries can be evaluated on this bar
            is_last_bar = i == n_bars - 1
            allow_bar_entry = i >= cfg.warmup_bars and not (
                is_last_bar and cfg.close_open_positions_at_end
            )

            if allow_bar_entry:
                # Step F: Causal strategy evaluation at bar `i`
                causal_slice = feat_df.iloc[: i + 1]
                bar_close_ts = bar_ts + tf_delta

                if signal_provider is not None:
                    candidate_signal = signal_provider(causal_slice, i, account_state)
                    supporting_ev: dict[str, object] = {"source": "signal_provider"}
                else:
                    causal_mtf: dict[Timeframe, pd.DataFrame] | None = None
                    if prepared_mtf is not None:
                        causal_mtf = {
                            tf: filter_closed_candles_as_of(t_df, tf, bar_close_ts)
                            for tf, t_df in prepared_mtf.items()
                        }
                    report = self._strategy_engine.analyze(
                        causal_slice,
                        symbol=cfg.symbol,
                        timeframe=cfg.timeframe,
                        as_of_index=None,
                        mtf_data=causal_mtf,
                        economic_events=economic_events,
                        account_state=None,
                        risk_parameters=None,
                        instrument_spec=None,
                        register_in_lifecycle=False,
                    )
                    candidate_signal = report.signal
                    supporting_ev = {
                        "regime": report.regime.value,
                        "buy_score": str(report.confluence.buy_score),
                        "sell_score": str(report.confluence.sell_score),
                    }

                # Step G, H, I: Route executable signal through SignalLifecycleManager, RiskEngine, and ExecutionSimulator
                if candidate_signal.is_executable:
                    dup_record = lifecycle_manager._is_duplicate_active_setup(candidate_signal)
                    if dup_record is None:
                        det_signal_id = f"sig-{i:05d}"
                        managed_rec = lifecycle_manager.register_signal(
                            candidate_signal,
                            supporting_evidence=supporting_ev,
                            signal_id=det_signal_id,
                        )

                        # Independent Phase 1 RiskEngine evaluation
                        risk_decision = self._risk_engine.evaluate(
                            signal=candidate_signal,
                            account=account_state,
                        )
                        managed_rec = lifecycle_manager.apply_risk_decision(
                            managed_rec.signal_id,
                            risk_decision,
                            bar_ts,
                        )

                        if not risk_decision.approved:
                            risk_rejections.append(
                                RiskRejectionRecord(
                                    bar_index=i,
                                    timestamp=bar_ts,
                                    signal_id=managed_rec.signal_id,
                                    symbol=candidate_signal.symbol,
                                    direction=candidate_signal.direction,
                                    strategy_id=candidate_signal.strategy_id,
                                    rejection_reason=risk_decision.rejection_reason
                                    or "RISK_REJECTED",
                                    account_state=account_state,
                                    risk_decision=risk_decision,
                                )
                            )
                        else:
                            next_pos_id = f"pos-{pos_counter + 1:05d}"
                            fill_eval = execution_sim.open_position(
                                position_id=next_pos_id,
                                signal_id=managed_rec.signal_id,
                                signal=candidate_signal,
                                risk_decision=risk_decision,
                                entry_bar_index=i,
                                entry_timestamp=bar_ts,
                                requested_entry_price=close_p,
                            )
                            if (
                                not fill_eval.approved
                                or fill_eval.position is None
                            ):
                                reason = (
                                    fill_eval.rejection_reason
                                    or "EXECUTION_REJECTED"
                                )
                                lifecycle_manager.transition(
                                    managed_rec.signal_id,
                                    SignalLifecycleState.CANCELLED,
                                    bar_ts,
                                    reason,
                                )
                                risk_rejections.append(
                                    RiskRejectionRecord(
                                        bar_index=i,
                                        timestamp=bar_ts,
                                        signal_id=managed_rec.signal_id,
                                        symbol=candidate_signal.symbol,
                                        direction=candidate_signal.direction,
                                        strategy_id=candidate_signal.strategy_id,
                                        rejection_reason=reason,
                                        account_state=account_state,
                                        risk_decision=risk_decision,
                                    )
                                )
                            elif fill_eval.position.used_margin > portfolio.free_margin:
                                reason = (
                                    f"{RiskRejectionCode.INSUFFICIENT_FREE_MARGIN.value}: "
                                    f"Required fill margin ({fill_eval.position.used_margin}) "
                                    f"exceeds free_margin ({portfolio.free_margin})."
                                )
                                lifecycle_manager.transition(
                                    managed_rec.signal_id,
                                    SignalLifecycleState.CANCELLED,
                                    bar_ts,
                                    reason,
                                )
                                risk_rejections.append(
                                    RiskRejectionRecord(
                                        bar_index=i,
                                        timestamp=bar_ts,
                                        signal_id=managed_rec.signal_id,
                                        symbol=candidate_signal.symbol,
                                        direction=candidate_signal.direction,
                                        strategy_id=candidate_signal.strategy_id,
                                        rejection_reason=reason,
                                        account_state=account_state,
                                        risk_decision=risk_decision,
                                    )
                                )
                            else:
                                pos_counter += 1
                                portfolio.add_open_position(fill_eval.position)
                                lifecycle_manager.transition(
                                    managed_rec.signal_id,
                                    SignalLifecycleState.ACTIVE,
                                    bar_ts,
                                    f"Position '{fill_eval.position.position_id}' opened at fill price "
                                    f"{fill_eval.position.entry_fill_price} ({fill_eval.position.lots} lots).",
                                )

            # End-of-dataset open position handling on the final bar
            if is_last_bar and cfg.close_open_positions_at_end:
                for pos in list(portfolio.open_positions):
                    trade_counter += 1
                    eod_trade_id = f"trd-{trade_counter:05d}"
                    eod_trade = execution_sim.close_position(
                        pos,
                        trade_id=eod_trade_id,
                        exit_bar_index=i,
                        exit_timestamp=bar_ts,
                        requested_exit_price=close_p,
                        exit_reason=ExitReason.END_OF_DATA,
                    )
                    portfolio.record_closed_trade(eod_trade)
                    lifecycle_manager.transition(
                        pos.signal_id,
                        SignalLifecycleState.CANCELLED,
                        bar_ts,
                        f"Position '{pos.position_id}' force-closed at END_OF_DATA "
                        f"at {eod_trade.exit_fill_price} (net_pnl={eod_trade.net_pnl}).",
                    )

            # Step J: Record bar-end EquityPoint
            portfolio.record_equity_point(i, bar_ts)

        start_ts = self._extract_bar_timestamp(feat_df.iloc[0])
        end_ts = self._extract_bar_timestamp(feat_df.iloc[-1])

        metrics = calculate_backtest_metrics(
            initial_balance=cfg.initial_balance,
            final_balance=portfolio.balance,
            final_equity=portfolio.equity,
            trades=portfolio.closed_trades,
            equity_curve=portfolio.equity_curve,
            annualization_periods=cfg.annualization_periods,
        )

        return BacktestResult(
            config=cfg,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            total_bars=n_bars,
            trades=portfolio.closed_trades,
            open_positions=portfolio.open_positions,
            equity_curve=portfolio.equity_curve,
            risk_rejections=tuple(risk_rejections),
            lifecycle_records=tuple(lifecycle_manager.list_records()),
            metrics=metrics,
        )
