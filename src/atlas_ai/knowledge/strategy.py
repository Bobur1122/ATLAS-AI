"""Deterministic Knowledge-Based Strategy evaluating APPROVED educational rules for ATLAS AI Phase 5."""

from decimal import Decimal, ROUND_DOWN

import pandas as pd

from atlas_ai.domain.market import Timeframe
from atlas_ai.domain.signals import SignalDirection
from atlas_ai.knowledge.config import KnowledgeConfig
from atlas_ai.knowledge.models import (
    ComparisonOperator,
    ConditionType,
    ExtractedTradingRule,
    KnowledgeRuleCondition,
    RuleApprovalStatus,
)
from atlas_ai.knowledge.repository import KnowledgeRepository
from atlas_ai.knowledge.validator import KnowledgeRuleValidator
from atlas_ai.strategies.base import BaseStrategy
from atlas_ai.strategies.fvg import FVGDetectionStrategy, FVGRetestStrategy
from atlas_ai.strategies.liquidity import LiquiditySweepStrategy
from atlas_ai.strategies.market_structure import MarketStructureStrategy
from atlas_ai.strategies.models import StrategyEvaluationResult
from atlas_ai.strategies.premium_discount import PremiumDiscountAnalyzer
from atlas_ai.strategies.sessions import KillZoneSessionAnalyzer, PreviousDayLevelsAnalyzer
from atlas_ai.strategies.trend import MarketRegimeDetector


class KnowledgeBasedStrategy(BaseStrategy):
    """Evaluates explicitly APPROVED educational trading rules causally against market data.

    Key Safety Invariants:
    1. Strictly implements `BaseStrategy` and slices `df.iloc[: as_of_index + 1]` via `prepare_causal_slice()`.
    2. Evaluates ONLY `RuleApprovalStatus.APPROVED` rules from `KnowledgeRepository`.
       `DRAFT`, `VALIDATED`, and `REJECTED` rules are strictly ignored.
    3. Every rule is re-validated via `KnowledgeRuleValidator` prior to evaluation.
    4. Never uses `eval()` or `exec()`; evaluates only declarative `KnowledgeRuleCondition` predicates.
    5. Emits a standard `StrategyEvaluationResult` containing `matched_rule_ids`, `source_document_ids`,
       `source_chunk_ids`, and `matched_conditions` in `supporting_evidence`.
    """

    def __init__(
        self,
        repository: KnowledgeRepository,
        *,
        symbol: str = "XAUUSD",
        timeframe: Timeframe = Timeframe.H1,
        config: KnowledgeConfig | None = None,
        validator: KnowledgeRuleValidator | None = None,
    ) -> None:
        self._repository = repository
        self._symbol = symbol.strip().upper()
        self._timeframe = timeframe
        self._config = config or repository.config
        self._validator = validator or repository.validator

        # Causal Phase 3 SMC/ICT & regime analyzers used to evaluate declarative rule predicates
        self._regime_detector = MarketRegimeDetector()
        self._market_structure = MarketStructureStrategy()
        self._liquidity = LiquiditySweepStrategy()
        self._fvg_detector = FVGDetectionStrategy()
        self._fvg_retest = FVGRetestStrategy()
        self._pd_analyzer = PremiumDiscountAnalyzer()
        self._pdh_pdl_analyzer = PreviousDayLevelsAnalyzer()
        self._killzone_analyzer = KillZoneSessionAnalyzer()

    @property
    def strategy_id(self) -> str:
        return "knowledge_based_strategy"

    @property
    def name(self) -> str:
        return "Educational Knowledge-Based Rule Strategy"

    @property
    def required_features(self) -> tuple[str, ...]:
        return ("timestamp", "open", "high", "low", "close", "atr_14")

    @property
    def repository(self) -> KnowledgeRepository:
        return self._repository

    def evaluate_with_context(
        self,
        df: pd.DataFrame,
        *,
        as_of_index: int | None = None,
        symbol: str | None = None,
        timeframe: Timeframe | None = None,
    ) -> StrategyEvaluationResult:
        """Evaluate approved knowledge rules for `(symbol, timeframe)` up to `as_of_index`."""
        sliced = self.prepare_causal_slice(df, as_of_index)
        last_row = sliced.iloc[-1]
        ts = self.extract_timestamp(last_row)

        target_sym = (symbol or self._symbol).strip().upper()
        target_tf = timeframe or self._timeframe

        # Retrieve ONLY explicitly APPROVED rules for target_sym and target_tf
        approved_rules = [
            r
            for r in self._repository.get_approved_rules(symbol=target_sym, timeframe=target_tf)
            if r.approval_status == RuleApprovalStatus.APPROVED
            and r.direction in (SignalDirection.BUY, SignalDirection.SELL)
            and len(r.conditions) > 0
        ]

        if not approved_rules:
            return self.build_wait_result(
                timestamp=ts,
                rationale=(
                    f"No APPROVED educational trading rules found for {target_sym} {target_tf.value}."
                ),
                rejection_reasons=["NO_APPROVED_KNOWLEDGE_RULES"],
                supporting_evidence={
                    "symbol": target_sym,
                    "timeframe": target_tf.value,
                    "matched_rule_ids": [],
                    "source_document_ids": [],
                    "matched_conditions": {},
                },
            )

        nan_cols = self.has_warmup_nan(last_row, ("atr_14",))
        if nan_cols:
            return self.build_wait_result(
                timestamp=ts,
                rationale=f"Warm-up NaN in required ATR column: {nan_cols}.",
                rejection_reasons=["WARMUP_NAN"],
                supporting_evidence={
                    "matched_rule_ids": [],
                    "source_document_ids": [],
                    "matched_conditions": {},
                },
            )

        # Compute causal market state snapshots once for the slice
        market_ctx = self._build_causal_market_context(sliced, last_row, ts)

        matched_buy_rules: list[ExtractedTradingRule] = []
        matched_sell_rules: list[ExtractedTradingRule] = []
        matched_conditions_by_rule: dict[str, list[str]] = {}

        for rule in approved_rules:
            # Re-validate safety invariants before evaluating any rule
            self._validator.validate_rule(rule)

            all_passed = True
            passed_descriptions: list[str] = []
            for cond in rule.conditions:
                passed, desc = self._evaluate_condition(cond, last_row, market_ctx)
                if not passed:
                    all_passed = False
                    break
                passed_descriptions.append(desc)

            if all_passed:
                matched_conditions_by_rule[rule.rule_id] = passed_descriptions
                if rule.direction == SignalDirection.BUY:
                    matched_buy_rules.append(rule)
                elif rule.direction == SignalDirection.SELL:
                    matched_sell_rules.append(rule)

        if not matched_buy_rules and not matched_sell_rules:
            return self.build_wait_result(
                timestamp=ts,
                rationale=(
                    f"Evaluated {len(approved_rules)} APPROVED knowledge rule(s) for "
                    f"{target_sym} {target_tf.value}; no rule had all conditions satisfied."
                ),
                rejection_reasons=["NO_KNOWLEDGE_RULE_CONDITIONS_MET"],
                supporting_evidence={
                    "evaluated_approved_rule_count": len(approved_rules),
                    "matched_rule_ids": [],
                    "source_document_ids": [],
                    "matched_conditions": {},
                },
            )

        if matched_buy_rules and matched_sell_rules:
            all_matched = matched_buy_rules + matched_sell_rules
            return self.build_wait_result(
                timestamp=ts,
                rationale=(
                    f"Conflicting APPROVED knowledge rules matched simultaneously: "
                    f"BUY={[r.rule_id for r in matched_buy_rules]} vs "
                    f"SELL={[r.rule_id for r in matched_sell_rules]}; returning WAIT."
                ),
                rejection_reasons=["CONFLICTING_KNOWLEDGE_RULES"],
                supporting_evidence={
                    "matched_rule_ids": [r.rule_id for r in all_matched],
                    "source_document_ids": sorted({r.source_document_id for r in all_matched}),
                    "source_chunk_ids": sorted({r.source_chunk_id for r in all_matched}),
                    "matched_conditions": matched_conditions_by_rule,
                },
            )

        winning_rules = matched_buy_rules if matched_buy_rules else matched_sell_rules
        winning_dir = SignalDirection.BUY if matched_buy_rules else SignalDirection.SELL

        # Deterministically select highest base_confidence rule (tie-break by rule_id)
        best_rule = sorted(
            winning_rules,
            key=lambda r: (r.base_confidence, r.rule_id),
            reverse=True,
        )[0]

        close_price = self.to_decimal(last_row["close"])
        atr_val = self.to_decimal(last_row["atr_14"])
        sl_dist = (atr_val * best_rule.stop_loss_atr_multiplier).quantize(
            Decimal("0.01"), rounding=ROUND_DOWN
        )
        if sl_dist <= Decimal("0.00"):
            sl_dist = Decimal("0.50")
        tp_dist = (sl_dist * best_rule.minimum_rr_ratio).quantize(
            Decimal("0.01"), rounding=ROUND_DOWN
        )

        if winning_dir == SignalDirection.BUY:
            sl_price = close_price - sl_dist
            tp_price = close_price + tp_dist
        else:
            sl_price = close_price + sl_dist
            tp_price = close_price - tp_dist

        # Small confidence bonus when multiple distinct approved rules corroborate the same direction
        corroboration_bonus = min(
            Decimal("0.15"),
            Decimal("0.05") * Decimal(len(winning_rules) - 1),
        )
        final_conf = min(Decimal("1.00"), best_rule.base_confidence + corroboration_bonus)

        matched_ids = [r.rule_id for r in winning_rules]
        source_docs = sorted({r.source_document_id for r in winning_rules})
        source_chunks = sorted({r.source_chunk_id for r in winning_rules})

        return StrategyEvaluationResult(
            strategy_id=self.strategy_id,
            direction=winning_dir,
            timestamp=ts,
            entry_price=close_price,
            stop_loss=sl_price,
            take_profit=tp_price,
            confidence=final_conf,
            rationale=(
                f"Knowledge rule(s) {matched_ids} from document(s) {source_docs} matched "
                f"for {winning_dir.value} ({best_rule.title})."
            ),
            supporting_evidence={
                "symbol": target_sym,
                "timeframe": target_tf.value,
                "selected_rule_id": best_rule.rule_id,
                "selected_rule_title": best_rule.title,
                "matched_rule_ids": matched_ids,
                "source_document_ids": source_docs,
                "source_chunk_ids": source_chunks,
                "matched_conditions": matched_conditions_by_rule,
            },
            rejection_reasons=(),
        )

    def evaluate(
        self,
        df: pd.DataFrame,
        as_of_index: int | None = None,
    ) -> StrategyEvaluationResult:
        """Evaluate approved knowledge rules using the strategy's default symbol and timeframe."""
        return self.evaluate_with_context(df, as_of_index=as_of_index)

    def _build_causal_market_context(
        self,
        sliced: pd.DataFrame,
        last_row: pd.Series,
        ts: object,
    ) -> dict[str, object]:
        """Compute causal SMC/ICT states on `sliced` for predicate matching."""
        # Regime requires full volatility/MA columns; degrade gracefully to UNKNOWN if absent
        regime_val = "UNKNOWN"
        if all(c in sliced.columns for c in self._regime_detector.required_features):
            regime_enum, _ = self._regime_detector.detect_regime(sliced)
            regime_val = regime_enum.value

        pd_state, _ = self._pd_analyzer.analyze_state(sliced)
        ms_res = self._market_structure.evaluate(sliced)
        liq_res = self._liquidity.evaluate(sliced)
        fvg_det_res = self._fvg_detector.evaluate(sliced)
        fvg_ret_res = self._fvg_retest.evaluate(sliced)
        pdl_state, _ = self._pdh_pdl_analyzer.analyze_levels(sliced)
        kz_info = self._killzone_analyzer.classify_bar_time(self.extract_timestamp(last_row))
        kz_res = self._killzone_analyzer.evaluate(sliced)

        return {
            "regime": regime_val,
            "pd_zone": pd_state.zone.value,
            "structure_event": str(ms_res.supporting_evidence.get("structure_event", "NONE")),
            "ms_direction": ms_res.direction.value,
            "liq_direction": liq_res.direction.value,
            "fvg_det_direction": fvg_det_res.direction.value,
            "fvg_ret_direction": fvg_ret_res.direction.value,
            "pdl_swept_and_rejected": pdl_state.pdl_swept_and_rejected,
            "pdh_swept_and_rejected": pdl_state.pdh_swept_and_rejected,
            "active_killzone": kz_info.get("active_killzone"),
            "swept_prior_session_low": bool(
                kz_res.supporting_evidence.get("swept_prior_session_low", False)
            ),
            "swept_prior_session_high": bool(
                kz_res.supporting_evidence.get("swept_prior_session_high", False)
            ),
        }

    def _evaluate_condition(
        self,
        condition: KnowledgeRuleCondition,
        last_row: pd.Series,
        market_ctx: dict[str, object],
    ) -> tuple[bool, str]:
        """Deterministically evaluate a single KnowledgeRuleCondition without eval() or exec()."""
        ctype = condition.condition_type
        param = condition.parameter.strip().upper()

        if ctype == ConditionType.REQUIRED_REGIME:
            actual = str(market_ctx.get("regime", "UNKNOWN")).upper()
            return (actual == param, f"REQUIRED_REGIME({param} == {actual})")

        if ctype == ConditionType.REQUIRED_PRICE_ZONE:
            actual = str(market_ctx.get("pd_zone", "UNKNOWN")).upper()
            return (actual == param, f"REQUIRED_PRICE_ZONE({param} == {actual})")

        if ctype == ConditionType.REQUIRED_STRUCTURE_EVENT:
            actual_ev = str(market_ctx.get("structure_event", "NONE")).upper()
            ms_dir = str(market_ctx.get("ms_direction", "WAIT")).upper()
            if param == "ANY_BULLISH_STRUCTURE":
                ok = actual_ev in ("BULLISH_BOS", "BULLISH_CHOCH") or ms_dir == "BUY"
            elif param == "ANY_BEARISH_STRUCTURE":
                ok = actual_ev in ("BEARISH_BOS", "BEARISH_CHOCH") or ms_dir == "SELL"
            else:
                ok = actual_ev == param
            return (ok, f"REQUIRED_STRUCTURE_EVENT({param} vs {actual_ev})")

        if ctype == ConditionType.REQUIRED_LIQUIDITY_SWEEP:
            if param == "SSL_OR_EQL_SWEEP":
                ok = market_ctx.get("liq_direction") == "BUY"
            elif param == "BSL_OR_EQH_SWEEP":
                ok = market_ctx.get("liq_direction") == "SELL"
            elif param == "PDL_SWEEP":
                ok = bool(market_ctx.get("pdl_swept_and_rejected", False))
            elif param == "PDH_SWEEP":
                ok = bool(market_ctx.get("pdh_swept_and_rejected", False))
            elif param == "PRIOR_SESSION_LOW_SWEEP":
                ok = bool(market_ctx.get("swept_prior_session_low", False))
            elif param == "PRIOR_SESSION_HIGH_SWEEP":
                ok = bool(market_ctx.get("swept_prior_session_high", False))
            else:
                ok = False
            return (ok, f"REQUIRED_LIQUIDITY_SWEEP({param})")

        if ctype == ConditionType.REQUIRED_FVG:
            if param == "BULLISH_FVG_RETEST":
                ok = market_ctx.get("fvg_ret_direction") == "BUY"
            elif param == "BEARISH_FVG_RETEST":
                ok = market_ctx.get("fvg_ret_direction") == "SELL"
            elif param == "BULLISH_FVG_ACTIVE":
                ok = (
                    market_ctx.get("fvg_det_direction") == "BUY"
                    or market_ctx.get("fvg_ret_direction") == "BUY"
                )
            elif param == "BEARISH_FVG_ACTIVE":
                ok = (
                    market_ctx.get("fvg_det_direction") == "SELL"
                    or market_ctx.get("fvg_ret_direction") == "SELL"
                )
            else:
                ok = False
            return (ok, f"REQUIRED_FVG({param})")

        if ctype == ConditionType.REQUIRED_KILLZONE:
            active_kz = market_ctx.get("active_killzone")
            if param == "ANY_KILL_ZONE":
                ok = active_kz in ("LONDON_KILL_ZONE", "NEW_YORK_KILL_ZONE")
            else:
                ok = active_kz == param
            return (ok, f"REQUIRED_KILLZONE({param} vs {active_kz})")

        if ctype == ConditionType.FEATURE_COMPARISON:
            lhs_col = condition.parameter.strip()
            if lhs_col not in last_row or self.has_warmup_nan(last_row, (lhs_col,)):
                return (False, f"FEATURE_COMPARISON({lhs_col} missing/NaN)")
            lhs_val = Decimal(str(last_row[lhs_col]))

            if condition.target_feature is not None:
                rhs_col = condition.target_feature.strip()
                if rhs_col not in last_row or self.has_warmup_nan(last_row, (rhs_col,)):
                    return (False, f"FEATURE_COMPARISON({rhs_col} missing/NaN)")
                rhs_val = Decimal(str(last_row[rhs_col]))
                rhs_label = f"{rhs_col}({rhs_val})"
            elif condition.threshold_value is not None:
                rhs_val = condition.threshold_value
                rhs_label = str(rhs_val)
            else:
                return (False, "FEATURE_COMPARISON(missing RHS)")

            op = condition.operator
            if op == ComparisonOperator.GT:
                ok = lhs_val > rhs_val
            elif op == ComparisonOperator.GE:
                ok = lhs_val >= rhs_val
            elif op == ComparisonOperator.LT:
                ok = lhs_val < rhs_val
            elif op == ComparisonOperator.LE:
                ok = lhs_val <= rhs_val
            else:
                ok = lhs_val == rhs_val
            return (ok, f"FEATURE_COMPARISON({lhs_col}({lhs_val}) {op.value} {rhs_label})")

        return (False, f"UNSUPPORTED_CONDITION({ctype})")
