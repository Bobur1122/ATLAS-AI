"""Multi-Strategy Confluence Engine for ATLAS AI."""

from decimal import Decimal, ROUND_DOWN
from typing import Mapping

from atlas_ai.domain.signals import SignalDirection
from atlas_ai.strategies.config import ConfluenceWeightsConfig
from atlas_ai.strategies.models import (
    ConfluenceResult,
    MultiTimeframeEvaluation,
    NewsFilterResult,
    StrategyEvaluationResult,
)

QUANTIZE_SCORE = Decimal("0.0001")


class ConfluenceEngine:
    r"""Combines primary directional strategies, contextual SMC/ICT filters, MTF alignment, and NewsFilter.

    Deterministic Scoring Formula:
    1. Primary Directional Modules (`ema_trend_following`, `smc_market_structure`, `liquidity_sweep`,
       `fvg_detection`, `fvg_retest`, and `multi_timeframe_analyzer` when enabled):
       - Let $W_{\text{total}} = \sum_{m \in \text{EnabledPrimary}} w_m$.
       - Note on correlated FVG evidence: if both `fvg_detection` and `fvg_retest` fire in the same
         direction on the same bar, only the higher-weighted `fvg_retest` contribution is counted so
         correlated FVG evidence is never double-counted.
       - $\text{buy\_score} = \frac{\sum_{m:\, d_m = \text{BUY}} w_m \cdot c_m}{W_{\text{total}}}$.
       - $\text{sell\_score} = \frac{\sum_{m:\, d_m = \text{SELL}} w_m \cdot c_m}{W_{\text{total}}}$.
       - $\text{wait\_weight\_ratio} = \frac{\sum_{m:\, d_m = \text{WAIT}} w_m}{W_{\text{total}}}$.
    2. Directional Conflict Check:
       - If both BUY and SELL primary votes exist and $|\text{buy\_score} - \text{sell\_score}| < \text{minimum\_directional\_edge}$,
         or if `mtf_evaluation.has_conflict` is True, the engine records the conflict and returns `WAIT`.
    3. Active Agreement Normalization & Contextual Confluence Modifiers:
       - For the winning primary direction $D \in \{\text{BUY}, \text{SELL}\}$, base confidence combines
         the active weighted confidence and total weight coverage:
         $\text{base\_conf} = 0.65 \cdot \left(\frac{\sum_{m:\, d_m = D} w_m \cdot c_m}{\sum_{m:\, d_m = D} w_m}\right) + 0.35 \cdot \text{score}_D$.
       - Contextual Filters (`premium_discount_analyzer`, `pdh_pdl_analyzer`, `killzone_session_analyzer`)
         do NOT trigger trades on their own; they add `+contextual_alignment_bonus` when aligned with $D$
         and subtract `-contextual_conflict_penalty` when opposing $D$ (e.g., trying to BUY in PREMIUM).
    4. News Filter Override:
       - If `news_result.allow_new_entries` is False, direction is forced to `WAIT`.
       - Otherwise, $\text{final\_confidence} = \text{clamp}(\text{adjusted\_conf} \times \text{news\_multiplier}, 0, 1)$.
    5. Final Threshold Gate:
       - If $\text{final\_confidence} < \text{minimum\_confidence\_threshold}$, returns `WAIT`.
    """

    def __init__(self, config: ConfluenceWeightsConfig | None = None) -> None:
        self._config = config or ConfluenceWeightsConfig()

    @property
    def config(self) -> ConfluenceWeightsConfig:
        return self._config

    def evaluate(
        self,
        *,
        evaluations: Mapping[str, StrategyEvaluationResult],
        mtf_evaluation: MultiTimeframeEvaluation | None = None,
        news_result: NewsFilterResult | None = None,
    ) -> ConfluenceResult:
        """Compute deterministic confluence score and directional decision."""
        cfg = self._config
        conflicts: list[str] = []
        missing_evidence: list[str] = []
        rejection_reasons: list[str] = []

        # Map primary module IDs to (enabled_flag, configured_weight)
        primary_spec: list[tuple[str, bool, Decimal]] = [
            ("ema_trend_following", cfg.enable_ema_trend, cfg.weight_ema_trend),
            ("smc_market_structure", cfg.enable_market_structure, cfg.weight_market_structure),
            ("liquidity_sweep", cfg.enable_liquidity_sweep, cfg.weight_liquidity_sweep),
            ("fvg_detection", cfg.enable_fvg, cfg.weight_fvg),
            ("fvg_retest", cfg.enable_fvg_retest, cfg.weight_fvg_retest),
            ("multi_timeframe_analyzer", cfg.enable_mtf, cfg.weight_mtf),
            ("knowledge_based_strategy", cfg.enable_knowledge_strategy, cfg.weight_knowledge_strategy),
            ("ml_signal_strategy", cfg.enable_ml_strategy, cfg.weight_ml_strategy),
        ]

        # Anti-double-counting rule for correlated FVG modules:
        # If both fvg_detection and fvg_retest are present and non-WAIT in the same direction,
        # suppress fvg_detection weight so FVG is not double-counted.
        fvg_det_ev = evaluations.get("fvg_detection")
        fvg_ret_ev = evaluations.get("fvg_retest")
        suppress_fvg_det = (
            fvg_det_ev is not None
            and fvg_ret_ev is not None
            and fvg_det_ev.direction != SignalDirection.WAIT
            and fvg_det_ev.direction == fvg_ret_ev.direction
        )

        total_weight = Decimal("0")
        buy_weighted_sum = Decimal("0")
        sell_weighted_sum = Decimal("0")
        buy_active_weight = Decimal("0")
        sell_active_weight = Decimal("0")
        wait_weight = Decimal("0")

        buy_modules: list[str] = []
        sell_modules: list[str] = []

        for mod_id, is_enabled, weight in primary_spec:
            if not is_enabled or weight <= Decimal("0"):
                continue
            if mod_id == "fvg_detection" and suppress_fvg_det:
                continue

            ev = evaluations.get(mod_id)
            if ev is None:
                missing_evidence.append(mod_id)
                total_weight += weight
                wait_weight += weight
                continue

            total_weight += weight
            if ev.direction == SignalDirection.BUY:
                buy_weighted_sum += weight * ev.confidence
                buy_active_weight += weight
                buy_modules.append(mod_id)
            elif ev.direction == SignalDirection.SELL:
                sell_weighted_sum += weight * ev.confidence
                sell_active_weight += weight
                sell_modules.append(mod_id)
            else:
                wait_weight += weight
                if "WARMUP_NAN" in ev.rejection_reasons or "INSUFFICIENT_WARMUP_DATA" in ev.rejection_reasons:
                    missing_evidence.append(f"{mod_id}:WARMUP")

        if mtf_evaluation is not None and cfg.enable_mtf:
            for m_tf in mtf_evaluation.missing_timeframes:
                missing_evidence.append(f"MTF_MISSING:{m_tf.value}")
            if mtf_evaluation.has_conflict:
                conflicts.append("MTF_DIRECTIONAL_CONFLICT")

        if total_weight <= Decimal("0"):
            return ConfluenceResult(
                direction=SignalDirection.WAIT,
                rejection_reasons=("NO_ENABLED_PRIMARY_MODULES",),
                missing_evidence=tuple(missing_evidence),
                component_evaluations=dict(evaluations),
                news_result=news_result,
                rationale="All primary directional modules are disabled or have zero weight.",
            )

        buy_score = (buy_weighted_sum / total_weight).quantize(QUANTIZE_SCORE, rounding=ROUND_DOWN)
        sell_score = (sell_weighted_sum / total_weight).quantize(QUANTIZE_SCORE, rounding=ROUND_DOWN)
        wait_ratio = (wait_weight / total_weight).quantize(QUANTIZE_SCORE, rounding=ROUND_DOWN)
        net_edge = abs(buy_score - sell_score).quantize(QUANTIZE_SCORE, rounding=ROUND_DOWN)

        if buy_modules and sell_modules:
            conflicts.append(
                f"PRIMARY_STRATEGY_CONFLICT: BUY({','.join(buy_modules)}) vs SELL({','.join(sell_modules)})"
            )

        # News filter hard override check
        if cfg.enable_news_filter and news_result is not None and not news_result.allow_new_entries:
            rejection_reasons.append(f"NEWS_FILTER_BLOCKED:{news_result.status.value}")
            return ConfluenceResult(
                direction=SignalDirection.WAIT,
                buy_score=buy_score,
                sell_score=sell_score,
                wait_weight_ratio=wait_ratio,
                net_directional_edge=net_edge,
                final_confidence=Decimal("0"),
                directional_agreement_count=0,
                conflicts=tuple(conflicts),
                missing_evidence=tuple(missing_evidence),
                rejection_reasons=tuple(rejection_reasons),
                component_evaluations=dict(evaluations),
                news_result=news_result,
                rationale=f"Confluence overridden by NewsFilter: {news_result.rationale}",
            )

        # Check KillZone hard requirement if enabled
        kz_ev = evaluations.get("killzone_session_analyzer")
        if cfg.enable_kill_zone and kz_ev is not None and "OUTSIDE_REQUIRED_KILLZONE" in kz_ev.rejection_reasons:
            rejection_reasons.append("OUTSIDE_REQUIRED_KILLZONE")
            return ConfluenceResult(
                direction=SignalDirection.WAIT,
                buy_score=buy_score,
                sell_score=sell_score,
                wait_weight_ratio=wait_ratio,
                net_directional_edge=net_edge,
                final_confidence=Decimal("0"),
                directional_agreement_count=0,
                conflicts=tuple(conflicts),
                missing_evidence=tuple(missing_evidence),
                rejection_reasons=tuple(rejection_reasons),
                component_evaluations=dict(evaluations),
                news_result=news_result,
                rationale=kz_ev.rationale,
            )

        # Check MTF hard conflict
        if "MTF_DIRECTIONAL_CONFLICT" in conflicts:
            rejection_reasons.append("MTF_DIRECTIONAL_CONFLICT")
            return ConfluenceResult(
                direction=SignalDirection.WAIT,
                buy_score=buy_score,
                sell_score=sell_score,
                wait_weight_ratio=wait_ratio,
                net_directional_edge=net_edge,
                final_confidence=Decimal("0"),
                directional_agreement_count=0,
                conflicts=tuple(conflicts),
                missing_evidence=tuple(missing_evidence),
                rejection_reasons=tuple(rejection_reasons),
                component_evaluations=dict(evaluations),
                news_result=news_result,
                rationale="Strong Multi-Timeframe directional conflict detected; returning WAIT.",
            )

        # Determine winning candidate direction
        if buy_score == Decimal("0") and sell_score == Decimal("0"):
            rejection_reasons.append("NO_DIRECTIONAL_SIGNALS")
            return ConfluenceResult(
                direction=SignalDirection.WAIT,
                buy_score=buy_score,
                sell_score=sell_score,
                wait_weight_ratio=wait_ratio,
                net_directional_edge=Decimal("0"),
                final_confidence=Decimal("0"),
                directional_agreement_count=0,
                conflicts=tuple(conflicts),
                missing_evidence=tuple(missing_evidence),
                rejection_reasons=tuple(rejection_reasons),
                component_evaluations=dict(evaluations),
                news_result=news_result,
                rationale="No primary directional strategies voted BUY or SELL; returning WAIT.",
            )

        if buy_modules and sell_modules and net_edge < cfg.minimum_directional_edge:
            rejection_reasons.append("CONFLICTING_EVIDENCE_BELOW_EDGE_THRESHOLD")
            return ConfluenceResult(
                direction=SignalDirection.WAIT,
                buy_score=buy_score,
                sell_score=sell_score,
                wait_weight_ratio=wait_ratio,
                net_directional_edge=net_edge,
                final_confidence=Decimal("0"),
                directional_agreement_count=max(len(buy_modules), len(sell_modules)),
                conflicts=tuple(conflicts),
                missing_evidence=tuple(missing_evidence),
                rejection_reasons=tuple(rejection_reasons),
                component_evaluations=dict(evaluations),
                news_result=news_result,
                rationale=(
                    f"Contradictory BUY ({buy_score}) and SELL ({sell_score}) evidence with net edge "
                    f"({net_edge}) < minimum_directional_edge ({cfg.minimum_directional_edge})."
                ),
            )

        if buy_score > sell_score:
            candidate_dir = SignalDirection.BUY
            agree_count = len(buy_modules)
            avg_active_conf = buy_weighted_sum / buy_active_weight
            raw_dir_score = buy_score
            opposing_score = sell_score
        else:
            candidate_dir = SignalDirection.SELL
            agree_count = len(sell_modules)
            avg_active_conf = sell_weighted_sum / sell_active_weight
            raw_dir_score = sell_score
            opposing_score = buy_score

        if agree_count < cfg.minimum_agreeing_directional_modules:
            rejection_reasons.append("INSUFFICIENT_AGREEING_MODULES")
            return ConfluenceResult(
                direction=SignalDirection.WAIT,
                buy_score=buy_score,
                sell_score=sell_score,
                wait_weight_ratio=wait_ratio,
                net_directional_edge=net_edge,
                final_confidence=Decimal("0"),
                directional_agreement_count=agree_count,
                conflicts=tuple(conflicts),
                missing_evidence=tuple(missing_evidence),
                rejection_reasons=tuple(rejection_reasons),
                component_evaluations=dict(evaluations),
                news_result=news_result,
                rationale=(
                    f"Only {agree_count} primary module(s) agreed on {candidate_dir.value}, "
                    f"below minimum_agreeing_directional_modules ({cfg.minimum_agreeing_directional_modules})."
                ),
            )

        # Base confidence combines average confidence of active agreeing modules (65%) and total weight coverage (35%),
        # minus any opposing primary score
        base_conf = (
            Decimal("0.65") * avg_active_conf + Decimal("0.35") * raw_dir_score - Decimal("0.50") * opposing_score
        )

        # Apply contextual filter adjustments (Premium/Discount, PDH/PDL, KillZone)
        context_spec: list[tuple[str, bool]] = [
            ("premium_discount_analyzer", cfg.enable_premium_discount),
            ("pdh_pdl_analyzer", cfg.enable_pdh_pdl),
            ("killzone_session_analyzer", cfg.enable_kill_zone),
        ]
        for ctx_id, ctx_enabled in context_spec:
            if not ctx_enabled:
                continue
            ctx_ev = evaluations.get(ctx_id)
            if ctx_ev is None:
                continue
            if ctx_ev.direction == candidate_dir:
                base_conf += cfg.contextual_alignment_bonus
            elif ctx_ev.direction in (SignalDirection.BUY, SignalDirection.SELL) and ctx_ev.direction != candidate_dir:
                base_conf -= cfg.contextual_conflict_penalty
                conflicts.append(f"CONTEXT_CONFLICT:{ctx_id}_FAVORS_{ctx_ev.direction.value}")

        # Apply NewsFilter confidence multiplier (e.g., 0.75 during MEDIUM impact window)
        if cfg.enable_news_filter and news_result is not None:
            base_conf = base_conf * news_result.confidence_multiplier

        final_conf = max(Decimal("0"), min(Decimal("1"), base_conf)).quantize(
            QUANTIZE_SCORE, rounding=ROUND_DOWN
        )

        if final_conf < cfg.minimum_confidence_threshold:
            rejection_reasons.append("CONFIDENCE_BELOW_THRESHOLD")
            return ConfluenceResult(
                direction=SignalDirection.WAIT,
                buy_score=buy_score,
                sell_score=sell_score,
                wait_weight_ratio=wait_ratio,
                net_directional_edge=net_edge,
                final_confidence=final_conf,
                directional_agreement_count=agree_count,
                conflicts=tuple(conflicts),
                missing_evidence=tuple(missing_evidence),
                rejection_reasons=tuple(rejection_reasons),
                component_evaluations=dict(evaluations),
                news_result=news_result,
                rationale=(
                    f"Confluence confidence ({final_conf}) for {candidate_dir.value} is below "
                    f"minimum_confidence_threshold ({cfg.minimum_confidence_threshold})."
                ),
            )

        return ConfluenceResult(
            direction=candidate_dir,
            buy_score=buy_score,
            sell_score=sell_score,
            wait_weight_ratio=wait_ratio,
            net_directional_edge=net_edge,
            final_confidence=final_conf,
            directional_agreement_count=agree_count,
            conflicts=tuple(conflicts),
            missing_evidence=tuple(missing_evidence),
            rejection_reasons=(),
            component_evaluations=dict(evaluations),
            news_result=news_result,
            rationale=(
                f"Confluence APPROVED {candidate_dir.value} with confidence={final_conf} "
                f"(buy_score={buy_score}, sell_score={sell_score}, agreeing_modules={agree_count})."
            ),
        )
