"""Deterministic schema, feature whitelist, and trading safety validator for extracted knowledge rules."""

import json
import re

from atlas_ai.domain.signals import SignalDirection
from atlas_ai.knowledge.config import KnowledgeConfig
from atlas_ai.knowledge.exceptions import KnowledgeSafetyError, KnowledgeValidationError
from atlas_ai.knowledge.models import (
    ConditionType,
    ExtractedTradingRule,
    KnowledgeRuleCondition,
    RuleApprovalStatus,
)

# Strictly forbidden patterns in any rule field or metadata (anti-code-execution, anti-lookahead, anti-risk-bypass)
FORBIDDEN_CODE_PATTERNS = re.compile(
    r"(?:\beval\s*\(|\bexec\s*\(|__import__|\bcompile\s*\(|\bsubprocess\b|\bos\.system\b|\bsys\.modules\b|\blambda\s+)",
    re.IGNORECASE,
)
FORBIDDEN_LOOKAHEAD_PATTERNS = re.compile(
    r"(?:\bnext_bar\b|\bfuture_close\b|\bfuture_high\b|\bfuture_low\b|\blookahead\b|\btomorrow_close\b|shift\s*\(\s*-\d+|\biloc\s*\[\s*\+)",
    re.IGNORECASE,
)
FORBIDDEN_RISK_BYPASS_PATTERNS = re.compile(
    r"(?:\boverride_risk\b|\bbypass_risk\b|\bdisable_risk_engine\b|\bno_stop_loss\b|\bignore_daily_loss\b|\bignore_drawdown\b|\bunlimited_lot|\benable_live_trading\b|\blive_trading_enabled\s*=\s*true\b|\bexecution_mode\s*=\s*['\"]?live\b)",
    re.IGNORECASE,
)

ALLOWED_REGIME_PARAMS: frozenset[str] = frozenset(
    {"UPTREND", "DOWNTREND", "RANGING", "HIGH_VOLATILITY", "LOW_VOLATILITY"}
)
ALLOWED_PRICE_ZONE_PARAMS: frozenset[str] = frozenset(
    {"DISCOUNT", "PREMIUM", "EQUILIBRIUM"}
)
ALLOWED_STRUCTURE_PARAMS: frozenset[str] = frozenset(
    {
        "BULLISH_BOS",
        "BEARISH_BOS",
        "BULLISH_CHOCH",
        "BEARISH_CHOCH",
        "ANY_BULLISH_STRUCTURE",
        "ANY_BEARISH_STRUCTURE",
    }
)
ALLOWED_LIQUIDITY_PARAMS: frozenset[str] = frozenset(
    {
        "SSL_OR_EQL_SWEEP",
        "BSL_OR_EQH_SWEEP",
        "PDL_SWEEP",
        "PDH_SWEEP",
        "PRIOR_SESSION_LOW_SWEEP",
        "PRIOR_SESSION_HIGH_SWEEP",
    }
)
ALLOWED_FVG_PARAMS: frozenset[str] = frozenset(
    {
        "BULLISH_FVG_RETEST",
        "BEARISH_FVG_RETEST",
        "BULLISH_FVG_ACTIVE",
        "BEARISH_FVG_ACTIVE",
    }
)
ALLOWED_KILLZONE_PARAMS: frozenset[str] = frozenset(
    {"LONDON_KILL_ZONE", "NEW_YORK_KILL_ZONE", "ANY_KILL_ZONE"}
)


class KnowledgeRuleValidator:
    """Validates ExtractedTradingRule instances against schema, feature whitelist, and RiskEngine safety laws."""

    def __init__(self, config: KnowledgeConfig | None = None) -> None:
        self._config = config or KnowledgeConfig()

    @property
    def config(self) -> KnowledgeConfig:
        return self._config

    def validate_rule(self, rule: ExtractedTradingRule) -> ExtractedTradingRule:
        """Validate a rule thoroughly.

        Raises:
            KnowledgeSafetyError: On any code injection, future data reference, stop-loss omission,
                risk-parameter override attempt, or live trading enablement attempt.
            KnowledgeValidationError: On invalid feature names, unsupported condition parameters,
                invalid RR/confidence thresholds, or attempting to mark an empty/WAIT rule as APPROVED.
        """
        cfg = self._config

        # 1. Scan all text and metadata for forbidden code, lookahead, or risk-bypass patterns
        text_blobs = [
            rule.rule_id,
            rule.title,
            rule.description,
            " ".join(rule.extraction_notes),
            json.dumps(rule.metadata, default=str),
        ]
        for cond in rule.conditions:
            text_blobs.append(cond.parameter)
            text_blobs.append(cond.rationale)
            if cond.target_feature:
                text_blobs.append(cond.target_feature)

        combined_blob = "\n".join(text_blobs)

        if FORBIDDEN_CODE_PATTERNS.search(combined_blob):
            raise KnowledgeSafetyError(
                f"SAFETY VIOLATION in rule '{rule.rule_id}': Arbitrary code execution expression detected."
            )
        if FORBIDDEN_LOOKAHEAD_PATTERNS.search(combined_blob):
            raise KnowledgeSafetyError(
                f"SAFETY VIOLATION in rule '{rule.rule_id}': Future-data or lookahead reference detected."
            )
        if FORBIDDEN_RISK_BYPASS_PATTERNS.search(combined_blob):
            raise KnowledgeSafetyError(
                f"SAFETY VIOLATION in rule '{rule.rule_id}': Attempt to override RiskEngine or enable live trading."
            )

        # Check explicit metadata keys for risk or live trading overrides
        forbidden_meta_keys = {
            "override_risk",
            "bypass_risk",
            "allow_live_trading",
            "live_trading_enabled",
            "max_daily_loss_percent",
            "max_total_drawdown_percent",
            "risk_per_trade_percent",
            "max_open_positions",
        }
        intersect = forbidden_meta_keys.intersection({k.lower() for k in rule.metadata})
        if intersect:
            raise KnowledgeSafetyError(
                f"SAFETY VIOLATION in rule '{rule.rule_id}': Rule metadata cannot override RiskParameters "
                f"or live trading flags ({sorted(intersect)})."
            )

        # 2. Stop-loss ATR multiplier safety validation
        if rule.stop_loss_atr_multiplier <= 0:
            raise KnowledgeSafetyError(
                f"SAFETY VIOLATION in rule '{rule.rule_id}': stop_loss_atr_multiplier ({rule.stop_loss_atr_multiplier}) "
                "must be strictly positive."
            )
        if not (
            cfg.min_allowed_sl_atr_multiplier
            <= rule.stop_loss_atr_multiplier
            <= cfg.max_allowed_sl_atr_multiplier
        ):
            raise KnowledgeValidationError(
                f"Rule '{rule.rule_id}' stop_loss_atr_multiplier ({rule.stop_loss_atr_multiplier}) "
                f"outside allowed bounds [{cfg.min_allowed_sl_atr_multiplier}, {cfg.max_allowed_sl_atr_multiplier}]."
            )

        # 3. Minimum Reward-to-Risk and confidence validation
        if not (cfg.min_allowed_rr_ratio <= rule.minimum_rr_ratio <= cfg.max_allowed_rr_ratio):
            raise KnowledgeValidationError(
                f"Rule '{rule.rule_id}' minimum_rr_ratio ({rule.minimum_rr_ratio}) "
                f"outside allowed bounds [{cfg.min_allowed_rr_ratio}, {cfg.max_allowed_rr_ratio}]."
            )

        if not rule.applicable_timeframes:
            raise KnowledgeValidationError(
                f"Rule '{rule.rule_id}' must specify at least one applicable_timeframe."
            )

        # 4. Validate every declarative condition
        for cond in rule.conditions:
            self._validate_condition(rule.rule_id, cond)

        # 5. Approval status gate: VALIDATED or APPROVED rules must be executable (BUY or SELL) and have >= 1 condition
        if rule.approval_status in (RuleApprovalStatus.VALIDATED, RuleApprovalStatus.APPROVED):
            if rule.direction not in (SignalDirection.BUY, SignalDirection.SELL):
                raise KnowledgeValidationError(
                    f"Rule '{rule.rule_id}' with direction={rule.direction.value} cannot be "
                    f"{rule.approval_status.value}; non-directional notes must remain DRAFT."
                )
            if not rule.conditions:
                raise KnowledgeValidationError(
                    f"Rule '{rule.rule_id}' has zero conditions and cannot be {rule.approval_status.value}."
                )

        return rule

    def _validate_condition(self, rule_id: str, condition: KnowledgeRuleCondition) -> None:
        ctype = condition.condition_type
        param = condition.parameter

        if ctype == ConditionType.REQUIRED_REGIME:
            if param.upper() not in ALLOWED_REGIME_PARAMS:
                raise KnowledgeValidationError(
                    f"Rule '{rule_id}' has unsupported REQUIRED_REGIME '{param}'. "
                    f"Allowed: {sorted(ALLOWED_REGIME_PARAMS)}."
                )
        elif ctype == ConditionType.REQUIRED_PRICE_ZONE:
            if param.upper() not in ALLOWED_PRICE_ZONE_PARAMS:
                raise KnowledgeValidationError(
                    f"Rule '{rule_id}' has unsupported REQUIRED_PRICE_ZONE '{param}'. "
                    f"Allowed: {sorted(ALLOWED_PRICE_ZONE_PARAMS)}."
                )
        elif ctype == ConditionType.REQUIRED_STRUCTURE_EVENT:
            if param.upper() not in ALLOWED_STRUCTURE_PARAMS:
                raise KnowledgeValidationError(
                    f"Rule '{rule_id}' has unsupported REQUIRED_STRUCTURE_EVENT '{param}'. "
                    f"Allowed: {sorted(ALLOWED_STRUCTURE_PARAMS)}."
                )
        elif ctype == ConditionType.REQUIRED_LIQUIDITY_SWEEP:
            if param.upper() not in ALLOWED_LIQUIDITY_PARAMS:
                raise KnowledgeValidationError(
                    f"Rule '{rule_id}' has unsupported REQUIRED_LIQUIDITY_SWEEP '{param}'. "
                    f"Allowed: {sorted(ALLOWED_LIQUIDITY_PARAMS)}."
                )
        elif ctype == ConditionType.REQUIRED_FVG:
            if param.upper() not in ALLOWED_FVG_PARAMS:
                raise KnowledgeValidationError(
                    f"Rule '{rule_id}' has unsupported REQUIRED_FVG '{param}'. "
                    f"Allowed: {sorted(ALLOWED_FVG_PARAMS)}."
                )
        elif ctype == ConditionType.REQUIRED_KILLZONE:
            if param.upper() not in ALLOWED_KILLZONE_PARAMS:
                raise KnowledgeValidationError(
                    f"Rule '{rule_id}' has unsupported REQUIRED_KILLZONE '{param}'. "
                    f"Allowed: {sorted(ALLOWED_KILLZONE_PARAMS)}."
                )
        elif ctype == ConditionType.FEATURE_COMPARISON:
            whitelist = self._config.whitelisted_features
            if param not in whitelist:
                raise KnowledgeValidationError(
                    f"Rule '{rule_id}' references unknown or non-whitelisted feature '{param}'. "
                    f"Whitelisted features: {sorted(whitelist)}."
                )
            has_target = condition.target_feature is not None
            has_thresh = condition.threshold_value is not None
            if has_target == has_thresh:
                raise KnowledgeValidationError(
                    f"Rule '{rule_id}' FEATURE_COMPARISON on '{param}' must specify exactly one of "
                    "'target_feature' or 'threshold_value'."
                )
            if condition.target_feature is not None and condition.target_feature not in whitelist:
                raise KnowledgeValidationError(
                    f"Rule '{rule_id}' references unknown or non-whitelisted target_feature "
                    f"'{condition.target_feature}'."
                )
