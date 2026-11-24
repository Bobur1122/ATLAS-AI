"""Deterministic and schema-validated rule extractors for ATLAS AI Phase 5."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal
import json
import re

from atlas_ai.domain.market import Timeframe
from atlas_ai.domain.signals import SignalDirection
from atlas_ai.knowledge.config import KnowledgeConfig
from atlas_ai.knowledge.exceptions import KnowledgeExtractionError
from atlas_ai.knowledge.models import (
    ComparisonOperator,
    ConditionType,
    ExtractedTradingRule,
    KnowledgeCategory,
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeRuleCondition,
    RuleApprovalStatus,
    compute_sha256_hex,
)
from atlas_ai.knowledge.validator import KnowledgeRuleValidator

FEATURE_COMP_RE = re.compile(
    r"\b(?P<lhs>close|open|high|low|ema_9|ema_21|ema_50|sma_20|sma_50|sma_100|sma_200|atr_14|atr_pct_14)\s*"
    r"(?P<op>>=|<=|>|<|==|above|below)\s*"
    r"(?P<rhs>close|open|high|low|ema_9|ema_21|ema_50|sma_20|sma_50|sma_100|sma_200|atr_14|atr_pct_14|\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
RR_MENTION_RE = re.compile(r"\b(?P<rr>\d+(?:\.\d+)?)\s*(?:r\b|:1\b|\s*rr\b)", re.IGNORECASE)
SL_ATR_MENTION_RE = re.compile(r"\b(?P<sl>\d+(?:\.\d+)?)\s*(?:x\s*)?atr\s*(?:stop|sl)\b", re.IGNORECASE)


class BaseRuleExtractor(ABC):
    """Abstract base interface for extracting structured trading rules from educational documents."""

    def __init__(
        self,
        config: KnowledgeConfig | None = None,
        validator: KnowledgeRuleValidator | None = None,
    ) -> None:
        self._config = config or KnowledgeConfig()
        self._validator = validator or KnowledgeRuleValidator(self._config)

    @property
    def config(self) -> KnowledgeConfig:
        return self._config

    @property
    def validator(self) -> KnowledgeRuleValidator:
        return self._validator

    @abstractmethod
    def extract_from_document(
        self,
        document: KnowledgeDocument,
        *,
        extracted_at: datetime | None = None,
    ) -> tuple[ExtractedTradingRule, ...]:
        """Extract and validate structured trading rules from a KnowledgeDocument."""


class DeterministicRuleExtractor(BaseRuleExtractor):
    """Offline, deterministic rule extractor that parses structured playbooks and explicit
    SMC/ICT & trend rule statements into DRAFT ExtractedTradingRule objects.

    Safety Invariants:
    1. Works 100% offline without external APIs or network calls.
    2. Never invents conditions not explicitly present in the source chunk.
    3. Always sets `approval_status = RuleApprovalStatus.DRAFT` upon extraction.
    4. Uncertain or ambiguous statements without clear direction and conditions remain
       non-executable (`direction = SignalDirection.WAIT`, `approval_status = DRAFT`).
    5. Every extracted rule is validated by `KnowledgeRuleValidator` before being returned.
    """

    def extract_from_document(
        self,
        document: KnowledgeDocument,
        *,
        extracted_at: datetime | None = None,
    ) -> tuple[ExtractedTradingRule, ...]:
        ts_extracted = extracted_at or document.ingested_at

        # 1. If the document is a structured JSON playbook with explicit 'playbook_rules', parse them directly
        playbook_raw = document.metadata.get("playbook_rules")
        if isinstance(playbook_raw, list) and playbook_raw:
            return self._extract_from_playbook_entries(
                document=document,
                entries=playbook_raw,
                extracted_at=ts_extracted,
            )

        # 2. Otherwise, deterministically extract rules chunk-by-chunk
        rules: list[ExtractedTradingRule] = []
        for chunk in document.chunks:
            chunk_rules = self.extract_from_chunk(
                chunk,
                document=document,
                extracted_at=ts_extracted,
            )
            rules.extend(chunk_rules)

        return tuple(rules)

    def _extract_from_playbook_entries(
        self,
        *,
        document: KnowledgeDocument,
        entries: list[object],
        extracted_at: datetime,
    ) -> tuple[ExtractedTradingRule, ...]:
        rules: list[ExtractedTradingRule] = []
        for idx, raw_entry in enumerate(entries):
            if not isinstance(raw_entry, dict):
                raise KnowledgeExtractionError(f"Playbook rule entry at index {idx} must be a dict.")

            chunk_idx = min(idx, len(document.chunks) - 1)
            source_chunk = document.chunks[chunk_idx]

            title = str(raw_entry.get("title") or f"{document.title} Rule #{idx + 1}").strip()
            description = str(raw_entry.get("description") or title).strip()
            cat_str = str(raw_entry.get("category") or KnowledgeCategory.SMC_ICT.value).strip().upper()
            try:
                category = KnowledgeCategory(cat_str)
            except ValueError as exc:
                raise KnowledgeExtractionError(f"Invalid KnowledgeCategory '{cat_str}'.") from exc

            dir_str = str(raw_entry.get("direction") or SignalDirection.WAIT.value).strip().upper()
            try:
                direction = SignalDirection(dir_str)
            except ValueError as exc:
                raise KnowledgeExtractionError(f"Invalid SignalDirection '{dir_str}'.") from exc

            raw_conds = raw_entry.get("conditions", [])
            if not isinstance(raw_conds, list):
                raise KnowledgeExtractionError("Playbook rule 'conditions' must be a list.")

            conditions: list[KnowledgeRuleCondition] = []
            for c_obj in raw_conds:
                if not isinstance(c_obj, dict):
                    raise KnowledgeExtractionError("Condition entry must be a dict.")
                ctype = ConditionType(str(c_obj["condition_type"]).strip().upper())
                op = ComparisonOperator(
                    str(c_obj.get("operator") or ComparisonOperator.EQ.value).strip().upper()
                )
                thresh_raw = c_obj.get("threshold_value")
                thresh_dec = Decimal(str(thresh_raw)) if thresh_raw is not None else None
                conditions.append(
                    KnowledgeRuleCondition(
                        condition_type=ctype,
                        parameter=str(c_obj["parameter"]).strip(),
                        operator=op,
                        target_feature=(
                            str(c_obj["target_feature"]).strip()
                            if c_obj.get("target_feature") is not None
                            else None
                        ),
                        threshold_value=thresh_dec,
                        rationale=str(c_obj.get("rationale") or "").strip(),
                    )
                )

            sl_mult = Decimal(str(raw_entry.get("stop_loss_atr_multiplier", "1.50")))
            min_rr = Decimal(str(raw_entry.get("minimum_rr_ratio", "2.00")))
            base_conf = Decimal(str(raw_entry.get("base_confidence", "0.75")))

            syms = document.symbol_tags
            if isinstance(raw_entry.get("target_symbols"), list) and raw_entry["target_symbols"]:
                syms = tuple(str(s).strip().upper() for s in raw_entry["target_symbols"])

            tfs = document.timeframe_tags
            if isinstance(raw_entry.get("applicable_timeframes"), list) and raw_entry["applicable_timeframes"]:
                tfs = tuple(Timeframe(str(t).strip().upper()) for t in raw_entry["applicable_timeframes"])

            hash_payload = (
                f"{document.document_id}:{source_chunk.chunk_id}:{direction.value}:"
                f"{[(c.condition_type.value, c.parameter, c.operator.value, c.target_feature, str(c.threshold_value)) for c in conditions]}"
            )
            c_hash = compute_sha256_hex(hash_payload)
            rule_id = str(raw_entry.get("rule_id") or f"rule-{c_hash[:12]}").strip()

            rule_meta = (
                dict(raw_entry["metadata"])
                if isinstance(raw_entry.get("metadata"), dict)
                else {}
            )

            # Force DRAFT status on newly ingested rules (never auto-approve)
            notes: list[str] = ["Extracted from structured JSON playbook."]
            if not conditions or direction == SignalDirection.WAIT:
                direction = SignalDirection.WAIT
                notes.append("AMBIGUOUS_OR_INCOMPLETE_STATEMENT: Kept as non-executable DRAFT.")

            rule = ExtractedTradingRule(
                rule_id=rule_id,
                source_document_id=document.document_id,
                source_chunk_id=source_chunk.chunk_id,
                title=title,
                description=description,
                category=category,
                approval_status=RuleApprovalStatus.DRAFT,
                target_symbols=syms,
                applicable_timeframes=tfs,
                direction=direction,
                conditions=tuple(conditions),
                stop_loss_atr_multiplier=sl_mult,
                minimum_rr_ratio=min_rr,
                base_confidence=base_conf,
                extracted_at=extracted_at,
                content_hash=c_hash,
                extraction_notes=tuple(notes),
                metadata=rule_meta,
            )
            self._validator.validate_rule(rule)
            rules.append(rule)

        return tuple(rules)

    def extract_from_chunk(
        self,
        chunk: KnowledgeChunk,
        *,
        document: KnowledgeDocument,
        extracted_at: datetime | None = None,
    ) -> tuple[ExtractedTradingRule, ...]:
        """Deterministically extract explicit trading rules from a single KnowledgeChunk."""
        ts_extracted = extracted_at or document.ingested_at
        text = chunk.text
        lower = text.lower()

        # 1. Detect explicit directional cues
        has_buy_cue = bool(
            re.search(
                r"\b(?:buy|long|bullish\s+setup|bullish\s+entry|enter\s+long|go\s+long)\b",
                lower,
            )
        )
        has_sell_cue = bool(
            re.search(
                r"\b(?:sell|short|bearish\s+setup|bearish\s+entry|enter\s+short|go\s+short)\b",
                lower,
            )
        )

        # 2. Detect explicit declarative conditions mentioned in the chunk text
        conditions: list[KnowledgeRuleCondition] = []

        # Kill Zones
        if "london kill zone" in lower or "london killzone" in lower:
            conditions.append(
                KnowledgeRuleCondition(
                    condition_type=ConditionType.REQUIRED_KILLZONE,
                    parameter="LONDON_KILL_ZONE",
                    rationale="Source explicitly requires London Kill Zone.",
                )
            )
        elif "new york kill zone" in lower or "ny kill zone" in lower or "ny killzone" in lower:
            conditions.append(
                KnowledgeRuleCondition(
                    condition_type=ConditionType.REQUIRED_KILLZONE,
                    parameter="NEW_YORK_KILL_ZONE",
                    rationale="Source explicitly requires New York Kill Zone.",
                )
            )

        # Dealing Range Premium / Discount Zone
        if re.search(r"\bdiscount(?:\s+zone|\s+array)?\b", lower):
            conditions.append(
                KnowledgeRuleCondition(
                    condition_type=ConditionType.REQUIRED_PRICE_ZONE,
                    parameter="DISCOUNT",
                    rationale="Source requires price in Discount zone (< 50% of dealing range).",
                )
            )
        elif re.search(r"\bpremium(?:\s+zone|\s+array)?\b", lower):
            conditions.append(
                KnowledgeRuleCondition(
                    condition_type=ConditionType.REQUIRED_PRICE_ZONE,
                    parameter="PREMIUM",
                    rationale="Source requires price in Premium zone (> 50% of dealing range).",
                )
            )

        # Fair Value Gaps (FVG)
        if re.search(r"\bbullish\s+(?:fvg|fair\s+value\s+gap)\s+retest\b|\bretest\s+of\s+bullish\s+(?:fvg|fair\s+value\s+gap)\b", lower):
            conditions.append(
                KnowledgeRuleCondition(
                    condition_type=ConditionType.REQUIRED_FVG,
                    parameter="BULLISH_FVG_RETEST",
                    rationale="Source requires a Bullish FVG retest.",
                )
            )
        elif re.search(r"\bbearish\s+(?:fvg|fair\s+value\s+gap)\s+retest\b|\bretest\s+of\s+bearish\s+(?:fvg|fair\s+value\s+gap)\b", lower):
            conditions.append(
                KnowledgeRuleCondition(
                    condition_type=ConditionType.REQUIRED_FVG,
                    parameter="BEARISH_FVG_RETEST",
                    rationale="Source requires a Bearish FVG retest.",
                )
            )
        elif re.search(r"\bbullish\s+(?:fvg|fair\s+value\s+gap)\b", lower):
            conditions.append(
                KnowledgeRuleCondition(
                    condition_type=ConditionType.REQUIRED_FVG,
                    parameter="BULLISH_FVG_ACTIVE",
                    rationale="Source requires an active Bullish FVG.",
                )
            )
        elif re.search(r"\bbearish\s+(?:fvg|fair\s+value\s+gap)\b", lower):
            conditions.append(
                KnowledgeRuleCondition(
                    condition_type=ConditionType.REQUIRED_FVG,
                    parameter="BEARISH_FVG_ACTIVE",
                    rationale="Source requires an active Bearish FVG.",
                )
            )

        # Liquidity Sweeps (PDL/PDH, EQL/SSL, EQH/BSL)
        if re.search(r"\b(?:pdl\s+sweep|sweep\s+of\s+pdl|previous\s+day\s+low\s+sweep|sweeps?\s+previous\s+day\s+low)\b", lower):
            conditions.append(
                KnowledgeRuleCondition(
                    condition_type=ConditionType.REQUIRED_LIQUIDITY_SWEEP,
                    parameter="PDL_SWEEP",
                    rationale="Source requires a Previous Day Low (PDL) liquidity sweep and rejection.",
                )
            )
        elif re.search(r"\b(?:pdh\s+sweep|sweep\s+of\s+pdh|previous\s+day\s+high\s+sweep|sweeps?\s+previous\s+day\s+high)\b", lower):
            conditions.append(
                KnowledgeRuleCondition(
                    condition_type=ConditionType.REQUIRED_LIQUIDITY_SWEEP,
                    parameter="PDH_SWEEP",
                    rationale="Source requires a Previous Day High (PDH) liquidity sweep and rejection.",
                )
            )
        elif re.search(r"\b(?:eql\s+sweep|equal\s+lows?\s+sweep|ssl\s+sweep|sell[- ]side\s+liquidity\s+sweep)\b", lower):
            conditions.append(
                KnowledgeRuleCondition(
                    condition_type=ConditionType.REQUIRED_LIQUIDITY_SWEEP,
                    parameter="SSL_OR_EQL_SWEEP",
                    rationale="Source requires a Sell-Side Liquidity (SSL/EQL) sweep and rejection.",
                )
            )
        elif re.search(r"\b(?:eqh\s+sweep|equal\s+highs?\s+sweep|bsl\s+sweep|buy[- ]side\s+liquidity\s+sweep)\b", lower):
            conditions.append(
                KnowledgeRuleCondition(
                    condition_type=ConditionType.REQUIRED_LIQUIDITY_SWEEP,
                    parameter="BSL_OR_EQH_SWEEP",
                    rationale="Source requires a Buy-Side Liquidity (BSL/EQH) sweep and rejection.",
                )
            )

        # Market Structure (BOS / CHoCH)
        if re.search(r"\bbullish\s+(?:bos|break\s+of\s+structure)\b", lower):
            conditions.append(
                KnowledgeRuleCondition(
                    condition_type=ConditionType.REQUIRED_STRUCTURE_EVENT,
                    parameter="BULLISH_BOS",
                    rationale="Source requires a confirmed Bullish Break of Structure (BOS).",
                )
            )
        elif re.search(r"\bbearish\s+(?:bos|break\s+of\s+structure)\b", lower):
            conditions.append(
                KnowledgeRuleCondition(
                    condition_type=ConditionType.REQUIRED_STRUCTURE_EVENT,
                    parameter="BEARISH_BOS",
                    rationale="Source requires a confirmed Bearish Break of Structure (BOS).",
                )
            )
        elif re.search(r"\bbullish\s+(?:choch|change\s+of\s+character)\b", lower):
            conditions.append(
                KnowledgeRuleCondition(
                    condition_type=ConditionType.REQUIRED_STRUCTURE_EVENT,
                    parameter="BULLISH_CHOCH",
                    rationale="Source requires a confirmed Bullish Change of Character (CHoCH).",
                )
            )
        elif re.search(r"\bbearish\s+(?:choch|change\s+of\s+character)\b", lower):
            conditions.append(
                KnowledgeRuleCondition(
                    condition_type=ConditionType.REQUIRED_STRUCTURE_EVENT,
                    parameter="BEARISH_CHOCH",
                    rationale="Source requires a confirmed Bearish Change of Character (CHoCH).",
                )
            )

        # Market Regime
        if re.search(r"\b(?:in\s+an?\s+uptrend|uptrend\s+regime|bullish\s+trend\s+regime)\b", lower):
            conditions.append(
                KnowledgeRuleCondition(
                    condition_type=ConditionType.REQUIRED_REGIME,
                    parameter="UPTREND",
                    rationale="Source requires an UPTREND market regime.",
                )
            )
        elif re.search(r"\b(?:in\s+a\s+downtrend|downtrend\s+regime|bearish\s+trend\s+regime)\b", lower):
            conditions.append(
                KnowledgeRuleCondition(
                    condition_type=ConditionType.REQUIRED_REGIME,
                    parameter="DOWNTREND",
                    rationale="Source requires a DOWNTREND market regime.",
                )
            )

        # Explicit Feature Comparisons (e.g., "close > ema_21", "ema_9 > ema_50")
        for f_match in FEATURE_COMP_RE.finditer(lower):
            lhs = f_match.group("lhs").lower()
            op_raw = f_match.group("op").lower()
            rhs = f_match.group("rhs").lower()

            op_map = {
                ">": ComparisonOperator.GT,
                "above": ComparisonOperator.GT,
                ">=": ComparisonOperator.GE,
                "<": ComparisonOperator.LT,
                "below": ComparisonOperator.LT,
                "<=": ComparisonOperator.LE,
                "==": ComparisonOperator.EQ,
            }
            op = op_map[op_raw]
            if re.match(r"^\d+(?:\.\d+)?$", rhs):
                conditions.append(
                    KnowledgeRuleCondition(
                        condition_type=ConditionType.FEATURE_COMPARISON,
                        parameter=lhs,
                        operator=op,
                        threshold_value=Decimal(rhs),
                        rationale=f"Extracted feature comparison: {lhs} {op.value} {rhs}.",
                    )
                )
            elif lhs != rhs:
                conditions.append(
                    KnowledgeRuleCondition(
                        condition_type=ConditionType.FEATURE_COMPARISON,
                        parameter=lhs,
                        operator=op,
                        target_feature=rhs,
                        rationale=f"Extracted feature comparison: {lhs} {op.value} {rhs}.",
                    )
                )

        # Determine direction and ambiguity status
        notes: list[str] = []
        if has_buy_cue and not has_sell_cue and conditions:
            direction = SignalDirection.BUY
            notes.append("Deterministic extraction found unambiguous BUY direction and explicit conditions.")
        elif has_sell_cue and not has_buy_cue and conditions:
            direction = SignalDirection.SELL
            notes.append("Deterministic extraction found unambiguous SELL direction and explicit conditions.")
        else:
            # Ambiguous or non-actionable educational statement -> keep as DRAFT with WAIT direction
            direction = SignalDirection.WAIT
            if has_buy_cue and has_sell_cue:
                notes.append(
                    "AMBIGUOUS_STATEMENT: Both BUY and SELL cues detected in the same chunk; kept as non-executable DRAFT."
                )
            elif not conditions:
                notes.append(
                    "INCOMPLETE_STATEMENT: No concrete testable market conditions found; kept as non-executable DRAFT."
                )
            else:
                notes.append(
                    "AMBIGUOUS_DIRECTION: Conditions detected without explicit BUY/SELL direction; kept as non-executable DRAFT."
                )

        # Parse optional SL ATR multiplier and minimum RR ratio if explicitly mentioned
        sl_mult = Decimal("1.50")
        sl_match = SL_ATR_MENTION_RE.search(lower)
        if sl_match:
            candidate_sl = Decimal(sl_match.group("sl"))
            if (
                self._config.min_allowed_sl_atr_multiplier
                <= candidate_sl
                <= self._config.max_allowed_sl_atr_multiplier
            ):
                sl_mult = candidate_sl

        min_rr = Decimal("2.00")
        rr_match = RR_MENTION_RE.search(lower)
        if rr_match:
            candidate_rr = Decimal(rr_match.group("rr"))
            if self._config.min_allowed_rr_ratio <= candidate_rr <= self._config.max_allowed_rr_ratio:
                min_rr = candidate_rr

        category = self._infer_category(conditions)
        rule_title = (
            f"{chunk.section_title} ({direction.value})"
            if chunk.section_title
            else f"{document.title} Chunk #{chunk.sequence_index + 1} ({direction.value})"
        )

        hash_payload = (
            f"{document.document_id}:{chunk.chunk_id}:{direction.value}:"
            f"{[(c.condition_type.value, c.parameter, c.operator.value, c.target_feature, str(c.threshold_value)) for c in conditions]}"
        )
        c_hash = compute_sha256_hex(hash_payload)
        rule_id = f"rule-{c_hash[:12]}"

        rule = ExtractedTradingRule(
            rule_id=rule_id,
            source_document_id=document.document_id,
            source_chunk_id=chunk.chunk_id,
            title=rule_title,
            description=text[:300],
            category=category,
            approval_status=RuleApprovalStatus.DRAFT,
            target_symbols=chunk.symbol_tags or document.symbol_tags,
            applicable_timeframes=chunk.timeframe_tags or document.timeframe_tags,
            direction=direction,
            conditions=tuple(conditions),
            stop_loss_atr_multiplier=sl_mult,
            minimum_rr_ratio=min_rr,
            base_confidence=Decimal("0.75"),
            extracted_at=ts_extracted,
            content_hash=c_hash,
            extraction_notes=tuple(notes),
            metadata={
                "start_timestamp": chunk.start_timestamp,
                "end_timestamp": chunk.end_timestamp,
                "section_title": chunk.section_title,
            },
        )
        self._validator.validate_rule(rule)
        return (rule,)

    @staticmethod
    def _infer_category(conditions: list[KnowledgeRuleCondition]) -> KnowledgeCategory:
        types = {c.condition_type for c in conditions}
        if ConditionType.REQUIRED_FVG in types:
            return KnowledgeCategory.FAIR_VALUE_GAP
        if ConditionType.REQUIRED_LIQUIDITY_SWEEP in types:
            return KnowledgeCategory.LIQUIDITY
        if ConditionType.REQUIRED_KILLZONE in types:
            return KnowledgeCategory.SESSION_KILLZONE
        if ConditionType.REQUIRED_STRUCTURE_EVENT in types or ConditionType.REQUIRED_PRICE_ZONE in types:
            return KnowledgeCategory.SMC_ICT
        if ConditionType.FEATURE_COMPARISON in types or ConditionType.REQUIRED_REGIME in types:
            return KnowledgeCategory.TREND_FOLLOWING
        return KnowledgeCategory.SMC_ICT


class LLMExtractorAdapter(BaseRuleExtractor):
    """Optional, strictly sandboxed adapter for LLM-based structured rule extraction.

    Safety Architecture:
    1. Does not make uncontrolled external calls; requires an injected `completion_fn(prompt) -> json_str`.
    2. Never trusts the LLM's `approval_status`; every LLM-extracted rule is forced to `RuleApprovalStatus.DRAFT`.
    3. Every LLM output is parsed into `ExtractedTradingRule` and validated by `KnowledgeRuleValidator`,
       which blocks any unknown feature names, `eval()`/`exec()` expressions, lookahead references,
       or attempts to bypass `RiskEngine`.
    """

    def __init__(
        self,
        completion_fn: Callable[[str], str],
        *,
        config: KnowledgeConfig | None = None,
        validator: KnowledgeRuleValidator | None = None,
    ) -> None:
        super().__init__(config=config, validator=validator)
        self._completion_fn = completion_fn

    def extract_from_document(
        self,
        document: KnowledgeDocument,
        *,
        extracted_at: datetime | None = None,
    ) -> tuple[ExtractedTradingRule, ...]:
        ts_extracted = extracted_at or datetime(2026, 11, 15, 12, 0, 0, tzinfo=timezone.utc)
        rules: list[ExtractedTradingRule] = []

        for chunk in document.chunks:
            prompt = (
                f"Extract structured trading rules in JSON format from chunk '{chunk.chunk_id}' "
                f"of document '{document.document_id}':\n{chunk.text}"
            )
            raw_response = self._completion_fn(prompt)
            try:
                parsed = json.loads(raw_response)
            except json.JSONDecodeError as exc:
                raise KnowledgeExtractionError(
                    f"LLMExtractorAdapter received malformed JSON for chunk '{chunk.chunk_id}': {exc}"
                ) from exc

            raw_rules = parsed.get("rules", [parsed]) if isinstance(parsed, dict) else parsed
            if not isinstance(raw_rules, list):
                raise KnowledgeExtractionError("LLMExtractorAdapter expected a JSON list of rules.")

            for idx, r_dict in enumerate(raw_rules):
                if not isinstance(r_dict, dict):
                    raise KnowledgeExtractionError("Each LLM extracted rule must be a JSON object.")

                raw_conds = r_dict.get("conditions", [])
                conds: list[KnowledgeRuleCondition] = []
                for c_dict in raw_conds:
                    conds.append(
                        KnowledgeRuleCondition(
                            condition_type=ConditionType(str(c_dict["condition_type"]).strip().upper()),
                            parameter=str(c_dict["parameter"]).strip(),
                            operator=ComparisonOperator(
                                str(c_dict.get("operator", "EQ")).strip().upper()
                            ),
                            target_feature=c_dict.get("target_feature"),
                            threshold_value=(
                                Decimal(str(c_dict["threshold_value"]))
                                if c_dict.get("threshold_value") is not None
                                else None
                            ),
                            rationale=str(c_dict.get("rationale", "")),
                        )
                    )

                direction = SignalDirection(str(r_dict.get("direction", "WAIT")).strip().upper())
                if not conds:
                    direction = SignalDirection.WAIT

                hash_payload = (
                    f"{document.document_id}:{chunk.chunk_id}:{idx}:{direction.value}:"
                    f"{[(c.condition_type.value, c.parameter) for c in conds]}"
                )
                c_hash = compute_sha256_hex(hash_payload)
                rule_id = str(r_dict.get("rule_id") or f"llm-rule-{c_hash[:12]}").strip()

                # Force DRAFT status regardless of what the LLM output claimed
                rule = ExtractedTradingRule(
                    rule_id=rule_id,
                    source_document_id=document.document_id,
                    source_chunk_id=chunk.chunk_id,
                    title=str(r_dict.get("title") or f"LLM Rule {idx + 1}"),
                    description=str(r_dict.get("description") or chunk.text[:200]),
                    category=KnowledgeCategory(
                        str(r_dict.get("category", KnowledgeCategory.SMC_ICT.value)).strip().upper()
                    ),
                    approval_status=RuleApprovalStatus.DRAFT,
                    target_symbols=chunk.symbol_tags or document.symbol_tags,
                    applicable_timeframes=chunk.timeframe_tags or document.timeframe_tags,
                    direction=direction,
                    conditions=tuple(conds),
                    stop_loss_atr_multiplier=Decimal(
                        str(r_dict.get("stop_loss_atr_multiplier", "1.50"))
                    ),
                    minimum_rr_ratio=Decimal(str(r_dict.get("minimum_rr_ratio", "2.00"))),
                    base_confidence=Decimal(str(r_dict.get("base_confidence", "0.70"))),
                    extracted_at=ts_extracted,
                    content_hash=c_hash,
                    extraction_notes=("Extracted via LLMExtractorAdapter (forced DRAFT).",),
                    metadata=dict(r_dict.get("metadata", {}))
                    if isinstance(r_dict.get("metadata"), dict)
                    else {},
                )
                # Mandatory deterministic safety & whitelist validation
                self._validator.validate_rule(rule)
                rules.append(rule)

        return tuple(rules)
