"""Persistent KnowledgeRepository with content-hash deduplication, approval workflow, and querying."""

import json
from pathlib import Path
from typing import Sequence

from atlas_ai.domain.market import Timeframe
from atlas_ai.domain.signals import SignalDirection
from atlas_ai.knowledge.config import KnowledgeConfig
from atlas_ai.knowledge.exceptions import KnowledgeRepositoryError
from atlas_ai.knowledge.models import (
    ConditionType,
    ExtractedTradingRule,
    KnowledgeCategory,
    KnowledgeDocument,
    KnowledgeQuery,
    RuleApprovalStatus,
)
from atlas_ai.knowledge.validator import KnowledgeRuleValidator


class KnowledgeRepository:
    """Stores, deduplicates, versions, queries, and persists KnowledgeDocument and ExtractedTradingRule records.

    Safety Invariants:
    1. Newly ingested rules are ALWAYS stored as `RuleApprovalStatus.DRAFT` (never auto-approved).
    2. Every rule added or updated must pass `KnowledgeRuleValidator.validate_rule()`.
    3. Transitioning a rule to `VALIDATED` or `APPROVED` re-runs `KnowledgeRuleValidator` to guarantee
       that non-directional (`WAIT`), empty-condition, or unsafe rules can never become `APPROVED`.
    """

    def __init__(
        self,
        config: KnowledgeConfig | None = None,
        validator: KnowledgeRuleValidator | None = None,
    ) -> None:
        self._config = config or KnowledgeConfig()
        self._validator = validator or KnowledgeRuleValidator(self._config)
        self._documents: dict[str, KnowledgeDocument] = {}
        self._doc_hash_index: dict[str, str] = {}
        self._rules: dict[str, ExtractedTradingRule] = {}
        self._rule_hash_index: dict[str, str] = {}

    @property
    def config(self) -> KnowledgeConfig:
        return self._config

    @property
    def validator(self) -> KnowledgeRuleValidator:
        return self._validator

    def add_document(self, document: KnowledgeDocument) -> KnowledgeDocument:
        """Store a KnowledgeDocument with content-hash deduplication."""
        if document.content_hash in self._doc_hash_index:
            existing_id = self._doc_hash_index[document.content_hash]
            return self._documents[existing_id]

        if document.document_id in self._documents:
            existing = self._documents[document.document_id]
            if existing.content_hash == document.content_hash:
                return existing
            raise KnowledgeRepositoryError(
                f"Document ID '{document.document_id}' already exists with different content_hash."
            )

        self._documents[document.document_id] = document
        self._doc_hash_index[document.content_hash] = document.document_id
        return document

    def get_document(self, document_id: str) -> KnowledgeDocument:
        """Retrieve a stored KnowledgeDocument by ID."""
        if document_id not in self._documents:
            raise KnowledgeRepositoryError(f"Unknown document_id '{document_id}'.")
        return self._documents[document_id]

    def list_documents(self) -> tuple[KnowledgeDocument, ...]:
        """Return all stored documents in deterministic order by `document_id`."""
        return tuple(self._documents[k] for k in sorted(self._documents))

    def add_rule(
        self,
        rule: ExtractedTradingRule,
        *,
        require_registered_document: bool = True,
    ) -> ExtractedTradingRule:
        """Validate and store an ExtractedTradingRule, forcing `DRAFT` status on initial ingestion
        and deduplicating by `content_hash`.
        """
        if require_registered_document and rule.source_document_id not in self._documents:
            raise KnowledgeRepositoryError(
                f"Cannot store rule '{rule.rule_id}': source_document_id "
                f"'{rule.source_document_id}' is not registered in KnowledgeRepository."
            )
        if require_registered_document:
            doc = self._documents[rule.source_document_id]
            chunk_ids = {c.chunk_id for c in doc.chunks}
            if rule.source_chunk_id not in chunk_ids:
                raise KnowledgeRepositoryError(
                    f"Cannot store rule '{rule.rule_id}': source_chunk_id "
                    f"'{rule.source_chunk_id}' does not belong to document '{doc.document_id}'."
                )

        # Enforce safety rule: newly ingested rules must never be automatically approved
        ingested_rule = (
            rule
            if rule.approval_status == RuleApprovalStatus.DRAFT
            else rule.model_copy(update={"approval_status": RuleApprovalStatus.DRAFT})
        )
        self._validator.validate_rule(ingested_rule)

        # Deduplicate by content_hash
        if ingested_rule.content_hash in self._rule_hash_index:
            existing_id = self._rule_hash_index[ingested_rule.content_hash]
            return self._rules[existing_id]

        # If updating an existing rule_id with new content, increment version
        if ingested_rule.rule_id in self._rules:
            prev = self._rules[ingested_rule.rule_id]
            if prev.content_hash == ingested_rule.content_hash:
                return prev
            ingested_rule = ingested_rule.model_copy(update={"version": prev.version + 1})

        self._rules[ingested_rule.rule_id] = ingested_rule
        self._rule_hash_index[ingested_rule.content_hash] = ingested_rule.rule_id
        return ingested_rule

    def add_rules(
        self,
        rules: Sequence[ExtractedTradingRule],
        *,
        require_registered_document: bool = True,
    ) -> tuple[ExtractedTradingRule, ...]:
        """Batch-add multiple rules with validation and deduplication."""
        return tuple(
            self.add_rule(r, require_registered_document=require_registered_document)
            for r in rules
        )

    def get_rule(self, rule_id: str) -> ExtractedTradingRule:
        """Retrieve a stored ExtractedTradingRule by `rule_id`."""
        if rule_id not in self._rules:
            raise KnowledgeRepositoryError(f"Unknown rule_id '{rule_id}'.")
        return self._rules[rule_id]

    def validate_rule_status(self, rule_id: str) -> ExtractedTradingRule:
        """Transition a `DRAFT` rule to `VALIDATED` after running full validator checks."""
        rule = self.get_rule(rule_id)
        if rule.approval_status == RuleApprovalStatus.REJECTED:
            raise KnowledgeRepositoryError(
                f"Rule '{rule_id}' is REJECTED and cannot transition to VALIDATED."
            )
        candidate = rule.model_copy(update={"approval_status": RuleApprovalStatus.VALIDATED})
        self._validator.validate_rule(candidate)
        self._rules[rule_id] = candidate
        return candidate

    def approve_rule(self, rule_id: str) -> ExtractedTradingRule:
        """Explicitly approve a rule (`RuleApprovalStatus.APPROVED`) after full safety validation."""
        rule = self.get_rule(rule_id)
        if rule.approval_status == RuleApprovalStatus.REJECTED:
            raise KnowledgeRepositoryError(
                f"Rule '{rule_id}' is REJECTED and cannot be APPROVED."
            )
        candidate = rule.model_copy(update={"approval_status": RuleApprovalStatus.APPROVED})
        self._validator.validate_rule(candidate)
        self._rules[rule_id] = candidate
        return candidate

    def reject_rule(self, rule_id: str, reason: str = "Rejected by reviewer.") -> ExtractedTradingRule:
        """Explicitly mark a rule as `RuleApprovalStatus.REJECTED`."""
        rule = self.get_rule(rule_id)
        updated_notes = (*rule.extraction_notes, f"REJECTED: {reason}")
        candidate = rule.model_copy(
            update={
                "approval_status": RuleApprovalStatus.REJECTED,
                "extraction_notes": updated_notes,
            }
        )
        self._rules[rule_id] = candidate
        return candidate

    def query_rules(self, query: KnowledgeQuery | None = None) -> tuple[ExtractedTradingRule, ...]:
        """Deterministically query stored rules by symbol, timeframe, category, direction, status, regime, or killzone."""
        q = query or KnowledgeQuery()
        matched: list[ExtractedTradingRule] = []

        for rule_id in sorted(self._rules):
            rule = self._rules[rule_id]

            if q.approval_status is not None and rule.approval_status != q.approval_status:
                continue
            if q.symbol is not None and q.symbol not in rule.target_symbols:
                continue
            if q.timeframe is not None and q.timeframe not in rule.applicable_timeframes:
                continue
            if q.category is not None and rule.category != q.category:
                continue
            if q.direction is not None and rule.direction != q.direction:
                continue
            if q.required_regime is not None:
                reg_upper = q.required_regime.strip().upper()
                has_reg = any(
                    c.condition_type == ConditionType.REQUIRED_REGIME
                    and c.parameter.upper() == reg_upper
                    for c in rule.conditions
                )
                if not has_reg:
                    continue
            if q.required_killzone is not None:
                kz_upper = q.required_killzone.strip().upper()
                has_kz = any(
                    c.condition_type == ConditionType.REQUIRED_KILLZONE
                    and c.parameter.upper() == kz_upper
                    for c in rule.conditions
                )
                if not has_kz:
                    continue

            matched.append(rule)

        return tuple(matched)

    def get_approved_rules(
        self,
        *,
        symbol: str | None = None,
        timeframe: Timeframe | None = None,
        direction: SignalDirection | None = None,
    ) -> tuple[ExtractedTradingRule, ...]:
        """Convenience helper returning ONLY `RuleApprovalStatus.APPROVED` rules."""
        return self.query_rules(
            KnowledgeQuery(
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                approval_status=RuleApprovalStatus.APPROVED,
            )
        )

    def save_to_json(self, file_path: str | Path) -> Path:
        """Persist all documents and rules to a deterministic JSON file on disk."""
        path = Path(file_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": "1.0.0",
                "documents": [
                    self._documents[doc_id].model_dump(mode="json")
                    for doc_id in sorted(self._documents)
                ],
                "rules": [
                    self._rules[r_id].model_dump(mode="json")
                    for r_id in sorted(self._rules)
                ],
            }
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except Exception as exc:
            raise KnowledgeRepositoryError(
                f"Failed to save KnowledgeRepository to '{path}': {exc}"
            ) from exc
        return path

    @classmethod
    def load_from_json(
        cls,
        file_path: str | Path,
        *,
        config: KnowledgeConfig | None = None,
    ) -> "KnowledgeRepository":
        """Load and re-validate a KnowledgeRepository from a JSON file on disk."""
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise KnowledgeRepositoryError(f"Repository JSON file not found: '{path}'.")

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise KnowledgeRepositoryError(
                f"Failed to parse KnowledgeRepository JSON from '{path}': {exc}"
            ) from exc

        if not isinstance(raw, dict) or "documents" not in raw or "rules" not in raw:
            raise KnowledgeRepositoryError(
                f"Invalid KnowledgeRepository schema in '{path}': missing 'documents' or 'rules'."
            )

        repo = cls(config=config)
        for doc_data in raw["documents"]:
            doc = KnowledgeDocument.model_validate(doc_data)
            repo.add_document(doc)

        for rule_data in raw["rules"]:
            rule = ExtractedTradingRule.model_validate(rule_data)
            # Re-validate every rule upon loading from disk
            repo.validator.validate_rule(rule)
            if rule.source_document_id not in repo._documents:
                raise KnowledgeRepositoryError(
                    f"Loaded rule '{rule.rule_id}' references missing document '{rule.source_document_id}'."
                )
            repo._rules[rule.rule_id] = rule
            repo._rule_hash_index[rule.content_hash] = rule.rule_id

        return repo
