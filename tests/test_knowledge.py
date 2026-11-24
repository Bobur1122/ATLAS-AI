"""Comprehensive Phase 5 offline unit & integration tests for Educational Knowledge Ingestion & Rule Extraction."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path

import pandas as pd
import pytest

from atlas_ai.backtest import BacktestConfig, BacktestEngine
from atlas_ai.data.features import FeaturePipeline
from atlas_ai.domain.market import InstrumentSpec, Timeframe
from atlas_ai.domain.risk import AccountState, RiskParameters, RiskRejectionCode
from atlas_ai.domain.signals import SignalDirection
from atlas_ai.knowledge import (
    ComparisonOperator,
    ConditionType,
    DeterministicRuleExtractor,
    ExtractedTradingRule,
    KnowledgeBasedStrategy,
    KnowledgeCategory,
    KnowledgeConfig,
    KnowledgeDocumentLoader,
    KnowledgeExtractionError,
    KnowledgeIngestionError,
    KnowledgeQuery,
    KnowledgeRepository,
    KnowledgeRepositoryError,
    KnowledgeRuleCondition,
    KnowledgeRuleValidator,
    KnowledgeSafetyError,
    KnowledgeSourceType,
    KnowledgeValidationError,
    LLMExtractorAdapter,
    RuleApprovalStatus,
    compute_sha256_hex,
)
from atlas_ai.strategies.config import ConfluenceWeightsConfig
from atlas_ai.strategies.engine import StrategyEngine
from atlas_ai.strategies.models import RiskEvaluationStatus, SignalLifecycleState

FIXED_TS = datetime(2026, 11, 15, 12, 0, 0, tzinfo=timezone.utc)


def _make_trending_xauusd_df(n_bars: int = 75, direction: str = "up") -> pd.DataFrame:
    """Create a deterministic trending XAUUSD H1 OHLCV DataFrame and run FeaturePipeline."""
    t0 = datetime(2026, 3, 2, 0, 0, tzinfo=timezone.utc)
    rows: list[dict[str, object]] = []
    price = 2600.0
    for i in range(n_bars):
        ts = t0 + timedelta(hours=i)
        step = 1.5 if direction == "up" else -1.5
        open_p = round(price, 2)
        close_p = round(price + step, 2)
        high_p = round(max(open_p, close_p) + 0.8, 2)
        low_p = round(min(open_p, close_p) - 0.5, 2)
        rows.append(
            {
                "timestamp": ts,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": 200.0 + i,
            }
        )
        price = close_p
    raw_df = pd.DataFrame(rows)
    return FeaturePipeline().transform(raw_df)


# ---------------------------------------------------------------------------
# 1. Document Loading, Transcript Sanitization & Chunking (.txt, .md, .srt, .vtt, .json)
# ---------------------------------------------------------------------------
def test_document_loader_and_transcript_chunking(tmp_path: Path) -> None:
    """Verify ingestion of .srt, .vtt, .md, .txt, and .json documents, timestamp preservation, and error handling."""
    loader = KnowledgeDocumentLoader(KnowledgeConfig(max_chunk_chars=300, min_chunk_chars=10))

    # 1A. SRT Transcript ingestion (strips sequence numbers, HTML tags, [Music], preserves timestamps)
    srt_content = (
        "1\n"
        "00:01:10,000 --> 00:01:25,500\n"
        "<i>Welcome</i> to the XAUUSD M15 London Kill Zone lesson. [Music]\n\n"
        "2\n"
        "00:01:26,000 --> 00:01:48,000\n"
        "When price enters the discount zone and sweeps previous day low, we look to BUY with a 1.5 ATR stop and 2.5R target.\n"
    )
    srt_file = tmp_path / "ict_gold_lesson.srt"
    srt_file.write_text(srt_content, encoding="utf-8")

    srt_doc = loader.load_file(srt_file, author="ICT Mentor", ingested_at=FIXED_TS)
    assert srt_doc.source_type == KnowledgeSourceType.TRANSCRIPT_SRT
    assert "XAUUSD" in srt_doc.symbol_tags
    assert Timeframe.M15 in srt_doc.timeframe_tags
    assert "<i>" not in srt_doc.normalized_text
    assert "[Music]" not in srt_doc.normalized_text
    assert len(srt_doc.chunks) >= 1
    assert srt_doc.chunks[0].start_timestamp == "00:01:10.000"
    assert srt_doc.chunks[-1].end_timestamp == "00:01:48.000"

    # Repeat load produces identical deterministic SHA-256 content_hash and document_id
    srt_doc_2 = loader.load_file(srt_file, author="ICT Mentor", ingested_at=FIXED_TS)
    assert srt_doc.content_hash == srt_doc_2.content_hash
    assert srt_doc.document_id == srt_doc_2.document_id

    # 1B. VTT Transcript ingestion
    vtt_content = (
        "WEBVTT\n\n"
        "00:05:00.000 --> 00:05:20.000\n"
        "On H1 XAUUSD in a downtrend regime, wait for premium zone and bearish FVG retest to SELL.\n"
    )
    vtt_doc = loader.load_text(
        vtt_content,
        title="VTT Bearish FVG Lesson",
        source_type=KnowledgeSourceType.TRANSCRIPT_VTT,
        ingested_at=FIXED_TS,
    )
    assert "WEBVTT" not in vtt_doc.normalized_text
    assert vtt_doc.chunks[0].start_timestamp == "00:05:00.000"

    # 1C. Markdown study notes with section headings
    md_content = (
        "# Gold Bullish Trend Checklist\n"
        "- Wait for XAUUSD H1 in an uptrend regime.\n"
        "- Confirm close > ema_21 and ema_9 > ema_50 before entering a BUY setup.\n\n"
        "## Bearish Reversal Checklist\n"
        "- In New York Kill Zone, wait for equal highs sweep in premium zone to SELL.\n"
    )
    md_doc = loader.load_text(
        md_content,
        title="SMC Study Guide",
        source_type=KnowledgeSourceType.MARKDOWN_NOTES,
        ingested_at=FIXED_TS,
    )
    assert len(md_doc.chunks) == 2
    assert md_doc.chunks[0].section_title == "Gold Bullish Trend Checklist"
    assert md_doc.chunks[1].section_title == "Bearish Reversal Checklist"

    # 1D. Empty document and malformed JSON rejection
    with pytest.raises(KnowledgeIngestionError, match="empty"):
        loader.load_text("   \n\t  ", title="Empty Doc")

    with pytest.raises(KnowledgeIngestionError, match="Malformed JSON"):
        loader.load_text("{bad_json:", title="Broken JSON", source_type=KnowledgeSourceType.PLAYBOOK_JSON)


# ---------------------------------------------------------------------------
# 2. Deterministic SMC/ICT & Trend Rule Extraction + LLM Adapter Validation
# ---------------------------------------------------------------------------
def test_deterministic_rule_extractor_smc_ict_and_trend_rules() -> None:
    """Verify offline extraction of SMC/ICT, trend, playbook JSON, and ambiguous rules."""
    loader = KnowledgeDocumentLoader()
    extractor = DeterministicRuleExtractor()

    # 2A. Natural language SMC/ICT & Trend Markdown extraction
    md_lesson = (
        "# Bullish London Setup\n"
        "On XAUUSD M15 during the London Kill Zone, when price is in the discount zone and completes a "
        "bullish FVG retest after a PDL sweep, enter a BUY setup with a 1.5 ATR stop and 2.5R target.\n\n"
        "# Bearish Trend Following\n"
        "On XAUUSD H1 in a downtrend regime where close < ema_21 and ema_9 < ema_50, enter a SELL setup "
        "with a 1.25 ATR stop and 2.0R target.\n\n"
        "# General Trading Psychology Note\n"
        "Always remain patient and accept that both buy and sell traders lose when they overtrade.\n"
    )
    doc = loader.load_text(
        md_lesson,
        title="Multi-Section Trading Lesson",
        source_type=KnowledgeSourceType.MARKDOWN_NOTES,
        ingested_at=FIXED_TS,
    )
    rules_1 = extractor.extract_from_document(doc, extracted_at=FIXED_TS)
    rules_2 = extractor.extract_from_document(doc, extracted_at=FIXED_TS)

    # Deterministic repeated extraction
    assert rules_1 == rules_2
    assert len(rules_1) == 3

    # Rule 0: Unambiguous BUY SMC/ICT rule (created as DRAFT)
    r_buy = rules_1[0]
    assert r_buy.direction == SignalDirection.BUY
    assert r_buy.approval_status == RuleApprovalStatus.DRAFT
    assert r_buy.stop_loss_atr_multiplier == Decimal("1.5")
    assert r_buy.minimum_rr_ratio == Decimal("2.5")
    cond_types_0 = {(c.condition_type, c.parameter) for c in r_buy.conditions}
    assert (ConditionType.REQUIRED_KILLZONE, "LONDON_KILL_ZONE") in cond_types_0
    assert (ConditionType.REQUIRED_PRICE_ZONE, "DISCOUNT") in cond_types_0
    assert (ConditionType.REQUIRED_FVG, "BULLISH_FVG_RETEST") in cond_types_0
    assert (ConditionType.REQUIRED_LIQUIDITY_SWEEP, "PDL_SWEEP") in cond_types_0

    # Rule 1: Unambiguous SELL Trend rule (created as DRAFT)
    r_sell = rules_1[1]
    assert r_sell.direction == SignalDirection.SELL
    assert r_sell.approval_status == RuleApprovalStatus.DRAFT
    assert r_sell.stop_loss_atr_multiplier == Decimal("1.25")
    assert r_sell.minimum_rr_ratio == Decimal("2.0")
    cond_types_1 = {(c.condition_type, c.parameter, c.target_feature) for c in r_sell.conditions}
    assert (ConditionType.REQUIRED_REGIME, "DOWNTREND", None) in cond_types_1
    assert (ConditionType.FEATURE_COMPARISON, "close", "ema_21") in cond_types_1
    assert (ConditionType.FEATURE_COMPARISON, "ema_9", "ema_50") in cond_types_1

    # Rule 2: Ambiguous psychology note -> remains DRAFT with WAIT direction and zero conditions
    r_ambig = rules_1[2]
    assert r_ambig.direction == SignalDirection.WAIT
    assert r_ambig.approval_status == RuleApprovalStatus.DRAFT
    assert len(r_ambig.conditions) == 0

    # 2B. LLMExtractorAdapter offline test: forces DRAFT even if LLM claims APPROVED, and blocks unsafe LLM output
    def safe_mock_llm(_prompt: str) -> str:
        return json.dumps(
            {
                "rules": [
                    {
                        "title": "LLM Extracted Bullish BOS",
                        "direction": "BUY",
                        "approval_status": "APPROVED",
                        "category": "SMC_ICT",
                        "stop_loss_atr_multiplier": "1.50",
                        "minimum_rr_ratio": "2.00",
                        "conditions": [
                            {
                                "condition_type": "REQUIRED_STRUCTURE_EVENT",
                                "parameter": "BULLISH_BOS",
                            }
                        ],
                    }
                ]
            }
        )

    single_chunk_doc = loader.load_text(
        "Bullish BOS on XAUUSD H1.",
        title="Short Note",
        ingested_at=FIXED_TS,
    )
    llm_adapter = LLMExtractorAdapter(safe_mock_llm)
    llm_rules = llm_adapter.extract_from_document(single_chunk_doc, extracted_at=FIXED_TS)
    assert len(llm_rules) == 1
    assert llm_rules[0].approval_status == RuleApprovalStatus.DRAFT

    def malicious_mock_llm(_prompt: str) -> str:
        return json.dumps(
            {
                "rules": [
                    {
                        "title": "Malicious Rule",
                        "direction": "BUY",
                        "conditions": [
                            {
                                "condition_type": "FEATURE_COMPARISON",
                                "parameter": "eval('os.system(\"rm -rf /\")')",
                                "operator": "GT",
                                "threshold_value": "0",
                            }
                        ],
                    }
                ]
            }
        )

    bad_llm_adapter = LLMExtractorAdapter(malicious_mock_llm)
    with pytest.raises(KnowledgeSafetyError, match="SAFETY VIOLATION"):
        bad_llm_adapter.extract_from_document(single_chunk_doc, extracted_at=FIXED_TS)


# ---------------------------------------------------------------------------
# 3. KnowledgeRuleValidator: Rejecting Unsafe, Non-Whitelisted & Risk-Bypass Rules
# ---------------------------------------------------------------------------
def test_knowledge_rule_validator_blocks_unsafe_or_non_whitelisted_rules() -> None:
    """Verify KnowledgeRuleValidator blocks unknown columns, future data, eval/exec, no SL, and risk overrides."""
    validator = KnowledgeRuleValidator()
    base_cond = KnowledgeRuleCondition(
        condition_type=ConditionType.FEATURE_COMPARISON,
        parameter="close",
        operator=ComparisonOperator.GT,
        target_feature="ema_21",
    )
    valid_rule = ExtractedTradingRule(
        rule_id="rule-valid-01",
        source_document_id="doc-01",
        source_chunk_id="chk-01",
        title="Valid EMA Rule",
        description="Buy when close > ema_21.",
        category=KnowledgeCategory.TREND_FOLLOWING,
        approval_status=RuleApprovalStatus.DRAFT,
        direction=SignalDirection.BUY,
        conditions=(base_cond,),
        stop_loss_atr_multiplier=Decimal("1.50"),
        minimum_rr_ratio=Decimal("2.00"),
        extracted_at=FIXED_TS,
        content_hash=compute_sha256_hex("rule-valid-01"),
    )
    assert validator.validate_rule(valid_rule) == valid_rule

    # 3A. Unknown / non-whitelisted feature column
    unknown_feat_rule = valid_rule.model_copy(
        update={
            "conditions": (
                KnowledgeRuleCondition(
                    condition_type=ConditionType.FEATURE_COMPARISON,
                    parameter="secret_proprietary_oscillator",
                    operator=ComparisonOperator.GT,
                    threshold_value=Decimal("50"),
                ),
            )
        }
    )
    with pytest.raises(KnowledgeValidationError, match="non-whitelisted feature"):
        validator.validate_rule(unknown_feat_rule)

    # 3B. Future-data / lookahead reference
    lookahead_rule = valid_rule.model_copy(
        update={"description": "Enter BUY when next_bar future_close > close."}
    )
    with pytest.raises(KnowledgeSafetyError, match="Future-data or lookahead"):
        validator.validate_rule(lookahead_rule)

    # 3C. Arbitrary code expression (eval / exec / lambda)
    code_exec_rule = valid_rule.model_copy(
        update={"description": "Run eval('place_market_order()') immediately."}
    )
    with pytest.raises(KnowledgeSafetyError, match="Arbitrary code execution"):
        validator.validate_rule(code_exec_rule)

    # 3D. Missing or non-positive stop-loss configuration
    no_sl_rule = valid_rule.model_copy(update={"stop_loss_atr_multiplier": Decimal("0")})
    with pytest.raises(KnowledgeSafetyError, match="stop_loss_atr_multiplier"):
        validator.validate_rule(no_sl_rule)

    # 3E. Attempt to override RiskParameters or enable live trading via metadata or text
    risk_override_rule = valid_rule.model_copy(
        update={"metadata": {"max_daily_loss_percent": "99.0"}}
    )
    with pytest.raises(KnowledgeSafetyError, match="cannot override RiskParameters"):
        validator.validate_rule(risk_override_rule)

    live_override_rule = valid_rule.model_copy(
        update={"description": "Set live_trading_enabled=true and bypass_risk."}
    )
    with pytest.raises(KnowledgeSafetyError, match="override RiskEngine or enable live trading"):
        validator.validate_rule(live_override_rule)

    # 3F. Ambiguous WAIT rule or zero-condition rule cannot transition to APPROVED
    wait_approved = valid_rule.model_copy(
        update={
            "direction": SignalDirection.WAIT,
            "approval_status": RuleApprovalStatus.APPROVED,
        }
    )
    with pytest.raises(KnowledgeValidationError, match="cannot be APPROVED"):
        validator.validate_rule(wait_approved)


# ---------------------------------------------------------------------------
# 4. KnowledgeRepository: Deduplication, Persistence, Provenance & Querying
# ---------------------------------------------------------------------------
def test_knowledge_repository_deduplication_persistence_and_querying(tmp_path: Path) -> None:
    """Verify repository deduplication, provenance enforcement, approval workflow, querying, and JSON persistence."""
    loader = KnowledgeDocumentLoader()
    extractor = DeterministicRuleExtractor()
    repo = KnowledgeRepository()

    playbook_json = json.dumps(
        {
            "title": "XAUUSD Institutional Playbook",
            "author": "Quantitative Desk",
            "symbols": ["XAUUSD"],
            "timeframes": ["H1", "M15"],
            "rules": [
                {
                    "rule_id": "rule-gold-buy-01",
                    "title": "Gold H1 Uptrend Pullback",
                    "description": "BUY XAUUSD in UPTREND when close > ema_21.",
                    "category": "TREND_FOLLOWING",
                    "direction": "BUY",
                    "stop_loss_atr_multiplier": "1.50",
                    "minimum_rr_ratio": "2.00",
                    "conditions": [
                        {"condition_type": "REQUIRED_REGIME", "parameter": "UPTREND"},
                        {
                            "condition_type": "FEATURE_COMPARISON",
                            "parameter": "close",
                            "operator": "GT",
                            "target_feature": "ema_21",
                        },
                    ],
                },
                {
                    "rule_id": "rule-gold-sell-02",
                    "title": "Gold London KillZone Bearish FVG",
                    "description": "SELL XAUUSD in London Kill Zone on Bearish FVG retest.",
                    "category": "SESSION_KILLZONE",
                    "direction": "SELL",
                    "stop_loss_atr_multiplier": "1.50",
                    "minimum_rr_ratio": "2.50",
                    "conditions": [
                        {"condition_type": "REQUIRED_KILLZONE", "parameter": "LONDON_KILL_ZONE"},
                        {"condition_type": "REQUIRED_FVG", "parameter": "BEARISH_FVG_RETEST"},
                    ],
                },
            ],
        }
    )

    doc = loader.load_text(
        playbook_json,
        title="Playbook",
        source_type=KnowledgeSourceType.PLAYBOOK_JSON,
        ingested_at=FIXED_TS,
    )

    # Reject adding a rule before its source document is registered (provenance check)
    extracted_rules = extractor.extract_from_document(doc, extracted_at=FIXED_TS)
    with pytest.raises(KnowledgeRepositoryError, match="not registered"):
        repo.add_rule(extracted_rules[0])

    # Register document (and verify duplicate add returns existing document)
    stored_doc = repo.add_document(doc)
    assert repo.add_document(doc) == stored_doc
    assert len(repo.list_documents()) == 1

    # Add extracted rules (verify both start as DRAFT and duplicate add is deduplicated)
    stored_rules = repo.add_rules(extracted_rules)
    assert len(stored_rules) == 2
    assert all(r.approval_status == RuleApprovalStatus.DRAFT for r in stored_rules)
    assert repo.add_rule(extracted_rules[0]) == stored_rules[0]

    # Initially zero APPROVED rules exist
    assert len(repo.get_approved_rules(symbol="XAUUSD")) == 0

    # Transition rule 1: DRAFT -> VALIDATED -> APPROVED
    val_r1 = repo.validate_rule_status("rule-gold-buy-01")
    assert val_r1.approval_status == RuleApprovalStatus.VALIDATED
    app_r1 = repo.approve_rule("rule-gold-buy-01")
    assert app_r1.approval_status == RuleApprovalStatus.APPROVED

    # Transition rule 2: DRAFT -> REJECTED (and verify REJECTED rule cannot be approved)
    rej_r2 = repo.reject_rule("rule-gold-sell-02", reason="Outperformed by structure model.")
    assert rej_r2.approval_status == RuleApprovalStatus.REJECTED
    with pytest.raises(KnowledgeRepositoryError, match="REJECTED and cannot be APPROVED"):
        repo.approve_rule("rule-gold-sell-02")

    # Query filtering
    approved_xau = repo.query_rules(
        KnowledgeQuery(
            symbol="XAUUSD",
            timeframe=Timeframe.H1,
            approval_status=RuleApprovalStatus.APPROVED,
            required_regime="UPTREND",
        )
    )
    assert len(approved_xau) == 1
    assert approved_xau[0].rule_id == "rule-gold-buy-01"

    # Persistence round-trip to JSON file
    repo_file = tmp_path / "knowledge_store.json"
    repo.save_to_json(repo_file)
    loaded_repo = KnowledgeRepository.load_from_json(repo_file)
    assert loaded_repo.list_documents() == repo.list_documents()
    assert loaded_repo.query_rules() == repo.query_rules()

    with pytest.raises(KnowledgeRepositoryError, match="not found"):
        KnowledgeRepository.load_from_json(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# 5. KnowledgeBasedStrategy Causal Evaluation, Unapproved Rule Exclusion & No Lookahead
# ---------------------------------------------------------------------------
def test_knowledge_based_strategy_causal_evaluation_and_no_lookahead() -> None:
    """Verify KnowledgeBasedStrategy evaluates only APPROVED rules, includes provenance, and has zero lookahead."""
    loader = KnowledgeDocumentLoader()
    extractor = DeterministicRuleExtractor()
    repo = KnowledgeRepository()

    doc = loader.load_text(
        "# XAUUSD H1 Trend Rule\n"
        "On XAUUSD H1 in an uptrend regime where close > ema_21 and ema_9 > ema_50, "
        "enter a BUY setup with a 1.5 ATR stop and 2.0R target.\n",
        title="H1 Trend Note",
        source_type=KnowledgeSourceType.MARKDOWN_NOTES,
        ingested_at=FIXED_TS,
    )
    repo.add_document(doc)
    rules = repo.add_rules(extractor.extract_from_document(doc, extracted_at=FIXED_TS))
    rule_id = rules[0].rule_id

    kb_strategy = KnowledgeBasedStrategy(repo, symbol="XAUUSD", timeframe=Timeframe.H1)
    df_full = _make_trending_xauusd_df(n_bars=75, direction="up")

    # 5A. While rule is still DRAFT, KnowledgeBasedStrategy must return WAIT (NO_APPROVED_KNOWLEDGE_RULES)
    res_draft = kb_strategy.evaluate(df_full, as_of_index=60)
    assert res_draft.direction == SignalDirection.WAIT
    assert "NO_APPROVED_KNOWLEDGE_RULES" in res_draft.rejection_reasons

    # 5B. While rule is VALIDATED (but not yet APPROVED), KnowledgeBasedStrategy must still return WAIT
    repo.validate_rule_status(rule_id)
    res_validated = kb_strategy.evaluate(df_full, as_of_index=60)
    assert res_validated.direction == SignalDirection.WAIT
    assert "NO_APPROVED_KNOWLEDGE_RULES" in res_validated.rejection_reasons

    # 5C. Once explicitly APPROVED, KnowledgeBasedStrategy evaluates the rule and emits BUY with provenance
    repo.approve_rule(rule_id)
    res_approved = kb_strategy.evaluate(df_full, as_of_index=60)
    assert res_approved.direction == SignalDirection.BUY
    assert res_approved.entry_price is not None
    assert res_approved.stop_loss is not None
    assert res_approved.take_profit is not None
    assert res_approved.stop_loss < res_approved.entry_price < res_approved.take_profit
    assert res_approved.supporting_evidence["matched_rule_ids"] == [rule_id]
    assert res_approved.supporting_evidence["source_document_ids"] == [doc.document_id]
    assert rule_id in res_approved.supporting_evidence["matched_conditions"]

    # 5D. Future-data invariance (strict zero lookahead):
    # Mutate future bars 61..74 with an extreme crash and verify evaluation at as_of_index=60 is untouched!
    df_mutated = df_full.copy()
    for idx in range(61, 75):
        df_mutated.loc[idx, "open"] = 1800.0
        df_mutated.loc[idx, "high"] = 1805.0
        df_mutated.loc[idx, "low"] = 1700.0
        df_mutated.loc[idx, "close"] = 1710.0

    res_after_future_mutation = kb_strategy.evaluate(df_mutated, as_of_index=60)
    assert res_after_future_mutation == res_approved


# ---------------------------------------------------------------------------
# 6. StrategyEngine, SignalLifecycleManager, RiskEngine & BacktestEngine Integration
# ---------------------------------------------------------------------------
def test_knowledge_strategy_cannot_bypass_risk_engine_or_backtest_gates() -> None:
    """Verify signals triggered by KnowledgeBasedStrategy pass through SignalLifecycleManager and RiskEngine."""
    loader = KnowledgeDocumentLoader()
    extractor = DeterministicRuleExtractor()
    repo = KnowledgeRepository()

    doc = loader.load_text(
        "# XAUUSD H1 Trend Rule\n"
        "On XAUUSD H1 in an uptrend regime where close > ema_21 and ema_9 > ema_50, "
        "enter a BUY setup with a 1.5 ATR stop and 2.0R target.\n",
        title="H1 Trend Playbook",
        source_type=KnowledgeSourceType.MARKDOWN_NOTES,
        ingested_at=FIXED_TS,
    )
    repo.add_document(doc)
    rules = repo.add_rules(extractor.extract_from_document(doc, extracted_at=FIXED_TS))
    repo.approve_rule(rules[0].rule_id)

    df = _make_trending_xauusd_df(n_bars=70, direction="up")

    # Configure StrategyEngine where ONLY knowledge_based_strategy is enabled as the primary trigger
    conf_cfg = ConfluenceWeightsConfig(
        enable_ema_trend=False,
        enable_market_structure=False,
        enable_liquidity_sweep=False,
        enable_fvg=False,
        enable_fvg_retest=False,
        enable_mtf=False,
        enable_knowledge_strategy=True,
        weight_knowledge_strategy=Decimal("1.00"),
        minimum_confidence_threshold=Decimal("0.50"),
    )
    strat_engine = StrategyEngine(
        confluence_config=conf_cfg,
        knowledge_repository=repo,
    )

    spec = InstrumentSpec(
        symbol="XAUUSD",
        contract_size=Decimal("100"),
        tick_size=Decimal("0.01"),
        tick_value=Decimal("1.00"),
        volume_step=Decimal("0.01"),
        minimum_volume=Decimal("0.01"),
        maximum_volume=Decimal("50.00"),
        margin_rate=Decimal("0.01"),
    )
    risk_params = RiskParameters(
        risk_per_trade_percent=Decimal("1.0"),
        max_daily_loss_percent=Decimal("3.0"),
        max_total_drawdown_percent=Decimal("10.0"),
        max_open_positions=2,
        minimum_risk_reward_ratio=Decimal("1.5"),
    )

    # 6A. Healthy account -> RiskEngine approves knowledge-triggered signal (GENERATED -> RISK_APPROVED)
    healthy_account = AccountState(
        balance=Decimal("10000.00"),
        equity=Decimal("10000.00"),
        free_margin=Decimal("10000.00"),
        daily_starting_balance=Decimal("10000.00"),
        current_daily_pnl=Decimal("0.00"),
        open_positions_count=0,
    )
    report_ok = strat_engine.analyze(
        df,
        symbol="XAUUSD",
        timeframe=Timeframe.H1,
        as_of_index=60,
        account_state=healthy_account,
        risk_parameters=risk_params,
        instrument_spec=spec,
    )
    assert report_ok.signal.direction == SignalDirection.BUY
    assert report_ok.risk_status == RiskEvaluationStatus.RISK_APPROVED
    assert report_ok.risk_decision is not None and report_ok.risk_decision.approved is True
    assert report_ok.lifecycle_record is not None
    assert report_ok.lifecycle_record.state == SignalLifecycleState.RISK_APPROVED

    # 6B. Breached daily loss account -> RiskEngine strictly REJECTS the knowledge-triggered signal!
    breached_account = AccountState(
        balance=Decimal("9650.00"),
        equity=Decimal("9650.00"),
        free_margin=Decimal("9650.00"),
        daily_starting_balance=Decimal("10000.00"),
        current_daily_pnl=Decimal("-350.00"),  # -3.5% daily loss > 3.0% limit
        open_positions_count=0,
    )
    strat_engine_2 = StrategyEngine(
        confluence_config=conf_cfg,
        knowledge_repository=repo,
    )
    report_rej = strat_engine_2.analyze(
        df,
        symbol="XAUUSD",
        timeframe=Timeframe.H1,
        as_of_index=60,
        account_state=breached_account,
        risk_parameters=risk_params,
        instrument_spec=spec,
    )
    assert report_rej.signal.direction == SignalDirection.BUY
    assert report_rej.risk_status == RiskEvaluationStatus.RISK_REJECTED
    assert report_rej.risk_decision is not None and report_rej.risk_decision.approved is False
    assert (
        report_rej.risk_decision.rejection_reason is not None
        and RiskRejectionCode.DAILY_LOSS_LIMIT_EXCEEDED.value in report_rej.risk_decision.rejection_reason
    )
    assert report_rej.lifecycle_record is not None
    assert report_rej.lifecycle_record.state == SignalLifecycleState.RISK_REJECTED

    # 6C. Full BacktestEngine run with knowledge-enabled StrategyEngine
    bt_engine = BacktestEngine(
        BacktestConfig(warmup_bars=55, risk_parameters=risk_params, instrument_spec=spec),
        strategy_engine=StrategyEngine(confluence_config=conf_cfg, knowledge_repository=repo),
    )
    bt_result = bt_engine.run(df)
    assert bt_result.total_bars == 70
    assert len(bt_result.lifecycle_records) >= 1
    for l_rec in bt_result.lifecycle_records:
        visited = [ev.to_state for ev in l_rec.history]
        assert SignalLifecycleState.GENERATED in visited
        assert (
            SignalLifecycleState.RISK_APPROVED in visited
            or SignalLifecycleState.RISK_REJECTED in visited
        )
    assert len(bt_result.trades) >= 1
    assert len(bt_result.risk_rejections) >= 1
