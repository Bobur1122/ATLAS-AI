"""Educational document ingestion, transcript sanitization, and deterministic chunking for ATLAS AI Phase 5."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Sequence

from atlas_ai.domain.market import Timeframe
from atlas_ai.knowledge.config import KnowledgeConfig
from atlas_ai.knowledge.exceptions import KnowledgeIngestionError
from atlas_ai.knowledge.models import (
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeSourceType,
    compute_sha256_hex,
)

# Regex patterns for SRT / VTT timecode lines and inline timestamps
TIMECODE_LINE_RE = re.compile(
    r"^\s*(?P<start>\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?)\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?)(?:\s+.*)?$"
)
INLINE_TIMESTAMP_RE = re.compile(r"\[(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]")
SRT_SEQ_LINE_RE = re.compile(r"^\s*\d+\s*$")
HTML_TAG_RE = re.compile(r"<[^>]+>")
NOISE_TAG_RE = re.compile(r"\[(?:music|applause|laughter|silence|inaudible)\]", re.IGNORECASE)
MARKDOWN_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$")

KNOWN_SYMBOLS = ("XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD")
KNOWN_TIMEFRAME_ALIASES: dict[str, Timeframe] = {
    "M1": Timeframe.M1,
    "1M": Timeframe.M1,
    "M5": Timeframe.M5,
    "5M": Timeframe.M5,
    "M15": Timeframe.M15,
    "15M": Timeframe.M15,
    "M30": Timeframe.M30,
    "30M": Timeframe.M30,
    "H1": Timeframe.H1,
    "1H": Timeframe.H1,
    "H4": Timeframe.H4,
    "4H": Timeframe.H4,
    "D1": Timeframe.D1,
    "1D": Timeframe.D1,
}


class TextChunker:
    """Splits normalized educational documents or transcripts into traceable KnowledgeChunk records."""

    def __init__(self, config: KnowledgeConfig | None = None) -> None:
        self._config = config or KnowledgeConfig()

    @property
    def config(self) -> KnowledgeConfig:
        return self._config

    def chunk_blocks(
        self,
        *,
        document_id: str,
        blocks: Sequence[dict[str, str | None]],
        default_symbols: tuple[str, ...],
        default_timeframes: tuple[Timeframe, ...],
    ) -> tuple[KnowledgeChunk, ...]:
        """Group structured paragraph/caption blocks into size-bounded KnowledgeChunk objects."""
        max_chars = self._config.max_chunk_chars
        min_chars = self._config.min_chunk_chars

        chunks: list[KnowledgeChunk] = []
        acc_texts: list[str] = []
        acc_len = 0
        curr_section: str | None = None
        curr_start_ts: str | None = None
        curr_end_ts: str | None = None

        def flush_accumulator() -> None:
            nonlocal acc_texts, acc_len, curr_start_ts, curr_end_ts
            combined = "\n\n".join(t for t in acc_texts if t.strip()).strip()
            if len(combined) >= min_chars or (combined and not chunks):
                seq_idx = len(chunks)
                c_hash = compute_sha256_hex(f"{document_id}:{seq_idx}:{combined}")
                chunk_id = f"chk-{document_id}-{seq_idx:04d}"
                sym_tags = _detect_symbols(combined, default_symbols)
                tf_tags = _detect_timeframes(combined, default_timeframes)
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        sequence_index=seq_idx,
                        text=combined,
                        content_hash=c_hash,
                        section_title=curr_section,
                        start_timestamp=curr_start_ts,
                        end_timestamp=curr_end_ts,
                        symbol_tags=sym_tags,
                        timeframe_tags=tf_tags,
                    )
                )
            acc_texts = []
            acc_len = 0
            curr_start_ts = None
            curr_end_ts = None

        for blk in blocks:
            raw_txt = (blk.get("text") or "").strip()
            if not raw_txt:
                continue
            blk_section = blk.get("section_title")
            blk_start = blk.get("start_timestamp")
            blk_end = blk.get("end_timestamp")

            # Flush if section changed or adding block would exceed max_chars
            section_changed = (
                blk_section is not None
                and curr_section is not None
                and blk_section != curr_section
                and acc_texts
            )
            exceeds_max = acc_texts and (acc_len + len(raw_txt) + 2 > max_chars)
            if section_changed or exceeds_max:
                flush_accumulator()

            if blk_section is not None:
                curr_section = blk_section
            if curr_start_ts is None and blk_start is not None:
                curr_start_ts = blk_start
            if blk_end is not None:
                curr_end_ts = blk_end
            elif blk_start is not None:
                curr_end_ts = blk_start

            # If a single block is larger than max_chars, slice it deterministically on sentence/line boundaries
            if len(raw_txt) > max_chars:
                sentences = re.split(r"(?<=[.!?])\s+|\n+", raw_txt)
                for sent in sentences:
                    s_clean = sent.strip()
                    if not s_clean:
                        continue
                    if acc_texts and (acc_len + len(s_clean) + 1 > max_chars):
                        flush_accumulator()
                    acc_texts.append(s_clean)
                    acc_len += len(s_clean) + 1
            else:
                acc_texts.append(raw_txt)
                acc_len += len(raw_txt) + 2

        if acc_texts:
            flush_accumulator()

        return tuple(chunks)


def _detect_symbols(text: str, default_symbols: tuple[str, ...]) -> tuple[str, ...]:
    upper_text = text.upper()
    found: list[str] = []
    for sym in KNOWN_SYMBOLS:
        if sym in upper_text:
            found.append(sym)
    if "GOLD" in upper_text and "XAUUSD" not in found:
        found.append("XAUUSD")
    for d_sym in default_symbols:
        if d_sym not in found:
            found.append(d_sym)
    return tuple(found)


def _detect_timeframes(text: str, default_timeframes: tuple[Timeframe, ...]) -> tuple[Timeframe, ...]:
    tokens = re.findall(r"\b(?:M1|1M|M5|5M|M15|15M|M30|30M|H1|1H|H4|4H|D1|1D)\b", text.upper())
    found: list[Timeframe] = []
    for tok in tokens:
        tf = KNOWN_TIMEFRAME_ALIASES.get(tok)
        if tf is not None and tf not in found:
            found.append(tf)
    if not found:
        return default_timeframes
    return tuple(found)


class KnowledgeDocumentLoader:
    """Ingests, sanitizes, and chunks educational trading materials (.txt, .md, .srt, .vtt, .json)."""

    def __init__(self, config: KnowledgeConfig | None = None) -> None:
        self._config = config or KnowledgeConfig()
        self._chunker = TextChunker(self._config)

    @property
    def config(self) -> KnowledgeConfig:
        return self._config

    def load_file(
        self,
        file_path: str | Path,
        *,
        title: str | None = None,
        author: str = "Unknown",
        source_type: KnowledgeSourceType | None = None,
        symbol_tags: Sequence[str] | None = None,
        timeframe_tags: Sequence[Timeframe] | None = None,
        ingested_at: datetime | None = None,
        document_id: str | None = None,
    ) -> KnowledgeDocument:
        """Load an educational file from disk into a validated KnowledgeDocument."""
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise KnowledgeIngestionError(f"Knowledge file not found: '{path}'.")

        suffix = path.suffix.lower()
        inferred_type = source_type or self._infer_source_type(suffix)
        try:
            raw_content = path.read_text(encoding="utf-8")
        except Exception as exc:
            raise KnowledgeIngestionError(f"Failed to read '{path}': {exc}") from exc

        doc_title = title or path.stem.replace("_", " ").replace("-", " ").title()
        return self.load_text(
            raw_content,
            title=doc_title,
            source_type=inferred_type,
            source_uri=str(path.as_posix()),
            author=author,
            symbol_tags=symbol_tags,
            timeframe_tags=timeframe_tags,
            ingested_at=ingested_at,
            document_id=document_id,
        )

    def load_text(
        self,
        raw_content: str,
        *,
        title: str,
        source_type: KnowledgeSourceType = KnowledgeSourceType.TEXT_NOTES,
        source_uri: str = "inline://document",
        author: str = "Unknown",
        symbol_tags: Sequence[str] | None = None,
        timeframe_tags: Sequence[Timeframe] | None = None,
        ingested_at: datetime | None = None,
        document_id: str | None = None,
    ) -> KnowledgeDocument:
        """Parse and chunk raw text, transcript content, or playbook JSON into a KnowledgeDocument."""
        if not isinstance(raw_content, str) or not raw_content.strip():
            raise KnowledgeIngestionError("Cannot ingest an empty or whitespace-only document.")

        ts_ingested = ingested_at or datetime(2026, 11, 15, 12, 0, 0, tzinfo=timezone.utc)
        default_syms = (
            tuple(s.strip().upper() for s in symbol_tags if s.strip())
            if symbol_tags
            else (self._config.default_symbol,)
        )
        default_tfs = (
            tuple(timeframe_tags)
            if timeframe_tags
            else (self._config.default_timeframe,)
        )

        metadata: dict[str, object] = {}
        eff_title = title.strip() or "Untitled Document"
        eff_author = author.strip() or "Unknown"

        if source_type in (KnowledgeSourceType.PLAYBOOK_JSON, KnowledgeSourceType.SCREENSHOT_ANNOTATION):
            blocks, normalized_text, json_meta, eff_title, eff_author, default_syms, default_tfs = (
                self._parse_json_content(
                    raw_content,
                    default_title=eff_title,
                    default_author=eff_author,
                    default_syms=default_syms,
                    default_tfs=default_tfs,
                )
            )
            metadata.update(json_meta)
        elif source_type in (KnowledgeSourceType.TRANSCRIPT_SRT, KnowledgeSourceType.TRANSCRIPT_VTT):
            blocks, normalized_text = self._parse_subtitle_transcript(raw_content)
        elif source_type == KnowledgeSourceType.MARKDOWN_NOTES:
            blocks, normalized_text = self._parse_markdown_notes(raw_content)
        else:
            blocks, normalized_text = self._parse_plain_text_or_transcript(raw_content)

        if not normalized_text.strip():
            raise KnowledgeIngestionError(
                f"Document '{eff_title}' contained no usable text after sanitization."
            )

        content_hash = compute_sha256_hex(normalized_text)
        doc_id = document_id or f"doc-{content_hash[:12]}"

        detected_syms = _detect_symbols(normalized_text, default_syms)
        detected_tfs = _detect_timeframes(normalized_text, default_tfs)

        chunks = self._chunker.chunk_blocks(
            document_id=doc_id,
            blocks=blocks,
            default_symbols=detected_syms,
            default_timeframes=detected_tfs,
        )
        if not chunks:
            raise KnowledgeIngestionError(
                f"Document '{eff_title}' produced zero chunks after parsing."
            )

        return KnowledgeDocument(
            document_id=doc_id,
            title=eff_title,
            source_type=source_type,
            source_uri=source_uri,
            author=eff_author,
            ingested_at=ts_ingested,
            content_hash=content_hash,
            raw_char_count=len(raw_content),
            normalized_text=normalized_text,
            symbol_tags=detected_syms,
            timeframe_tags=detected_tfs,
            chunks=chunks,
            metadata=metadata,
        )

    @staticmethod
    def _infer_source_type(suffix: str) -> KnowledgeSourceType:
        mapping = {
            ".srt": KnowledgeSourceType.TRANSCRIPT_SRT,
            ".vtt": KnowledgeSourceType.TRANSCRIPT_VTT,
            ".md": KnowledgeSourceType.MARKDOWN_NOTES,
            ".markdown": KnowledgeSourceType.MARKDOWN_NOTES,
            ".json": KnowledgeSourceType.PLAYBOOK_JSON,
            ".txt": KnowledgeSourceType.TEXT_NOTES,
        }
        if suffix not in mapping:
            raise KnowledgeIngestionError(
                f"Unsupported knowledge file extension '{suffix}'. Supported: {list(mapping.keys())}."
            )
        return mapping[suffix]

    @staticmethod
    def _sanitize_line(line: str) -> str:
        """Remove HTML tags, subtitle noise tags, and collapse redundant whitespace."""
        cleaned = HTML_TAG_RE.sub("", line)
        cleaned = NOISE_TAG_RE.sub("", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _parse_subtitle_transcript(
        self, raw_content: str
    ) -> tuple[list[dict[str, str | None]], str]:
        """Parse .srt or .vtt transcript files, stripping sequence numbers and capturing timestamps."""
        lines = raw_content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        blocks: list[dict[str, str | None]] = []
        norm_lines: list[str] = []

        curr_start: str | None = None
        curr_end: str | None = None
        curr_caption_lines: list[str] = []

        def flush_cue() -> None:
            nonlocal curr_caption_lines, curr_start, curr_end
            if curr_caption_lines:
                txt = " ".join(curr_caption_lines).strip()
                if txt:
                    blocks.append(
                        {
                            "text": txt,
                            "section_title": "Transcript",
                            "start_timestamp": curr_start,
                            "end_timestamp": curr_end,
                        }
                    )
                    norm_lines.append(txt)
            curr_caption_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                flush_cue()
                continue
            if stripped.upper().startswith("WEBVTT") or stripped.startswith("NOTE"):
                continue
            if SRT_SEQ_LINE_RE.match(stripped):
                continue

            tc_match = TIMECODE_LINE_RE.match(stripped)
            if tc_match:
                flush_cue()
                curr_start = tc_match.group("start").replace(",", ".")
                curr_end = tc_match.group("end").replace(",", ".")
                continue

            sanitized = self._sanitize_line(stripped)
            if sanitized:
                curr_caption_lines.append(sanitized)

        flush_cue()
        return blocks, "\n\n".join(norm_lines)

    def _parse_markdown_notes(
        self, raw_content: str
    ) -> tuple[list[dict[str, str | None]], str]:
        """Parse Markdown notes into section-titled paragraph blocks."""
        lines = raw_content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        blocks: list[dict[str, str | None]] = []
        norm_paragraphs: list[str] = []

        curr_section: str | None = None
        para_lines: list[str] = []

        def flush_para() -> None:
            nonlocal para_lines
            if para_lines:
                txt = " ".join(para_lines).strip()
                if txt:
                    blocks.append(
                        {
                            "text": txt,
                            "section_title": curr_section,
                            "start_timestamp": None,
                            "end_timestamp": None,
                        }
                    )
                    norm_paragraphs.append(txt)
            para_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                flush_para()
                continue

            h_match = MARKDOWN_HEADING_RE.match(stripped)
            if h_match:
                flush_para()
                curr_section = self._sanitize_line(h_match.group("title"))
                continue

            cleaned = self._sanitize_line(stripped.lstrip("-*+ "))
            if cleaned:
                para_lines.append(cleaned)

        flush_para()
        return blocks, "\n\n".join(norm_paragraphs)

    def _parse_plain_text_or_transcript(
        self, raw_content: str
    ) -> tuple[list[dict[str, str | None]], str]:
        """Parse plain text or inline-timestamped lesson transcripts."""
        paragraphs = re.split(r"\n\s*\n", raw_content.replace("\r\n", "\n"))
        blocks: list[dict[str, str | None]] = []
        norm_paragraphs: list[str] = []

        for para in paragraphs:
            ts_matches = INLINE_TIMESTAMP_RE.findall(para)
            start_ts = ts_matches[0] if ts_matches else None
            end_ts = ts_matches[-1] if len(ts_matches) > 1 else start_ts
            no_ts_para = INLINE_TIMESTAMP_RE.sub("", para)
            cleaned = self._sanitize_line(no_ts_para)
            if cleaned:
                blocks.append(
                    {
                        "text": cleaned,
                        "section_title": None,
                        "start_timestamp": start_ts,
                        "end_timestamp": end_ts,
                    }
                )
                norm_paragraphs.append(cleaned)

        return blocks, "\n\n".join(norm_paragraphs)

    def _parse_json_content(
        self,
        raw_content: str,
        *,
        default_title: str,
        default_author: str,
        default_syms: tuple[str, ...],
        default_tfs: tuple[Timeframe, ...],
    ) -> tuple[
        list[dict[str, str | None]],
        str,
        dict[str, object],
        str,
        str,
        tuple[str, ...],
        tuple[Timeframe, ...],
    ]:
        """Parse structured trading playbook or screenshot annotation JSON."""
        try:
            payload = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise KnowledgeIngestionError(f"Malformed JSON knowledge document: {exc}") from exc

        if not isinstance(payload, dict) or not payload:
            raise KnowledgeIngestionError("JSON knowledge document must be a non-empty JSON object.")

        eff_title = str(payload.get("title") or default_title).strip()
        eff_author = str(payload.get("author") or default_author).strip()

        syms = default_syms
        if isinstance(payload.get("symbols"), list) and payload["symbols"]:
            syms = tuple(str(s).strip().upper() for s in payload["symbols"] if str(s).strip())

        tfs = default_tfs
        if isinstance(payload.get("timeframes"), list) and payload["timeframes"]:
            parsed_tfs: list[Timeframe] = []
            for tf_raw in payload["timeframes"]:
                tf_key = str(tf_raw).strip().upper()
                if tf_key in KNOWN_TIMEFRAME_ALIASES:
                    parsed_tfs.append(KNOWN_TIMEFRAME_ALIASES[tf_key])
            if parsed_tfs:
                tfs = tuple(parsed_tfs)

        blocks: list[dict[str, str | None]] = []
        norm_parts: list[str] = []
        metadata: dict[str, object] = {}

        if isinstance(payload.get("rules"), list):
            metadata["playbook_rules"] = payload["rules"]
            for idx, r_obj in enumerate(payload["rules"]):
                if isinstance(r_obj, dict):
                    r_title = str(r_obj.get("title") or f"Rule {idx + 1}").strip()
                    r_desc = str(r_obj.get("description") or r_title).strip()
                    txt = f"{r_title}: {r_desc}"
                    blocks.append(
                        {
                            "text": txt,
                            "section_title": r_title,
                            "start_timestamp": None,
                            "end_timestamp": None,
                        }
                    )
                    norm_parts.append(txt)

        if isinstance(payload.get("sections"), list):
            for sec in payload["sections"]:
                if isinstance(sec, dict):
                    sec_title = str(sec.get("title") or "Section").strip()
                    sec_text = self._sanitize_line(str(sec.get("content") or sec.get("text") or ""))
                    if sec_text:
                        blocks.append(
                            {
                                "text": sec_text,
                                "section_title": sec_title,
                                "start_timestamp": (
                                    str(sec["timestamp"]) if sec.get("timestamp") else None
                                ),
                                "end_timestamp": None,
                            }
                        )
                        norm_parts.append(sec_text)

        if isinstance(payload.get("notes") or payload.get("content"), str):
            note_txt = self._sanitize_line(str(payload.get("notes") or payload.get("content")))
            if note_txt:
                blocks.append(
                    {
                        "text": note_txt,
                        "section_title": eff_title,
                        "start_timestamp": None,
                        "end_timestamp": None,
                    }
                )
                norm_parts.append(note_txt)

        if not blocks:
            raise KnowledgeIngestionError(
                "JSON knowledge document must contain 'rules', 'sections', 'notes', or 'content'."
            )

        return blocks, "\n\n".join(norm_parts), metadata, eff_title, eff_author, syms, tfs
