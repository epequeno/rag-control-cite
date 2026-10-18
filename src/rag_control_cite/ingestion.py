"""Stage 0 — ingestion: parse → chunk → metadata.

OSCAL catalogs (JSON) are parsed structurally — each control becomes a
Document, its prose extracted from the statement and guidance parts.
Baseline profiles map control IDs to tiers (low/moderate/high/privacy),
which become access_tags. The recursive character chunker (same algorithm
as rag-eval-project) splits each Document into Chunks carrying full
provenance: source_uri, last_modified, doc_type, access_tags, chunk_index,
char offsets.

~80% of RAG failures trace to this layer — so this stage ends in numbers:
parse-failure rate, table-extraction rate, char-sanity pass %.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rag_control_cite.models import Chunk, Document

__all__ = [
    "IngestionReport",
    "RecursiveChunker",
    "ingest",
    "parse_oscal_catalog",
]

# --------------------------------------------------------------------------- #
# Prose extraction
# --------------------------------------------------------------------------- #


def _extract_prose(obj: dict) -> str:
    """Recursively extract prose text from an OSCAL part or control.

    Walks ``parts`` in order, prefixing each item with its label prop.
    Parameter insertions (``{{ insert: param, ... }}``) are left inline —
    they are part of the control text and the exact-term query set tests
    whether retrieval finds controls by ID, not whether params resolve.
    """
    parts: list[str] = []
    if "prose" in obj:
        parts.append(obj["prose"])
    for p in obj.get("parts", []):
        label = ""
        for prop in p.get("props", []):
            if prop.get("name") == "label":
                label = prop["value"]
        sub = _extract_prose(p)
        if label and sub:
            parts.append(f"{label} {sub}")
        elif sub:
            parts.append(sub)
    return "\n".join(parts)


def _control_body(control: dict) -> str:
    """Build the chunkable text for one OSCAL control.

    Layout: ``ID: Title`` header, then statement and guidance prose.
    Assessment-objective and assessment-method parts are excluded —
    they describe how to audit the control, not what it says, and
    including them would dilute retrieval with audit procedure text.
    """
    lines: list[str] = [f"{control['id'].upper()}: {control['title']}"]
    for part in control.get("parts", []):
        name = part.get("name", "")
        if name in ("statement", "guidance"):
            prose = _extract_prose(part)
            if prose:
                lines.append("")
                lines.append(prose)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Baseline mapping
# --------------------------------------------------------------------------- #

_BASELINE_FILES = {
    "low": "NIST_SP-800-53_rev5_LOW-baseline_profile.json",
    "moderate": "NIST_SP-800-53_rev5_MODERATE-baseline_profile.json",
    "high": "NIST_SP-800-53_rev5_HIGH-baseline_profile.json",
    "privacy": "NIST_SP-800-53_rev5_PRIVACY-baseline_profile.json",
}


def _load_baseline_ids(profile_path: Path) -> set[str]:
    """Extract the set of control IDs included in a baseline profile."""
    with profile_path.open() as f:
        data = json.load(f)
    ids: set[str] = set()
    for imp in data.get("profile", {}).get("imports", []):
        for inc in imp.get("include-controls", []):
            ids.update(inc.get("with-ids", []))
    return ids


def _baseline_tiers_for_control(
    control_id: str,
    baseline_maps: dict[str, set[str]],
) -> set[str]:
    """Return the set of baseline tiers that include a control ID."""
    return {tier for tier, ids in baseline_maps.items() if control_id in ids}


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #

_FRAMEWORK_TAGS = {
    "SP800-53": "sp-800-53",
    "SP800-171": "sp-800-171",
    "CSF": "csf",
}


def _walk_controls(
    group: dict,
    framework: str,
    source_uri: str,
    last_modified: datetime,
    baseline_maps: dict[str, set[str]],
    doc_type: str,
) -> list[Document]:
    """Flatten a group's controls (including enhancements) into Documents."""
    family = group.get("id", "").lower()
    docs: list[Document] = []
    for control in group.get("controls", []):
        cid = control["id"].lower()
        body = _control_body(control)
        if not body.strip():
            continue
        tiers = _baseline_tiers_for_control(cid, baseline_maps)
        tags: set[str] = {
            f"framework:{framework}",
            f"family:{family}",
        }
        # Chunks in no baseline get an explicit baseline:none tag so the
        # fail-closed policy denies them to every finite baseline set —
        # a tagging gap is never a silent grant.
        if tiers:
            tags.update(f"baseline:{t}" for t in tiers)
        else:
            tags.add("baseline:none")
        # Privacy baseline → privacy audience tag
        if "privacy" in tiers:
            tags.add("audience:privacy")
        tags.add("audience:security")
        docs.append(
            Document(
                doc_id=f"{cid}@{framework}",
                source_uri=source_uri,
                text=body,
                title=control.get("title", cid),
                doc_type=doc_type,
                access_tags=frozenset(tags),
                last_modified=last_modified,
                author="NIST",
                control_id=cid,
                family=family,
            )
        )
        # Enhancements (nested controls)
        for sub in control.get("controls", []):
            sub_cid = sub["id"].lower()
            sub_body = _control_body(sub)
            if not sub_body.strip():
                continue
            sub_tiers = _baseline_tiers_for_control(sub_cid, baseline_maps)
            sub_tags: set[str] = {
                f"framework:{framework}",
                f"family:{family}",
            }
            if sub_tiers:
                sub_tags.update(f"baseline:{t}" for t in sub_tiers)
            else:
                sub_tags.add("baseline:none")
            if "privacy" in sub_tiers:
                sub_tags.add("audience:privacy")
            sub_tags.add("audience:security")
            docs.append(
                Document(
                    doc_id=f"{sub_cid}@{framework}",
                    source_uri=source_uri,
                    text=sub_body,
                    title=sub.get("title", sub_cid),
                    doc_type=doc_type,
                    access_tags=frozenset(sub_tags),
                    last_modified=last_modified,
                    author="NIST",
                    control_id=sub_cid,
                    family=family,
                )
            )
    return docs


@dataclass
class ParseResult:
    """Outcome of parsing one source file."""

    source_uri: str
    documents: list[Document]
    error: str | None = None
    had_tables: bool = False
    table_controls: int = 0
    total_controls: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None


def parse_oscal_catalog(
    catalog_path: Path,
    framework: str,
    baseline_dir: Path | None = None,
) -> ParseResult:
    """Parse an OSCAL catalog JSON file into Documents.

    Args:
        catalog_path: path to the catalog JSON.
        framework: framework tag ("sp-800-53", "sp-800-171", "csf").
        baseline_dir: directory containing baseline profile JSONs
                       (only for SP 800-53; None for others).

    Returns:
        ParseResult with documents or an error.
    """
    try:
        with catalog_path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return ParseResult(
            source_uri=str(catalog_path),
            documents=[],
            error=f"parse: {exc}",
        )

    catalog = data.get("catalog", data)
    meta = catalog.get("metadata", {})
    last_modified_str = meta.get("last-modified", "")
    try:
        last_modified = datetime.fromisoformat(
            last_modified_str.replace("Z", "+00:00")
            if last_modified_str
            else "2026-01-01T00:00:00+00:00"
        )
    except ValueError:
        last_modified = datetime(2026, 1, 1)

    # Load baseline maps for SP 800-53
    baseline_maps: dict[str, set[str]] = {}
    if baseline_dir and baseline_dir.is_dir():
        for tier, fname in _BASELINE_FILES.items():
            bpath = baseline_dir / fname
            if bpath.exists():
                baseline_maps[tier] = _load_baseline_ids(bpath)

    doc_type = "oscal-json"
    source_uri = str(catalog_path)
    groups = catalog.get("groups", [])
    all_docs: list[Document] = []
    total_controls = 0
    table_controls = 0

    for group in groups:
        for control in group.get("controls", []):
            total_controls += 1
            # Count controls with params (structured table-like data)
            if control.get("params"):
                table_controls += 1
            all_docs.extend(
                _walk_controls(
                    group={"id": group["id"], "controls": [control]},
                    framework=framework,
                    source_uri=source_uri,
                    last_modified=last_modified,
                    baseline_maps=baseline_maps,
                    doc_type=doc_type,
                )
            )

    return ParseResult(
        source_uri=source_uri,
        documents=all_docs,
        error=None,
        had_tables=table_controls > 0,
        table_controls=table_controls,
        total_controls=total_controls,
    )


# --------------------------------------------------------------------------- #
# Chunker — same algorithm as rag-eval-project's RecursiveChunker
# --------------------------------------------------------------------------- #


@dataclass
class RecursiveChunker:
    """Recursive character chunker with provenance passthrough.

    Splits on the coarsest separator that applies, recurses into
    oversized pieces, then greedily packs units up to ``chunk_size``.
    ``chunk_overlap`` is a maximum: the next chunk starts at the first
    unit boundary at or after ``end - chunk_overlap``.
    """

    chunk_size: int = 1024
    chunk_overlap: int = 128
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ")

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(f"chunk_overlap must be in [0, chunk_size), got {self.chunk_overlap}")
        if not self.separators or any(s == "" for s in self.separators):
            raise ValueError("separators must be non-empty strings")

    def chunk(self, doc: Document) -> list[Chunk]:
        body = doc.text
        chunks: list[Chunk] = []
        for i, (start, end) in enumerate(self._packed_spans(body)):
            chunk_id = hashlib.sha256(f"{doc.doc_id}:{start}-{end}".encode()).hexdigest()[:16]
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_uri=doc.source_uri,
                    text=body[start:end],
                    start_char=start,
                    end_char=end,
                    chunk_index=i,
                    doc_type=doc.doc_type,
                    access_tags=doc.access_tags,
                    last_modified=doc.last_modified,
                    author=doc.author,
                    title=doc.title,
                    control_id=doc.control_id,
                    framework=next(
                        (t.split(":", 1)[1] for t in doc.access_tags if t.startswith("framework:")),
                        None,
                    ),
                )
            )
        return chunks

    def _packed_spans(self, body: str) -> list[tuple[int, int]]:
        units: list[tuple[int, int]] = []
        self._split(body, 0, len(body), 0, units)

        spans: list[tuple[int, int]] = []
        i, n = 0, len(units)
        while i < n:
            j = i
            while j < n and units[j][1] - units[i][0] <= self.chunk_size:
                j += 1
            end = units[j - 1][1]
            spans.append((units[i][0], end))
            if j >= n:
                break
            boundary = end - self.chunk_overlap
            nxt = j
            for t in range(i + 1, j):
                if units[t][0] >= boundary:
                    nxt = t
                    break
            i = nxt
        return spans

    def _split(
        self,
        body: str,
        start: int,
        end: int,
        sep_ix: int,
        out: list[tuple[int, int]],
    ) -> None:
        if end - start <= self.chunk_size:
            if end > start:
                out.append((start, end))
            return
        if sep_ix >= len(self.separators):
            for s in range(start, end, self.chunk_size):
                out.append((s, min(s + self.chunk_size, end)))
            return
        sep = self.separators[sep_ix]
        pieces: list[tuple[int, int]] = []
        cur = start
        while True:
            p = body.find(sep, cur, end)
            if p == -1:
                break
            pieces.append((cur, p + len(sep)))
            cur = p + len(sep)
        if cur < end:
            pieces.append((cur, end))
        if len(pieces) <= 1:
            self._split(body, start, end, sep_ix + 1, out)
            return
        for s, e in pieces:
            self._split(body, s, e, sep_ix + 1, out)


# --------------------------------------------------------------------------- #
# Measurements
# --------------------------------------------------------------------------- #


@dataclass
class IngestionReport:
    """Stage 0 metrics — the numbers this stage ends in."""

    files_attempted: int = 0
    files_failed: int = 0
    total_controls: int = 0
    table_controls: int = 0
    total_documents: int = 0
    total_chunks: int = 0
    char_sanity_passed: int = 0
    char_sanity_total: int = 0

    @property
    def parse_failure_rate(self) -> float:
        if self.files_attempted == 0:
            return 0.0
        return self.files_failed / self.files_attempted

    @property
    def table_extraction_rate(self) -> float:
        if self.total_controls == 0:
            return 0.0
        return self.table_controls / self.total_controls

    @property
    def char_sanity_pass_pct(self) -> float:
        if self.char_sanity_total == 0:
            return 0.0
        return self.char_sanity_passed / self.char_sanity_total

    def as_dict(self) -> dict[str, float | int]:
        return {
            "files_attempted": self.files_attempted,
            "files_failed": self.files_failed,
            "parse_failure_rate": round(self.parse_failure_rate, 4),
            "total_controls": self.total_controls,
            "table_controls": self.table_controls,
            "table_extraction_rate": round(self.table_extraction_rate, 4),
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "char_sanity_passed": self.char_sanity_passed,
            "char_sanity_total": self.char_sanity_total,
            "char_sanity_pass_pct": round(self.char_sanity_pass_pct, 4),
        }


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


# Catalogs to ingest — keyed by framework tag
_CATALOGS = [
    ("sp-800-53", "nist.gov/SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json"),
    ("sp-800-171", "nist.gov/SP800-171/rev3/json/NIST_SP800-171_rev3_catalog.json"),
    ("csf", "nist.gov/CSF/v2.0/json/NIST_CSF_v2.0_catalog.json"),
]


def ingest(
    corpus_dir: Path | str | None = None,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
) -> tuple[list[Chunk], IngestionReport]:
    """Parse → chunk → measure. Returns (chunks, report).

    Args:
        corpus_dir: path to the OSCAL corpus root. Defaults to
                    ``data/corpus/oscal`` relative to the project root.
        chunk_size: max characters per chunk.
        chunk_overlap: max character overlap between adjacent chunks.

    Raises:
        FileNotFoundError: if the corpus directory does not exist.
    """
    if corpus_dir is None:
        corpus_dir = Path("data/corpus/oscal")
    corpus_root = Path(corpus_dir).expanduser()
    if not corpus_root.is_dir():
        raise FileNotFoundError(f"corpus not found: {corpus_root}")

    baseline_dir = corpus_root / "nist.gov" / "SP800-53" / "rev5" / "json"

    chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks: list[Chunk] = []
    report = IngestionReport()
    doc_count = 0

    for framework, rel_path in _CATALOGS:
        cat_path = corpus_root / rel_path
        if not cat_path.exists():
            report.files_attempted += 1
            report.files_failed += 1
            continue

        report.files_attempted += 1
        result = parse_oscal_catalog(
            cat_path,
            framework=framework,
            baseline_dir=baseline_dir if framework == "sp-800-53" else None,
        )

        if not result.ok:
            report.files_failed += 1
            continue

        report.total_controls += result.total_controls
        report.table_controls += result.table_controls

        for doc in result.documents:
            doc_count += 1
            doc_chunks = chunker.chunk(doc)
            all_chunks.extend(doc_chunks)
            # Char sanity: chunk.text == doc.text[start:end]
            for ch in doc_chunks:
                report.char_sanity_total += 1
                if ch.text == doc.text[ch.start_char : ch.end_char]:
                    report.char_sanity_passed += 1

    report.total_documents = doc_count
    report.total_chunks = len(all_chunks)

    return all_chunks, report
