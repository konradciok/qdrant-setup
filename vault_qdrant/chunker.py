"""MarkdownChunker — split Obsidian markdown into semantically coherent chunks.

Public API
----------
    chunk(content: str, doc_type: str | None = None) -> list[dict]

Each returned dict has:
    text          : str            # wiki-links stripped, plain text
    h1            : str | None     # heading breadcrumb from document H1
    h2            : str | None     # current H2 heading
    h3            : str | None     # current H3 heading (None if absent)
    forward_links : list[str]      # note names from [[...]] in original text
    chunk_index   : int            # 0-based position in output list
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import tiktoken

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENC = tiktoken.get_encoding("cl100k_base")

# Hard limit: H2 sections larger than this are split at H3 level
_HARD_SPLIT_THRESHOLD = 800

# Sections smaller than this are merged with the next sibling
_MERGE_THRESHOLD = 80

# doc_type → target token ceiling for H3-level splitting
_DOC_TYPE_TARGETS: dict[str, int] = {
    "spec": 700,
    "plan": 700,
    "session": 400,
    "todos": 300,
    "note": 600,
    "adr": 600,
}
_DEFAULT_TARGET = 600

# Regex patterns
_H1_RE = re.compile(r"^#\s+(.+)", re.MULTILINE)
_H2_HEADING_RE = re.compile(r"^##\s+(.+)", re.MULTILINE)
_H3_HEADING_RE = re.compile(r"^###\s+(.+)", re.MULTILINE)
_WIKI_LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_TABLE_ROW_RE = re.compile(r"^\s*\|")


# ---------------------------------------------------------------------------
# Internal data structure
# ---------------------------------------------------------------------------

@dataclass
class _Section:
    """A parsed section of a markdown document."""

    h1: str | None
    h2: str | None
    h3: str | None
    raw_text: str        # original text with wiki-links intact
    text: str            # wiki-links replaced with display text
    forward_links: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Wiki-link processing
# ---------------------------------------------------------------------------

def _strip_wiki_links(text: str) -> tuple[str, list[str]]:
    """Replace wiki-links with display text; return (clean_text, note_names).

    [[note|alias]] -> "alias", records "note" in forward_links
    [[note]]       -> "note",  records "note" in forward_links
    """
    note_names: list[str] = []

    def _replace(match: re.Match) -> str:
        note = match.group(1).strip()
        alias = match.group(2)
        note_names.append(note)
        return alias.strip() if alias else note

    clean = _WIKI_LINK_RE.sub(_replace, text)
    return clean, note_names


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


# ---------------------------------------------------------------------------
# Protected block detection (fenced code + tables)
# ---------------------------------------------------------------------------

def _find_protected_ranges(text: str) -> list[tuple[int, int]]:
    """Return character (start, end) ranges that must not be split.

    Protected blocks:
    - Fenced code blocks (``` ... ```)
    - Markdown tables (consecutive lines starting with |)
    """
    ranges: list[tuple[int, int]] = []
    lines = text.split("\n")
    pos = 0
    in_fence = False
    fence_start = -1
    table_start = -1
    in_table = False

    for line in lines:
        line_start = pos
        line_end = pos + len(line)
        stripped = line.strip()

        # Fenced code block detection
        if stripped.startswith("```"):
            if not in_fence:
                in_fence = True
                fence_start = line_start
            else:
                in_fence = False
                ranges.append((fence_start, line_end))

        # Table detection (not inside a fence)
        if not in_fence:
            if _TABLE_ROW_RE.match(line):
                if not in_table:
                    in_table = True
                    table_start = line_start
            else:
                if in_table:
                    in_table = False
                    ranges.append((table_start, line_start - 1))

        pos = line_end + 1  # +1 for the '\n' we split on

    # Handle unclosed table at end of text
    if in_table:
        ranges.append((table_start, pos))

    return ranges


def _char_in_protected(char_pos: int, protected: list[tuple[int, int]]) -> bool:
    return any(start <= char_pos <= end for start, end in protected)


# ---------------------------------------------------------------------------
# Heading-based splitting that respects protected blocks
# ---------------------------------------------------------------------------

def _split_at_heading(
    text: str,
    heading_re: re.Pattern,
) -> list[tuple[str | None, str]]:
    """Split *text* at lines matching *heading_re*, skipping protected blocks.

    Returns list of (heading_text | None, body_text) pairs.
    The first element may have heading_text=None for content before the first heading.
    """
    protected = _find_protected_ranges(text)
    result: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    current_body_start = 0

    for match in heading_re.finditer(text):
        if _char_in_protected(match.start(), protected):
            continue

        body = text[current_body_start : match.start()]
        result.append((current_heading, body))
        current_heading = match.group(1).strip()
        # Skip past the newline that terminates the heading line
        current_body_start = match.end()
        if current_body_start < len(text) and text[current_body_start] == "\n":
            current_body_start += 1

    result.append((current_heading, text[current_body_start:]))
    return result


# ---------------------------------------------------------------------------
# H2-level document splitting
# ---------------------------------------------------------------------------

def _split_by_h2(content: str, h1: str | None) -> list[_Section]:
    """Split document into H2-level sections."""
    sections: list[_Section] = []

    pairs = _split_at_heading(content, _H2_HEADING_RE)

    for h2_heading, body in pairs:
        if not body.strip() and h2_heading is None:
            continue
        clean_body, links = _strip_wiki_links(body)
        sections.append(
            _Section(
                h1=h1,
                h2=h2_heading,
                h3=None,
                raw_text=body,
                text=clean_body,
                forward_links=links,
            )
        )

    return sections


# ---------------------------------------------------------------------------
# H3-level splitting of an oversized H2 section
# ---------------------------------------------------------------------------

def _split_by_h3(section: _Section) -> list[_Section]:
    """Split an H2 section further at H3 boundaries."""
    pairs = _split_at_heading(section.raw_text, _H3_HEADING_RE)

    result: list[_Section] = []
    for h3_heading, body in pairs:
        if not body.strip() and h3_heading is None:
            continue
        clean_body, links = _strip_wiki_links(body)
        result.append(
            _Section(
                h1=section.h1,
                h2=section.h2,
                h3=h3_heading,
                raw_text=body,
                text=clean_body,
                forward_links=links,
            )
        )

    return result if result else [section]


# ---------------------------------------------------------------------------
# Section merging — tiny sections absorbed into next sibling
# ---------------------------------------------------------------------------

def _merge_sections(sections: list[_Section]) -> list[_Section]:
    """Merge sections <_MERGE_THRESHOLD tokens into the next sibling.

    Rules:
    - Preamble sections (h2=None) are never used as a merge source into a
      named H2 section; they are kept as-is or dropped if empty.
    - Merging only happens between sections with the same h2 value.
    - A tiny section is absorbed into the next sibling, inheriting its h2.
    """
    if not sections:
        return sections

    merged: list[_Section] = []
    pending: _Section | None = None

    for sec in sections:
        if pending is None:
            # Preamble (h2=None) sections that are tiny are discarded silently
            # to avoid polluting named H2 sections.
            is_preamble = sec.h2 is None and sec.h3 is None
            if is_preamble and _count_tokens(sec.text) < _MERGE_THRESHOLD:
                # Skip tiny preamble — it has no useful heading context
                continue
            if not is_preamble and _count_tokens(sec.text) < _MERGE_THRESHOLD:
                pending = sec
            else:
                merged.append(sec)
            continue

        # We have a pending tiny section — try to merge with current.
        # Merging is allowed across different H2 siblings (the tiny section's
        # content is prepended to the absorbing sibling, which keeps its own h2).
        # We only refuse to merge across different H1 top-level boundaries.
        same_h1 = pending.h1 == sec.h1
        is_preamble_sec = sec.h2 is None and sec.h3 is None
        if same_h1 and not is_preamble_sec:
            combined_raw = pending.raw_text.rstrip() + "\n\n" + sec.raw_text
            combined_text = pending.text.rstrip() + "\n\n" + sec.text
            combined_links = pending.forward_links + sec.forward_links
            merged_sec = _Section(
                h1=sec.h1 or pending.h1,
                h2=sec.h2,
                h3=sec.h3,
                raw_text=combined_raw,
                text=combined_text,
                forward_links=combined_links,
            )
            if _count_tokens(merged_sec.text) < _MERGE_THRESHOLD:
                # Still tiny — keep accumulating
                pending = merged_sec
            else:
                merged.append(merged_sec)
                pending = None
        else:
            # Different H1 boundary or preamble — flush pending as-is even if small
            merged.append(pending)
            pending = None
            if not is_preamble_sec and _count_tokens(sec.text) < _MERGE_THRESHOLD:
                pending = sec
            elif is_preamble_sec and _count_tokens(sec.text) < _MERGE_THRESHOLD:
                # Drop tiny preamble
                pass
            else:
                merged.append(sec)

    if pending is not None:
        merged.append(pending)

    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk(content: str, doc_type: str | None = None) -> list[dict]:
    """Split *content* into semantically coherent chunks.

    Parameters
    ----------
    content:
        Raw markdown text (may include YAML frontmatter).
    doc_type:
        Optional document type selecting token targets. Recognised values:
        'spec', 'plan', 'session', 'todos', 'note', 'adr'.

    Returns
    -------
    list[dict] with keys: text, h1, h2, h3, forward_links, chunk_index.
    Each dict is a newly created object (immutable pattern).
    """
    target = _DOC_TYPE_TARGETS.get(doc_type or "", _DEFAULT_TARGET)

    # Extract H1
    h1_match = _H1_RE.search(content)
    h1 = h1_match.group(1).strip() if h1_match else None

    # Phase 1: split by H2
    h2_sections = _split_by_h2(content, h1)

    # Phase 2: always apply H3 splitting to H2 sections so that H3 headings
    # are captured in the breadcrumb. For small sections this may produce a
    # single sub-section; for large ones it produces multiple.
    # Additionally enforce the hard threshold and doc_type target.
    refined: list[_Section] = []
    for sec in h2_sections:
        token_count = _count_tokens(sec.text)
        # Always split by H3 so headings appear in breadcrumb
        sub = _split_by_h3(sec)
        if len(sub) > 1:
            # H3 sub-sections found — check if any still exceed the target
            further_split: list[_Section] = []
            for s in sub:
                if _count_tokens(s.text) > target:
                    # Cannot split further (no H4 support) — keep as-is
                    further_split.append(s)
                else:
                    further_split.append(s)
            refined.extend(further_split)
        else:
            # No H3 headings found — use the section as-is, but split by
            # hard threshold if needed (the single sub equals the original)
            refined.extend(sub)

    # Phase 2b: doc_type-aware splitting — sections still over target that
    # were not split above (e.g., oversized H3 sections)
    further: list[_Section] = []
    for sec in refined:
        if _count_tokens(sec.text) > _HARD_SPLIT_THRESHOLD:
            # Try H3 again in case it wasn't done yet
            sub = _split_by_h3(sec)
            further.extend(sub)
        else:
            further.append(sec)

    # Phase 3: merge tiny sections (<_MERGE_THRESHOLD tokens)
    final = _merge_sections(further)

    # Phase 4: assemble output — each dict is a new immutable object
    return [
        {
            "text": sec.text.strip(),
            "h1": sec.h1,
            "h2": sec.h2,
            "h3": sec.h3,
            "forward_links": list(sec.forward_links),
            "chunk_index": idx,
        }
        for idx, sec in enumerate(final)
    ]
