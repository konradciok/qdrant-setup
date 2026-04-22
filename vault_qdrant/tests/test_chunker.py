"""Tests for MarkdownChunker — written BEFORE implementation (TDD)."""
import textwrap

import pytest
import tiktoken

from vault_qdrant.chunker import chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENC = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_ENC.encode(text))


def _repeat_words(seed: str, target_tokens: int) -> str:
    """Return text that is approximately *target_tokens* tokens long."""
    words = (seed + " ") * (target_tokens // _token_count(seed) + 1)
    # Trim to rough target
    while _token_count(words) > target_tokens + 5:
        words = words.rsplit(" ", 1)[0]
    return words.strip()


def _big_paragraph(target_tokens: int, seed: str = "word") -> str:
    """Generate a paragraph of *target_tokens* tokens."""
    return _repeat_words(seed, target_tokens)


# ---------------------------------------------------------------------------
# Test 1 — H2 splitting
# ---------------------------------------------------------------------------

class TestH2Splitting:
    def test_h2_splitting(self):
        """Content with 2 H2 sections produces 2 chunks with correct h2 payloads."""
        content = textwrap.dedent("""\
            # Doc Title

            ## Section Alpha

            {alpha}

            ## Section Beta

            {beta}
        """).format(
            alpha=_big_paragraph(200, "alpha"),
            beta=_big_paragraph(200, "beta"),
        )

        chunks = chunk(content)

        assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
        headings = {c["h2"] for c in chunks}
        assert headings == {"Section Alpha", "Section Beta"}


# ---------------------------------------------------------------------------
# Test 2 — H3 splitting when H2 section is too large
# ---------------------------------------------------------------------------

class TestH3SplittingWhenH2TooLarge:
    def test_h3_splitting_when_h2_too_large(self):
        """A single H2 section >800 tokens splits further at H3 boundaries."""
        content = textwrap.dedent("""\
            # Big Doc

            ## Huge Section

            ### Part One

            {part1}

            ### Part Two

            {part2}
        """).format(
            part1=_big_paragraph(500, "part1"),
            part2=_big_paragraph(500, "part2"),
        )

        chunks = chunk(content)

        # The H2 is >800 tokens total, so it must be split at H3 level
        assert len(chunks) >= 2, "Expected at least 2 chunks after H3 splitting"
        h3_values = {c["h3"] for c in chunks}
        assert "Part One" in h3_values
        assert "Part Two" in h3_values


# ---------------------------------------------------------------------------
# Test 3 — Short section merging
# ---------------------------------------------------------------------------

class TestShortSectionMerging:
    def test_short_section_merging(self):
        """A section <80 tokens merges with next sibling (within same H2).

        Interpretation: when an H2 section has <80 tokens of body text, its
        content is merged into the following H2 sibling. The merged chunk carries
        the heading of the absorbing sibling. The key assertion is that no
        standalone chunk exists for the tiny section with <80 tokens.
        """
        content = textwrap.dedent("""\
            # Doc

            ## Section A

            tiny text here

            ## Section B

            {body}
        """).format(body=_big_paragraph(200, "body"))

        chunks = chunk(content)

        # The tiny Section A (<80 tokens) must not appear as a standalone chunk
        # with fewer than 80 tokens. It should be merged into Section B.
        for c in chunks:
            if c["h2"] == "Section A" and "Section B" not in (c.get("merged_from") or ""):
                # If Section A appears alone, it must have been merged (>=80 tokens)
                assert _token_count(c["text"]) >= 80, (
                    f"Tiny section appeared standalone with {_token_count(c['text'])} tokens"
                )

        # After merging: total chunks should be 1 (A merged into B)
        assert len(chunks) == 1, f"Expected 1 merged chunk, got {len(chunks)}"


# ---------------------------------------------------------------------------
# Test 4 — Code fence not split
# ---------------------------------------------------------------------------

class TestCodeFenceNotSplit:
    def test_code_fence_not_split(self):
        """A triple-backtick code block is never broken across chunk boundaries."""
        fenced_code = "```python\n" + "\n".join(f"x_{i} = {i}" for i in range(100)) + "\n```"
        content = textwrap.dedent("""\
            # Doc

            ## Section

            {fence}
        """).format(fence=fenced_code)

        chunks = chunk(content)

        # Every chunk must have balanced fence markers (even count or zero)
        for c in chunks:
            text = c["text"]
            opens = text.count("```")
            assert opens % 2 == 0, (
                f"Unbalanced fence markers in chunk {c['chunk_index']}: "
                f"{opens} backtick markers"
            )


# ---------------------------------------------------------------------------
# Test 5 — Table not split
# ---------------------------------------------------------------------------

class TestTableNotSplit:
    def test_table_not_split(self):
        """A markdown table is never broken across chunk boundaries."""
        table_rows = ["| Col A | Col B |", "| --- | --- |"] + [
            f"| row_{i} | value_{i} |" for i in range(50)
        ]
        table = "\n".join(table_rows)

        content = textwrap.dedent("""\
            # Doc

            ## Section

            {table}
        """).format(table=table)

        chunks = chunk(content)

        # The table header must appear in the same chunk as separator and data rows
        for c in chunks:
            text = c["text"]
            if "| Col A |" in text:
                assert "| --- |" in text, "Table header separated from separator row"
                assert "| row_" in text, "Table header chunk missing data rows"


# ---------------------------------------------------------------------------
# Test 6 — Wiki link with display alias
# ---------------------------------------------------------------------------

class TestWikiLinkDisplayAlias:
    def test_wiki_link_display_alias(self):
        """[[note|alias]] → 'alias' in chunk text."""
        content = "# Doc\n\n## Section\n\nSee [[some-note|the alias text]] for details.\n"

        chunks = chunk(content)

        assert len(chunks) >= 1
        combined_text = " ".join(c["text"] for c in chunks)
        assert "the alias text" in combined_text
        assert "[[" not in combined_text
        assert "some-note" not in combined_text


# ---------------------------------------------------------------------------
# Test 7 — Wiki link plain (no alias)
# ---------------------------------------------------------------------------

class TestWikiLinkPlain:
    def test_wiki_link_plain(self):
        """[[note]] → 'note' in chunk text."""
        content = "# Doc\n\n## Section\n\nSee [[my-note]] for more info.\n"

        chunks = chunk(content)

        assert len(chunks) >= 1
        combined_text = " ".join(c["text"] for c in chunks)
        assert "my-note" in combined_text
        assert "[[" not in combined_text


# ---------------------------------------------------------------------------
# Test 8 — Forward links extracted
# ---------------------------------------------------------------------------

class TestForwardLinksExtracted:
    def test_forward_links_extracted(self):
        """forward_links in payload contains note names from wiki-links."""
        content = textwrap.dedent("""\
            # Doc

            ## Section

            See [[note-alpha]] and [[note-beta|Beta Note]] for info.
        """)

        chunks = chunk(content)

        all_links: list[str] = []
        for c in chunks:
            all_links.extend(c["forward_links"])

        assert "note-alpha" in all_links
        # The note name (not the alias) is recorded in forward_links
        assert "note-beta" in all_links


# ---------------------------------------------------------------------------
# Test 9 — Heading breadcrumb
# ---------------------------------------------------------------------------

class TestHeadingBreadcrumb:
    def test_heading_breadcrumb(self):
        """Each chunk payload has h1, h2, h3 keys (None if absent)."""
        content = textwrap.dedent("""\
            # My H1

            ## My H2

            ### My H3

            Some content here that is meaningful text.

            ## Another H2

            Content without H3.
        """)

        chunks = chunk(content)

        assert len(chunks) >= 1
        for c in chunks:
            assert "h1" in c
            assert "h2" in c
            assert "h3" in c

        # Find the chunk with H3
        h3_chunks = [c for c in chunks if c["h3"] is not None]
        assert len(h3_chunks) >= 1
        assert h3_chunks[0]["h1"] == "My H1"
        assert h3_chunks[0]["h2"] == "My H2"
        assert h3_chunks[0]["h3"] == "My H3"

        # Find chunk without H3 under "Another H2"
        no_h3_chunks = [c for c in chunks if c["h2"] == "Another H2"]
        assert len(no_h3_chunks) >= 1
        assert no_h3_chunks[0]["h3"] is None


# ---------------------------------------------------------------------------
# Test 10 — doc_type token targets
# ---------------------------------------------------------------------------

class TestDocTypeTokenTargets:
    """Test that doc_type-aware targets are respected.

    Interpretation: We test that for a large document that requires splitting,
    spec chunks land <= 900 tokens (target 500-700 + margin) and session chunks
    land <= 600 tokens (target 300-400 + margin). We do not test exact ranges
    because splits depend on H3 boundary positions in the content.
    """

    def _make_large_doc(self, num_h2: int, tokens_per_section: int) -> str:
        """Create a document with multiple H2 sections of given token size."""
        lines = ["# Large Document\n"]
        for i in range(num_h2):
            lines.append(f"\n## Section {i}\n")
            lines.append(f"\n### Sub {i}a\n")
            lines.append(_big_paragraph(tokens_per_section // 2, f"word{i}a"))
            lines.append(f"\n\n### Sub {i}b\n")
            lines.append(_big_paragraph(tokens_per_section // 2, f"word{i}b"))
            lines.append("\n")
        return "\n".join(lines)

    def test_spec_token_target(self):
        """spec doc_type produces chunks within the 500-700 token target range."""
        content = self._make_large_doc(num_h2=3, tokens_per_section=1200)
        chunks = chunk(content, doc_type="spec")

        assert len(chunks) > 0

        oversized = [
            c for c in chunks
            if _token_count(c["text"]) > 900  # 500-700 target + reasonable margin
        ]
        assert len(oversized) == 0, (
            f"spec chunks too large for target 500-700: "
            f"{[_token_count(c['text']) for c in oversized]}"
        )

    def test_session_token_target(self):
        """session doc_type produces chunks within the 300-400 token target range."""
        content = self._make_large_doc(num_h2=3, tokens_per_section=900)
        chunks = chunk(content, doc_type="session")

        assert len(chunks) > 0

        oversized = [
            c for c in chunks
            if _token_count(c["text"]) > 600  # 300-400 target + margin
        ]
        assert len(oversized) == 0, (
            f"session chunks too large for target 300-400: "
            f"{[_token_count(c['text']) for c in oversized]}"
        )
