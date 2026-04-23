"""Tests for VaultScanner — written BEFORE implementation (TDD)."""
import hashlib
from pathlib import Path

import pytest

from vault_qdrant.scanner import scan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _md(tmp_path: Path, rel: str, body: str) -> Path:
    """Write a markdown file at *tmp_path / rel* and return its Path."""
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


SIMPLE_MD = "# Hello\n\nSome content here.\n"

FRONTMATTER_MD = """\
---
tags:
  - python
  - qdrant
type: note
created: "2024-01-15"
status: active
projects:
  - medusa
---
# Body

Content below the frontmatter.
"""

NO_FRONTMATTER_MD = "# Just a heading\n\nNo YAML here.\n"


# ---------------------------------------------------------------------------
# Exclusion tests
# ---------------------------------------------------------------------------

class TestExclusions:
    def test_skip_obsidian_dir(self, tmp_path: Path):
        """Files inside .obsidian/ must not appear in results."""
        _md(tmp_path, ".obsidian/config.md", SIMPLE_MD)
        _md(tmp_path, "note.md", SIMPLE_MD)

        results = scan(tmp_path)

        paths = [r["file_path"] for r in results]
        assert not any(".obsidian" in p for p in paths)
        assert any("note.md" in p for p in paths)

    def test_skip_templates_dir(self, tmp_path: Path):
        """Files inside templates/ must not appear in results."""
        _md(tmp_path, "templates/daily.md", SIMPLE_MD)
        _md(tmp_path, "note.md", SIMPLE_MD)

        results = scan(tmp_path)

        paths = [r["file_path"] for r in results]
        assert not any("templates" in p for p in paths)

    def test_skip_attachments_dir(self, tmp_path: Path):
        """Files inside attachments/ must not appear in results."""
        _md(tmp_path, "attachments/image.md", SIMPLE_MD)
        _md(tmp_path, "note.md", SIMPLE_MD)

        results = scan(tmp_path)

        paths = [r["file_path"] for r in results]
        assert not any("attachments" in p for p in paths)

    def test_skip_non_markdown(self, tmp_path: Path):
        """Non-.md files (.txt, .png placeholders) must be excluded."""
        txt = tmp_path / "readme.txt"
        txt.write_text("plain text", encoding="utf-8")
        png = tmp_path / "photo.png"
        png.write_bytes(b"\x89PNG")
        _md(tmp_path, "note.md", SIMPLE_MD)

        results = scan(tmp_path)

        paths = [r["file_path"] for r in results]
        assert all(p.endswith(".md") for p in paths)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Frontmatter tests
# ---------------------------------------------------------------------------

class TestFrontmatter:
    def test_frontmatter_parsing(self, tmp_path: Path):
        """YAML frontmatter fields are correctly extracted."""
        _md(tmp_path, "note.md", FRONTMATTER_MD)

        results = scan(tmp_path)

        assert len(results) == 1
        doc = results[0]
        assert doc["tags"] == ["python", "qdrant"]
        assert doc["type"] == "note"
        assert doc["created"] == "2024-01-15"
        assert doc["status"] == "active"
        assert doc["projects"] == ["medusa"]

    def test_frontmatter_missing(self, tmp_path: Path):
        """Files without frontmatter get empty/None defaults (type is inferred, not None)."""
        _md(tmp_path, "bare.md", NO_FRONTMATTER_MD)

        results = scan(tmp_path)

        assert len(results) == 1
        doc = results[0]
        assert doc["tags"] == []
        # type is now inferred from folder path (no folder match → "note")
        assert doc["type"] == "note"
        assert doc["created"] is None
        assert doc["status"] is None
        assert doc["projects"] == []


# ---------------------------------------------------------------------------
# Hash tests
# ---------------------------------------------------------------------------

class TestDocHash:
    def test_doc_hash_deterministic(self, tmp_path: Path):
        """SHA-256 of file content is stable across two scan calls."""
        _md(tmp_path, "note.md", SIMPLE_MD)

        first = scan(tmp_path)[0]["doc_hash"]
        second = scan(tmp_path)[0]["doc_hash"]

        assert first == second
        expected = hashlib.sha256(SIMPLE_MD.encode()).hexdigest()
        assert first == expected


# ---------------------------------------------------------------------------
# Return shape test
# ---------------------------------------------------------------------------

class TestScanOutput:
    def test_scan_returns_list_of_scanned_docs(self, tmp_path: Path):
        """scan() returns a list of dicts with the required keys."""
        _md(tmp_path, "a.md", SIMPLE_MD)
        _md(tmp_path, "sub/b.md", FRONTMATTER_MD)

        results = scan(tmp_path)

        required_keys = {
            "file_path", "content", "tags", "type",
            "created", "status", "projects", "doc_hash",
        }
        assert isinstance(results, list)
        assert len(results) == 2
        for doc in results:
            assert isinstance(doc, dict)
            assert required_keys.issubset(doc.keys()), (
                f"Missing keys: {required_keys - doc.keys()}"
            )
            assert isinstance(doc["file_path"], str)
            assert isinstance(doc["content"], str)
            assert isinstance(doc["tags"], list)
            assert isinstance(doc["projects"], list)
            assert isinstance(doc["doc_hash"], str)
            assert len(doc["doc_hash"]) == 64  # SHA-256 hex length


# ---------------------------------------------------------------------------
# Doc-type inference tests
# ---------------------------------------------------------------------------

class TestDocTypeInference:
    def test_sessions_folder_infers_session(self, tmp_path: Path):
        _md(tmp_path, "sessions/2026-01-01-daily.md", NO_FRONTMATTER_MD)
        result = scan(tmp_path)[0]
        assert result["type"] == "session"

    def test_todos_folder_infers_todos(self, tmp_path: Path):
        _md(tmp_path, "todos/Active TODOs.md", NO_FRONTMATTER_MD)
        result = scan(tmp_path)[0]
        assert result["type"] == "todos"

    def test_projects_folder_infers_project(self, tmp_path: Path):
        _md(tmp_path, "projects/medusa/overview.md", NO_FRONTMATTER_MD)
        result = scan(tmp_path)[0]
        assert result["type"] == "project"

    def test_architecture_adr_pattern(self, tmp_path: Path):
        _md(tmp_path, "projects/medusa/architecture/0003-hosting.md", NO_FRONTMATTER_MD)
        result = scan(tmp_path)[0]
        assert result["type"] == "adr"

    def test_components_folder_infers_component(self, tmp_path: Path):
        _md(tmp_path, "components/button.md", NO_FRONTMATTER_MD)
        result = scan(tmp_path)[0]
        assert result["type"] == "component"

    def test_unknown_folder_infers_note(self, tmp_path: Path):
        _md(tmp_path, "random/deep/path.md", NO_FRONTMATTER_MD)
        result = scan(tmp_path)[0]
        assert result["type"] == "note"

    def test_frontmatter_type_takes_precedence(self, tmp_path: Path):
        _md(tmp_path, "sessions/2026-01-01-daily.md", FRONTMATTER_MD)
        result = scan(tmp_path)[0]
        # FRONTMATTER_MD has type: note in frontmatter — must win over folder inference
        assert result["type"] == "note"

    def test_type_source_frontmatter_when_explicit(self, tmp_path: Path):
        _md(tmp_path, "sessions/daily.md", FRONTMATTER_MD)
        result = scan(tmp_path)[0]
        assert result["type_source"] == "frontmatter"

    def test_type_source_inferred_when_missing(self, tmp_path: Path):
        _md(tmp_path, "sessions/daily.md", NO_FRONTMATTER_MD)
        result = scan(tmp_path)[0]
        assert result["type_source"] == "inferred"

    def test_specs_folder_infers_spec(self, tmp_path: Path):
        _md(tmp_path, "projects/medusa/specs/api-design.md", NO_FRONTMATTER_MD)
        result = scan(tmp_path)[0]
        assert result["type"] == "spec"


# ---------------------------------------------------------------------------
# Inline tag harvesting tests
# ---------------------------------------------------------------------------

class TestInlineTagHarvesting:
    def test_inline_tags_merged_with_frontmatter_tags(self, tmp_path: Path):
        content = "---\ntags: [existing]\n---\n\nSome text with #newtag here."
        _md(tmp_path, "notes/test.md", content)
        result = scan(tmp_path)[0]
        assert "existing" in result["tags"]
        assert "newtag" in result["tags"]

    def test_inline_tags_deduplicated(self, tmp_path: Path):
        content = "---\ntags: [mytag]\n---\n\n#mytag appears again."
        _md(tmp_path, "notes/test.md", content)
        result = scan(tmp_path)[0]
        assert result["tags"].count("mytag") == 1

    def test_inline_tags_skip_code_fence(self, tmp_path: Path):
        content = "---\ntags: []\n---\n\nCode:\n```\n#notatag\n```\n"
        _md(tmp_path, "notes/test.md", content)
        result = scan(tmp_path)[0]
        assert "notatag" not in result["tags"]
