# Contextualizer Prompt Caching & Chunk-ID Collision Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two data-correctness and cost bugs found in the Qdrant RAG pipeline audit: (1) add Anthropic prompt caching to the contextualizer so the full document is cached across N chunk calls instead of being re-sent each time, and (2) add `chunk_index` to the `_chunk_id` hash to prevent silent ID collisions when a file has duplicate heading names.

**Architecture:** Two surgical, independent changes. Task 1 rewires `contextualizer.py` to use structured content blocks with `cache_control: {"type": "ephemeral"}` on the document block. Task 2 adds `chunk_index` to the SHA-256 input in `upserter.py`. Task 3 is a one-line type annotation verification in `cli.py`. Each task is self-contained with no inter-task dependencies.

**Tech Stack:** Python 3.12, `anthropic` SDK (prompt caching via `cache_control`), `hashlib`, pytest, `unittest.mock`.

---

## File Map

| File | Change |
|------|--------|
| `vault_qdrant/contextualizer.py` | Restructure `messages` to use content blocks; add `cache_control` on document block |
| `vault_qdrant/tests/test_contextualizer.py` | New test file covering caching behaviour |
| `vault_qdrant/upserter.py` | Add `chunk_index` parameter to `_chunk_id()` and update call site |
| `vault_qdrant/tests/test_upserter.py` | Add two collision tests; update `_expected_id` helper |
| `vault_qdrant/cli.py` | Verify `_contextualize_chunks` annotation (no `| None`) |

---

## Task 1: Prompt caching in `contextualizer.py`

**Files:**
- Modify: `vault_qdrant/contextualizer.py`
- Create: `vault_qdrant/tests/test_contextualizer.py`

### Background

Anthropic's prompt caching works by prefix-matching: if the beginning of the `messages` array is byte-identical to a prior call within the TTL window, the cached tokens are reused. By placing the stable document block *before* the volatile chunk block and tagging it with `cache_control: {"type": "ephemeral"}`, all N chunk calls for the same document pay full price only on the first call; the remaining N-1 calls hit the cache.

The current single-string `content` makes this impossible — the full prompt string changes with every chunk call.

- [x] **Step 1: Write the failing tests**

Create `vault_qdrant/tests/test_contextualizer.py`:

```python
"""Tests for contextualizer prompt caching behaviour."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from vault_qdrant.contextualizer import Contextualizer


def _make_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    return resp


class TestContextualizerCaching:
    def test_document_block_has_cache_control(self) -> None:
        """The document content block must carry cache_control=ephemeral."""
        with patch("vault_qdrant.contextualizer.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = _make_response("Context sentence.")
            MockAnthropic.return_value = mock_client

            c = Contextualizer(api_key="test-key")
            c.contextualize("full document text", "just chunk text")

            create_kwargs = mock_client.messages.create.call_args.kwargs
            messages = create_kwargs["messages"]
            assert len(messages) == 1
            user_content = messages[0]["content"]
            doc_block = user_content[0]
            assert doc_block["cache_control"] == {"type": "ephemeral"}, (
                "Document block must have cache_control={'type': 'ephemeral'}"
            )

    def test_chunk_block_has_no_cache_control(self) -> None:
        """The chunk content block must NOT carry cache_control (it is volatile)."""
        with patch("vault_qdrant.contextualizer.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = _make_response("Context sentence.")
            MockAnthropic.return_value = mock_client

            c = Contextualizer(api_key="test-key")
            c.contextualize("full document text", "just chunk text")

            create_kwargs = mock_client.messages.create.call_args.kwargs
            user_content = create_kwargs["messages"][0]["content"]
            chunk_block = user_content[1]
            assert "cache_control" not in chunk_block, (
                "Chunk block must not have cache_control — it changes per call"
            )

    def test_document_appears_before_chunk(self) -> None:
        """Document block must precede chunk block (prefix caching requirement)."""
        with patch("vault_qdrant.contextualizer.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = _make_response("Context sentence.")
            MockAnthropic.return_value = mock_client

            c = Contextualizer(api_key="test-key")
            c.contextualize("THE DOCUMENT", "THE CHUNK")

            user_content = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
            assert "THE DOCUMENT" in user_content[0]["text"]
            assert "THE CHUNK" in user_content[1]["text"]

    def test_return_value_prepends_context(self) -> None:
        """contextualize() must return '<context>\\n\\n<original chunk>'."""
        with patch("vault_qdrant.contextualizer.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = _make_response("Intro sentence.")
            MockAnthropic.return_value = mock_client

            c = Contextualizer(api_key="test-key")
            result = c.contextualize("doc", "chunk text")

            assert result == "Intro sentence.\n\nchunk text"
```

- [x] **Step 2: Run the failing tests**

```bash
cd /Users/konradciok/repositories/medusa/qdrant
uv run pytest vault_qdrant/tests/test_contextualizer.py -v 2>&1 | head -40
```

Expected: `FAILED` on the `cache_control` assertion — current code passes a string content, not a list of blocks.

- [x] **Step 3: Apply the fix to `vault_qdrant/contextualizer.py`**

Replace the `contextualize` method. The key diff is replacing the single-string `content` with a two-block list:

```python
    def contextualize(self, document: str, chunk: str) -> str:
        """Add contextual prefix to a chunk.

        Calls Claude to generate 1-2 sentences situating the chunk within
        the document. Returns the context prepended to the chunk.

        The document block is tagged cache_control=ephemeral so repeated calls
        for different chunks of the same document reuse the cached tokens.

        Args:
            document: Full document text
            chunk: Chunk of text from the document

        Returns:
            Contextual summary + chunk (separated by two newlines)

        Raises:
            anthropic.APIError: If API call fails
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=150,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Here is a document:\n<document>\n{document}\n</document>",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "\n\nSituate this chunk in the document in 1-2 sentences:\n"
                                f"<chunk>\n{chunk}\n</chunk>"
                            ),
                        },
                    ],
                }
            ],
        )

        context = response.content[0].text
        return f"{context}\n\n{chunk}"
```

- [x] **Step 4: Run the new tests — must pass**

```bash
cd /Users/konradciok/repositories/medusa/qdrant
uv run pytest vault_qdrant/tests/test_contextualizer.py -v
```

Expected: 4 tests `PASSED`.

- [x] **Step 5: Run the full test suite**

```bash
cd /Users/konradciok/repositories/medusa/qdrant
uv run pytest vault_qdrant/tests/ -v 2>&1 | tail -20
```

Expected: all tests pass. The existing `test_cli_contextualizer.py::test_contextualizer_prepends_context` still passes because the return-value contract is unchanged.

- [x] **Step 6: Commit**

```bash
cd /Users/konradciok/repositories/medusa/qdrant
git add vault_qdrant/contextualizer.py vault_qdrant/tests/test_contextualizer.py
git commit -m "perf: add prompt caching to contextualizer (cache_control on document block)"
```

---

## Task 2: Collision-safe `_chunk_id` in `upserter.py`

**Files:**
- Modify: `vault_qdrant/upserter.py`
- Modify: `vault_qdrant/tests/test_upserter.py`

### Background

`_chunk_id(file_path, h2, h3)` hashes only the heading path. A file with two sections that share the same heading (e.g., two `## Notes` sections) produces identical IDs — the second upsert silently overwrites the first. Adding `chunk_index` (already present in every chunk dict) to the hash input makes every chunk ID unique by construction.

The change requires updating one call site in `_build_point` and the `_expected_id` test helper.

- [x] **Step 1: Write the failing tests**

Add to the bottom of `vault_qdrant/tests/test_upserter.py`:

```python
def test_duplicate_headings_different_chunk_index_no_collision() -> None:
    """Two chunks with identical h2/h3 but different chunk_index must produce different IDs."""
    id_first = _chunk_id("notes.md", "Notes", None, 0)
    id_second = _chunk_id("notes.md", "Notes", None, 1)
    assert id_first != id_second, (
        "Chunks with duplicate headings but different chunk_index must not collide"
    )


def test_chunk_id_includes_chunk_index() -> None:
    """chunk_index alone must change the resulting ID."""
    id_zero = _chunk_id("file.md", "Section", "Sub", 0)
    id_one = _chunk_id("file.md", "Section", "Sub", 1)
    assert id_zero != id_one
```

Ensure `_chunk_id` is imported at the top of `test_upserter.py`. If it is not already imported, add:

```python
from vault_qdrant.upserter import _chunk_id, delete_orphans, upsert_chunks
```

- [x] **Step 2: Run the failing tests**

```bash
cd /Users/konradciok/repositories/medusa/qdrant
uv run pytest vault_qdrant/tests/test_upserter.py::test_duplicate_headings_different_chunk_index_no_collision vault_qdrant/tests/test_upserter.py::test_chunk_id_includes_chunk_index -v
```

Expected: `FAILED` — `TypeError: _chunk_id() takes from 2 to 3 positional arguments but 4 were given`.

- [x] **Step 3: Update `_chunk_id` in `vault_qdrant/upserter.py` (lines 42–45)**

```python
# Before
def _chunk_id(file_path: str, h2: str | None, h3: str | None = None) -> str:
    """Return a deterministic 32-char hex ID from SHA-256(file_path + h2 + h3)."""
    raw = (file_path + (h2 or "") + (h3 or "")).encode()
    return hashlib.sha256(raw).hexdigest()[:32]

# After
def _chunk_id(file_path: str, h2: str | None, h3: str | None, chunk_index: int) -> str:
    """Return a deterministic 32-char hex ID from SHA-256(file_path + h2 + h3 + chunk_index)."""
    raw = (file_path + (h2 or "") + (h3 or "") + str(chunk_index)).encode()
    return hashlib.sha256(raw).hexdigest()[:32]
```

- [x] **Step 4: Update the call site in `_build_point` (line 88)**

```python
# Before
        id=_chunk_id(doc["file_path"], chunk.get("h2"), chunk.get("h3")),

# After
        id=_chunk_id(
            doc["file_path"],
            chunk.get("h2"),
            chunk.get("h3"),
            chunk.get("chunk_index", 0),
        ),
```

- [x] **Step 5: Update `_expected_id` helper in `test_upserter.py`**

Find the `_expected_id` helper function in `test_upserter.py` and add `chunk_index`:

```python
# Before
def _expected_id(file_path: str, h2: str | None, h3: str | None = None) -> str:
    raw = (file_path + (h2 or "") + (h3 or "")).encode()
    return hashlib.sha256(raw).hexdigest()[:32]

# After
def _expected_id(file_path: str, h2: str | None, h3: str | None, chunk_index: int = 0) -> str:
    raw = (file_path + (h2 or "") + (h3 or "") + str(chunk_index)).encode()
    return hashlib.sha256(raw).hexdigest()[:32]
```

Also update the call inside `test_chunk_ids_are_deterministic` to pass `chunk_index`:

```python
# Before
expected = _expected_id(sample_doc["file_path"], chunk["h2"], chunk.get("h3"))

# After
expected = _expected_id(sample_doc["file_path"], chunk["h2"], chunk.get("h3"), chunk.get("chunk_index", 0))
```

- [x] **Step 6: Run all upserter tests**

```bash
cd /Users/konradciok/repositories/medusa/qdrant
uv run pytest vault_qdrant/tests/test_upserter.py -v
```

Expected: all tests `PASSED`, including both new collision tests.

- [x] **Step 7: Run the full test suite**

```bash
cd /Users/konradciok/repositories/medusa/qdrant
uv run pytest vault_qdrant/tests/ -v 2>&1 | tail -20
```

Expected: all tests pass.

- [x] **Step 8: Commit**

```bash
cd /Users/konradciok/repositories/medusa/qdrant
git add vault_qdrant/upserter.py vault_qdrant/tests/test_upserter.py
git commit -m "fix: add chunk_index to _chunk_id hash to prevent heading-collision"
```

---

## Task 3: Verify `_contextualize_chunks` type annotation in `cli.py`

**Files:**
- Check: `vault_qdrant/cli.py` (line 101)

### Background

The internal helper `_contextualize_chunks(contextualizer: Contextualizer, ...)` is only called after the `if contextualizer is not None` guard in `_sync_doc`. The annotation should be `Contextualizer`, not `Contextualizer | None`. This task confirms the annotation is already clean (no `| None` present) — if it is, no change is needed.

- [x] **Step 1: Check the annotation**

```bash
grep -n "def _contextualize_chunks" /Users/konradciok/repositories/medusa/qdrant/vault_qdrant/cli.py
```

Expected:
```
101:def _contextualize_chunks(
```

Then read lines 101–105 to confirm the signature reads `contextualizer: Contextualizer` with no `| None`.

- [x] **Step 2: If annotation is already clean — no commit needed**

Note it as resolved and run the full test suite one final time:

```bash
cd /Users/konradciok/repositories/medusa/qdrant
uv run pytest vault_qdrant/tests/ -v 2>&1 | tail -5
```

Expected: all tests pass.

- [x] **Step 3: If `| None` is present — apply fix and commit**

Change:
```python
# Before
    contextualizer: Contextualizer | None,

# After
    contextualizer: Contextualizer,
```

```bash
cd /Users/konradciok/repositories/medusa/qdrant
git add vault_qdrant/cli.py
git commit -m "chore: tighten _contextualize_chunks type annotation"
```

---

## Self-Review

**Spec coverage:**
- Prompt caching → Task 1 ✓
- Chunk-ID collision prevention → Task 2 ✓
- Type annotation verification → Task 3 ✓

**Placeholder scan:** No TBDs. All code blocks are complete and runnable.

**Type consistency:** `_chunk_id` receives 4 args everywhere — implementation, call site in `_build_point`, test helper `_expected_id`, and new collision tests. Return types unchanged throughout.
