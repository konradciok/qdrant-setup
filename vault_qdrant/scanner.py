"""VaultScanner — walk an Obsidian vault and return parsed document records."""
import hashlib
import re
from pathlib import Path

from yaml import safe_load

# Directory names whose files are excluded from scanning
_EXCLUDED_DIRS = frozenset({".obsidian", "templates", "attachments"})

# Regex to detect and extract YAML frontmatter at the very start of a file
_FRONTMATTER_RE = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)

# Ordered rules mapping folder path patterns to doc_type values
_FOLDER_TYPE_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^sessions/"), "session"),
    (re.compile(r"^todos/"), "todos"),
    (re.compile(r".*/architecture/\d{4}-.+"), "adr"),
    (re.compile(r".*/specs/"), "spec"),
    (re.compile(r"^projects/"), "project"),
    (re.compile(r"^components/"), "component"),
]

_INLINE_TAG_RE = re.compile(r"(?<![`\w])#([a-zA-Z][a-zA-Z0-9_/-]+)")
_CODE_FENCE_RE = re.compile(r"^```", re.MULTILINE)


def _infer_doc_type(file_path: str) -> str:
    """Infer doc_type from folder path using ordered rules; default is 'note'."""
    for pattern, doc_type in _FOLDER_TYPE_RULES:
        if pattern.search(file_path):
            return doc_type
    return "note"


def _extract_inline_tags(content: str) -> list[str]:
    """Extract #hashtag patterns from content, skipping fenced code blocks."""
    fences = list(_CODE_FENCE_RE.finditer(content))
    blocked: list[tuple[int, int]] = []
    for i in range(0, len(fences) - 1, 2):
        blocked.append((fences[i].start(), fences[i + 1].end()))
    if len(fences) % 2 != 0:
        blocked.append((fences[-1].start(), len(content)))

    tags: list[str] = []
    for m in _INLINE_TAG_RE.finditer(content):
        pos = m.start()
        if any(start <= pos <= end for start, end in blocked):
            continue
        tags.append(m.group(1))
    return tags


def _is_excluded(path: Path, vault_root: Path) -> bool:
    """Return True if any path component matches an excluded directory name."""
    relative = path.relative_to(vault_root)
    return bool(_EXCLUDED_DIRS.intersection(relative.parts[:-1]))


def _parse_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter fields; return defaults when absent."""
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {"tags": [], "type": None, "created": None, "status": None, "projects": []}

    raw = safe_load(match.group(1)) or {}
    return {
        "tags": _as_list(raw.get("tags")),
        "type": raw.get("type"),
        "created": _as_str(raw.get("created")),
        "status": raw.get("status"),
        "projects": _as_list(raw.get("projects")),
    }


def _as_list(value) -> list:
    """Coerce a value to a list; None becomes []."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_str(value) -> str | None:
    """Coerce a value to str, preserving None."""
    if value is None:
        return None
    return str(value)


def _hash(content: str) -> str:
    """Return the SHA-256 hex digest of *content*."""
    return hashlib.sha256(content.encode()).hexdigest()


def scan(vault_path: str | Path) -> list[dict]:
    """Walk *vault_path* and return one record per eligible markdown file.

    Each record contains:
        file_path   (str)              — relative path from vault_path
        content     (str)              — full raw file content
        tags        (list)             — frontmatter tags merged with inline #tags
        type        (str)              — from frontmatter or inferred from folder path
        type_source (str)              — "frontmatter" or "inferred"
        created     (str|None)
        status      (str|None)
        projects    (list)             — from frontmatter, default []
        doc_hash    (str)              — SHA-256 of content
    """
    root = Path(vault_path)
    results: list[dict] = []

    for md_file in root.rglob("*.md"):
        if _is_excluded(md_file, root):
            continue

        content = md_file.read_text(encoding="utf-8")
        frontmatter = _parse_frontmatter(content)
        file_path = md_file.relative_to(root).as_posix()

        if frontmatter["type"] is not None:
            doc_type = frontmatter["type"]
            type_source = "frontmatter"
        else:
            doc_type = _infer_doc_type(file_path)
            type_source = "inferred"

        inline_tags = _extract_inline_tags(content)
        all_tags = list(dict.fromkeys(frontmatter["tags"] + inline_tags))

        results.append(
            {
                "file_path": file_path,
                "content": content,
                "doc_hash": _hash(content),
                "tags": all_tags,
                "type": doc_type,
                "type_source": type_source,
                "created": frontmatter["created"],
                "status": frontmatter["status"],
                "projects": frontmatter["projects"],
            }
        )

    return results
