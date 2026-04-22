"""VaultScanner — walk an Obsidian vault and return parsed document records."""
import hashlib
import re
from pathlib import Path

from yaml import safe_load

# Directory names whose files are excluded from scanning
_EXCLUDED_DIRS = frozenset({".obsidian", "templates", "attachments"})

# Regex to detect and extract YAML frontmatter at the very start of a file
_FRONTMATTER_RE = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)


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
        file_path (str)   — relative path from vault_path
        content   (str)   — full raw file content
        tags      (list)  — from frontmatter, default []
        type      (str|None)
        created   (str|None)
        status    (str|None)
        projects  (list)  — from frontmatter, default []
        doc_hash  (str)   — SHA-256 of content
    """
    root = Path(vault_path)
    results: list[dict] = []

    for md_file in root.rglob("*.md"):
        if _is_excluded(md_file, root):
            continue

        content = md_file.read_text(encoding="utf-8")
        frontmatter = _parse_frontmatter(content)

        results.append(
            {
                "file_path": str(md_file.relative_to(root)),
                "content": content,
                "doc_hash": _hash(content),
                **frontmatter,
            }
        )

    return results
