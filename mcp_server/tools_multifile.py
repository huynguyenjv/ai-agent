"""Multi-file Atomic Edits — Phase 17.1.

Apply edits across several files as one transaction: back up originals, detect
conflicts, apply all, and roll back every file if any step fails. Supports
dry-run (preview unified diffs without writing).

Enhancement over tools.apply_edits: in-memory backup + guaranteed rollback +
conflict detection.
"""

from __future__ import annotations

import difflib
import logging
import os

logger = logging.getLogger("mcp_server.tools_multifile")


def _within_repo(repo_path: str, file_path: str) -> bool:
    real_repo = os.path.realpath(repo_path)
    real_file = os.path.realpath(os.path.join(repo_path, file_path))
    return real_file == real_repo or real_file.startswith(real_repo + os.sep)


def _read(repo_path: str, file_path: str) -> str | None:
    abs_path = os.path.join(repo_path, file_path)
    if not os.path.isfile(abs_path):
        return None
    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _compute_new_content(original: str | None, edit: dict) -> tuple[str | None, str | None]:
    """Return (new_content, conflict_reason). conflict_reason set → abort."""
    if "new_content" in edit and edit["new_content"] is not None:
        return edit["new_content"], None

    search = edit.get("search")
    replace = edit.get("replace", "")
    if search is None:
        return None, "edit needs 'new_content' or 'search'(+replace)"
    if original is None:
        return None, "file not found for search/replace"
    if search not in original:
        return None, f"search text not found: {search[:60]!r}"
    return original.replace(search, replace), None


def apply_multi_file_edits(repo_path: str, edits: list[dict], dry_run: bool = False) -> dict:
    """Atomically apply a list of file edits with rollback on failure."""
    if not edits:
        return {"success": True, "applied": 0, "files": []}

    # Validate paths + compute target contents first (pre-flight, no writes)
    planned: list[tuple[str, str | None, str]] = []  # (file_path, original, new_content)
    for edit in edits:
        file_path = edit.get("file_path")
        if not file_path:
            return {"success": False, "error": "edit missing file_path", "rolled_back": False}
        if not _within_repo(repo_path, file_path):
            return {"success": False, "error": f"path outside repo: {file_path}", "rolled_back": False}

        original = _read(repo_path, file_path)
        new_content, conflict = _compute_new_content(original, edit)
        if conflict:
            return {"success": False, "error": f"conflict in {file_path}: {conflict}",
                    "rolled_back": False, "conflict": True}
        planned.append((file_path, original, new_content))

    if dry_run:
        diffs = []
        for file_path, original, new_content in planned:
            diff = "".join(difflib.unified_diff(
                (original or "").splitlines(keepends=True),
                (new_content or "").splitlines(keepends=True),
                fromfile=f"a/{file_path}", tofile=f"b/{file_path}",
            ))
            diffs.append({"file_path": file_path, "diff": diff})
        return {"success": True, "dry_run": True, "files": [p[0] for p in planned], "diffs": diffs}

    # Apply with rollback
    written: list[tuple[str, str | None]] = []  # (file_path, original) for rollback
    try:
        for file_path, original, new_content in planned:
            abs_path = os.path.join(repo_path, file_path)
            os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            written.append((file_path, original))

        # Verify
        for file_path, _, new_content in planned:
            if _read(repo_path, file_path) != new_content:
                raise IOError(f"verification failed for {file_path}")

        return {"success": True, "applied": len(planned), "files": [p[0] for p in planned]}

    except Exception as e:
        # Roll back everything we wrote
        for file_path, original in written:
            abs_path = os.path.join(repo_path, file_path)
            try:
                if original is None:
                    if os.path.isfile(abs_path):
                        os.remove(abs_path)
                else:
                    with open(abs_path, "w", encoding="utf-8") as f:
                        f.write(original)
            except OSError:
                logger.error("rollback failed for %s", file_path)
        logger.warning("apply_multi_file_edits failed, rolled back: %s", e)
        return {"success": False, "error": str(e), "rolled_back": True}
