"""Phase 3 tools: get_project_skeleton and index_with_deps.

Section 5 of the implementation plan.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import defaultdict
from pathlib import Path

from mcp_server.dep_classifier import DepClassifier, DepType
from mcp_server.hash_store import HashStore
from mcp_server.models import CodeChunk, ExtractionMode, SKIP_DIRS, SKIP_EXTENSIONS
from mcp_server.plugins.registry import PluginRegistry
from mcp_server.token_budget import TokenBudget
from mcp_server.uploader import Uploader

logger = logging.getLogger("mcp_server.tools")


def get_project_skeleton(
    repo_path: str,
    registry: PluginRegistry,
    include_methods: bool = True,
) -> dict:
    """Section 5, Tool: get_project_skeleton.

    Return a compact structural overview of the entire repository.
    Read-only local operation. Does NOT upload to server.
    Output must fit within ~2000 tokens.
    """
    packages: dict[str, list[dict]] = defaultdict(list)
    total_classes = 0
    total_packages = set()

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for fname in files:
            ext = Path(fname).suffix.lower()
            if ext in SKIP_EXTENSIONS:
                continue

            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, repo_path).replace("\\", "/")

            plugin = registry.get_plugin(full_path)

            try:
                with open(full_path, "rb") as f:
                    source = f.read()
            except OSError:
                continue

            try:
                skeleton = plugin.extract_skeleton(rel_path, source, include_methods)
            except Exception as e:
                logger.debug("Failed to extract skeleton from %s: %s", rel_path, e)
                continue

            # Derive package path from relative location
            pkg_path = str(Path(rel_path).parent)
            if pkg_path == ".":
                pkg_path = "(root)"

            total_packages.add(pkg_path)

            for cls in skeleton.get("classes", []):
                total_classes += 1
                packages[pkg_path].append(cls)

            if include_methods:
                for func in skeleton.get("functions", []):
                    packages[pkg_path].append(func)

    return {
        "packages": dict(packages),
        "stats": {
            "total_classes": total_classes,
            "total_packages": len(total_packages),
            "hint": "Use index_with_deps to index specific files, or read_file to view content.",
        },
    }


async def index_with_deps(
    repo_path: str,
    registry: PluginRegistry,
    hash_store: HashStore,
    uploader: Uploader,
    dep_classifier: DepClassifier,
    file_path: str,
    depth: int = 2,
    token_budget: int = 8000,
) -> dict:
    """Section 5, Tool: index_with_deps.

    Parse a file and its project-local deps up to given depth,
    then upload changed chunks to the server.

    BFS traversal with hash checking and fail-safe hash store writes.
    """
    depth = min(depth, 3)  # Maximum 3
    budget = TokenBudget(token_budget)

    # BFS state
    queue: list[tuple[str, int]] = [(file_path, 0)]  # (rel_path, depth)
    visited: set[str] = set()
    all_chunks: list[CodeChunk] = []
    files_processed = 0
    files_skipped = 0
    file_hashes: list[tuple[str, str]] = []  # (path, hash) for batch update

    while queue:
        rel_path, current_depth = queue.pop(0)

        if current_depth > depth:
            continue

        abs_path = os.path.join(repo_path, rel_path)
        real_path = os.path.realpath(abs_path)

        if real_path in visited:
            continue
        visited.add(real_path)

        if not os.path.isfile(abs_path):
            continue

        files_processed += 1

        # Read file and compute hash
        try:
            with open(abs_path, "rb") as f:
                source_bytes = f.read()
        except OSError:
            continue

        current_hash = hashlib.md5(source_bytes).hexdigest()

        # Check hash store — skip if unchanged
        stored_hash = hash_store.get_hash(rel_path)
        if stored_hash == current_hash:
            files_skipped += 1
            continue

        # Detect plugin
        plugin = registry.get_plugin(abs_path)

        # Get extraction mode based on depth
        mode = budget.get_mode(current_depth)

        # Extract chunks
        try:
            chunks = plugin.extract_chunks(rel_path, source_bytes, mode)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", rel_path, e)
            # Fall back to FallbackPlugin per Section 14
            fallback = registry.fallback
            try:
                chunks = fallback.extract_chunks(rel_path, source_bytes, mode)
            except Exception:
                continue

        # Apply token budget — truncate if over budget
        filtered_chunks: list[CodeChunk] = []
        for chunk in chunks:
            text = chunk.embed_text + chunk.body
            if budget.can_add(current_depth, text):
                budget.add(current_depth, text)
                filtered_chunks.append(chunk)

        all_chunks.extend(filtered_chunks)
        file_hashes.append((rel_path, current_hash))

        # Resolve deps and add to BFS queue
        if current_depth < depth:
            for chunk in chunks:
                for raw_import in chunk.raw_imports:
                    dep_type, resolved = dep_classifier.classify(
                        raw_import,
                        chunk.lang,
                        plugin,
                        rel_path,
                        repo_path,
                    )
                    if dep_type == DepType.project and resolved is not None:
                        dep_rel = str(resolved).replace("\\", "/")
                        queue.append((dep_rel, current_depth + 1))

    # If no chunks to upload
    if not all_chunks:
        return {
            "indexed": 0,
            "skipped": files_skipped,
            "files_processed": files_processed,
        }

    # Upload to server
    result = await uploader.upload(all_chunks)

    # Only update hash store after successful upload (Section 5, P6)
    if "error" not in result:
        hash_store.set_hashes_batch(file_hashes)

    return {
        "indexed": len(all_chunks),
        "skipped": files_skipped,
        "files_processed": files_processed,
        **({"error": result["error"]} if "error" in result else {}),
    }
