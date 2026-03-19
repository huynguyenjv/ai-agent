"""
MCP Server for Java Codebase Indexing.

Provides tools for Continue.dev (or any MCP client) to index Java source
files into Qdrant via the AI Agent's /v1/index-file endpoint.

Usage (stdio):
    python scripts/mcp_index_server.py

Continue.dev config (config.yaml):
    mcpServers:
      - name: java-indexer
        command: python
        args:
          - C:/path/to/ai-agent/scripts/mcp_index_server.py
        env:
          AI_AGENT_URL: http://localhost:8080
"""

import os
import json
import glob
import urllib.request
import urllib.error
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ── Configuration ──────────────────────────────────────────────────
AI_AGENT_URL = os.getenv("AI_AGENT_URL", "http://localhost:8080")
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "java_codebase")

mcp = FastMCP(
    "java-indexer",
    instructions="Index Java source files into Qdrant via AI Agent server",
)


# ── Helpers ────────────────────────────────────────────────────────

def _post_json(path: str, payload: dict) -> dict:
    """Send a JSON POST request to the AI Agent server."""
    url = f"{AI_AGENT_URL}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        return {"error": f"HTTP {e.code}: {body}"}
    except urllib.error.URLError as e:
        return {"error": f"Connection failed: {e.reason}"}


# ── Tools ──────────────────────────────────────────────────────────

@mcp.tool()
def index_file(file_path: str, collection: str = "") -> str:
    """Index a single Java file into Qdrant.

    Reads the file from disk, sends its content to the AI Agent server
    for Java-aware parsing (dependency tracking, Lombok detection, etc.)
    and vector indexing.

    Args:
        file_path: Absolute path to the .java file.
        collection: Qdrant collection name (optional, uses default if empty).
    """
    path = Path(file_path)
    if not path.exists():
        return f"❌ File not found: {file_path}"
    if not path.suffix == ".java":
        return f"❌ Not a Java file: {file_path}"

    content = path.read_text(encoding="utf-8")

    payload = {
        "file_path": str(path),
        "content": content,
    }
    if collection:
        payload["collection"] = collection

    result = _post_json("/v1/index-file", payload)

    if result.get("error"):
        return f"❌ Indexing failed: {result['error']}"
    if result.get("success"):
        return (
            f"✅ Indexed `{path.name}`\n"
            f"   Collection: {result.get('collection', DEFAULT_COLLECTION)}\n"
            f"   Classes: {result.get('classes_indexed', 0)}\n"
            f"   Points: {result.get('points_created', 0)}"
        )
    return f"⚠️ Unexpected response: {json.dumps(result, indent=2)}"


@mcp.tool()
def index_directory(directory_path: str, collection: str = "") -> str:
    """Index all Java files in a directory (recursive) into Qdrant.

    Walks the directory, finds all .java files, and indexes each one
    via the AI Agent server.

    Args:
        directory_path: Absolute path to the directory containing Java files.
        collection: Qdrant collection name (optional, uses default if empty).
    """
    dirpath = Path(directory_path)
    if not dirpath.is_dir():
        return f"❌ Directory not found: {directory_path}"

    java_files = sorted(dirpath.rglob("*.java"))
    if not java_files:
        return f"⚠️ No .java files found in {directory_path}"

    results = []
    success_count = 0
    fail_count = 0
    total_points = 0

    for jf in java_files:
        content = jf.read_text(encoding="utf-8")
        payload = {"file_path": str(jf), "content": content}
        if collection:
            payload["collection"] = collection

        resp = _post_json("/v1/index-file", payload)

        if resp.get("success"):
            success_count += 1
            total_points += resp.get("points_created", 0)
        else:
            fail_count += 1
            results.append(f"  ❌ {jf.name}: {resp.get('error', 'unknown')}")

    summary = (
        f"📊 Indexed {success_count}/{len(java_files)} files\n"
        f"   Total points: {total_points}\n"
        f"   Collection: {collection or DEFAULT_COLLECTION}"
    )
    if fail_count:
        summary += f"\n   Failures ({fail_count}):\n" + "\n".join(results)

    return summary


@mcp.tool()
def index_current_file(file_path: str) -> str:
    """Quick-index the currently open Java file.

    Designed for use with Continue.dev — pass the current file path
    to index it immediately.

    Args:
        file_path: Path to the currently open .java file.
    """
    return index_file(file_path)


# ── Resources ──────────────────────────────────────────────────────

@mcp.resource("indexer://status")
def get_indexer_status() -> str:
    """Check AI Agent server health and index stats."""
    try:
        url = f"{AI_AGENT_URL}/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Server unreachable: {e}"


# ── Entrypoint ─────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
