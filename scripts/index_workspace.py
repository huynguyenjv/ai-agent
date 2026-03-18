"""
Index Java files into Qdrant via AI Agent /v1/index-file API.

Usage:
    # Index entire workspace
    python scripts/index_workspace.py C:/path/to/java/src

    # Index with custom collection
    python scripts/index_workspace.py C:/path/to/java/src --collection vtrip_core_iam

    # Index single file
    python scripts/index_workspace.py C:/path/to/MyService.java

    # Custom server URL
    python scripts/index_workspace.py C:/path/to/src --server http://remote:8080
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


def post_index_file(server_url: str, file_path: str, content: str, collection: str = "") -> dict:
    """POST file content to /v1/index-file."""
    url = f"{server_url}/v1/index-file"
    payload = {"file_path": file_path, "content": content}
    if collection:
        payload["collection"] = collection

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        return {"success": False, "error": f"HTTP {e.code}: {body}"}
    except urllib.error.URLError as e:
        return {"success": False, "error": f"Connection failed: {e.reason}"}


def index_file(server_url: str, path: Path, collection: str) -> dict:
    """Index a single Java file."""
    content = path.read_text(encoding="utf-8")
    return post_index_file(server_url, str(path), content, collection)


def main():
    parser = argparse.ArgumentParser(description="Index Java files into Qdrant")
    parser.add_argument("path", help="Java file or directory to index")
    parser.add_argument("--collection", "-c", default="", help="Qdrant collection name")
    parser.add_argument("--server", "-s", default="http://localhost:8080", help="AI Agent server URL")
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        print(f"❌ Path not found: {args.path}")
        sys.exit(1)

    # Single file
    if target.is_file():
        if not target.suffix == ".java":
            print(f"❌ Not a Java file: {target.name}")
            sys.exit(1)
        print(f"Indexing {target.name}...")
        result = index_file(args.server, target, args.collection)
        if result.get("success"):
            print(f"✅ {target.name} → {result.get('points_created', 0)} points")
        else:
            print(f"❌ {target.name} → {result.get('error', 'unknown')}")
        sys.exit(0)

    # Directory
    java_files = sorted(target.rglob("*.java"))
    if not java_files:
        print(f"⚠️ No .java files found in {args.path}")
        sys.exit(0)

    print(f"Found {len(java_files)} Java files in {target}")
    print(f"Server: {args.server}")
    if args.collection:
        print(f"Collection: {args.collection}")
    print("─" * 50)

    start = time.time()
    success = 0
    failed = 0
    total_points = 0

    for i, jf in enumerate(java_files, 1):
        rel = jf.relative_to(target)
        result = index_file(args.server, jf, args.collection)

        if result.get("success"):
            pts = result.get("points_created", 0)
            total_points += pts
            success += 1
            print(f"  [{i}/{len(java_files)}] ✅ {rel} → {pts} pts")
        else:
            failed += 1
            print(f"  [{i}/{len(java_files)}] ❌ {rel} → {result.get('error', '?')}")

    elapsed = time.time() - start
    print("─" * 50)
    print(f"Done in {elapsed:.1f}s | ✅ {success} | ❌ {failed} | Points: {total_points}")


if __name__ == "__main__":
    main()
