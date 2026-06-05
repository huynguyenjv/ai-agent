#!/usr/bin/env python
"""AI Agent CLI — Simple client for testing the API."""

import argparse
import json
import os
import sys

import httpx

DEFAULT_URL = "http://localhost:9090"
DEFAULT_API_KEY = os.environ.get("API_KEY", "dev-secret-key")


def cmd_health(args):
    """Check server health (use --deep for dependency probes)."""
    path = "/health/deep" if getattr(args, "deep", False) else "/health"
    url = f"{args.url}{path}"
    try:
        resp = httpx.get(url, timeout=10)
        print(json.dumps(resp.json(), indent=2))
        return 0 if resp.status_code == 200 else 1
    except httpx.ConnectError:
        print(f"Error: Cannot connect to {url}")
        return 1


def cmd_config(args):
    """Validate local CLI config + server reachability (Phase 20.4)."""
    checks = []
    api_key = args.api_key
    checks.append(("API key set", bool(api_key and api_key != "dev-secret-key")))
    checks.append(("Server URL", bool(args.url)))

    reachable = False
    try:
        resp = httpx.get(f"{args.url}/health", timeout=5)
        reachable = resp.status_code == 200
    except httpx.HTTPError:
        reachable = False
    checks.append(("Server reachable", reachable))

    print("Config validation:")
    ok = True
    for name, passed in checks:
        print(f"  [{'OK' if passed else 'WARN'}] {name}")
        ok = ok and passed
    print(f"\nResolved: url={args.url} api_key={'***' if api_key else '(none)'}")
    return 0 if ok else 1


def cmd_chat(args):
    """Send chat request."""
    url = f"{args.url}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": args.api_key,
    }
    message = args.message
    if getattr(args, "agents", False):
        message = f"/agents {message}"  # opt-in multi-agent (see docs/multi-agent-design.md)
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": message}],
        "stream": True,
    }

    try:
        with httpx.stream("POST", url, json=payload, headers=headers, timeout=120) as resp:
            if resp.status_code != 200:
                print(f"Error {resp.status_code}: {resp.text}")
                return 1

            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        print()
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        pass
        return 0
    except httpx.ConnectError:
        print(f"Error: Cannot connect to {url}")
        return 1


def cmd_review(args):
    """Send code review request."""
    url = f"{args.url}/review/analyze"
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": args.api_key,
    }

    # Read diff from file or stdin
    if args.diff_file:
        with open(args.diff_file) as f:
            diff = f.read()
    else:
        print("Enter diff (Ctrl+D to end):")
        diff = sys.stdin.read()

    payload = {
        "diff": diff,
        "repo_path": args.repo or os.getcwd(),
    }

    try:
        with httpx.stream("POST", url, json=payload, headers=headers, timeout=300) as resp:
            if resp.status_code != 200:
                print(f"Error {resp.status_code}: {resp.text}")
                return 1

            for line in resp.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        content = chunk.get("content", "")
                        if content:
                            print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        pass
            print()
        return 0
    except httpx.ConnectError:
        print(f"Error: Cannot connect to {url}")
        return 1


def cmd_index(args):
    """Index a repository."""
    url = f"{args.url}/index"
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": args.api_key,
    }
    payload = {
        "repo_path": args.repo or os.getcwd(),
    }

    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=300)
        print(json.dumps(resp.json(), indent=2))
        return 0 if resp.status_code == 200 else 1
    except httpx.ConnectError:
        print(f"Error: Cannot connect to {url}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="AI Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py health
  python cli.py chat "Write a hello world in Python"
  python cli.py chat -m "Explain this code"
  python cli.py review --diff-file changes.diff
  python cli.py index --repo /path/to/repo
        """,
    )
    parser.add_argument(
        "--url", "-u",
        default=DEFAULT_URL,
        help=f"Server URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--api-key", "-k",
        default=DEFAULT_API_KEY,
        help="API key for authentication",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # health
    health_parser = subparsers.add_parser("health", help="Check server health")
    health_parser.add_argument("--deep", action="store_true", help="Deep dependency probe")

    # config
    subparsers.add_parser("config", help="Validate CLI config + server reachability")

    # chat
    chat_parser = subparsers.add_parser("chat", help="Send chat message")
    chat_parser.add_argument(
        "message", nargs="?",
        help="Message to send",
    )
    chat_parser.add_argument(
        "-m", "--message",
        dest="message_flag",
        help="Message to send (alternative)",
    )
    chat_parser.add_argument(
        "--agents", action="store_true",
        help="Run multi-agent workflow for this request (/agents)",
    )

    # review
    review_parser = subparsers.add_parser("review", help="Code review")
    review_parser.add_argument(
        "--diff-file", "-f",
        help="File containing diff (or stdin)",
    )
    review_parser.add_argument(
        "--repo", "-r",
        help="Repository path",
    )

    # index
    index_parser = subparsers.add_parser("index", help="Index repository")
    index_parser.add_argument(
        "--repo", "-r",
        help="Repository path to index",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "health":
        return cmd_health(args)
    elif args.command == "config":
        return cmd_config(args)
    elif args.command == "chat":
        msg = args.message or getattr(args, "message_flag", None)
        if not msg:
            print("Error: Message required")
            return 1
        args.message = msg
        return cmd_chat(args)
    elif args.command == "review":
        return cmd_review(args)
    elif args.command == "index":
        return cmd_index(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
