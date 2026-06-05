"""Trace → Dataset — Phase 16.3.

Collects production traces (with user feedback) into a versioned JSONL dataset
for later fine-tuning. Secrets/PII are redacted before anything is written.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("eval.trace_collector")

DEFAULT_DIR = os.environ.get("DATASET_DIR", "data/datasets")


def _redact(text: str) -> str:
    if not text:
        return text or ""
    try:
        from server.utils.secret_scanner import redact

        return redact(text)[0]
    except Exception:
        return text


@dataclass
class TraceExample:
    messages: list[dict]
    response: str
    label: str                       # "positive" | "negative"
    intent: str = ""
    metadata: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "messages": [
                {"role": m.get("role", ""), "content": _redact(str(m.get("content", "")))}
                for m in self.messages
            ],
            "response": _redact(self.response),
            "label": self.label,
            "intent": self.intent,
            "metadata": self.metadata,
        }


class TraceCollector:
    """Accumulates labeled examples and exports a versioned JSONL dataset."""

    def __init__(self, output_dir: str = DEFAULT_DIR):
        self._dir = output_dir
        self._examples: list[TraceExample] = []

    def add(self, messages: list[dict], response: str, label: str,
            intent: str = "", metadata: dict | None = None) -> None:
        if label not in ("positive", "negative"):
            raise ValueError("label must be 'positive' or 'negative'")
        self._examples.append(
            TraceExample(messages, response, label, intent, metadata or {})
        )

    def add_positive(self, messages, response, **kw) -> None:
        self.add(messages, response, "positive", **kw)

    def add_negative(self, messages, response, **kw) -> None:
        self.add(messages, response, "negative", **kw)

    def __len__(self) -> int:
        return len(self._examples)

    def stats(self) -> dict:
        pos = sum(1 for e in self._examples if e.label == "positive")
        return {"total": len(self._examples), "positive": pos,
                "negative": len(self._examples) - pos}

    def export_jsonl(self, version: str | None = None) -> str:
        """Write all examples to a versioned JSONL file. Returns the path."""
        os.makedirs(self._dir, exist_ok=True)
        version = version or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self._dir, f"dataset-{version}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for ex in self._examples:
                f.write(json.dumps(ex.to_record(), ensure_ascii=False) + "\n")
        logger.info("Exported %d examples → %s", len(self._examples), path)
        return path
