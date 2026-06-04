"""Secret Scanning — Phase 10.3.

Detects and redacts secrets (API keys, tokens, private keys, credentials) in
text — primarily tool outputs (e.g. an LLM reading a .env file) before they
enter the model context or get returned to the user.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

logger = logging.getLogger("server.secret_scanner")

REDACTED = "[REDACTED]"

# (name, compiled regex). Group 0 is redacted unless the pattern captures the
# secret in group 1 (used to keep surrounding context like the key name).
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("aws_access_key_id", re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("private_key_block", re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"
    )),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    ("bearer_token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{20,}")),
    ("url_basic_auth", re.compile(r"\b[a-z][a-z0-9+.\-]*://[^\s:@/]+:[^\s:@/]+@")),
    # key=value style secrets in config/.env
    # No \b around the keyword so prefixed identifiers (DB_PASSWORD, MY_API_KEY)
    # are also caught.
    ("assigned_secret", re.compile(
        r"(?i)(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|private[_-]?key)\s*[:=]\s*['\"]?([^\s'\"]{6,})['\"]?"
    )),
]

# Entropy detection for long opaque strings
_ENTROPY_TOKEN = re.compile(r"\b[A-Za-z0-9+/=_\-]{24,}\b")
_ENTROPY_THRESHOLD = 4.0


@dataclass
class SecretFinding:
    kind: str
    preview: str  # masked preview, never the raw secret


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _mask(secret: str) -> str:
    secret = secret.strip()
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:3]}…{secret[-2:]} ({len(secret)} chars)"


def scan(text: str | None) -> list[SecretFinding]:
    """Return secret findings without modifying the text."""
    if not text:
        return []

    findings: list[SecretFinding] = []
    spans: list[tuple[int, int]] = []

    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            secret = m.group(1) if m.groups() else m.group(0)
            findings.append(SecretFinding(kind=kind, preview=_mask(secret)))
            spans.append(m.span(1) if m.groups() else m.span(0))

    # Entropy-based catch-all for high-entropy tokens not already matched
    for m in _ENTROPY_TOKEN.finditer(text):
        if any(s <= m.start() < e or s < m.end() <= e for s, e in spans):
            continue
        token = m.group(0)
        if _shannon_entropy(token) >= _ENTROPY_THRESHOLD:
            findings.append(SecretFinding(kind="high_entropy", preview=_mask(token)))

    return findings


def redact(text: str | None) -> tuple[str, list[SecretFinding]]:
    """Return (redacted_text, findings). Replaces detected secrets with [REDACTED]."""
    if not text:
        return text or "", []

    findings: list[SecretFinding] = []

    def _replace(value: str) -> str:
        return value if not value else REDACTED

    out = text
    for kind, pattern in _PATTERNS:
        def _sub(m: re.Match) -> str:
            secret = m.group(1) if m.groups() else m.group(0)
            findings.append(SecretFinding(kind=kind, preview=_mask(secret)))
            if m.groups():
                # keep everything except the captured secret
                start, end = m.span(1)
                return m.group(0)[: start - m.start()] + REDACTED + m.group(0)[end - m.start():]
            return REDACTED
        out = pattern.sub(_sub, out)

    # Entropy pass
    def _entropy_sub(m: re.Match) -> str:
        token = m.group(0)
        if _shannon_entropy(token) >= _ENTROPY_THRESHOLD:
            findings.append(SecretFinding(kind="high_entropy", preview=_mask(token)))
            return REDACTED
        return token

    out = _ENTROPY_TOKEN.sub(_entropy_sub, out)

    if findings:
        logger.warning("secret_scanner: redacted %d secret(s): %s",
                       len(findings), [f.kind for f in findings])
    return out, findings


def has_secrets(text: str | None) -> bool:
    return bool(scan(text))
