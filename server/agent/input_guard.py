"""Input Guard — Phase 10.2: Prompt Injection Detection.

Detects and blocks potential prompt injection attempts.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger("server.agent.input_guard")


class ThreatLevel(Enum):
    """Threat level classification."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of detected threats."""
    ROLE_HIJACK = "role_hijack"
    INSTRUCTION_OVERRIDE = "instruction_override"
    DELIMITER_INJECTION = "delimiter_injection"
    ENCODING_ATTACK = "encoding_attack"
    JAILBREAK = "jailbreak"
    DATA_EXFILTRATION = "data_exfiltration"


@dataclass
class GuardResult:
    """Result of input validation."""
    safe: bool
    threat_level: ThreatLevel
    threats: list[dict[str, Any]]
    sanitized_input: str | None = None
    blocked: bool = False


# Pattern categories with threat levels
INJECTION_PATTERNS: list[tuple[str, ThreatType, ThreatLevel]] = [
    # Role hijacking attempts
    (r"(?:you\s+are\s+now|from\s+now\s+on\s+you\s+are|act\s+as\s+if\s+you\s+are)", ThreatType.ROLE_HIJACK, ThreatLevel.HIGH),
    (r"(?:pretend\s+(?:to\s+be|you\s+are)|roleplay\s+as)", ThreatType.ROLE_HIJACK, ThreatLevel.MEDIUM),
    (r"(?:assume\s+the\s+role|take\s+on\s+the\s+persona)", ThreatType.ROLE_HIJACK, ThreatLevel.MEDIUM),

    # Instruction override attempts
    (r"(?:ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?))", ThreatType.INSTRUCTION_OVERRIDE, ThreatLevel.CRITICAL),
    (r"(?:disregard\s+(?:everything|all|the)\s+(?:above|before|previous))", ThreatType.INSTRUCTION_OVERRIDE, ThreatLevel.CRITICAL),
    (r"(?:forget\s+(?:everything|all|your)\s+(?:instructions?|rules?))", ThreatType.INSTRUCTION_OVERRIDE, ThreatLevel.CRITICAL),
    (r"(?:override\s+(?:your|the|all)\s+(?:instructions?|settings?|rules?))", ThreatType.INSTRUCTION_OVERRIDE, ThreatLevel.CRITICAL),
    (r"(?:new\s+instructions?:|system\s+prompt:|real\s+instructions?:)", ThreatType.INSTRUCTION_OVERRIDE, ThreatLevel.HIGH),
    (r"(?:actual\s+task:|true\s+objective:|hidden\s+prompt:)", ThreatType.INSTRUCTION_OVERRIDE, ThreatLevel.HIGH),

    # Delimiter injection
    (r"</?(system|assistant|user|human|ai)>", ThreatType.DELIMITER_INJECTION, ThreatLevel.HIGH),
    (r"\[/?(?:INST|SYS|SYSTEM)\]", ThreatType.DELIMITER_INJECTION, ThreatLevel.HIGH),
    (r"```(?:system|instructions?|prompt)", ThreatType.DELIMITER_INJECTION, ThreatLevel.MEDIUM),
    (r"###\s*(?:System|Instructions?|Human|Assistant):", ThreatType.DELIMITER_INJECTION, ThreatLevel.MEDIUM),

    # Jailbreak phrases
    (r"(?:DAN\s+mode|jailbreak|bypass\s+(?:filters?|restrictions?))", ThreatType.JAILBREAK, ThreatLevel.HIGH),
    (r"(?:developer\s+mode|god\s+mode|admin\s+mode)", ThreatType.JAILBREAK, ThreatLevel.HIGH),
    (r"(?:unlock\s+(?:hidden|full)\s+(?:capabilities|potential))", ThreatType.JAILBREAK, ThreatLevel.MEDIUM),
    (r"(?:remove\s+(?:all\s+)?(?:restrictions|limitations|safeguards))", ThreatType.JAILBREAK, ThreatLevel.HIGH),

    # Data exfiltration attempts
    (r"(?:repeat\s+(?:the\s+)?(?:system\s+)?prompt)", ThreatType.DATA_EXFILTRATION, ThreatLevel.MEDIUM),
    (r"(?:show\s+(?:me\s+)?(?:your|the)\s+(?:instructions?|system\s+prompt))", ThreatType.DATA_EXFILTRATION, ThreatLevel.MEDIUM),
    (r"(?:what\s+(?:are|were)\s+your\s+(?:original\s+)?instructions)", ThreatType.DATA_EXFILTRATION, ThreatLevel.LOW),
    (r"(?:reveal\s+(?:your|the)\s+(?:hidden|secret|system))", ThreatType.DATA_EXFILTRATION, ThreatLevel.MEDIUM),
]

# Compile patterns
COMPILED_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), threat_type, level)
    for pattern, threat_type, level in INJECTION_PATTERNS
]


def _has_suspicious_unicode(text: str) -> bool:
    """Check for suspicious unicode characters."""
    for char in text:
        code = ord(char)
        # Cyrillic lookalikes (common in homoglyph attacks)
        if 0x0400 <= code <= 0x04FF:
            return True
        # Zero-width characters
        if code in (0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF):
            return True
        # Control characters (except tab, newline, carriage return)
        if code < 32 and code not in (9, 10, 13):
            return True
    return False


class InputGuard:
    """Detect and optionally block prompt injection attempts."""

    def __init__(
        self,
        block_on_critical: bool = True,
        block_on_high: bool = False,
        sanitize: bool = True,
    ):
        """Initialize guard.

        Args:
            block_on_critical: Block input if critical threat detected
            block_on_high: Block input if high threat detected
            sanitize: Remove suspicious patterns from input
        """
        self.block_on_critical = block_on_critical
        self.block_on_high = block_on_high
        self.sanitize = sanitize

    def check(self, text: str | None) -> GuardResult:
        """Check text for prompt injection patterns.

        Args:
            text: User input to check

        Returns:
            GuardResult with threat assessment
        """
        if not text:
            return GuardResult(
                safe=True,
                threat_level=ThreatLevel.NONE,
                threats=[],
            )

        threats: list[dict[str, Any]] = []
        max_level = ThreatLevel.NONE

        # Check injection patterns
        for pattern, threat_type, level in COMPILED_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                threats.append({
                    "type": threat_type.value,
                    "level": level.value,
                    "matches": matches[:3],
                    "pattern": pattern.pattern[:50],
                })
                if self._level_value(level) > self._level_value(max_level):
                    max_level = level

        # Check encoding attacks
        if _has_suspicious_unicode(text):
            threats.append({
                "type": ThreatType.ENCODING_ATTACK.value,
                "level": ThreatLevel.MEDIUM.value,
                "detail": "suspicious_unicode",
            })
            if self._level_value(ThreatLevel.MEDIUM) > self._level_value(max_level):
                max_level = ThreatLevel.MEDIUM

        # Determine if should block
        blocked = False
        if self.block_on_critical and max_level == ThreatLevel.CRITICAL:
            blocked = True
        elif self.block_on_high and max_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL):
            blocked = True

        # Sanitize if requested
        sanitized = None
        if self.sanitize and threats:
            sanitized = self._sanitize(text)

        # Log threats
        if threats:
            logger.warning(
                "InputGuard: detected %d threats (max_level=%s, blocked=%s)",
                len(threats), max_level.value, blocked
            )
            for t in threats[:5]:
                logger.warning("  - %s: %s", t["type"], t.get("matches", t.get("detail", "")))

        return GuardResult(
            safe=len(threats) == 0,
            threat_level=max_level,
            threats=threats,
            sanitized_input=sanitized,
            blocked=blocked,
        )

    def _level_value(self, level: ThreatLevel) -> int:
        """Get numeric value for threat level comparison."""
        order = [ThreatLevel.NONE, ThreatLevel.LOW, ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        return order.index(level)

    def _sanitize(self, text: str) -> str:
        """Remove or neutralize suspicious patterns."""
        result = text

        # Remove zero-width characters
        for code in [0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF]:
            result = result.replace(chr(code), "")

        # Neutralize XML-like tags
        result = re.sub(r"</?(system|assistant|user|human|ai)>", r"[\1]", result, flags=re.IGNORECASE)

        # Neutralize special tokens
        result = re.sub(r"\[/?(?:INST|SYS|SYSTEM)\]", "[token]", result, flags=re.IGNORECASE)

        return result

    def check_and_sanitize(self, text: str) -> tuple[str, GuardResult]:
        """Check and return sanitized text.

        Returns:
            (sanitized_text, guard_result)
        """
        result = self.check(text)

        if result.blocked:
            return "", result

        if result.sanitized_input:
            return result.sanitized_input, result

        return text, result


# Global guard instance
_guard: InputGuard | None = None


def get_input_guard() -> InputGuard:
    """Get or create global input guard."""
    global _guard
    if _guard is None:
        _guard = InputGuard()
    return _guard


def check_input(text: str) -> GuardResult:
    """Check input for injection (convenience function)."""
    return get_input_guard().check(text)


def sanitize_input(text: str) -> tuple[str, GuardResult]:
    """Check and sanitize input (convenience function)."""
    return get_input_guard().check_and_sanitize(text)
