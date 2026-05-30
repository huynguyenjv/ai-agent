"""Feedback analyzer for prompt refinement.

Analyzes user feedback patterns to suggest prompt improvements.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger("server.feedback_analyzer")


@dataclass
class FeedbackEntry:
    """A single feedback entry."""
    timestamp: datetime
    session_id: str
    query: str
    response_preview: str
    feedback_type: str  # positive, negative, correction, retry
    feedback_text: str | None = None
    intent: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class FeedbackPattern:
    """A detected feedback pattern."""
    pattern_type: str
    description: str
    frequency: int
    examples: list[str]
    suggested_action: str
    severity: str  # low, medium, high


class FeedbackAnalyzer:
    """Analyzes feedback to identify improvement patterns."""

    def __init__(self, max_entries: int = 1000):
        self.entries: list[FeedbackEntry] = []
        self.max_entries = max_entries
        self._pattern_cache: dict[str, Any] = {}

    def add_feedback(
        self,
        session_id: str,
        query: str,
        response: str,
        feedback_type: str,
        feedback_text: str | None = None,
        intent: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Record user feedback.

        Args:
            session_id: Session identifier
            query: Original query
            response: Agent response
            feedback_type: positive/negative/correction/retry
            feedback_text: Optional feedback text
            intent: Classified intent
            tags: Optional tags
        """
        entry = FeedbackEntry(
            timestamp=datetime.now(),
            session_id=session_id,
            query=query,
            response_preview=response[:500] if response else "",
            feedback_type=feedback_type,
            feedback_text=feedback_text,
            intent=intent,
            tags=tags or [],
        )

        self.entries.append(entry)

        # Trim old entries
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

        # Invalidate cache
        self._pattern_cache.clear()

    def analyze_patterns(
        self,
        time_window: timedelta | None = None,
        min_frequency: int = 3,
    ) -> list[FeedbackPattern]:
        """Analyze feedback for patterns.

        Args:
            time_window: Only analyze feedback within window
            min_frequency: Minimum occurrences for pattern

        Returns:
            List of detected patterns
        """
        # Filter by time
        entries = self.entries
        if time_window:
            cutoff = datetime.now() - time_window
            entries = [e for e in entries if e.timestamp >= cutoff]

        if not entries:
            return []

        patterns: list[FeedbackPattern] = []

        # Pattern 1: High retry rate for intent
        patterns.extend(self._detect_retry_patterns(entries, min_frequency))

        # Pattern 2: Negative feedback keywords
        patterns.extend(self._detect_keyword_patterns(entries, min_frequency))

        # Pattern 3: Correction patterns
        patterns.extend(self._detect_correction_patterns(entries, min_frequency))

        # Pattern 4: Length issues
        patterns.extend(self._detect_length_patterns(entries, min_frequency))

        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        patterns.sort(key=lambda p: (severity_order.get(p.severity, 3), -p.frequency))

        return patterns

    def _detect_retry_patterns(
        self,
        entries: list[FeedbackEntry],
        min_freq: int,
    ) -> list[FeedbackPattern]:
        """Detect intents with high retry rates."""
        patterns = []

        # Count retries per intent
        intent_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "retry": 0})

        for entry in entries:
            intent = entry.intent or "unknown"
            intent_counts[intent]["total"] += 1
            if entry.feedback_type == "retry":
                intent_counts[intent]["retry"] += 1

        for intent, counts in intent_counts.items():
            if counts["total"] >= min_freq:
                retry_rate = counts["retry"] / counts["total"]
                if retry_rate > 0.3:
                    patterns.append(FeedbackPattern(
                        pattern_type="high_retry_rate",
                        description=f"Intent '{intent}' has {retry_rate:.0%} retry rate",
                        frequency=counts["retry"],
                        examples=[e.query for e in entries if e.intent == intent and e.feedback_type == "retry"][:3],
                        suggested_action=f"Review and improve prompts for '{intent}' intent",
                        severity="high" if retry_rate > 0.5 else "medium",
                    ))

        return patterns

    def _detect_keyword_patterns(
        self,
        entries: list[FeedbackEntry],
        min_freq: int,
    ) -> list[FeedbackPattern]:
        """Detect negative feedback keyword patterns."""
        patterns = []

        # Keywords indicating problems
        problem_keywords = {
            "wrong": "incorrect_output",
            "error": "error_prone",
            "missing": "incomplete",
            "incomplete": "incomplete",
            "too long": "verbosity",
            "too short": "brevity",
            "slow": "performance",
            "confus": "clarity",
            "unclear": "clarity",
            "doesn't work": "broken",
            "broken": "broken",
        }

        keyword_examples: dict[str, list[str]] = defaultdict(list)

        for entry in entries:
            if entry.feedback_type == "negative" and entry.feedback_text:
                text_lower = entry.feedback_text.lower()
                for keyword, category in problem_keywords.items():
                    if keyword in text_lower:
                        keyword_examples[category].append(entry.feedback_text)

        for category, examples in keyword_examples.items():
            if len(examples) >= min_freq:
                patterns.append(FeedbackPattern(
                    pattern_type="keyword_pattern",
                    description=f"Feedback frequently mentions '{category}' issues",
                    frequency=len(examples),
                    examples=examples[:3],
                    suggested_action=_get_action_for_category(category),
                    severity="medium",
                ))

        return patterns

    def _detect_correction_patterns(
        self,
        entries: list[FeedbackEntry],
        min_freq: int,
    ) -> list[FeedbackPattern]:
        """Detect patterns in user corrections."""
        patterns = []

        corrections = [e for e in entries if e.feedback_type == "correction" and e.feedback_text]

        if len(corrections) < min_freq:
            return patterns

        # Look for common correction types
        correction_types: dict[str, list[str]] = defaultdict(list)

        for entry in corrections:
            text = entry.feedback_text or ""

            if re.search(r"\b(use|instead|not)\b", text.lower()):
                correction_types["style_preference"].append(text)
            elif re.search(r"\b(add|include|also)\b", text.lower()):
                correction_types["missing_info"].append(text)
            elif re.search(r"\b(remove|don't|no)\b", text.lower()):
                correction_types["unwanted_content"].append(text)

        for corr_type, examples in correction_types.items():
            if len(examples) >= min_freq:
                patterns.append(FeedbackPattern(
                    pattern_type="correction_pattern",
                    description=f"Users frequently correct '{corr_type}'",
                    frequency=len(examples),
                    examples=examples[:3],
                    suggested_action=_get_action_for_correction(corr_type),
                    severity="medium",
                ))

        return patterns

    def _detect_length_patterns(
        self,
        entries: list[FeedbackEntry],
        min_freq: int,
    ) -> list[FeedbackPattern]:
        """Detect response length issues."""
        patterns = []

        # Check for too long/short feedback
        too_long = [e for e in entries if e.feedback_type == "negative" and
                    e.feedback_text and "long" in e.feedback_text.lower()]
        too_short = [e for e in entries if e.feedback_type == "negative" and
                     e.feedback_text and "short" in e.feedback_text.lower()]

        if len(too_long) >= min_freq:
            patterns.append(FeedbackPattern(
                pattern_type="length_issue",
                description="Responses frequently too verbose",
                frequency=len(too_long),
                examples=[e.feedback_text or "" for e in too_long[:3]],
                suggested_action="Add conciseness instruction to system prompt",
                severity="low",
            ))

        if len(too_short) >= min_freq:
            patterns.append(FeedbackPattern(
                pattern_type="length_issue",
                description="Responses frequently too brief",
                frequency=len(too_short),
                examples=[e.feedback_text or "" for e in too_short[:3]],
                suggested_action="Add detail instruction to system prompt",
                severity="low",
            ))

        return patterns

    def get_stats(self) -> dict[str, Any]:
        """Get feedback statistics.

        Returns:
            Statistics dict
        """
        if not self.entries:
            return {
                "total": 0,
                "by_type": {},
                "by_intent": {},
                "satisfaction_rate": None,
                "oldest": None,
                "newest": None,
            }

        by_type: dict[str, int] = defaultdict(int)
        by_intent: dict[str, int] = defaultdict(int)

        for entry in self.entries:
            by_type[entry.feedback_type] += 1
            if entry.intent:
                by_intent[entry.intent] += 1

        positive = by_type.get("positive", 0)
        negative = by_type.get("negative", 0)
        total_rated = positive + negative

        return {
            "total": len(self.entries),
            "by_type": dict(by_type),
            "by_intent": dict(by_intent),
            "satisfaction_rate": positive / total_rated if total_rated > 0 else None,
            "oldest": self.entries[0].timestamp.isoformat() if self.entries else None,
            "newest": self.entries[-1].timestamp.isoformat() if self.entries else None,
        }

    def suggest_prompt_improvements(self) -> list[dict[str, str]]:
        """Generate prompt improvement suggestions.

        Returns:
            List of suggestions
        """
        patterns = self.analyze_patterns()
        suggestions = []

        for pattern in patterns:
            suggestions.append({
                "issue": pattern.description,
                "action": pattern.suggested_action,
                "severity": pattern.severity,
                "evidence_count": pattern.frequency,
            })

        return suggestions


def _get_action_for_category(category: str) -> str:
    """Get suggested action for feedback category."""
    actions = {
        "incorrect_output": "Add validation step or fact-checking instruction",
        "error_prone": "Add error handling guidance to prompts",
        "incomplete": "Add completeness checklist to prompts",
        "verbosity": "Add conciseness instruction (e.g., 'be brief')",
        "brevity": "Add detail instruction (e.g., 'provide full explanation')",
        "performance": "Optimize tool selection or add caching",
        "clarity": "Add structure guidelines (headers, bullets)",
        "broken": "Review tool integration and error handling",
    }
    return actions.get(category, "Review and refine prompts")


def _get_action_for_correction(corr_type: str) -> str:
    """Get suggested action for correction type."""
    actions = {
        "style_preference": "Update style guidelines in system prompt",
        "missing_info": "Add completeness checklist to prompts",
        "unwanted_content": "Add negative examples to prompts",
    }
    return actions.get(corr_type, "Review correction patterns and update prompts")


# Singleton instance
_analyzer: FeedbackAnalyzer | None = None


def get_feedback_analyzer() -> FeedbackAnalyzer:
    """Get singleton feedback analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = FeedbackAnalyzer()
    return _analyzer
