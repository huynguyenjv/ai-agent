"""Tests for feedback analyzer."""

import pytest
from datetime import datetime, timedelta

from server.feedback_analyzer import (
    FeedbackAnalyzer,
    FeedbackEntry,
    FeedbackPattern,
    get_feedback_analyzer,
)


class TestFeedbackAnalyzer:
    """Tests for FeedbackAnalyzer."""

    def test_add_feedback(self):
        analyzer = FeedbackAnalyzer()

        analyzer.add_feedback(
            session_id="sess1",
            query="Write a function",
            response="def foo(): pass",
            feedback_type="positive",
            intent="code_gen",
        )

        assert len(analyzer.entries) == 1
        assert analyzer.entries[0].feedback_type == "positive"

    def test_max_entries_limit(self):
        analyzer = FeedbackAnalyzer(max_entries=5)

        for i in range(10):
            analyzer.add_feedback(
                session_id=f"sess{i}",
                query=f"query{i}",
                response="response",
                feedback_type="positive",
            )

        assert len(analyzer.entries) == 5
        assert analyzer.entries[0].session_id == "sess5"

    def test_get_stats(self):
        analyzer = FeedbackAnalyzer()

        analyzer.add_feedback("s1", "q1", "r1", "positive", intent="code_gen")
        analyzer.add_feedback("s2", "q2", "r2", "negative", intent="code_gen")
        analyzer.add_feedback("s3", "q3", "r3", "positive", intent="explain")

        stats = analyzer.get_stats()

        assert stats["total"] == 3
        assert stats["by_type"]["positive"] == 2
        assert stats["by_type"]["negative"] == 1
        assert stats["satisfaction_rate"] == pytest.approx(2/3)

    def test_empty_stats(self):
        analyzer = FeedbackAnalyzer()
        stats = analyzer.get_stats()

        assert stats["total"] == 0
        assert stats["satisfaction_rate"] is None


class TestPatternDetection:
    """Tests for pattern detection."""

    def test_retry_pattern(self):
        analyzer = FeedbackAnalyzer()

        # Add retries for same intent
        for i in range(5):
            analyzer.add_feedback(
                f"sess{i}", f"query{i}", "response", "retry", intent="code_review"
            )

        patterns = analyzer.analyze_patterns(min_frequency=3)

        retry_patterns = [p for p in patterns if p.pattern_type == "high_retry_rate"]
        assert len(retry_patterns) > 0
        assert "code_review" in retry_patterns[0].description

    def test_keyword_pattern(self):
        analyzer = FeedbackAnalyzer()

        for i in range(4):
            analyzer.add_feedback(
                f"sess{i}", f"query{i}", "response", "negative",
                feedback_text="The answer was wrong and incomplete",
            )

        patterns = analyzer.analyze_patterns(min_frequency=3)

        keyword_patterns = [p for p in patterns if p.pattern_type == "keyword_pattern"]
        assert len(keyword_patterns) > 0

    def test_length_pattern(self):
        analyzer = FeedbackAnalyzer()

        for i in range(4):
            analyzer.add_feedback(
                f"sess{i}", f"query{i}", "response", "negative",
                feedback_text="Response was too long",
            )

        patterns = analyzer.analyze_patterns(min_frequency=3)

        length_patterns = [p for p in patterns if p.pattern_type == "length_issue"]
        assert len(length_patterns) > 0
        assert "verbose" in length_patterns[0].description


class TestPromptSuggestions:
    """Tests for prompt improvement suggestions."""

    def test_suggest_improvements(self):
        analyzer = FeedbackAnalyzer()

        # Add some negative patterns
        for i in range(5):
            analyzer.add_feedback(
                f"sess{i}", f"query{i}", "response", "retry", intent="unit_test"
            )

        suggestions = analyzer.suggest_prompt_improvements()

        assert len(suggestions) > 0
        assert all("action" in s for s in suggestions)

    def test_no_suggestions_when_no_patterns(self):
        analyzer = FeedbackAnalyzer()

        analyzer.add_feedback("s1", "q1", "r1", "positive")

        suggestions = analyzer.suggest_prompt_improvements()

        assert suggestions == []


class TestSingleton:
    """Tests for singleton accessor."""

    def test_get_feedback_analyzer(self):
        analyzer1 = get_feedback_analyzer()
        analyzer2 = get_feedback_analyzer()

        assert analyzer1 is analyzer2
