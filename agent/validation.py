"""
Validation Pipeline — multi-pass, severity-aware code validation.
Replaces binary validation with a structured pipeline driven by a rule-engine.
"""

from __future__ import annotations

import re
import structlog
from dataclasses import dataclass, field
from .validation_schema import IssueSeverity, IssueCategory, ValidationIssue, ValidationResult

# Delay import to avoid circular dependencies if any
from . import validation_rules as rules

logger = structlog.get_logger()


# ── Severity & Issue ─────────────────────────────────────────────────
# Types moved to validation_schema.py to break cycles.


# ── Validation Pipeline ─────────────────────────────────────────────

class ValidationPipeline:
    """Multi-pass code validation with dynamic rule loading."""

    def validate(self, code: str, rag_chunks: list[Any] | None = None) -> ValidationResult:
        result = ValidationResult(code_length=len(code))

        if not code or len(code.strip()) < 50:
            result.issues.append(ValidationIssue(
                message="Generated code is empty or too short",
                severity=IssueSeverity.ERROR,
                category=IssueCategory.STRUCTURAL,
            ))
            return result

        # Pass 1: Structural checks (Syntax/Braces)
        self._check_structure(code, result)

        # Pass 2 & 3: Forbidden & Required patterns (from config)
        self._check_configured_patterns(code, result)

        # Pass 4: Pattern logic (AAA, Quality)
        self._check_aaa_pattern(code, result)
        self._check_quality(code, result)

        # Pass 5: Advanced Anti-patterns (Dynamic Rule Engine)
        self._check_advanced_rules(code, result)

        # Pass 6: RAG-aware construction cross-check
        if rag_chunks:
            self._check_construction_patterns(code, rag_chunks, result)

        # Metrics
        result.test_count = len(re.findall(r"@Test\b", code))
        result.mock_count = len(re.findall(r"@Mock\b", code))

        logger.info("Validation complete", summary=result.get_summary())
        return result

    # ── Internal Pass Implementations ─────────────────────────────────

    def _check_structure(self, code: str, result: ValidationResult) -> None:
        if not re.search(r"\bclass\s+\w+", code):
            result.issues.append(ValidationIssue("No class declaration found", IssueSeverity.ERROR, IssueCategory.STRUCTURAL))

        if code.count("{") != code.count("}"):
            result.issues.append(ValidationIssue("Unbalanced braces found", IssueSeverity.ERROR, IssueCategory.STRUCTURAL, suggestion="Check for missing or extra braces"))

        if "import " not in code:
            result.issues.append(ValidationIssue("No import statements found", IssueSeverity.ERROR, IssueCategory.MISSING_IMPORT))

    def _check_configured_patterns(self, code: str, result: ValidationResult) -> None:
        # Forbidden
        for pattern, suggestion in rules.FORBIDDEN_PATTERNS:
            if pattern in code:
                line_idx = next((i for i, line in enumerate(code.split("\n"), 1) if pattern in line), None)
                result.issues.append(ValidationIssue(f"Forbidden pattern: {pattern}", IssueSeverity.ERROR, IssueCategory.FORBIDDEN_PATTERN, line=line_idx, suggestion=suggestion))
        
        # Required
        for pattern, sev, cat in rules.REQUIRED_PATTERNS:
            if pattern not in code:
                result.issues.append(ValidationIssue(f"Missing required pattern: {pattern}", sev, cat, suggestion=f"Add {pattern} to the test class"))

    def _check_aaa_pattern(self, code: str, result: ValidationResult) -> None:
        missing = [p for p in ["// Arrange", "// Act", "// Assert"] if p.lower() not in code.lower()]
        if missing:
            result.issues.append(ValidationIssue(f"AAA pattern incomplete. Missing: {', '.join(missing)}", IssueSeverity.WARNING, IssueCategory.MISSING_AAA, suggestion="Add // Arrange, // Act, // Assert comments"))

    def _check_quality(self, code: str, result: ValidationResult) -> None:
        if "verify(" not in code:
            result.issues.append(ValidationIssue("No verify() calls found", IssueSeverity.WARNING, IssueCategory.MISSING_VERIFY))
        
        if "lenient()" in code:
            result.issues.append(ValidationIssue("lenient() usage detected — avoid in unit tests", IssueSeverity.WARNING, IssueCategory.QUALITY, suggestion="Use strict stubbing"))

        if "@BeforeEach" not in code:
            result.issues.append(ValidationIssue("No @BeforeEach setup method found", IssueSeverity.INFO, IssueCategory.QUALITY))

    def _check_advanced_rules(self, code: str, result: ValidationResult) -> None:
        """Dynamic rule engine for complex anti-patterns."""
        for rule in rules.ANTI_PATTERNS:
            match = re.search(rule["pattern"], code)
            if not match:
                continue

            # check exclusion
            if rule.get("exclude") and re.search(rule["exclude"], code):
                continue
            
            # check mandatory imports
            if rule.get("required_import") and rule["required_import"] not in code:
                # This rule triggers if pattern IS present but import NOT present
                pass # logic below handles issuing
            
            # check custom function
            if rule.get("check_func") == "check_inject_mocks_has_mocks":
                if "@InjectMocks" in code and "@Mock" not in code:
                    pass # trigger
                else: continue

            # Format message with named groups if available
            msg = rule["message"].format(**match.groupdict()) if match.groupdict() else rule["message"]
            result.issues.append(ValidationIssue(msg, rule["severity"], rule["category"], suggestion=rule.get("suggestion")))

        # Check static utils ( Pass 6.2)
        for util in rules.STATIC_UTILS:
            if re.search(util["pattern"], code):
                has_mock = bool(re.search(rf"MockedStatic\s*<\s*{util['class']}\s*>", code))
                if not has_mock and bool(re.search(rf"void\s+\w+\(.*?\).*?\{{[^}}]*{re.escape(util['name'])}", code, re.DOTALL)):
                    result.issues.append(ValidationIssue(f"{util['name']} in test data without MockedStatic", IssueSeverity.WARNING, IssueCategory.DATETIME_NOW_IN_TEST, suggestion=f"Use MockedStatic<{util['class']}>"))

    def _check_construction_patterns(self, code: str, rag_chunks: list[Any], result: ValidationResult) -> None:
        """Pass 7: Cross-check construction patterns against RAG metadata."""
        for chunk in rag_chunks:
            name = getattr(chunk, "class_name", None)
            if not name: continue

            java_type = getattr(chunk, "java_type", None) or getattr(chunk, "type", None)
            has_builder = getattr(chunk, "has_builder", False)

            if f"{name}.builder(" in code and not has_builder:
                suggestion = f"Use canonical constructor: new {name}(...)" if java_type == "record" else f"Use constructor or mock({name}.class)"
                result.issues.append(ValidationIssue(f"{name}.builder() used but {name} has NO @Builder", IssueSeverity.ERROR, IssueCategory.WRONG_CONSTRUCTION_PATTERN, suggestion=suggestion))

            # Field name checks in builder
            known_fields = set()
            components = getattr(chunk, "record_components", None)
            fields = getattr(chunk, "fields", None)
            if components: known_fields = {rc.name for rc in components}
            elif fields: known_fields = {f.name for f in fields}

            if known_fields and f"{name}.builder()" in code:
                # Logic to extract fields from chain (simplified for this refactor demo)
                pass 