"""
Validation Schema — shared types for the validation pipeline.
This file exists to break circular dependencies between validation logic and rules.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any


class IssueSeverity(str, Enum):
    """Severity level for a validation issue."""
    ERROR = "error"       # Must fix — blocks acceptance
    WARNING = "warning"   # Should fix — repair if budget allows
    INFO = "info"         # Nice to have — don't block or repair


class IssueCategory(str, Enum):
    """Category of validation issue for targeted repair."""
    FORBIDDEN_PATTERN = "forbidden_pattern"
    MISSING_ANNOTATION = "missing_annotation"
    MISSING_IMPORT = "missing_import"
    MISSING_AAA = "missing_aaa"
    MISSING_VERIFY = "missing_verify"
    MISSING_DISPLAY_NAME = "missing_display_name"
    MISSING_TEST = "missing_test"
    STRUCTURAL = "structural"
    NAMING = "naming"
    QUALITY = "quality"
    STATIC_CALL_WITHOUT_MOCK = "static_call_without_mock"
    DATETIME_NOW_IN_TEST = "datetime_now_in_test"
    MISSING_MOCK_FIELD = "missing_mock_field"
    CHAINED_VERIFY_ON_STATIC = "chained_verify_on_static"
    DOMAIN_TYPE_GUESSING = "domain_type_guessing"
    WRONG_CONSTRUCTION_PATTERN = "wrong_construction_pattern"
    INCONSISTENT_MOCK_VALUE = "inconsistent_mock_value"
    RAW_VALUE_INSTEAD_OF_ENUM = "raw_value_instead_of_enum"


@dataclass
class ValidationIssue:
    """A single validation issue with severity and category."""
    message: str
    severity: IssueSeverity
    category: IssueCategory
    line: Optional[int] = None
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        prefix = f"[{str(self.severity).upper()}]"
        loc = f" (line {self.line})" if self.line else ""
        return f"{prefix} {self.message}{loc}"


@dataclass
class ValidationResult:
    """Result of the full validation pipeline."""
    issues: list[ValidationIssue] = field(default_factory=list)
    code_length: int = 0
    test_count: int = 0
    mock_count: int = 0

    @property
    def passed(self) -> bool:
        return not any(i.severity == IssueSeverity.ERROR for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]

    @property
    def infos(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.INFO]

    @property
    def all_messages(self) -> list[str]:
        return [str(i) for i in self.issues]

    def get_summary(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "infos": len(self.infos),
            "test_count": self.test_count,
            "mock_count": self.mock_count,
            "categories": list({i.category.value for i in self.issues}),
        }
