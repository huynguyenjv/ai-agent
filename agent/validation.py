"""
Validation Pipeline — multi-pass, severity-aware code validation.

Replaces the flat boolean validation with a structured pipeline:

    1. Structural validation  — syntax, imports, class structure
    2. Pattern validation     — JUnit5/Mockito patterns
    3. Quality validation     — AAA pattern, naming, coverage heuristics
    4. Safety validation      — forbidden patterns, Spring context leaks

Each issue carries a severity (ERROR / WARNING / INFO) so the repair
loop can prioritise fixes and the orchestrator can decide whether to
accept code with only warnings vs. blocking on errors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import structlog

logger = structlog.get_logger()


# ── Severity & Issue ─────────────────────────────────────────────────

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


@dataclass
class ValidationIssue:
    """A single validation issue with severity and category."""

    message: str
    severity: IssueSeverity
    category: IssueCategory
    line: Optional[int] = None          # Line number if applicable
    suggestion: Optional[str] = None    # Suggested fix

    def __str__(self) -> str:
        prefix = f"[{self.severity.value.upper()}]"
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
        """True if no ERROR-level issues."""
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
    def error_messages(self) -> list[str]:
        """Flat list of error messages (backward compat with old validator)."""
        return [str(i) for i in self.errors]

    @property
    def all_messages(self) -> list[str]:
        """Flat list of all messages."""
        return [str(i) for i in self.issues]

    def get_issues_by_category(self, category: IssueCategory) -> list[ValidationIssue]:
        return [i for i in self.issues if i.category == category]

    def get_summary(self) -> dict:
        return {
            "passed": self.passed,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "infos": len(self.infos),
            "test_count": self.test_count,
            "mock_count": self.mock_count,
            "categories": list({i.category.value for i in self.issues}),
        }


# ── Validation Pipeline ─────────────────────────────────────────────

class ValidationPipeline:
    """Multi-pass code validation with severity levels.

    Replaces ``TestRules.validate_generated_code()`` with a richer
    pipeline that feeds the repair strategy selector.
    """

    # Forbidden patterns (ERROR — must fix)
    FORBIDDEN_PATTERNS = [
        ("@SpringBootTest", "Use @ExtendWith(MockitoExtension.class) instead"),
        ("@DataJpaTest", "Use @ExtendWith(MockitoExtension.class) instead"),
        ("@WebMvcTest", "Use @ExtendWith(MockitoExtension.class) instead"),
        ("@SpringExtension", "Use MockitoExtension instead"),
        ("@ContextConfiguration", "Not needed for unit tests"),
        ("@RunWith(SpringRunner", "Use @ExtendWith(MockitoExtension.class)"),
        ("@Autowired", "Use @InjectMocks / @Mock instead"),
        ("@MockBean", "Use @Mock instead of @MockBean"),
        ("TestRestTemplate", "Use direct service calls with mocks"),
        ("MockMvc", "Use direct service calls with mocks"),
    ]

    # Required patterns with severity
    REQUIRED_PATTERNS = [
        ("@ExtendWith(MockitoExtension.class)", IssueSeverity.ERROR, IssueCategory.MISSING_ANNOTATION),
        ("@Mock", IssueSeverity.ERROR, IssueCategory.MISSING_ANNOTATION),
        ("@InjectMocks", IssueSeverity.ERROR, IssueCategory.MISSING_ANNOTATION),
        ("@Test", IssueSeverity.ERROR, IssueCategory.MISSING_TEST),
        ("@DisplayName", IssueSeverity.WARNING, IssueCategory.MISSING_DISPLAY_NAME),
    ]

    def validate(self, code: str) -> ValidationResult:
        """Run the full validation pipeline on generated test code."""
        result = ValidationResult(code_length=len(code))

        if not code or len(code.strip()) < 50:
            result.issues.append(ValidationIssue(
                message="Generated code is empty or too short",
                severity=IssueSeverity.ERROR,
                category=IssueCategory.STRUCTURAL,
            ))
            return result

        # Pass 1: Structural checks
        self._check_structure(code, result)

        # Pass 2: Forbidden patterns
        self._check_forbidden(code, result)

        # Pass 3: Required patterns
        self._check_required(code, result)

        # Pass 4: AAA pattern
        self._check_aaa_pattern(code, result)

        # Pass 5: Quality checks
        self._check_quality(code, result)

        # Count tests and mocks
        result.test_count = len(re.findall(r"@Test\b", code))
        result.mock_count = len(re.findall(r"@Mock\b", code))

        logger.info(
            "Validation complete",
            passed=result.passed,
            errors=len(result.errors),
            warnings=len(result.warnings),
            tests=result.test_count,
            mocks=result.mock_count,
        )

        return result

    # ── Pass implementations ─────────────────────────────────────────

    def _check_structure(self, code: str, result: ValidationResult) -> None:
        """Pass 1: Structural checks."""
        # Must have a class declaration
        if not re.search(r"\bclass\s+\w+", code):
            result.issues.append(ValidationIssue(
                message="No class declaration found",
                severity=IssueSeverity.ERROR,
                category=IssueCategory.STRUCTURAL,
            ))

        # Must have balanced braces
        open_braces = code.count("{")
        close_braces = code.count("}")
        if open_braces != close_braces:
            result.issues.append(ValidationIssue(
                message=f"Unbalanced braces: {open_braces} open, {close_braces} close",
                severity=IssueSeverity.ERROR,
                category=IssueCategory.STRUCTURAL,
                suggestion="Check for missing or extra braces at end of file",
            ))

        # Check for import statements
        if "import " not in code:
            result.issues.append(ValidationIssue(
                message="No import statements found",
                severity=IssueSeverity.ERROR,
                category=IssueCategory.MISSING_IMPORT,
                suggestion="Add import org.junit.jupiter.api.Test and other required imports",
            ))

    def _check_forbidden(self, code: str, result: ValidationResult) -> None:
        """Pass 2: Forbidden pattern detection."""
        for pattern, suggestion in self.FORBIDDEN_PATTERNS:
            if pattern in code:
                # Find line number
                line_num = None
                for i, line in enumerate(code.split("\n"), 1):
                    if pattern in line:
                        line_num = i
                        break

                result.issues.append(ValidationIssue(
                    message=f"Forbidden pattern: {pattern}",
                    severity=IssueSeverity.ERROR,
                    category=IssueCategory.FORBIDDEN_PATTERN,
                    line=line_num,
                    suggestion=suggestion,
                ))

    def _check_required(self, code: str, result: ValidationResult) -> None:
        """Pass 3: Required pattern checks."""
        for pattern, severity, category in self.REQUIRED_PATTERNS:
            if pattern not in code:
                result.issues.append(ValidationIssue(
                    message=f"Missing required pattern: {pattern}",
                    severity=severity,
                    category=category,
                    suggestion=f"Add {pattern} to the test class",
                ))

    def _check_aaa_pattern(self, code: str, result: ValidationResult) -> None:
        """Pass 4: AAA (Arrange-Act-Assert) pattern check."""
        has_arrange = "// Arrange" in code or "// arrange" in code
        has_act = "// Act" in code or "// act" in code
        has_assert = "// Assert" in code or "// assert" in code

        if not (has_arrange and has_act and has_assert):
            missing = []
            if not has_arrange:
                missing.append("// Arrange")
            if not has_act:
                missing.append("// Act")
            if not has_assert:
                missing.append("// Assert")

            result.issues.append(ValidationIssue(
                message=f"AAA pattern incomplete. Missing: {', '.join(missing)}",
                severity=IssueSeverity.WARNING,
                category=IssueCategory.MISSING_AAA,
                suggestion="Add // Arrange, // Act, // Assert comments to each test method",
            ))

    def _check_quality(self, code: str, result: ValidationResult) -> None:
        """Pass 5: Quality and best-practice checks."""
        # Check for verify() calls
        if "verify(" not in code:
            result.issues.append(ValidationIssue(
                message="No verify() calls found — consider adding interaction verification",
                severity=IssueSeverity.WARNING,
                category=IssueCategory.MISSING_VERIFY,
                suggestion="Add verify(mockObject).method() after Act section",
            ))

        # Check test naming convention
        test_methods = re.findall(r"void\s+(\w+)\s*\(", code)
        for method in test_methods:
            if method in ("setUp", "tearDown", "beforeAll", "afterAll"):
                continue
            # Expected: method_WhenCondition_ShouldResult
            if not re.match(r"\w+_\w+_\w+", method) and not method.startswith("should"):
                result.issues.append(ValidationIssue(
                    message=f"Test method '{method}' doesn't follow naming convention: method_WhenCondition_ShouldResult",
                    severity=IssueSeverity.INFO,
                    category=IssueCategory.NAMING,
                ))

        # Check for lenient() usage
        if "lenient()" in code:
            result.issues.append(ValidationIssue(
                message="lenient() usage detected — avoid in unit tests",
                severity=IssueSeverity.WARNING,
                category=IssueCategory.QUALITY,
                suggestion="Remove lenient() and use strict stubbing",
            ))

        # Check for @BeforeEach
        if "@BeforeEach" not in code:
            result.issues.append(ValidationIssue(
                message="No @BeforeEach setup method found",
                severity=IssueSeverity.INFO,
                category=IssueCategory.QUALITY,
                suggestion="Add @BeforeEach void setUp() for common test data initialization",
            ))
