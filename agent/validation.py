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
    # Anti-pattern categories
    STATIC_CALL_WITHOUT_MOCK = "static_call_without_mock"
    INCONSISTENT_MOCK_VALUE = "inconsistent_mock_value"
    DATETIME_NOW_IN_TEST = "datetime_now_in_test"
    MISSING_MOCK_FIELD = "missing_mock_field"
    RAW_VALUE_INSTEAD_OF_ENUM = "raw_value_instead_of_enum"
    CHAINED_VERIFY_ON_STATIC = "chained_verify_on_static"
    DOMAIN_TYPE_GUESSING = "domain_type_guessing"
    WRONG_CONSTRUCTION_PATTERN = "wrong_construction_pattern"


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

    def validate(self, code: str, rag_chunks: list | None = None) -> ValidationResult:
        """Run the full validation pipeline on generated test code.

        Args:
            code: The generated Java test code.
            rag_chunks: Optional RAG context chunks. When provided, enables
                        Pass 7 (construction pattern cross-check) which detects
                        .builder() on records without @Builder, wrong field
                        names, etc.
        """
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

        # Pass 6: Anti-pattern detection (complex service issues)
        self._check_anti_patterns(code, result)

        # Pass 7: RAG-aware construction pattern cross-check
        if rag_chunks:
            self._check_construction_patterns(code, rag_chunks, result)

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

    def _check_anti_patterns(self, code: str, result: ValidationResult) -> None:
        """Pass 6: Detect common anti-patterns that cause compilation/runtime failures.

        These are the most frequent issues when generating tests for complex services
        with SecurityContextHolder, multiple dependencies, and domain models.
        """
        # --- AP1: SecurityContextHolder used without MockedStatic ---
        has_security_context_holder = bool(
            re.search(r"\bSecurityContextHolder\b", code)
        )
        has_mocked_static_security = bool(
            re.search(r"MockedStatic\s*<\s*SecurityContextHolder\s*>", code)
        )
        if has_security_context_holder and not has_mocked_static_security:
            # Direct call like SecurityContextHolder.setContext() or .getContext()
            result.issues.append(ValidationIssue(
                message="SecurityContextHolder used without MockedStatic — will leak state between tests",
                severity=IssueSeverity.ERROR,
                category=IssueCategory.STATIC_CALL_WITHOUT_MOCK,
                suggestion=(
                    "Use MockedStatic<SecurityContextHolder> with try-with-resources: "
                    "try (MockedStatic<SecurityContextHolder> m = mockStatic(SecurityContextHolder.class)) { ... }"
                ),
            ))

        # --- AP2: Other common static utility calls without MockedStatic ---
        static_utils = [
            (r"\bLocalDateTime\.now\(\)", "LocalDateTime.now()", "LocalDateTime"),
            (r"\bLocalDate\.now\(\)", "LocalDate.now()", "LocalDate"),
            (r"\bInstant\.now\(\)", "Instant.now()", "Instant"),
            (r"\bUUID\.randomUUID\(\)", "UUID.randomUUID()", "UUID"),
        ]
        for pattern, name, class_name in static_utils:
            if re.search(pattern, code):
                has_mock = bool(re.search(
                    rf"MockedStatic\s*<\s*{class_name}\s*>", code
                ))
                if not has_mock:
                    # Check if it's in test data (problematic) vs in import (ok)
                    # Only flag if used in method body, not in import
                    in_method = bool(re.search(
                        rf"void\s+\w+\(.*?\).*?\{{[^}}]*{re.escape(name)}",
                        code, re.DOTALL
                    ))
                    if in_method:
                        result.issues.append(ValidationIssue(
                            message=(
                                f"{name} in test data — value will differ from runtime. "
                                f"Use ArgumentCaptor, any() matcher, or MockedStatic<{class_name}>"
                            ),
                            severity=IssueSeverity.WARNING,
                            category=IssueCategory.DATETIME_NOW_IN_TEST,
                            suggestion=f"Replace {name} with a fixed value, ArgumentCaptor, or MockedStatic",
                        ))

        # --- AP3: @InjectMocks but potentially missing @Mock fields ---
        # Extract the @InjectMocks class type
        inject_match = re.search(
            r"@InjectMocks\s+(?:private\s+)?(\w+)\s+\w+", code
        )
        mock_fields = re.findall(
            r"@Mock\s+(?:private\s+)?(\w+)\s+\w+", code
        )
        if inject_match and len(mock_fields) == 0:
            result.issues.append(ValidationIssue(
                message=f"@InjectMocks {inject_match.group(1)} has no @Mock fields — all dependencies will be null",
                severity=IssueSeverity.ERROR,
                category=IssueCategory.MISSING_MOCK_FIELD,
                suggestion="Add @Mock field for every constructor parameter of the service under test",
            ))

        # --- AP4: SecurityContextHolder.setContext() direct call ---
        if re.search(r"SecurityContextHolder\.setContext\s*\(", code):
            result.issues.append(ValidationIssue(
                message="SecurityContextHolder.setContext() called directly — use MockedStatic instead",
                severity=IssueSeverity.ERROR,
                category=IssueCategory.STATIC_CALL_WITHOUT_MOCK,
                suggestion=(
                    "Replace SecurityContextHolder.setContext(ctx) with "
                    "MockedStatic: securityMock.when(SecurityContextHolder::getContext).thenReturn(ctx)"
                ),
            ))

        # --- AP5: SecurityContextHolder.getContext() direct call (without MockedStatic) ---
        if (re.search(r"SecurityContextHolder\.getContext\s*\(", code)
                and not has_mocked_static_security):
            result.issues.append(ValidationIssue(
                message="SecurityContextHolder.getContext() called directly without MockedStatic",
                severity=IssueSeverity.ERROR,
                category=IssueCategory.STATIC_CALL_WITHOUT_MOCK,
                suggestion="Wrap in MockedStatic<SecurityContextHolder> try-with-resources block",
            ))

        # --- AP6: import MockedStatic if SecurityContextHolder is used ---
        if has_security_context_holder:
            if "import org.mockito.MockedStatic" not in code:
                result.issues.append(ValidationIssue(
                    message="Missing import for MockedStatic (needed for SecurityContextHolder)",
                    severity=IssueSeverity.WARNING,
                    category=IssueCategory.MISSING_IMPORT,
                    suggestion="Add: import org.mockito.MockedStatic;",
                ))

        # --- AP7: verify() on chained static calls ---
        # Detect: verify(SecurityContextHolder.getContext().getAuthentication()).getName()
        # This is fragile/wrong — should verify on the mock variable directly
        if re.search(
            r"verify\s*\(\s*SecurityContextHolder\s*\.\s*getContext\s*\(",
            code,
        ):
            result.issues.append(ValidationIssue(
                message=(
                    "verify() used on chained SecurityContextHolder.getContext()... call — "
                    "verify the mock object variable directly instead"
                ),
                severity=IssueSeverity.ERROR,
                category=IssueCategory.CHAINED_VERIFY_ON_STATIC,
                suggestion=(
                    "Replace verify(SecurityContextHolder.getContext().getAuthentication()).getName() "
                    "with verify(authentication).getName() — verify on the mock variable, not via static chain"
                ),
            ))

        # --- AP8: Overly verbose setUp building domain objects with many fields ---
        # Heuristic: if setUp has more than 8 builder calls, it's likely guessing fields
        setup_match = re.search(
            r"void\s+setUp\s*\(\s*\)\s*\{(.*?)(?=\n\s*(?:@Test|void\s+\w+_))",
            code, re.DOTALL,
        )
        if setup_match:
            setup_body = setup_match.group(1)
            builder_field_count = len(re.findall(r"\.\w+\([^)]*\)", setup_body))
            if builder_field_count > 30:
                result.issues.append(ValidationIssue(
                    message=(
                        f"setUp has {builder_field_count} builder/setter calls — likely guessing "
                        f"domain type fields. Use mock(Type.class) + stub only accessed fields"
                    ),
                    severity=IssueSeverity.WARNING,
                    category=IssueCategory.DOMAIN_TYPE_GUESSING,
                    suggestion=(
                        "Instead of building domain objects with many guessed fields in setUp, "
                        "use mock(Type.class) and stub only the accessors called in the method under test"
                    ),
                ))
    def _check_construction_patterns(
        self, code: str, rag_chunks: list, result: ValidationResult
    ) -> None:
        """Pass 7: Cross-check construction patterns against RAG metadata.

        Detects:
        - .builder() used on record/class that has NO @Builder
        - new Constructor() used on class that HAS @Builder (should use builder)
        - Wrong field names in builder chains (field not in record_components/fields)
        """
        for chunk in rag_chunks:
            name = chunk.class_name
            if not name:
                continue

            java_type = getattr(chunk, "java_type", None) or chunk.type
            has_builder = getattr(chunk, "has_builder", False)

            # --- Check 1: .builder() on type WITHOUT @Builder ---
            builder_pattern = rf"\b{re.escape(name)}\.builder\s*\("
            if re.search(builder_pattern, code) and not has_builder:
                if java_type == "record":
                    components = getattr(chunk, "record_components", None)
                    args = ", ".join(rc.name for rc in components) if components else "..."
                    result.issues.append(ValidationIssue(
                        message=(
                            f"{name}.builder() used but {name} is a record WITHOUT @Builder. "
                            f"Use canonical constructor: new {name}({args})"
                        ),
                        severity=IssueSeverity.ERROR,
                        category=IssueCategory.WRONG_CONSTRUCTION_PATTERN,
                        suggestion=(
                            f"Replace {name}.builder()...build() with "
                            f"new {name}({args})"
                        ),
                    ))
                else:
                    result.issues.append(ValidationIssue(
                        message=(
                            f"{name}.builder() used but {name} does NOT have @Builder annotation"
                        ),
                        severity=IssueSeverity.ERROR,
                        category=IssueCategory.WRONG_CONSTRUCTION_PATTERN,
                        suggestion=(
                            f"Use constructor or mock({name}.class) instead of .builder()"
                        ),
                    ))

            # --- Check 2: Wrong fields in builder chain ---
            # If builder IS used (with or without @Builder), check field names
            # Use parenthesis-depth tracking to extract only chain-level methods,
            # ignoring nested calls like UUID.fromString() inside arguments.
            known_fields: set[str] = set()
            components = getattr(chunk, "record_components", None)
            fields = getattr(chunk, "fields", None)
            if components:
                known_fields = {rc.name for rc in components}
            elif fields:
                known_fields = {f.name for f in fields}

            if known_fields:
                for m in re.finditer(
                    rf"\b{re.escape(name)}\.builder\s*\(\)", code
                ):
                    chain_fields = self._extract_builder_chain_fields(
                        code, m.end()
                    )
                    wrong = [f for f in chain_fields if f not in known_fields]
                    if wrong:
                        result.issues.append(ValidationIssue(
                            message=(
                                f"{name}: builder uses fields {wrong} "
                                f"which do NOT exist. Known fields: {sorted(known_fields)}"
                            ),
                            severity=IssueSeverity.ERROR,
                            category=IssueCategory.WRONG_CONSTRUCTION_PATTERN,
                            suggestion=(
                                f"Use only these fields for {name}: {sorted(known_fields)}"
                            ),
                        ))

    @staticmethod
    def _extract_builder_chain_fields(code: str, start: int) -> list[str]:
        """Extract top-level builder method names using paren-depth tracking.

        Scans from `start` (right after `.builder()`) and collects `.fieldName(`
        only at depth 0, ignoring nested calls like UUID.fromString() inside args.
        """
        fields: list[str] = []
        depth = 0
        i = start
        length = len(code)
        while i < length:
            c = code[i]
            if c == '(':
                depth += 1
                i += 1
            elif c == ')':
                depth -= 1
                if depth < 0:
                    break
                i += 1
            elif c == '.' and depth == 0:
                m = re.match(r'\.(\w+)\s*\(', code[i:])
                if m:
                    name = m.group(1)
                    if name not in ("build", "toBuilder"):
                        fields.append(name)
                    i += m.end() - 1  # position at '(' — next iter handles depth
                else:
                    i += 1
            elif c == ';':
                break
            else:
                i += 1
        return fields