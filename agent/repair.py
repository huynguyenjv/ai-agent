"""
Repair Strategies — targeted fix strategies based on validation issue categories.

Instead of blindly re-generating code, the repair system:
  1. Classifies each validation issue by category
  2. Selects a repair strategy per category
  3. Builds a focused repair prompt with specific instructions

This results in much higher repair success rates because the LLM gets
precise instructions like "remove @SpringBootTest and replace with
@ExtendWith(MockitoExtension.class)" instead of "fix the issues".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import structlog

from .validation import ValidationResult, ValidationIssue, IssueCategory, IssueSeverity

logger = structlog.get_logger()


# ── Strategy types ───────────────────────────────────────────────────

class RepairAction(str, Enum):
    """Type of repair action to take."""
    REPLACE_PATTERN = "replace_pattern"    # Replace forbidden with correct
    ADD_PATTERN = "add_pattern"            # Add missing pattern
    RESTRUCTURE = "restructure"            # Major restructuring needed
    MINOR_FIX = "minor_fix"               # Small fix (naming, comments)
    REGENERATE = "regenerate"              # Full re-generation needed


@dataclass
class RepairInstruction:
    """A single repair instruction for the LLM."""

    action: RepairAction
    description: str
    priority: int            # Lower = more important
    category: IssueCategory
    search_pattern: Optional[str] = None   # What to find
    replacement: Optional[str] = None      # What to replace with
    context: Optional[str] = None          # Additional context


@dataclass
class RepairPlan:
    """A plan consisting of ordered repair instructions."""

    instructions: list[RepairInstruction] = field(default_factory=list)
    original_issues: list[ValidationIssue] = field(default_factory=list)
    attempt_number: int = 0
    strategy: str = "targeted"  # "targeted" or "regenerate"

    @property
    def has_structural_issues(self) -> bool:
        return any(i.category == IssueCategory.STRUCTURAL for i in self.original_issues)

    @property
    def instruction_count(self) -> int:
        return len(self.instructions)

    def get_repair_prompt_section(self) -> str:
        """Generate the repair instruction section for the LLM prompt."""
        if not self.instructions:
            return ""

        lines = ["## Repair Instructions (MUST FOLLOW IN ORDER)\n"]

        for i, inst in enumerate(sorted(self.instructions, key=lambda x: x.priority), 1):
            lines.append(f"### Fix {i}: {inst.description}")
            if inst.search_pattern and inst.replacement:
                lines.append(f"- Find: `{inst.search_pattern}`")
                lines.append(f"- Replace with: `{inst.replacement}`")
            if inst.context:
                lines.append(f"- Note: {inst.context}")
            lines.append("")

        return "\n".join(lines)


# ── Strategy Selector ────────────────────────────────────────────────

class RepairStrategySelector:
    """Selects and builds repair strategies based on validation results."""

    # Maps issue category → repair strategy builder
    _STRATEGY_MAP = {
        IssueCategory.FORBIDDEN_PATTERN: "_strategy_forbidden",
        IssueCategory.MISSING_ANNOTATION: "_strategy_missing_annotation",
        IssueCategory.MISSING_IMPORT: "_strategy_missing_import",
        IssueCategory.MISSING_AAA: "_strategy_missing_aaa",
        IssueCategory.MISSING_VERIFY: "_strategy_missing_verify",
        IssueCategory.MISSING_DISPLAY_NAME: "_strategy_missing_display_name",
        IssueCategory.MISSING_TEST: "_strategy_missing_test",
        IssueCategory.STRUCTURAL: "_strategy_structural",
        IssueCategory.NAMING: "_strategy_naming",
        IssueCategory.QUALITY: "_strategy_quality",
        # Anti-pattern strategies
        IssueCategory.STATIC_CALL_WITHOUT_MOCK: "_strategy_static_call",
        IssueCategory.INCONSISTENT_MOCK_VALUE: "_strategy_inconsistent_values",
        IssueCategory.DATETIME_NOW_IN_TEST: "_strategy_datetime_now",
        IssueCategory.MISSING_MOCK_FIELD: "_strategy_missing_mock_field",
        IssueCategory.RAW_VALUE_INSTEAD_OF_ENUM: "_strategy_raw_value",
        IssueCategory.CHAINED_VERIFY_ON_STATIC: "_strategy_chained_verify",
        IssueCategory.DOMAIN_TYPE_GUESSING: "_strategy_domain_guessing",
        IssueCategory.WRONG_CONSTRUCTION_PATTERN: "_strategy_wrong_construction",
    }

    def build_repair_plan(
        self,
        validation_result: ValidationResult,
        attempt_number: int = 1,
        max_attempts: int = 2,
    ) -> RepairPlan:
        """Build a repair plan from validation results.

        If this is the last attempt and there are structural issues,
        falls back to full regeneration strategy.
        """
        plan = RepairPlan(
            original_issues=[i for i in validation_result.issues if i.severity != IssueSeverity.INFO],
            attempt_number=attempt_number,
        )

        # On last attempt with structural issues → regenerate
        if attempt_number >= max_attempts and validation_result.errors:
            plan.strategy = "regenerate"
            plan.instructions.append(RepairInstruction(
                action=RepairAction.REGENERATE,
                description="Full regeneration — previous repair attempts failed",
                priority=0,
                category=IssueCategory.STRUCTURAL,
                context="Start from scratch. Follow ALL rules strictly. "
                        "Previous issues: " + "; ".join(str(i) for i in validation_result.errors),
            ))
            return plan

        # Targeted strategy: build instruction for each issue
        plan.strategy = "targeted"
        seen_categories: set[IssueCategory] = set()

        for issue in validation_result.issues:
            if issue.severity == IssueSeverity.INFO:
                continue  # Skip info-level in repair

            # Deduplicate by category (one instruction per category)
            if issue.category in seen_categories:
                continue
            seen_categories.add(issue.category)

            strategy_method = self._STRATEGY_MAP.get(issue.category)
            if strategy_method:
                method = getattr(self, strategy_method)
                instructions = method(issue, validation_result)
                plan.instructions.extend(instructions)

        logger.info(
            "Repair plan built",
            attempt=attempt_number,
            strategy=plan.strategy,
            instructions=len(plan.instructions),
            error_categories=list(seen_categories),
        )

        return plan

    # ── Individual strategy builders ─────────────────────────────────

    def _strategy_forbidden(
        self, issue: ValidationIssue, result: ValidationResult
    ) -> list[RepairInstruction]:
        """Replace forbidden patterns with correct alternatives."""
        instructions = []
        forbidden_issues = result.get_issues_by_category(IssueCategory.FORBIDDEN_PATTERN)

        for fi in forbidden_issues:
            # Extract the forbidden pattern from the message
            pattern = fi.message.replace("Forbidden pattern: ", "")
            instructions.append(RepairInstruction(
                action=RepairAction.REPLACE_PATTERN,
                description=f"Remove forbidden pattern: {pattern}",
                priority=10,
                category=IssueCategory.FORBIDDEN_PATTERN,
                search_pattern=pattern,
                replacement="",  # Remove it
                context=fi.suggestion,
            ))

        return instructions

    def _strategy_missing_annotation(
        self, issue: ValidationIssue, result: ValidationResult
    ) -> list[RepairInstruction]:
        """Add missing JUnit5/Mockito annotations."""
        missing_issues = result.get_issues_by_category(IssueCategory.MISSING_ANNOTATION)
        instructions = []

        for mi in missing_issues:
            pattern = mi.message.replace("Missing required pattern: ", "")
            instructions.append(RepairInstruction(
                action=RepairAction.ADD_PATTERN,
                description=f"Add missing annotation: {pattern}",
                priority=15,
                category=IssueCategory.MISSING_ANNOTATION,
                replacement=pattern,
                context=f"Ensure {pattern} is present in the test class. {mi.suggestion or ''}",
            ))

        return instructions

    def _strategy_missing_import(
        self, issue: ValidationIssue, result: ValidationResult
    ) -> list[RepairInstruction]:
        return [RepairInstruction(
            action=RepairAction.ADD_PATTERN,
            description="Add missing import statements",
            priority=5,
            category=IssueCategory.MISSING_IMPORT,
            context="Add all required imports: org.junit.jupiter.api.*, org.mockito.*, static assertions and matchers",
        )]

    def _strategy_missing_aaa(
        self, issue: ValidationIssue, result: ValidationResult
    ) -> list[RepairInstruction]:
        return [RepairInstruction(
            action=RepairAction.MINOR_FIX,
            description="Add AAA pattern comments to each test method",
            priority=30,
            category=IssueCategory.MISSING_AAA,
            context="Add // Arrange, // Act, // Assert comments in each @Test method body",
        )]

    def _strategy_missing_verify(
        self, issue: ValidationIssue, result: ValidationResult
    ) -> list[RepairInstruction]:
        return [RepairInstruction(
            action=RepairAction.ADD_PATTERN,
            description="Add verify() calls for mock interaction verification",
            priority=25,
            category=IssueCategory.MISSING_VERIFY,
            context="After each Act section, add verify(mock).method(args) to verify interactions",
        )]

    def _strategy_missing_display_name(
        self, issue: ValidationIssue, result: ValidationResult
    ) -> list[RepairInstruction]:
        return [RepairInstruction(
            action=RepairAction.MINOR_FIX,
            description="Add @DisplayName to all test methods",
            priority=35,
            category=IssueCategory.MISSING_DISPLAY_NAME,
            context="Each @Test method should have @DisplayName(\"descriptive name\")",
        )]

    def _strategy_missing_test(
        self, issue: ValidationIssue, result: ValidationResult
    ) -> list[RepairInstruction]:
        return [RepairInstruction(
            action=RepairAction.RESTRUCTURE,
            description="No @Test methods found — add test methods",
            priority=5,
            category=IssueCategory.MISSING_TEST,
            context="The class must contain at least one method annotated with @Test",
        )]

    def _strategy_structural(
        self, issue: ValidationIssue, result: ValidationResult
    ) -> list[RepairInstruction]:
        return [RepairInstruction(
            action=RepairAction.RESTRUCTURE,
            description=f"Fix structural issue: {issue.message}",
            priority=1,
            category=IssueCategory.STRUCTURAL,
            context=issue.suggestion or "Fix the class structure to be valid Java",
        )]

    def _strategy_naming(
        self, issue: ValidationIssue, result: ValidationResult
    ) -> list[RepairInstruction]:
        return [RepairInstruction(
            action=RepairAction.MINOR_FIX,
            description="Fix test method naming convention",
            priority=40,
            category=IssueCategory.NAMING,
            context="Use method_WhenCondition_ShouldResult naming pattern",
        )]

    def _strategy_quality(
        self, issue: ValidationIssue, result: ValidationResult
    ) -> list[RepairInstruction]:
        return [RepairInstruction(
            action=RepairAction.MINOR_FIX,
            description=f"Fix quality issue: {issue.message}",
            priority=40,
            category=IssueCategory.QUALITY,
            context=issue.suggestion or "",
        )]

    # ── Anti-pattern repair strategies ───────────────────────────────

    def _strategy_static_call(
        self, issue: ValidationIssue, result: ValidationResult
    ) -> list[RepairInstruction]:
        """Replace direct static calls with MockedStatic try-with-resources."""
        return [RepairInstruction(
            action=RepairAction.RESTRUCTURE,
            description="Wrap static utility calls in MockedStatic try-with-resources",
            priority=5,
            category=IssueCategory.STATIC_CALL_WITHOUT_MOCK,
            context=(
                "CRITICAL: SecurityContextHolder and other static utilities MUST be mocked with "
                "MockedStatic<SecurityContextHolder>. Replace any direct SecurityContextHolder.setContext() "
                "or .getContext() calls with:\n"
                "  try (MockedStatic<SecurityContextHolder> securityMock = mockStatic(SecurityContextHolder.class)) {\n"
                "      securityMock.when(SecurityContextHolder::getContext).thenReturn(securityContext);\n"
                "      when(securityContext.getAuthentication()).thenReturn(authentication);\n"
                "      // ... rest of test ...\n"
                "  }\n"
                "Also add: import org.mockito.MockedStatic;"
            ),
        )]

    def _strategy_inconsistent_values(
        self, issue: ValidationIssue, result: ValidationResult
    ) -> list[RepairInstruction]:
        """Fix inconsistent literal values between setUp and test methods."""
        return [RepairInstruction(
            action=RepairAction.MINOR_FIX,
            description="Fix inconsistent mock/test data values",
            priority=10,
            category=IssueCategory.INCONSISTENT_MOCK_VALUE,
            context=(
                "Ensure all literal values are consistent: if setUp creates data with "
                "'encoded-password', then when(encoder.matches(raw, encoded)) must use "
                "'encoded-password' — not a different string. Trace every literal value "
                "to verify consistency between setUp, when(), verify(), and assertEquals()."
            ),
        )]

    def _strategy_datetime_now(
        self, issue: ValidationIssue, result: ValidationResult
    ) -> list[RepairInstruction]:
        """Replace LocalDateTime.now() in test data with fixed values or captures."""
        return [RepairInstruction(
            action=RepairAction.REPLACE_PATTERN,
            description="Replace LocalDateTime.now()/Instant.now() with fixed values",
            priority=12,
            category=IssueCategory.DATETIME_NOW_IN_TEST,
            search_pattern="LocalDateTime.now()",
            replacement="LocalDateTime.of(2024, 1, 15, 10, 30, 0)",
            context=(
                "Never use .now() in test expected values — the value at construction time will "
                "differ from the runtime value inside the method under test. Options:\n"
                "1. Use a fixed value: LocalDateTime.of(2024, 1, 15, 10, 30, 0)\n"
                "2. Use ArgumentCaptor to capture the actual value\n"
                "3. Use any(LocalDateTime.class) in verify()/when() matchers\n"
                "4. If the method creates the timestamp internally, verify with ArgumentCaptor"
            ),
        )]

    def _strategy_missing_mock_field(
        self, issue: ValidationIssue, result: ValidationResult
    ) -> list[RepairInstruction]:
        """Add missing @Mock fields for service dependencies."""
        return [RepairInstruction(
            action=RepairAction.ADD_PATTERN,
            description="Add missing @Mock fields for all service dependencies",
            priority=3,
            category=IssueCategory.MISSING_MOCK_FIELD,
            context=(
                "CRITICAL: @InjectMocks will set unmocked dependencies to null, causing NPE. "
                "List ALL constructor parameters of the service under test and add @Mock for each. "
                "Check the service source code constructor or @RequiredArgsConstructor fields."
            ),
        )]

    def _strategy_raw_value(
        self, issue: ValidationIssue, result: ValidationResult
    ) -> list[RepairInstruction]:
        """Replace raw literals with proper enum constants."""
        return [RepairInstruction(
            action=RepairAction.MINOR_FIX,
            description="Replace raw int/string values with enum constants",
            priority=20,
            category=IssueCategory.RAW_VALUE_INSTEAD_OF_ENUM,
            context=(
                "If source code uses enum constants like UserStatus.ACTIVE, tests must also use "
                "UserStatus.ACTIVE — not integer 1 or string 'ACTIVE'. Check all when()/verify() "
                "arguments and assertEquals() expectations for raw values that should be enums."
            ),
        )]

    def _strategy_chained_verify(
        self, issue: ValidationIssue, result: ValidationResult
    ) -> list[RepairInstruction]:
        """Fix verify() calls that chain through static utility methods."""
        return [RepairInstruction(
            action=RepairAction.REPLACE_PATTERN,
            description="Replace chained verify on SecurityContextHolder with direct mock verify",
            priority=8,
            category=IssueCategory.CHAINED_VERIFY_ON_STATIC,
            search_pattern="verify(SecurityContextHolder.getContext().getAuthentication())",
            replacement="verify(authentication)",
            context=(
                "Inside MockedStatic block, verify on the mock variable directly:\n"
                "  verify(authentication).getName();  // CORRECT\n"
                "NOT: verify(SecurityContextHolder.getContext().getAuthentication()).getName()  // WRONG\n"
                "The mock objects (authentication, securityContext) should be declared as local variables "
                "before the try-with-resources block."
            ),
        )]

    def _strategy_domain_guessing(
        self, issue: ValidationIssue, result: ValidationResult
    ) -> list[RepairInstruction]:
        """Fix overly verbose setUp that guesses domain type fields."""
        return [RepairInstruction(
            action=RepairAction.RESTRUCTURE,
            description="Replace guessed domain builders with mock(Type.class) + stub accessors",
            priority=7,
            category=IssueCategory.DOMAIN_TYPE_GUESSING,
            context=(
                "The setUp method has too many builder/setter calls, suggesting domain type fields "
                "are being guessed. Replace with:\n"
                "  User user = mock(User.class);\n"
                "  when(user.id()).thenReturn(UUID.fromString(userId));\n"
                "  when(user.email()).thenReturn(email);\n"
                "Only stub the accessors that the method under test actually calls. "
                "Move domain object creation inside each test method for self-containment."
            ),
        )]

    def _strategy_wrong_construction(
        self, issue: ValidationIssue, result: ValidationResult
    ) -> list[RepairInstruction]:
        """Fix wrong construction pattern (e.g., .builder() on record without @Builder)."""
        return [RepairInstruction(
            action=RepairAction.REPLACE,
            description=issue.suggestion or "Fix construction pattern to match actual class definition",
            priority=9,  # High priority — causes compilation failure
            category=IssueCategory.WRONG_CONSTRUCTION_PATTERN,
            context=(
                f"COMPILATION ERROR: {issue.message}\n"
                "The generated code uses a construction pattern that does not exist for this type. "
                "Check the Domain Types section in context for the exact construction hint.\n"
                "- Records without @Builder: use new RecordName(field1, field2, ...)\n"
                "- Records with @Builder: use RecordName.builder().field(value).build()\n"
                "- If fields are wrong, use ONLY the fields listed in the context.\n"
                "- If the type source is unknown, use mock(Type.class) + stub accessors."
            ),
        )]
