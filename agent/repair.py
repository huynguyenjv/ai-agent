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
