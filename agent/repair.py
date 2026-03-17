"""
Repair Strategies — targeted fix strategies with root-cause reasoning,
failure memory, and escalating repair levels.

Architecture:
  Level 1 — Targeted Fix:   Category-based instructions (existing)
  Level 2 — Reasoning Fix:  LLM explains WHY the issue occurred, then fixes
  Level 3 — Regenerate:     Full regeneration with accumulated failure context

Key improvements over naive repair:
  - FailureMemory: tracks what was tried and what persisted → avoids repeating
  - RepairReasoningEngine: asks LLM to reason about root-cause before fixing
  - Escalation: each attempt uses a progressively deeper strategy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import structlog

from .validation_schema import ValidationResult, ValidationIssue, IssueCategory, IssueSeverity

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════
# Failure Memory — tracks what was tried and what failed
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RepairAttemptRecord:
    """Record of a single repair attempt."""
    attempt_number: int
    strategy: str                        # "targeted", "reasoning", "regenerate"
    instructions_used: list[str]         # Human-readable list of instructions given
    issues_before: list[str]             # Issues that existed before this attempt
    issues_after: list[str]              # Issues that remained after this attempt
    issues_fixed: list[str]             # Issues that were resolved
    issues_introduced: list[str]         # NEW issues introduced by the repair
    reasoning: Optional[str] = None      # LLM reasoning (if Level 2+)


class FailureMemory:
    """Tracks repair history so we don't repeat failed fixes.

    Used by RepairStrategySelector to build smarter prompts on each attempt:
      - "Previous attempts tried X but issue Y persists"
      - "Do NOT repeat: [list of failed approaches]"
      - "Issues Z were introduced by the last repair — revert those changes"
    """

    def __init__(self):
        self._attempts: list[RepairAttemptRecord] = []

    @property
    def attempt_count(self) -> int:
        return len(self._attempts)

    @property
    def attempts(self) -> list[RepairAttemptRecord]:
        return list(self._attempts)

    def record_attempt(
        self,
        attempt_number: int,
        strategy: str,
        instructions_used: list[str],
        issues_before: list[str],
        issues_after: list[str],
        reasoning: Optional[str] = None,
    ) -> RepairAttemptRecord:
        """Record a repair attempt and calculate diffs."""
        before_set = set(issues_before)
        after_set = set(issues_after)

        record = RepairAttemptRecord(
            attempt_number=attempt_number,
            strategy=strategy,
            instructions_used=instructions_used,
            issues_before=issues_before,
            issues_after=issues_after,
            issues_fixed=sorted(before_set - after_set),
            issues_introduced=sorted(after_set - before_set),
            reasoning=reasoning,
        )
        self._attempts.append(record)

        logger.info(
            "Repair attempt recorded",
            attempt=attempt_number,
            strategy=strategy,
            fixed=len(record.issues_fixed),
            introduced=len(record.issues_introduced),
            remaining=len(record.issues_after),
        )
        return record

    def get_persistent_issues(self) -> list[str]:
        """Issues that have persisted through ALL attempts."""
        if not self._attempts:
            return []
        # Issues present in the LAST attempt's 'after' that were also in the first attempt's 'before'
        first_issues = set(self._attempts[0].issues_before)
        last_issues = set(self._attempts[-1].issues_after)
        return sorted(first_issues & last_issues)

    def get_failed_instructions(self) -> list[str]:
        """Instructions that were tried but did NOT resolve the target issues."""
        failed: list[str] = []
        for record in self._attempts:
            if record.issues_after:
                # These instructions didn't fully resolve everything
                failed.extend(record.instructions_used)
        return failed

    def get_regression_warnings(self) -> list[str]:
        """Issues that were INTRODUCED by previous repairs (regressions)."""
        regressions: list[str] = []
        for record in self._attempts:
            regressions.extend(record.issues_introduced)
        return regressions

    def build_memory_section(self) -> str:
        """Build a prompt section summarizing repair history.

        This is injected into the repair prompt so the LLM understands:
          1. What was already tried
          2. What didn't work
          3. What regressions to avoid
        """
        if not self._attempts:
            return ""

        lines = ["## Previous Repair Attempts (DO NOT repeat failed approaches)\n"]

        for record in self._attempts:
            lines.append(f"### Attempt {record.attempt_number} ({record.strategy})")
            if record.reasoning:
                lines.append(f"Reasoning: {record.reasoning[:200]}")
            lines.append(f"Instructions tried: {', '.join(record.instructions_used[:5])}")
            if record.issues_fixed:
                lines.append(f"✅ Fixed: {', '.join(record.issues_fixed[:3])}")
            if record.issues_introduced:
                lines.append(f"❌ REGRESSIONS introduced: {', '.join(record.issues_introduced[:3])}")
            if record.issues_after:
                lines.append(f"⚠️ Still broken: {', '.join(record.issues_after[:3])}")
            lines.append("")

        persistent = self.get_persistent_issues()
        if persistent:
            lines.append(f"**PERSISTENT ISSUES (survived all attempts):** {', '.join(persistent[:5])}")
            lines.append("These need a fundamentally different approach.\n")

        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all memory (e.g., for a new generation)."""
        self._attempts.clear()


# ═══════════════════════════════════════════════════════════════════════
# Repair Reasoning Engine
# ═══════════════════════════════════════════════════════════════════════

class RepairReasoningEngine:
    """Asks the LLM to reason about WHY validation failed before fixing.

    Instead of blindly saying "fix this", we first ask:
      "Given this test code and these validation errors, explain:
       1. What is the ROOT CAUSE of each error?
       2. What specific code changes would fix each error?
       3. Are any errors related (fixing one fixes others)?"

    The reasoning output is then included in the repair prompt.
    """

    @staticmethod
    def build_reasoning_prompt(
        test_code: str,
        validation_issues: list[str],
        memory: Optional[FailureMemory] = None,
    ) -> tuple[str, str]:
        """Build a prompt that asks the LLM to reason about failures.

        Returns (system_prompt, user_prompt).
        """
        system_prompt = (
            "You are a Java test debugging expert. Your job is to analyze "
            "validation errors in JUnit5/Mockito test code and explain the ROOT CAUSE "
            "of each error. Output ONLY a JSON object, no code.\n\n"
            "For each error, determine:\n"
            "1. root_cause: WHY this error exists (not just what it is)\n"
            "2. fix_action: Specific code change needed\n"
            "3. related_errors: Other errors that share the same root cause\n"
            "4. risk: What could go wrong if this fix is applied naively"
        )

        memory_section = ""
        if memory and memory.attempt_count > 0:
            memory_section = (
                "\n## CRITICAL: Previous repair attempts FAILED\n"
                + memory.build_memory_section()
                + "\nYou MUST propose a DIFFERENT approach than what was already tried.\n"
            )

        user_prompt = f"""Analyze these validation errors and explain the ROOT CAUSE of each.

## Test Code
```java
{test_code}
```

## Validation Errors
{chr(10).join(f'- {issue}' for issue in validation_issues)}
{memory_section}

## Output JSON Schema
{{
    "analyses": [
        {{
            "error": "the error message",
            "root_cause": "why this error exists",
            "fix_action": "specific code change to fix it",
            "related_errors": ["other error messages with same root cause"],
            "risk": "what could go wrong with this fix"
        }}
    ],
    "overall_strategy": "high-level approach to fix all issues at once",
    "fix_order": ["error1", "error2", "...in order they should be fixed"]
}}

Output ONLY valid JSON."""

        return system_prompt, user_prompt

    @staticmethod
    def parse_reasoning(response_content: str) -> Optional[dict]:
        """Parse the reasoning JSON from LLM response."""
        import json
        import re

        # Try code block
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response_content, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            # Try raw JSON
            match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if match:
                text = match.group(0).strip()
            else:
                return None

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse reasoning JSON")
            return None


# ═══════════════════════════════════════════════════════════════════════
# Repair Strategy Types
# ═══════════════════════════════════════════════════════════════════════

class RepairAction(str, Enum):
    """Type of repair action to take."""
    REPLACE_PATTERN = "replace_pattern"    # Replace forbidden with correct
    ADD_PATTERN = "add_pattern"            # Add missing pattern
    RESTRUCTURE = "restructure"            # Major restructuring needed
    MINOR_FIX = "minor_fix"               # Small fix (naming, comments)
    REGENERATE = "regenerate"              # Full re-generation needed


class RepairLevel(str, Enum):
    """Escalation level of repair strategy."""
    TARGETED = "targeted"                  # Level 1: Category-based quick fix
    REASONING = "reasoning"                # Level 2: LLM reasons about WHY, then fixes
    REGENERATE = "regenerate"              # Level 3: Full regeneration with failure context


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
    strategy: str = "targeted"  # "targeted", "reasoning", or "regenerate"
    level: RepairLevel = RepairLevel.TARGETED
    reasoning: Optional[dict] = None       # LLM reasoning output (Level 2+)
    memory_section: str = ""               # Failure memory context

    @property
    def has_structural_issues(self) -> bool:
        return any(i.category == IssueCategory.STRUCTURAL for i in self.original_issues)

    @property
    def instruction_count(self) -> int:
        return len(self.instructions)

    def get_repair_prompt_section(self) -> str:
        """Generate the repair instruction section for the LLM prompt."""
        lines = []

        # Include failure memory context (tells LLM what was already tried)
        if self.memory_section:
            lines.append(self.memory_section)

        # Include LLM reasoning (Level 2+)
        if self.reasoning:
            lines.append("## Root Cause Analysis\n")
            overall = self.reasoning.get("overall_strategy", "")
            if overall:
                lines.append(f"**Strategy:** {overall}\n")
            for analysis in self.reasoning.get("analyses", [])[:5]:
                lines.append(f"- **{analysis.get('error', 'unknown')}**")
                lines.append(f"  Root cause: {analysis.get('root_cause', 'unknown')}")
                lines.append(f"  Fix: {analysis.get('fix_action', 'unknown')}")
                risk = analysis.get("risk", "")
                if risk:
                    lines.append(f"  ⚠️ Risk: {risk}")
                lines.append("")

            fix_order = self.reasoning.get("fix_order", [])
            if fix_order:
                lines.append(f"**Fix order:** {' → '.join(fix_order[:5])}\n")

        # Include targeted instructions
        if self.instructions:
            lines.append("## Repair Instructions (MUST FOLLOW IN ORDER)\n")
        for i, inst in enumerate(sorted(self.instructions, key=lambda x: x.priority), 1):
            lines.append(f"### Fix {i}: {inst.description}")
            if inst.search_pattern and inst.replacement:
                lines.append(f"- Find: `{inst.search_pattern}`")
                lines.append(f"- Replace with: `{inst.replacement}`")
            if inst.context:
                lines.append(f"- Note: {inst.context}")
            lines.append("")

        return "\n".join(lines) if lines else ""


# ═══════════════════════════════════════════════════════════════════════
# Strategy Selector
# ═══════════════════════════════════════════════════════════════════════

class RepairStrategySelector:
    """Selects and builds repair strategies with escalation + memory.

    Escalation levels:
      Attempt 1 → Level 1 (TARGETED): Category-based instructions
      Attempt 2 → Level 2 (REASONING): Ask LLM to reason about WHY
      Attempt 3 → Level 3 (REGENERATE): Full regeneration with failure context
    """

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

    def determine_level(
        self,
        attempt_number: int,
        max_attempts: int,
        memory: Optional[FailureMemory] = None,
    ) -> RepairLevel:
        """Determine repair level based on attempt number and history.

        Escalation strategy:
          Attempt 1 → TARGETED (quick category-based fix)
          Attempt 2 → REASONING (LLM reasons about WHY, then fixes)
          Attempt 3+ → REGENERATE (full regen with all failure context)

        Special: if memory shows regressions, escalate faster.
        """
        # Fast-track to regenerate if previous attempts introduced regressions
        if memory and memory.get_regression_warnings():
            regression_count = len(memory.get_regression_warnings())
            if regression_count >= 2:
                logger.info(
                    "Escalating to REGENERATE due to regressions",
                    regressions=regression_count,
                )
                return RepairLevel.REGENERATE

        if attempt_number >= max_attempts:
            return RepairLevel.REGENERATE
        elif attempt_number >= 2:
            return RepairLevel.REASONING
        else:
            return RepairLevel.TARGETED

    def build_repair_plan(
        self,
        validation_result: ValidationResult,
        attempt_number: int = 1,
        max_attempts: int = 3,
        memory: Optional[FailureMemory] = None,
        reasoning: Optional[dict] = None,
    ) -> RepairPlan:
        """Build a repair plan from validation results.

        Uses escalation levels:
          Level 1 (TARGETED): Category-based instructions
          Level 2 (REASONING): Includes LLM root-cause analysis
          Level 3 (REGENERATE): Full regeneration with failure context
        """
        level = self.determine_level(attempt_number, max_attempts, memory)

        plan = RepairPlan(
            original_issues=[i for i in validation_result.issues if i.severity != IssueSeverity.INFO],
            attempt_number=attempt_number,
            level=level,
            reasoning=reasoning,
            memory_section=memory.build_memory_section() if memory else "",
        )

        if level == RepairLevel.REGENERATE:
            plan.strategy = "regenerate"
            persistent = memory.get_persistent_issues() if memory else []
            context_lines = [
                "Start from scratch. Follow ALL rules strictly.",
            ]
            if persistent:
                context_lines.append(
                    f"Previous attempts failed to fix: {'; '.join(persistent[:5])}. "
                    "Use a FUNDAMENTALLY DIFFERENT approach."
                )
            if memory:
                failed = memory.get_failed_instructions()
                if failed:
                    context_lines.append(
                        f"DO NOT repeat these approaches: {'; '.join(failed[:5])}"
                    )

            plan.instructions.append(RepairInstruction(
                action=RepairAction.REGENERATE,
                description="Full regeneration — previous repair attempts failed",
                priority=0,
                category=IssueCategory.STRUCTURAL,
                context=" ".join(context_lines),
            ))
            return plan

        # Level 1 or 2: build targeted instructions
        plan.strategy = level.value
        seen_categories: set[IssueCategory] = set()

        for issue in validation_result.issues:
            if issue.severity == IssueSeverity.INFO:
                continue

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
            level=level.value,
            strategy=plan.strategy,
            instructions=len(plan.instructions),
            error_categories=list(seen_categories),
            has_reasoning=reasoning is not None,
            has_memory=memory is not None and memory.attempt_count > 0,
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
            pattern = fi.message.replace("Forbidden pattern: ", "")
            instructions.append(RepairInstruction(
                action=RepairAction.REPLACE_PATTERN,
                description=f"Remove forbidden pattern: {pattern}",
                priority=10,
                category=IssueCategory.FORBIDDEN_PATTERN,
                search_pattern=pattern,
                replacement="",
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
            context='Each @Test method should have @DisplayName("descriptive name")',
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
                "  }\n"
                "Also add: import org.mockito.MockedStatic;"
            ),
        )]

    def _strategy_inconsistent_values(
        self, issue: ValidationIssue, result: ValidationResult
    ) -> list[RepairInstruction]:
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
        return [RepairInstruction(
            action=RepairAction.REPLACE_PATTERN,
            description="Replace LocalDateTime.now()/Instant.now() with fixed values",
            priority=12,
            category=IssueCategory.DATETIME_NOW_IN_TEST,
            search_pattern="LocalDateTime.now()",
            replacement="LocalDateTime.of(2024, 1, 15, 10, 30, 0)",
            context=(
                "Never use .now() in test expected values. Options:\n"
                "1. Use a fixed value: LocalDateTime.of(2024, 1, 15, 10, 30, 0)\n"
                "2. Use ArgumentCaptor to capture the actual value\n"
                "3. Use any(LocalDateTime.class) in verify()/when() matchers"
            ),
        )]

    def _strategy_missing_mock_field(
        self, issue: ValidationIssue, result: ValidationResult
    ) -> list[RepairInstruction]:
        return [RepairInstruction(
            action=RepairAction.ADD_PATTERN,
            description="Add missing @Mock fields for all service dependencies",
            priority=3,
            category=IssueCategory.MISSING_MOCK_FIELD,
            context=(
                "CRITICAL: @InjectMocks will set unmocked dependencies to null, causing NPE. "
                "List ALL constructor parameters of the service under test and add @Mock for each."
            ),
        )]

    def _strategy_raw_value(
        self, issue: ValidationIssue, result: ValidationResult
    ) -> list[RepairInstruction]:
        return [RepairInstruction(
            action=RepairAction.MINOR_FIX,
            description="Replace raw int/string values with enum constants",
            priority=20,
            category=IssueCategory.RAW_VALUE_INSTEAD_OF_ENUM,
            context=(
                "If source code uses enum constants like UserStatus.ACTIVE, tests must also use "
                "UserStatus.ACTIVE — not integer 1 or string 'ACTIVE'."
            ),
        )]

    def _strategy_chained_verify(
        self, issue: ValidationIssue, result: ValidationResult
    ) -> list[RepairInstruction]:
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
                "NOT: verify(SecurityContextHolder.getContext().getAuthentication()).getName()  // WRONG"
            ),
        )]

    def _strategy_domain_guessing(
        self, issue: ValidationIssue, result: ValidationResult
    ) -> list[RepairInstruction]:
        return [RepairInstruction(
            action=RepairAction.RESTRUCTURE,
            description="Replace guessed domain builders with mock(Type.class) + stub accessors",
            priority=7,
            category=IssueCategory.DOMAIN_TYPE_GUESSING,
            context=(
                "The setUp method has too many builder/setter calls suggesting domain type fields "
                "are being guessed. Replace with:\n"
                "  User user = mock(User.class);\n"
                "  when(user.id()).thenReturn(UUID.fromString(userId));\n"
                "Only stub the accessors that the method under test actually calls."
            ),
        )]

    def _strategy_wrong_construction(
        self, issue: ValidationIssue, result: ValidationResult
    ) -> list[RepairInstruction]:
        return [RepairInstruction(
            action=RepairAction.REPLACE_PATTERN,
            description=issue.suggestion or "Fix construction pattern to match actual class definition",
            priority=9,
            category=IssueCategory.WRONG_CONSTRUCTION_PATTERN,
            context=(
                f"COMPILATION ERROR: {issue.message}\n"
                "Check the Domain Types section for exact construction patterns.\n"
                "- Records without @Builder: use new RecordName(field1, field2, ...)\n"
                "- Records with @Builder: use RecordName.builder().field(value).build()\n"
                "- If the type source is unknown, use mock(Type.class) + stub accessors."
            ),
        )]
