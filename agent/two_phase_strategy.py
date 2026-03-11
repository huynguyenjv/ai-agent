"""
Two-Phase Generation Strategy — Analyze → Generate → Self-Review → Validate → Repair.

Architecture (iterative):

  Phase 1 (Analysis):
    - LLM analyzes service source code → structured JSON plan
    - Identifies dependencies, domain types, exceptions
    - NO code generation

  Phase 2 (Enhanced Generation with iterative refinement):
    - Enriches RAG context by searching for missing types
    - Generates COMPLETE test class with analysis + registry context
    - **Self-Review**: LLM reviews its own output for issues
    - **Validate**: Runs validation pipeline IN THE LOOP
    - **Smart Repair**: Uses escalating repair strategy:
        Level 1 — Targeted (category-based)
        Level 2 — Reasoning (LLM explains WHY, then fixes)
        Level 3 — Regenerate (full regen with failure memory)
    - FailureMemory tracks what was tried → prevents repeating failed fixes

Key insight: Phase 2 is a BETTER single-pass with richer context + iterative refinement.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import structlog

from .analysis_prompt import (
    AnalysisPromptBuilder,
    AnalysisResult,
    extract_json_from_response,
)
from .prompt import PromptBuilder
from .rules import TestRules
from .validation import ValidationPipeline, ValidationResult
from .repair import (
    RepairStrategySelector,
    RepairReasoningEngine,
    FailureMemory,
    RepairLevel,
)
from .events import Event, EventType, get_event_bus
from rag.client import RAGClient
from rag.schema import CodeChunk
from vllm.client import VLLMClient

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TwoPhaseConfig:
    """Configuration for Two-Phase Generation."""

    # Phase 1 settings
    analysis_temperature: float = 0.1       # Lower temp for structured output
    analysis_max_tokens: int = 2000         # Max tokens for analysis

    # Phase 2 settings
    generation_temperature: float = 0.2     # Standard temp for code gen
    generation_max_tokens: int = 4000       # Max tokens for complete class

    # Self-review settings
    self_review_enabled: bool = True        # Enable LLM self-review after gen
    self_review_temperature: float = 0.1    # Low temp for review accuracy
    self_review_max_tokens: int = 1500      # Max tokens for review output

    # Repair settings (escalating)
    max_repair_attempts: int = 3            # 3 levels: targeted→reasoning→regen
    reasoning_enabled: bool = True          # Enable LLM reasoning in Level 2
    reasoning_max_tokens: int = 1500        # Max tokens for reasoning output

    # Fallback settings
    fallback_to_single_pass: bool = True    # Fallback if two-phase fails


class TwoPhaseState(str, Enum):
    """States of the two-phase generation process."""
    IDLE = "idle"
    ANALYZING = "analyzing"
    ENRICHING = "enriching"
    GENERATING = "generating"
    SELF_REVIEWING = "self_reviewing"
    VALIDATING = "validating"
    REPAIRING = "repairing"
    COMPLETED = "completed"
    FAILED = "failed"


# ═══════════════════════════════════════════════════════════════════════
# Result Types
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TwoPhaseResult:
    """Complete result of two-phase generation."""
    success: bool
    test_code: Optional[str] = None
    class_name: str = ""

    # Phase results
    analysis_result: Optional[AnalysisResult] = None
    extra_chunks_found: int = 0

    # Validation
    validation_passed: bool = True
    validation_issues: list[str] = field(default_factory=list)
    validation_result: Optional[ValidationResult] = None

    # Self-review
    self_review_issues: list[str] = field(default_factory=list)
    self_review_applied: bool = False

    # Repair metrics
    total_tokens_used: int = 0
    analysis_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0
    repair_attempts: int = 0
    repair_levels_used: list[str] = field(default_factory=list)  # ["targeted", "reasoning"]

    # Error info
    error: Optional[str] = None
    failed_phase: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
# Two-Phase Strategy
# ═══════════════════════════════════════════════════════════════════════

class TwoPhaseStrategy:
    """Two-Phase Generation: Analyze → Enrich → Generate → Self-Review → Validate → Repair.

    Phase 2 generates the COMPLETE test class in a single LLM call,
    using the same PromptBuilder as single-pass but with enriched context.
    After generation, an iterative refinement loop runs:

      1. Self-Review: LLM checks its own output for issues
      2. Validate: Structural/pattern validation
      3. Repair (if needed): Escalating strategy with failure memory

    Usage::

        strategy = TwoPhaseStrategy(
            vllm_client=vllm,
            domain_registry=registry,
            prompt_builder=prompt_builder,
            test_rules=test_rules,
            rag_client=rag_client,
        )

        # Non-streaming (full iterative loop)
        result = strategy.generate(source_code, class_name, file_path, rag_chunks)

        # Streaming (orchestrator drives the loop)
        analysis = strategy.analyze(source_code, class_name, file_path)
        extra = strategy.search_missing_context(analysis, rag_chunks, source_code)
        sys_prompt, usr_prompt = strategy.build_generation_prompt(
            analysis, class_name, file_path, rag_chunks + extra, source_code)
        # then stream with vllm.stream_generate(sys_prompt, usr_prompt)
        # then call strategy.self_review(code) + strategy.iterative_repair(...)
    """

    def __init__(
        self,
        vllm_client: VLLMClient,
        domain_registry,  # DomainTypeRegistry | None
        prompt_builder: PromptBuilder,
        test_rules: TestRules,
        rag_client: RAGClient,
        config: Optional[TwoPhaseConfig] = None,
    ):
        self.vllm = vllm_client
        self.registry = domain_registry
        self.prompt_builder = prompt_builder
        self.test_rules = test_rules
        self.rag = rag_client
        self.config = config or TwoPhaseConfig()

        self.analysis_builder = AnalysisPromptBuilder()
        self.validator = ValidationPipeline()
        self.repair_selector = RepairStrategySelector()
        self.reasoning_engine = RepairReasoningEngine()
        self.event_bus = get_event_bus()

        self._state = TwoPhaseState.IDLE

    @property
    def state(self) -> TwoPhaseState:
        return self._state

    # ═══════════════════════════════════════════════════════════════════
    # Public API — called by both non-streaming generate() and
    #              streaming orchestrator
    # ═══════════════════════════════════════════════════════════════════

    def analyze(
        self,
        source_code: str,
        class_name: str,
        file_path: str,
    ) -> Optional[AnalysisResult]:
        """Phase 1: Analyze the service structure.

        Returns structured AnalysisResult or None on failure.
        This is a standalone public method so orchestrator can call it
        and stream progress separately.
        """
        self._state = TwoPhaseState.ANALYZING

        system_prompt = self.analysis_builder.build_system_prompt()
        user_prompt = self.analysis_builder.build_analysis_prompt(
            source_code=source_code,
            class_name=class_name,
            file_path=file_path,
        )

        logger.debug("Phase 1: Running analysis", class_name=class_name)

        response = self.vllm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.analysis_temperature,
            max_tokens=self.config.analysis_max_tokens,
        )

        if not response.success:
            logger.error("Analysis LLM call failed", error=response.error)
            self._state = TwoPhaseState.FAILED
            return None

        try:
            json_str = extract_json_from_response(response.content)
            result = AnalysisResult.from_json(json_str)

            logger.info(
                "Phase 1 complete",
                class_name=class_name,
                methods=len(result.methods),
                domain_types=result.all_domain_types,
                dependencies=result.constructor_dependencies,
                complexity=result.complexity_level,
            )
            return result

        except Exception as e:
            logger.error("Failed to parse analysis result", error=str(e))
            self._state = TwoPhaseState.FAILED
            return None

    def search_missing_context(
        self,
        analysis_result: AnalysisResult,
        rag_chunks: list[CodeChunk],
        source_code: str,
        collection_name: Optional[str] = None,
    ) -> list[CodeChunk]:
        """Search RAG for types/exceptions not yet in rag_chunks.

        Finds:
          1. Domain types from analysis not in RAG results
          2. Dependencies from analysis not in RAG results
          3. Exception classes referenced in source code

        Returns a list of additional CodeChunks to merge into rag_chunks.
        """
        self._state = TwoPhaseState.ENRICHING
        existing_names = {c.class_name for c in rag_chunks}
        types_to_find: set[str] = set()

        # 1. Domain types from analysis
        for t in analysis_result.all_domain_types:
            if t not in existing_names:
                types_to_find.add(t)

        # 2. Dependencies from analysis
        for d in analysis_result.constructor_dependencies:
            if d not in existing_names:
                types_to_find.add(d)

        # 3. Exception classes from source code (generic, no hardcoding)
        for match in re.finditer(r'\bthrow\s+new\s+(\w+(?:Exception|Error))\b', source_code):
            name = match.group(1)
            if name not in existing_names:
                types_to_find.add(name)
        for match in re.finditer(r'\bcatch\s*\(\s*(\w+(?:Exception|Error))\b', source_code):
            name = match.group(1)
            if name not in existing_names:
                types_to_find.add(name)
        for match in re.finditer(r'import\s+[\w.]+\.(\w+(?:Exception|Error))\s*;', source_code):
            name = match.group(1)
            if name not in existing_names:
                types_to_find.add(name)

        if not types_to_find:
            logger.debug("No missing types to search for")
            return []

        logger.info(
            "Searching for missing types",
            types=sorted(types_to_find),
            count=len(types_to_find),
        )

        extra_chunks: list[CodeChunk] = []

        def _fetch_one(type_name: str) -> Optional[CodeChunk]:
            try:
                result = self.rag.search_by_class(
                    class_name=type_name,
                    top_k=1,
                    include_dependencies=False,
                    collection_name=collection_name,
                )
                return result.chunks[0] if result.chunks else None
            except Exception as e:
                logger.debug("RAG search failed for type", type_name=type_name, error=str(e))
                return None

        # Parallel search
        with ThreadPoolExecutor(max_workers=min(5, len(types_to_find))) as executor:
            future_map = {executor.submit(_fetch_one, t): t for t in types_to_find}
            for future in as_completed(future_map):
                type_name = future_map[future]
                chunk = future.result()
                if chunk and chunk.class_name not in existing_names:
                    extra_chunks.append(chunk)
                    existing_names.add(chunk.class_name)
                    logger.debug("Found missing type", type_name=type_name)
                elif not chunk:
                    logger.debug("Type not found in index", type_name=type_name)

        logger.info(
            "Missing context search complete",
            searched=len(types_to_find),
            found=len(extra_chunks),
        )
        return extra_chunks

    def build_generation_prompt(
        self,
        analysis_result: AnalysisResult,
        class_name: str,
        file_path: str,
        rag_chunks: list[CodeChunk],
        source_code: str,
        collection_name: Optional[str] = None,
    ) -> tuple[str, str]:
        """Build enhanced prompt for Phase 2 (complete test class generation).

        Uses the existing PromptBuilder (same as single-pass) but enriches
        the prompt with:
          1. Analysis summary (methods, scenarios, complexity)
          2. Registry construction patterns (if available)
          3. Source code (always included explicitly)

        Returns (system_prompt, user_prompt).
        """
        # Build analysis summary section
        analysis_section = self._build_analysis_section(analysis_result)

        # Build task description with source code + analysis
        task_desc = (
            f"Generate unit tests for `{class_name}`\n\n"
            f"## Source Code Under Test\n```java\n{source_code}\n```\n\n"
            f"{analysis_section}"
        )

        # Build registry context for domain types
        registry_context = ""
        if self.registry and analysis_result.all_domain_types:
            try:
                registry_context = self.registry.get_prompt_section(
                    class_names=analysis_result.all_domain_types,
                    collection_name=collection_name,
                )
            except Exception as e:
                logger.warning("Registry lookup failed", error=str(e))

        # Use existing PromptBuilder for consistent prompt quality
        system_prompt = self.prompt_builder.build_system_prompt()

        if registry_context:
            user_prompt = self.prompt_builder.build_registry_enhanced_prompt(
                class_name=class_name,
                file_path=file_path,
                rag_chunks=rag_chunks,
                registry_context=registry_context,
                task_description=task_desc,
            )
        else:
            user_prompt = self.prompt_builder.build_test_generation_prompt(
                class_name=class_name,
                file_path=file_path,
                rag_chunks=rag_chunks,
                task_description=task_desc,
            )

        return system_prompt, user_prompt

    def self_review(
        self,
        test_code: str,
        source_code: str,
        class_name: str,
        rag_chunks: list[CodeChunk],
    ) -> tuple[str, list[str]]:
        """Self-Review: Ask LLM to review its own generated test code.

        The LLM checks for:
          1. Missing imports for types used in the test
          2. Wrong method names (compared to source code)
          3. Missing @BeforeEach setup
          4. Incorrect mock configurations
          5. Hallucinated methods or constructors

        Returns (corrected_code, issues_found).
        If no issues found, returns (original_code, []).
        """
        if not self.config.self_review_enabled:
            return test_code, []

        self._state = TwoPhaseState.SELF_REVIEWING

        system_prompt = (
            "You are a Java test code reviewer. Review the generated test code "
            "against the original source code and fix ALL issues.\n\n"
            "CHECK FOR:\n"
            "1. Missing imports — every class used in test must be imported\n"
            "2. Wrong method names — method calls must EXACTLY match the source code\n"
            "3. Wrong constructor/builder patterns — check source for actual constructors\n"
            "4. Missing @BeforeEach setUp — shared test data should be in setUp()\n"
            "5. Hallucinated methods — don't call methods that don't exist in source\n"
            "6. Missing verify() — every when() mock should have a matching verify()\n\n"
            "Output the COMPLETE corrected test class. If no changes needed, "
            "output the original code unchanged."
        )

        # Extract dependency info from RAG chunks for the reviewer
        dep_info_parts = []
        for chunk in rag_chunks[:8]:  # Top 8 chunks
            if chunk.class_name and chunk.fully_qualified_name:
                dep_info_parts.append(
                    f"- {chunk.class_name}: {chunk.fully_qualified_name} "
                    f"(type={chunk.type}, layer={chunk.layer})"
                )

        dep_context = "\n".join(dep_info_parts) if dep_info_parts else "No dependency info available"

        user_prompt = f"""Review this generated test code against the source code.

## Source Code Under Test
```java
{source_code}
```

## Available Types (from RAG index — use these FQNs for imports)
{dep_context}

## Generated Test Code to Review
```java
{test_code}
```

## Review Checklist
1. Are ALL imports correct? (check each used class has an import)
2. Do method calls match EXACTLY what's in the source code?
3. Are domain type constructors/builders correct?
4. Is there a @BeforeEach setUp() for shared variables?
5. Are there any hallucinated (non-existent) methods?
6. Is every when() mock paired with a verify()?

If you find issues, output the CORRECTED complete test class.
If no issues, output the original test class unchanged.
Start your response with the issues found as comments, then the corrected code."""

        response = self.vllm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.self_review_temperature,
            max_tokens=self.config.self_review_max_tokens + 2000,  # Extra for full class
        )

        if not response.success:
            logger.warning("Self-review LLM call failed", error=response.error)
            return test_code, []

        # Extract issues found and corrected code
        review_content = response.content
        issues_found = self._extract_review_issues(review_content)
        corrected_code = self._extract_code(review_content)

        if issues_found:
            logger.info(
                "Self-review found issues",
                issues_count=len(issues_found),
                issues=issues_found[:5],
            )
        else:
            logger.info("Self-review: no issues found")

        self._publish_event(EventType.STEP_COMPLETED, {
            "phase": "self_review",
            "issues_found": len(issues_found),
            "code_changed": corrected_code != test_code,
        })

        return corrected_code, issues_found

    def iterative_repair(
        self,
        test_code: str,
        validation_result: ValidationResult,
        rag_chunks: list[CodeChunk],
        source_code: str = "",
    ) -> tuple[str, int, list[str]]:
        """Iterative repair with escalating strategy + failure memory.

        Repair loop:
          Attempt 1 → Level 1 (TARGETED): Category-based instructions
          Attempt 2 → Level 2 (REASONING): LLM reasons about WHY, then fixes
          Attempt 3 → Level 3 (REGENERATE): Full regen with failure context

        Returns (repaired_code, attempts_used, levels_used).
        """
        memory = FailureMemory()
        current_code = test_code
        current_validation = validation_result
        levels_used: list[str] = []

        for attempt in range(1, self.config.max_repair_attempts + 1):
            self._state = TwoPhaseState.REPAIRING

            issues_before = current_validation.error_messages

            # Determine escalation level
            level = self.repair_selector.determine_level(
                attempt_number=attempt,
                max_attempts=self.config.max_repair_attempts,
                memory=memory,
            )
            levels_used.append(level.value)

            logger.info(
                "Repair attempt",
                attempt=attempt,
                level=level.value,
                errors=len(current_validation.errors),
                persistent=len(memory.get_persistent_issues()),
            )

            self._publish_event(EventType.REPAIR_STARTED, {
                "attempt": attempt,
                "level": level.value,
                "errors": len(current_validation.errors),
            })

            # Level 2+: Run reasoning engine first
            reasoning = None
            if level == RepairLevel.REASONING and self.config.reasoning_enabled:
                reasoning = self._run_reasoning(
                    current_code, current_validation.error_messages, memory,
                )

            # Build repair plan (uses memory + reasoning)
            repair_plan = self.repair_selector.build_repair_plan(
                validation_result=current_validation,
                attempt_number=attempt,
                max_attempts=self.config.max_repair_attempts,
                memory=memory,
                reasoning=reasoning,
            )

            # Build repair prompt
            repair_section = repair_plan.get_repair_prompt_section()

            feedback = (
                f"{repair_section}\n\n"
                f"Repair attempt {attempt}/{self.config.max_repair_attempts} "
                f"(Level: {level.value}). Fix these validation issues:\n"
                + "\n".join(f"- {msg}" for msg in current_validation.error_messages)
            )

            system_prompt = self.prompt_builder.build_system_prompt()
            user_prompt = self.prompt_builder.build_refinement_prompt(
                original_code=current_code,
                feedback=feedback,
                validation_issues=current_validation.error_messages,
                rag_chunks=rag_chunks,
            )

            response = self.vllm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=self.config.generation_max_tokens,
            )

            if not response.success:
                logger.warning("Repair LLM call failed", error=response.error)
                # Record failed attempt
                memory.record_attempt(
                    attempt_number=attempt,
                    strategy=level.value,
                    instructions_used=[inst.description for inst in repair_plan.instructions],
                    issues_before=issues_before,
                    issues_after=issues_before,  # Same issues (call failed)
                    reasoning=reasoning.get("overall_strategy") if reasoning else None,
                )
                continue

            repaired_code = self._extract_code(response.content)

            # Validate immediately (in-loop validation)
            current_validation = self.validator.validate(repaired_code, rag_chunks=rag_chunks)

            # Record attempt with full diff
            record = memory.record_attempt(
                attempt_number=attempt,
                strategy=level.value,
                instructions_used=[inst.description for inst in repair_plan.instructions],
                issues_before=issues_before,
                issues_after=current_validation.error_messages,
                reasoning=reasoning.get("overall_strategy") if reasoning else None,
            )

            self._publish_event(EventType.REPAIR_COMPLETED, {
                "attempt": attempt,
                "level": level.value,
                "success": current_validation.passed,
                "fixed": len(record.issues_fixed),
                "introduced": len(record.issues_introduced),
            })

            if current_validation.passed:
                logger.info(
                    "Repair successful",
                    attempt=attempt,
                    level=level.value,
                )
                return repaired_code, attempt, levels_used

            current_code = repaired_code

            # Log progress
            logger.info(
                "Repair attempt result",
                attempt=attempt,
                fixed=len(record.issues_fixed),
                introduced=len(record.issues_introduced),
                remaining=len(current_validation.errors),
            )

        # Return last attempt even if not fully fixed
        return current_code, self.config.max_repair_attempts, levels_used

    # ═══════════════════════════════════════════════════════════════════
    # Full Non-Streaming Generation
    # ═══════════════════════════════════════════════════════════════════

    def generate(
        self,
        source_code: str,
        class_name: str,
        file_path: str,
        rag_chunks: list[CodeChunk],
        collection_name: Optional[str] = None,
    ) -> TwoPhaseResult:
        """Execute full two-phase generation with iterative refinement.

        Full flow:
          Phase 1: Analyze → AnalysisResult
          Enrich: Search missing context
          Phase 2: Generate complete test class
          Self-Review: LLM checks its own output
          Validate: Structural/pattern validation
          Repair (if needed): Escalating strategy with failure memory
        """
        start_time = time.time()
        result = TwoPhaseResult(success=False, class_name=class_name)

        try:
            # ── Phase 1: Analysis ─────────────────────────────────────
            analysis_start = time.time()
            analysis_result = self.analyze(source_code, class_name, file_path)
            result.analysis_time_ms = (time.time() - analysis_start) * 1000

            if not analysis_result:
                result.error = "Analysis phase failed"
                result.failed_phase = "analysis"
                self._state = TwoPhaseState.FAILED
                return result

            result.analysis_result = analysis_result

            self._publish_event(EventType.STEP_COMPLETED, {
                "phase": "analysis",
                "methods": len(analysis_result.methods),
                "domain_types": len(analysis_result.all_domain_types),
                "complexity": analysis_result.complexity_level,
            })

            # ── Enrich context ────────────────────────────────────────
            extra_chunks = self.search_missing_context(
                analysis_result=analysis_result,
                rag_chunks=rag_chunks,
                source_code=source_code,
                collection_name=collection_name,
            )
            all_chunks = rag_chunks + extra_chunks
            result.extra_chunks_found = len(extra_chunks)

            # ── Phase 2: Generate complete test class ─────────────────
            self._state = TwoPhaseState.GENERATING

            gen_start = time.time()
            system_prompt, user_prompt = self.build_generation_prompt(
                analysis_result=analysis_result,
                class_name=class_name,
                file_path=file_path,
                rag_chunks=all_chunks,
                source_code=source_code,
                collection_name=collection_name,
            )

            self._publish_event(EventType.STEP_STARTED, {
                "phase": "generation",
                "class_name": class_name,
            })

            response = self.vllm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.config.generation_temperature,
                max_tokens=self.config.generation_max_tokens,
            )
            result.generation_time_ms = (time.time() - gen_start) * 1000

            if not response.success:
                result.error = f"Generation failed: {response.error}"
                result.failed_phase = "generation"
                self._state = TwoPhaseState.FAILED
                return result

            test_code = self._extract_code(response.content)
            result.test_code = test_code
            result.total_tokens_used = response.tokens_used

            self._publish_event(EventType.STEP_COMPLETED, {
                "phase": "generation",
                "tokens_used": response.tokens_used,
            })

            # ── Self-Review ───────────────────────────────────────────
            reviewed_code, review_issues = self.self_review(
                test_code=test_code,
                source_code=source_code,
                class_name=class_name,
                rag_chunks=all_chunks,
            )

            if review_issues:
                result.self_review_issues = review_issues
                result.self_review_applied = (reviewed_code != test_code)
                test_code = reviewed_code
                result.test_code = test_code

            # ── Validate (IN the loop, not just at end) ──────────────
            self._state = TwoPhaseState.VALIDATING

            validation_result = self.validator.validate(test_code, rag_chunks=all_chunks)
            result.validation_result = validation_result
            result.validation_passed = validation_result.passed
            result.validation_issues = validation_result.error_messages

            self._publish_event(EventType.VALIDATION_COMPLETED, {
                "passed": validation_result.passed,
                "errors": len(validation_result.errors),
                "warnings": len(validation_result.warnings),
            })

            # ── Iterative Repair (escalating, with memory) ───────────
            if not validation_result.passed:
                repaired_code, repair_attempts, levels_used = self.iterative_repair(
                    test_code=test_code,
                    validation_result=validation_result,
                    rag_chunks=all_chunks,
                    source_code=source_code,
                )
                result.repair_attempts = repair_attempts
                result.repair_levels_used = levels_used
                result.test_code = repaired_code

                # Re-validate after all repairs
                validation_result = self.validator.validate(repaired_code, rag_chunks=all_chunks)
                result.validation_result = validation_result
                result.validation_passed = validation_result.passed
                result.validation_issues = validation_result.error_messages

            # ── Success ───────────────────────────────────────────────
            result.success = True
            result.total_time_ms = (time.time() - start_time) * 1000
            self._state = TwoPhaseState.COMPLETED

            self._publish_event(EventType.GENERATION_COMPLETED, {
                "success": True,
                "class_name": class_name,
                "total_time_ms": round(result.total_time_ms, 1),
                "validation_passed": result.validation_passed,
                "self_review_applied": result.self_review_applied,
                "repair_levels": result.repair_levels_used,
            })

            logger.info(
                "Two-phase generation complete",
                class_name=class_name,
                success=True,
                validation_passed=result.validation_passed,
                total_time_ms=round(result.total_time_ms, 1),
                repair_attempts=result.repair_attempts,
                repair_levels=result.repair_levels_used,
                self_review_applied=result.self_review_applied,
                extra_chunks=len(extra_chunks),
            )
            return result

        except Exception as e:
            logger.error("Two-phase generation failed", error=str(e), class_name=class_name)
            result.error = str(e)
            result.failed_phase = self._state.value
            self._state = TwoPhaseState.FAILED

            self._publish_event(EventType.ERROR_OCCURRED, {
                "error": str(e),
                "phase": self._state.value,
            })
            return result

    # ═══════════════════════════════════════════════════════════════════
    # Private helpers
    # ═══════════════════════════════════════════════════════════════════

    def _build_analysis_section(self, analysis_result: AnalysisResult) -> str:
        """Build an analysis summary to include in the Phase 2 prompt."""
        lines = [
            "## Phase 1 Analysis Summary",
            f"Complexity: {analysis_result.complexity_level} (score: {analysis_result.complexity_score})",
            f"Dependencies to @Mock: {', '.join(analysis_result.constructor_dependencies) or 'None'}",
            f"Domain Types: {', '.join(analysis_result.all_domain_types) or 'None'}",
            "",
            "### Methods and Required Test Scenarios",
            "Generate tests ONLY for the scenarios listed below. Do NOT invent extra scenarios.",
        ]

        for m in analysis_result.methods:
            params_str = ", ".join(f"{t} {n}" for t, n in m.parameters) or "none"
            lines.append(f"\n**{m.return_type} {m.name}({params_str})** — {m.complexity}")
            if m.dependencies_called:
                lines.append(f"  Calls: {', '.join(m.dependencies_called)}")
            for s in m.test_scenarios:
                priority_label = {1: "MUST", 2: "SHOULD", 3: "NICE"}.get(s.priority, "")
                lines.append(f"  - [{priority_label}] {s.name}: {s.description}")
                if s.expected_behavior:
                    lines.append(f"    → Expected: {s.expected_behavior}")

        return "\n".join(lines)

    def _run_reasoning(
        self,
        test_code: str,
        validation_issues: list[str],
        memory: FailureMemory,
    ) -> Optional[dict]:
        """Run the reasoning engine: ask LLM WHY validation failed.

        Returns parsed reasoning dict or None on failure.
        """
        logger.info("Running repair reasoning engine", issues=len(validation_issues))

        system_prompt, user_prompt = self.reasoning_engine.build_reasoning_prompt(
            test_code=test_code,
            validation_issues=validation_issues,
            memory=memory,
        )

        response = self.vllm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=self.config.reasoning_max_tokens,
        )

        if not response.success:
            logger.warning("Reasoning LLM call failed", error=response.error)
            return None

        reasoning = self.reasoning_engine.parse_reasoning(response.content)
        if reasoning:
            logger.info(
                "Reasoning complete",
                analyses=len(reasoning.get("analyses", [])),
                strategy=reasoning.get("overall_strategy", "")[:100],
            )
        else:
            logger.warning("Failed to parse reasoning output")

        return reasoning

    def _extract_review_issues(self, review_content: str) -> list[str]:
        """Extract issues found during self-review from LLM response."""
        issues = []
        # Look for issues in comment format: // ISSUE: ...
        for match in re.finditer(r'//\s*(?:ISSUE|FIX|FIXED|CHANGE|CHANGED|PROBLEM):\s*(.+)', review_content):
            issues.append(match.group(1).strip())
        # Look for bullet points before code: - Found: ...
        for match in re.finditer(r'^[-*]\s*((?:Found|Fixed|Missing|Wrong|Added|Removed|Changed):.+)', review_content, re.MULTILINE):
            issues.append(match.group(1).strip())
        return issues

    def _extract_code(self, response: str) -> str:
        """Extract Java code from LLM response."""
        # Try markdown code block
        code_match = re.search(r'```(?:java)?\s*\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Try to find class declaration
        class_match = re.search(
            r'((?:package.*?\n)?(?:import.*?\n)*\s*(?:@\w+.*?\n)*\s*(?:public\s+)?class\s+\w+.*?\{.*\})',
            response,
            re.DOTALL,
        )
        if class_match:
            return class_match.group(1).strip()

        return response.strip()

    def _publish_event(self, event_type: EventType, data: dict) -> None:
        """Publish an event to the event bus."""
        self.event_bus.publish(Event(
            type=event_type,
            data=data,
            source="two_phase_strategy",
        ))


# ═══════════════════════════════════════════════════════════════════════
# Complexity Calculator
# ═══════════════════════════════════════════════════════════════════════

class ComplexityCalculator:
    """Calculate service complexity to decide generation strategy."""

    # Thresholds
    SIMPLE_THRESHOLD = 5
    MEDIUM_THRESHOLD = 15

    def calculate_from_rag(self, main_chunk: CodeChunk) -> tuple[int, str]:
        """Calculate complexity score from RAG chunk metadata.

        Returns:
            (score, level) where level is "simple", "medium", or "complex"
        """
        score = 0

        # Dependencies (each = 2 points)
        deps = len(main_chunk.dependencies or [])
        score += deps * 2

        # Used types (each = 3 points)
        used_types = len(main_chunk.used_types or [])
        score += used_types * 3

        # Method count (each = 1 point)
        methods = main_chunk.method_count or 0
        score += methods

        # Determine level
        if score <= self.SIMPLE_THRESHOLD:
            level = "simple"
        elif score <= self.MEDIUM_THRESHOLD:
            level = "medium"
        else:
            level = "complex"

        return score, level

    def should_use_two_phase(
        self,
        main_chunk: CodeChunk,
        threshold: int = 10,
    ) -> bool:
        """Determine if two-phase strategy should be used."""
        score, _ = self.calculate_from_rag(main_chunk)
        return score >= threshold
