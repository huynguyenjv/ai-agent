"""
Two-Phase Generation Strategy — Analyze first, then generate with enriched context.

This strategy splits test generation into two distinct phases:

Phase 1 (Analysis):
    - LLM analyzes the service source code
    - Outputs structured JSON with method analysis and test scenarios
    - Identifies all dependencies, domain types, exceptions
    - NO code generation in this phase

Phase 2 (Enhanced Generation):
    - Enriches RAG context by searching for missing types (exceptions, domain types)
    - Builds a COMPLETE test class prompt with analysis summary + registry context
    - Generates the COMPLETE test class in ONE LLM call (same quality as single-pass)
    - Uses PromptBuilder for consistent prompt style

Key insight: Phase 2 is essentially a BETTER single-pass — same full-class generation,
but with richer context. This avoids the quality degradation of per-method generation.
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
from .repair import RepairStrategySelector
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

    # Repair settings
    max_repair_attempts: int = 2            # Max repair attempts

    # Fallback settings
    fallback_to_single_pass: bool = True    # Fallback if two-phase fails


class TwoPhaseState(str, Enum):
    """States of the two-phase generation process."""
    IDLE = "idle"
    ANALYZING = "analyzing"
    ENRICHING = "enriching"
    GENERATING = "generating"
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

    # Metrics
    total_tokens_used: int = 0
    analysis_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0
    repair_attempts: int = 0

    # Error info
    error: Optional[str] = None
    failed_phase: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
# Two-Phase Strategy
# ═══════════════════════════════════════════════════════════════════════

class TwoPhaseStrategy:
    """Two-Phase Generation: Analyze → Enrich → Generate.

    Phase 2 generates the COMPLETE test class in a single LLM call,
    using the same PromptBuilder as single-pass but with enriched context.

    Usage::

        strategy = TwoPhaseStrategy(
            vllm_client=vllm,
            domain_registry=registry,
            prompt_builder=prompt_builder,
            test_rules=test_rules,
            rag_client=rag_client,
        )

        # Non-streaming
        result = strategy.generate(source_code, class_name, file_path, rag_chunks)

        # Streaming (used by orchestrator)
        analysis = strategy.analyze(source_code, class_name, file_path)
        extra = strategy.search_missing_context(analysis, rag_chunks, source_code)
        sys_prompt, usr_prompt = strategy.build_generation_prompt(
            analysis, class_name, file_path, rag_chunks + extra, source_code)
        # then stream with vllm.stream_generate(sys_prompt, usr_prompt)
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
        """Execute full two-phase generation (non-streaming).

        Orchestrates: analyze → enrich → build prompt → generate → validate → repair.
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

            # ── Validate ──────────────────────────────────────────────
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

            # ── Repair if needed ──────────────────────────────────────
            if not validation_result.passed:
                self._state = TwoPhaseState.REPAIRING

                repaired_code, repair_attempts = self._run_repair(
                    test_code=test_code,
                    validation_result=validation_result,
                    rag_chunks=all_chunks,
                )
                result.repair_attempts = repair_attempts

                if repaired_code:
                    result.test_code = repaired_code
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
            })

            logger.info(
                "Two-phase generation complete",
                class_name=class_name,
                success=True,
                validation_passed=result.validation_passed,
                total_time_ms=round(result.total_time_ms, 1),
                repair_attempts=result.repair_attempts,
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
        """Build an analysis summary to include in the Phase 2 prompt.

        Tells the LLM exactly what methods to test and what scenarios
        to cover, reducing hallucination of non-existent behaviors.
        """
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

    def _run_repair(
        self,
        test_code: str,
        validation_result: ValidationResult,
        rag_chunks: list[CodeChunk],
    ) -> tuple[Optional[str], int]:
        """Run repair loop using PromptBuilder.build_refinement_prompt."""
        current_code = test_code

        for attempt in range(1, self.config.max_repair_attempts + 1):
            logger.info(
                "Running repair attempt",
                attempt=attempt,
                errors=len(validation_result.errors),
            )

            self._publish_event(EventType.REPAIR_STARTED, {
                "attempt": attempt,
                "errors": len(validation_result.errors),
            })

            # Build repair prompt using existing PromptBuilder (same as single-pass)
            repair_plan = self.repair_selector.build_repair_plan(
                validation_result=validation_result,
                attempt_number=attempt,
                max_attempts=self.config.max_repair_attempts,
            )
            repair_section = repair_plan.get_repair_prompt_section()

            feedback = (
                f"{repair_section}\n\n"
                f"Auto-repair attempt {attempt}: fix these validation issues: "
                f"{', '.join(validation_result.error_messages)}"
            )

            system_prompt = self.prompt_builder.build_system_prompt()
            user_prompt = self.prompt_builder.build_refinement_prompt(
                original_code=current_code,
                feedback=feedback,
                validation_issues=validation_result.error_messages,
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
                continue

            repaired_code = self._extract_code(response.content)
            validation_result = self.validator.validate(repaired_code, rag_chunks=rag_chunks)

            self._publish_event(EventType.REPAIR_COMPLETED, {
                "attempt": attempt,
                "success": validation_result.passed,
            })

            if validation_result.passed:
                logger.info("Repair successful", attempt=attempt)
                return repaired_code, attempt

            current_code = repaired_code

        # Return last attempt even if not fully fixed
        return current_code, self.config.max_repair_attempts

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
