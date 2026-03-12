"""
Planner Agent — separates *what to do* from *how to do it*.

The Planner inspects the incoming request and produces a structured
``ExecutionPlan`` that the Orchestrator can execute step-by-step
under the control of the StateMachine.

Responsibilities:
  - Determine task type (test generation / refinement / general chat)
  - Extract class name and file path
  - Decide which RAG queries are needed
  - Estimate complexity
  - Build ordered list of PlanSteps
"""

import re
from typing import Optional

import structlog

from .plan import ExecutionPlan, PlanStep, StepAction, TaskType

logger = structlog.get_logger()


class Planner:
    """Creates structured execution plans from user requests."""

    # ── Public API ───────────────────────────────────────────────────

    def plan_test_generation(
        self,
        file_path: str,
        class_name: Optional[str] = None,
        task_description: Optional[str] = None,
        session_id: Optional[str] = None,
        max_repair_attempts: int = 2,
    ) -> ExecutionPlan:
        """Build an execution plan for test generation.

        Steps produced (in order):
          1. extract_class_info   — resolve class name from file path
          2. retrieve_context     — fetch RAG chunks for target + deps
          3. build_prompt         — construct system + user prompt
          4. generate_code        — call LLM
          5. extract_code         — extract Java code from response
          6. validate_code        — run rule-based validation
          7. record_session       — persist to session memory
        """
        resolved_class = class_name or self._extract_class_name(file_path)

        plan = ExecutionPlan(
            task_type=TaskType.TEST_GENERATION,
            class_name=resolved_class or "",
            file_path=file_path,
            max_repair_attempts=max_repair_attempts,
            metadata={
                "session_id": session_id,
                "task_description": task_description,
                "has_inline_source": bool(
                    task_description and "```" in task_description
                ),
            },
        )

        # Step 1: Resolve class name (may already be done, but step is
        # kept so the orchestrator can validate / override)
        plan.add_step(
            action=StepAction.EXTRACT_CLASS_INFO,
            description=f"Resolve class name from {file_path}",
            file_path=file_path,
            class_name=resolved_class,
        )

        # Step 2: Retrieve RAG context
        plan.add_step(
            action=StepAction.RETRIEVE_CONTEXT,
            description=f"Fetch RAG context for {resolved_class or '?'}",
            class_name=resolved_class,
            file_path=file_path,
            inline_source=task_description,
            session_id=session_id,
        )

        # Step 3: Build prompts
        plan.add_step(
            action=StepAction.BUILD_PROMPT,
            description="Construct system + user prompt with RAG context",
            class_name=resolved_class,
            file_path=file_path,
            task_description=task_description,
            session_id=session_id,
        )

        # Step 4: Generate code via LLM
        plan.add_step(
            action=StepAction.GENERATE_CODE,
            description="Call LLM to generate unit test code",
        )

        # Step 5: Extract code from LLM response
        plan.add_step(
            action=StepAction.EXTRACT_CODE,
            description="Extract Java code block from LLM response",
        )

        # Step 6: Validate generated code
        plan.add_step(
            action=StepAction.VALIDATE_CODE,
            description="Validate against JUnit5/Mockito rules",
        )

        # Step 7: Record in session
        plan.add_step(
            action=StepAction.RECORD_SESSION,
            description="Persist results to session memory",
            session_id=session_id,
        )

        logger.info(
            "Plan created",
            plan_id=plan.plan_id,
            task_type=plan.task_type.value,
            class_name=plan.class_name,
            steps=len(plan.steps),
        )

        return plan

    def plan_refinement(
        self,
        session_id: str,
        feedback: str,
        last_class_name: str = "",
        last_test_code: str = "",
        max_repair_attempts: int = 2,
    ) -> ExecutionPlan:
        """Build an execution plan for refining a previously generated test.

        Steps produced:
          1. retrieve_context     — re-fetch RAG context from session cache
          2. build_prompt         — construct refinement prompt
          3. generate_code        — call LLM
          4. extract_code         — extract Java code
          5. validate_code        — validate
          6. record_session       — persist
        """
        plan = ExecutionPlan(
            task_type=TaskType.REFINEMENT,
            class_name=last_class_name,
            max_repair_attempts=max_repair_attempts,
            metadata={
                "session_id": session_id,
                "feedback": feedback,
                "original_code": last_test_code,
            },
        )

        plan.add_step(
            action=StepAction.RETRIEVE_CONTEXT,
            description=f"Re-fetch cached RAG context for {last_class_name}",
            session_id=session_id,
            class_name=last_class_name,
        )

        plan.add_step(
            action=StepAction.BUILD_PROMPT,
            description="Construct refinement prompt with feedback",
            feedback=feedback,
            original_code=last_test_code,
            session_id=session_id,
        )

        plan.add_step(
            action=StepAction.GENERATE_CODE,
            description="Call LLM to refine test code",
        )

        plan.add_step(
            action=StepAction.EXTRACT_CODE,
            description="Extract Java code from response",
        )

        plan.add_step(
            action=StepAction.VALIDATE_CODE,
            description="Validate refined code",
        )

        plan.add_step(
            action=StepAction.RECORD_SESSION,
            description="Persist refinement to session",
            session_id=session_id,
        )

        logger.info(
            "Refinement plan created",
            plan_id=plan.plan_id,
            class_name=last_class_name,
            steps=len(plan.steps),
        )

        return plan

    def plan_repair(
        self,
        original_plan: ExecutionPlan,
        validation_issues: list[str],
        generated_code: str,
    ) -> None:
        """Mutate an existing plan to add repair steps.

        Appends new steps (build_prompt → generate → extract → validate)
        and increments the repair counter.
        """
        original_plan.current_repair_attempt += 1
        attempt = original_plan.current_repair_attempt

        logger.info(
            "Planning repair",
            plan_id=original_plan.plan_id,
            attempt=attempt,
            max_attempts=original_plan.max_repair_attempts,
            issues=validation_issues,
        )

        # Add repair step (builds a refined prompt with issues)
        original_plan.add_step(
            action=StepAction.REPAIR_CODE,
            description=f"Repair attempt {attempt}: fix validation issues",
            validation_issues=validation_issues,
            generated_code=generated_code,
            attempt=attempt,
        )

        # Re-generate
        original_plan.add_step(
            action=StepAction.GENERATE_CODE,
            description=f"Re-generate code (repair attempt {attempt})",
        )

        # Re-extract
        original_plan.add_step(
            action=StepAction.EXTRACT_CODE,
            description=f"Extract code (repair attempt {attempt})",
        )

        # Re-validate
        original_plan.add_step(
            action=StepAction.VALIDATE_CODE,
            description=f"Validate repaired code (attempt {attempt})",
        )

        # Re-record
        original_plan.add_step(
            action=StepAction.RECORD_SESSION,
            description=f"Record repair result (attempt {attempt})",
            session_id=original_plan.metadata.get("session_id"),
        )

    def plan_incremental_update(
        self,
        file_path: str,
        existing_test_code: str,
        class_name: Optional[str] = None,
        task_description: Optional[str] = None,
        changed_methods: Optional[list[str]] = None,
        session_id: Optional[str] = None,
        max_repair_attempts: int = 2,
    ) -> ExecutionPlan:
        """Build an execution plan for incremental test update.

        Used when a test file already exists and only new/changed methods
        need additional tests.  The CI script provides ``existing_test_code``
        explicitly — the agent does NOT auto-detect anything.

        Steps produced (in order):
          1. extract_class_info      — resolve class name from file path
          2. analyze_existing_test   — parse existing test to find covered methods
          3. retrieve_context        — fetch RAG chunks for target + deps
          4. build_prompt            — incremental prompt with existing test
          5. generate_code           — call LLM
          6. extract_code            — extract Java code from response
          7. merge_tests             — merge new tests into existing test class
          8. validate_code           — run rule-based validation
          9. record_session          — persist to session memory
        """
        resolved_class = class_name or self._extract_class_name(file_path)

        plan = ExecutionPlan(
            task_type=TaskType.INCREMENTAL_UPDATE,
            class_name=resolved_class or "",
            file_path=file_path,
            max_repair_attempts=max_repair_attempts,
            metadata={
                "session_id": session_id,
                "task_description": task_description,
                "existing_test_code": existing_test_code,
                "changed_methods": changed_methods or [],
            },
        )

        # Step 1: Resolve class name
        plan.add_step(
            action=StepAction.EXTRACT_CLASS_INFO,
            description=f"Resolve class name from {file_path}",
            file_path=file_path,
            class_name=resolved_class,
        )

        # Step 2: Analyze existing test file
        plan.add_step(
            action=StepAction.ANALYZE_EXISTING_TEST,
            description="Parse existing test to find already-covered methods",
            existing_test_code=existing_test_code,
        )

        # Step 3: Retrieve RAG context
        plan.add_step(
            action=StepAction.RETRIEVE_CONTEXT,
            description=f"Fetch RAG context for {resolved_class or '?'}",
            class_name=resolved_class,
            file_path=file_path,
            session_id=session_id,
        )

        # Step 4: Build incremental prompt
        plan.add_step(
            action=StepAction.BUILD_PROMPT,
            description="Construct incremental update prompt with existing test",
            class_name=resolved_class,
            file_path=file_path,
            task_description=task_description,
            session_id=session_id,
        )

        # Step 5: Generate code via LLM
        plan.add_step(
            action=StepAction.GENERATE_CODE,
            description="Call LLM to generate incremental tests",
        )

        # Step 6: Extract code from response
        plan.add_step(
            action=StepAction.EXTRACT_CODE,
            description="Extract Java code block from LLM response",
        )

        # Step 7: Merge new tests into existing test class
        plan.add_step(
            action=StepAction.MERGE_TESTS,
            description="Merge newly generated tests into existing test class",
            existing_test_code=existing_test_code,
        )

        # Step 8: Validate merged code
        plan.add_step(
            action=StepAction.VALIDATE_CODE,
            description="Validate merged test code against JUnit5/Mockito rules",
        )

        # Step 9: Record in session
        plan.add_step(
            action=StepAction.RECORD_SESSION,
            description="Persist results to session memory",
            session_id=session_id,
        )

        logger.info(
            "Incremental update plan created",
            plan_id=plan.plan_id,
            task_type=plan.task_type.value,
            class_name=plan.class_name,
            steps=len(plan.steps),
            changed_methods=changed_methods,
        )

        return plan

    # ── Two-Phase Generation Planning ─────────────────────────────────

    def plan_two_phase_generation(
        self,
        file_path: str,
        class_name: Optional[str] = None,
        task_description: Optional[str] = None,
        session_id: Optional[str] = None,
        max_repair_attempts: int = 2,
        complexity_score: int = 0,
    ) -> ExecutionPlan:
        """Build an execution plan for two-phase test generation.

        Two-phase generation is used for complex services where single-pass
        generation tends to hallucinate domain types.

        Steps produced (in order):
          1. extract_class_info      — resolve class name from file path
          2. retrieve_context        — fetch RAG chunks for target + deps
          3. analyze_service         — Phase 1: LLM analyzes service, outputs JSON
          4. lookup_registry         — Lookup domain types in registry
          5. generate_method_tests   — Phase 2: Generate tests per method
          6. assemble_test_class     — Assemble final test class
          7. validate_code           — run rule-based validation
          8. record_session          — persist to session memory
        """
        resolved_class = class_name or self._extract_class_name(file_path)

        plan = ExecutionPlan(
            task_type=TaskType.TWO_PHASE_GENERATION,
            class_name=resolved_class or "",
            file_path=file_path,
            max_repair_attempts=max_repair_attempts,
            metadata={
                "session_id": session_id,
                "task_description": task_description,
                "complexity_score": complexity_score,
                "strategy": "two_phase",
                "has_inline_source": bool(
                    task_description and "```" in task_description
                ),
            },
        )

        # Step 1: Resolve class name
        plan.add_step(
            action=StepAction.EXTRACT_CLASS_INFO,
            description=f"Resolve class name from {file_path}",
            file_path=file_path,
            class_name=resolved_class,
        )

        # Step 2: Retrieve RAG context
        plan.add_step(
            action=StepAction.RETRIEVE_CONTEXT,
            description=f"Fetch RAG context for {resolved_class or '?'}",
            class_name=resolved_class,
            file_path=file_path,
            inline_source=task_description,
            session_id=session_id,
        )

        # Step 3: Phase 1 - Analyze service
        plan.add_step(
            action=StepAction.ANALYZE_SERVICE,
            description="Phase 1: Analyze service structure and plan tests",
            class_name=resolved_class,
            file_path=file_path,
        )

        # Step 4: Lookup domain types in registry
        plan.add_step(
            action=StepAction.LOOKUP_REGISTRY,
            description="Lookup domain type construction patterns from registry",
        )

        # Step 5: Phase 2 - Generate tests per method
        plan.add_step(
            action=StepAction.GENERATE_METHOD_TESTS,
            description="Phase 2: Generate focused tests for each method",
        )

        # Step 6: Assemble final test class
        plan.add_step(
            action=StepAction.ASSEMBLE_TEST_CLASS,
            description="Assemble method tests into complete test class",
        )

        # Step 7: Validate generated code
        plan.add_step(
            action=StepAction.VALIDATE_CODE,
            description="Validate against JUnit5/Mockito rules",
        )

        # Step 8: Record in session
        plan.add_step(
            action=StepAction.RECORD_SESSION,
            description="Persist results to session memory",
            session_id=session_id,
        )

        logger.info(
            "Two-phase plan created",
            plan_id=plan.plan_id,
            task_type=plan.task_type.value,
            class_name=plan.class_name,
            steps=len(plan.steps),
            complexity_score=complexity_score,
        )

        return plan

    def plan_complexity_check(
        self,
        file_path: str,
        class_name: Optional[str] = None,
    ) -> ExecutionPlan:
        """Build a minimal plan to check service complexity.

        Used to decide whether to use single-pass or two-phase generation.
        """
        resolved_class = class_name or self._extract_class_name(file_path)

        plan = ExecutionPlan(
            task_type=TaskType.TEST_GENERATION,  # Will be changed after check
            class_name=resolved_class or "",
            file_path=file_path,
            metadata={"purpose": "complexity_check"},
        )

        plan.add_step(
            action=StepAction.EXTRACT_CLASS_INFO,
            description=f"Resolve class name from {file_path}",
            file_path=file_path,
            class_name=resolved_class,
        )

        plan.add_step(
            action=StepAction.RETRIEVE_CONTEXT,
            description=f"Fetch RAG context for complexity analysis",
            class_name=resolved_class,
            file_path=file_path,
        )

        plan.add_step(
            action=StepAction.CALCULATE_COMPLEXITY,
            description="Calculate service complexity score",
        )

        return plan

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_class_name(file_path: str) -> Optional[str]:
        """Extract class name from a Java file path."""
        file_name = file_path.replace("\\", "/").split("/")[-1]
        if file_name.endswith(".java"):
            return file_name[:-5]
        return None
