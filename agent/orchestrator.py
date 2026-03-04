"""
Agent orchestrator for coordinating test generation workflow.

Phase 1 refactor: the Orchestrator now delegates planning to the
``Planner`` and drives execution through the ``StateMachine``.
The public API (``generate_test``, ``refine_test``) is unchanged
so the server layer requires zero modifications.
"""

import re
from typing import Optional, Generator
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import structlog

from .plan import ExecutionPlan, PlanStep, StepAction, StepStatus, TaskType
from .planner import Planner
from .prompt import PromptBuilder
from .rules import TestRules
from .memory import SessionMemory, MemoryManager
from .state_machine import AgentState, StateMachine, TransitionError
from .validation import ValidationPipeline, ValidationResult, IssueSeverity
from .repair import RepairStrategySelector, RepairPlan
from .events import EventBus, Event, EventType, get_event_bus
from .metrics import MetricsCollector
from rag.client import RAGClient
from rag.schema import SearchQuery, CodeChunk, MetadataFilter
from vllm.client import VLLMClient

# Phase 2: optional context builder
try:
    from context.context_builder import ContextBuilder, ContextResult
    _CONTEXT_BUILDER_AVAILABLE = True
except ImportError:
    _CONTEXT_BUILDER_AVAILABLE = False
    ContextBuilder = ContextResult = None  # type: ignore

logger = structlog.get_logger()


@dataclass
class GenerationRequest:
    """Request for test generation."""

    file_path: str
    class_name: Optional[str] = None
    task_description: Optional[str] = None
    session_id: Optional[str] = None
    existing_test_code: Optional[str] = None
    changed_methods: Optional[list[str]] = None


@dataclass
class GenerationResult:
    """Result of test generation."""

    success: bool
    test_code: Optional[str] = None
    class_name: str = ""
    validation_passed: bool = True
    validation_issues: list[str] = None
    error: Optional[str] = None
    rag_chunks_used: int = 0
    tokens_used: int = 0
    plan_summary: Optional[dict] = None
    validation_summary: Optional[dict] = None   # Phase 3: detailed validation
    repair_attempts: int = 0                     # Phase 3: how many repairs

    def __post_init__(self):
        if self.validation_issues is None:
            self.validation_issues = []


class StreamPhase(str, Enum):
    """Phases emitted during streaming test generation."""
    PLANNING = "planning"
    RETRIEVING = "retrieving"
    GENERATING = "generating"
    VALIDATING = "validating"
    REPAIRING = "repairing"
    DONE = "done"
    ERROR = "error"


@dataclass
class StreamEvent:
    """A single event in the streaming test generation pipeline.

    - ``phase`` — current pipeline phase
    - ``content`` — text content (delta for GENERATING, full msg for others)
    - ``delta`` — True if ``content`` is an incremental token (GENERATING phase)
    - ``metadata`` — extra info (chunk count, validation result, etc.)
    """
    phase: StreamPhase
    content: str = ""
    delta: bool = False
    metadata: dict = field(default_factory=dict)


class AgentOrchestrator:
    """Orchestrates the test generation workflow using StateMachine + Planner."""

    def __init__(
        self,
        rag_client: RAGClient,
        vllm_client: VLLMClient,
        top_k_results: int = 10,
        max_context_tokens: int = 4000,
        max_repair_attempts: int = 2,
        repo_path: Optional[str] = None,
        token_budget: int = 6000,
    ):
        self.rag = rag_client
        self.vllm = vllm_client
        self.prompt_builder = PromptBuilder()
        self.test_rules = TestRules()
        self.memory_manager = MemoryManager()
        self.planner = Planner()
        self.validator = ValidationPipeline()         # Phase 3
        self.repair_selector = RepairStrategySelector()  # Phase 3

        # Phase 4: Event system + Metrics
        self.event_bus = get_event_bus()
        self.metrics = MetricsCollector(self.event_bus)

        self.top_k = top_k_results
        self.max_context_tokens = max_context_tokens
        self.max_repair_attempts = max_repair_attempts

        # Phase 2: Context Builder (optional, graceful degradation)
        self.context_builder: Optional[object] = None
        if _CONTEXT_BUILDER_AVAILABLE and ContextBuilder is not None:
            self.context_builder = ContextBuilder(
                rag_client=rag_client,
                repo_path=repo_path,
                token_budget=token_budget,
            )
            if repo_path:
                try:
                    self.context_builder.init_intelligence(repo_path)
                except Exception as e:
                    logger.warning("Intelligence init failed, using RAG-only", error=str(e))

        logger.info(
            "Agent orchestrator initialized (Phase 1-4: SM + Planner + Context + Validation + Events)",
            intelligence=self.context_builder.intelligence_ready if self.context_builder else False,
        )

    # =====================================================================
    # Public API  (backward-compatible)
    # =====================================================================

    def generate_test(self, request: GenerationRequest) -> GenerationResult:
        """Generate unit tests for a class.

        Workflow via StateMachine:
          IDLE → PLANNING → RETRIEVING → GENERATING → VALIDATING → COMPLETED
                                                          ↓
                                                      REPAIRING → GENERATING → …
        """
        sm = StateMachine()

        try:
            # ── IDLE → PLANNING ──────────────────────────────────────
            sm.transition_to(AgentState.PLANNING, request_file=request.file_path)

            # Choose plan based on whether existing test code is provided
            if request.existing_test_code:
                plan = self.planner.plan_incremental_update(
                    file_path=request.file_path,
                    existing_test_code=request.existing_test_code,
                    class_name=request.class_name,
                    task_description=request.task_description,
                    changed_methods=request.changed_methods,
                    session_id=request.session_id,
                    max_repair_attempts=self.max_repair_attempts,
                )
            else:
                plan = self.planner.plan_test_generation(
                    file_path=request.file_path,
                    class_name=request.class_name,
                    task_description=request.task_description,
                    session_id=request.session_id,
                    max_repair_attempts=self.max_repair_attempts,
                )
            sm.plan = plan

            # Phase 4: Publish plan created + generation started events
            self.event_bus.publish(Event(
                type=EventType.PLAN_CREATED,
                data={"plan_id": plan.plan_id, "steps": len(plan.steps), "task": "generate"},
                source="orchestrator",
                plan_id=plan.plan_id,
                session_id=request.session_id,
            ))
            self.event_bus.publish(Event(
                type=EventType.GENERATION_STARTED,
                data={"plan_id": plan.plan_id, "class_name": plan.class_name},
                source="orchestrator",
                plan_id=plan.plan_id,
                session_id=request.session_id,
            ))

            class_name = plan.class_name
            if not class_name:
                sm.fail(error="Could not determine class name from file path")
                return GenerationResult(success=False, error=sm.error)

            logger.info(
                "Starting test generation",
                class_name=class_name,
                file_path=request.file_path,
                plan_id=plan.plan_id,
            )

            # ── Execute plan steps ───────────────────────────────────
            result = self._execute_plan(sm, plan)

            # Phase 4: Publish generation completed
            self.event_bus.publish(Event(
                type=EventType.GENERATION_COMPLETED,
                data={
                    "plan_id": plan.plan_id,
                    "success": result.success,
                    "tokens_used": result.tokens_used,
                },
                source="orchestrator",
                plan_id=plan.plan_id,
                session_id=request.session_id,
            ))
            return result

        except TransitionError as e:
            logger.error("State transition error", error=str(e))
            return GenerationResult(success=False, error=str(e))
        except Exception as e:
            logger.error("Test generation failed", error=str(e))
            if not sm.is_terminal:
                sm.fail(error=str(e))
            return GenerationResult(success=False, error=str(e))

    # =====================================================================
    # Streaming API  (progress phases + token-by-token code gen)
    # =====================================================================

    def generate_test_streaming(
        self, request: GenerationRequest
    ) -> Generator[StreamEvent, None, None]:
        """Generate unit tests with real-time streaming.

        Yields ``StreamEvent`` objects for each phase:
          PLANNING    → "Planning test generation…"
          RETRIEVING  → "Searching codebase context…" + chunk count
          GENERATING  → token-by-token code deltas from vLLM
          VALIDATING  → "Validating generated code…" + result
          REPAIRING   → (if needed) "Repairing code…" + re-generate stream
          DONE        → final metadata (validation summary, tokens)
          ERROR       → error message

        Usage in API layer::

            for event in orchestrator.generate_test_streaming(req):
                if event.delta:
                    send_sse(event.content)  # token delta
                else:
                    send_sse_status(event)   # phase update
        """
        try:
            # ── PLANNING ─────────────────────────────────────────────
            yield StreamEvent(
                phase=StreamPhase.PLANNING,
                content=f"📋 Planning test generation for `{request.file_path}`…",
            )

            plan = self.planner.plan_test_generation(
                file_path=request.file_path,
                class_name=request.class_name,
                task_description=request.task_description,
                session_id=request.session_id,
                max_repair_attempts=self.max_repair_attempts,
            )

            class_name = plan.class_name
            if not class_name:
                yield StreamEvent(
                    phase=StreamPhase.ERROR,
                    content="Could not determine class name from file path",
                )
                return

            # ── RETRIEVING ───────────────────────────────────────────
            yield StreamEvent(
                phase=StreamPhase.RETRIEVING,
                content=f"🔍 Searching codebase context for `{class_name}`…",
            )

            # Resolve session
            session = None
            session_id = request.session_id
            if session_id:
                session = self.memory_manager.get_or_create_session(session_id)
                session.set_context(class_name=class_name, file_path=request.file_path)

            # Retrieve RAG context (same logic as _step_retrieve_context)
            rag_chunks: list[CodeChunk] = []
            if self.context_builder:
                try:
                    context_result = self.context_builder.build_context(
                        class_name=class_name,
                        file_path=request.file_path,
                        session=session,
                        top_k=self.top_k,
                    )
                    rag_chunks = context_result.rag_chunks
                except Exception:
                    rag_chunks = self._get_rag_context(
                        class_name=class_name,
                        file_path=request.file_path,
                        session=session,
                    )
            else:
                rag_chunks = self._get_rag_context(
                    class_name=class_name,
                    file_path=request.file_path,
                    session=session,
                )

            yield StreamEvent(
                phase=StreamPhase.RETRIEVING,
                content=f"✅ Found {len(rag_chunks)} context chunks",
                metadata={"chunks_count": len(rag_chunks)},
            )

            # ── Build prompt ─────────────────────────────────────────
            system_prompt = self.prompt_builder.build_system_prompt()
            user_prompt = self.prompt_builder.build_test_generation_prompt(
                class_name=class_name,
                file_path=request.file_path,
                rag_chunks=rag_chunks,
                task_description=request.task_description,
                session=session,
            )

            # ── GENERATING (stream tokens) ───────────────────────────
            yield StreamEvent(
                phase=StreamPhase.GENERATING,
                content=f"⚡ Generating tests for `{class_name}`…\n\n",
            )

            # Accumulate full response while streaming
            code_parts: list[str] = []
            for delta_token in self.vllm.stream_generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            ):
                code_parts.append(delta_token)
                yield StreamEvent(
                    phase=StreamPhase.GENERATING,
                    content=delta_token,
                    delta=True,
                )

            full_response = "".join(code_parts)
            extracted_code = self._extract_code(full_response)

            # ── VALIDATING ───────────────────────────────────────────
            yield StreamEvent(
                phase=StreamPhase.VALIDATING,
                content="\n\n🔎 Validating generated code…",
            )

            validation_result = self.validator.validate(
                extracted_code, rag_chunks=rag_chunks,
            )

            tokens_used = len(full_response) // 4  # rough estimate

            if validation_result.passed:
                yield StreamEvent(
                    phase=StreamPhase.VALIDATING,
                    content=f"✅ Validation passed ({validation_result.test_count} test(s), 0 errors)",
                    metadata={"passed": True, **validation_result.get_summary()},
                )
            else:
                yield StreamEvent(
                    phase=StreamPhase.VALIDATING,
                    content=(
                        f"⚠️ Validation: {len(validation_result.errors)} errors, "
                        f"{len(validation_result.warnings)} warnings"
                    ),
                    metadata={"passed": False, **validation_result.get_summary()},
                )

                # ── REPAIRING ────────────────────────────────────────
                for attempt in range(1, self.max_repair_attempts + 1):
                    yield StreamEvent(
                        phase=StreamPhase.REPAIRING,
                        content=f"\n\n🔧 Repair attempt {attempt}/{self.max_repair_attempts}…",
                        metadata={"attempt": attempt},
                    )

                    # Build repair prompt
                    repair_plan = self.repair_selector.build_repair_plan(
                        validation_result=validation_result,
                        attempt_number=attempt,
                        max_attempts=self.max_repair_attempts,
                    )
                    repair_section = repair_plan.get_repair_prompt_section()
                    feedback = (
                        f"{repair_section}\n\n"
                        f"Auto-repair attempt {attempt}: fix the following validation issues: "
                        f"{', '.join(validation_result.error_messages)}"
                    )

                    repair_system = self.prompt_builder.build_system_prompt()
                    repair_user = self.prompt_builder.build_refinement_prompt(
                        original_code=extracted_code,
                        feedback=feedback,
                        validation_issues=validation_result.error_messages,
                        rag_chunks=rag_chunks,
                    )

                    # Stream repair generation
                    yield StreamEvent(
                        phase=StreamPhase.GENERATING,
                        content=f"\n\n⚡ Re-generating (repair {attempt})…\n\n",
                    )

                    repair_parts: list[str] = []
                    for delta_token in self.vllm.stream_generate(
                        system_prompt=repair_system,
                        user_prompt=repair_user,
                    ):
                        repair_parts.append(delta_token)
                        yield StreamEvent(
                            phase=StreamPhase.GENERATING,
                            content=delta_token,
                            delta=True,
                        )

                    full_response = "".join(repair_parts)
                    extracted_code = self._extract_code(full_response)

                    # Re-validate
                    yield StreamEvent(
                        phase=StreamPhase.VALIDATING,
                        content="\n\n🔎 Re-validating…",
                    )
                    validation_result = self.validator.validate(
                        extracted_code, rag_chunks=rag_chunks,
                    )

                    if validation_result.passed:
                        yield StreamEvent(
                            phase=StreamPhase.VALIDATING,
                            content=f"✅ Validation passed after repair {attempt}",
                            metadata={"passed": True, **validation_result.get_summary()},
                        )
                        break
                    else:
                        yield StreamEvent(
                            phase=StreamPhase.VALIDATING,
                            content=(
                                f"⚠️ Still {len(validation_result.errors)} errors "
                                f"after repair {attempt}"
                            ),
                            metadata={"passed": False, **validation_result.get_summary()},
                        )

            # Record in session
            if session:
                session.add_user_message(
                    request.task_description or f"Generate tests for {class_name}",
                    metadata={"file_path": request.file_path},
                )
                session.add_assistant_message(
                    extracted_code,
                    metadata={"validation_passed": validation_result.passed},
                )
                session.record_generated_test(
                    class_name=class_name,
                    test_code=extracted_code,
                    success=validation_result.passed,
                )

            # ── DONE ─────────────────────────────────────────────────
            yield StreamEvent(
                phase=StreamPhase.DONE,
                content="",
                metadata={
                    "success": True,
                    "class_name": class_name,
                    "validation_passed": validation_result.passed,
                    "validation_issues": validation_result.error_messages,
                    "tokens_used": tokens_used,
                    "rag_chunks_used": len(rag_chunks),
                },
            )

        except Exception as e:
            logger.error("Streaming test generation failed", error=str(e))
            yield StreamEvent(
                phase=StreamPhase.ERROR,
                content=f"Error: {e}",
            )

    def refine_test(
        self,
        session_id: str,
        feedback: str,
    ) -> GenerationResult:
        """Refine previously generated test based on feedback."""
        sm = StateMachine()

        try:
            session = self.memory_manager.get_session(session_id)
            if not session:
                return GenerationResult(success=False, error="Session not found or expired")

            if not session.generated_tests:
                return GenerationResult(success=False, error="No previous test generation in this session")

            last_test = session.generated_tests[-1]

            # ── IDLE → PLANNING ──────────────────────────────────────
            sm.transition_to(AgentState.PLANNING, refinement=True, session_id=session_id)

            plan = self.planner.plan_refinement(
                session_id=session_id,
                feedback=feedback,
                last_class_name=last_test.class_name,
                last_test_code=last_test.test_code,
                max_repair_attempts=self.max_repair_attempts,
            )
            sm.plan = plan

            return self._execute_plan(sm, plan)

        except TransitionError as e:
            logger.error("State transition error", error=str(e))
            return GenerationResult(success=False, error=str(e))
        except Exception as e:
            logger.error("Refinement failed", error=str(e))
            if not sm.is_terminal:
                sm.fail(error=str(e))
            return GenerationResult(success=False, error=str(e))

    # =====================================================================
    # Plan execution engine
    # =====================================================================

    def _execute_plan(self, sm: StateMachine, plan: ExecutionPlan) -> GenerationResult:
        """Execute an ``ExecutionPlan`` step-by-step, driven by the StateMachine.

        The engine loops through pending steps, executing each one and
        transitioning the state machine accordingly.  If validation fails
        and the plan allows repair, it appends repair steps and continues.
        """
        # Shared context accumulated during execution
        ctx: dict = {
            "class_name": plan.class_name,
            "file_path": plan.file_path,
            "session": None,
            "rag_chunks": [],
            "system_prompt": None,
            "user_prompt": None,
            "full_response": None,
            "extracted_code": None,
            "validation_passed": True,
            "validation_issues": [],
            "tokens_used": 0,
        }

        # Resolve session once
        session_id = plan.metadata.get("session_id")
        if session_id:
            ctx["session"] = self.memory_manager.get_or_create_session(session_id)
            if plan.task_type == TaskType.TEST_GENERATION:
                ctx["session"].set_context(
                    class_name=plan.class_name,
                    file_path=plan.file_path,
                )

        # ── Step loop ────────────────────────────────────────────────
        while True:
            step = plan.get_next_pending_step()
            if step is None:
                break

            step.start()

            # Phase 4: Publish step started
            self.event_bus.publish(Event(
                type=EventType.STEP_STARTED,
                data={"step_id": step.step_id, "action": step.action.value, "description": step.description},
                source="orchestrator",
                plan_id=plan.plan_id,
            ))

            try:
                self._execute_step(sm, plan, step, ctx)
                step.complete()

                # Phase 4: Publish step completed
                self.event_bus.publish(Event(
                    type=EventType.STEP_COMPLETED,
                    data={"step_id": step.step_id, "action": step.action.value, "result": step.result},
                    source="orchestrator",
                    plan_id=plan.plan_id,
                ))
            except _ValidationFailed as vf:
                # Validation failed — decide repair or accept
                step.complete(result={"passed": False, "issues": vf.issues})

                # Phase 4: Publish step failed + validation event
                self.event_bus.publish(Event(
                    type=EventType.STEP_FAILED,
                    data={"step_id": step.step_id, "action": step.action.value, "issues": vf.issues},
                    source="orchestrator",
                    plan_id=plan.plan_id,
                ))

                # Store validation result in ctx for repair strategy
                if vf.validation_result:
                    ctx["validation_result"] = vf.validation_result

                if plan.can_repair:
                    # ── VALIDATING → REPAIRING ───────────────────────
                    sm.transition_to(
                        AgentState.REPAIRING,
                        attempt=plan.current_repair_attempt + 1,
                        issues=vf.issues,
                    )
                    # Phase 4: Publish repair started
                    self.event_bus.publish(Event(
                        type=EventType.REPAIR_STARTED,
                        data={"attempt": plan.current_repair_attempt + 1, "issues": vf.issues},
                        source="orchestrator",
                        plan_id=plan.plan_id,
                    ))
                    self.planner.plan_repair(
                        original_plan=plan,
                        validation_issues=vf.issues,
                        generated_code=ctx.get("extracted_code", ""),
                    )
                    # REPAIRING → GENERATING will happen on next loop iteration
                    sm.transition_to(AgentState.GENERATING, repair=True)
                    continue
                else:
                    # Accept with issues
                    ctx["validation_passed"] = False
                    ctx["validation_issues"] = vf.issues
                    logger.warning(
                        "Max repair attempts reached, accepting with issues",
                        plan_id=plan.plan_id,
                        issues=vf.issues,
                    )
            except Exception as e:
                step.fail(str(e))
                # Phase 4: Publish step failed + error
                self.event_bus.publish(Event(
                    type=EventType.STEP_FAILED,
                    data={"step_id": step.step_id, "action": step.action.value, "error": str(e)},
                    source="orchestrator",
                    plan_id=plan.plan_id,
                ))
                self.event_bus.publish(Event(
                    type=EventType.ERROR_OCCURRED,
                    data={"error": str(e), "step_id": step.step_id},
                    source="orchestrator",
                    plan_id=plan.plan_id,
                ))
                sm.fail(error=str(e))
                return GenerationResult(
                    success=False,
                    class_name=ctx.get("class_name", ""),
                    error=str(e),
                    plan_summary=plan.get_summary(),
                )

        # ── Terminal state ───────────────────────────────────────────
        if not sm.is_terminal:
            sm.transition_to(AgentState.COMPLETED)

        # Build validation summary
        validation_summary = None
        if ctx.get("validation_result"):
            validation_summary = ctx["validation_result"].get_summary()

        return GenerationResult(
            success=True,
            test_code=ctx.get("full_response"),
            class_name=ctx.get("class_name", ""),
            validation_passed=ctx.get("validation_passed", True),
            validation_issues=ctx.get("validation_issues", []),
            rag_chunks_used=len(ctx.get("rag_chunks", [])),
            tokens_used=ctx.get("tokens_used", 0),
            plan_summary=plan.get_summary(),
            validation_summary=validation_summary,
            repair_attempts=plan.current_repair_attempt,
        )

    # =====================================================================
    # Step executors (one per StepAction)
    # =====================================================================

    def _execute_step(
        self,
        sm: StateMachine,
        plan: ExecutionPlan,
        step: PlanStep,
        ctx: dict,
    ) -> None:
        """Dispatch a single plan step to its executor."""
        action = step.action

        if action == StepAction.EXTRACT_CLASS_INFO:
            self._step_extract_class_info(sm, plan, step, ctx)

        elif action == StepAction.RETRIEVE_CONTEXT:
            self._step_retrieve_context(sm, plan, step, ctx)

        elif action == StepAction.BUILD_PROMPT:
            self._step_build_prompt(sm, plan, step, ctx)

        elif action == StepAction.GENERATE_CODE:
            self._step_generate_code(sm, plan, step, ctx)

        elif action == StepAction.EXTRACT_CODE:
            self._step_extract_code(sm, plan, step, ctx)

        elif action == StepAction.VALIDATE_CODE:
            self._step_validate_code(sm, plan, step, ctx)

        elif action == StepAction.RECORD_SESSION:
            self._step_record_session(sm, plan, step, ctx)

        elif action == StepAction.REPAIR_CODE:
            self._step_repair_code(sm, plan, step, ctx)

        elif action == StepAction.ANALYZE_EXISTING_TEST:
            self._step_analyze_existing_test(sm, plan, step, ctx)

        elif action == StepAction.MERGE_TESTS:
            self._step_merge_tests(sm, plan, step, ctx)

        else:
            raise ValueError(f"Unknown step action: {action}")

    # ── Individual step implementations ──────────────────────────────

    def _step_extract_class_info(
        self, sm: StateMachine, plan: ExecutionPlan, step: PlanStep, ctx: dict
    ) -> None:
        class_name = step.params.get("class_name") or self._extract_class_name(
            step.params.get("file_path", plan.file_path)
        )
        if not class_name:
            raise ValueError("Could not determine class name from file path")

        ctx["class_name"] = class_name
        plan.class_name = class_name
        step.result = {"class_name": class_name}

    def _step_retrieve_context(
        self, sm: StateMachine, plan: ExecutionPlan, step: PlanStep, ctx: dict
    ) -> None:
        # Transition to RETRIEVING (only if not already in GENERATING for repair)
        if sm.state == AgentState.PLANNING:
            sm.transition_to(AgentState.RETRIEVING, class_name=ctx["class_name"])

        if plan.task_type == TaskType.REFINEMENT:
            # Load from session cache
            rag_chunks = self._get_cached_rag_context(ctx, step)
            ctx["rag_chunks"] = rag_chunks
            ctx["context_result"] = None

            # Try to enrich with context builder for refinement
            if self.context_builder and rag_chunks:
                try:
                    ctx["context_result"] = self.context_builder.build_refinement_context(
                        class_name=ctx["class_name"],
                        rag_chunks=rag_chunks,
                    )
                except Exception as e:
                    logger.warning("ContextBuilder refinement failed, using RAG-only", error=str(e))
        elif self.context_builder:
            # Phase 2 path: use ContextBuilder for full pipeline
            try:
                context_result = self.context_builder.build_context(
                    class_name=ctx["class_name"],
                    file_path=ctx.get("file_path", plan.file_path),
                    inline_source=step.params.get("inline_source"),
                    session=ctx.get("session"),
                    top_k=self.top_k,
                )
                ctx["rag_chunks"] = context_result.rag_chunks
                ctx["context_result"] = context_result
                logger.info(
                    "ContextBuilder result",
                    snippets=len(context_result.snippets),
                    intelligence=context_result.intelligence_available,
                    mock_types=context_result.mock_types,
                    elapsed_ms=round(context_result.elapsed_ms, 1),
                )
            except Exception as e:
                logger.warning("ContextBuilder failed, falling back to RAG-only", error=str(e))
                rag_chunks = self._get_rag_context(
                    class_name=ctx["class_name"],
                    file_path=ctx.get("file_path", plan.file_path),
                    session=ctx.get("session"),
                    inline_source=step.params.get("inline_source"),
                )
                ctx["rag_chunks"] = rag_chunks
                ctx["context_result"] = None
        else:
            # Legacy path: RAG-only
            rag_chunks = self._get_rag_context(
                class_name=ctx["class_name"],
                file_path=ctx.get("file_path", plan.file_path),
                session=ctx.get("session"),
                inline_source=step.params.get("inline_source"),
            )
            ctx["rag_chunks"] = rag_chunks
            ctx["context_result"] = None

        step.result = {"chunks_count": len(ctx.get("rag_chunks", []))}

        # Phase 4: Publish context retrieved
        self.event_bus.publish(Event(
            type=EventType.CONTEXT_RETRIEVED,
            data={
                "chunks_count": len(ctx.get("rag_chunks", [])),
                "intelligence": bool(ctx.get("context_result")),
            },
            source="orchestrator",
        ))

    def _step_build_prompt(
        self, sm: StateMachine, plan: ExecutionPlan, step: PlanStep, ctx: dict
    ) -> None:
        if plan.task_type == TaskType.REFINEMENT:
            feedback = step.params.get("feedback") or plan.metadata.get("feedback", "")
            original_code = step.params.get("original_code") or plan.metadata.get("original_code", "")
            _, issues = self.test_rules.validate_generated_code(original_code)

            ctx["system_prompt"] = self.prompt_builder.build_system_prompt()
            ctx["user_prompt"] = self.prompt_builder.build_refinement_prompt(
                original_code=original_code,
                feedback=feedback,
                validation_issues=issues,
                rag_chunks=ctx.get("rag_chunks", []),
            )
        elif plan.task_type == TaskType.INCREMENTAL_UPDATE:
            ctx["system_prompt"] = self.prompt_builder.build_system_prompt()
            ctx["user_prompt"] = self.prompt_builder.build_incremental_update_prompt(
                class_name=ctx["class_name"],
                file_path=ctx.get("file_path", plan.file_path),
                rag_chunks=ctx.get("rag_chunks", []),
                existing_test_code=plan.metadata.get("existing_test_code", ""),
                tested_methods=ctx.get("tested_methods", []),
                changed_methods=plan.metadata.get("changed_methods", []),
                task_description=plan.metadata.get("task_description"),
            )
        else:
            ctx["system_prompt"] = self.prompt_builder.build_system_prompt()
            ctx["user_prompt"] = self.prompt_builder.build_test_generation_prompt(
                class_name=ctx["class_name"],
                file_path=ctx.get("file_path", plan.file_path),
                rag_chunks=ctx.get("rag_chunks", []),
                task_description=plan.metadata.get("task_description"),
                session=ctx.get("session"),
            )

        # Record user message in session
        session = ctx.get("session")
        if session and plan.task_type == TaskType.TEST_GENERATION:
            session.add_user_message(
                plan.metadata.get("task_description") or f"Generate tests for {ctx['class_name']}",
                metadata={"file_path": plan.file_path},
            )
        elif session and plan.task_type == TaskType.REFINEMENT:
            session.add_user_message(
                plan.metadata.get("feedback", ""),
                metadata={"type": "refinement"},
            )

    def _step_generate_code(
        self, sm: StateMachine, plan: ExecutionPlan, step: PlanStep, ctx: dict
    ) -> None:
        # Transition to GENERATING if not already there (repair transitions already done)
        if sm.state not in (AgentState.GENERATING,):
            sm.transition_to(AgentState.GENERATING, class_name=ctx["class_name"])

        response = self.vllm.generate(
            system_prompt=ctx["system_prompt"],
            user_prompt=ctx["user_prompt"],
        )

        if not response.success:
            raise RuntimeError(f"LLM generation failed: {response.error}")

        ctx["full_response"] = response.content
        ctx["tokens_used"] = response.tokens_used
        step.result = {"tokens_used": response.tokens_used}

    def _step_extract_code(
        self, sm: StateMachine, plan: ExecutionPlan, step: PlanStep, ctx: dict
    ) -> None:
        extracted = self._extract_code(ctx["full_response"])
        ctx["extracted_code"] = extracted

    def _step_validate_code(
        self, sm: StateMachine, plan: ExecutionPlan, step: PlanStep, ctx: dict
    ) -> None:
        if sm.state != AgentState.VALIDATING:
            sm.transition_to(AgentState.VALIDATING)

        # Phase 3: use ValidationPipeline for severity-aware validation
        # Pass RAG chunks for construction pattern cross-check (Pass 7)
        rag_chunks = ctx.get("rag_chunks")
        validation_result = self.validator.validate(
            ctx.get("extracted_code", ""),
            rag_chunks=rag_chunks,
        )
        ctx["validation_result"] = validation_result
        ctx["validation_passed"] = validation_result.passed
        ctx["validation_issues"] = validation_result.error_messages

        logger.info(
            "Validation result",
            plan_id=plan.plan_id,
            passed=validation_result.passed,
            errors=len(validation_result.errors),
            warnings=len(validation_result.warnings),
            tests=validation_result.test_count,
        )

        # Phase 4: Publish validation completed
        self.event_bus.publish(Event(
            type=EventType.VALIDATION_COMPLETED,
            data={
                "passed": validation_result.passed,
                "errors": len(validation_result.errors),
                "warnings": len(validation_result.warnings),
                "test_count": validation_result.test_count,
            },
            source="orchestrator",
            plan_id=plan.plan_id,
        ))

        if not validation_result.passed:
            raise _ValidationFailed(
                validation_result.error_messages,
                validation_result=validation_result,
            )

    def _step_record_session(
        self, sm: StateMachine, plan: ExecutionPlan, step: PlanStep, ctx: dict
    ) -> None:
        session: Optional[SessionMemory] = ctx.get("session")
        if not session:
            return

        session.add_assistant_message(
            ctx.get("extracted_code", ""),
            metadata={"validation_passed": ctx.get("validation_passed", True)},
        )
        session.record_generated_test(
            class_name=ctx.get("class_name", plan.class_name),
            test_code=ctx.get("extracted_code", ""),
            success=ctx.get("validation_passed", True),
            feedback=plan.metadata.get("feedback"),
        )

    def _step_repair_code(
        self, sm: StateMachine, plan: ExecutionPlan, step: PlanStep, ctx: dict
    ) -> None:
        """Phase 3: Build a targeted repair prompt using RepairStrategySelector."""
        issues = step.params.get("validation_issues", [])
        generated_code = step.params.get("generated_code", ctx.get("extracted_code", ""))
        validation_result = ctx.get("validation_result")
        attempt = step.params.get("attempt", 1)

        # Build targeted repair plan if we have structured validation
        repair_section = ""
        if validation_result and isinstance(validation_result, ValidationResult):
            repair_plan = self.repair_selector.build_repair_plan(
                validation_result=validation_result,
                attempt_number=attempt,
                max_attempts=self.max_repair_attempts,
            )
            repair_section = repair_plan.get_repair_prompt_section()
            ctx["repair_plan"] = repair_plan

            logger.info(
                "Targeted repair plan built",
                plan_id=plan.plan_id,
                attempt=attempt,
                strategy=repair_plan.strategy,
                instructions=repair_plan.instruction_count,
            )

            # Phase 4: Publish repair completed (plan built)
            self.event_bus.publish(Event(
                type=EventType.REPAIR_COMPLETED,
                data={
                    "attempt": attempt,
                    "strategy": repair_plan.strategy,
                    "success": True,  # repair plan built, actual success determined later
                },
                source="orchestrator",
                plan_id=plan.plan_id,
            ))

        # Build repair prompt
        feedback_parts = []
        if repair_section:
            feedback_parts.append(repair_section)
        feedback_parts.append(
            f"Auto-repair attempt {attempt}: fix the following validation issues: {', '.join(issues)}"
        )

        ctx["system_prompt"] = self.prompt_builder.build_system_prompt()
        ctx["user_prompt"] = self.prompt_builder.build_refinement_prompt(
            original_code=generated_code,
            feedback="\n\n".join(feedback_parts),
            validation_issues=issues,
            rag_chunks=ctx.get("rag_chunks", []),
        )

    # =====================================================================
    # New step executors for incremental update
    # =====================================================================

    def _step_analyze_existing_test(
        self, sm: StateMachine, plan: ExecutionPlan, step: PlanStep, ctx: dict
    ) -> None:
        """Parse existing test code to find which methods are already tested."""
        existing_code = step.params.get("existing_test_code", "")
        if not existing_code:
            ctx["tested_methods"] = []
            step.result = {"tested_methods": []}
            return

        tested = self._extract_tested_methods(existing_code)
        ctx["tested_methods"] = tested
        step.result = {"tested_methods": tested}
        logger.info(
            "Analyzed existing test",
            class_name=ctx.get("class_name"),
            tested_methods=tested,
        )

    def _step_merge_tests(
        self, sm: StateMachine, plan: ExecutionPlan, step: PlanStep, ctx: dict
    ) -> None:
        """Merge newly generated tests into existing test class.

        Strategy: The LLM is instructed to output ONLY new test methods
        (not the full class), so we insert them into the existing class body.
        If the LLM returns a full class, we use it as-is (full replacement).
        """
        existing_code = step.params.get("existing_test_code", "")
        new_code = ctx.get("extracted_code", "")

        if not existing_code or not new_code:
            # Nothing to merge — use generated code as-is
            step.result = {"merged": False}
            return

        # Check if LLM returned a full class (has class declaration)
        if re.search(r'class\s+\w+Test\s*\{', new_code):
            # Full class returned — use as-is (full replacement)
            step.result = {"merged": False, "strategy": "full_replacement"}
            return

        # LLM returned only test methods — merge into existing class
        merged = self._merge_test_methods(existing_code, new_code)
        if merged:
            ctx["extracted_code"] = merged
            ctx["full_response"] = merged
            step.result = {"merged": True, "strategy": "method_insert"}
        else:
            step.result = {"merged": False, "strategy": "fallback"}

    @staticmethod
    def _extract_tested_methods(test_code: str) -> list[str]:
        """Extract method names being tested from test method names.

        Convention: testMethodName_WhenX_ShouldY or methodName_WhenX_ShouldY
        """
        tested = set()
        # Match @Test method names
        for match in re.finditer(
            r'(?:@Test|@DisplayName)\s*(?:\([^)]*\))?\s*\n\s*(?:public\s+|private\s+|protected\s+)?void\s+(\w+)',
            test_code,
        ):
            method_name = match.group(1)
            # Strip test prefix if present
            if method_name.startswith("test"):
                method_name = method_name[4:]
                if method_name:
                    method_name = method_name[0].lower() + method_name[1:]
            # Extract the base method name (before _When or _Should)
            base = method_name.split("_")[0] if "_" in method_name else method_name
            if base:
                tested.add(base)
        return sorted(tested)

    @staticmethod
    def _merge_test_methods(existing_code: str, new_methods: str) -> Optional[str]:
        """Insert new test methods into existing test class body."""
        # Find the last closing brace of the class
        last_brace = existing_code.rfind("}")
        if last_brace == -1:
            return None

        # Insert new methods before the final closing brace
        merged = (
            existing_code[:last_brace].rstrip()
            + "\n\n    // ── Incrementally generated tests ──────────────────────\n\n"
            + new_methods.strip()
            + "\n"
            + existing_code[last_brace:]
        )
        return merged

    # =====================================================================
    # Helpers (preserved from original orchestrator)
    # =====================================================================

    def _get_cached_rag_context(self, ctx: dict, step: PlanStep) -> list[CodeChunk]:
        """Load RAG context from session cache (for refinement)."""
        session: Optional[SessionMemory] = ctx.get("session")
        if not session:
            return []

        class_name = step.params.get("class_name") or ctx.get("class_name", "")
        cache_key = f"{class_name}:{session.current_file or ''}"
        cached = session.get_cached_rag_context(cache_key)
        if cached:
            return [CodeChunk(**c) for c in cached]
        return []

    def _get_rag_context(
        self,
        class_name: str,
        file_path: str,
        session: Optional[SessionMemory],
        inline_source: Optional[str] = None,
    ) -> list[CodeChunk]:
        """Exact graph traversal: fetch target class → deps → used_types.

        The Qdrant payload stores:
          - ``dependencies``  — FQNs of all referenced classes
          - ``used_types``    — simple names of domain model types
          - ``referenced_models`` — rich model info (fields, builders, etc.)

        We merge ``dependencies`` (converted to simple names) with
        ``used_types`` to determine which related classes to fetch.
        If both are empty, we fall back to inline source parsing.
        """
        cache_key = f"{class_name}:{file_path}"
        if session:
            cached = session.get_cached_rag_context(cache_key)
            if cached:
                logger.debug("Using cached RAG context", class_name=class_name)
                return [CodeChunk(**c) for c in cached]

        chunks: list[CodeChunk] = []
        existing_fqns: set[str] = set()

        # Query 1: fetch the target service chunk
        main_result = self.rag.search_by_class(
            class_name=class_name,
            top_k=1,
            include_dependencies=False,
        )
        main_chunk = main_result.chunks[0] if main_result.chunks else None
        if not main_chunk:
            logger.warning("Service target not found in index", class_name=class_name)
            return []
        chunks.append(main_chunk)
        existing_fqns.add(main_chunk.fully_qualified_name)

        # ── Collect types to fetch ──────────────────────────────────
        # dependencies are FQNs → extract simple name for search
        dep_simple_names: set[str] = set()
        for dep_fqn in (main_chunk.dependencies or []):
            simple = dep_fqn.rsplit(".", 1)[-1] if "." in dep_fqn else dep_fqn
            dep_simple_names.add(simple)

        # used_types are already simple names (added by build_index)
        used = set(main_chunk.used_types or [])

        # Merge both sets
        types_to_fetch = (dep_simple_names | used) - {main_chunk.class_name, class_name}

        # Fallback: parse inline source if Qdrant metadata is empty
        if not types_to_fetch and inline_source:
            fallback_types = self._extract_types_from_source(inline_source, class_name)
            if fallback_types:
                types_to_fetch = fallback_types
                logger.info(
                    "Using fallback type extraction from inline source",
                    class_name=class_name,
                    fallback_types=sorted(types_to_fetch),
                )

        if not types_to_fetch:
            logger.info(
                "RAG context retrieved (target only, no deps/used_types)",
                class_name=class_name,
            )
            return chunks

        # Parallel fetch for each dependency/used_type
        fetched_names: set[str] = set()
        unfound_types: set[str] = set()

        def _fetch_one(type_name: str) -> Optional[CodeChunk]:
            try:
                result = self.rag.search_by_class(
                    class_name=type_name,
                    top_k=1,
                    include_dependencies=False,
                )
                return result.chunks[0] if result.chunks else None
            except Exception as e:
                logger.warning(
                    "Failed to fetch dependency chunk",
                    type_name=type_name,
                    error=str(e),
                )
                return None

        with ThreadPoolExecutor(max_workers=min(5, len(types_to_fetch))) as executor:
            future_map = {
                executor.submit(_fetch_one, t): t for t in types_to_fetch
            }
            for future in as_completed(future_map):
                type_name = future_map[future]
                chunk = future.result()
                if chunk and chunk.fully_qualified_name not in existing_fqns:
                    chunks.append(chunk)
                    existing_fqns.add(chunk.fully_qualified_name)
                    fetched_names.add(type_name)
                elif not chunk:
                    unfound_types.add(type_name)
                    logger.warning(
                        "Type NOT found in Qdrant index — LLM will be told to mock it",
                        type_name=type_name,
                        class_name=class_name,
                    )

        # Attach unfound types to main_chunk metadata so prompt builder can warn LLM
        if unfound_types and main_chunk:
            main_chunk.unfound_types = sorted(unfound_types)

        # Cache in session
        if session:
            session.cache_rag_context(
                cache_key,
                [c.model_dump() for c in chunks],
            )

        logger.info(
            "RAG context retrieved (target+deps+used_types)",
            class_name=class_name,
            total_chunks=len(chunks),
            deps_count=len(dep_simple_names),
            used_types_count=len(used),
            types_requested=len(types_to_fetch),
            fetched=len(chunks) - 1,
            unfound=sorted(unfound_types) if unfound_types else [],
        )

        # Return ALL chunks — do NOT truncate with [:top_k].
        # Domain types are critical for correct construction patterns.
        # The prompt builder handles token budgeting separately.
        return chunks

    def _extract_types_from_source(self, source: str, class_name: str) -> set[str]:
        """Extract domain type names from inline Java source code."""
        types: set[str] = set()

        _SKIP_PREFIXES = {
            "java.", "javax.", "jakarta.",
            "org.springframework.", "org.junit.", "org.mockito.",
            "org.slf4j.", "org.apache.", "com.fasterxml.",
            "lombok.",
        }
        _SKIP_SIMPLE = {
            "String", "Integer", "Long", "Double", "Float", "Boolean",
            "UUID", "Map", "Set", "List", "Page", "Pageable", "Optional",
            "BigDecimal", "LocalDate", "LocalDateTime", "Instant",
            "ResponseEntity", "HttpStatus",
        }

        for line in source.split("\n"):
            line = line.strip()

            if line.startswith("import "):
                import_path = line.replace("import ", "").replace(";", "").strip()
                if any(import_path.startswith(p) for p in _SKIP_PREFIXES):
                    continue
                simple_name = import_path.split(".")[-1]
                if simple_name != class_name and simple_name not in _SKIP_SIMPLE:
                    types.add(simple_name)

            field_match = re.match(
                r'\s*(?:private|protected|public)?\s*(?:final\s+)?(\w+)\s+\w+\s*;',
                line,
            )
            if field_match:
                field_type = field_match.group(1)
                if (
                    field_type[0].isupper()
                    and field_type != class_name
                    and field_type not in _SKIP_SIMPLE
                ):
                    types.add(field_type)

        return types

    def _extract_class_name(self, file_path: str) -> Optional[str]:
        """Extract class name from file path."""
        file_name = file_path.replace("\\", "/").split("/")[-1]
        if file_name.endswith(".java"):
            return file_name[:-5]
        return None

    def _extract_code(self, response: str) -> str:
        """Extract Java code from LLM response."""
        code_block_pattern = r"```(?:java)?\s*\n(.*?)```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        if matches:
            return max(matches, key=len).strip()

        class_pattern = r"((?:import.*?\n)*\s*(?:@\w+.*?\n)*\s*(?:public\s+)?class\s+\w+.*?\{.*\})"
        class_match = re.search(class_pattern, response, re.DOTALL)

        if class_match:
            return class_match.group(1).strip()

        return response.strip()

    # ── Session / status helpers ─────────────────────────────────────

    def get_session_info(self, session_id: str) -> Optional[dict]:
        """Get information about a session."""
        session = self.memory_manager.get_session(session_id)
        if session:
            return session.get_session_summary()
        return None

    def cleanup_sessions(self) -> int:
        """Clean up expired sessions."""
        return self.memory_manager.cleanup_expired()


# ── Internal exceptions ──────────────────────────────────────────────

class _ValidationFailed(Exception):
    """Raised internally when code validation fails."""

    def __init__(self, issues: list[str], validation_result: Optional[ValidationResult] = None):
        self.issues = issues
        self.validation_result = validation_result
        super().__init__(f"Validation failed: {issues}")
