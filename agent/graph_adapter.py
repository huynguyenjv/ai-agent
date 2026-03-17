"""
Graph adapter — bridges the LangGraph agent with existing API models.

Provides ``create_graph_orchestrator()`` and ``GraphOrchestrator`` which
implement the same interface as ``AgentOrchestrator`` (generate_test,
generate_test_streaming) but delegate to the LangGraph StateGraph.

This allows server/api.py to swap between legacy and graph backends
with minimal changes.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Optional, Generator, AsyncGenerator, cast

import structlog

from agent.graph import create_agent_graph
from agent.state import UnitTestState
from agent.models import StreamEvent, StreamPhase
from rag.client import RAGClient
from vllm.client import VLLMClient

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════
# Reuse existing data classes for backward compatibility
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GraphGenerationRequest:
    """Maps 1:1 with the existing GenerationRequest fields."""
    file_path: str
    class_name: Optional[str] = None
    task_description: Optional[str] = None
    session_id: Optional[str] = None
    existing_test_code: Optional[str] = None
    changed_methods: Optional[list[str]] = None
    collection_name: Optional[str] = None
    source_code: Optional[str] = None
    force_two_phase: bool = False
    force_single_pass: bool = False
    complexity_threshold: int = 10
    require_human_review: bool = False


@dataclass
class GraphGenerationResult:
    """Maps 1:1 with the existing GenerationResult fields."""
    success: bool = False
    test_code: Optional[str] = None
    class_name: str = ""
    validation_passed: bool = True
    validation_issues: list[str] = field(default_factory=list)
    error: Optional[str] = None
    rag_chunks_used: int = 0
    tokens_used: int = 0
    strategy_used: str = "single_pass"
    complexity_score: int = 0
    repair_attempts: int = 0
    analysis_result: Optional[dict] = None
    run_id: Optional[str] = None  # For human review tracking


class GraphOrchestrator:
    """Drop-in replacement for AgentOrchestrator, backed by LangGraph.

    Implements the same public API:
      - generate_test(request) → result
      - generate_test_streaming(request) → Generator[StreamEvent]
      - refine_test(session_id, feedback) → result
    """

    def __init__(
        self,
        rag_client,
        vllm_client,
        *,
        repo_path: Optional[str] = None,
        token_budget: int = 6000,
        max_repair_attempts: int = 3,
        checkpoint_db: str = "checkpoints.db",
    ):
        self.rag = rag_client
        self.vllm = vllm_client
        self.max_repair_attempts = max_repair_attempts

        # Import existing modules
        from agent.prompt import PromptBuilder
        from agent.validation import ValidationPipeline
        from agent.repair import RepairStrategySelector
        from agent.memory import MemoryManager
        from agent.events import get_event_bus
        from agent.metrics import MetricsCollector

        self.prompt_builder = PromptBuilder()
        self.validator = ValidationPipeline()
        self.repair_selector = RepairStrategySelector()
        self.memory_manager = MemoryManager()
        self.event_bus = get_event_bus()
        self.metrics = MetricsCollector(self.event_bus)

        # Optional components
        self.context_builder = None
        self.two_phase_strategy = None
        self.domain_registry = None

        try:
            from context.context_builder import ContextBuilder
            if ContextBuilder:
                self.context_builder = ContextBuilder(
                    rag_client=rag_client,
                    repo_path=repo_path,
                    token_budget=token_budget,
                )
                if repo_path:
                    try:
                        self.context_builder.init_intelligence(repo_path)
                    except Exception:
                        pass
        except ImportError:
            pass

        try:
            from agent.two_phase_strategy import (
                TwoPhaseStrategy, TwoPhaseConfig,
                ComplexityCalculator,
            )
            # get_domain_registry may not exist in all versions
            try:
                from agent.two_phase_strategy import get_domain_registry
                self.domain_registry = get_domain_registry(
                    qdrant_client=rag_client.qdrant,
                    default_collection="java_codebase",
                )
            except ImportError:
                self.domain_registry = None
            self.two_phase_strategy = TwoPhaseStrategy(
                vllm_client=vllm_client,
                domain_registry=self.domain_registry,
                prompt_builder=self.prompt_builder,
                test_rules=None,
                rag_client=rag_client,
                config=TwoPhaseConfig(),
            )
        except (ImportError, Exception) as e:
            logger.warning("Two-Phase Strategy unavailable for graph", error=str(e))

        # Cache service
        try:
            from utils.cache_service import get_cache_service
            self.cache_service = get_cache_service()
        except ImportError:
            self.cache_service = None

        # LangGraph instance (initialized via initialize())
        self._graph = None

    async def initialize(self, checkpoint_db: str = "checkpoints.db"):
        """Initialize the LangGraph instance asynchronously.
        
        Required because AsyncSqliteSaver requires async setup.
        """
        if self._graph is not None:
            return

        from agent.graph import create_agent_graph
        self._graph = await create_agent_graph(
            rag_client=self.rag,
            vllm_client=self.vllm,
            prompt_builder=self.prompt_builder,
            validation_pipeline=self.validator,
            repair_selector=self.repair_selector,
            context_builder=self.context_builder,
            two_phase_strategy=self.two_phase_strategy,
            domain_registry=self.domain_registry,
            memory_manager=self.memory_manager,
            cache_service=self.cache_service,
            checkpoint_db=checkpoint_db,
        )
        logger.info("GraphOrchestrator initialized (LangGraph backend)")

    def _build_input_state(self, request, run_id: str) -> dict:
        """Build LangGraph input state from a request object."""
        class_name = getattr(request, "class_name", None)
        if not class_name:
            fp = getattr(request, "file_path", "")
            name = fp.replace("\\", "/").rsplit("/", 1)[-1]
            class_name = name[:-5] if name.endswith(".java") else name

        return {
            "user_input": getattr(request, "task_description", None) or f"Generate tests for {class_name}",
            "session_id": getattr(request, "session_id", None) or run_id,
            "file_path": getattr(request, "file_path", ""),
            "class_name": class_name,
            "task_description": getattr(request, "task_description", None) or "",
            "collection_name": getattr(request, "collection_name", None),
            "source_code": getattr(request, "source_code", None) or "",
            "existing_test_code": getattr(request, "existing_test_code", None) or "",
            "changed_methods": getattr(request, "changed_methods", None) or [],
            "require_human_review": getattr(request, "require_human_review", False),
            "force_two_phase": getattr(request, "force_two_phase", False),
            "force_single_pass": getattr(request, "force_single_pass", False),
            "complexity_threshold": getattr(request, "complexity_threshold", 10),
            "retry_count": 0,
            "max_retries": self.max_repair_attempts,
            "repo_path": getattr(request, "repo_path", None) or os.getenv("JAVA_REPO_PATH", ""),
        }

    async def generate_test(self, request) -> GraphGenerationResult:
        """Generate unit tests via LangGraph (async).

        Accepts either GraphGenerationRequest or the legacy GenerationRequest.
        Handles GraphInterrupt — returns partial result with run_id for resume.
        """
        run_id = str(uuid.uuid4())
        input_state = self._build_input_state(request, run_id)
        class_name = input_state["class_name"]
        config = {"configurable": {"thread_id": run_id}}

        try:
            result_state = await self._graph.ainvoke(input_state, config=config)

            # Parse subgraph result
            subgraph_json = result_state.get("subgraph_result", "{}")
            try:
                result_data = json.loads(subgraph_json) if isinstance(subgraph_json, str) else subgraph_json
            except (json.JSONDecodeError, TypeError):
                result_data = {}

            return GraphGenerationResult(
                success=bool(result_data.get("success", True)),
                test_code=str(result_data.get("test_code", result_state.get("final_test_code", ""))) or None,
                class_name=str(result_data.get("class_name", class_name)),
                validation_passed=bool(result_data.get("validation_passed", True)),
                validation_issues=cast(list[str], result_data.get("validation_issues", [])),
                error=str(result_data.get("error", "")) or None,
                rag_chunks_used=int(result_data.get("rag_chunks_used", 0)),
                tokens_used=int(result_data.get("tokens_used", 0)),
                strategy_used=str(result_data.get("strategy_used", "single_pass")),
                complexity_score=int(result_data.get("complexity_score", 0)),
                repair_attempts=int(result_data.get("repair_attempts", 0)),
                analysis_result=cast(Optional[dict], result_data.get("analysis_result")) if isinstance(result_data.get("analysis_result"), dict) else None,
                run_id=run_id,
            )

        except Exception as e:
            # Check if it's a GraphInterrupt (human review pause)
            if "GraphInterrupt" in type(e).__name__ or "interrupt" in str(type(e)).lower():
                logger.info("Graph interrupted for human review", run_id=run_id)
                # Get partial state — return test_code generated so far
                state = await self.get_run_state(run_id)
                test_code = ""
                if state:
                    test_code = state.get("test_code", "")
                return GraphGenerationResult(
                    success=bool(test_code),
                    test_code=test_code,
                    class_name=class_name,
                    error="awaiting_human_review" if not test_code else None,
                    run_id=run_id,
                )

            logger.error("Graph execution failed", error=str(e), run_id=run_id)
            return GraphGenerationResult(
                success=False,
                error=str(e),
                class_name=class_name,
                run_id=run_id,
            )

    async def refine_test(self, session_id: str, feedback: str) -> GraphGenerationResult:
        """Refine a previously generated test based on feedback (async)."""
        config = {"configurable": {"thread_id": session_id}}
        
        # Get current state to preserve context
        current_state = await self.get_run_state(session_id)
        if not current_state:
            return GraphGenerationResult(success=False, error="Session not found", run_id=session_id)

        # Update state with feedback
        input_update = {
            "human_feedback": feedback,
            "human_approved": False, # Treat refinement as a "not yet approved" state
            "retry_count": 0, # Reset retry count for fresh refinement
        }

        try:
            # resume the graph with feedback
            result_state = await self._graph.ainvoke(input_update, config=config)
            
            subgraph_json = result_state.get("subgraph_result", "{}")
            try:
                result_data = json.loads(subgraph_json) if isinstance(subgraph_json, str) else subgraph_json
            except (json.JSONDecodeError, TypeError):
                result_data = {}

            return GraphGenerationResult(
                success=bool(result_data.get("success", True)),
                test_code=str(result_data.get("test_code", result_state.get("final_test_code", ""))) or None,
                class_name=str(result_data.get("class_name", current_state.get("class_name", ""))),
                validation_passed=bool(result_data.get("validation_passed", True)),
                validation_issues=cast(list[str], result_data.get("validation_issues", [])),
                error=str(result_data.get("error", "")) or None,
                run_id=session_id,
                tokens_used=int(result_data.get("tokens_used", 0)),
            )
        except Exception as e:
            logger.error("Refine test failed", error=str(e), session_id=session_id)
            return GraphGenerationResult(success=False, error=str(e), run_id=session_id)

    # ── Node name → StreamPhase mapping ─────────────────────────────
    _NODE_PHASE_MAP = {
        "supervisor": StreamPhase.PLANNING,
        "retrieve": StreamPhase.RETRIEVING,
        "check_strategy": StreamPhase.PLANNING,
        "analyze": StreamPhase.ANALYZING,
        "build_prompt": StreamPhase.PLANNING,
        "call_llm": StreamPhase.GENERATING,
        "validate": StreamPhase.VALIDATING,
        "repair": StreamPhase.REPAIRING,
        "human_review": StreamPhase.VALIDATING,
        "save_result": StreamPhase.DONE,
    }

    _NODE_MSG_MAP = {
        "supervisor": "🔍 Classifying request intent...",
        "retrieve": "📚 Retrieving RAG context...",
        "check_strategy": "🧮 Evaluating complexity and choosing strategy...",
        "analyze": "🔬 Phase 1: Analyzing service structure...",
        "build_prompt": "📝 Building generation prompt...",
        "call_llm": "🤖 Generating test code...",
        "validate": "✅ Running 7-pass validation...",
        "repair": "🔧 Repairing validation issues...",
        "human_review": "👤 Submitting for review...",
        "save_result": "💾 Saving result...",
    }

    async def generate_test_streaming(self, request) -> AsyncGenerator[StreamEvent]:
        """Stream test generation via LangGraph's .astream() API.

        Yields StreamEvent objects compatible with the existing
        ``_stream_test_generation`` SSE pipeline in server/api.py.
        """
        run_id = str(uuid.uuid4())
        input_state = self._build_input_state(request, run_id)
        class_name = input_state["class_name"]
        config = {"configurable": {"thread_id": run_id}}

        rag_chunks_used = 0
        tokens_used = 0
        validation_passed = True
        validation_issues = []
        test_code = ""
        final_error = None
        strategy_used = "single_pass"

        try:
            async for event in self._graph.astream(
                input_state,
                config=config,
                stream_mode="updates",
                subgraphs=True,
            ):
                # With subgraphs=True, event is a tuple:
                #   (namespace_tuple, {node_name: state_update_dict})
                if isinstance(event, tuple) and len(event) == 2:
                    namespace, update_dict = event
                else:
                    # Fallback: treat as normal dict
                    namespace = ()
                    update_dict = event if isinstance(event, dict) else {}

                if not isinstance(update_dict, dict):
                    continue

                for node_name, state_update in update_dict.items():
                    if not isinstance(state_update, dict):
                        continue

                    phase = self._NODE_PHASE_MAP.get(node_name, StreamPhase.PLANNING)
                    msg = self._NODE_MSG_MAP.get(node_name, f"Running {node_name}...")

                    # Emit progress message
                    yield StreamEvent(
                        phase=phase,
                        content=f"> {msg}\n\n",
                        delta=False,
                    )

                    # Track key state values as they flow through
                    if "rag_chunks" in state_update and state_update["rag_chunks"]:
                        rag_chunks_used = len(state_update["rag_chunks"])

                    if "tokens_used" in state_update:
                        tokens_used = state_update.get("tokens_used", 0) or 0

                    if "strategy" in state_update and state_update["strategy"]:
                        strategy_used = state_update["strategy"]

                    if "validation_passed" in state_update:
                        validation_passed = state_update["validation_passed"]

                    if "validation_issues" in state_update:
                        validation_issues = state_update["validation_issues"] or []

                    if "error" in state_update and state_update["error"]:
                        final_error = state_update["error"]

                    # When call_llm finishes, stream the generated code
                    if node_name == "call_llm" and "test_code" in state_update:
                        code = state_update.get("test_code", "")
                        if code:
                            test_code = code
                            yield StreamEvent(
                                phase=StreamPhase.GENERATING,
                                content="```java\n",
                                delta=True,
                            )
                            chunk_size = 60
                            for i in range(0, len(code), chunk_size):
                                yield StreamEvent(
                                    phase=StreamPhase.GENERATING,
                                    content=code[i:i + chunk_size],
                                    delta=True,
                                )
                            yield StreamEvent(
                                phase=StreamPhase.GENERATING,
                                content="\n```\n\n",
                                delta=True,
                            )

                    # When repair finishes, stream the repaired code
                    if node_name == "repair" and "test_code" in state_update:
                        new_code = state_update.get("test_code", "")
                        if new_code and new_code != test_code:
                            test_code = new_code
                            yield StreamEvent(
                                phase=StreamPhase.REPAIRING,
                                content="\n\n> 🔧 **Repaired code:**\n\n```java\n",
                                delta=False,
                            )
                            chunk_size = 60
                            for i in range(0, len(new_code), chunk_size):
                                yield StreamEvent(
                                    phase=StreamPhase.GENERATING,
                                    content=new_code[i:i + chunk_size],
                                    delta=True,
                                )
                            yield StreamEvent(
                                phase=StreamPhase.GENERATING,
                                content="\n```\n\n",
                                delta=True,
                            )

            # Done — emit final event with metadata
            if final_error:
                yield StreamEvent(
                    phase=StreamPhase.ERROR,
                    content=f"Error: {final_error}",
                )
            else:
                yield StreamEvent(
                    phase=StreamPhase.DONE,
                    content="",
                    metadata={
                        "validation_passed": validation_passed,
                        "validation_issues": validation_issues,
                        "rag_chunks_used": rag_chunks_used,
                        "tokens_used": tokens_used,
                        "class_name": class_name,
                        "strategy_used": strategy_used,
                    },
                )

        except Exception as e:
            logger.error("Graph streaming failed", error=str(e), run_id=run_id)
            yield StreamEvent(
                phase=StreamPhase.ERROR,
                content=f"Error: {e}",
            )

    async def get_run_state(self, run_id: str) -> Optional[dict]:
        """Get the state of a graph run (for polling/review endpoints)."""
        config = {"configurable": {"thread_id": run_id}}
        try:
            state = await self._graph.aget_state(config)
            return state.values if state else None
        except Exception as e:
            logger.error("Failed to get run state", run_id=run_id, error=str(e))
            return None

    async def submit_review(self, run_id: str, approved: bool, feedback: str = "") -> GraphGenerationResult:
        """Resume a paused graph after human review (async)."""
        config = {"configurable": {"thread_id": run_id}}

        try:
            # Get current state to preserve class_name
            current_state = await self.get_run_state(run_id)
            initial_class_name = current_state.get("class_name", "") if current_state else ""

            # Update state with review decision
            await self._graph.aupdate_state(
                config,
                {"human_approved": approved, "human_feedback": feedback},
            )

            # Resume execution
            result_state = await self._graph.ainvoke(None, config=config)

            subgraph_json = result_state.get("subgraph_result", "{}")
            try:
                result_data = json.loads(subgraph_json) if isinstance(subgraph_json, str) else subgraph_json
            except (json.JSONDecodeError, TypeError):
                result_data = {}

            return GraphGenerationResult(
                success=bool(result_data.get("success", True)),
                test_code=str(result_data.get("test_code", "")) or None,
                class_name=str(result_data.get("class_name", initial_class_name)),
                validation_passed=bool(result_data.get("validation_passed", True)),
                validation_issues=cast(list[str], result_data.get("validation_issues", [])),
                error=str(result_data.get("error", "")) or None,
                run_id=run_id,
            )

        except Exception as e:
            logger.error("Review resume failed", run_id=run_id, error=str(e))
            return GraphGenerationResult(success=False, error=str(e), run_id=run_id)


async def create_graph_orchestrator(
    rag_client=None,
    vllm_client=None,
) -> GraphOrchestrator:
    """Factory function matching create_orchestrator() signature.

    Uses same env vars as the legacy create_orchestrator().
    """
    if rag_client is None:
        from rag.client import RAGClient
        rag_client = RAGClient(
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            collection_name=os.getenv("QDRANT_COLLECTION", "java_codebase"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2-onnx"),
        )

    if vllm_client is None:
        from vllm.client import VLLMClient
        vllm_client = VLLMClient(
            base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("VLLM_API_KEY", "token-abc123"),
            model=os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"),
        )

    orchestrator = GraphOrchestrator(
        rag_client=rag_client,
        vllm_client=vllm_client,
        repo_path=os.getenv("JAVA_REPO_PATH") or None,
    )
    
    await orchestrator.initialize(
        checkpoint_db=os.getenv("LANGGRAPH_CHECKPOINT_DB", "checkpoints.db")
    )
    
    return orchestrator
