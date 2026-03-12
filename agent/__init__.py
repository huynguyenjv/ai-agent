"""Agent orchestration module.

Architecture (Phase 1-4 — Legacy):
  - StateMachine       — finite-state runtime (IDLE → PLAN → … → DONE)
  - Planner            — separates planning from execution
  - ExecutionPlan      — structured step list
  - Orchestrator       — executes plans via the state machine
  - ValidationPipeline — multi-pass severity-aware code validation
  - RepairStrategy     — targeted fix strategies per issue category
  - EventBus           — decoupled pub/sub for lifecycle events
  - MetricsCollector   — observability (timing, counters, rates)

Architecture (LangGraph — New):
  - AgentState / UnitTestState — typed state dicts
  - Supervisor        — regex-based intent classifier
  - SubGraphs         — task-specific StateGraphs (UnitTest, ...)
  - GraphOrchestrator — drop-in replacement for AgentOrchestrator
  - create_agent_graph — factory function for the compiled graph
"""

# ── Legacy exports (preserved for backward compat) ──
from .orchestrator import AgentOrchestrator, GenerationRequest, GenerationResult, StreamEvent, StreamPhase
from .plan import ExecutionPlan, PlanStep, StepAction, TaskType
from .planner import Planner
from .prompt import PromptBuilder
from .rules import TestRules
from .memory import SessionMemory, MemoryManager
from .memory_store import MemoryStore, InMemoryStore, RedisStore, create_memory_store
from .state_machine import AgentState as LegacyAgentState, StateMachine, TransitionError
from .validation import ValidationPipeline, ValidationResult, IssueSeverity, IssueCategory
from .repair import RepairStrategySelector, RepairPlan, RepairAction
from .events import EventBus, Event, EventType, get_event_bus, reset_event_bus
from .metrics import MetricsCollector

# ── LangGraph exports (new) ──
from .state import AgentState as GraphAgentState, UnitTestState
from .supervisor import classify_intent, supervisor_node
from .graph import create_agent_graph
from .graph_adapter import GraphOrchestrator, create_graph_orchestrator

__all__ = [
    # Core orchestration (legacy)
    "AgentOrchestrator",
    "GenerationRequest",
    "GenerationResult",
    "StreamEvent",
    "StreamPhase",
    # Phase 1: State machine + Planner (legacy)
    "LegacyAgentState",
    "StateMachine",
    "TransitionError",
    "Planner",
    "ExecutionPlan",
    "PlanStep",
    "StepAction",
    "TaskType",
    # Prompt & rules
    "PromptBuilder",
    "TestRules",
    # Memory
    "SessionMemory",
    "MemoryManager",
    "MemoryStore",
    "InMemoryStore",
    "RedisStore",
    "create_memory_store",
    # Phase 3: Validation + Repair
    "ValidationPipeline",
    "ValidationResult",
    "IssueSeverity",
    "IssueCategory",
    "RepairStrategySelector",
    "RepairPlan",
    "RepairAction",
    # Phase 4: Events + Metrics
    "EventBus",
    "Event",
    "EventType",
    "get_event_bus",
    "reset_event_bus",
    "MetricsCollector",
    # LangGraph (new)
    "GraphAgentState",
    "UnitTestState",
    "classify_intent",
    "supervisor_node",
    "create_agent_graph",
    "GraphOrchestrator",
    "create_graph_orchestrator",
]

