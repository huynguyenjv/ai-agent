"""Agent orchestration module.

Architecture (Phase 1-4):
  - StateMachine       — finite-state runtime (IDLE → PLAN → … → DONE)
  - Planner            — separates planning from execution
  - ExecutionPlan      — structured step list
  - Orchestrator       — executes plans via the state machine
  - ValidationPipeline — multi-pass severity-aware code validation
  - RepairStrategy     — targeted fix strategies per issue category
  - EventBus           — decoupled pub/sub for lifecycle events
  - MetricsCollector   — observability (timing, counters, rates)
"""

from .orchestrator import AgentOrchestrator, GenerationRequest, GenerationResult
from .plan import ExecutionPlan, PlanStep, StepAction, TaskType
from .planner import Planner
from .prompt import PromptBuilder
from .rules import TestRules
from .memory import SessionMemory, MemoryManager
from .memory_store import MemoryStore, InMemoryStore, RedisStore, create_memory_store
from .state_machine import AgentState, StateMachine, TransitionError
from .validation import ValidationPipeline, ValidationResult, IssueSeverity, IssueCategory
from .repair import RepairStrategySelector, RepairPlan, RepairAction
from .events import EventBus, Event, EventType, get_event_bus, reset_event_bus
from .metrics import MetricsCollector

__all__ = [
    # Core orchestration
    "AgentOrchestrator",
    "GenerationRequest",
    "GenerationResult",
    # Phase 1: State machine + Planner
    "AgentState",
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
]

