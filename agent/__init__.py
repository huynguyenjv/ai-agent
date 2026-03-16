"""Agent orchestration module.

Architecture (LangGraph):
  - AgentState / UnitTestState — typed state dicts
  - Supervisor        — regex-based intent classifier
  - SubGraphs         — task-specific StateGraphs (UnitTest, ...)
  - GraphOrchestrator — LangGraph backend
  - create_agent_graph — factory function for the compiled graph
"""

# ── Shared Models ──
from .models import GenerationRequest, GenerationResult, StreamEvent, StreamPhase

# ── Support components ──
from .prompt import PromptBuilder
from .rules import TestRules
from .memory import SessionMemory, MemoryManager
from .memory_store import MemoryStore, InMemoryStore, RedisStore, create_memory_store
from .validation import ValidationPipeline, ValidationResult, IssueSeverity, IssueCategory
from .repair import RepairStrategySelector, RepairPlan, RepairAction
from .events import EventBus, Event, EventType, get_event_bus, reset_event_bus
from .metrics import MetricsCollector

# ── LangGraph Architecture ──
from .state import AgentState, UnitTestState
from .supervisor import classify_intent, supervisor_node
from .graph import create_agent_graph
from .graph_adapter import GraphOrchestrator, create_graph_orchestrator

__all__ = [
    # Models
    "GenerationRequest",
    "GenerationResult",
    "StreamEvent",
    "StreamPhase",
    
    # Support
    "PromptBuilder",
    "TestRules",
    "SessionMemory",
    "MemoryManager",
    "MemoryStore",
    "InMemoryStore",
    "RedisStore",
    "create_memory_store",
    "ValidationPipeline",
    "ValidationResult",
    "IssueSeverity",
    "IssueCategory",
    "RepairStrategySelector",
    "RepairPlan",
    "RepairAction",
    "EventBus",
    "Event",
    "EventType",
    "get_event_bus",
    "reset_event_bus",
    "MetricsCollector",
    
    # LangGraph
    "AgentState",
    "UnitTestState",
    "classify_intent",
    "supervisor_node",
    "create_agent_graph",
    "GraphOrchestrator",
    "create_graph_orchestrator",
]

