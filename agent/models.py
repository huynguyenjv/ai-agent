"""
Shared data models for the AI Agent.

Contains request/result/event types used by both LangGraph nodes
and the server API layer.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerationRequest:
    """Request for test generation."""

    file_path: str
    class_name: Optional[str] = None
    task_description: Optional[str] = None
    session_id: Optional[str] = None
    existing_test_code: Optional[str] = None
    changed_methods: Optional[list[str]] = None
    collection_name: Optional[str] = None
    source_code: Optional[str] = None
    # Two-Phase Strategy options
    force_two_phase: bool = False
    force_single_pass: bool = False
    complexity_threshold: int = 10


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
    validation_summary: Optional[dict] = None
    repair_attempts: int = 0
    strategy_used: str = "single_pass"
    complexity_score: int = 0
    analysis_result: Optional[dict] = None

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
    ANALYZING = "analyzing"
    REGISTRY_LOOKUP = "registry_lookup"
    METHOD_GENERATING = "method_generating"
    ASSEMBLING = "assembling"


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
