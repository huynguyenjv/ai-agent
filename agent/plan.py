"""
Execution plan data structures for the agent workflow.

An ExecutionPlan is a structured representation of what the agent
intends to do, produced by the Planner before any code generation.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class TaskType(str, Enum):
    """Type of task the agent is performing."""

    TEST_GENERATION = "test_generation"
    REFINEMENT = "refinement"
    GENERAL_CHAT = "general_chat"


class StepAction(str, Enum):
    """Actions that a plan step can perform."""

    EXTRACT_CLASS_INFO = "extract_class_info"
    RETRIEVE_CONTEXT = "retrieve_context"
    BUILD_PROMPT = "build_prompt"
    GENERATE_CODE = "generate_code"
    VALIDATE_CODE = "validate_code"
    EXTRACT_CODE = "extract_code"
    RECORD_SESSION = "record_session"
    REPAIR_CODE = "repair_code"


class StepStatus(str, Enum):
    """Status of a plan step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class PlanStep:
    """A single step in an execution plan."""

    step_id: int
    action: StepAction
    description: str
    params: dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def start(self) -> None:
        """Mark step as in-progress."""
        self.status = StepStatus.IN_PROGRESS
        self.started_at = time.time()

    def complete(self, result: Any = None) -> None:
        """Mark step as completed with optional result."""
        self.status = StepStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()

    def fail(self, error: str) -> None:
        """Mark step as failed."""
        self.status = StepStatus.FAILED
        self.error = error
        self.completed_at = time.time()

    def skip(self, reason: str = "") -> None:
        """Mark step as skipped."""
        self.status = StepStatus.SKIPPED
        self.error = reason
        self.completed_at = time.time()

    @property
    def duration_ms(self) -> Optional[float]:
        """Duration in milliseconds, if completed."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return None


@dataclass
class ExecutionPlan:
    """A structured plan for the agent to execute.

    Produced by the Planner, consumed by the StateMachine/Orchestrator.
    """

    plan_id: str = field(default_factory=lambda: f"plan-{uuid.uuid4().hex[:8]}")
    task_type: TaskType = TaskType.TEST_GENERATION
    class_name: str = ""
    file_path: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    max_repair_attempts: int = 2
    current_repair_attempt: int = 0

    # Metadata collected during planning
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(
        self,
        action: StepAction,
        description: str,
        **params: Any,
    ) -> PlanStep:
        """Add a step to the plan."""
        step = PlanStep(
            step_id=len(self.steps) + 1,
            action=action,
            description=description,
            params=params,
        )
        self.steps.append(step)
        return step

    def get_next_pending_step(self) -> Optional[PlanStep]:
        """Get the next pending step to execute."""
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                return step
        return None

    def get_step_by_action(self, action: StepAction) -> Optional[PlanStep]:
        """Get the first step with a given action."""
        for step in self.steps:
            if step.action == action:
                return step
        return None

    @property
    def is_complete(self) -> bool:
        """Check if all steps are completed or skipped."""
        return all(
            s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
            for s in self.steps
        )

    @property
    def has_failed(self) -> bool:
        """Check if any step has failed."""
        return any(s.status == StepStatus.FAILED for s in self.steps)

    @property
    def can_repair(self) -> bool:
        """Check if repair is still possible (hasn't exceeded max attempts)."""
        return self.current_repair_attempt < self.max_repair_attempts

    def get_summary(self) -> dict:
        """Get a summary of the plan execution."""
        return {
            "plan_id": self.plan_id,
            "task_type": self.task_type.value,
            "class_name": self.class_name,
            "total_steps": len(self.steps),
            "completed": sum(1 for s in self.steps if s.status == StepStatus.COMPLETED),
            "failed": sum(1 for s in self.steps if s.status == StepStatus.FAILED),
            "skipped": sum(1 for s in self.steps if s.status == StepStatus.SKIPPED),
            "repair_attempts": self.current_repair_attempt,
            "is_complete": self.is_complete,
            "has_failed": self.has_failed,
        }
