"""
State Machine Runtime for the AI Agent.

Defines the agent lifecycle as a finite state machine:
    IDLE → PLANNING → RETRIEVING → GENERATING → VALIDATING → DONE
                                                      ↓
                                                  REPAIRING ←→ VALIDATING
                                                      ↓
                                                   FAILED

Each state transition is explicit and logged, making the workflow
predictable, debuggable, and extensible.
"""

import time
from enum import Enum
from typing import Any, Callable, Optional

import structlog

from .plan import ExecutionPlan, PlanStep, StepAction, StepStatus, TaskType

logger = structlog.get_logger()


class AgentState(str, Enum):
    """All possible states of the agent runtime."""

    IDLE = "idle"
    PLANNING = "planning"
    RETRIEVING = "retrieving"
    GENERATING = "generating"
    VALIDATING = "validating"
    REPAIRING = "repairing"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Valid transitions ────────────────────────────────────────────────
# Maps current_state → set of allowed next states
VALID_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.IDLE: {AgentState.PLANNING, AgentState.FAILED},
    AgentState.PLANNING: {AgentState.RETRIEVING, AgentState.GENERATING, AgentState.FAILED},
    AgentState.RETRIEVING: {AgentState.GENERATING, AgentState.FAILED},
    AgentState.GENERATING: {AgentState.VALIDATING, AgentState.FAILED},
    AgentState.VALIDATING: {AgentState.COMPLETED, AgentState.REPAIRING, AgentState.FAILED},
    AgentState.REPAIRING: {AgentState.GENERATING, AgentState.FAILED},
    AgentState.COMPLETED: {AgentState.IDLE},
    AgentState.FAILED: {AgentState.IDLE},
}


class TransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    def __init__(self, from_state: AgentState, to_state: AgentState):
        self.from_state = from_state
        self.to_state = to_state
        allowed = VALID_TRANSITIONS.get(from_state, set())
        super().__init__(
            f"Invalid transition {from_state.value} → {to_state.value}. "
            f"Allowed: {', '.join(s.value for s in allowed)}"
        )


class StateTransition:
    """Record of a single state transition."""

    __slots__ = ("from_state", "to_state", "timestamp", "metadata")

    def __init__(
        self,
        from_state: AgentState,
        to_state: AgentState,
        metadata: Optional[dict] = None,
    ):
        self.from_state = from_state
        self.to_state = to_state
        self.timestamp = time.time()
        self.metadata = metadata or {}


class StateMachine:
    """Finite state machine that drives the agent runtime.

    Usage::

        sm = StateMachine()
        sm.transition_to(AgentState.PLANNING, plan=my_plan)
        sm.transition_to(AgentState.RETRIEVING)
        # …and so on

    The machine enforces valid transitions, logs every state change,
    and maintains a full transition history for debugging.
    """

    def __init__(self) -> None:
        self._state: AgentState = AgentState.IDLE
        self._plan: Optional[ExecutionPlan] = None
        self._history: list[StateTransition] = []
        self._on_enter_hooks: dict[AgentState, list[Callable]] = {}
        self._on_exit_hooks: dict[AgentState, list[Callable]] = {}
        self._error: Optional[str] = None
        self._started_at: Optional[float] = None

    # ── Properties ───────────────────────────────────────────────────

    @property
    def state(self) -> AgentState:
        """Current state."""
        return self._state

    @property
    def plan(self) -> Optional[ExecutionPlan]:
        """Current execution plan."""
        return self._plan

    @plan.setter
    def plan(self, plan: ExecutionPlan) -> None:
        self._plan = plan

    @property
    def error(self) -> Optional[str]:
        """Last error message, if any."""
        return self._error

    @property
    def history(self) -> list[StateTransition]:
        """Full transition history."""
        return list(self._history)

    @property
    def is_terminal(self) -> bool:
        """Whether the machine is in a terminal state."""
        return self._state in (AgentState.COMPLETED, AgentState.FAILED)

    @property
    def elapsed_ms(self) -> Optional[float]:
        """Elapsed time since the machine left IDLE, in milliseconds."""
        if self._started_at is None:
            return None
        return (time.time() - self._started_at) * 1000

    # ── Transitions ──────────────────────────────────────────────────

    def transition_to(
        self,
        new_state: AgentState,
        **metadata: Any,
    ) -> None:
        """Transition to a new state.

        Raises ``TransitionError`` if the transition is not allowed.
        """
        old_state = self._state

        # Validate
        allowed = VALID_TRANSITIONS.get(old_state, set())
        if new_state not in allowed:
            raise TransitionError(old_state, new_state)

        # Record
        transition = StateTransition(old_state, new_state, metadata=metadata)
        self._history.append(transition)

        # Track timing
        if old_state == AgentState.IDLE and new_state != AgentState.IDLE:
            self._started_at = time.time()

        # Run exit hooks
        for hook in self._on_exit_hooks.get(old_state, []):
            try:
                hook(old_state, new_state, metadata)
            except Exception as e:
                logger.warning("on_exit hook failed", hook=hook, error=str(e))

        # Transition
        self._state = new_state

        # Store error if transitioning to FAILED
        if new_state == AgentState.FAILED:
            self._error = metadata.get("error", "Unknown error")

        logger.info(
            "State transition",
            from_state=old_state.value,
            to_state=new_state.value,
            elapsed_ms=self.elapsed_ms,
            **{k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))},
        )

        # Run enter hooks
        for hook in self._on_enter_hooks.get(new_state, []):
            try:
                hook(old_state, new_state, metadata)
            except Exception as e:
                logger.warning("on_enter hook failed", hook=hook, error=str(e))

    def fail(self, error: str) -> None:
        """Shortcut to transition to FAILED state."""
        self.transition_to(AgentState.FAILED, error=error)

    def reset(self) -> None:
        """Reset machine to IDLE (only from terminal states)."""
        if self._state in (AgentState.COMPLETED, AgentState.FAILED):
            self.transition_to(AgentState.IDLE)
        else:
            # Force reset
            logger.warning("Force-resetting state machine", current_state=self._state.value)
            self._state = AgentState.IDLE

        self._plan = None
        self._error = None
        self._started_at = None

    # ── Hooks ────────────────────────────────────────────────────────

    def on_enter(self, state: AgentState, callback: Callable) -> None:
        """Register a callback to run when entering a state."""
        self._on_enter_hooks.setdefault(state, []).append(callback)

    def on_exit(self, state: AgentState, callback: Callable) -> None:
        """Register a callback to run when exiting a state."""
        self._on_exit_hooks.setdefault(state, []).append(callback)

    # ── Serialization ────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Get current machine status for debugging/API."""
        return {
            "state": self._state.value,
            "error": self._error,
            "elapsed_ms": self.elapsed_ms,
            "transition_count": len(self._history),
            "plan_summary": self._plan.get_summary() if self._plan else None,
            "history": [
                {
                    "from": t.from_state.value,
                    "to": t.to_state.value,
                    "timestamp": t.timestamp,
                }
                for t in self._history[-10:]  # last 10 transitions
            ],
        }
