"""Quick smoke test for Phase 1 components.

Uses importlib.util to load modules from file paths directly,
bypassing the agent/__init__.py which triggers heavy torch imports.
"""

import sys
import os
import importlib.util

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register so cross-imports work
    spec.loader.exec_module(mod)
    return mod


# Load modules directly from file — register under dotted names so
# relative imports within the agent package still work.
# We need a fake "agent" package module to satisfy "from .plan import ..."
import types

agent_pkg = types.ModuleType("agent")
agent_pkg.__path__ = [os.path.join(base, "agent")]
agent_pkg.__package__ = "agent"
sys.modules["agent"] = agent_pkg

# Stub structlog to avoid needing the heavy dependency chain
structlog_stub = types.ModuleType("structlog")
structlog_stub.get_logger = lambda: type("Logger", (), {
    "info": lambda self, *a, **kw: None,
    "debug": lambda self, *a, **kw: None,
    "warning": lambda self, *a, **kw: None,
    "error": lambda self, *a, **kw: None,
})()
sys.modules["structlog"] = structlog_stub

plan_mod = _load("agent.plan", os.path.join(base, "agent", "plan.py"))
sm_mod = _load("agent.state_machine", os.path.join(base, "agent", "state_machine.py"))
planner_mod = _load("agent.planner", os.path.join(base, "agent", "planner.py"))

ExecutionPlan = plan_mod.ExecutionPlan
PlanStep = plan_mod.PlanStep
StepAction = plan_mod.StepAction
StepStatus = plan_mod.StepStatus
TaskType = plan_mod.TaskType

AgentState = sm_mod.AgentState
StateMachine = sm_mod.StateMachine
TransitionError = sm_mod.TransitionError
VALID_TRANSITIONS = sm_mod.VALID_TRANSITIONS

Planner = planner_mod.Planner

# ====================================================================
# Test 1: Plan module
# ====================================================================

plan = ExecutionPlan(
    task_type=TaskType.TEST_GENERATION,
    class_name="UserService",
    file_path="src/main/java/com/example/UserService.java",
)
plan.add_step(StepAction.EXTRACT_CLASS_INFO, "Resolve class name", file_path="test.java")
plan.add_step(StepAction.RETRIEVE_CONTEXT, "Fetch RAG context")
plan.add_step(StepAction.BUILD_PROMPT, "Build prompts")
plan.add_step(StepAction.GENERATE_CODE, "Call LLM")
plan.add_step(StepAction.VALIDATE_CODE, "Validate code")

assert len(plan.steps) == 5
assert plan.task_type == TaskType.TEST_GENERATION
assert not plan.is_complete
assert not plan.has_failed

# Execute steps
for step in plan.steps:
    step.start()
    step.complete(result="ok")

assert plan.is_complete
print(f"[PASS] Plan module: {plan.plan_id}, {len(plan.steps)} steps, complete={plan.is_complete}")

# Test 2: State Machine module
sm = StateMachine()
assert sm.state == AgentState.IDLE

sm.transition_to(AgentState.PLANNING)
assert sm.state == AgentState.PLANNING

sm.transition_to(AgentState.RETRIEVING)
assert sm.state == AgentState.RETRIEVING

sm.transition_to(AgentState.GENERATING)
assert sm.state == AgentState.GENERATING

sm.transition_to(AgentState.VALIDATING)
assert sm.state == AgentState.VALIDATING

sm.transition_to(AgentState.COMPLETED)
assert sm.state == AgentState.COMPLETED
assert sm.is_terminal

print(f"[PASS] State machine: happy path IDLE->PLAN->RETRIEVE->GEN->VALIDATE->DONE")

# Test invalid transition
sm2 = StateMachine()
try:
    sm2.transition_to(AgentState.GENERATING)  # IDLE -> GENERATING not allowed
    assert False, "Should have raised TransitionError"
except TransitionError as e:
    print(f"[PASS] Invalid transition caught: {e}")

# Test repair loop
sm3 = StateMachine()
sm3.transition_to(AgentState.PLANNING)
sm3.transition_to(AgentState.RETRIEVING)
sm3.transition_to(AgentState.GENERATING)
sm3.transition_to(AgentState.VALIDATING)
sm3.transition_to(AgentState.REPAIRING)  # validation failed
sm3.transition_to(AgentState.GENERATING)  # retry
sm3.transition_to(AgentState.VALIDATING)  # re-validate
sm3.transition_to(AgentState.COMPLETED)   # pass

assert sm3.state == AgentState.COMPLETED
assert len(sm3.history) == 8
print(f"[PASS] State machine: repair loop path, {len(sm3.history)} transitions")

# Test fail path
sm4 = StateMachine()
sm4.transition_to(AgentState.PLANNING)
sm4.fail(error="Something broke")
assert sm4.state == AgentState.FAILED
assert sm4.error == "Something broke"
print(f"[PASS] State machine: fail path, error='{sm4.error}'")

# Test reset
sm4.reset()
assert sm4.state == AgentState.IDLE
print(f"[PASS] State machine: reset to IDLE")

# Test 3: Planner module
planner = Planner()

gen_plan = planner.plan_test_generation(
    file_path="src/main/java/com/example/service/OrderService.java",
    task_description="Generate unit tests for OrderService",
    session_id="test-session-123",
)

assert gen_plan.task_type == TaskType.TEST_GENERATION
assert gen_plan.class_name == "OrderService"
assert len(gen_plan.steps) == 7  # 7 steps for test generation
assert gen_plan.steps[0].action == StepAction.EXTRACT_CLASS_INFO
assert gen_plan.steps[1].action == StepAction.RETRIEVE_CONTEXT
assert gen_plan.steps[2].action == StepAction.BUILD_PROMPT
assert gen_plan.steps[3].action == StepAction.GENERATE_CODE
assert gen_plan.steps[4].action == StepAction.EXTRACT_CODE
assert gen_plan.steps[5].action == StepAction.VALIDATE_CODE
assert gen_plan.steps[6].action == StepAction.RECORD_SESSION
print(f"[PASS] Planner: test generation plan, {len(gen_plan.steps)} steps, class={gen_plan.class_name}")

# Test refinement plan
ref_plan = planner.plan_refinement(
    session_id="test-session-123",
    feedback="Add more edge cases",
    last_class_name="OrderService",
    last_test_code="// some test code",
)

assert ref_plan.task_type == TaskType.REFINEMENT
assert len(ref_plan.steps) == 6  # 6 steps for refinement
print(f"[PASS] Planner: refinement plan, {len(ref_plan.steps)} steps")

# Test repair planning
planner.plan_repair(
    original_plan=gen_plan,
    validation_issues=["Missing @Test annotation", "No verify() calls"],
    generated_code="// broken code",
)

assert gen_plan.current_repair_attempt == 1
assert len(gen_plan.steps) == 12  # original 7 + 5 repair steps
assert gen_plan.steps[7].action == StepAction.REPAIR_CODE
print(f"[PASS] Planner: repair plan appended, total steps={len(gen_plan.steps)}, attempt={gen_plan.current_repair_attempt}")

# Test status output
status = sm3.get_status()
assert "state" in status
assert "history" in status
print(f"[PASS] Status output: state={status['state']}, transitions={status['transition_count']}")

print("\n" + "=" * 60)
print("ALL PHASE 1 TESTS PASSED")
print("=" * 60)
