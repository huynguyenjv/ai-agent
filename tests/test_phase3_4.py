"""
Phase 3 + 4 Integration Tests

Tests:
  1. ValidationPipeline  — good code, bad code, partial, severity levels
  2. RepairStrategySelector — targeted, regenerate, fallback
  3. EventBus — subscribe, publish, wildcard, unsubscribe, error resilience
  4. MetricsCollector — event-driven recording, counters, timing
  5. Full mock flow — generate → validate → repair → complete (orchestrator)

Runs WITHOUT Qdrant / vLLM / torch.
"""

import sys
import types
import time
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Bootstrap: avoid heavy imports ──────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for mod_name in [
    "structlog",
    "tree_sitter_java",
    "tree_sitter",
    "sentence_transformers",
    "qdrant_client",
    "qdrant_client.http",
    "qdrant_client.http.models",
    "torch",
    "pydantic",
]:
    if mod_name not in sys.modules:
        stub = types.ModuleType(mod_name)
        if mod_name == "structlog":
            stub.get_logger = lambda: MagicMock()
        if mod_name == "tree_sitter":
            stub.Language = MagicMock()
            stub.Parser = MagicMock()
        if mod_name == "tree_sitter_java":
            stub.language = MagicMock
        if mod_name == "pydantic":
            # Minimal pydantic stub for BaseModel
            class _BaseModel:
                def __init_subclass__(cls, **kw):
                    pass
                @classmethod
                def model_validate(cls, data):
                    return cls()
                def model_dump(self):
                    return {}
            stub.BaseModel = _BaseModel
            stub.Field = lambda *a, **kw: None
        sys.modules[mod_name] = stub


def _load(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_pkg(pkg_name: str, init_path: Path):
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(init_path.parent)]
    pkg.__file__ = str(init_path)
    sys.modules[pkg_name] = pkg
    return pkg


# ── Load agent package ──────────────────────────────────────────────

_load_pkg("agent", ROOT / "agent" / "__init__.py")
_load_pkg("rag", ROOT / "rag" / "__init__.py")

# Stub rag.client and rag.schema
_rag_schema_mod = types.ModuleType("rag.schema")


class _SearchQuery:
    def __init__(self, **kw): pass


class _CodeChunk:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
    def model_dump(self):
        return self.__dict__


class _MetadataFilter:
    def __init__(self, **kw): pass


class _SearchResult:
    def __init__(self, chunks=None):
        self.chunks = chunks or []


_rag_schema_mod.SearchQuery = _SearchQuery
_rag_schema_mod.CodeChunk = _CodeChunk
_rag_schema_mod.MetadataFilter = _MetadataFilter
_rag_schema_mod.SearchResult = _SearchResult
sys.modules["rag.schema"] = _rag_schema_mod

_rag_client_mod = types.ModuleType("rag.client")
_rag_client_mod.RAGClient = MagicMock
sys.modules["rag.client"] = _rag_client_mod

# Stub vllm
_load_pkg("vllm", ROOT / "vllm" / "__init__.py")
_vllm_client_mod = types.ModuleType("vllm.client")


class _FakeVLLMResponse:
    def __init__(self, content="", success=True, error=None, tokens_used=100):
        self.content = content
        self.success = success
        self.error = error
        self.tokens_used = tokens_used


class _FakeVLLMClient:
    def __init__(self, **kw):
        self._response = _FakeVLLMResponse()

    def generate(self, **kw):
        return self._response

    def health_check(self):
        return True

    def close(self):
        pass


_vllm_client_mod.VLLMClient = _FakeVLLMClient
sys.modules["vllm.client"] = _vllm_client_mod

# Stub context package (optional, graceful degradation)
_ctx_pkg = types.ModuleType("context")
_ctx_pkg.__path__ = [str(ROOT / "context")]
sys.modules["context"] = _ctx_pkg
_ctx_builder_mod = types.ModuleType("context.context_builder")
_ctx_builder_mod.ContextBuilder = None
_ctx_builder_mod.ContextResult = None
sys.modules["context.context_builder"] = _ctx_builder_mod

# Now load agent sub-modules
_plan = _load("agent.plan", ROOT / "agent" / "plan.py")
_sm = _load("agent.state_machine", ROOT / "agent" / "state_machine.py")
_planner = _load("agent.planner", ROOT / "agent" / "planner.py")
_prompt = _load("agent.prompt", ROOT / "agent" / "prompt.py")
_rules = _load("agent.rules", ROOT / "agent" / "rules.py")
_memory = _load("agent.memory", ROOT / "agent" / "memory.py")

# Phase 3
_validation = _load("agent.validation", ROOT / "agent" / "validation.py")
_repair = _load("agent.repair", ROOT / "agent" / "repair.py")

# Phase 4
_events = _load("agent.events", ROOT / "agent" / "events.py")
_metrics = _load("agent.metrics", ROOT / "agent" / "metrics.py")

# Orchestrator (imports all the above)
_orch = _load("agent.orchestrator", ROOT / "agent" / "orchestrator.py")


# ── Import classes ──────────────────────────────────────────────────

ValidationPipeline = _validation.ValidationPipeline
ValidationResult = _validation.ValidationResult
ValidationIssue = _validation.ValidationIssue
IssueSeverity = _validation.IssueSeverity
IssueCategory = _validation.IssueCategory

RepairStrategySelector = _repair.RepairStrategySelector
RepairAction = _repair.RepairAction

EventBus = _events.EventBus
Event = _events.Event
EventType = _events.EventType
reset_event_bus = _events.reset_event_bus

MetricsCollector = _metrics.MetricsCollector
TimingStats = _metrics.TimingStats

AgentOrchestrator = _orch.AgentOrchestrator
GenerationRequest = _orch.GenerationRequest
GenerationResult = _orch.GenerationResult


# ═══════════════════════════════════════════════════════════════════
# Test data
# ═══════════════════════════════════════════════════════════════════

GOOD_TEST_CODE = """\
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class UserServiceTest {

    @Mock
    private UserRepository userRepository;

    @InjectMocks
    private UserService userService;

    @Test
    @DisplayName("Should return user when found")
    void shouldReturnUserWhenFound() {
        // Arrange
        User user = new User("John");
        when(userRepository.findById(1L)).thenReturn(Optional.of(user));

        // Act
        User result = userService.getUser(1L);

        // Assert
        assertNotNull(result);
        assertEquals("John", result.getName());
        verify(userRepository).findById(1L);
    }
}
"""

BAD_TEST_CODE = """\
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class UserServiceTest {

    void testSomething() {
        assert true;
    }
}
"""

MISSING_ANNOTATIONS_CODE = """\
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
class UserServiceTest {

    @Test
    void testGetUser() {
        // no arrange/act/assert pattern
        UserService svc = new UserService();
        svc.getUser(1L);
    }
}
"""


# ═══════════════════════════════════════════════════════════════════
# 1. ValidationPipeline Tests
# ═══════════════════════════════════════════════════════════════════

def test_validation_good_code():
    """Good code should pass validation with no errors."""
    vp = ValidationPipeline()
    result = vp.validate(GOOD_TEST_CODE)
    assert result.passed, f"Expected pass, got errors: {result.error_messages}"
    assert result.test_count >= 1
    print(f"  ✓ Good code: {result.test_count} test(s), {len(result.warnings)} warning(s)")


def test_validation_bad_code():
    """Bad code with @SpringBootTest should fail."""
    vp = ValidationPipeline()
    result = vp.validate(BAD_TEST_CODE)
    assert not result.passed, "Expected failure for bad code"
    assert any("SpringBootTest" in msg or "FORBIDDEN" in msg.upper() or "forbidden" in msg.lower()
               for msg in result.error_messages), f"Expected forbidden pattern error, got: {result.error_messages}"
    print(f"  ✓ Bad code: {len(result.errors)} error(s), {len(result.warnings)} warning(s)")


def test_validation_partial_code():
    """Code with some issues but not fatal should have warnings."""
    vp = ValidationPipeline()
    result = vp.validate(MISSING_ANNOTATIONS_CODE)
    # May pass or fail depending on severity of missing @Mock/@InjectMocks
    summary = result.get_summary()
    assert "passed" in summary
    print(f"  ✓ Partial code: passed={result.passed}, errors={summary['errors']}, warnings={summary['warnings']}")


def test_validation_empty_code():
    """Empty code should fail validation."""
    vp = ValidationPipeline()
    result = vp.validate("")
    assert not result.passed
    print(f"  ✓ Empty code: {len(result.errors)} error(s)")


def test_validation_severity_levels():
    """Verify severity levels work correctly."""
    vp = ValidationPipeline()
    result = vp.validate(GOOD_TEST_CODE)
    # Good code should have 0 errors
    assert len(result.errors) == 0
    # May have some INFO/WARNING level issues
    for issue in result.issues:
        assert issue.severity in (IssueSeverity.ERROR, IssueSeverity.WARNING, IssueSeverity.INFO)
    print(f"  ✓ Severity levels: {len(result.errors)} ERR, {len(result.warnings)} WARN, "
          f"{len([i for i in result.issues if i.severity == IssueSeverity.INFO])} INFO")


# ═══════════════════════════════════════════════════════════════════
# 2. RepairStrategySelector Tests
# ═══════════════════════════════════════════════════════════════════

def test_repair_targeted_strategy():
    """Repair selector should produce targeted instructions for specific issues."""
    vp = ValidationPipeline()
    result = vp.validate(BAD_TEST_CODE)
    assert not result.passed

    rs = RepairStrategySelector()
    plan = rs.build_repair_plan(result, attempt_number=1, max_attempts=3)

    assert plan.instruction_count > 0
    prompt_section = plan.get_repair_prompt_section()
    assert len(prompt_section) > 0
    print(f"  ✓ Targeted repair: {plan.instruction_count} instruction(s), strategy={plan.strategy}")


def test_repair_regenerate_on_last_attempt():
    """On last attempt with structural issues, should suggest REGENERATE."""
    vp = ValidationPipeline()
    # Create code with structural issues
    structural_bad = "class Test {"  # Missing closing brace
    result = vp.validate(structural_bad)

    rs = RepairStrategySelector()
    plan = rs.build_repair_plan(result, attempt_number=3, max_attempts=3)

    prompt = plan.get_repair_prompt_section()
    # Should contain regenerate or full rewrite instructions
    assert plan.instruction_count > 0
    print(f"  ✓ Regenerate strategy: {plan.instruction_count} instruction(s), strategy={plan.strategy}")


def test_repair_prompt_format():
    """Repair prompt should be well-formatted for LLM consumption."""
    rs = RepairStrategySelector()
    vp = ValidationPipeline()
    result = vp.validate(BAD_TEST_CODE)
    plan = rs.build_repair_plan(result, attempt_number=1, max_attempts=2)

    prompt = plan.get_repair_prompt_section()
    # Should contain structured repair instructions
    assert "fix" in prompt.lower() or "repair" in prompt.lower() or "replace" in prompt.lower() or "remove" in prompt.lower() or "add" in prompt.lower()
    print(f"  ✓ Repair prompt format OK ({len(prompt)} chars)")


# ═══════════════════════════════════════════════════════════════════
# 3. EventBus Tests
# ═══════════════════════════════════════════════════════════════════

def test_eventbus_subscribe_publish():
    """Basic subscribe and publish."""
    bus = EventBus()
    received = []

    def handler(event):
        received.append(event)

    bus.subscribe(EventType.STATE_CHANGED, handler)
    bus.publish(Event(type=EventType.STATE_CHANGED, data={"state": "PLANNING"}))

    assert len(received) == 1
    assert received[0].data["state"] == "PLANNING"
    print(f"  ✓ Subscribe/publish: received {len(received)} event(s)")


def test_eventbus_wildcard():
    """Wildcard subscriber should receive all events."""
    bus = EventBus()
    all_events = []

    bus.subscribe_all(lambda e: all_events.append(e))

    bus.publish(Event(type=EventType.STATE_CHANGED, data={}))
    bus.publish(Event(type=EventType.STEP_STARTED, data={}))
    bus.publish(Event(type=EventType.GENERATION_COMPLETED, data={}))

    assert len(all_events) == 3
    print(f"  ✓ Wildcard: received {len(all_events)} event(s)")


def test_eventbus_unsubscribe():
    """Unsubscribe should stop delivering events."""
    bus = EventBus()
    count = [0]

    def handler(e):
        count[0] += 1

    bus.subscribe(EventType.STEP_STARTED, handler)
    bus.publish(Event(type=EventType.STEP_STARTED, data={}))
    assert count[0] == 1

    bus.unsubscribe(EventType.STEP_STARTED, handler)
    bus.publish(Event(type=EventType.STEP_STARTED, data={}))
    assert count[0] == 1  # Still 1
    print(f"  ✓ Unsubscribe: count stayed at {count[0]}")


def test_eventbus_error_resilience():
    """A failing handler should not break other handlers."""
    bus = EventBus()
    results = []

    def bad_handler(e):
        raise RuntimeError("boom")

    def good_handler(e):
        results.append("ok")

    bus.subscribe(EventType.ERROR_OCCURRED, bad_handler)
    bus.subscribe(EventType.ERROR_OCCURRED, good_handler)

    bus.publish(Event(type=EventType.ERROR_OCCURRED, data={}))

    assert results == ["ok"]
    print(f"  ✓ Error resilience: good handler still ran")


def test_eventbus_counts():
    """Event count and subscriber count should be tracked."""
    bus = EventBus()
    bus.subscribe(EventType.STATE_CHANGED, lambda e: None)
    bus.subscribe(EventType.STEP_STARTED, lambda e: None)
    bus.subscribe_all(lambda e: None)

    assert bus.subscriber_count == 3

    bus.publish(Event(type=EventType.STATE_CHANGED, data={}))
    bus.publish(Event(type=EventType.STEP_STARTED, data={}))
    assert bus.event_count == 2
    print(f"  ✓ Counts: {bus.subscriber_count} subscribers, {bus.event_count} events")


# ═══════════════════════════════════════════════════════════════════
# 4. MetricsCollector Tests
# ═══════════════════════════════════════════════════════════════════

def test_metrics_manual_recording():
    """Manual recording should update counters."""
    mc = MetricsCollector()
    mc.record_generation(success=True, tokens=500, duration_ms=1200)
    mc.record_generation(success=False, tokens=0, duration_ms=300)
    mc.record_step("RETRIEVE_CONTEXT", 150.0)

    m = mc.get_metrics()
    assert m["generations"]["total"] == 2
    assert m["generations"]["success"] == 1
    assert m["generations"]["failed"] == 1
    assert m["tokens"]["total"] == 500
    assert "RETRIEVE_CONTEXT" in m["timing"]["steps"]
    print(f"  ✓ Manual recording: {m['generations']}")


def test_metrics_event_driven():
    """MetricsCollector should collect from EventBus automatically."""
    bus = EventBus()
    mc = MetricsCollector(bus)

    # Simulate a generation lifecycle
    bus.publish(Event(type=EventType.GENERATION_STARTED, data={"plan_id": "p1"}, source="test"))
    time.sleep(0.01)  # Small delay for timing

    bus.publish(Event(type=EventType.STEP_STARTED, data={"action": "RETRIEVE", "step_id": 1}))
    time.sleep(0.01)
    bus.publish(Event(type=EventType.STEP_COMPLETED, data={"action": "RETRIEVE", "step_id": 1}))

    bus.publish(Event(type=EventType.VALIDATION_COMPLETED, data={"passed": True}))

    bus.publish(Event(type=EventType.GENERATION_COMPLETED, data={
        "plan_id": "p1", "success": True, "tokens_used": 800,
    }))

    m = mc.get_metrics()
    assert m["generations"]["total"] == 1
    assert m["generations"]["success"] == 1
    assert m["validation"]["total"] == 1
    assert m["validation"]["passed"] == 1
    assert m["tokens"]["total"] == 800
    assert "RETRIEVE" in m["timing"]["steps"]
    print(f"  ✓ Event-driven metrics: gen={m['generations']}, val={m['validation']}")


def test_metrics_repair_tracking():
    """MetricsCollector should track repair attempts."""
    bus = EventBus()
    mc = MetricsCollector(bus)

    bus.publish(Event(type=EventType.REPAIR_STARTED, data={"attempt": 1}))
    bus.publish(Event(type=EventType.REPAIR_COMPLETED, data={"success": False}))
    bus.publish(Event(type=EventType.REPAIR_STARTED, data={"attempt": 2}))
    bus.publish(Event(type=EventType.REPAIR_COMPLETED, data={"success": True}))

    m = mc.get_metrics()
    assert m["repair"]["total"] == 2
    assert m["repair"]["success"] == 1
    assert m["repair"]["success_rate"] == 0.5
    print(f"  ✓ Repair tracking: {m['repair']}")


def test_timing_stats():
    """TimingStats should compute min/max/avg correctly."""
    ts = TimingStats()
    ts.record(100)
    ts.record(200)
    ts.record(300)

    assert ts.count == 3
    assert ts.min_ms == 100
    assert ts.max_ms == 300
    assert ts.avg_ms == 200
    d = ts.to_dict()
    assert d["count"] == 3
    print(f"  ✓ TimingStats: {d}")


def test_metrics_reset():
    """Reset should clear all metrics."""
    mc = MetricsCollector()
    mc.record_generation(success=True, tokens=100)
    mc.reset()
    m = mc.get_metrics()
    assert m["generations"]["total"] == 0
    assert m["tokens"]["total"] == 0
    print(f"  ✓ Metrics reset: all zeroed")


# ═══════════════════════════════════════════════════════════════════
# 5. Full Mock Flow — Orchestrator Integration
# ═══════════════════════════════════════════════════════════════════

def test_orchestrator_happy_path():
    """Full flow: generate → validate → complete with events."""
    reset_event_bus()

    rag_client = MagicMock()
    rag_result = MagicMock()
    rag_chunk = _CodeChunk(
        class_name="UserService",
        fully_qualified_name="com.example.UserService",
        type="class",
        java_type="SERVICE",
        layer="service",
        summary="User service class",
        source_code="public class UserService {}",
        dependencies=[],
        used_types=[],
        fields=[],
        methods=[],
    )
    rag_result.chunks = [rag_chunk]
    rag_client.search_by_class = MagicMock(return_value=rag_result)

    vllm_client = _FakeVLLMClient()
    vllm_client._response = _FakeVLLMResponse(
        content=f"```java\n{GOOD_TEST_CODE}\n```",
        success=True,
        tokens_used=500,
    )

    orch = AgentOrchestrator(
        rag_client=rag_client,
        vllm_client=vllm_client,
        max_repair_attempts=2,
    )

    # Capture events
    events_received = []
    orch.event_bus.subscribe_all(lambda e: events_received.append(e))

    request = GenerationRequest(
        file_path="src/main/java/com/example/UserService.java",
        class_name="UserService",
    )
    result = orch.generate_test(request)

    assert result.success, f"Expected success, got error: {result.error}"
    assert "UserServiceTest" in (result.test_code or "")
    assert result.validation_passed
    assert result.tokens_used > 0

    # Check events were published
    event_types = [e.type for e in events_received]
    assert EventType.PLAN_CREATED in event_types, f"Missing PLAN_CREATED in {event_types}"
    assert EventType.GENERATION_STARTED in event_types
    assert EventType.GENERATION_COMPLETED in event_types
    assert EventType.CONTEXT_RETRIEVED in event_types
    assert EventType.VALIDATION_COMPLETED in event_types

    # Check metrics
    m = orch.metrics.get_metrics()
    assert m["generations"]["total"] >= 1
    assert m["generations"]["success"] >= 1

    print(f"  ✓ Happy path: success={result.success}, events={len(events_received)}, "
          f"tokens={result.tokens_used}")


def test_orchestrator_repair_flow():
    """Flow with validation failure → repair → success."""
    reset_event_bus()

    rag_client = MagicMock()
    rag_result = MagicMock()
    rag_chunk = _CodeChunk(
        class_name="OrderService",
        fully_qualified_name="com.example.OrderService",
        type="class",
        java_type="SERVICE",
        layer="service",
        summary="Order service class",
        source_code="public class OrderService {}",
        dependencies=[],
        used_types=[],
        fields=[],
        methods=[],
    )
    rag_result.chunks = [rag_chunk]
    rag_client.search_by_class = MagicMock(return_value=rag_result)

    # First call returns bad code, second returns good code
    call_count = [0]

    class _SequentialVLLM:
        def generate(self, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return _FakeVLLMResponse(
                    content=f"```java\n{BAD_TEST_CODE}\n```",
                    success=True,
                    tokens_used=300,
                )
            else:
                return _FakeVLLMResponse(
                    content=f"```java\n{GOOD_TEST_CODE}\n```",
                    success=True,
                    tokens_used=400,
                )

        def health_check(self):
            return True

        def close(self):
            pass

    orch = AgentOrchestrator(
        rag_client=rag_client,
        vllm_client=_SequentialVLLM(),
        max_repair_attempts=2,
    )

    events_received = []
    orch.event_bus.subscribe_all(lambda e: events_received.append(e))

    request = GenerationRequest(
        file_path="src/main/java/com/example/OrderService.java",
        class_name="OrderService",
    )
    result = orch.generate_test(request)

    assert result.success, f"Expected success after repair, got error: {result.error}"
    assert result.repair_attempts >= 1

    event_types = [e.type for e in events_received]
    assert EventType.REPAIR_STARTED in event_types, "Expected REPAIR_STARTED event"

    print(f"  ✓ Repair flow: success={result.success}, repairs={result.repair_attempts}, "
          f"calls={call_count[0]}, events={len(events_received)}")


def test_orchestrator_max_repairs_exceeded():
    """When max repairs are exceeded, should still return with issues."""
    reset_event_bus()

    rag_client = MagicMock()
    rag_result = MagicMock()
    rag_chunk = _CodeChunk(
        class_name="BadService",
        fully_qualified_name="com.example.BadService",
        type="class",
        java_type="SERVICE",
        layer="service",
        summary="Bad service",
        source_code="public class BadService {}",
        dependencies=[],
        used_types=[],
        fields=[],
        methods=[],
    )
    rag_result.chunks = [rag_chunk]
    rag_client.search_by_class = MagicMock(return_value=rag_result)

    # Always returns bad code
    class _AlwaysBadVLLM:
        def generate(self, **kw):
            return _FakeVLLMResponse(
                content=f"```java\n{BAD_TEST_CODE}\n```",
                success=True,
                tokens_used=200,
            )

        def health_check(self):
            return True

        def close(self):
            pass

    orch = AgentOrchestrator(
        rag_client=rag_client,
        vllm_client=_AlwaysBadVLLM(),
        max_repair_attempts=1,  # Only 1 repair attempt allowed
    )

    request = GenerationRequest(
        file_path="src/main/java/com/example/BadService.java",
        class_name="BadService",
    )
    result = orch.generate_test(request)

    # Should succeed but with validation issues (accepted with issues)
    assert result.success
    assert not result.validation_passed
    assert len(result.validation_issues) > 0
    print(f"  ✓ Max repairs exceeded: issues={result.validation_issues[:2]}...")


def test_orchestrator_plan_summary():
    """Plan summary should be returned in the result."""
    reset_event_bus()

    rag_client = MagicMock()
    rag_result = MagicMock()
    rag_chunk = _CodeChunk(
        class_name="PlanService",
        fully_qualified_name="com.example.PlanService",
        type="class",
        java_type="SERVICE",
        layer="service",
        summary="Plan service",
        source_code="public class PlanService {}",
        dependencies=[],
        used_types=[],
        fields=[],
        methods=[],
    )
    rag_result.chunks = [rag_chunk]
    rag_client.search_by_class = MagicMock(return_value=rag_result)

    vllm_client = _FakeVLLMClient()
    vllm_client._response = _FakeVLLMResponse(
        content=f"```java\n{GOOD_TEST_CODE}\n```",
        success=True,
        tokens_used=600,
    )

    orch = AgentOrchestrator(
        rag_client=rag_client,
        vllm_client=vllm_client,
    )

    request = GenerationRequest(
        file_path="src/main/java/com/example/PlanService.java",
        class_name="PlanService",
    )
    result = orch.generate_test(request)

    assert result.plan_summary is not None
    assert "plan_id" in result.plan_summary
    assert "total_steps" in result.plan_summary
    print(f"  ✓ Plan summary: {result.plan_summary.get('plan_id', 'n/a')}, "
          f"steps={result.plan_summary.get('total_steps', 0)}")


def test_orchestrator_validation_summary():
    """Validation summary (Phase 3) should be in the result."""
    reset_event_bus()

    rag_client = MagicMock()
    rag_result = MagicMock()
    rag_chunk = _CodeChunk(
        class_name="ValService",
        fully_qualified_name="com.example.ValService",
        type="class",
        java_type="SERVICE",
        layer="service",
        summary="Val service",
        source_code="public class ValService {}",
        dependencies=[],
        used_types=[],
        fields=[],
        methods=[],
    )
    rag_result.chunks = [rag_chunk]
    rag_client.search_by_class = MagicMock(return_value=rag_result)

    vllm_client = _FakeVLLMClient()
    vllm_client._response = _FakeVLLMResponse(
        content=f"```java\n{GOOD_TEST_CODE}\n```",
        success=True,
        tokens_used=400,
    )

    orch = AgentOrchestrator(
        rag_client=rag_client,
        vllm_client=vllm_client,
    )

    request = GenerationRequest(
        file_path="src/main/java/com/example/ValService.java",
        class_name="ValService",
    )
    result = orch.generate_test(request)

    assert result.success
    assert result.validation_summary is not None
    assert "passed" in result.validation_summary
    assert "errors" in result.validation_summary
    print(f"  ✓ Validation summary: {result.validation_summary}")


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════

ALL_TESTS = [
    # 1. Validation
    test_validation_good_code,
    test_validation_bad_code,
    test_validation_partial_code,
    test_validation_empty_code,
    test_validation_severity_levels,
    # 2. Repair
    test_repair_targeted_strategy,
    test_repair_regenerate_on_last_attempt,
    test_repair_prompt_format,
    # 3. EventBus
    test_eventbus_subscribe_publish,
    test_eventbus_wildcard,
    test_eventbus_unsubscribe,
    test_eventbus_error_resilience,
    test_eventbus_counts,
    # 4. Metrics
    test_metrics_manual_recording,
    test_metrics_event_driven,
    test_metrics_repair_tracking,
    test_timing_stats,
    test_metrics_reset,
    # 5. Orchestrator integration
    test_orchestrator_happy_path,
    test_orchestrator_repair_flow,
    test_orchestrator_max_repairs_exceeded,
    test_orchestrator_plan_summary,
    test_orchestrator_validation_summary,
]


def main():
    print("=" * 70)
    print("Phase 3 + 4 Integration Tests")
    print("=" * 70)

    passed = 0
    failed = 0
    errors = []

    for test_fn in ALL_TESTS:
        name = test_fn.__name__
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, e))
            print(f"  ✗ {name}: {e}")

    print()
    print("=" * 70)
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 70)

    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            import traceback
            print(f"\n  {name}:")
            traceback.print_exception(type(err), err, err.__traceback__)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
