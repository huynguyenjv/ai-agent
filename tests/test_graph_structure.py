"""Quick smoke test: verify graph compiles and node wiring is correct.

Stubs qdrant_client/tree_sitter/redis before ANY agent imports
to avoid the agent.__init__ → orchestrator import chain.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Stub ALL heavy optional deps BEFORE any agent imports ────────────
_MOCK_MODULES = [
    "tree_sitter_java", "tree_sitter",
    "sentence_transformers",
    "qdrant_client", "qdrant_client.http", "qdrant_client.http.models",
    "qdrant_client.models",
    "torch",
    "redis", "redis.asyncio",
]
for mod_name in _MOCK_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# ═══════════════════════════════════════════════════════════════════════
# Test 1: Supervisor intent classification
# ═══════════════════════════════════════════════════════════════════════

from agent.supervisor import classify_intent

assert classify_intent("Generate unit tests for UserService") == "unit_test"
assert classify_intent("Write tests for OrderService") == "unit_test"
assert classify_intent("") == "unit_test"
assert classify_intent("Review my code quality") == "code_review"
assert classify_intent("Refactor this class") == "refactor"
assert classify_intent("Generate documentation") == "doc_gen"
print("[PASS] Test 1: Supervisor intent classification")


# ═══════════════════════════════════════════════════════════════════════
# Test 2: State schemas
# ═══════════════════════════════════════════════════════════════════════

from agent.state import AgentState, UnitTestState

assert "intent" in AgentState.__annotations__
assert "subgraph_result" in AgentState.__annotations__
assert len(UnitTestState.__annotations__) == 36
print(f"[PASS] Test 2: AgentState has {len(AgentState.__annotations__)} fields, UnitTestState has {len(UnitTestState.__annotations__)} fields")


# ═══════════════════════════════════════════════════════════════════════
# Test 3: UnitTest SubGraph compiles
# ═══════════════════════════════════════════════════════════════════════

from agent.subgraphs.unit_test import build_unit_test_graph
from agent.prompt import PromptBuilder
from agent.validation import ValidationPipeline
from agent.repair import RepairStrategySelector

compiled = build_unit_test_graph(
    rag_client=MagicMock(),
    vllm_client=MagicMock(),
    prompt_builder=PromptBuilder(),
    validation_pipeline=ValidationPipeline(),
    repair_selector=RepairStrategySelector(),
)

graph_nodes = compiled.get_graph().nodes
expected_nodes = {
    "retrieve", "check_strategy", "analyze", "build_prompt",
    "call_llm", "validate", "repair", "human_review", "save_result",
    "__start__", "__end__",
}
actual_nodes = set(graph_nodes.keys())
missing = expected_nodes - actual_nodes
assert not missing, f"Missing nodes: {missing}"
print(f"[PASS] Test 3: UnitTest SubGraph compiled with {len(actual_nodes)} nodes: {sorted(actual_nodes - {'__start__', '__end__'})}")


# ═══════════════════════════════════════════════════════════════════════
# Test 4: Routing functions
# ═══════════════════════════════════════════════════════════════════════

from agent.subgraphs.unit_test import route_strategy, route_after_validate, route_after_review

assert route_strategy({"strategy": "single_pass"}) == "single_pass"
assert route_strategy({"strategy": "two_phase"}) == "two_phase"
assert route_strategy({}) == "single_pass"

assert route_after_validate({"validation_passed": True}) == "pass"
assert route_after_validate({"validation_passed": False, "retry_count": 1, "max_retries": 3}) == "retry"
assert route_after_validate({"validation_passed": False, "retry_count": 3, "max_retries": 3}) == "max_retries"

assert route_after_review({"human_approved": True}) == "approve"
assert route_after_review({"human_approved": False}) == "reject"
assert route_after_review({"human_approved": None}) == "approve"
print("[PASS] Test 4: All routing edge conditions")


# ═══════════════════════════════════════════════════════════════════════
# Test 5: check_strategy node
# ═══════════════════════════════════════════════════════════════════════

from agent.nodes.check_strategy import check_strategy_node

r = check_strategy_node({"force_single_pass": True})
assert r["strategy"] == "single_pass"

r = check_strategy_node({"force_two_phase": True})
assert r["strategy"] == "two_phase"

r = check_strategy_node({"existing_test_code": "// test"})
assert r["strategy"] == "single_pass"

r = check_strategy_node({
    "rag_chunks": [{"class_name": "Svc", "dependencies": ["A", "B", "C"],
                     "used_types": ["X", "Y", "Z", "W"], "methods": ["m1", "m2"]}],
    "class_name": "Svc",
    "complexity_threshold": 10,
})
assert r["strategy"] == "two_phase"
assert r["complexity_score"] == 20
print(f"[PASS] Test 5: check_strategy complexity={r['complexity_score']}")


# ═══════════════════════════════════════════════════════════════════════
# Test 6: call_llm code extraction
# ═══════════════════════════════════════════════════════════════════════

from agent.nodes.call_llm import _extract_code

assert _extract_code("```java\nint x = 1;\n```") == "int x = 1;"
assert _extract_code("text\n```java\nclass T {}\n```\nmore") == "class T {}"
assert _extract_code("```\nraw code\n```") == "raw code"
assert _extract_code("just text") == "just text"
assert _extract_code("") == ""
print("[PASS] Test 6: Code extraction from LLM responses")


# ═══════════════════════════════════════════════════════════════════════
# Test 7: validate node (basic)
# ═══════════════════════════════════════════════════════════════════════

from agent.nodes.validate import validate_node

vr = validate_node({"test_code": ""}, validation_pipeline=ValidationPipeline())
assert vr["validation_passed"] is False
assert "Empty test code" in vr["validation_issues"]
print("[PASS] Test 7: Validate node handles empty code")


print("\n" + "=" * 60)
print("ALL 7 GRAPH STRUCTURE TESTS PASSED ✅")
print("=" * 60)
