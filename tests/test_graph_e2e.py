"""End-to-end smoke test: invoke graph with mocked RAG + vLLM.

Catches runtime bugs in state propagation, subgraph nesting,
node function signatures, and result serialization.
"""

import sys
import os
from unittest.mock import MagicMock, patch
from rag.schema import CodeChunk
from agent.graph_adapter import GraphOrchestrator


# Stub heavy deps BEFORE any project imports
for m in [
    'tree_sitter_java', 'tree_sitter', 'sentence_transformers',
    'qdrant_client', 'qdrant_client.http', 'qdrant_client.http.models',
    'qdrant_client.models', 'torch', 'redis', 'redis.asyncio',
]:
    if m not in sys.modules:
        sys.modules[m] = MagicMock()


# ── Mock RAG client ─────────────────────────────────────────────────
class MockRAGClient:
    """Simulates RAGClient.search_by_class()."""
    def __init__(self):
        self.qdrant = MagicMock()

    def search_by_class(self, class_name, top_k=1,
                        include_dependencies=False, collection_name=None):
        """Return a fake search result with one chunk."""
        chunk = CodeChunk(
            class_name=class_name,
            fully_qualified_name=f"com.example.{class_name}",
            package="com.example",
            element_type="class",
            code=f"public class {class_name} {{}}\n",
            summary=f"Service class {class_name}",
            dependencies=["com.example.UserRepository"],
            used_types=["UserRepository"],
            methods=["findById", "create"],
        )
        result = MagicMock()
        result.chunks = [chunk]
        return result


# ── Mock vLLM client ────────────────────────────────────────────────
MOCK_JAVA_CODE = """
package com.example;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class UserServiceTest {

    @Mock
    private UserRepository userRepository;

    @InjectMocks
    private UserService userService;

    @Test
    void testFindById() {
        // Arrange
        when(userRepository.findById(1L)).thenReturn(new User());
        // Act
        User result = userService.findById(1L);
        // Assert
        assertNotNull(result);
        verify(userRepository).findById(1L);
    }
}
""".strip()


class MockVLLMClient:
    """Simulates VLLMClient.generate()."""
    def generate(self, system_prompt="", user_prompt="", **kwargs):
        resp = MagicMock()
        resp.success = True
        resp.content = f"```java\n{MOCK_JAVA_CODE}\n```"
        resp.tokens_used = 150
        resp.error = None
        return resp

    def health_check(self):
        return True
    
    def close(self):
        pass


# ═══════════════════════════════════════════════════════════════════════
# Test: Full graph invocation
# ═══════════════════════════════════════════════════════════════════════

def test_full_graph_invocation():
    """Test complete flow: supervisor → retrieve → check_strategy →
    build_prompt → call_llm → validate → human_review → save_result."""


    rag = MockRAGClient()
    vllm = MockVLLMClient()

    # Create graph orchestrator (uses MemorySaver since in-memory test)
    orch = GraphOrchestrator(
        rag_client=rag,
        vllm_client=vllm,
        checkpoint_db=":memory:",
    )

    print("GraphOrchestrator created OK")

    # Simulate a GenerationRequest-like object
    class FakeRequest:
        file_path = "src/main/java/com/example/UserService.java"
        class_name = "UserService"
        task_description = "Generate unit tests for UserService"
        session_id = None
        existing_test_code = None
        changed_methods = None
        collection_name = None
        source_code = "public class UserService { }"
        force_two_phase = False
        force_single_pass = False
        complexity_threshold = 10
        require_human_review = False

    result = orch.generate_test(FakeRequest())

    print(f"Result type: {type(result).__name__}")
    print(f"  success: {result.success}")
    print(f"  class_name: {result.class_name}")
    print(f"  test_code length: {len(result.test_code or '')}")
    print(f"  validation_passed: {result.validation_passed}")
    print(f"  validation_issues: {result.validation_issues}")
    print(f"  tokens_used: {result.tokens_used}")
    print(f"  strategy_used: {result.strategy_used}")
    print(f"  repair_attempts: {result.repair_attempts}")
    print(f"  run_id: {result.run_id}")
    print(f"  error: {result.error}")

    if result.error:
        print(f"\n[FAIL] Graph execution error: {result.error}")
        return False
    
    if not result.test_code:
        print(f"\n[FAIL] No test code generated")
        return False

    print(f"\n[PASS] Full graph invocation completed successfully")
    return True


if __name__ == "__main__":
    success = test_full_graph_invocation()
    sys.exit(0 if success else 1)
