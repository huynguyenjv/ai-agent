"""
Agent orchestrator for coordinating test generation workflow.
"""

import re
from typing import Optional
from dataclasses import dataclass

import structlog

from .prompt import PromptBuilder
from .rules import TestRules
from .memory import SessionMemory, MemoryManager
from rag.client import RAGClient
from rag.schema import SearchQuery, CodeChunk
from vllm.client import VLLMClient

logger = structlog.get_logger()


@dataclass
class GenerationRequest:
    """Request for test generation."""

    file_path: str
    class_name: Optional[str] = None
    task_description: Optional[str] = None
    session_id: Optional[str] = None


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

    def __post_init__(self):
        if self.validation_issues is None:
            self.validation_issues = []


class AgentOrchestrator:
    """Orchestrates the test generation workflow."""

    def __init__(
        self,
        rag_client: RAGClient,
        vllm_client: VLLMClient,
        top_k_results: int = 10,
        max_context_tokens: int = 4000,
    ):
        self.rag = rag_client
        self.vllm = vllm_client
        self.prompt_builder = PromptBuilder()
        self.test_rules = TestRules()
        self.memory_manager = MemoryManager()
        self.top_k = top_k_results
        self.max_context_tokens = max_context_tokens

        logger.info("Agent orchestrator initialized")

    def generate_test(self, request: GenerationRequest) -> GenerationResult:
        """Generate unit tests for a class."""
        try:
            # Extract class name from file path if not provided
            class_name = request.class_name or self._extract_class_name(request.file_path)
            if not class_name:
                return GenerationResult(
                    success=False,
                    error="Could not determine class name from file path",
                )

            logger.info(
                "Starting test generation",
                class_name=class_name,
                file_path=request.file_path,
            )

            # Get or create session
            session = None
            if request.session_id:
                session = self.memory_manager.get_or_create_session(request.session_id)
                session.set_context(class_name=class_name, file_path=request.file_path)

            # Query RAG for context
            rag_chunks = self._get_rag_context(class_name, request.file_path, session)

            # Build prompts
            system_prompt = self.prompt_builder.build_system_prompt()
            user_prompt = self.prompt_builder.build_test_generation_prompt(
                class_name=class_name,
                file_path=request.file_path,
                rag_chunks=rag_chunks,
                task_description=request.task_description,
                session=session,
            )

            # Record user message in session
            if session:
                session.add_user_message(
                    request.task_description or f"Generate tests for {class_name}",
                    metadata={"file_path": request.file_path},
                )

            # Call vLLM
            response = self.vllm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            if not response.success:
                return GenerationResult(
                    success=False,
                    class_name=class_name,
                    error=response.error,
                )

            # Extract code from response
            test_code = self._extract_code(response.content)

            # Validate generated code
            is_valid, issues = self.test_rules.validate_generated_code(test_code)

            # Record in session
            if session:
                session.add_assistant_message(
                    test_code,
                    metadata={"validation_passed": is_valid},
                )
                session.record_generated_test(
                    class_name=class_name,
                    test_code=test_code,
                    success=is_valid,
                )

            logger.info(
                "Test generation complete",
                class_name=class_name,
                validation_passed=is_valid,
                issues=issues,
            )

            return GenerationResult(
                success=True,
                test_code=test_code,
                class_name=class_name,
                validation_passed=is_valid,
                validation_issues=issues,
                rag_chunks_used=len(rag_chunks),
                tokens_used=response.tokens_used,
            )

        except Exception as e:
            logger.error("Test generation failed", error=str(e))
            return GenerationResult(
                success=False,
                error=str(e),
            )

    def refine_test(
        self,
        session_id: str,
        feedback: str,
    ) -> GenerationResult:
        """Refine previously generated test based on feedback."""
        session = self.memory_manager.get_session(session_id)
        if not session:
            return GenerationResult(
                success=False,
                error="Session not found or expired",
            )

        if not session.generated_tests:
            return GenerationResult(
                success=False,
                error="No previous test generation in this session",
            )

        last_test = session.generated_tests[-1]

        # Build refinement prompt
        _, issues = self.test_rules.validate_generated_code(last_test.test_code)
        refinement_prompt = self.prompt_builder.build_refinement_prompt(
            original_code=last_test.test_code,
            feedback=feedback,
            validation_issues=issues,
        )

        # Record feedback
        session.add_user_message(feedback, metadata={"type": "refinement"})

        # Call vLLM
        response = self.vllm.generate(
            system_prompt=self.prompt_builder.build_system_prompt(),
            user_prompt=refinement_prompt,
        )

        if not response.success:
            return GenerationResult(
                success=False,
                class_name=last_test.class_name,
                error=response.error,
            )

        # Extract and validate
        test_code = self._extract_code(response.content)
        is_valid, new_issues = self.test_rules.validate_generated_code(test_code)

        # Record result
        session.add_assistant_message(test_code, metadata={"validation_passed": is_valid})
        session.record_generated_test(
            class_name=last_test.class_name,
            test_code=test_code,
            success=is_valid,
            feedback=feedback,
        )

        return GenerationResult(
            success=True,
            test_code=test_code,
            class_name=last_test.class_name,
            validation_passed=is_valid,
            validation_issues=new_issues,
            tokens_used=response.tokens_used,
        )

    def _get_rag_context(
        self,
        class_name: str,
        file_path: str,
        session: Optional[SessionMemory],
    ) -> list[CodeChunk]:
        """Get relevant context from RAG."""
        # Check session cache first
        cache_key = f"{class_name}:{file_path}"
        if session:
            cached = session.get_cached_rag_context(cache_key)
            if cached:
                logger.debug("Using cached RAG context", class_name=class_name)
                return [CodeChunk(**c) for c in cached]

        chunks = []

        # Search for the main class
        main_result = self.rag.search_by_class(
            class_name=class_name,
            top_k=1,
            include_dependencies=True,
        )
        chunks.extend(main_result.chunks)

        # Search for related code
        related_query = SearchQuery(
            query=f"service {class_name} dependencies methods",
            top_k=self.top_k - len(chunks),
            score_threshold=0.4,
        )
        related_result = self.rag.search(related_query)

        # Add unique chunks
        existing_names = {c.fully_qualified_name for c in chunks}
        for chunk in related_result.chunks:
            if chunk.fully_qualified_name not in existing_names:
                chunks.append(chunk)
                existing_names.add(chunk.fully_qualified_name)

        # Cache in session
        if session:
            session.cache_rag_context(
                cache_key,
                [c.model_dump() for c in chunks],
            )

        logger.info(
            "RAG context retrieved",
            class_name=class_name,
            chunks=len(chunks),
        )

        return chunks[:self.top_k]

    def _extract_class_name(self, file_path: str) -> Optional[str]:
        """Extract class name from file path."""
        # Handle both Unix and Windows paths
        file_name = file_path.replace("\\", "/").split("/")[-1]
        if file_name.endswith(".java"):
            return file_name[:-5]  # Remove .java extension
        return None

    def _extract_code(self, response: str) -> str:
        """Extract Java code from LLM response."""
        # Try to find code block
        code_block_pattern = r"```(?:java)?\s*\n(.*?)```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        if matches:
            # Return the longest code block (likely the main test class)
            return max(matches, key=len).strip()

        # If no code block, try to find class definition
        class_pattern = r"((?:import.*?\n)*\s*(?:@\w+.*?\n)*\s*(?:public\s+)?class\s+\w+.*?\{.*\})"
        class_match = re.search(class_pattern, response, re.DOTALL)

        if class_match:
            return class_match.group(1).strip()

        # Return the whole response if no patterns match
        return response.strip()

    def get_session_info(self, session_id: str) -> Optional[dict]:
        """Get information about a session."""
        session = self.memory_manager.get_session(session_id)
        if session:
            return session.get_session_summary()
        return None

    def cleanup_sessions(self) -> int:
        """Clean up expired sessions."""
        return self.memory_manager.cleanup_expired()

