"""
Agent orchestrator for coordinating test generation workflow.
"""

import re
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import structlog

from .prompt import PromptBuilder
from .rules import TestRules
from .memory import SessionMemory, MemoryManager
from rag.client import RAGClient
from rag.schema import SearchQuery, CodeChunk, MetadataFilter
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

            # Query RAG for context, with fallback extraction from inline source
            rag_chunks = self._get_rag_context(
                class_name, request.file_path, session,
                inline_source=request.task_description,
            )

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

            # Full LLM response = Analysis Report + Test Plan + Code + Coverage Notes
            full_response = response.content

            # Extract code for validation only (don't strip the response)
            extracted_code = self._extract_code(full_response)

            # Validate generated code
            is_valid, issues = self.test_rules.validate_generated_code(extracted_code)

            # Record in session (store extracted code for refinement)
            if session:
                session.add_assistant_message(
                    extracted_code,
                    metadata={"validation_passed": is_valid},
                )
                session.record_generated_test(
                    class_name=class_name,
                    test_code=extracted_code,
                    success=is_valid,
                )

            logger.info(
                "Test generation complete",
                class_name=class_name,
                validation_passed=is_valid,
                issues=issues,
            )

            # Return FULL response (not just extracted code) so user sees
            # Analysis Report + Test Plan + Implementation + Coverage Notes
            return GenerationResult(
                success=True,
                test_code=full_response,
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

        # Lấy lại RAG context từ session cache để LLM có đủ thông tin source code
        rag_chunks: list[CodeChunk] = []
        if session.current_file:
            cache_key = f"{last_test.class_name}:{session.current_file}"
            cached = session.get_cached_rag_context(cache_key)
            if cached:
                rag_chunks = [CodeChunk(**c) for c in cached]

        # Build refinement prompt
        _, issues = self.test_rules.validate_generated_code(last_test.test_code)
        refinement_prompt = self.prompt_builder.build_refinement_prompt(
            original_code=last_test.test_code,
            feedback=feedback,
            validation_issues=issues,
            rag_chunks=rag_chunks,
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
        inline_source: Optional[str] = None,
    ) -> list[CodeChunk]:
        """Exact graph traversal: lấy target class → phân tích deps+used_types → parallel fetch tất cả.
        
        If deps/used_types from Qdrant are empty, falls back to extracting
        type names from the inline source code (Continue IDE sends code inline).
        """
        cache_key = f"{class_name}:{file_path}"
        if session:
            cached = session.get_cached_rag_context(cache_key)
            if cached:
                logger.debug("Using cached RAG context", class_name=class_name)
                return [CodeChunk(**c) for c in cached]

        chunks: list[CodeChunk] = []
        existing_fqns: set[str] = set()

        # Query 1: lấy đúng 1 chunk service target
        main_result = self.rag.search_by_class(
            class_name=class_name,
            top_k=1,
            include_dependencies=False,
        )
        main_chunk = main_result.chunks[0] if main_result.chunks else None
        if not main_chunk:
            logger.warning("Service target not found in index", class_name=class_name)
            return []
        chunks.append(main_chunk)
        existing_fqns.add(main_chunk.fully_qualified_name)

        # Gộp dependencies + used_types, loại trùng lặp và loại chính nó
        deps = set(main_chunk.dependencies or [])
        used = set(main_chunk.used_types or [])
        types_to_fetch = (deps | used) - {main_chunk.class_name, class_name}

        # Fallback: nếu Qdrant không có deps/used_types, parse từ inline source
        if not types_to_fetch and inline_source:
            fallback_types = self._extract_types_from_source(inline_source, class_name)
            if fallback_types:
                types_to_fetch = fallback_types
                logger.info(
                    "Using fallback type extraction from inline source",
                    class_name=class_name,
                    fallback_types=sorted(types_to_fetch),
                )

        if not types_to_fetch:
            logger.info(
                "RAG context retrieved (target only, no deps/used_types)",
                class_name=class_name,
            )
            return chunks

        # Parallel fetch: lấy chunk cho từng dependency/used_type cùng lúc
        def _fetch_one(type_name: str) -> Optional[CodeChunk]:
            try:
                result = self.rag.search_by_class(
                    class_name=type_name,
                    top_k=1,
                    include_dependencies=False,
                )
                return result.chunks[0] if result.chunks else None
            except Exception as e:
                logger.warning(
                    "Failed to fetch dependency chunk",
                    type_name=type_name,
                    error=str(e),
                )
                return None

        with ThreadPoolExecutor(max_workers=min(5, len(types_to_fetch))) as executor:
            future_map = {
                executor.submit(_fetch_one, t): t for t in types_to_fetch
            }
            for future in as_completed(future_map):
                chunk = future.result()
                if chunk and chunk.fully_qualified_name not in existing_fqns:
                    chunks.append(chunk)
                    existing_fqns.add(chunk.fully_qualified_name)

        # Cache in session
        if session:
            session.cache_rag_context(
                cache_key,
                [c.model_dump() for c in chunks],
            )

        logger.info(
            "RAG context retrieved (target+deps+used_types)",
            class_name=class_name,
            total_chunks=len(chunks),
            types_requested=len(types_to_fetch),
            fetched=len(chunks) - 1,
        )

        return chunks[:self.top_k]

    def _extract_types_from_source(self, source: str, class_name: str) -> set[str]:
        """Extract domain type names from inline Java source code.
        
        Fallback for when Qdrant has empty dependencies/used_types.
        Parses import statements and field declarations to find
        domain-specific types that should be fetched from RAG.
        """
        types: set[str] = set()
        
        # Well-known prefixes to skip
        _SKIP_PREFIXES = {
            "java.", "javax.", "jakarta.",
            "org.springframework.", "org.junit.", "org.mockito.",
            "org.slf4j.", "org.apache.", "com.fasterxml.",
            "lombok.",
        }
        _SKIP_SIMPLE = {
            "String", "Integer", "Long", "Double", "Float", "Boolean",
            "UUID", "Map", "Set", "List", "Page", "Pageable", "Optional",
            "BigDecimal", "LocalDate", "LocalDateTime", "Instant",
            "ResponseEntity", "HttpStatus",
        }
        
        for line in source.split("\n"):
            line = line.strip()
            
            # Parse import statements: import com.example.domain.Resource;
            if line.startswith("import "):
                import_path = line.replace("import ", "").replace(";", "").strip()
                # Skip well-known framework imports
                if any(import_path.startswith(p) for p in _SKIP_PREFIXES):
                    continue
                # Extract simple class name from FQN
                simple_name = import_path.split(".")[-1]
                if simple_name != class_name and simple_name not in _SKIP_SIMPLE:
                    types.add(simple_name)
            
            # Parse field declarations: private final ResourceQueryService resourceQueryService;
            field_match = re.match(
                r'\s*(?:private|protected|public)?\s*(?:final\s+)?(\w+)\s+\w+\s*;',
                line,
            )
            if field_match:
                field_type = field_match.group(1)
                if (field_type[0].isupper() and 
                    field_type != class_name and 
                    field_type not in _SKIP_SIMPLE):
                    types.add(field_type)
        
        return types

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

