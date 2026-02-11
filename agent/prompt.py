"""
Prompt builder for test generation.
Constructs compact, effective prompts using RAG context.
"""

from typing import Optional

from .rules import TestRules, LayerRules
from .memory import SessionMemory
from rag.schema import CodeChunk


class PromptBuilder:
    """Builds prompts for test generation."""

    def __init__(self):
        self.test_rules = TestRules()
        self.layer_rules = LayerRules()

    def build_system_prompt(self) -> str:
        """Build the system prompt for test generation."""
        return """
    You are a Senior Java Backend Developer writing UNIT TESTS only.

SCOPE
- Java 21, Spring Boot 3.x
- JUnit 5 + Mockito
- Service layer ONLY (NO integration tests)

INPUT (I will paste)
- ONE service class source code
- Any referenced domain/DTO/exception/mappers signatures needed to compile tests

OUTPUT (MUST)
1) Analysis Report: complexity + test plan with priorities
2) Complete JUnit5+Mockito test class that compiles:
    - Path: src/test/java/{same package as service}/{ServiceName}Test.java
3) Coverage Notes: expected coverage %, what’s tested vs not tested, and rationale

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — CODE ANALYSIS (MANDATORY)
- Dependencies = number of @Mock-able collaborators
- Branches = if/else, switch cases, try/catch
- Service Calls = calls to dependencies
- Complexity Score = (Dependencies×2) + (Branches×3) + (Service Calls×2)

Classification:
- 0–5: SIMPLE  → 2–3 tests → 90–100%
- 6–15: MEDIUM → 4–6 tests → 60–80%
- 16+: COMPLEX → 7–10 tests → 40–60%

Add this analysis as a JavaDoc comment at the top of the test class.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — TEST PLANNING
- List ALL public methods of the service.
- For each method: 1 happy path + 1 critical failure path.
- Add edge cases ONLY when there are meaningful branches.
- Priority 1 (MUST): happy + critical errors
- Priority 2 (SHOULD): key branch edges
- Priority 3 (NICE): rare scenarios, concurrency (only if relevant)

Stop when you have the minimum correct set (prefer fewer correct tests over many).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — IMPLEMENTATION RULES (STRICT)

✅ MUST
- Use @ExtendWith(MockitoExtension.class) ONLY (NO Spring test annotations)
- Mock ALL dependencies with @Mock, service with @InjectMocks
- AAA pattern (Given/When/Then)
- @BeforeEach: initialize DATA only (NO stubbing in setup)
- Verify mock interactions (verify/never/times/inOrder when needed)
- Use eq(...) for important args; use any()/anyString() ONLY for verify negative or when arg truly irrelevant
- Tests must be independent (no shared mutable state across tests)

❌ NEVER
- NO @SpringBootTest / @DataJpaTest / @WebMvcTest / @Autowired / @MockBean
- NO testing private methods
- NO lenient()
- NO when(voidMethod()).thenReturn(...) (void: no stub or doThrow/doNothing if needed)
- NO guessing fields/constructors/methods/classes not present in provided source
- NO mocking domain models unless unavoidable (prefer real objects)

DOMAIN SOURCE RULE (CRITICAL)
- Use accessors exactly as in source:
    - record → .field()
    - class → getX()/isX() as defined
- Create objects based on annotations:
    - @Builder → MUST use Builder pattern: ClassName.builder().field(value).build()
    - @Builder(toBuilder=true) → can also use obj.toBuilder().field(newValue).build()
    - @Data/@Getter/@Setter → use new ClassName() + setters OR all-args constructor
    - @NoArgsConstructor → new ClassName() is available
    - @AllArgsConstructor → new ClassName(all, fields, in, order) is available
    - record WITHOUT @Builder → use canonical constructor: new RecordName(field1, field2, ...)
    - record WITH @Builder → MUST use RecordName.builder().field(value).build()
- Do NOT use new ClassName(...) if class/record has @Builder (use builder pattern instead)
- Do NOT assume .toBuilder() exists unless @Builder(toBuilder=true) is present

STATIC UTILITIES (ONLY IF PRESENT IN SOURCE)
- If service uses static utilities: mock via Mockito MockedStatic (try-with-resources).

NAMING
- method_WhenCondition_ShouldExpectedResult (do NOT prefix with “test”)

If any required signature is missing, DO NOT INVENT:
- Write only safe tests + list exactly what missing source is needed to complete.

NOW: analyze and write tests for the following service:
 
"""

    def build_test_generation_prompt(
        self,
        class_name: str,
        file_path: str,
        rag_chunks: list[CodeChunk],
        task_description: Optional[str] = None,
        session: Optional[SessionMemory] = None,
    ) -> str:
        """Build the user prompt for test generation."""
        parts = []

        # Task description
        if task_description:
            parts.append(f"## Task\n{task_description}")
        else:
            parts.append(f"## Task\nGenerate comprehensive unit tests for `{class_name}`")

        # Target class info
        parts.append(f"\n## Target Class\n- Class: `{class_name}`\n- File: `{file_path}`")

        # Detect layer and add focus
        layer = self.layer_rules.detect_layer(class_name)
        if layer != "unknown":
            focus = self.layer_rules.get_test_focus(layer)
            parts.append(f"- Layer: {layer}\n- Testing Focus: {focus}")

        # RAG context - class summaries
        if rag_chunks:
            parts.append("\n## Codebase Context")
            parts.append("Use this context to understand the class and its dependencies:\n")

            # Group by type
            main_class = None
            dependencies = []
            related = []

            for chunk in rag_chunks:
                if chunk.class_name == class_name:
                    main_class = chunk
                elif chunk.class_name in [c.class_name for c in rag_chunks if c.class_name == class_name]:
                    dependencies.append(chunk)
                else:
                    related.append(chunk)

            # Main class summary
            if main_class:
                lombok_info = self._get_lombok_info(main_class)
                if lombok_info:
                    parts.append(f"### Target Class Summary\nLombok: {lombok_info}\n```\n{main_class.summary}\n```")
                else:
                    parts.append(f"### Target Class Summary\n```\n{main_class.summary}\n```")

            # Dependencies
            if dependencies:
                parts.append("\n### Dependencies")
                for dep in dependencies[:5]:
                    lombok_info = self._get_lombok_info(dep)
                    type_info = dep.type
                    if lombok_info:
                        type_info += f", {lombok_info}"
                    parts.append(f"\n**{dep.class_name}** ({type_info})")
                    parts.append(f"```\n{dep.summary[:300]}...\n```" if len(dep.summary) > 300 else f"```\n{dep.summary}\n```")

            # Related code
            if related:
                parts.append("\n### Related Code")
                for rel in related[:5]:
                    # Build type info with Lombok annotations
                    type_info = f"{rel.type}, {rel.layer}"
                    if hasattr(rel, 'java_type') and rel.java_type:
                        type_info = f"{rel.java_type}, {type_info}"
                    
                    # Add Lombok info
                    lombok_info = self._get_lombok_info(rel)
                    if lombok_info:
                        type_info += f", {lombok_info}"
                    
                    parts.append(f"\n**{rel.class_name}** ({type_info})")
                    summary = rel.summary[:200] + "..." if len(rel.summary) > 200 else rel.summary
                    parts.append(f"```\n{summary}\n```")

        # Session context
        if session:
            # Previous conversation context
            conv_context = session.get_conversation_context(max_turns=3)
            if conv_context:
                parts.append(f"\n## Previous Context\n{conv_context}")

            # Previous tests for reference
            if session.generated_tests:
                recent_test = session.generated_tests[-1]
                if recent_test.success and recent_test.class_name != class_name:
                    parts.append(f"\n## Reference: Previous Test Style\nFollow similar patterns as the previously generated test for `{recent_test.class_name}`")

        # Requirements reminder
        parts.append("""
## Requirements
1. Use JUnit 5 + Mockito ONLY (no Spring test context)
2. Follow AAA pattern with comments
3. Include @DisplayName for all tests
4. Verify all mock interactions
5. Cover: happy path, edge cases, null inputs, exceptions
6. Generate complete, compilable code with imports""")

        return "\n".join(parts)

    def build_refinement_prompt(
        self,
        original_code: str,
        feedback: str,
        validation_issues: list[str],
    ) -> str:
        """Build prompt for refining generated test code."""
        parts = [
            "## Refinement Request",
            f"The following test code needs improvements:\n\n```java\n{original_code}\n```",
        ]

        if feedback:
            parts.append(f"\n## User Feedback\n{feedback}")

        if validation_issues:
            parts.append("\n## Validation Issues")
            for issue in validation_issues:
                parts.append(f"- {issue}")

        parts.append("""
## Instructions
1. Fix all validation issues
2. Address user feedback
3. Maintain JUnit 5 + Mockito patterns
4. Keep AAA structure
5. Return the complete corrected test code""")

        return "\n".join(parts)

    def build_context_summary(self, chunks: list[CodeChunk], max_tokens: int = 2000) -> str:
        """Build a compact context summary from RAG chunks."""
        if not chunks:
            return ""

        summaries = []
        current_tokens = 0
        approx_chars_per_token = 4

        for chunk in chunks:
            # Include java_type and Lombok info
            java_type = getattr(chunk, 'java_type', None) or chunk.type
            lombok_info = self._get_lombok_info(chunk)
            
            type_label = f"{java_type}"
            if lombok_info:
                type_label += f" {lombok_info}"
            
            summary = f"[{type_label}] {chunk.class_name}: {chunk.summary}"
            summary_tokens = len(summary) // approx_chars_per_token

            if current_tokens + summary_tokens > max_tokens:
                break

            summaries.append(summary)
            current_tokens += summary_tokens

        return "\n\n".join(summaries)

    def _get_lombok_info(self, chunk: CodeChunk) -> str:
        """Extract Lombok annotation info from a chunk for prompt context."""
        lombok_parts = []
        
        if getattr(chunk, 'has_builder', False):
            if getattr(chunk, 'has_builder_to_builder', False):
                lombok_parts.append("@Builder(toBuilder=true)")
            else:
                lombok_parts.append("@Builder")
        
        if getattr(chunk, 'has_data', False):
            lombok_parts.append("@Data")
        elif getattr(chunk, 'has_value', False):
            lombok_parts.append("@Value")
        else:
            if getattr(chunk, 'has_getter', False):
                lombok_parts.append("@Getter")
            if getattr(chunk, 'has_setter', False):
                lombok_parts.append("@Setter")
        
        return " ".join(lombok_parts)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4

