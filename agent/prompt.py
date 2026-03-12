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
3) Coverage Notes: expected coverage %, what's tested vs not tested, and rationale

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ CRITICAL ACCURACY RULE (MUST FOLLOW)

You MUST test ONLY behaviors that actually exist in the source code.
- Read EACH method body carefully before deciding what to test.
- If a method just DELEGATES to a dependency (calls one method and returns the result), test ONLY:
  1. That it calls the dependency with the correct arguments
  2. That it returns exactly what the dependency returns
  DO NOT invent exception scenarios for delegation methods.
- Add failure/exception test ONLY IF the method body contains:
  - explicit throw statement
  - if/else or switch with error handling
  - try/catch blocks
- NEVER assume a method throws an exception unless you can SEE the throw statement in its body.
- NEVER invent behaviors, constructors, methods, or fields not present in the provided source.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — CODE ANALYSIS (MANDATORY)
- Dependencies = number of @Mock-able collaborators
- Branches = if/else, switch cases, try/catch (count ONLY what EXISTS in source)
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
- For each method:
  - 1 happy path test (ALWAYS)
  - failure/exception test ONLY IF the method body has explicit throw/if/switch/try-catch
  - If the method just delegates, DO NOT add a failure test for it.
- Priority 1 (MUST): happy path
- Priority 2 (SHOULD): failure paths that EXIST in source code
- Priority 3 (NICE): edge cases for complex branching logic

Stop when you have the minimum correct set (prefer fewer correct tests over many).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — IMPLEMENTATION RULES (STRICT)

✅ MUST
- Use @ExtendWith(MockitoExtension.class) ONLY (NO Spring test annotations)
- Mock ALL dependencies with @Mock, service with @InjectMocks
- AAA pattern with comments: // Arrange, // Act, // Assert
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
- NO inventing exception scenarios not present in code

DOMAIN SOURCE RULE (CRITICAL)
- Use accessors exactly as in source:
    - record → .field()
    - class → getX()/isX() as defined
    - @Builder → MUST use Builder pattern: ClassName.builder().field(value).build()
    - @Builder(toBuilder=true) → can also use obj.toBuilder().field(newValue).build()
    - @Data/@Getter/@Setter → use new ClassName() + setters OR all-args constructor
    - @NoArgsConstructor → new ClassName() is available
    - @AllArgsConstructor → new ClassName(all, fields, in, order) is available
    - record WITHOUT @Builder → use canonical constructor: new RecordName(field1, field2, ...)
    - record WITH @Builder → MUST use RecordName.builder().field(value).build()

IMPORTANT:
If record does NOT have @Builder annotation, NEVER use builder pattern.
Do NOT use new ClassName(...) if class/record has @Builder (use builder pattern instead)
Do NOT assume .toBuilder() exists unless @Builder(toBuilder=true) is present

IMPORT RULE (CRITICAL — COPY EXACTLY)
- Copy ALL imports verbatim from the source service code.
- In particular, check the EXACT package for each type:
    E.g., if the service imports `vtrip.tech.microservice.core.starter.security.ActorType`,
    then the test MUST import from `vtrip.tech.microservice.core.starter.security.ActorType`
    — NOT from a guessed package like `vtrip.app.domain.auth.ActorType`.
- For EVERY type used in the test that also appears in the service source code,
  use the EXACT same import statement. Do NOT guess or infer a different package.

STATIC UTILITIES (ONLY IF PRESENT IN SOURCE)
- If service uses static utilities: mock via Mockito MockedStatic (try-with-resources).
- SecurityContextHolder, LocalDateTime.now(), UUID.randomUUID() → MUST use MockedStatic.
  Example (MANDATORY pattern for SecurityContextHolder):
  ```
  // Declare mock objects BEFORE the try block
  Authentication authentication = mock(Authentication.class);
  SecurityContext securityContext = mock(SecurityContext.class);

  try (MockedStatic<SecurityContextHolder> securityMock =
           mockStatic(SecurityContextHolder.class)) {
      securityMock.when(SecurityContextHolder::getContext).thenReturn(securityContext);
      when(securityContext.getAuthentication()).thenReturn(authentication);
      when(authentication.getName()).thenReturn("user@example.com");
      // ... Act & Assert ...

      // VERIFY on the mock object directly:
      verify(authentication).getName();          // ✅ CORRECT
      // verify(SecurityContextHolder.getContext().getAuthentication()).getName();  ❌ WRONG
  }
  ```
  NEVER call SecurityContextHolder.setContext() or .getContext() directly in tests.
  NEVER verify via chained static calls — verify on the mock object variable instead.

INTERMEDIATE MOCK CALLS (CRITICAL)
- If the source code calls a dependency method and passes the result to another call:
  ```
  jwtUseCase.createToken(ActorType.USER_ACTOR, id,
      jwtUseCase.buildClaims(ActorType.USER_ACTOR, refId, provider));
  ```
  You MUST mock BOTH calls:
  ```
  Map<String, Object> claims = Map.of("role", "user");
  when(jwtUseCase.buildClaims(eq(ActorType.USER_ACTOR), eq(refId), eq(provider))).thenReturn(claims);
  when(jwtUseCase.createToken(eq(ActorType.USER_ACTOR), eq(id), eq(claims))).thenReturn(jwtToken);
  ```
  If you skip mocking `buildClaims()`, it returns null, and `createToken()` gets null as argument.
  Trace EVERY dependency call in order, mock ALL of them.

DOMAIN TYPE SAFETY (CRITICAL)
- If a domain type (UserProfile, User, etc.) is NOT provided in the Codebase Context section:
  DO NOT guess its builder fields, constructor args, or accessor methods.
  Instead:
  1. Use `mock(ClassName.class)` and stub ONLY the accessors actually called in the method body.
     Example: if the source calls `user.id()`, `user.email()`, `user.refUserId()`, stub ONLY those:
     ```
     User user = mock(User.class);
     when(user.id()).thenReturn(UUID.fromString(userId));
     when(user.email()).thenReturn(email);
     when(user.refUserId()).thenReturn(UUID.fromString(refUserId));
     ```
  2. If the domain type IS provided in Context, use ONLY the fields/types shown there.
  3. NEVER invent builder fields not shown in the source or context.
- This is especially important for types returned by external APIs (e.g., OpenAPIRepository).

NAMING
- method_WhenCondition_ShouldExpectedResult (do NOT prefix with "test")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — ANTI-PATTERNS (MUST AVOID)

1. **IMPORT PACKAGE MISMATCH**: Copy imports exactly from the source code.
   If source has `import vtrip.tech.microservice.core.starter.security.ActorType`,
   the test MUST use the same import. Do NOT guess a different package.

2. **INCONSISTENT MOCK VALUES**: If setUp creates `userPassword` with field "encoded-password",
   then when(encoder.matches(raw, encoded)) MUST use "encoded-password" — NOT a different string.
   Rule: every literal in when()/verify() MUST match the data used in that test method.

3. **MISSING @Mock DEPENDENCIES**: Count ALL constructor parameters of the service under test.
   Each one MUST have a corresponding @Mock field. If you miss one, @InjectMocks will fail.
   BEFORE writing tests, LIST all constructor params and create @Mock for each.

4. **LocalDateTime.now() / Instant.now() IN TEST DATA**: Never construct expected objects
   with `.now()` — the value at construction time ≠ the value inside the method under test.
   Instead: use ArgumentCaptor to capture the actual object, OR use any() matcher, OR
   mock the clock.

5. **OVERLY VERBOSE setUp**: @BeforeEach should initialize ONLY shared data like userId, email.
   DO NOT pre-build every domain object in setUp — build them inside each test method so
   each test is self-contained and values are guaranteed consistent.
   If a domain type source is NOT provided, use mock(Type.class) + stub accessors.

6. **MISSING INTERMEDIATE MOCK CHAINS**: If method calls `dep.foo()` and passes the result
   to `dep.bar(dep.foo())`, you MUST mock BOTH `dep.foo()` AND `dep.bar()`.
   Trace the FULL call chain in the method body, mock EVERY dependency call in order.
   DO NOT rely on `any()` to skip mocking — the unmocked call returns null → NPE.

7. **GUESSING DOMAIN TYPE BUILDERS**: If the exact source code of a domain type is NOT
   provided in the Codebase Context, DO NOT build it with .builder() or constructor.
   Use mock(Type.class) + stub the accessors the method actually calls.
   This prevents type mismatches, wrong field names, and missing fields.

8. **RAW VALUES INSTEAD OF ENUMS**: If source code uses `user.status().ordinal()`, then
   `status` is an enum — use the enum constant (e.g., UserStatus.ACTIVE), NOT integer 1.

9. **WRONG CONSTRUCTION PATTERN**: DO NOT use .builder() if the class/record has no @Builder.
   DO NOT use new Constructor() if the class has @Builder. Follow the construction hints
   provided in the Domain Types section EXACTLY.

10. **VERIFY ON CHAINED STATIC CALLS**: Inside MockedStatic, verify mock objects DIRECTLY:
    `verify(authentication).getName()` ✅ — NOT `verify(SecurityContextHolder.getContext().getAuthentication()).getName()` ❌

If any required signature is missing, DO NOT INVENT:
- Use mock(Type.class) + stub accessors for unknown domain types.
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

        # Extract source code from task_description if present (Continue IDE inline)
        source_code = None
        task_text = task_description or ""
        if task_description:
            # Parse out the code block from Continue IDE format
            import re
            code_match = re.search(r'```(?:[^\n]*\.java)?\s*\n(.*?)```', task_description, re.DOTALL)
            if code_match:
                source_code = code_match.group(1).strip()
                # Remove the code block from task text to get clean instruction
                task_text = re.sub(r'```[^`]*```', '', task_description, flags=re.DOTALL).strip()
                if not task_text:
                    task_text = f"Generate unit tests for `{class_name}`"

        parts.append(f"## Task\n{task_text}")

        # Source code section (clean, separate from task instruction)
        if source_code:
            parts.append(f"\n## Source Code Under Test\n```java\n{source_code}\n```")

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

            # Group chunks by role
            main_class = None
            used_type_names: set[str] = set()
            dep_names: set[str] = set()
            domain_types = []   # models/DTOs/entities from used_types
            dependencies = []   # service-level deps
            related = []        # other related code

            # Determine used_type names and dependency names from main class chunk
            for chunk in rag_chunks:
                if chunk.class_name == class_name:
                    main_class = chunk
                    used_type_names = set(chunk.used_types or [])
                    dep_names = set(chunk.dependencies or [])
                    break

            # Service suffixes for smart classification
            _SERVICE_SUFFIXES = ("Service", "Repository", "Client", "Gateway", "Handler")
            # Domain type indicators
            _DOMAIN_TYPES = {"entity", "dto", "model", "record", "domain", "vo", "request", "response"}

            for chunk in rag_chunks:
                if chunk.class_name == class_name:
                    continue  # already handled as main_class

                # Strategy 1: use known lists if available
                if chunk.class_name in used_type_names:
                    domain_types.append(chunk)
                elif chunk.class_name in dep_names:
                    dependencies.append(chunk)
                # Strategy 2: fallback — classify by chunk metadata
                elif any(chunk.class_name.endswith(s) for s in _SERVICE_SUFFIXES):
                    dependencies.append(chunk)
                elif (
                    (chunk.java_type in ("record", "class", "enum"))
                    and (
                        (chunk.type or "").lower() in _DOMAIN_TYPES
                        or (chunk.layer or "").lower() in ("domain", "model", "dto")
                        or not any(chunk.class_name.endswith(s) for s in _SERVICE_SUFFIXES)
                    )
                ):
                    domain_types.append(chunk)
                else:
                    related.append(chunk)

            # Main class summary
            if main_class:
                lombok_info = self._get_lombok_info(main_class)
                if lombok_info:
                    parts.append(f"### Target Class Summary\nLombok: {lombok_info}\n```\n{main_class.summary}\n```")
                else:
                    parts.append(f"### Target Class Summary\n```\n{main_class.summary}\n```")

            # Domain types: FULL summary — LLM needs exact field/builder info
            if domain_types:
                parts.append("\n### Domain Types Used by This Class")
                parts.append("⚠️ Use ONLY the construction pattern shown for each type. DO NOT guess.\n")
                for dt in domain_types:
                    java_type = dt.java_type or dt.type
                    lombok_info = self._get_lombok_info(dt)
                    type_label = java_type
                    if lombok_info:
                        type_label += f", {lombok_info}"

                    # Construction hint — explicit instruction
                    construction = self._get_construction_hint(dt)
                    parts.append(f"\n**{dt.class_name}** ({type_label})")
                    if construction:
                        parts.append(construction)

                    # List record components or fields — ALL of them, with exact types
                    if (java_type == "record") and getattr(dt, "record_components", None):
                        comp_str = ", ".join(f"{rc.type} {rc.name}" for rc in dt.record_components)
                        parts.append(f"Record components (EXACT — do not add/change): `{comp_str}`")
                    elif getattr(dt, "fields", None):
                        # Show ALL fields, not truncated — exact types matter
                        field_str = ", ".join(f"{f.type} {f.name}" for f in dt.fields)
                        parts.append(f"Fields (EXACT — do not add/change): `{field_str}`")

                    parts.append(f"```\n{dt.summary}\n```")

            # ── Unfound domain types: explicitly instruct LLM to mock ──
            unfound = getattr(main_class, "unfound_types", []) if main_class else []
            if unfound:
                parts.append("\n### ⚠️ Domain Types NOT FOUND in Codebase Index")
                parts.append(
                    "The following types are referenced by imports/dependencies but "
                    "their source code is NOT available. **DO NOT guess their fields, "
                    "constructors, or builder patterns.** Use `mock(Type.class)` and "
                    "stub ONLY the accessors actually called in the method body.\n"
                )
                for uf in unfound:
                    parts.append(f"- **{uf}** → `{uf} obj = mock({uf}.class);` then `when(obj.field()).thenReturn(value);`")
                parts.append("")

            # Service dependencies (truncated)
            if dependencies:
                parts.append("\n### Service Dependencies (Mocks)")
                for dep in dependencies[:5]:
                    lombok_info = self._get_lombok_info(dep)
                    type_info = dep.type
                    if lombok_info:
                        type_info += f", {lombok_info}"
                    parts.append(f"\n**{dep.class_name}** ({type_info})")
                    parts.append(f"```\n{dep.summary[:300]}...\n```" if len(dep.summary) > 300 else f"```\n{dep.summary}\n```")

            # Related code (brief)
            if related:
                parts.append("\n### Related Code")
                for rel in related[:3]:
                    type_info = f"{rel.java_type or rel.type}, {rel.layer}"
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
2. Follow AAA pattern with comments: // Arrange, // Act, // Assert
3. Include @DisplayName for all tests
4. Verify all mock interactions
5. Test ONLY behaviors visible in the source code — do NOT invent exceptions
6. Generate complete, compilable code with correct imports (COPY imports from source)
7. If method just delegates, test delegation only (no failure path)
8. ALL constructor dependencies MUST have @Mock — count them before writing
9. SecurityContextHolder / static utilities → MUST use MockedStatic with try-with-resources
10. NEVER use LocalDateTime.now() in expected test data — use ArgumentCaptor or any()
11. Keep literal values consistent: same value in setUp/when/verify/assertEquals
12. If domain type source NOT provided → use mock(Type.class) + stub only accessed fields
13. Mock ALL intermediate dependency calls (e.g., buildClaims before createToken)
14. Verify MockedStatic via mock object variables, NOT chained static calls
15. Use enum constants from source, NOT raw int/string equivalents""")

        return "\n".join(parts)

    def build_refinement_prompt(
        self,
        original_code: str,
        feedback: str,
        validation_issues: list[str],
        rag_chunks: list[CodeChunk] = None,
    ) -> str:
        """Build prompt for refining generated test code."""
        parts = [
            "## Refinement Request",
            f"The following test code needs improvements:\n\n```java\n{original_code}\n```",
        ]

        # Đưa codebase context vào để LLM không hallucinate khi sửa test
        if rag_chunks:
            context_summary = self.build_context_summary(rag_chunks)
            if context_summary:
                parts.append(f"\n## Codebase Context (Reference)\nUse this to verify correct class signatures, constructors, and method names:\n\n{context_summary}")

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
5. Use ONLY accessors, constructors, and builders exactly as shown in Codebase Context
6. Return the complete corrected test code""")

        return "\n".join(parts)

    def build_incremental_update_prompt(
        self,
        class_name: str,
        file_path: str,
        rag_chunks: list[CodeChunk],
        existing_test_code: str,
        tested_methods: list[str] = None,
        changed_methods: list[str] = None,
        task_description: Optional[str] = None,
    ) -> str:
        """Build prompt for incremental test update.

        The LLM receives the existing test file and is asked to generate
        ONLY new/updated test methods for uncovered or changed methods.
        """
        parts = []

        # Task
        if changed_methods:
            methods_str = ", ".join(f"`{m}`" for m in changed_methods)
            parts.append(
                f"## Task\nThe following methods were changed/added in `{class_name}`: {methods_str}.\n"
                f"Generate ONLY the additional test methods needed for these changes."
            )
        else:
            parts.append(
                f"## Task\nGenerate additional unit tests for `{class_name}` "
                f"covering methods NOT yet tested in the existing test file."
            )

        if task_description:
            parts.append(f"\nAdditional instructions: {task_description}")

        # Target class info
        parts.append(f"\n## Target Class\n- Class: `{class_name}`\n- File: `{file_path}`")

        # Existing test code
        parts.append(
            f"\n## Existing Test File\n"
            f"This test file already exists. DO NOT regenerate tests for methods already covered.\n\n"
            f"```java\n{existing_test_code}\n```"
        )

        # Already tested methods
        if tested_methods:
            methods_list = ", ".join(f"`{m}`" for m in tested_methods)
            parts.append(
                f"\n## Already Tested Methods\n"
                f"The following methods already have tests — DO NOT duplicate them: {methods_list}"
            )

        # RAG context (reuse same logic as test generation)
        if rag_chunks:
            context_summary = self.build_context_summary(rag_chunks)
            if context_summary:
                parts.append(f"\n## Codebase Context\n{context_summary}")

        # Output instructions
        parts.append("""
## Output Requirements
1. Generate ONLY new test methods — not a complete test class
2. Follow the EXACT same style, naming conventions, and patterns as the existing test file
3. Use the same @Mock fields already declared in the existing test
4. If new @Mock fields are needed, list them at the top with a comment: // NEW MOCK FIELDS NEEDED
5. Each test method must have @Test and @DisplayName
6. Follow AAA pattern with comments: // Arrange, // Act, // Assert
7. DO NOT include class declaration, imports, or @BeforeEach — only new test methods
8. Use method naming: methodName_WhenCondition_ShouldExpectedResult""")

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

    def _get_construction_hint(self, chunk: CodeChunk) -> str:
        """Generate explicit construction instruction for a domain type.
        
        Returns a clear, unambiguous instruction telling the LLM exactly
        how to instantiate this type. Critical for preventing hallucination
        of .builder() on records/classes that don't have @Builder.
        """
        java_type = getattr(chunk, 'java_type', None) or chunk.type
        has_builder = getattr(chunk, 'has_builder', False)
        has_builder_to_builder = getattr(chunk, 'has_builder_to_builder', False)
        has_data = getattr(chunk, 'has_data', False)
        has_value = getattr(chunk, 'has_value', False)
        has_getter = getattr(chunk, 'has_getter', False)
        has_setter = getattr(chunk, 'has_setter', False)
        has_no_args = getattr(chunk, 'has_no_args_constructor', False)
        has_all_args = getattr(chunk, 'has_all_args_constructor', False)
        name = chunk.class_name

        # --- Records ---
        if java_type == "record":
            if has_builder:
                hint = f"✅ HAS @Builder → MUST use: `{name}.builder().field(value).build()`"
                if has_builder_to_builder:
                    hint += f"\n   Also supports: `obj.toBuilder().field(newValue).build()`"
                return hint
            else:
                # Build canonical constructor example from record_components
                components = getattr(chunk, 'record_components', None)
                if components:
                    args = ", ".join(rc.name for rc in components)
                    return (f"⚠️ NO @Builder → MUST use canonical constructor: "
                            f"`new {name}({args})`\n"
                            f"   NEVER use {name}.builder(). It does NOT exist.")
                else:
                    return (f"⚠️ NO @Builder → MUST use canonical constructor: "
                            f"`new {name}(...)`\n"
                            f"   NEVER use {name}.builder(). It does NOT exist.")

        # --- Enums ---
        if java_type == "enum":
            return f"Enum — use `{name}.VALUE` constants directly."

        # --- Interfaces ---
        if java_type == "interface":
            return "Interface — mock or use implementing class."

        # --- Classes ---
        if has_builder:
            hint = f"✅ HAS @Builder → MUST use: `{name}.builder().field(value).build()`"
            if has_builder_to_builder:
                hint += f"\n   Also supports: `obj.toBuilder().field(newValue).build()`"
            return hint

        # @Value → immutable, all-args constructor
        if has_value:
            fields = getattr(chunk, 'fields', None)
            if fields:
                args = ", ".join(f.name for f in fields)
                return f"@Value (immutable) → use: `new {name}({args})`"
            return f"@Value (immutable) → use all-args constructor: `new {name}(...)`"

        # @Data or @Getter+@Setter → new + setters
        if has_data or (has_getter and has_setter):
            if has_all_args:
                return (f"@Data + @AllArgsConstructor → use: `new {name}(all, args, in, order)` "
                        f"OR `new {name}()` + setters")
            if has_no_args:
                return f"@Data → use: `new {name}()` then call `.setField(value)` for each field"
            return f"@Data → use: `new {name}()` + setters, or all-args constructor if available"

        # @AllArgsConstructor only
        if has_all_args:
            if has_no_args:
                return (f"Use: `new {name}(all, args, ...)` "
                        f"OR `new {name}()` + setters (if @Setter present)")
            return f"Use: `new {name}(all, args, in, order)`"

        # @NoArgsConstructor only
        if has_no_args:
            return f"Use: `new {name}()` then set fields via setters"

        # Plain class — no Lombok
        return ""

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        from utils.tokenizer import count_tokens
        return count_tokens(text)

