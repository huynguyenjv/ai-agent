# agent/ — Core Orchestration

The brain of the system. Orchestrates the full test generation workflow using a finite state machine + planner architecture.

## Pipeline

```
Request → Planner → ExecutionPlan
                        │
              StateMachine executes:
              IDLE → PLANNING → RETRIEVING → GENERATING → VALIDATING
                                                              │
                                                   ┌──────────┴──────────┐
                                                   ▼                     ▼
                                              COMPLETED             REPAIRING
                                                                        │
                                                                        ▼
                                                              (back to VALIDATING)
                                                                  or FAILED
```

## Files

| File | Description |
|------|-------------|
| `orchestrator.py` | **Main coordinator.** `AgentOrchestrator` executes plans via state machine. Entry points: `generate_test()`, `refine_test()`. Delegates to RAG, vLLM, validation, repair. |
| `state_machine.py` | `StateMachine` — finite-state runtime with transition guards and history tracking. |
| `planner.py` | `Planner` — separates planning from execution. Produces `ExecutionPlan` from requests. |
| `plan.py` | Data structures: `ExecutionPlan`, `PlanStep`, `StepAction`, `TaskType`, `StepStatus`. |
| `prompt.py` | `PromptBuilder` — constructs system/user prompts with RAG context, rules, field info, Lombok hints, unfound-type warnings. |
| `rules.py` | `TestRules` + `LayerRules` — JUnit5/Mockito constraints, forbidden patterns, layer-specific templates. |
| `validation.py` | `ValidationPipeline` — 7-pass severity-aware validation (structural, patterns, quality, safety, construction cross-check). |
| `repair.py` | `RepairStrategySelector` — targeted repair strategies per issue category. Builds focused repair prompts. |
| `memory.py` | `SessionMemory` + `MemoryManager` — session state, conversation history, RAG context caching. |
| `memory_store.py` | `MemoryStore` ABC, `InMemoryStore`, `RedisStore` — pluggable persistence backends. |
| `events.py` | `EventBus` — decoupled pub/sub for lifecycle events. Thread-safe. |
| `metrics.py` | `MetricsCollector` — subscribes to EventBus, collects timing/counts/quality metrics. |

## Validation Passes

| Pass | Check |
|------|-------|
| 1 | Structural — has `@Test`, `@ExtendWith`, class declaration |
| 2 | Import completeness — missing imports for used types |
| 3 | Mock patterns — `@Mock`/`@InjectMocks` consistency |
| 4 | Anti-patterns — forbidden Spring annotations, `new` on service classes, `Mockito.mock()` in fields |
| 5 | Quality — test naming, assertion presence, method coverage |
| 6 | Safety — hardcoded credentials, Thread.sleep, System.exit |
| 7 | Construction cross-check — `.builder()` on records without `@Builder`, wrong field names vs RAG metadata |

## Public API

```python
from agent import AgentOrchestrator, GenerationRequest

orchestrator = AgentOrchestrator(rag_client, vllm_client)
result = orchestrator.generate_test(GenerationRequest(
    class_name="UserService",
    file_path="UserService.java",
    session_id="abc-123",
))
# result.code → generated JUnit5 test source
# result.validation → ValidationResult with issues
```

## Dependencies

- `rag/` — vector search for context retrieval
- `vllm/` — LLM for code generation
- `context/` — smart context assembly (optional, Phase 2)
