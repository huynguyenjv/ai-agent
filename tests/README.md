# tests/ — Development Test Suites

Test files for validating the agent pipeline. Organized by development phase.

## Files

| File | Description | Runner |
|------|-------------|--------|
| `test_phase1.py` | State machine, planner, execution plan. | `python tests/test_phase1.py` |
| `test_phase2.py` | Intelligence layer, context builder, snippet selector, token optimizer. | `python tests/test_phase2.py` |
| `test_phase3_4.py` | Validation pipeline, repair strategies, event bus, metrics collector. | `pytest tests/test_phase3_4.py -v` |
| `_test_e2e_trace.py` | End-to-end trace: simulates AuthUseCaseService with UserProfile (record without @Builder). Verifies prompt, validation, and construction cross-check. | `python tests/_test_e2e_trace.py` |

## Running All Tests

```bash
# Phase 3/4 (pytest)
python -m pytest tests/test_phase3_4.py -v

# Phase 1 & 2 (standalone scripts — stub heavy deps to avoid torch/qdrant imports)
python tests/test_phase1.py
python tests/test_phase2.py

# E2E trace
python tests/_test_e2e_trace.py
```

## Notes

- Phase 1 and 2 tests use module-level stubs to avoid loading `torch`, `sentence-transformers`, and `qdrant-client` at import time.
- Phase 3/4 tests use standard pytest with mocking.
- The E2E trace test uses synthetic `CodeChunk` data to verify the full prompt → validation → repair chain.
