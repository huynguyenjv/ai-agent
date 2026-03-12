# Two-Phase Generation Strategy

## Overview

Two-Phase Generation is a new strategy designed to improve test generation accuracy for **complex services** that have many dependencies and domain types. The traditional single-pass approach often "hallucinates" incorrect construction patterns when the context window becomes too large.

## How It Works

### Phase 1: Analysis (No Code Generation)

The LLM **only analyzes** the service and outputs a structured JSON plan:

```json
{
  "service_name": "OrderService",
  "complexity_score": 18,
  "complexity_level": "complex",
  "constructor_dependencies": ["OrderRepository", "UserService", "PaymentGateway"],
  "all_domain_types": ["OrderRequest", "Order", "User", "PaymentResult"],
  "methods": [
    {
      "name": "createOrder",
      "return_type": "Order",
      "dependencies_called": ["orderRepository", "userService"],
      "domain_types_used": ["OrderRequest", "Order", "User"],
      "test_scenarios": [
        {"name": "happy_path", "description": "Create order successfully", "priority": 1},
        {"name": "user_not_found", "description": "User does not exist", "priority": 2}
      ]
    }
  ]
}
```

### Phase 2: Focused Generation

For each method, generate tests with:
- **Minimal context** (only the method being tested)
- **Exact construction patterns** from the Domain Registry
- **Specific test scenarios** from Phase 1 analysis

### Domain Type Registry

A pre-indexed cache of construction patterns for all domain types:

```java
// OrderRequest - record, @Builder
OrderRequest request = OrderRequest.builder()
    .userId(userId)
    .items(items)
    .build();

// User - NOT IN INDEX, use mock
User user = mock(User.class);
when(user.id()).thenReturn(userId);
```

## Benefits

| Aspect | Single-Pass | Two-Phase |
|--------|-------------|-----------|
| Context Size | Large (all at once) | Small (per method) |
| LLM Focus | Analyze + Generate | Separate concerns |
| Construction Patterns | LLM guesses | Registry provides exact patterns |
| Debugging | Hard to trace | Each phase is traceable |
| Complex Services | Often hallucinates | Much more accurate |

## When to Use

| Service Type | Complexity Score | Recommended Strategy |
|--------------|------------------|---------------------|
| Simple (2-3 deps) | 0-5 | Single-pass |
| Medium (4-6 deps) | 6-15 | Either (auto-detect) |
| Complex (7+ deps) | 16+ | Two-phase |

## API Usage

### Check Complexity

```bash
GET /v1/complexity/OrderService?collection=my_project
```

Response:
```json
{
  "class_name": "OrderService",
  "complexity_score": 18,
  "complexity_level": "complex",
  "recommended_strategy": "two_phase"
}
```

### Force Two-Phase Generation

```bash
POST /pipeline/generate-two-phase
{
  "file_path": "src/main/java/com/example/OrderService.java",
  "source_code": "package com.example; ...",
  "collection": "my_project"
}
```

### Auto-Detect Strategy

```bash
POST /pipeline/generate
{
  "file_path": "src/main/java/com/example/OrderService.java",
  "source_code": "...",
  "force_two_phase": false,
  "force_single_pass": false,
  "complexity_threshold": 10
}
```

### Registry Endpoints

```bash
# Get registry stats
GET /v1/registry/stats?collection=my_project

# Rebuild registry after reindex
POST /v1/registry/rebuild?collection=my_project

# Lookup a domain type
GET /v1/registry/lookup/OrderRequest?collection=my_project

# Get prompt section for multiple types
GET /v1/registry/prompt-section?class_names=OrderRequest,User,Order
```

## Configuration

In `config/agent.yaml`:

```yaml
two_phase:
  enabled: true
  complexity_threshold: 10
  
  analysis:
    temperature: 0.1
    max_tokens: 2000
    timeout: 60
  
  generation:
    temperature: 0.2
    max_tokens: 2000
    methods_per_batch: 1
    parallel_generation: false
  
  repair:
    max_attempts: 2
  
  fallback:
    to_single_pass: true

domain_registry:
  enabled: true
  default_collection: "java_codebase"
  auto_rebuild_on_index: true
```

## Response Format

The response includes additional metadata:

```json
{
  "success": true,
  "test_code": "...",
  "strategy_used": "two_phase",
  "complexity_score": 18,
  "analysis_result": {
    "service_name": "OrderService",
    "complexity_level": "complex",
    "methods": [...],
    "total_test_count": 8,
    "priority_1_count": 4
  }
}
```

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Two-Phase Generation                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Request    │───▶│  Complexity  │───▶│   Router     │       │
│  │              │    │  Calculator  │    │              │       │
│  └──────────────┘    └──────────────┘    └──────┬───────┘       │
│                                                  │               │
│                      ┌───────────────────────────┼───────────┐  │
│                      │                           │           │  │
│                      ▼                           ▼           │  │
│              ┌──────────────┐           ┌──────────────┐     │  │
│              │ Single-Pass  │           │  Two-Phase   │     │  │
│              │  (simple)    │           │  (complex)   │     │  │
│              └──────────────┘           └──────┬───────┘     │  │
│                                                │             │  │
│                                    ┌───────────┴───────────┐ │  │
│                                    │                       │ │  │
│                                    ▼                       ▼ │  │
│                           ┌──────────────┐        ┌──────────┴─┐│
│                           │   Phase 1    │        │   Domain   ││
│                           │  Analysis    │        │  Registry  ││
│                           │  (JSON)      │        │  Lookup    ││
│                           └──────┬───────┘        └──────┬─────┘│
│                                  │                       │      │
│                                  └───────────┬───────────┘      │
│                                              │                  │
│                                              ▼                  │
│                                     ┌──────────────┐            │
│                                     │   Phase 2    │            │
│                                     │  Generation  │            │
│                                     │ (per method) │            │
│                                     └──────┬───────┘            │
│                                            │                    │
│                                            ▼                    │
│                                     ┌──────────────┐            │
│                                     │   Assembly   │            │
│                                     │  Test Class  │            │
│                                     └──────┬───────┘            │
│                                            │                    │
│                                            ▼                    │
│                                     ┌──────────────┐            │
│                                     │  Validation  │            │
│                                     │   & Repair   │            │
│                                     └──────────────┘            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### Registry is empty

```bash
# Rebuild after indexing
POST /v1/registry/rebuild?collection=my_project
```

### Two-phase not working

Check if enabled:
```bash
GET /v1/two-phase/status
```

### Analysis fails

- Check if source code is provided
- Verify the LLM is responding correctly
- Check logs for JSON parsing errors

### Wrong construction patterns

1. Verify the type is in the registry:
   ```bash
   GET /v1/registry/lookup/OrderRequest
   ```

2. Check if the index has correct Lombok annotations:
   ```bash
   GET /index/lookup/OrderRequest
   ```

3. Reindex if needed:
   ```bash
   POST /reindex
   {
     "repo_path": "/path/to/repo",
     "recreate": true
   }
   ```

