"""
Analysis Prompt Builder — Prompts for Phase 1 (Analysis) of Two-Phase Generation.

Phase 1 asks the LLM to ANALYZE the service and output a structured JSON plan,
WITHOUT generating any test code. This separation of concerns allows:

1. Focused attention on understanding the service structure
2. Explicit identification of dependencies and domain types
3. Clear test scenario planning before code generation
4. Validation of the analysis before proceeding to generation

The analysis output drives Phase 2 (focused generation) and Registry lookups.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import structlog

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════
# Analysis Result Schema
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TestScenario:
    """A single test scenario for a method."""
    name: str                           # e.g., "happy_path", "user_not_found"
    description: str                    # What this scenario tests
    expected_behavior: str              # Expected outcome
    priority: int = 1                   # 1 = must test, 2 = should test, 3 = nice to have


@dataclass
class MethodAnalysis:
    """Analysis of a single method to test."""
    name: str                           # Method name
    return_type: str                    # Return type
    parameters: list[tuple[str, str]]   # [(type, name), ...]
    dependencies_called: list[str]      # Dependencies this method calls
    domain_types_used: list[str]        # Domain types used in this method
    test_scenarios: list[TestScenario]  # Planned test scenarios
    complexity: str = "simple"          # simple, medium, complex
    has_branching: bool = False         # Has if/else/switch
    has_exception_handling: bool = False  # Has try/catch or throws


@dataclass
class AnalysisResult:
    """Complete analysis result from Phase 1."""
    
    service_name: str
    service_fqn: str                    # Fully qualified name
    complexity_score: int               # Calculated complexity
    complexity_level: str               # simple, medium, complex
    
    # Dependencies
    constructor_dependencies: list[str]  # All @Mock-able dependencies
    
    # Domain types
    all_domain_types: list[str]         # All domain types used
    
    # Method analysis
    methods: list[MethodAnalysis]
    
    # Summary
    total_test_count: int = 0           # Estimated total tests
    priority_1_count: int = 0           # Must-have tests
    
    # Raw JSON (for debugging)
    raw_json: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisResult":
        """Parse analysis result from dictionary payload (e.g., LangGraph state)."""
        # Parse methods
        methods = []
        for m in data.get("methods", []):
            scenarios = [
                TestScenario(
                    name=s.get("name", ""),
                    description=s.get("description", ""),
                    expected_behavior=s.get("expected_behavior", ""),
                    priority=s.get("priority", 1),
                )
                for s in m.get("test_scenarios", [])
            ]
            
            methods.append(MethodAnalysis(
                name=m.get("name", ""),
                return_type=m.get("return_type", "void"),
                parameters=[(p.get("type", ""), p.get("name", "")) 
                           for p in m.get("parameters", [])],
                dependencies_called=m.get("dependencies_called", []),
                domain_types_used=m.get("domain_types_used", []),
                test_scenarios=scenarios,
                complexity=m.get("complexity", "simple"),
                has_branching=m.get("has_branching", False),
                has_exception_handling=m.get("has_exception_handling", False),
            ))
        
        # Calculate totals
        total_tests = sum(len(m.test_scenarios) for m in methods)
        priority_1 = sum(
            len([s for s in m.test_scenarios if s.priority == 1])
            for m in methods
        )
        
        return cls(
            service_name=data.get("service_name", ""),
            service_fqn=data.get("service_fqn", ""),
            complexity_score=data.get("complexity_score", 0),
            complexity_level=data.get("complexity_level", "simple"),
            constructor_dependencies=data.get("constructor_dependencies", []),
            all_domain_types=data.get("all_domain_types", []),
            methods=methods,
            total_test_count=data.get("total_test_count", total_tests),
            priority_1_count=data.get("priority_1_count", priority_1),
            raw_json=data.get("raw_json", None),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "AnalysisResult":
        """Parse analysis result from LLM JSON output."""
        try:
            data = json.loads(json_str)
            data["raw_json"] = json_str
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse analysis JSON", error=str(e))
            raise ValueError(f"Invalid analysis JSON: {e}")
    
    def get_method(self, method_name: str) -> Optional[MethodAnalysis]:
        """Get analysis for a specific method."""
        return next((m for m in self.methods if m.name == method_name), None)
    
    def get_types_for_method(self, method_name: str) -> list[str]:
        """Get domain types needed for a specific method's tests."""
        method = self.get_method(method_name)
        if method:
            return method.domain_types_used
        return []
    
    def get_scenarios_for_method(self, method_name: str) -> list[TestScenario]:
        """Get test scenarios for a specific method."""
        method = self.get_method(method_name)
        if method:
            return method.test_scenarios
        return []
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "service_name": self.service_name,
            "service_fqn": self.service_fqn,
            "complexity_score": self.complexity_score,
            "complexity_level": self.complexity_level,
            "constructor_dependencies": self.constructor_dependencies,
            "all_domain_types": self.all_domain_types,
            "methods": [
                {
                    "name": m.name,
                    "return_type": m.return_type,
                    "parameters": [{"type": t, "name": n} for t, n in m.parameters],
                    "dependencies_called": m.dependencies_called,
                    "domain_types_used": m.domain_types_used,
                    "test_scenarios": [
                        {
                            "name": s.name,
                            "description": s.description,
                            "expected_behavior": s.expected_behavior,
                            "priority": s.priority,
                        }
                        for s in m.test_scenarios
                    ],
                    "complexity": m.complexity,
                    "has_branching": m.has_branching,
                    "has_exception_handling": m.has_exception_handling,
                }
                for m in self.methods
            ],
            "total_test_count": self.total_test_count,
            "priority_1_count": self.priority_1_count,
        }


# ═══════════════════════════════════════════════════════════════════════
# Analysis Prompt Builder
# ═══════════════════════════════════════════════════════════════════════

class AnalysisPromptBuilder:
    """Builds prompts for Phase 1 (Analysis) of Two-Phase Generation."""
    
    def build_system_prompt(self) -> str:
        """Build system prompt for analysis phase."""
        return """You are a Senior Java Test Architect analyzing a service class.

YOUR TASK: Analyze the service and output a structured JSON plan.
DO NOT generate any test code. Only analyze and plan.

ANALYSIS RULES:
1. Identify ALL constructor dependencies (these become @Mock fields)
2. Identify ALL domain types used (records, entities, DTOs, models)
3. For each public method:
   - List dependencies it calls
   - List domain types it uses
   - Plan test scenarios based on ACTUAL code behavior
   - Mark complexity (simple/medium/complex)

TEST SCENARIO RULES:
- Priority 1 (MUST): Happy path for every method
- Priority 2 (SHOULD): Error paths that EXIST in code (explicit throw, if/else with error)
- Priority 3 (NICE): Edge cases for complex branching
- DO NOT invent scenarios for behaviors not in the code
- If method just delegates (calls dependency and returns), only test delegation

COMPLEXITY SCORING:
- Dependencies count × 2
- Branching points (if/else/switch) × 3
- Service calls × 2
- Total 0-5: simple, 6-15: medium, 16+: complex

OUTPUT FORMAT: Valid JSON only, no markdown, no explanation."""

    def build_analysis_prompt(
        self,
        source_code: str,
        class_name: str,
        file_path: str,
        available_types: Optional[list[str]] = None,
    ) -> str:
        """Build the analysis prompt for Phase 1."""
        
        available_types_section = ""
        if available_types:
            available_types_list = "\n".join(f"- {t}" for t in sorted(set(available_types)))
            available_types_section = f"""
## Available Types in Context
(Only select domain types and constructor dependencies that appear in this list, or in the source code directly. Do NOT invent types that don't exist here.)
{available_types_list}
"""

        return f"""Analyze this Java service and output a JSON test plan.

## Service Source Code
```java
{source_code}
```
{available_types_section}
## Output JSON Schema
{{
    "service_name": "{class_name}",
    "service_fqn": "full.package.{class_name}",
    "complexity_score": <number>,
    "complexity_level": "simple|medium|complex",
    "constructor_dependencies": [
        "DependencyName1",
        "DependencyName2"
    ],
    "all_domain_types": [
        "DomainType1",
        "DomainType2"
    ],
    "methods": [
        {{
            "name": "methodName",
            "return_type": "ReturnType",
            "parameters": [
                {{"type": "ParamType", "name": "paramName"}}
            ],
            "dependencies_called": ["dep1", "dep2"],
            "domain_types_used": ["Type1", "Type2"],
            "test_scenarios": [
                {{
                    "name": "scenario_name",
                    "description": "What this tests",
                    "expected_behavior": "Expected outcome",
                    "priority": 1
                }}
            ],
            "complexity": "simple|medium|complex",
            "has_branching": true|false,
            "has_exception_handling": true|false
        }}
    ]
}}

IMPORTANT:
1. Output ONLY valid JSON, no markdown code blocks
2. Include ALL public methods
3. Include ALL constructor parameters as dependencies
4. Include ALL domain types (records, entities, DTOs, models)
5. Test scenarios must reflect ACTUAL code behavior only
6. DO NOT invent exception scenarios not in the code"""

    def build_focused_generation_prompt(
        self,
        method_analysis: MethodAnalysis,
        registry_context: str,
        mock_fields: list[str],
        class_name: str,
    ) -> str:
        """Build focused prompt for Phase 2 (single method generation).
        
        This prompt is much smaller and focused compared to full generation.
        """
        
        # Format scenarios
        scenarios_text = "\n".join(
            f"  {i+1}. **{s.name}** (Priority {s.priority}): {s.description}\n"
            f"     Expected: {s.expected_behavior}"
            for i, s in enumerate(method_analysis.test_scenarios)
        )
        
        # Format parameters
        params_text = ", ".join(
            f"{t} {n}" for t, n in method_analysis.parameters
        ) or "none"
        
        return f"""Generate test methods for `{method_analysis.name}` in `{class_name}`.

## Method Signature
```java
{method_analysis.return_type} {method_analysis.name}({params_text})
```

## Test Scenarios to Implement
{scenarios_text}

## Mock Fields Available
```java
{chr(10).join(f'@Mock {m};' for m in mock_fields)}
@InjectMocks {class_name} {class_name[0].lower() + class_name[1:]};
```

## Dependencies This Method Calls
{', '.join(method_analysis.dependencies_called) or 'None'}

{registry_context}

## Output Requirements
1. Generate ONLY test methods for the scenarios above
2. Use @Test and @DisplayName for each method
3. Follow AAA pattern with comments: // Arrange, // Act, // Assert
4. Use EXACT construction patterns from Domain Types section
5. Verify mock interactions with verify()
6. Method naming: methodName_WhenCondition_ShouldResult

Output the test methods only, no class declaration or imports."""

    def build_assembly_prompt(
        self,
        class_name: str,
        package_name: str,
        mock_fields: list[str],
        test_methods: list[str],
        imports: list[str],
    ) -> str:
        """Build prompt for assembling final test class.
        
        This is used to combine generated test methods into a complete class.
        """
        
        mock_fields_code = "\n    ".join(f"@Mock\n    {m};" for m in mock_fields)
        test_methods_code = "\n\n".join(test_methods)
        imports_code = "\n".join(f"import {imp};" for imp in sorted(set(imports)))
        
        return f"""Assemble these test methods into a complete JUnit5 test class.

## Package
{package_name}

## Required Imports
{imports_code}

## Mock Fields
{mock_fields_code}

## Test Methods
{test_methods_code}

## Output Requirements
1. Complete, compilable test class
2. @ExtendWith(MockitoExtension.class) on class
3. All imports at top
4. @Mock fields before @InjectMocks
5. @BeforeEach setUp() if needed for shared data
6. All test methods included

Output the complete test class."""


# ═══════════════════════════════════════════════════════════════════════
# JSON Extraction Helper
# ═══════════════════════════════════════════════════════════════════════

def extract_json_from_response(response: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks."""
    import re
    
    # Try to find JSON in code block
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    
    # Try to find raw JSON (starts with { and ends with })
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json_match.group(0).strip()
    
    # Return as-is and let JSON parser handle errors
    return response.strip()

