"""
Test generation rules and constraints.
Enforces JUnit5 + Mockito patterns without Spring test context.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TestRules:
    """Rules and constraints for test generation."""

    # Required imports
    required_imports: list[str] = field(
        default_factory=lambda: [
            "org.junit.jupiter.api.Test",
            "org.junit.jupiter.api.BeforeEach",
            "org.junit.jupiter.api.DisplayName",
            "org.junit.jupiter.api.extension.ExtendWith",
            "org.mockito.Mock",
            "org.mockito.InjectMocks",
            "org.mockito.junit.jupiter.MockitoExtension",
            "static org.mockito.Mockito.*",
            "static org.junit.jupiter.api.Assertions.*",
        ]
    )

    # Forbidden imports/annotations
    forbidden_patterns: list[str] = field(
        default_factory=lambda: [
            "@SpringBootTest",
            "@DataJpaTest",
            "@WebMvcTest",
            "@SpringExtension",
            "@ContextConfiguration",
            "@RunWith(SpringRunner",
            "@Autowired",  # In tests, use @InjectMocks instead
            "SpringBootTest",
            "TestRestTemplate",
            "MockMvc",  # Use direct method calls instead
        ]
    )

    # Required patterns
    required_patterns: list[str] = field(
        default_factory=lambda: [
            "@ExtendWith(MockitoExtension.class)",
            "@Mock",
            "@InjectMocks",
            "@Test",
            "@DisplayName",
        ]
    )

    # AAA pattern comments
    aaa_comments: dict[str, str] = field(
        default_factory=lambda: {
            "arrange": "// Arrange",
            "act": "// Act",
            "assert": "// Assert",
        }
    )

    def get_test_class_template(self, class_name: str) -> str:
        """Get the template for a test class."""
        return f'''import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;
import static org.mockito.ArgumentMatchers.*;

@ExtendWith(MockitoExtension.class)
@DisplayName("{class_name} Tests")
class {class_name}Test {{

    // TODO: Add @Mock fields for dependencies

    @InjectMocks
    private {class_name} underTest;

    @BeforeEach
    void setUp() {{
        // Additional setup if needed
    }}

    // TODO: Add test methods
}}'''

    def get_test_method_template(
        self, method_name: str, description: str = ""
    ) -> str:
        """Get the template for a test method."""
        display_name = description or f"should {method_name} successfully"
        return f'''
    @Test
    @DisplayName("{display_name}")
    void {method_name}_shouldSucceed() {{
        // Arrange
        // TODO: Set up test data and mock behaviors

        // Act
        // TODO: Call the method under test

        // Assert
        // TODO: Verify results and interactions
    }}'''

    def get_mock_setup_example(self, dependency_type: str, method: str) -> str:
        """Get example mock setup code."""
        return f"when({dependency_type.lower()}.{method}(any())).thenReturn(/* expected value */);"

    def get_verify_example(self, dependency_type: str, method: str) -> str:
        """Get example verification code."""
        return f"verify({dependency_type.lower()}).{method}(any());"

    def validate_generated_code(self, code: str) -> tuple[bool, list[str]]:
        """Validate generated test code against rules."""
        issues = []

        # Check for forbidden patterns
        for pattern in self.forbidden_patterns:
            if pattern in code:
                issues.append(f"Forbidden pattern found: {pattern}")

        # Check for required patterns (at least some should be present)
        required_found = sum(1 for p in self.required_patterns if p in code)
        if required_found < 3:
            issues.append("Missing required JUnit5/Mockito annotations")

        # Check for AAA pattern
        has_arrange = "// Arrange" in code or "// arrange" in code.lower()
        has_act = "// Act" in code or "// act" in code.lower()
        has_assert = "// Assert" in code or "// assert" in code.lower()

        if not (has_arrange and has_act and has_assert):
            issues.append("AAA pattern comments missing (// Arrange, // Act, // Assert)")

        # Check for @Test annotation
        if "@Test" not in code:
            issues.append("No @Test annotations found")

        # Check for verify calls (interaction testing)
        if "verify(" not in code:
            issues.append("No verify() calls found - consider adding interaction verification")

        return len(issues) == 0, issues


@dataclass
class LayerRules:
    """Rules for DDD layer detection and handling."""

    layer_patterns: dict[str, list[str]] = field(
        default_factory=lambda: {
            "application": [
                "*Service",
                "*ServiceImpl",
                "*UseCase",
                "*Handler",
                "*Facade",
                "*ApplicationService",
            ],
            "domain": [
                "*Entity",
                "*ValueObject",
                "*DomainService",
                "*Aggregate",
                "*AggregateRoot",
                "*DomainEvent",
                "*Specification",
            ],
            "infrastructure": [
                "*Repository",
                "*RepositoryImpl",
                "*Adapter",
                "*Client",
                "*Gateway",
                "*Mapper",
                "*JpaRepository",
            ],
        }
    )

    def detect_layer(self, class_name: str, package: str = "") -> str:
        """Detect the DDD layer for a class."""
        # Check package first
        package_lower = package.lower()
        if "application" in package_lower or "service" in package_lower:
            return "application"
        if "domain" in package_lower or "model" in package_lower:
            return "domain"
        if "infrastructure" in package_lower or "persistence" in package_lower:
            return "infrastructure"

        # Check class name patterns
        for layer, patterns in self.layer_patterns.items():
            for pattern in patterns:
                pattern_suffix = pattern.replace("*", "")
                if class_name.endswith(pattern_suffix):
                    return layer

        return "unknown"

    def get_test_focus(self, layer: str) -> str:
        """Get testing focus based on layer."""
        focus_map = {
            "application": "Test business logic, orchestration, and service interactions",
            "domain": "Test domain rules, invariants, and value object behavior",
            "infrastructure": "Test data mapping, repository queries, and external integrations",
        }
        return focus_map.get(layer, "Test core functionality and edge cases")

