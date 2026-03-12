"""
Domain Type Registry — Pre-indexed construction patterns for domain types.

This registry provides deterministic, pre-computed construction information
for all domain types (records, entities, DTOs, models) in the codebase.

Instead of letting the LLM guess how to construct domain objects, the registry
provides exact, copy-paste ready examples based on actual source code analysis.

Usage::

    registry = DomainTypeRegistry(qdrant_client)
    registry.build_from_collection("java_codebase")
    
    info = registry.lookup("OrderRequest")
    # info.construction_pattern = "builder"
    # info.example_code = "OrderRequest.builder().userId(id).items(items).build()"
    
    # Batch lookup for multiple types
    infos = registry.lookup_batch(["OrderRequest", "User", "Order"])

The registry is built from Qdrant index metadata, which already contains:
- java_type (record, class, enum, interface)
- has_builder, has_data, has_value (Lombok annotations)
- fields, record_components (field information)
- has_no_args_constructor, has_all_args_constructor
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = structlog.get_logger()


class ConstructionPattern(str, Enum):
    """How to construct a domain type."""
    BUILDER = "builder"                    # Use .builder()...build()
    CONSTRUCTOR = "constructor"            # Use new ClassName(...)
    SETTER = "setter"                      # Use new ClassName() + setters
    FACTORY = "factory"                    # Use factory method
    MOCK = "mock"                          # Use mock(ClassName.class)
    ENUM_CONSTANT = "enum_constant"        # Use EnumName.VALUE


@dataclass
class FieldInfo:
    """Information about a field/component."""
    name: str
    type: str
    nullable: bool = True
    has_default: bool = False


@dataclass
class DomainTypeInfo:
    """Pre-computed construction information for a domain type."""
    
    class_name: str
    fully_qualified_name: str
    java_type: str                          # record, class, enum, interface
    construction_pattern: ConstructionPattern
    example_code: str                       # Copy-paste ready example
    fields: list[FieldInfo] = field(default_factory=list)
    
    # Lombok info
    has_builder: bool = False
    has_builder_to_builder: bool = False
    has_data: bool = False
    has_value: bool = False
    has_getter: bool = False
    has_setter: bool = False
    has_no_args_constructor: bool = False
    has_all_args_constructor: bool = False
    
    # Enum constants (for enum types)
    enum_constants: list[str] = field(default_factory=list)
    
    # Metadata
    found_in_index: bool = True
    reason: str = ""                        # Why this pattern was chosen
    
    def get_field_names(self) -> list[str]:
        """Get list of field names."""
        return [f.name for f in self.fields]
    
    def get_field_types_map(self) -> dict[str, str]:
        """Get field name → type mapping."""
        return {f.name: f.type for f in self.fields}
    
    def get_stub_example(self, field_name: str) -> str:
        """Get example stub for a field accessor."""
        field_info = next((f for f in self.fields if f.name == field_name), None)
        if not field_info:
            return f"when(obj.{field_name}()).thenReturn(/* value */);"
        
        # Generate appropriate return value based on type
        return_value = self._get_default_value(field_info.type)
        
        # Record uses field() accessor, class uses getField()
        if self.java_type == "record":
            return f"when(obj.{field_name}()).thenReturn({return_value});"
        else:
            getter = f"get{field_name[0].upper()}{field_name[1:]}"
            return f"when(obj.{getter}()).thenReturn({return_value});"
    
    def _get_default_value(self, type_str: str) -> str:
        """Get a sensible default value for a type."""
        type_lower = type_str.lower()
        
        if "string" in type_lower:
            return '"test"'
        if "uuid" in type_lower:
            return 'UUID.randomUUID()'
        if "long" in type_lower:
            return "1L"
        if "int" in type_lower:
            return "1"
        if "boolean" in type_lower:
            return "true"
        if "double" in type_lower or "float" in type_lower:
            return "1.0"
        if "bigdecimal" in type_lower:
            return 'BigDecimal.valueOf(100)'
        if "list" in type_lower:
            return "List.of()"
        if "set" in type_lower:
            return "Set.of()"
        if "map" in type_lower:
            return "Map.of()"
        if "optional" in type_lower:
            return "Optional.empty()"
        if "localdate" in type_lower:
            return "LocalDate.of(2024, 1, 15)"
        if "localdatetime" in type_lower:
            return "LocalDateTime.of(2024, 1, 15, 10, 30)"
        if "instant" in type_lower:
            return "Instant.now()"
        
        # Unknown type - suggest mock
        return f"mock({type_str}.class)"


@dataclass
class RegistryStats:
    """Statistics about the registry."""
    total_types: int = 0
    by_pattern: dict[str, int] = field(default_factory=dict)
    by_java_type: dict[str, int] = field(default_factory=dict)
    build_time_ms: float = 0.0


class DomainTypeRegistry:
    """Registry of domain types with pre-computed construction patterns.
    
    The registry is built from Qdrant index metadata and provides
    deterministic construction information for test generation.
    """
    
    # Types to include in registry (domain types only)
    _DOMAIN_LAYERS = {"domain", "model", "dto", "entity", "vo"}
    _DOMAIN_TYPES = {"entity", "dto", "model", "record", "domain", "vo", "request", "response"}
    
    def __init__(
        self,
        qdrant_client: QdrantClient,
        default_collection: str = "java_codebase",
    ):
        self.qdrant = qdrant_client
        self.default_collection = default_collection
        
        # In-memory cache: collection_name → {class_name → DomainTypeInfo}
        self._cache: dict[str, dict[str, DomainTypeInfo]] = {}
        self._stats: dict[str, RegistryStats] = {}
    
    # ═══════════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════════
    
    def build_from_collection(
        self,
        collection_name: Optional[str] = None,
        force_rebuild: bool = False,
    ) -> RegistryStats:
        """Build/rebuild the registry from a Qdrant collection.
        
        Args:
            collection_name: Qdrant collection to scan. Defaults to default_collection.
            force_rebuild: If True, rebuild even if cache exists.
        
        Returns:
            RegistryStats with build information.
        """
        collection = collection_name or self.default_collection
        
        if not force_rebuild and collection in self._cache:
            logger.debug("Registry cache hit", collection=collection)
            return self._stats.get(collection, RegistryStats())
        
        start_time = time.time()
        logger.info("Building domain type registry", collection=collection)
        
        # Scroll through all domain types in the collection
        registry: dict[str, DomainTypeInfo] = {}
        stats = RegistryStats()
        
        try:
            offset = None
            while True:
                points, offset = self.qdrant.scroll(
                    collection_name=collection,
                    scroll_filter=self._build_domain_filter(),
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset,
                )
                
                if not points:
                    break
                
                for point in points:
                    info = self._build_type_info(point.payload)
                    if info:
                        registry[info.class_name] = info
                        
                        # Update stats
                        pattern = info.construction_pattern.value
                        stats.by_pattern[pattern] = stats.by_pattern.get(pattern, 0) + 1
                        stats.by_java_type[info.java_type] = stats.by_java_type.get(info.java_type, 0) + 1
                
                if offset is None:
                    break
            
            stats.total_types = len(registry)
            stats.build_time_ms = (time.time() - start_time) * 1000
            
            # Cache results
            self._cache[collection] = registry
            self._stats[collection] = stats
            
            logger.info(
                "Domain type registry built",
                collection=collection,
                total_types=stats.total_types,
                by_pattern=stats.by_pattern,
                build_time_ms=round(stats.build_time_ms, 1),
            )
            
            return stats
            
        except Exception as e:
            logger.error("Failed to build registry", collection=collection, error=str(e))
            return RegistryStats()
    
    def lookup(
        self,
        class_name: str,
        collection_name: Optional[str] = None,
    ) -> DomainTypeInfo:
        """Look up construction info for a domain type.
        
        Args:
            class_name: Simple class name (e.g., "OrderRequest").
            collection_name: Qdrant collection. Defaults to default_collection.
        
        Returns:
            DomainTypeInfo with construction pattern and example.
            If not found, returns a mock-based fallback.
        """
        collection = collection_name or self.default_collection
        
        # Ensure registry is built
        if collection not in self._cache:
            self.build_from_collection(collection)
        
        registry = self._cache.get(collection, {})
        
        if class_name in registry:
            return registry[class_name]
        
        # Not found - return mock fallback
        return DomainTypeInfo(
            class_name=class_name,
            fully_qualified_name=class_name,
            java_type="unknown",
            construction_pattern=ConstructionPattern.MOCK,
            example_code=f"{class_name} obj = mock({class_name}.class);",
            found_in_index=False,
            reason="Type not found in index - use mock and stub only accessed methods",
        )
    
    def lookup_batch(
        self,
        class_names: list[str],
        collection_name: Optional[str] = None,
    ) -> dict[str, DomainTypeInfo]:
        """Look up construction info for multiple domain types.
        
        Args:
            class_names: List of simple class names.
            collection_name: Qdrant collection.
        
        Returns:
            Dict mapping class_name → DomainTypeInfo.
        """
        return {
            name: self.lookup(name, collection_name)
            for name in class_names
        }
    
    def get_construction_code(
        self,
        class_name: str,
        collection_name: Optional[str] = None,
        variable_name: Optional[str] = None,
    ) -> str:
        """Get copy-paste ready construction code for a type.
        
        Args:
            class_name: Simple class name.
            collection_name: Qdrant collection.
            variable_name: Variable name to use. Defaults to camelCase of class_name.
        
        Returns:
            Java code string for constructing the type.
        """
        info = self.lookup(class_name, collection_name)
        var_name = variable_name or self._to_camel_case(class_name)
        
        if info.construction_pattern == ConstructionPattern.MOCK:
            return f"{class_name} {var_name} = mock({class_name}.class);"
        
        return f"{class_name} {var_name} = {info.example_code};"
    
    def get_prompt_section(
        self,
        class_names: list[str],
        collection_name: Optional[str] = None,
    ) -> str:
        """Generate a prompt section with construction examples for multiple types.
        
        This is designed to be inserted directly into the LLM prompt.
        
        Args:
            class_names: List of domain type names.
            collection_name: Qdrant collection.
        
        Returns:
            Formatted prompt section with construction examples.
        """
        if not class_names:
            return ""
        
        lines = [
            "## Domain Type Construction (COPY EXACTLY)",
            "",
            "Use these EXACT patterns to construct domain objects.",
            "DO NOT guess or invent different patterns.",
            "",
            "```java",
        ]
        
        for class_name in class_names:
            info = self.lookup(class_name, collection_name)
            
            # Add comment with pattern info
            if info.found_in_index:
                pattern_hint = f"// {class_name} - {info.java_type}"
                if info.has_builder:
                    pattern_hint += ", @Builder"
                if info.has_data:
                    pattern_hint += ", @Data"
                if info.has_value:
                    pattern_hint += ", @Value"
            else:
                pattern_hint = f"// {class_name} - NOT IN INDEX, use mock"
            
            lines.append(pattern_hint)
            lines.append(self.get_construction_code(class_name, collection_name))
            
            # Add field info for reference
            if info.fields and info.found_in_index:
                field_str = ", ".join(f"{f.type} {f.name}" for f in info.fields[:5])
                if len(info.fields) > 5:
                    field_str += ", ..."
                lines.append(f"// Fields: {field_str}")
            
            lines.append("")
        
        lines.append("```")
        
        return "\n".join(lines)
    
    def get_stats(self, collection_name: Optional[str] = None) -> RegistryStats:
        """Get registry statistics."""
        collection = collection_name or self.default_collection
        return self._stats.get(collection, RegistryStats())
    
    def clear_cache(self, collection_name: Optional[str] = None) -> None:
        """Clear the registry cache."""
        if collection_name:
            self._cache.pop(collection_name, None)
            self._stats.pop(collection_name, None)
        else:
            self._cache.clear()
            self._stats.clear()
        logger.info("Registry cache cleared", collection=collection_name or "all")
    
    def is_cached(self, collection_name: Optional[str] = None) -> bool:
        """Check if registry is cached for a collection."""
        collection = collection_name or self.default_collection
        return collection in self._cache
    
    # ═══════════════════════════════════════════════════════════════════
    # Internal Methods
    # ═══════════════════════════════════════════════════════════════════
    
    def _build_domain_filter(self) -> models.Filter:
        """Build Qdrant filter for domain types."""
        # Filter for class-level elements (not methods)
        # that are likely domain types
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="element_type",
                    match=models.MatchValue(value="class"),
                ),
            ],
            should=[
                # Match by layer
                models.FieldCondition(
                    key="layer",
                    match=models.MatchAny(any=list(self._DOMAIN_LAYERS)),
                ),
                # Match by type
                models.FieldCondition(
                    key="type",
                    match=models.MatchAny(any=list(self._DOMAIN_TYPES)),
                ),
                # Match records (always domain types)
                models.FieldCondition(
                    key="java_type",
                    match=models.MatchValue(value="record"),
                ),
                # Match enums (often domain types)
                models.FieldCondition(
                    key="java_type",
                    match=models.MatchValue(value="enum"),
                ),
            ],
        )
    
    def _build_type_info(self, payload: dict) -> Optional[DomainTypeInfo]:
        """Build DomainTypeInfo from Qdrant payload."""
        class_name = payload.get("class_name")
        if not class_name:
            return None
        
        java_type = payload.get("java_type", "class")
        
        # Extract fields
        fields = self._extract_fields(payload)
        
        # Determine construction pattern
        pattern, example, reason = self._determine_construction(
            class_name=class_name,
            java_type=java_type,
            payload=payload,
            fields=fields,
        )
        
        # Extract enum constants if enum
        enum_constants = []
        if java_type == "enum":
            # Enum constants might be in annotations or a dedicated field
            enum_constants = payload.get("enum_constants", [])
        
        return DomainTypeInfo(
            class_name=class_name,
            fully_qualified_name=payload.get("fully_qualified_name", class_name),
            java_type=java_type,
            construction_pattern=pattern,
            example_code=example,
            fields=fields,
            has_builder=payload.get("has_builder", False),
            has_builder_to_builder=payload.get("has_builder_to_builder", False),
            has_data=payload.get("has_data", False),
            has_value=payload.get("has_value", False),
            has_getter=payload.get("has_getter", False),
            has_setter=payload.get("has_setter", False),
            has_no_args_constructor=payload.get("has_no_args_constructor", False),
            has_all_args_constructor=payload.get("has_all_args_constructor", False),
            enum_constants=enum_constants,
            found_in_index=True,
            reason=reason,
        )
    
    def _extract_fields(self, payload: dict) -> list[FieldInfo]:
        """Extract field information from payload."""
        fields = []
        
        # Try record_components first (for records)
        components = payload.get("record_components", [])
        if components:
            for comp in components:
                if isinstance(comp, dict):
                    fields.append(FieldInfo(
                        name=comp.get("name", ""),
                        type=comp.get("type", "Object"),
                    ))
            return fields
        
        # Fall back to fields (for classes)
        field_list = payload.get("fields", [])
        for f in field_list:
            if isinstance(f, dict):
                fields.append(FieldInfo(
                    name=f.get("name", ""),
                    type=f.get("type", "Object"),
                ))
        
        return fields
    
    def _determine_construction(
        self,
        class_name: str,
        java_type: str,
        payload: dict,
        fields: list[FieldInfo],
    ) -> tuple[ConstructionPattern, str, str]:
        """Determine the best construction pattern for a type.
        
        Returns:
            (pattern, example_code, reason)
        """
        has_builder = payload.get("has_builder", False)
        has_builder_to_builder = payload.get("has_builder_to_builder", False)
        has_data = payload.get("has_data", False)
        has_value = payload.get("has_value", False)
        has_getter = payload.get("has_getter", False)
        has_setter = payload.get("has_setter", False)
        has_no_args = payload.get("has_no_args_constructor", False)
        has_all_args = payload.get("has_all_args_constructor", False)
        
        # ─── Enum ───────────────────────────────────────────────────────
        if java_type == "enum":
            constants = payload.get("enum_constants", [])
            if constants:
                example = f"{class_name}.{constants[0]}"
            else:
                example = f"{class_name}.VALUE"
            return (
                ConstructionPattern.ENUM_CONSTANT,
                example,
                "Enum type - use constant directly",
            )
        
        # ─── Interface ──────────────────────────────────────────────────
        if java_type == "interface":
            return (
                ConstructionPattern.MOCK,
                f"mock({class_name}.class)",
                "Interface - must mock or use implementation",
            )
        
        # ─── Record ─────────────────────────────────────────────────────
        if java_type == "record":
            if has_builder:
                example = self._build_builder_example(class_name, fields, has_builder_to_builder)
                return (
                    ConstructionPattern.BUILDER,
                    example,
                    "Record with @Builder - use builder pattern",
                )
            else:
                # Canonical constructor
                example = self._build_constructor_example(class_name, fields)
                return (
                    ConstructionPattern.CONSTRUCTOR,
                    example,
                    "Record without @Builder - use canonical constructor",
                )
        
        # ─── Class with @Builder ────────────────────────────────────────
        if has_builder:
            example = self._build_builder_example(class_name, fields, has_builder_to_builder)
            return (
                ConstructionPattern.BUILDER,
                example,
                "@Builder annotation - use builder pattern",
            )
        
        # ─── @Value (immutable) ─────────────────────────────────────────
        if has_value:
            example = self._build_constructor_example(class_name, fields)
            return (
                ConstructionPattern.CONSTRUCTOR,
                example,
                "@Value (immutable) - use all-args constructor",
            )
        
        # ─── @Data or @Getter + @Setter ─────────────────────────────────
        if has_data or (has_getter and has_setter):
            if has_all_args:
                example = self._build_constructor_example(class_name, fields)
                return (
                    ConstructionPattern.CONSTRUCTOR,
                    example,
                    "@Data + @AllArgsConstructor - use constructor",
                )
            elif has_no_args:
                example = self._build_setter_example(class_name, fields)
                return (
                    ConstructionPattern.SETTER,
                    example,
                    "@Data - use no-args constructor + setters",
                )
        
        # ─── @AllArgsConstructor ────────────────────────────────────────
        if has_all_args:
            example = self._build_constructor_example(class_name, fields)
            return (
                ConstructionPattern.CONSTRUCTOR,
                example,
                "@AllArgsConstructor - use all-args constructor",
            )
        
        # ─── @NoArgsConstructor + setters ───────────────────────────────
        if has_no_args and has_setter:
            example = self._build_setter_example(class_name, fields)
            return (
                ConstructionPattern.SETTER,
                example,
                "@NoArgsConstructor + @Setter - use setters",
            )
        
        # ─── Fallback: try constructor if fields known ──────────────────
        if fields:
            example = self._build_constructor_example(class_name, fields)
            return (
                ConstructionPattern.CONSTRUCTOR,
                example,
                "Plain class with fields - try constructor",
            )
        
        # ─── Last resort: mock ──────────────────────────────────────────
        return (
            ConstructionPattern.MOCK,
            f"mock({class_name}.class)",
            "Unknown construction pattern - use mock for safety",
        )
    
    def _build_builder_example(
        self,
        class_name: str,
        fields: list[FieldInfo],
        has_to_builder: bool = False,
    ) -> str:
        """Build a builder pattern example."""
        if not fields:
            return f"{class_name}.builder().build()"
        
        # Use first 3 fields for example
        field_calls = []
        for f in fields[:3]:
            value = self._get_example_value(f.type, f.name)
            field_calls.append(f".{f.name}({value})")
        
        if len(fields) > 3:
            field_calls.append("/* ... */")
        
        return f"{class_name}.builder(){''.join(field_calls)}.build()"
    
    def _build_constructor_example(
        self,
        class_name: str,
        fields: list[FieldInfo],
    ) -> str:
        """Build a constructor example."""
        if not fields:
            return f"new {class_name}()"
        
        args = []
        for f in fields[:5]:
            value = self._get_example_value(f.type, f.name)
            args.append(value)
        
        if len(fields) > 5:
            args.append("/* ... */")
        
        return f"new {class_name}({', '.join(args)})"
    
    def _build_setter_example(
        self,
        class_name: str,
        fields: list[FieldInfo],
    ) -> str:
        """Build a setter-based construction example."""
        var_name = self._to_camel_case(class_name)
        
        lines = [f"new {class_name}()"]
        
        for f in fields[:3]:
            setter = f"set{f.name[0].upper()}{f.name[1:]}"
            value = self._get_example_value(f.type, f.name)
            lines.append(f"    .{setter}({value})")
        
        if len(fields) > 3:
            lines.append("    /* ... */")
        
        # Note: This returns a fluent-style example, but actual setters
        # might not be fluent. The LLM should adapt.
        return ";\n".join(lines)
    
    def _get_example_value(self, type_str: str, field_name: str) -> str:
        """Get an example value for a field."""
        type_lower = type_str.lower()
        name_lower = field_name.lower()
        
        # Context-aware values based on field name
        if "id" in name_lower:
            if "uuid" in type_lower:
                return "UUID.randomUUID()"
            return "1L"
        if "email" in name_lower:
            return '"test@example.com"'
        if "name" in name_lower:
            return '"Test Name"'
        if "status" in name_lower:
            # Likely an enum - use the type name
            if type_str[0].isupper() and "string" not in type_lower:
                return f"{type_str}.ACTIVE"
            return '"ACTIVE"'
        if "date" in name_lower or "time" in name_lower:
            if "instant" in type_lower:
                return "Instant.now()"
            if "localdatetime" in type_lower:
                return "LocalDateTime.of(2024, 1, 15, 10, 30)"
            if "localdate" in type_lower:
                return "LocalDate.of(2024, 1, 15)"
        if "amount" in name_lower or "price" in name_lower:
            if "bigdecimal" in type_lower:
                return "BigDecimal.valueOf(100)"
            return "100.0"
        if "count" in name_lower or "quantity" in name_lower:
            return "1"
        if "enabled" in name_lower or "active" in name_lower or "flag" in name_lower:
            return "true"
        
        # Type-based fallback
        if "string" in type_lower:
            return f'"{field_name}Value"'
        if "uuid" in type_lower:
            return "UUID.randomUUID()"
        if "long" in type_lower:
            return "1L"
        if "int" in type_lower:
            return "1"
        if "boolean" in type_lower:
            return "true"
        if "double" in type_lower or "float" in type_lower:
            return "1.0"
        if "bigdecimal" in type_lower:
            return "BigDecimal.ONE"
        if "list" in type_lower:
            return "List.of()"
        if "set" in type_lower:
            return "Set.of()"
        if "map" in type_lower:
            return "Map.of()"
        if "optional" in type_lower:
            return "Optional.empty()"
        
        # Unknown type - use variable name as placeholder
        return f"{field_name}Value"
    
    def _to_camel_case(self, class_name: str) -> str:
        """Convert ClassName to camelCase variable name."""
        if not class_name:
            return "obj"
        return class_name[0].lower() + class_name[1:]


# ═══════════════════════════════════════════════════════════════════════
# Singleton accessor for global registry
# ═══════════════════════════════════════════════════════════════════════

_global_registry: Optional[DomainTypeRegistry] = None


def get_domain_registry(
    qdrant_client: Optional[QdrantClient] = None,
    default_collection: str = "java_codebase",
) -> DomainTypeRegistry:
    """Get or create the global domain type registry.
    
    Args:
        qdrant_client: Qdrant client. Required on first call.
        default_collection: Default collection name.
    
    Returns:
        The global DomainTypeRegistry instance.
    """
    global _global_registry
    
    if _global_registry is None:
        if qdrant_client is None:
            raise ValueError("qdrant_client required on first call to get_domain_registry")
        _global_registry = DomainTypeRegistry(qdrant_client, default_collection)
    
    return _global_registry


def reset_domain_registry() -> None:
    """Reset the global registry (for testing)."""
    global _global_registry
    _global_registry = None

