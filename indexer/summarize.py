"""
Code summarizer for generating semantic summaries of Java code.
Produces compact summaries (150-250 tokens) for RAG indexing.
"""

from typing import Optional

import structlog

from .parse_java import ClassInfo, MethodInfo

logger = structlog.get_logger()


class CodeSummarizer:
    """Generate semantic summaries for Java code elements."""

    def __init__(self, min_tokens: int = 150, max_tokens: int = 250):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def summarize_class(self, class_info: ClassInfo) -> str:
        """Generate a semantic summary for a class/record/enum/interface."""
        parts = []

        # Class type and name
        class_type_label = self._get_class_type_label(class_info)
        parts.append(f"{class_type_label}: {class_info.fully_qualified_name}")

        # Java type (class, record, interface, enum, annotation)
        parts.append(f"Type: {class_info.class_type}")

        # Modifiers (abstract, final, sealed, etc.)
        if hasattr(class_info, 'modifiers') and class_info.modifiers:
            parts.append(f"Modifiers: {', '.join(class_info.modifiers)}")

        # Annotations (important for understanding purpose)
        if class_info.annotations:
            parts.append(f"Annotations: {', '.join(class_info.annotations)}")

        # Lombok annotations summary
        if hasattr(class_info, 'lombok_annotations') and class_info.lombok_annotations:
            parts.append(f"Lombok: {', '.join(class_info.lombok_annotations)}")
        
        # Builder info
        if hasattr(class_info, 'has_builder') and class_info.has_builder:
            builder_info = "@Builder"
            if class_info.has_builder_to_builder:
                builder_info += "(toBuilder=true)"
            parts.append(f"Builder: {builder_info}")

        # Inheritance
        if class_info.extends:
            parts.append(f"Extends: {class_info.extends}")
        if class_info.implements:
            parts.append(f"Implements: {', '.join(class_info.implements)}")

        # Record components (for records)
        if hasattr(class_info, 'record_components') and class_info.record_components:
            components = [f"{c.type} {c.name}" for c in class_info.record_components]
            parts.append(f"Record Components: {', '.join(components)}")
            # Add usage hint for records
            if class_info.has_builder:
                parts.append(f"Usage: {class_info.name}.builder().field(value).build()")
            else:
                args = ", ".join([c.name for c in class_info.record_components])
                parts.append(f"Usage: new {class_info.name}({args})")

        # Enum constants
        if hasattr(class_info, 'enum_constants') and class_info.enum_constants:
            constants = class_info.enum_constants[:10]  # Limit to 10
            if len(class_info.enum_constants) > 10:
                constants.append("...")
            parts.append(f"Constants: {', '.join(constants)}")

        # Javadoc summary
        if class_info.javadoc:
            doc_summary = self._extract_javadoc_summary(class_info.javadoc)
            if doc_summary:
                parts.append(f"Purpose: {doc_summary}")

        # Dependencies (injected fields) - only for classes
        if class_info.class_type == "class":
            dependencies = self._extract_dependencies(class_info)
            if dependencies:
                parts.append(f"Dependencies: {', '.join(dependencies)}")

        # Detailed fields (prefer over legacy format)
        if hasattr(class_info, 'detailed_fields') and class_info.detailed_fields:
            field_summaries = []
            for f in class_info.detailed_fields[:8]:
                field_str = f"{f.type} {f.name}"
                if f.annotations:
                    # Show important annotations
                    important_anns = [a for a in f.annotations if any(
                        x in a.lower() for x in ['@id', '@column', '@notnull', '@nullable', '@valid', '@jsonproperty']
                    )]
                    if important_anns:
                        field_str = f"{' '.join(important_anns)} {field_str}"
                field_summaries.append(field_str)
            if len(class_info.detailed_fields) > 8:
                field_summaries.append(f"... +{len(class_info.detailed_fields) - 8} more")
            parts.append(f"Fields: {'; '.join(field_summaries)}")
        elif class_info.class_type != "record" and class_info.fields:
            # Fallback to legacy format
            field_summaries = [f"{t} {n}" for t, n, _ in class_info.fields[:5]]
            if len(class_info.fields) > 5:
                field_summaries.append("...")
            parts.append(f"Fields: {', '.join(field_summaries)}")

        # Public methods summary
        public_methods = [m for m in class_info.methods if self._is_public_method(m)]
        if public_methods:
            method_signatures = [self._method_signature(m) for m in public_methods[:10]]
            parts.append(f"Methods: {'; '.join(method_signatures)}")

        summary = "\n".join(parts)
        return self._truncate_to_tokens(summary, self.max_tokens)

    def summarize_method(self, method: MethodInfo, class_info: ClassInfo) -> str:
        """Generate a semantic summary for a method."""
        parts = []

        # Method signature
        parts.append(f"Method: {class_info.fully_qualified_name}.{method.name}")
        parts.append(f"Signature: {self._method_signature(method)}")

        # Annotations
        if method.annotations:
            parts.append(f"Annotations: {', '.join(method.annotations)}")

        # Javadoc
        if method.javadoc:
            doc_summary = self._extract_javadoc_summary(method.javadoc)
            if doc_summary:
                parts.append(f"Purpose: {doc_summary}")

        # Method body summary (key operations)
        if method.body:
            body_summary = self._summarize_method_body(method.body)
            if body_summary:
                parts.append(f"Operations: {body_summary}")

        summary = "\n".join(parts)
        return self._truncate_to_tokens(summary, self.max_tokens)

    def _get_class_type_label(self, class_info: ClassInfo) -> str:
        """Determine the semantic type of the class."""
        name = class_info.name
        annotations = " ".join(class_info.annotations).lower()
        class_type = class_info.class_type

        # Check Java type first (record, enum, interface, annotation)
        if class_type == "record":
            if name.endswith("Request") or name.endswith("Response"):
                return "DTO Record"
            if name.endswith("Event"):
                return "Event Record"
            if name.endswith("Command"):
                return "Command Record"
            return "Record"
        
        if class_type == "enum":
            return "Enum"
        
        if class_type == "annotation":
            return "Annotation"
        
        if class_type == "interface":
            if name.endswith("Repository"):
                return "Repository Interface"
            if name.endswith("Service"):
                return "Service Interface"
            if name.endswith("UseCase"):
                return "UseCase Interface"
            if name.endswith("Port"):
                return "Port Interface"
            if name.endswith("Gateway"):
                return "Gateway Interface"
            return "Interface"

        # Check annotations
        if "@service" in annotations:
            return "Service"
        if "@repository" in annotations:
            return "Repository"
        if "@entity" in annotations:
            return "Entity"
        if "@controller" in annotations or "@restcontroller" in annotations:
            return "Controller"
        if "@component" in annotations:
            return "Component"

        # Check naming conventions
        if name.endswith("Service") or name.endswith("ServiceImpl"):
            return "Service"
        if name.endswith("Repository") or name.endswith("RepositoryImpl"):
            return "Repository"
        if name.endswith("Entity") or "@entity" in annotations:
            return "Entity"
        if name.endswith("Controller"):
            return "Controller"
        if name.endswith("Handler"):
            return "Handler"
        if name.endswith("UseCase"):
            return "UseCase"
        if name.endswith("Adapter"):
            return "Adapter"
        if name.endswith("Gateway"):
            return "Gateway"
        if name.endswith("Factory"):
            return "Factory"
        if name.endswith("Mapper"):
            return "Mapper"

        # Check if it's an interface
        if class_info.class_type == "interface":
            if name.endswith("Repository"):
                return "Repository Interface"
            return "Interface"

        return "Class"

    def _extract_javadoc_summary(self, javadoc: str) -> Optional[str]:
        """Extract the main description from javadoc."""
        if not javadoc:
            return None

        # Remove comment markers
        lines = javadoc.replace("/**", "").replace("*/", "").split("\n")
        summary_lines = []

        for line in lines:
            line = line.strip().lstrip("*").strip()
            # Stop at tags
            if line.startswith("@"):
                break
            if line:
                summary_lines.append(line)

        summary = " ".join(summary_lines)
        # Truncate to first sentence or 100 chars
        if "." in summary:
            summary = summary.split(".")[0] + "."
        return summary[:200] if summary else None

    def _extract_dependencies(self, class_info: ClassInfo) -> list[str]:
        """Extract injected dependencies from fields."""
        dependencies = []
        injection_annotations = {"@Autowired", "@Inject", "@Resource"}

        for field_type, field_name, annotations in class_info.fields:
            # Check for injection annotations
            has_injection = any(
                ann.startswith(inj) for ann in annotations for inj in injection_annotations
            )
            # Also consider final fields in constructor-injected classes
            if has_injection or (
                field_type.endswith("Service")
                or field_type.endswith("Repository")
                or field_type.endswith("Client")
                or field_type.endswith("Gateway")
            ):
                dependencies.append(field_type)

        return list(set(dependencies))

    def _is_public_method(self, method: MethodInfo) -> bool:
        """Check if method is public (no private/protected annotation)."""
        for ann in method.annotations:
            if "private" in ann.lower() or "protected" in ann.lower():
                return False
        # In Java, methods without explicit modifier in interface are public
        return True

    def _method_signature(self, method: MethodInfo) -> str:
        """Generate a compact method signature."""
        params = ", ".join(f"{t} {n}" for t, n in method.parameters)
        return f"{method.return_type} {method.name}({params})"

    def _summarize_method_body(self, body: str) -> Optional[str]:
        """Extract key operations from method body."""
        if not body:
            return None

        operations = []

        # Look for common patterns
        if "repository." in body.lower() or "dao." in body.lower():
            operations.append("database access")
        if ".save(" in body or ".persist(" in body:
            operations.append("persist entity")
        if ".find" in body or ".get" in body:
            operations.append("query data")
        if ".delete(" in body or ".remove(" in body:
            operations.append("delete operation")
        if "throw new" in body:
            operations.append("throws exception")
        if ".map(" in body or ".stream(" in body:
            operations.append("stream processing")
        if "validate" in body.lower():
            operations.append("validation")
        if ".publish(" in body or ".send(" in body:
            operations.append("event publishing")
        if "log." in body.lower() or "logger." in body.lower():
            operations.append("logging")

        return ", ".join(operations) if operations else None

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximate token count."""
        # Rough approximation: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit(" ", 1)[0] + "..."

    def detect_layer(self, class_info: ClassInfo) -> str:
        """Detect DDD layer for a class."""
        name = class_info.name.lower()
        package = class_info.package.lower()
        annotations = " ".join(class_info.annotations).lower()

        # Infrastructure layer
        if any(x in package for x in ["infrastructure", "adapter", "persistence", "repository"]):
            return "infrastructure"
        if any(x in name for x in ["repository", "adapter", "client", "gateway"]):
            return "infrastructure"
        if "@repository" in annotations:
            return "infrastructure"

        # Application layer
        if any(x in package for x in ["application", "service", "usecase"]):
            return "application"
        if any(x in name for x in ["service", "usecase", "handler", "facade"]):
            return "application"
        if "@service" in annotations:
            return "application"

        # Domain layer
        if any(x in package for x in ["domain", "model", "entity"]):
            return "domain"
        if any(x in name for x in ["entity", "valueobject", "aggregate", "domainservice"]):
            return "domain"
        if "@entity" in annotations:
            return "domain"

        return "unknown"

    def detect_type(self, class_info: ClassInfo) -> str:
        """Detect the semantic type of a class/record/enum/interface."""
        # First check Java type
        java_type = class_info.class_type
        if java_type == "record":
            return "record"
        if java_type == "enum":
            return "enum"
        if java_type == "annotation":
            return "annotation"
        if java_type == "interface":
            return "interface"
        
        # Then check semantic label
        label = self._get_class_type_label(class_info)
        type_mapping = {
            "Service": "service",
            "Repository": "repository",
            "Repository Interface": "repository",
            "Service Interface": "interface",
            "UseCase Interface": "interface",
            "Port Interface": "interface",
            "Gateway Interface": "interface",
            "Entity": "entity",
            "Controller": "controller",
            "Handler": "service",
            "UseCase": "service",
            "Adapter": "adapter",
            "Gateway": "gateway",
            "Interface": "interface",
            "Class": "class",
            "Record": "record",
            "DTO Record": "record",
            "Event Record": "record",
            "Command Record": "record",
            "Enum": "enum",
            "Annotation": "annotation",
        }
        return type_mapping.get(label, "class")

