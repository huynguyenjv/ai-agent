"""
Code summarizer for generating semantic summaries of Java code.
Produces compact summaries (150-250 tokens) for RAG indexing.
Includes usage hint generation so LLMs know HOW to instantiate each model.
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

    # ------------------------------------------------------------------
    # Public summarizers
    # ------------------------------------------------------------------

    def summarize_class(self, class_info: ClassInfo) -> str:
        """Generate a semantic summary for a class/record/enum/interface."""
        parts = []

        class_type_label = self._get_class_type_label(class_info)
        parts.append(f"{class_type_label}: {class_info.fully_qualified_name}")
        parts.append(f"Type: {class_info.class_type}")

        if class_info.modifiers:
            parts.append(f"Modifiers: {', '.join(class_info.modifiers)}")
        if class_info.annotations:
            parts.append(f"Annotations: {', '.join(class_info.annotations)}")
        if class_info.lombok_annotations:
            parts.append(f"Lombok: {', '.join(class_info.lombok_annotations)}")
        if class_info.extends:
            parts.append(f"Extends: {class_info.extends}")
        if class_info.implements:
            parts.append(f"Implements: {', '.join(class_info.implements)}")

        # Record components
        if class_info.record_components:
            components = [f"{c.type} {c.name}" for c in class_info.record_components]
            parts.append(f"Record Components: {', '.join(components)}")

        # Enum constants
        if class_info.enum_constants:
            constants = class_info.enum_constants[:10]
            if len(class_info.enum_constants) > 10:
                constants.append("...")
            parts.append(f"Constants: {', '.join(constants)}")

        # Javadoc
        if class_info.javadoc:
            doc_summary = self._extract_javadoc_summary(class_info.javadoc)
            if doc_summary:
                parts.append(f"Purpose: {doc_summary}")

        # Usage hint — always include so LLM knows how to instantiate
        hint = self.generate_usage_hint(class_info)
        if hint:
            parts.append(f"Instantiation:\n{hint}")

        # Dependencies for service/component classes
        if class_info.class_type == "class":
            dependencies = self._extract_dependencies(class_info)
            if dependencies:
                parts.append(f"Dependencies: {', '.join(dependencies)}")

        # Fields
        if class_info.detailed_fields:
            field_summaries = []
            for f in class_info.detailed_fields[:8]:
                field_str = f"{f.type} {f.name}"
                important_anns = [
                    a for a in f.annotations
                    if any(x in a.lower() for x in ['@id', '@column', '@notnull', '@nullable', '@valid', '@jsonproperty'])
                ]
                if important_anns:
                    field_str = f"{' '.join(important_anns)} {field_str}"
                field_summaries.append(field_str)
            if len(class_info.detailed_fields) > 8:
                field_summaries.append(f"... +{len(class_info.detailed_fields) - 8} more")
            parts.append(f"Fields: {'; '.join(field_summaries)}")
        elif class_info.class_type != "record" and class_info.fields:
            field_summaries = [f"{t} {n}" for t, n, _ in class_info.fields[:5]]
            if len(class_info.fields) > 5:
                field_summaries.append("...")
            parts.append(f"Fields: {', '.join(field_summaries)}")

        # Methods
        public_methods = [m for m in class_info.methods if self._is_public_method(m)]
        if public_methods:
            sigs = [self._method_signature(m) for m in public_methods[:10]]
            parts.append(f"Methods: {'; '.join(sigs)}")

        # Referenced types (compact — full info goes in index payload)
        if class_info.referenced_class_names:
            parts.append(f"Uses: {', '.join(class_info.referenced_class_names[:10])}")

        summary = "\n".join(parts)
        return self._truncate_to_tokens(summary, self.max_tokens)

    def summarize_method(self, method: MethodInfo, class_info: ClassInfo) -> str:
        """Generate a semantic summary for a method."""
        parts = []
        parts.append(f"Method: {class_info.fully_qualified_name}.{method.name}")
        parts.append(f"Signature: {self._method_signature(method)}")
        if method.annotations:
            parts.append(f"Annotations: {', '.join(method.annotations)}")
        if method.javadoc:
            doc_summary = self._extract_javadoc_summary(method.javadoc)
            if doc_summary:
                parts.append(f"Purpose: {doc_summary}")
        if method.body:
            body_summary = self._summarize_method_body(method.body)
            if body_summary:
                parts.append(f"Operations: {body_summary}")
        return self._truncate_to_tokens("\n".join(parts), self.max_tokens)

    # ------------------------------------------------------------------
    # Usage hint generation (NEW — core feature)
    # ------------------------------------------------------------------

    def generate_usage_hint(self, class_info: ClassInfo) -> str:
        """
        Generate the correct instantiation pattern for a class.

        Priority:
          1. @Builder / @Builder(toBuilder=true)
          2. record (canonical constructor)
          3. @Value (immutable, all-args constructor)
          4. @AllArgsConstructor
          5. @Data / @Setter (no-args + setters)
          6. @NoArgsConstructor only
          7. Plain class fallback
        """
        name = class_info.name

        # ---- Record cases ----
        if class_info.is_record:
            if class_info.has_builder:
                return self._builder_hint(name, [c.name for c in class_info.record_components])
            # Canonical constructor
            args = ", ".join(c.name for c in class_info.record_components)
            return f"new {name}({args})"

        # ---- @Builder on regular class ----
        if class_info.has_builder:
            instance_fields = [
                f.name for f in class_info.detailed_fields if "static" not in f.modifiers
            ]
            hint = self._builder_hint(name, instance_fields)
            if class_info.has_builder_to_builder:
                hint += f"\n// toBuilder: existing.toBuilder().field(newValue).build()"
            return hint

        # ---- @Value (Lombok immutable) ----
        if class_info.has_value:
            fields = [f.name for f in class_info.detailed_fields if "static" not in f.modifiers]
            args = ", ".join(fields)
            return f"new {name}({args})  // @Value: all-args, no setters"

        # ---- @AllArgsConstructor ----
        if class_info.has_all_args_constructor:
            fields = [f.name for f in class_info.detailed_fields if "static" not in f.modifiers]
            args = ", ".join(fields)
            return f"new {name}({args})"

        # ---- @Data or @Setter (mutable bean) ----
        if class_info.has_data or class_info.has_setter:
            fields = [
                f.name for f in class_info.detailed_fields
                if "static" not in f.modifiers
            ][:6]
            setter_lines = "\n".join(
                f"obj.set{f[0].upper()}{f[1:]}(value);" for f in fields
            )
            extra = ""
            if len([f for f in class_info.detailed_fields if "static" not in f.modifiers]) > 6:
                extra = "\n// ... more setters"
            return f"{name} obj = new {name}();\n{setter_lines}{extra}"

        # ---- @NoArgsConstructor only ----
        if class_info.has_no_args_constructor:
            return f"new {name}()  // no-args constructor only"

        # ---- @RequiredArgsConstructor (final fields) ----
        if class_info.has_required_args_constructor:
            final_fields = [
                f.name for f in class_info.detailed_fields
                if "final" in f.modifiers and "static" not in f.modifiers
            ]
            args = ", ".join(final_fields) if final_fields else "/* final fields */"
            return f"new {name}({args})  // @RequiredArgsConstructor"

        # ---- Spring @Component/@Service/@Repository — injected, not new'd ----
        annotations_lower = " ".join(class_info.annotations).lower()
        if any(x in annotations_lower for x in ("@service", "@component", "@repository", "@controller", "@restcontroller")):
            return f"// Spring-managed bean — inject via @Autowired or constructor injection"

        # ---- Interface / abstract class ----
        if class_info.is_interface or class_info.is_abstract:
            return f"// Interface/Abstract — use an implementing class"

        # ---- Enum ----
        if class_info.is_enum:
            if class_info.enum_constants:
                return f"{name}.{class_info.enum_constants[0]}  // or any of: {', '.join(class_info.enum_constants[:5])}"
            return f"{name}.CONSTANT_NAME"

        return f"new {name}()  // check for available constructors"

    def _builder_hint(self, class_name: str, field_names: list[str]) -> str:
        """Format a builder chain hint."""
        if not field_names:
            return f"{class_name}.builder().build()"
        shown = field_names[:6]
        lines = "\n  ".join(f".{f}(value)" for f in shown)
        extra = f"\n  // ... +{len(field_names) - 6} more fields" if len(field_names) > 6 else ""
        return f"{class_name}.builder()\n  {lines}{extra}\n  .build()"

    # ------------------------------------------------------------------
    # Layer / type detection
    # ------------------------------------------------------------------

    def detect_layer(self, class_info: ClassInfo) -> str:
        name = class_info.name.lower()
        package = class_info.package.lower()
        annotations = " ".join(class_info.annotations).lower()

        if any(x in package for x in ["infrastructure", "adapter", "persistence", "repository"]):
            return "infrastructure"
        if any(x in name for x in ["repository", "adapter", "client", "gateway"]):
            return "infrastructure"
        if "@repository" in annotations:
            return "infrastructure"

        if any(x in package for x in ["application", "service", "usecase"]):
            return "application"
        if any(x in name for x in ["service", "usecase", "handler", "facade"]):
            return "application"
        if "@service" in annotations:
            return "application"

        if any(x in package for x in ["domain", "model", "entity"]):
            return "domain"
        if any(x in name for x in ["entity", "valueobject", "aggregate", "domainservice"]):
            return "domain"
        if "@entity" in annotations:
            return "domain"

        # DTO / model layer
        if any(x in name for x in ["request", "response", "dto", "command", "event", "query", "payload"]):
            return "dto"

        return "unknown"

    def detect_type(self, class_info: ClassInfo) -> str:
        java_type = class_info.class_type
        if java_type == "record":
            return "record"
        if java_type == "enum":
            return "enum"
        if java_type == "annotation":
            return "annotation"
        if java_type == "interface":
            return "interface"

        label = self._get_class_type_label(class_info)
        return {
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
            "Mapper": "mapper",
            "Factory": "factory",
        }.get(label, "class")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_class_type_label(self, class_info: ClassInfo) -> str:
        name = class_info.name
        annotations = " ".join(class_info.annotations).lower()
        class_type = class_info.class_type

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

        return "Class"

    def _extract_javadoc_summary(self, javadoc: str) -> Optional[str]:
        if not javadoc:
            return None
        lines = javadoc.replace("/**", "").replace("*/", "").split("\n")
        summary_lines = []
        for line in lines:
            line = line.strip().lstrip("*").strip()
            if line.startswith("@"):
                break
            if line:
                summary_lines.append(line)
        summary = " ".join(summary_lines)
        if "." in summary:
            summary = summary.split(".")[0] + "."
        return summary[:200] if summary else None

    def _extract_dependencies(self, class_info: ClassInfo) -> list[str]:
        dependencies = []
        injection_annotations = {"@Autowired", "@Inject", "@Resource"}
        for field_type, field_name, annotations in class_info.fields:
            has_injection = any(
                ann.startswith(inj) for ann in annotations for inj in injection_annotations
            )
            if has_injection or any(
                field_type.endswith(s)
                for s in ["Service", "Repository", "Client", "Gateway", "Handler"]
            ):
                dependencies.append(field_type)
        return list(set(dependencies))

    def _is_public_method(self, method: MethodInfo) -> bool:
        for ann in method.annotations:
            if "private" in ann.lower() or "protected" in ann.lower():
                return False
        return True

    def _method_signature(self, method: MethodInfo) -> str:
        params = ", ".join(f"{t} {n}" for t, n in method.parameters)
        return f"{method.return_type} {method.name}({params})"

    def _summarize_method_body(self, body: str) -> Optional[str]:
        if not body:
            return None
        operations = []
        bl = body.lower()
        if "repository." in bl or "dao." in bl:
            operations.append("database access")
        if ".save(" in bl or ".persist(" in bl:
            operations.append("persist entity")
        if ".find" in bl or ".get" in bl:
            operations.append("query data")
        if ".delete(" in bl or ".remove(" in bl:
            operations.append("delete operation")
        if "throw new" in bl:
            operations.append("throws exception")
        if ".map(" in bl or ".stream(" in bl:
            operations.append("stream processing")
        if "validate" in bl:
            operations.append("validation")
        if ".publish(" in bl or ".send(" in bl:
            operations.append("event publishing")
        if "log." in bl or "logger." in bl:
            operations.append("logging")
        return ", ".join(operations) if operations else None

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit(" ", 1)[0] + "..."