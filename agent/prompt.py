import os
import re
from typing import Optional, Any
from jinja2 import Environment, FileSystemLoader

from .rules import TestRules, LayerRules
from .memory import SessionMemory
from rag.schema import CodeChunk


class TemplateEngine:
    """Helper for rendering Jinja2 templates."""

    def __init__(self):
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        self.env = Environment(loader=FileSystemLoader(template_dir))

    def render(self, template_name: str, **kwargs: Any) -> str:
        template = self.env.get_template(template_name)
        return template.render(**kwargs)


class PromptBuilder:
    """Builds prompts for test generation."""

    def __init__(self):
        self.test_rules = TestRules()
        self.layer_rules = LayerRules()
        self.engine = TemplateEngine()

    def build_system_prompt(self) -> str:
        """Build the system prompt for test generation."""
        return self.engine.render("system_prompt.jinja2")

    def build_test_generation_prompt(
        self,
        class_name: str,
        file_path: str,
        rag_chunks: list[CodeChunk],
        task_description: Optional[str] = None,
        session: Optional[SessionMemory] = None,
    ) -> str:
        """Build the user prompt for test generation."""
        # Extract source code from task_description if present (Continue IDE inline)
        source_code = None
        task_text = task_description or ""
        if task_description:
            code_match = re.search(r'```(?:[^\n]*\.java)?\s*\n(.*?)```', task_description, re.DOTALL)
            if code_match:
                source_code = code_match.group(1).strip()
                task_text = re.sub(r'```[^`]*```', '', task_description, flags=re.DOTALL).strip()
                if not task_text:
                    task_text = f"Generate unit tests for `{class_name}`"

        # Detect layer and focus
        layer = self.layer_rules.detect_layer(class_name)
        focus = self.layer_rules.get_test_focus(layer) if layer != "unknown" else None

        # Process RAG chunks
        main_class, domain_types, dependencies, related, unfound = self._process_rag_chunks(class_name, rag_chunks)

        # Session context
        session_context = session.get_conversation_context(max_turns=3) if session else None
        reference_test_class = None
        if session and session.generated_tests:
            recent_test = session.generated_tests[-1]
            if recent_test.success and recent_test.class_name != class_name:
                reference_test_class = recent_test.class_name

        return self.engine.render(
            "generation_prompt.jinja2",
            task_text=task_text,
            source_code=source_code,
            class_name=class_name,
            file_path=file_path,
            layer=layer,
            focus=focus,
            rag_chunks=bool(rag_chunks),
            main_class=main_class,
            main_class_lombok=self._get_lombok_info(main_class) if main_class else None,
            domain_types=domain_types,
            unfound=unfound,
            dependencies=dependencies,
            related=related,
            session_context=session_context,
            reference_test=bool(reference_test_class),
            reference_test_class=reference_test_class,
        )

    def _process_rag_chunks(
        self, 
        class_name: str, 
        rag_chunks: list[CodeChunk]
    ) -> tuple[Optional[CodeChunk], list[dict], list[dict], list[dict], list[str]]:
        """Helper to classify RAG chunks for the template."""
        if not rag_chunks:
            return None, [], [], [], []

        main_class: Optional[CodeChunk] = None
        used_type_names: set[str] = set()
        dep_names: set[str] = set()
        
        for chunk in rag_chunks:
            if chunk.class_name == class_name:
                main_class = chunk
                used_type_names = set(chunk.used_types)
                dep_names = set(chunk.dependencies)
                break

        _SERVICE_SUFFIXES = ("Service", "Repository", "Client", "Gateway", "Handler")
        _DOMAIN_TYPES = {"entity", "dto", "model", "record", "domain", "vo", "request", "response"}

        domain_types_list: list[dict[str, Any]] = []
        dependencies: list[dict[str, Any]] = []
        related: list[dict[str, Any]] = []

        for chunk in rag_chunks:
            if chunk.class_name == class_name:
                continue

            # Classify
            is_domain = False
            c_name = str(chunk.class_name)
            if c_name in used_type_names:
                is_domain = True
            elif (chunk.java_type in ("record", "class", "enum")) and (
                (chunk.type or "").lower() in _DOMAIN_TYPES
                or (chunk.layer or "").lower() in ("domain", "model", "dto")
                or not any(c_name.endswith(s) for s in _SERVICE_SUFFIXES)
            ):
                is_domain = True

            if is_domain:
                java_type = str(chunk.java_type or chunk.type)
                construction = self._get_construction_hint(chunk)
                
                # Format for template
                dt_info: dict[str, Any] = {
                    "class_name": c_name,
                    "type_label": f"{java_type}, {self._get_lombok_info(chunk)}" if self._get_lombok_info(chunk) else java_type,
                    "construction": bool(construction),
                    "construction_comments": "\n".join(f"// {line}" for line in construction.split("\n")) if construction else "",
                    "summary": chunk.summary
                }
                
                if (java_type == "record") and getattr(chunk, "record_components", None):
                    dt_info["comp_str"] = ", ".join(f"{rc.type} {rc.name}" for rc in chunk.record_components)
                elif getattr(chunk, "fields", None):
                    dt_info["field_str"] = ", ".join(f"{f.type} {f.name}" for f in chunk.fields)
                
                domain_types_list.append(dt_info)
            elif c_name in dep_names or any(c_name.endswith(s) for s in _SERVICE_SUFFIXES):
                dependencies.append({
                    "class_name": c_name,
                    "type_info": f"{chunk.type}, {self._get_lombok_info(chunk)}" if self._get_lombok_info(chunk) else chunk.type,
                    "summary_truncated": chunk.summary[:300] + ("..." if len(chunk.summary) > 300 else "")
                })
            else:
                related.append({
                    "class_name": c_name,
                    "type_info": f"{getattr(chunk, 'java_type', None) or chunk.type}, {chunk.layer}",
                    "summary_truncated": chunk.summary[:200] + ("..." if len(chunk.summary) > 200 else "")
                })

        unfound = list(getattr(main_class, "unfound_types", [])) if main_class else []
        
        # Slicing with explicit list conversion
        final_deps = list(dependencies[:5])
        final_related = list(related[:3])
        return main_class, domain_types_list, final_deps, final_related, unfound

    def build_refinement_prompt(
        self,
        original_code: str,
        feedback: str,
        validation_issues: list[str],
        rag_chunks: Optional[list[CodeChunk]] = None,
    ) -> str:
        """Build prompt for refining generated test code."""
        context_summary = self.build_context_summary(rag_chunks) if rag_chunks else None
        return self.engine.render(
            "refinement_prompt.jinja2",
            original_code=original_code,
            context_summary=context_summary,
            feedback=feedback,
            validation_issues=validation_issues,
        )

    def build_incremental_update_prompt(
        self,
        class_name: str,
        file_path: str,
        rag_chunks: list[CodeChunk],
        existing_test_code: str,
        tested_methods: Optional[list[str]] = None,
        changed_methods: Optional[list[str]] = None,
        task_description: Optional[str] = None,
    ) -> str:
        """Build prompt for incremental test update."""
        context_summary = self.build_context_summary(rag_chunks) if rag_chunks else None
        return self.engine.render(
            "incremental_prompt.jinja2",
            class_name=class_name,
            changed_methods=bool(changed_methods),
            changed_methods_str=", ".join(f"`{m}`" for m in changed_methods) if changed_methods else "",
            task_description=task_description,
            file_path=file_path,
            existing_test_code=existing_test_code,
            tested_methods_str=", ".join(f"`{m}`" for m in tested_methods) if tested_methods else "",
            context_summary=context_summary,
        )

    def build_context_summary(self, chunks: list[CodeChunk], max_tokens: int = 2000) -> str:
        """Build a compact context summary from RAG chunks."""
        if not chunks:
            return ""

        summaries = []
        current_tokens = 0
        approx_chars_per_token = 4

        for chunk in chunks:
            java_type = getattr(chunk, 'java_type', None) or chunk.type
            lombok_info = self._get_lombok_info(chunk)
            type_label = f"{java_type} {lombok_info}".strip()
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
            lombok_parts.append("@Builder(toBuilder=true)" if getattr(chunk, 'has_builder_to_builder', False) else "@Builder")
        if getattr(chunk, 'has_data', False):
            lombok_parts.append("@Data")
        elif getattr(chunk, 'has_value', False):
            lombok_parts.append("@Value")
        else:
            if getattr(chunk, 'has_getter', False): lombok_parts.append("@Getter")
            if getattr(chunk, 'has_setter', False): lombok_parts.append("@Setter")
        return " ".join(lombok_parts)

    def _get_construction_hint(self, chunk: CodeChunk) -> str:
        """Generate explicit construction instruction for a domain type."""
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

        if java_type == "record":
            if has_builder:
                hint = f"✅ HAS @Builder → MUST use: `{name}.builder().field(value).build()`"
                if has_builder_to_builder: hint += f"\n   Also supports: `obj.toBuilder().field(newValue).build()`"
                return hint
            components = getattr(chunk, 'record_components', None)
            args = ", ".join(rc.name for rc in components) if components else "..."
            return f"⚠️ NO @Builder → MUST use canonical constructor: `new {name}({args})`\n   NEVER use {name}.builder(). It does NOT exist."

        if java_type == "enum": return f"Enum — use `{name}.VALUE` constants directly."
        if java_type == "interface": return "Interface — mock or use implementing class."

        if has_builder:
            hint = f"✅ HAS @Builder → MUST use: `{name}.builder().field(value).build()`"
            if has_builder_to_builder: hint += f"\n   Also supports: `obj.toBuilder().field(newValue).build()`"
            return hint
        if has_value:
            fields = getattr(chunk, 'fields', None)
            args = ", ".join(f.name for f in fields) if fields else "..."
            return f"@Value (immutable) → use: `new {name}({args})`"
        if has_data or (has_getter and has_setter):
            if has_all_args: return f"@Data + @AllArgsConstructor → use: `new {name}(all, args, in, order)` OR `new {name}()` + setters"
            if has_no_args: return f"@Data → use: `new {name}()` then call `.setField(value)` for each field"
            return f"@Data → use: `new {name}()` + setters, or all-args constructor if available"
        if has_all_args: return f"Use: `new {name}(all, args, ...)`" if not has_no_args else f"Use: `new {name}(all, args, ...)` OR `new {name}()` + setters"
        if has_no_args: return f"Use: `new {name}()` then set fields via setters"
        return ""

    def estimate_tokens(self, text: str) -> int:
        from utils.tokenizer import count_tokens
        return count_tokens(text)

    def build_registry_enhanced_prompt(
        self,
        class_name: str,
        file_path: str,
        rag_chunks: list[CodeChunk],
        registry_context: str,
        task_description: Optional[str] = None,
        session: Optional[SessionMemory] = None,
    ) -> str:
        """Enhanced version of build_test_generation_prompt with registry context."""
        base_prompt = self.build_test_generation_prompt(class_name, file_path, rag_chunks, task_description, session)
        if not registry_context: return base_prompt
        parts = base_prompt.split("\n## Requirements")
        return f"{parts[0]}\n\n{registry_context}\n\n## Requirements{parts[1]}" if len(parts) == 2 else f"{base_prompt}\n\n{registry_context}"

    def build_focused_method_prompt(
        self,
        class_name: str,
        method_name: str,
        method_signature: str,
        test_scenarios: list[dict],
        mock_fields: list[str],
        registry_context: str,
        dependencies_called: Optional[list[str]] = None,
    ) -> str:
        """Focused prompt for generating tests for a single method."""
        return self.engine.render(
            "focused_method_prompt.jinja2",
            class_name=class_name,
            class_name_lower=class_name[0].lower() + class_name[1:],
            method_name=method_name,
            method_signature=method_signature,
            test_scenarios=test_scenarios,
            mock_fields=mock_fields,
            registry_context=registry_context,
            dependencies_called=bool(dependencies_called),
            dependencies_called_str=", ".join(dependencies_called) if dependencies_called else "",
        )
