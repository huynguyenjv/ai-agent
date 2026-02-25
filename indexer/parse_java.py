"""
Java source code parser using tree-sitter.
Extracts services, methods, entities, repository interfaces,
referenced types, and dependency relationships.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

import structlog

logger = structlog.get_logger()


@dataclass
class MethodInfo:
    """Extracted method information."""
    name: str
    return_type: str
    parameters: list[tuple[str, str]]  # (type, name)
    body: str
    annotations: list[str]
    javadoc: Optional[str] = None
    start_line: int = 0
    end_line: int = 0


@dataclass
class RecordComponent:
    """Record component (field in record declaration)."""
    name: str
    type: str
    annotations: list[str] = field(default_factory=list)


@dataclass
class FieldInfo:
    """Detailed field information."""
    name: str
    type: str
    annotations: list[str] = field(default_factory=list)
    modifiers: list[str] = field(default_factory=list)
    default_value: Optional[str] = None
    javadoc: Optional[str] = None


@dataclass
class TypeReference:
    """A reference to another class/type used within a class."""
    type_name: str          # Simple name, e.g. "OrderRequest"
    context: str            # "field", "parameter", "return_type", "extends", "implements"
    field_or_method: str    # The field name or method name where it appears
    is_generic_arg: bool = False   # True if inside List<X>, Optional<X>, etc.


@dataclass
class ClassInfo:
    """Extracted class/interface/record/enum information."""
    name: str
    package: str
    file_path: str
    class_type: str  # class, interface, enum, record, annotation
    annotations: list[str]
    implements: list[str]
    extends: Optional[str]
    fields: list[tuple[str, str, list[str]]]  # legacy: (type, name, annotations)
    methods: list[MethodInfo]
    imports: list[str]
    javadoc: Optional[str] = None
    inner_classes: list["ClassInfo"] = field(default_factory=list)
    record_components: list[RecordComponent] = field(default_factory=list)
    enum_constants: list[str] = field(default_factory=list)
    modifiers: list[str] = field(default_factory=list)
    detailed_fields: list[FieldInfo] = field(default_factory=list)

    # --- NEW: type reference tracking ---
    referenced_types: list[TypeReference] = field(default_factory=list)
    # Simple set of unique referenced class names (for quick lookup)
    referenced_class_names: list[str] = field(default_factory=list)

    # Lombok
    has_builder: bool = False
    has_builder_to_builder: bool = False
    has_data: bool = False
    has_getter: bool = False
    has_setter: bool = False
    has_no_args_constructor: bool = False
    has_all_args_constructor: bool = False
    has_required_args_constructor: bool = False
    has_value: bool = False
    has_equals_and_hash_code: bool = False
    has_to_string: bool = False

    @property
    def fully_qualified_name(self) -> str:
        return f"{self.package}.{self.name}" if self.package else self.name

    @property
    def is_interface(self) -> bool:
        return self.class_type == "interface"

    @property
    def is_record(self) -> bool:
        return self.class_type == "record"

    @property
    def is_enum(self) -> bool:
        return self.class_type == "enum"

    @property
    def is_annotation(self) -> bool:
        return self.class_type in ("annotation", "annotation_type")

    @property
    def is_abstract(self) -> bool:
        return "abstract" in self.modifiers

    @property
    def is_final(self) -> bool:
        return "final" in self.modifiers

    @property
    def is_sealed(self) -> bool:
        return "sealed" in self.modifiers

    @property
    def is_lombok_data_class(self) -> bool:
        return self.has_data or (self.has_getter and self.has_setter)

    @property
    def is_immutable(self) -> bool:
        return self.is_record or self.has_value

    @property
    def lombok_annotations(self) -> list[str]:
        annotations = []
        if self.has_builder:
            annotations.append("@Builder" + ("(toBuilder=true)" if self.has_builder_to_builder else ""))
        if self.has_data:
            annotations.append("@Data")
        if self.has_value:
            annotations.append("@Value")
        if self.has_getter:
            annotations.append("@Getter")
        if self.has_setter:
            annotations.append("@Setter")
        if self.has_no_args_constructor:
            annotations.append("@NoArgsConstructor")
        if self.has_all_args_constructor:
            annotations.append("@AllArgsConstructor")
        if self.has_required_args_constructor:
            annotations.append("@RequiredArgsConstructor")
        if self.has_equals_and_hash_code:
            annotations.append("@EqualsAndHashCode")
        if self.has_to_string:
            annotations.append("@ToString")
        return annotations


# ---------------------------------------------------------------------------
# Primitives & well-known JDK types to exclude from referenced_types
# ---------------------------------------------------------------------------
_EXCLUDED_TYPES: set[str] = {
    # primitives
    "void", "int", "long", "double", "float", "boolean", "byte", "short", "char",
    # java.lang
    "String", "Integer", "Long", "Double", "Float", "Boolean", "Byte", "Short",
    "Character", "Number", "Object", "Class", "Enum", "Record",
    # java.util
    "List", "Map", "Set", "Collection", "Optional", "ArrayList", "HashMap",
    "HashSet", "LinkedList", "LinkedHashMap", "TreeMap", "TreeSet",
    "Iterator", "Iterable", "Comparator",
    # java.util.function
    "Function", "BiFunction", "Consumer", "Supplier", "Predicate",
    # Spring / common framework types
    "Page", "Pageable", "Sort", "ResponseEntity", "HttpStatus", "HttpHeaders",
    "MultiValueMap", "ModelAndView",
    # java.time
    "LocalDate", "LocalDateTime", "LocalTime", "ZonedDateTime",
    "OffsetDateTime", "Instant", "Duration", "Period",
    # misc
    "UUID", "BigDecimal", "BigInteger", "URL", "URI",
    "InputStream", "OutputStream", "Reader", "Writer",
    "Exception", "RuntimeException", "Throwable", "Error",
    "Void", "var",
}


class JavaParser:
    """Parse Java source files using tree-sitter."""

    def __init__(self):
        self.language = Language(tsjava.language(), "java")
        self.parser = Parser()
        self.parser.set_language(self.language)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_file(self, file_path: str) -> Optional[ClassInfo]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
            return self.parse_source(source_code, file_path)
        except Exception as e:
            logger.error("Failed to parse file", file_path=file_path, error=str(e))
            return None

    def parse_source(self, source_code: str, file_path: str = "") -> Optional[ClassInfo]:
        tree = self.parser.parse(bytes(source_code, "utf-8"))
        root_node = tree.root_node

        package = self._extract_package(root_node, source_code)
        imports = self._extract_imports(root_node, source_code)

        class_node = self._find_class_declaration(root_node)
        if not class_node:
            return None

        class_info = self._extract_class_info(class_node, source_code, package, imports, file_path)

        # Post-process: extract all type references
        if class_info:
            refs = self._extract_all_type_references(class_info)
            class_info.referenced_types = refs
            class_info.referenced_class_names = list({r.type_name for r in refs})

        return class_info

    def parse_directory(self, directory: str) -> list[ClassInfo]:
        classes = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".java"):
                    file_path = os.path.join(root, file)
                    class_info = self.parse_file(file_path)
                    if class_info:
                        classes.append(class_info)
                        logger.info("Parsed class", class_name=class_info.name, file_path=file_path)
        return classes

    # ------------------------------------------------------------------
    # Type reference extraction (NEW)
    # ------------------------------------------------------------------

    def _extract_all_type_references(self, class_info: ClassInfo) -> list[TypeReference]:
        """
        Walk fields, method signatures (params + return types),
        extends, and implements to collect every user-defined type reference.
        """
        refs: list[TypeReference] = []
        seen: set[tuple[str, str, str]] = set()

        def add(type_name: str, context: str, member: str, is_generic: bool = False):
            # Strip array suffix
            type_name = type_name.replace("[]", "").strip()
            if not type_name or type_name in _EXCLUDED_TYPES:
                return
            if not type_name[0].isupper():
                return
            key = (type_name, context, member)
            if key not in seen:
                seen.add(key)
                refs.append(TypeReference(
                    type_name=type_name,
                    context=context,
                    field_or_method=member,
                    is_generic_arg=is_generic,
                ))

        # extends
        if class_info.extends:
            for t in self._decompose_type(class_info.extends):
                add(t, "extends", class_info.extends, t != class_info.extends)

        # implements
        for iface in class_info.implements:
            for t in self._decompose_type(iface):
                add(t, "implements", iface, t != iface)

        # fields
        for f in class_info.detailed_fields:
            for t in self._decompose_type(f.type):
                add(t, "field", f.name, t != self._strip_generics(f.type))

        # record components
        for comp in class_info.record_components:
            for t in self._decompose_type(comp.type):
                add(t, "field", comp.name, t != self._strip_generics(comp.type))

        # methods: return type + parameters
        for m in class_info.methods:
            for t in self._decompose_type(m.return_type):
                add(t, "return_type", m.name, t != self._strip_generics(m.return_type))
            for param_type, param_name in m.parameters:
                for t in self._decompose_type(param_type):
                    add(t, "parameter", f"{m.name}.{param_name}", t != self._strip_generics(param_type))

        return refs

    def _decompose_type(self, type_str: str) -> list[str]:
        """
        Extract all concrete type names from a possibly generic type string.
        Examples:
            "OrderRequest"           -> ["OrderRequest"]
            "List<OrderRequest>"     -> ["List", "OrderRequest"]   (List filtered out later)
            "Map<String, OrderDto>"  -> ["Map", "String", "OrderDto"]
            "ResponseEntity<Page<OrderResponse>>" -> [...]
        """
        if not type_str:
            return []
        # Remove array markers and whitespace
        type_str = type_str.replace("[]", "").strip()
        # Find all identifiers (sequences of word chars starting with uppercase or lowercase)
        tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*', type_str)
        return tokens

    def _strip_generics(self, type_str: str) -> str:
        """Return outermost type without generic args: List<X> -> List"""
        if not type_str:
            return type_str
        idx = type_str.find('<')
        return type_str[:idx].strip() if idx != -1 else type_str.strip()

    # ------------------------------------------------------------------
    # Existing extraction helpers (unchanged except minor additions)
    # ------------------------------------------------------------------

    def _extract_package(self, root_node, source_code: str) -> str:
        for child in root_node.children:
            if child.type == "package_declaration":
                for node in child.children:
                    if node.type in ("scoped_identifier", "identifier"):
                        return self._get_node_text(node, source_code)
        return ""

    def _extract_imports(self, root_node, source_code: str) -> list[str]:
        imports = []
        for child in root_node.children:
            if child.type == "import_declaration":
                import_text = self._get_node_text(child, source_code)
                import_text = import_text.replace("import ", "").replace(";", "").strip()
                imports.append(import_text)
        return imports

    def _find_class_declaration(self, root_node):
        declaration_types = (
            "class_declaration",
            "record_declaration",
            "interface_declaration",
            "enum_declaration",
            "annotation_type_declaration",
        )
        for child in root_node.children:
            if child.type in declaration_types:
                return child
        return None

    def _extract_class_info(self, class_node, source_code, package, imports, file_path) -> ClassInfo:
        node_type = class_node.type
        class_type = self._normalize_class_type(node_type)

        name = ""
        annotations = []
        modifiers = []
        implements = []
        extends = None
        javadoc = self._get_preceding_javadoc(class_node, source_code)
        record_components = []
        enum_constants = []

        for child in class_node.children:
            if child.type == "identifier":
                name = self._get_node_text(child, source_code)
            elif child.type == "modifiers":
                annotations = self._extract_annotations(child, source_code)
                modifiers = self._extract_modifiers(child, source_code)
            elif child.type == "superclass":
                extends = self._extract_superclass(child, source_code)
            elif child.type in ("super_interfaces", "extends_interfaces"):
                implements = self._extract_interfaces(child, source_code)
            elif child.type == "formal_parameters":
                record_components = self._extract_record_components(child, source_code)

        fields = []
        detailed_fields = []
        methods = []
        inner_classes = []

        body_node = (
            self._find_child_by_type(class_node, "class_body")
            or self._find_child_by_type(class_node, "interface_body")
            or self._find_child_by_type(class_node, "enum_body")
            or self._find_child_by_type(class_node, "record_body")
            or self._find_child_by_type(class_node, "annotation_type_body")
        )

        if body_node:
            for child in body_node.children:
                if child.type == "field_declaration":
                    f = self._extract_field(child, source_code)
                    if f:
                        fields.append(f)
                    df = self._extract_detailed_field(child, source_code)
                    if df:
                        detailed_fields.append(df)
                elif child.type == "method_declaration":
                    m = self._extract_method(child, source_code)
                    if m:
                        methods.append(m)
                elif child.type == "compact_constructor_declaration":
                    m = self._extract_compact_constructor(child, source_code, name)
                    if m:
                        methods.append(m)
                elif child.type == "enum_constant":
                    c = self._extract_enum_constant(child, source_code)
                    if c:
                        enum_constants.append(c)
                elif child.type in ("class_declaration", "interface_declaration",
                                    "record_declaration", "enum_declaration"):
                    ic = self._extract_class_info(child, source_code, package, [], file_path)
                    if ic:
                        inner_classes.append(ic)

        lombok_info = self._detect_lombok_annotations(annotations)

        return ClassInfo(
            name=name,
            package=package,
            file_path=file_path,
            class_type=class_type,
            annotations=annotations,
            modifiers=modifiers,
            implements=implements,
            extends=extends,
            fields=fields,
            detailed_fields=detailed_fields,
            methods=methods,
            imports=imports,
            javadoc=javadoc,
            inner_classes=inner_classes,
            record_components=record_components,
            enum_constants=enum_constants,
            **lombok_info,
        )

    def _normalize_class_type(self, node_type: str) -> str:
        return {
            "class_declaration": "class",
            "interface_declaration": "interface",
            "enum_declaration": "enum",
            "record_declaration": "record",
            "annotation_type_declaration": "annotation",
        }.get(node_type, node_type.replace("_declaration", ""))

    def _extract_modifiers(self, modifiers_node, source_code: str) -> list[str]:
        modifier_types = {
            "public", "private", "protected", "static", "final",
            "abstract", "synchronized", "native", "transient", "volatile",
            "sealed", "non-sealed", "strictfp",
        }
        return [
            self._get_node_text(child, source_code)
            for child in modifiers_node.children
            if self._get_node_text(child, source_code) in modifier_types
        ]

    def _extract_record_components(self, params_node, source_code: str) -> list[RecordComponent]:
        components = []
        for child in params_node.children:
            if child.type == "formal_parameter":
                comp_type = comp_name = ""
                comp_annotations = []
                for pc in child.children:
                    if pc.type == "modifiers":
                        comp_annotations = self._extract_annotations(pc, source_code)
                    elif pc.type in ("type_identifier", "generic_type", "array_type",
                                     "integral_type", "floating_point_type", "boolean_type"):
                        comp_type = self._get_node_text(pc, source_code)
                    elif pc.type == "identifier":
                        comp_name = self._get_node_text(pc, source_code)
                if comp_type and comp_name:
                    components.append(RecordComponent(name=comp_name, type=comp_type, annotations=comp_annotations))
        return components

    def _extract_compact_constructor(self, node, source_code, class_name) -> Optional[MethodInfo]:
        annotations = []
        body = ""
        for child in node.children:
            if child.type == "modifiers":
                annotations = self._extract_annotations(child, source_code)
            elif child.type == "block":
                body = self._get_node_text(child, source_code)
        return MethodInfo(
            name=class_name, return_type="", parameters=[], body=body,
            annotations=annotations, start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
        )

    def _extract_enum_constant(self, node, source_code) -> Optional[str]:
        for child in node.children:
            if child.type == "identifier":
                return self._get_node_text(child, source_code)
        return None

    def _extract_annotations(self, modifiers_node, source_code) -> list[str]:
        return [
            self._get_node_text(child, source_code)
            for child in modifiers_node.children
            if child.type in ("marker_annotation", "annotation")
        ]

    def _extract_superclass(self, superclass_node, source_code) -> Optional[str]:
        for child in superclass_node.children:
            if child.type in ("type_identifier", "generic_type"):
                return self._get_node_text(child, source_code)
        return None

    def _extract_interfaces(self, interfaces_node, source_code) -> list[str]:
        interfaces = []
        for child in interfaces_node.children:
            if child.type in ("type_identifier", "generic_type"):
                interfaces.append(self._get_node_text(child, source_code))
            elif child.type == "type_list":
                for tc in child.children:
                    if tc.type in ("type_identifier", "generic_type"):
                        interfaces.append(self._get_node_text(tc, source_code))
        return interfaces

    def _extract_field(self, field_node, source_code) -> Optional[tuple]:
        field_type = field_name = ""
        annotations = []
        for child in field_node.children:
            if child.type == "modifiers":
                annotations = self._extract_annotations(child, source_code)
            elif child.type in ("type_identifier", "generic_type", "array_type",
                                 "integral_type", "floating_point_type", "boolean_type"):
                field_type = self._get_node_text(child, source_code)
            elif child.type == "variable_declarator":
                for vc in child.children:
                    if vc.type == "identifier":
                        field_name = self._get_node_text(vc, source_code)
                        break
        if field_type and field_name:
            return (field_type, field_name, annotations)
        return None

    def _extract_detailed_field(self, field_node, source_code) -> Optional[FieldInfo]:
        field_type = field_name = ""
        annotations = []
        modifiers = []
        default_value = None
        javadoc = self._get_preceding_javadoc(field_node, source_code)

        for child in field_node.children:
            if child.type == "modifiers":
                annotations = self._extract_annotations(child, source_code)
                modifiers = self._extract_modifiers(child, source_code)
            elif child.type in ("type_identifier", "generic_type", "array_type",
                                 "integral_type", "floating_point_type", "boolean_type",
                                 "scoped_type_identifier"):
                field_type = self._get_node_text(child, source_code)
            elif child.type == "variable_declarator":
                for vc in child.children:
                    if vc.type == "identifier":
                        field_name = self._get_node_text(vc, source_code)
                    elif vc.type not in ("identifier", "dimensions"):
                        default_value = self._get_node_text(vc, source_code)
        if field_type and field_name:
            return FieldInfo(
                name=field_name, type=field_type, annotations=annotations,
                modifiers=modifiers, default_value=default_value, javadoc=javadoc,
            )
        return None

    def _detect_lombok_annotations(self, annotations: list[str]) -> dict:
        result = {
            "has_builder": False, "has_builder_to_builder": False,
            "has_data": False, "has_getter": False, "has_setter": False,
            "has_no_args_constructor": False, "has_all_args_constructor": False,
            "has_required_args_constructor": False, "has_value": False,
            "has_equals_and_hash_code": False, "has_to_string": False,
        }
        for ann in annotations:
            al = ann.lower()
            if "@builder" in al:
                result["has_builder"] = True
                if "tobuilder" in al and "true" in al:
                    result["has_builder_to_builder"] = True
            if "@data" in al:
                result["has_data"] = True
                result["has_getter"] = result["has_setter"] = True
                result["has_to_string"] = result["has_equals_and_hash_code"] = True
            if al == "@value" or al.startswith("@value("):
                result["has_value"] = True
                result["has_getter"] = result["has_to_string"] = True
                result["has_equals_and_hash_code"] = True
            if "@getter" in al:
                result["has_getter"] = True
            if "@setter" in al:
                result["has_setter"] = True
            if "@noargsconstructor" in al:
                result["has_no_args_constructor"] = True
            if "@allargsconstructor" in al:
                result["has_all_args_constructor"] = True
            if "@requiredargsconstructor" in al:
                result["has_required_args_constructor"] = True
            if "@equalsandhashcode" in al:
                result["has_equals_and_hash_code"] = True
            if "@tostring" in al:
                result["has_to_string"] = True
        return result

    def _extract_method(self, method_node, source_code) -> Optional[MethodInfo]:
        name = ""
        return_type = "void"
        parameters = []
        annotations = []
        body = ""
        javadoc = self._get_preceding_javadoc(method_node, source_code)

        for child in method_node.children:
            if child.type == "modifiers":
                annotations = self._extract_annotations(child, source_code)
            elif child.type in ("type_identifier", "generic_type", "array_type",
                                 "integral_type", "floating_point_type", "boolean_type", "void_type"):
                return_type = self._get_node_text(child, source_code)
            elif child.type == "identifier":
                name = self._get_node_text(child, source_code)
            elif child.type == "formal_parameters":
                parameters = self._extract_parameters(child, source_code)
            elif child.type == "block":
                body = self._get_node_text(child, source_code)

        if name:
            return MethodInfo(
                name=name, return_type=return_type, parameters=parameters,
                body=body, annotations=annotations, javadoc=javadoc,
                start_line=method_node.start_point[0] + 1,
                end_line=method_node.end_point[0] + 1,
            )
        return None

    def _extract_parameters(self, params_node, source_code) -> list[tuple[str, str]]:
        parameters = []
        for child in params_node.children:
            if child.type in ("formal_parameter", "spread_parameter"):
                param_type = param_name = ""
                for pc in child.children:
                    if pc.type in ("type_identifier", "generic_type", "array_type",
                                   "integral_type", "floating_point_type", "boolean_type"):
                        param_type = self._get_node_text(pc, source_code)
                    elif pc.type == "identifier":
                        param_name = self._get_node_text(pc, source_code)
                if param_type and param_name:
                    parameters.append((param_type, param_name))
        return parameters

    def _get_preceding_javadoc(self, node, source_code: str) -> Optional[str]:
        start_byte = node.start_byte
        preceding_text = source_code[:start_byte].strip()
        if preceding_text.endswith("*/"):
            javadoc_start = preceding_text.rfind("/**")
            if javadoc_start != -1:
                return preceding_text[javadoc_start:]
        return None

    def _find_child_by_type(self, node, type_name: str):
        for child in node.children:
            if child.type == type_name:
                return child
        return None

    def _get_node_text(self, node, source_code: str) -> str:
        return source_code[node.start_byte:node.end_byte]