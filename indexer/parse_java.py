"""
Java source code parser using tree-sitter.
Extracts services, methods, entities, and repository interfaces.
"""

import os
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
    modifiers: list[str] = field(default_factory=list)  # private, final, static, etc.
    default_value: Optional[str] = None
    javadoc: Optional[str] = None


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
    fields: list[tuple[str, str, list[str]]]  # (type, name, annotations) - legacy format
    methods: list[MethodInfo]
    imports: list[str]
    javadoc: Optional[str] = None
    inner_classes: list["ClassInfo"] = field(default_factory=list)
    # Record-specific
    record_components: list[RecordComponent] = field(default_factory=list)
    # Enum-specific
    enum_constants: list[str] = field(default_factory=list)
    # Modifiers
    modifiers: list[str] = field(default_factory=list)  # public, abstract, final, sealed, etc.
    # Detailed fields (new format)
    detailed_fields: list[FieldInfo] = field(default_factory=list)
    # Lombok annotations detection
    has_builder: bool = False
    has_builder_to_builder: bool = False
    has_data: bool = False
    has_getter: bool = False
    has_setter: bool = False
    has_no_args_constructor: bool = False
    has_all_args_constructor: bool = False
    has_required_args_constructor: bool = False
    has_value: bool = False  # @Value (immutable)
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
        return self.class_type == "annotation" or self.class_type == "annotation_type"

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
        """Check if this is a Lombok @Data class."""
        return self.has_data or (self.has_getter and self.has_setter)

    @property
    def is_immutable(self) -> bool:
        """Check if this is an immutable class (record or @Value)."""
        return self.is_record or self.has_value

    @property
    def lombok_annotations(self) -> list[str]:
        """Get list of Lombok annotations present."""
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


class JavaParser:
    """Parse Java source files using tree-sitter."""

    def __init__(self):
        self.language = Language(tsjava.language(), "java")
        self.parser = Parser()
        self.parser.set_language(self.language)

    def parse_file(self, file_path: str) -> Optional[ClassInfo]:
        """Parse a single Java file and extract class information."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
            return self.parse_source(source_code, file_path)
        except Exception as e:
            logger.error("Failed to parse file", file_path=file_path, error=str(e))
            return None

    def parse_source(self, source_code: str, file_path: str = "") -> Optional[ClassInfo]:
        """Parse Java source code and extract class information."""
        tree = self.parser.parse(bytes(source_code, "utf-8"))
        root_node = tree.root_node

        # Extract package
        package = self._extract_package(root_node, source_code)

        # Extract imports
        imports = self._extract_imports(root_node, source_code)

        # Find main class/interface declaration
        class_node = self._find_class_declaration(root_node)
        if not class_node:
            return None

        return self._extract_class_info(
            class_node, source_code, package, imports, file_path
        )

    def _extract_package(self, root_node, source_code: str) -> str:
        """Extract package declaration."""
        for child in root_node.children:
            if child.type == "package_declaration":
                # Get the scoped identifier
                for node in child.children:
                    if node.type == "scoped_identifier" or node.type == "identifier":
                        return self._get_node_text(node, source_code)
        return ""

    def _extract_imports(self, root_node, source_code: str) -> list[str]:
        """Extract import statements."""
        imports = []
        for child in root_node.children:
            if child.type == "import_declaration":
                import_text = self._get_node_text(child, source_code)
                # Clean up the import statement
                import_text = import_text.replace("import ", "").replace(";", "").strip()
                imports.append(import_text)
        return imports

    def _find_class_declaration(self, root_node):
        """Find the main class/interface/record/enum/annotation declaration."""
        # Priority order: class, record, interface, enum, annotation
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

    def _extract_class_info(
        self,
        class_node,
        source_code: str,
        package: str,
        imports: list[str],
        file_path: str,
    ) -> ClassInfo:
        """Extract detailed class information from AST node."""
        # Determine class type from node type
        node_type = class_node.type
        class_type = self._normalize_class_type(node_type)
        
        name = ""
        annotations = []
        modifiers = []
        implements = []
        extends = None
        javadoc = None
        record_components = []
        enum_constants = []

        # Get preceding javadoc/annotations
        javadoc = self._get_preceding_javadoc(class_node, source_code)

        for child in class_node.children:
            if child.type == "identifier":
                name = self._get_node_text(child, source_code)
            elif child.type == "modifiers":
                annotations = self._extract_annotations(child, source_code)
                modifiers = self._extract_modifiers(child, source_code)
            elif child.type == "superclass":
                extends = self._extract_superclass(child, source_code)
            elif child.type == "super_interfaces" or child.type == "extends_interfaces":
                implements = self._extract_interfaces(child, source_code)
            elif child.type == "formal_parameters":
                # Record components (for record declarations)
                record_components = self._extract_record_components(child, source_code)

        # Extract fields and methods from class body
        fields = []
        detailed_fields = []
        methods = []
        inner_classes = []

        body_node = self._find_child_by_type(class_node, "class_body") or \
                    self._find_child_by_type(class_node, "interface_body") or \
                    self._find_child_by_type(class_node, "enum_body") or \
                    self._find_child_by_type(class_node, "record_body") or \
                    self._find_child_by_type(class_node, "annotation_type_body")

        if body_node:
            for child in body_node.children:
                if child.type == "field_declaration":
                    field_info = self._extract_field(child, source_code)
                    if field_info:
                        fields.append(field_info)
                    # Also extract detailed field info
                    detailed_field = self._extract_detailed_field(child, source_code)
                    if detailed_field:
                        detailed_fields.append(detailed_field)
                elif child.type == "method_declaration":
                    method_info = self._extract_method(child, source_code)
                    if method_info:
                        methods.append(method_info)
                elif child.type == "compact_constructor_declaration":
                    # Record compact constructor
                    method_info = self._extract_compact_constructor(child, source_code, name)
                    if method_info:
                        methods.append(method_info)
                elif child.type == "enum_constant":
                    # Enum constants
                    const_name = self._extract_enum_constant(child, source_code)
                    if const_name:
                        enum_constants.append(const_name)
                elif child.type in ("class_declaration", "interface_declaration", 
                                   "record_declaration", "enum_declaration"):
                    inner_class = self._extract_class_info(
                        child, source_code, package, [], file_path
                    )
                    if inner_class:
                        inner_classes.append(inner_class)

        # Detect Lombok annotations
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
        """Normalize node type to class type string."""
        type_mapping = {
            "class_declaration": "class",
            "interface_declaration": "interface",
            "enum_declaration": "enum",
            "record_declaration": "record",
            "annotation_type_declaration": "annotation",
        }
        return type_mapping.get(node_type, node_type.replace("_declaration", ""))

    def _extract_modifiers(self, modifiers_node, source_code: str) -> list[str]:
        """Extract modifiers (public, private, abstract, final, sealed, etc.)."""
        modifiers = []
        modifier_types = {
            "public", "private", "protected", "static", "final", 
            "abstract", "synchronized", "native", "transient", "volatile",
            "sealed", "non-sealed", "strictfp"
        }
        for child in modifiers_node.children:
            text = self._get_node_text(child, source_code)
            if text in modifier_types:
                modifiers.append(text)
        return modifiers

    def _extract_record_components(self, params_node, source_code: str) -> list[RecordComponent]:
        """Extract record components from formal parameters."""
        components = []
        for child in params_node.children:
            if child.type == "formal_parameter":
                comp_type = ""
                comp_name = ""
                comp_annotations = []
                for param_child in child.children:
                    if param_child.type == "modifiers":
                        comp_annotations = self._extract_annotations(param_child, source_code)
                    elif param_child.type in ("type_identifier", "generic_type", "array_type",
                                             "integral_type", "floating_point_type", "boolean_type"):
                        comp_type = self._get_node_text(param_child, source_code)
                    elif param_child.type == "identifier":
                        comp_name = self._get_node_text(param_child, source_code)
                if comp_type and comp_name:
                    components.append(RecordComponent(
                        name=comp_name,
                        type=comp_type,
                        annotations=comp_annotations,
                    ))
        return components

    def _extract_compact_constructor(self, node, source_code: str, class_name: str) -> Optional[MethodInfo]:
        """Extract compact constructor from record."""
        annotations = []
        body = ""
        for child in node.children:
            if child.type == "modifiers":
                annotations = self._extract_annotations(child, source_code)
            elif child.type == "block":
                body = self._get_node_text(child, source_code)
        
        return MethodInfo(
            name=class_name,  # Constructor has same name as class
            return_type="",
            parameters=[],
            body=body,
            annotations=annotations,
            javadoc=None,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
        )

    def _extract_enum_constant(self, node, source_code: str) -> Optional[str]:
        """Extract enum constant name."""
        for child in node.children:
            if child.type == "identifier":
                return self._get_node_text(child, source_code)
        return None

    def _extract_annotations(self, modifiers_node, source_code: str) -> list[str]:
        """Extract annotations from modifiers node."""
        annotations = []
        for child in modifiers_node.children:
            if child.type == "marker_annotation" or child.type == "annotation":
                annotations.append(self._get_node_text(child, source_code))
        return annotations

    def _extract_superclass(self, superclass_node, source_code: str) -> Optional[str]:
        """Extract superclass name."""
        for child in superclass_node.children:
            if child.type == "type_identifier" or child.type == "generic_type":
                return self._get_node_text(child, source_code)
        return None

    def _extract_interfaces(self, interfaces_node, source_code: str) -> list[str]:
        """Extract implemented/extended interfaces."""
        interfaces = []
        for child in interfaces_node.children:
            if child.type == "type_identifier" or child.type == "generic_type":
                interfaces.append(self._get_node_text(child, source_code))
            elif child.type == "type_list":
                for type_child in child.children:
                    if type_child.type == "type_identifier" or type_child.type == "generic_type":
                        interfaces.append(self._get_node_text(type_child, source_code))
        return interfaces

    def _extract_field(
        self, field_node, source_code: str
    ) -> Optional[tuple[str, str, list[str]]]:
        """Extract field information (legacy format)."""
        field_type = ""
        field_name = ""
        annotations = []

        for child in field_node.children:
            if child.type == "modifiers":
                annotations = self._extract_annotations(child, source_code)
            elif child.type in ("type_identifier", "generic_type", "array_type", 
                               "integral_type", "floating_point_type", "boolean_type"):
                field_type = self._get_node_text(child, source_code)
            elif child.type == "variable_declarator":
                for var_child in child.children:
                    if var_child.type == "identifier":
                        field_name = self._get_node_text(var_child, source_code)
                        break

        if field_type and field_name:
            return (field_type, field_name, annotations)
        return None

    def _extract_detailed_field(self, field_node, source_code: str) -> Optional[FieldInfo]:
        """Extract detailed field information."""
        field_type = ""
        field_name = ""
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
                for var_child in child.children:
                    if var_child.type == "identifier":
                        field_name = self._get_node_text(var_child, source_code)
                    elif var_child.type not in ("identifier", "dimensions"):
                        # This is the initializer/default value
                        default_value = self._get_node_text(var_child, source_code)

        if field_type and field_name:
            return FieldInfo(
                name=field_name,
                type=field_type,
                annotations=annotations,
                modifiers=modifiers,
                default_value=default_value,
                javadoc=javadoc,
            )
        return None

    def _detect_lombok_annotations(self, annotations: list[str]) -> dict:
        """Detect Lombok annotations and return flags."""
        result = {
            "has_builder": False,
            "has_builder_to_builder": False,
            "has_data": False,
            "has_getter": False,
            "has_setter": False,
            "has_no_args_constructor": False,
            "has_all_args_constructor": False,
            "has_required_args_constructor": False,
            "has_value": False,
            "has_equals_and_hash_code": False,
            "has_to_string": False,
        }

        for ann in annotations:
            ann_lower = ann.lower()
            
            # @Builder detection
            if "@builder" in ann_lower:
                result["has_builder"] = True
                # Check for toBuilder=true
                if "tobuilder" in ann_lower and "true" in ann_lower:
                    result["has_builder_to_builder"] = True
            
            # @Data
            if "@data" in ann_lower:
                result["has_data"] = True
                # @Data implies @Getter, @Setter, @ToString, @EqualsAndHashCode
                result["has_getter"] = True
                result["has_setter"] = True
                result["has_to_string"] = True
                result["has_equals_and_hash_code"] = True
            
            # @Value (immutable)
            if ann_lower == "@value" or ann_lower.startswith("@value("):
                result["has_value"] = True
                result["has_getter"] = True
                result["has_to_string"] = True
                result["has_equals_and_hash_code"] = True
            
            # Individual annotations
            if "@getter" in ann_lower:
                result["has_getter"] = True
            if "@setter" in ann_lower:
                result["has_setter"] = True
            if "@noargsconstructor" in ann_lower:
                result["has_no_args_constructor"] = True
            if "@allargsconstructor" in ann_lower:
                result["has_all_args_constructor"] = True
            if "@requiredargsconstructor" in ann_lower:
                result["has_required_args_constructor"] = True
            if "@equalsandhashcode" in ann_lower:
                result["has_equals_and_hash_code"] = True
            if "@tostring" in ann_lower:
                result["has_to_string"] = True

        return result

    def _extract_method(self, method_node, source_code: str) -> Optional[MethodInfo]:
        """Extract method information."""
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
                name=name,
                return_type=return_type,
                parameters=parameters,
                body=body,
                annotations=annotations,
                javadoc=javadoc,
                start_line=method_node.start_point[0] + 1,
                end_line=method_node.end_point[0] + 1,
            )
        return None

    def _extract_parameters(
        self, params_node, source_code: str
    ) -> list[tuple[str, str]]:
        """Extract method parameters."""
        parameters = []
        for child in params_node.children:
            if child.type == "formal_parameter" or child.type == "spread_parameter":
                param_type = ""
                param_name = ""
                for param_child in child.children:
                    if param_child.type in ("type_identifier", "generic_type", "array_type",
                                           "integral_type", "floating_point_type", "boolean_type"):
                        param_type = self._get_node_text(param_child, source_code)
                    elif param_child.type == "identifier":
                        param_name = self._get_node_text(param_child, source_code)
                if param_type and param_name:
                    parameters.append((param_type, param_name))
        return parameters

    def _get_preceding_javadoc(self, node, source_code: str) -> Optional[str]:
        """Get javadoc comment preceding a node."""
        # Look for block comment before the node
        start_byte = node.start_byte
        preceding_text = source_code[:start_byte].strip()
        
        if preceding_text.endswith("*/"):
            # Find the start of the javadoc
            javadoc_start = preceding_text.rfind("/**")
            if javadoc_start != -1:
                return preceding_text[javadoc_start:]
        return None

    def _find_child_by_type(self, node, type_name: str):
        """Find first child node of given type."""
        for child in node.children:
            if child.type == type_name:
                return child
        return None

    def _get_node_text(self, node, source_code: str) -> str:
        """Get text content of a node."""
        return source_code[node.start_byte:node.end_byte]

    def parse_directory(self, directory: str) -> list[ClassInfo]:
        """Parse all Java files in a directory recursively."""
        classes = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".java"):
                    file_path = os.path.join(root, file)
                    class_info = self.parse_file(file_path)
                    if class_info:
                        classes.append(class_info)
                        logger.info(
                            "Parsed class",
                            class_name=class_info.name,
                            file_path=file_path,
                        )
        return classes

