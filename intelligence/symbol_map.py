"""
Symbol Map — global symbol table for the repository.

Provides fast O(1) lookups for:
  • Class → its methods, fields, annotations, type info
  • Method name → which class(es) define it
  • Field type → which classes use it as a dependency

This replaces slow vector-search for structural queries like
"what methods does OrderService have?" or "who injects UserRepository?".

Usage::

    from intelligence.repo_scanner import RepoScanner
    from intelligence.symbol_map import SymbolMap

    snapshot = RepoScanner().scan("/path/to/repo")
    symbols = SymbolMap.build(snapshot)

    entry = symbols.lookup("OrderService")
    print(entry.methods)     # [MethodSymbol(...), ...]
    print(entry.fields)      # [FieldSymbol(...), ...]
    print(entry.annotations) # ["@Service", "@RequiredArgsConstructor"]
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import structlog

from .repo_scanner import RepoSnapshot
from indexer.parse_java import ClassInfo, MethodInfo, FieldInfo

logger = structlog.get_logger()


# ── Symbol data structures ───────────────────────────────────────────

@dataclass
class MethodSymbol:
    """Compact representation of a method for symbol lookups."""

    name: str
    return_type: str
    parameters: list[tuple[str, str]]  # [(type, name), ...]
    annotations: list[str]
    is_public: bool = True


@dataclass
class FieldSymbol:
    """Compact representation of a field."""

    name: str
    type: str
    annotations: list[str]
    is_injectable: bool = False  # True if @Autowired, @Inject, or constructor-injected


@dataclass
class SymbolEntry:
    """Full symbol table entry for a single class/interface/record/enum."""

    name: str
    fqn: str
    package: str
    file_path: str
    java_type: str  # class | interface | record | enum
    annotations: list[str]
    extends: Optional[str]
    implements: list[str]
    methods: list[MethodSymbol]
    fields: list[FieldSymbol]
    # Quick classification flags
    is_service: bool = False
    is_repository: bool = False
    is_controller: bool = False
    is_model: bool = False
    is_interface: bool = False
    # Lombok flags
    has_builder: bool = False
    has_data: bool = False
    has_value: bool = False


# ── Symbol Map ───────────────────────────────────────────────────────

class SymbolMap:
    """Global symbol table for a scanned repository.

    Provides multiple index views over the same data:
      • by name (simple class name)
      • by FQN
      • by method name → classes that define it
      • by field type → classes that inject/use it
      • by annotation → classes that have it
    """

    def __init__(self) -> None:
        # Primary indexes
        self._by_name: dict[str, list[SymbolEntry]] = defaultdict(list)
        self._by_fqn: dict[str, SymbolEntry] = {}

        # Reverse indexes
        self._method_to_classes: dict[str, list[str]] = defaultdict(list)
        self._field_type_to_classes: dict[str, list[str]] = defaultdict(list)
        self._annotation_to_classes: dict[str, list[str]] = defaultdict(list)

        # Classification indexes
        self._services: list[str] = []
        self._repositories: list[str] = []
        self._controllers: list[str] = []
        self._models: list[str] = []
        self._interfaces: list[str] = []

    # ── Construction ─────────────────────────────────────────────────

    @classmethod
    def build(cls, snapshot: RepoSnapshot) -> SymbolMap:
        """Build symbol map from a ``RepoSnapshot``."""
        start = time.time()
        sm = cls()

        for ci in snapshot.classes:
            entry = sm._class_to_entry(ci)
            sm._register(entry)

        elapsed = (time.time() - start) * 1000
        logger.info(
            "Symbol map built",
            entries=len(sm._by_fqn),
            methods=sum(len(e.methods) for e in sm._by_fqn.values()),
            fields=sum(len(e.fields) for e in sm._by_fqn.values()),
            elapsed_ms=round(elapsed, 1),
        )
        return sm

    # ── Primary lookups ──────────────────────────────────────────────

    def lookup(self, name: str) -> Optional[SymbolEntry]:
        """Lookup by simple class name (first match)."""
        matches = self._by_name.get(name)
        return matches[0] if matches else None

    def lookup_all(self, name: str) -> list[SymbolEntry]:
        """Lookup all entries with a given simple name."""
        return self._by_name.get(name, [])

    def lookup_fqn(self, fqn: str) -> Optional[SymbolEntry]:
        """Lookup by fully-qualified name."""
        return self._by_fqn.get(fqn)

    # ── Reverse lookups ──────────────────────────────────────────────

    def classes_with_method(self, method_name: str) -> list[str]:
        """Find all classes that define a method with the given name."""
        return self._method_to_classes.get(method_name, [])

    def classes_injecting(self, type_name: str) -> list[str]:
        """Find all classes that have a field of the given type (dependency)."""
        return self._field_type_to_classes.get(type_name, [])

    def classes_with_annotation(self, annotation: str) -> list[str]:
        """Find all classes with the given annotation."""
        # Normalize: accept both "@Service" and "Service"
        key = annotation if annotation.startswith("@") else f"@{annotation}"
        return self._annotation_to_classes.get(key, [])

    # ── Classification queries ───────────────────────────────────────

    @property
    def services(self) -> list[str]:
        return list(self._services)

    @property
    def repositories(self) -> list[str]:
        return list(self._repositories)

    @property
    def controllers(self) -> list[str]:
        return list(self._controllers)

    @property
    def models(self) -> list[str]:
        return list(self._models)

    @property
    def interfaces(self) -> list[str]:
        return list(self._interfaces)

    # ── Method-level queries ─────────────────────────────────────────

    def get_method_signatures(self, class_name: str) -> list[str]:
        """Get all method signatures for a class (for prompt context)."""
        entry = self.lookup(class_name)
        if not entry:
            return []
        sigs = []
        for m in entry.methods:
            params = ", ".join(f"{t} {n}" for t, n in m.parameters)
            sigs.append(f"{m.return_type} {m.name}({params})")
        return sigs

    def get_injectable_dependencies(self, class_name: str) -> list[FieldSymbol]:
        """Get all injectable fields (dependencies) for a class."""
        entry = self.lookup(class_name)
        if not entry:
            return []
        return [f for f in entry.fields if f.is_injectable]

    # ── Statistics ───────────────────────────────────────────────────

    def get_summary(self) -> dict:
        return {
            "total_symbols": len(self._by_fqn),
            "services": len(self._services),
            "repositories": len(self._repositories),
            "controllers": len(self._controllers),
            "models": len(self._models),
            "interfaces": len(self._interfaces),
            "unique_methods": len(self._method_to_classes),
        }

    # ── Internal helpers ─────────────────────────────────────────────

    def _class_to_entry(self, ci: ClassInfo) -> SymbolEntry:
        """Convert a ``ClassInfo`` to a ``SymbolEntry``."""
        annotations_lower = " ".join(ci.annotations).lower()

        is_service = any(
            a in annotations_lower
            for a in ("@service", "@component")
        )
        is_repo = "@repository" in annotations_lower
        is_controller = any(
            a in annotations_lower
            for a in ("@controller", "@restcontroller")
        )
        is_model = (
            ci.is_record
            or ci.has_data
            or ci.has_value
            or any(
                ci.name.endswith(s)
                for s in ("Request", "Response", "Dto", "DTO", "Entity", "Command", "Event")
            )
            or "@entity" in annotations_lower
        )

        # Methods
        methods = [
            MethodSymbol(
                name=m.name,
                return_type=m.return_type,
                parameters=m.parameters,
                annotations=m.annotations,
                is_public=not any(
                    a in (" ".join(m.annotations) + " " + " ".join(getattr(m, "modifiers", [])))
                    for a in ("private", "protected")
                ),
            )
            for m in ci.methods
        ]

        # Fields
        fields = []
        injectable_annotations = {"@autowired", "@inject", "@mock"}
        for f in ci.detailed_fields:
            ann_lower = " ".join(f.annotations).lower()
            is_injectable = (
                any(a in ann_lower for a in injectable_annotations)
                or (
                    "final" in f.modifiers
                    and ci.has_required_args_constructor
                    and f.type[0].isupper()  # Likely a class type, not primitive
                )
                or (
                    "final" in f.modifiers
                    and "@requiredargsconstructor" in annotations_lower
                    and f.type[0].isupper()
                )
            )
            fields.append(
                FieldSymbol(
                    name=f.name,
                    type=f.type,
                    annotations=f.annotations,
                    is_injectable=is_injectable,
                )
            )

        return SymbolEntry(
            name=ci.name,
            fqn=ci.fully_qualified_name,
            package=ci.package,
            file_path=ci.file_path,
            java_type=ci.class_type,
            annotations=ci.annotations,
            extends=ci.extends,
            implements=ci.implements,
            methods=methods,
            fields=fields,
            is_service=is_service,
            is_repository=is_repo,
            is_controller=is_controller,
            is_model=is_model,
            is_interface=ci.is_interface,
            has_builder=ci.has_builder,
            has_data=ci.has_data,
            has_value=ci.has_value,
        )

    def _register(self, entry: SymbolEntry) -> None:
        """Register a symbol entry in all indexes."""
        self._by_name[entry.name].append(entry)
        self._by_fqn[entry.fqn] = entry

        # Reverse: method → classes
        for m in entry.methods:
            if m.name not in self._method_to_classes or entry.fqn not in self._method_to_classes[m.name]:
                self._method_to_classes[m.name].append(entry.fqn)

        # Reverse: field type → classes
        for f in entry.fields:
            if f.is_injectable:
                self._field_type_to_classes[f.type].append(entry.fqn)

        # Reverse: annotation → classes
        for ann in entry.annotations:
            key = ann if ann.startswith("@") else f"@{ann}"
            self._annotation_to_classes[key].append(entry.fqn)

        # Classification
        if entry.is_service:
            self._services.append(entry.fqn)
        if entry.is_repository:
            self._repositories.append(entry.fqn)
        if entry.is_controller:
            self._controllers.append(entry.fqn)
        if entry.is_model:
            self._models.append(entry.fqn)
        if entry.is_interface:
            self._interfaces.append(entry.fqn)
