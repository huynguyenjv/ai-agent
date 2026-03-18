"""
Build and manage the Qdrant vector index for Java codebase.

New in this version:
  - Dependency graph: tracks which classes use which other classes
  - Referenced type resolution: resolves simple names to fully-qualified names
  - Model-aware indexing: model/DTO classes get usage hints embedded in their summary
  - Enriched service summaries: include resolved model shapes for referenced types
"""

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models

from utils.embedding import ONNXEmbedder
from .parse_java import JavaParser, ClassInfo, TypeReference
from .summarize import CodeSummarizer

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Dependency graph
# ---------------------------------------------------------------------------

@dataclass
class DependencyEdge:
    """A directed edge: source_class --[context]--> target_class"""
    source: str          # fully qualified name of the using class
    target: str          # fully qualified name (or simple name) of the used class
    context: str         # "field", "parameter", "return_type", "extends", "implements"
    member: str          # field/method name where dependency appears


@dataclass
class DependencyGraph:
    """
    Bidirectional dependency graph over all parsed classes.

    outgoing[A] = list of classes A depends on
    incoming[B] = list of classes that depend on B
    """
    edges: list[DependencyEdge] = field(default_factory=list)
    outgoing: dict[str, list[DependencyEdge]] = field(default_factory=lambda: defaultdict(list))
    incoming: dict[str, list[DependencyEdge]] = field(default_factory=lambda: defaultdict(list))

    def add_edge(self, edge: DependencyEdge):
        self.edges.append(edge)
        self.outgoing[edge.source].append(edge)
        self.incoming[edge.target].append(edge)

    def dependencies_of(self, fqn: str) -> list[str]:
        """Classes that `fqn` directly depends on."""
        return list({e.target for e in self.outgoing.get(fqn, [])})

    def dependents_of(self, fqn: str) -> list[str]:
        """Classes that directly use `fqn`."""
        return list({e.source for e in self.incoming.get(fqn, [])})

    def to_payload(self, fqn: str) -> dict:
        """Serialize graph neighbourhood for a class into Qdrant payload."""
        return {
            "dependencies": self.dependencies_of(fqn),
            "dependents": self.dependents_of(fqn),
            "dependency_edges": [
                {"target": e.target, "context": e.context, "member": e.member}
                for e in self.outgoing.get(fqn, [])
            ],
        }


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

class IndexBuilder:
    """Build and manage the vector index for Java code."""

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "java_codebase",
        embedding_model: str = "all-MiniLM-L6-v2-onnx",
    ):
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.embedder = ONNXEmbedder(embedding_model)
        self.vector_size = self.embedder.get_sentence_embedding_dimension()
        self.parser = JavaParser()
        self.summarizer = CodeSummarizer()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(self, recreate: bool = False, collection_name: Optional[str] = None) -> None:
        effective_collection = collection_name or self.collection_name
        collections = self.qdrant.get_collections().collections
        exists = any(c.name == effective_collection for c in collections)

        if exists and recreate:
            logger.info("Deleting existing collection", collection=effective_collection)
            self.qdrant.delete_collection(effective_collection)
            exists = False

        if not exists:
            logger.info("Creating collection", collection=effective_collection)
            self.qdrant.create_collection(
                collection_name=effective_collection,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            for fname, ftype in [
                ("type", models.PayloadSchemaType.KEYWORD),
                ("layer", models.PayloadSchemaType.KEYWORD),
                ("class_name", models.PayloadSchemaType.KEYWORD),
                ("package", models.PayloadSchemaType.KEYWORD),
                ("java_type", models.PayloadSchemaType.KEYWORD),
                ("is_model", models.PayloadSchemaType.KEYWORD),
            ]:
                self.qdrant.create_payload_index(
                    collection_name=effective_collection,
                    field_name=fname,
                    field_schema=ftype,
                )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def index_repository(
        self, repo_path: str, recreate: bool = False,
        collection_name: Optional[str] = None,
    ) -> int:
        effective_collection = collection_name or self.collection_name
        logger.info("Starting repository indexing", repo_path=repo_path, collection=effective_collection)
        self.create_collection(recreate=recreate, collection_name=effective_collection)

        # 1. Parse all Java files
        classes = self.parser.parse_directory(repo_path)
        logger.info("Parsed Java files", count=len(classes))

        # 2. Build lookup maps
        #    name_map: simple_name -> ClassInfo  (last-wins if duplicate simple names)
        #    fqn_map:  fully_qualified_name -> ClassInfo
        name_map: dict[str, ClassInfo] = {}
        fqn_map: dict[str, ClassInfo] = {}
        for c in classes:
            name_map[c.name] = c
            fqn_map[c.fully_qualified_name] = c

        # 3. Build dependency graph
        graph = self._build_dependency_graph(classes, name_map, fqn_map)
        logger.info("Built dependency graph", edges=len(graph.edges))

        # 4. Build index points
        points: list[models.PointStruct] = []
        point_id = 0

        for class_info in classes:
            class_point = self._create_class_point(
                class_info, point_id, name_map, fqn_map, graph
            )
            if class_point:
                points.append(class_point)
                point_id += 1

            # Index methods individually for large classes
            if len(class_info.methods) > 5:
                for method in class_info.methods:
                    method_point = self._create_method_point(method, class_info, point_id)
                    if method_point:
                        points.append(method_point)
                        point_id += 1

        # 5. Batch upsert
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i: i + batch_size]
            self.qdrant.upsert(collection_name=effective_collection, points=batch)
            logger.info("Indexed batch", start=i, end=i + len(batch))

        logger.info("Indexing complete", total_points=len(points))
        return len(points)

    # ------------------------------------------------------------------
    # Single-file indexing (for Continue.dev / IDE integration)
    # ------------------------------------------------------------------

    def index_single_file(
        self,
        source_code: str,
        file_path: str,
        collection_name: Optional[str] = None,
    ) -> dict:
        """Index a single Java file's content into Qdrant.

        Reuses the same Java-aware parsing, summarization, and embedding
        pipeline as ``index_repository()`` but operates on one file.

        Steps:
            1. Parse source_code with JavaParser.parse_source()
            2. Delete old points for this file_path (idempotent update)
            3. Create class + method points
            4. Upsert to Qdrant

        Note:
            Cross-class dependency graph is *not* resolved here because
            we only have one file.  For full graph resolution, use
            ``index_repository()`` or ``/reindex``.

        Returns:
            dict with ``success``, ``classes_indexed``, ``points_created``,
            and ``error`` (if any).
        """
        effective_collection = collection_name or self.collection_name
        logger.info(
            "Indexing single file",
            file_path=file_path,
            collection=effective_collection,
        )

        try:
            # Ensure collection exists
            self.create_collection(recreate=False, collection_name=effective_collection)

            # 1. Parse
            class_info = self.parser.parse_source(source_code, file_path)
            if not class_info:
                return {
                    "success": False,
                    "classes_indexed": 0,
                    "points_created": 0,
                    "error": f"No Java class found in {file_path}",
                }

            classes = [class_info]
            # Include inner classes if present
            if class_info.inner_classes:
                classes.extend(class_info.inner_classes)

            # 2. Delete old points for this file_path
            self._delete_points_by_file_path(effective_collection, file_path)

            # 3. Build minimal dependency graph (single-file scope)
            name_map = {c.name: c for c in classes}
            fqn_map = {c.fully_qualified_name: c for c in classes}
            graph = self._build_dependency_graph(classes, name_map, fqn_map)

            # 4. Create points — use UUIDs for point IDs to avoid collisions
            import hashlib

            points: list[models.PointStruct] = []
            for cls in classes:
                # Deterministic ID from FQN so re-indexing replaces the same point
                point_id = self._fqn_to_point_id(cls.fully_qualified_name)
                class_point = self._create_class_point(
                    cls, point_id, name_map, fqn_map, graph,
                )
                if class_point:
                    points.append(class_point)

                # Index methods individually for large classes
                if len(cls.methods) > 5:
                    for method in cls.methods:
                        method_id = self._fqn_to_point_id(
                            f"{cls.fully_qualified_name}.{method.name}"
                        )
                        method_point = self._create_method_point(method, cls, method_id)
                        if method_point:
                            points.append(method_point)

            # 5. Upsert
            if points:
                self.qdrant.upsert(
                    collection_name=effective_collection, points=points,
                )

            logger.info(
                "Single file indexed",
                file_path=file_path,
                classes=len(classes),
                points=len(points),
            )
            return {
                "success": True,
                "classes_indexed": len(classes),
                "points_created": len(points),
                "error": None,
            }

        except Exception as e:
            logger.error(
                "Single file indexing failed",
                file_path=file_path,
                error=str(e),
            )
            return {
                "success": False,
                "classes_indexed": 0,
                "points_created": 0,
                "error": str(e),
            }

    def _delete_points_by_file_path(
        self, collection_name: str, file_path: str,
    ) -> None:
        """Delete all existing points for a given file_path (idempotent)."""
        try:
            self.qdrant.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="file_path",
                                match=models.MatchValue(value=file_path),
                            )
                        ]
                    )
                ),
            )
            logger.debug("Deleted old points", file_path=file_path)
        except Exception as e:
            # Collection may not exist yet or no points match — safe to ignore
            logger.debug("Delete old points skipped", file_path=file_path, reason=str(e))

    @staticmethod
    def _fqn_to_point_id(fqn: str) -> int:
        """Convert a fully-qualified name to a deterministic Qdrant point ID.

        Uses a hash to produce a positive integer that fits Qdrant's
        uint64 point ID range.
        """
        import hashlib
        digest = hashlib.sha256(fqn.encode()).hexdigest()
        # Take first 15 hex chars → fits in a positive 64-bit integer
        return int(digest[:15], 16)

    # ------------------------------------------------------------------
    # Dependency graph construction
    # ------------------------------------------------------------------

    def _build_dependency_graph(
        self,
        classes: list[ClassInfo],
        name_map: dict[str, ClassInfo],
        fqn_map: dict[str, ClassInfo],
    ) -> DependencyGraph:
        """
        For every class, resolve its referenced_types to FQNs (where possible)
        and build a directed dependency graph.
        """
        graph = DependencyGraph()

        for cls in classes:
            source_fqn = cls.fully_qualified_name

            for ref in cls.referenced_types:
                # Resolve to FQN if we have it in the codebase
                target_fqn = self._resolve_type(ref.type_name, cls, name_map, fqn_map)
                if target_fqn is None:
                    # External dependency — still track with simple name
                    target_fqn = ref.type_name

                # Avoid self-loops
                if target_fqn == source_fqn:
                    continue

                graph.add_edge(DependencyEdge(
                    source=source_fqn,
                    target=target_fqn,
                    context=ref.context,
                    member=ref.field_or_method,
                ))

        return graph

    def _resolve_type(
        self,
        simple_name: str,
        cls: ClassInfo,
        name_map: dict[str, ClassInfo],
        fqn_map: dict[str, ClassInfo],
    ) -> Optional[str]:
        """
        Try to resolve a simple type name to a fully-qualified name.
        Strategy:
          1. Check same package first
          2. Check import statements
          3. Fall back to global name_map
        """
        # 1. Same package
        candidate_fqn = f"{cls.package}.{simple_name}" if cls.package else simple_name
        if candidate_fqn in fqn_map:
            return candidate_fqn

        # 2. Explicit import
        for imp in cls.imports:
            if imp.endswith(f".{simple_name}"):
                if imp in fqn_map:
                    return imp
                # Even if not in fqn_map, use the import string as the target
                return imp

        # 3. Global lookup by simple name (may be ambiguous)
        if simple_name in name_map:
            return name_map[simple_name].fully_qualified_name

        return None

    # ------------------------------------------------------------------
    # Point creation
    # ------------------------------------------------------------------

    def _create_class_point(
        self,
        class_info: ClassInfo,
        point_id: int,
        name_map: dict[str, ClassInfo],
        fqn_map: dict[str, ClassInfo],
        graph: DependencyGraph,
    ) -> Optional[models.PointStruct]:
        try:
            is_model = self._is_model_class(class_info)
            fqn = class_info.fully_qualified_name

            # --- Build summary ---
            summary = self.summarizer.summarize_class(class_info)

            # For service/component classes: append resolved model shapes
            # so that the embedding captures HOW to use the models they depend on
            if not is_model:
                resolved_model_hints = self._resolve_model_hints(
                    class_info, name_map, fqn_map
                )
                if resolved_model_hints:
                    hint_block = "\n\nReferenced Model Shapes:\n" + "\n---\n".join(resolved_model_hints)
                    # Keep total length reasonable
                    combined = summary + hint_block
                    summary = combined[:4000]  # character cap; fine for embedding

            # --- Dependency graph neighbourhood ---
            graph_payload = graph.to_payload(fqn)

            # --- Separate used_types (domain models) from service deps ---
            all_dep_fqns = set(graph_payload.get("dependencies", []))
            service_deps: list[str] = []   # simple names of service-level deps
            used_types: list[str] = []     # simple names of domain model types

            for dep_fqn in all_dep_fqns:
                simple = dep_fqn.rsplit(".", 1)[-1] if "." in dep_fqn else dep_fqn
                dep_class = fqn_map.get(dep_fqn) or name_map.get(simple)
                if dep_class and self._is_model_class(dep_class):
                    used_types.append(simple)
                else:
                    service_deps.append(simple)

            # --- Resolved referenced models (for payload, not just embedding) ---
            referenced_models_detail = self._build_referenced_models_payload(
                class_info, name_map, fqn_map
            )

            # --- Embedding ---
            embedding = self.embedder.encode(summary).tolist()

            return models.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    # Identity
                    "type": self.summarizer.detect_type(class_info),
                    "java_type": class_info.class_type,
                    "layer": self.summarizer.detect_layer(class_info),
                    "class_name": class_info.name,
                    "package": class_info.package,
                    "file_path": class_info.file_path,
                    "fully_qualified_name": fqn,
                    "element_type": "class",
                    # Content
                    "summary": summary,
                    "annotations": class_info.annotations,
                    "modifiers": class_info.modifiers,
                    "method_count": len(class_info.methods),
                    "field_count": len(class_info.detailed_fields),
                    "fields": [
                        {"type": f.type, "name": f.name, "annotations": f.annotations}
                        for f in class_info.detailed_fields[:20]
                    ],
                    # Record components (for Java records)
                    "record_components": [
                        {"type": c.type, "name": c.name, "annotations": getattr(c, 'annotations', [])}
                        for c in class_info.record_components
                    ],
                    # Inheritance / interfaces
                    "extends": class_info.extends,
                    "implements": class_info.implements,
                    # Model / instantiation info
                    "is_model": is_model,
                    "usage_hint": self.summarizer.generate_usage_hint(class_info),
                    # Lombok flags
                    "has_builder": class_info.has_builder,
                    "has_builder_to_builder": class_info.has_builder_to_builder,
                    "has_data": class_info.has_data,
                    "has_getter": class_info.has_getter,
                    "has_setter": class_info.has_setter,
                    "has_value": class_info.has_value,
                    "is_immutable": class_info.is_immutable,
                    # Constructor flags
                    "has_no_args_constructor": class_info.has_no_args_constructor,
                    "has_all_args_constructor": class_info.has_all_args_constructor,
                    "has_required_args_constructor": class_info.has_required_args_constructor,
                    # Type references (simple names)
                    "referenced_class_names": class_info.referenced_class_names,
                    # Separated dependency lists (simple names)
                    "used_types": used_types,          # domain models/DTOs/entities
                    # Resolved model info for RAG context injection
                    "referenced_models": referenced_models_detail,
                    # Dependency graph (FQNs)
                    **graph_payload,
                },
            )
        except Exception as e:
            logger.error("Failed to create class point", class_name=class_info.name, error=str(e))
            return None

    def _create_method_point(
        self, method, class_info: ClassInfo, point_id: int
    ) -> Optional[models.PointStruct]:
        try:
            summary = self.summarizer.summarize_method(method, class_info)
            embedding = self.embedder.encode(summary).tolist()
            return models.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "type": "method",
                    "layer": self.summarizer.detect_layer(class_info),
                    "class_name": class_info.name,
                    "method_name": method.name,
                    "package": class_info.package,
                    "file_path": class_info.file_path,
                    "fully_qualified_name": f"{class_info.fully_qualified_name}.{method.name}",
                    "summary": summary,
                    "annotations": method.annotations,
                    "element_type": "method",
                    "start_line": method.start_line,
                    "end_line": method.end_line,
                },
            )
        except Exception as e:
            logger.error("Failed to create method point", method_name=method.name,
                         class_name=class_info.name, error=str(e))
            return None

    # ------------------------------------------------------------------
    # Model detection
    # ------------------------------------------------------------------

    def _is_model_class(self, class_info: ClassInfo) -> bool:
        """
        Detect if a class is a model/DTO that needs usage hints
        (as opposed to a service/component that is Spring-managed).
        """
        name = class_info.name
        annotations_lower = " ".join(class_info.annotations).lower()

        # Records are almost always DTOs
        if class_info.is_record:
            return True

        # Naming conventions
        if any(name.endswith(s) for s in [
            "Request", "Response", "Dto", "DTO", "Command",
            "Event", "Query", "Payload", "Form", "Model",
        ]):
            return True

        # JPA entity
        if "@entity" in annotations_lower:
            return True

        # Package-based detection: types in domain/model/dto/vo/openapi packages
        pkg = (class_info.package or "").lower()
        if any(seg in pkg.split(".") for seg in (
            "domain", "model", "dto", "vo", "entity", "openapi", "payload", "event",
        )):
            # But exclude Spring-managed classes in domain packages
            is_spring_managed = any(
                x in annotations_lower
                for x in ("@service", "@component", "@repository", "@controller", "@restcontroller", "@configuration")
            )
            if not is_spring_managed:
                return True

        # Lombok data/value/builder on a non-service class
        is_spring_managed = any(
            x in annotations_lower
            for x in ("@service", "@component", "@repository", "@controller", "@restcontroller")
        )
        if not is_spring_managed and (class_info.has_data or class_info.has_value or class_info.has_builder):
            return True

        return False

    # ------------------------------------------------------------------
    # Referenced model resolution helpers
    # ------------------------------------------------------------------

    def _resolve_model_hints(
        self,
        class_info: ClassInfo,
        name_map: dict[str, ClassInfo],
        fqn_map: dict[str, ClassInfo],
    ) -> list[str]:
        """
        For each referenced class that exists in the codebase AND is a model,
        generate a compact shape+usage string to embed alongside the parent class.
        """
        hints = []
        for ref_name in class_info.referenced_class_names:
            ref_class = name_map.get(ref_name)
            if ref_class and self._is_model_class(ref_class):
                hint = self.summarizer.generate_usage_hint(ref_class)
                fields_preview = ", ".join(
                    f"{f.type} {f.name}" for f in ref_class.detailed_fields[:5]
                )
                if not fields_preview and ref_class.record_components:
                    fields_preview = ", ".join(
                        f"{c.type} {c.name}" for c in ref_class.record_components
                    )
                hints.append(
                    f"{ref_class.fully_qualified_name} ({ref_class.class_type})\n"
                    f"Fields: {fields_preview}\n"
                    f"Usage:\n{hint}"
                )
        return hints

    def _build_referenced_models_payload(
        self,
        class_info: ClassInfo,
        name_map: dict[str, ClassInfo],
        fqn_map: dict[str, ClassInfo],
    ) -> list[dict]:
        """
        Build a structured payload entry for each referenced class found in the codebase.
        Stored in Qdrant so retrieval-time context can inject model shapes without re-querying.
        """
        result = []
        for ref_name in class_info.referenced_class_names:
            ref_class = name_map.get(ref_name)
            if not ref_class:
                # External type — just record the name
                result.append({"class_name": ref_name, "resolved": False})
                continue

            result.append({
                "class_name": ref_class.name,
                "fully_qualified_name": ref_class.fully_qualified_name,
                "java_type": ref_class.class_type,
                "resolved": True,
                "is_model": self._is_model_class(ref_class),
                "usage_hint": self.summarizer.generate_usage_hint(ref_class),
                "fields": [
                    {"type": f.type, "name": f.name}
                    for f in ref_class.detailed_fields[:10]
                ],
                "record_components": [
                    {"type": c.type, "name": c.name}
                    for c in ref_class.record_components
                ],
                "has_builder": ref_class.has_builder,
                "is_immutable": ref_class.is_immutable,
            })
        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_collection_info(self) -> dict:
        try:
            info = self.qdrant.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value,
            }
        except Exception as e:
            logger.error("Failed to get collection info", error=str(e))
            return {"error": str(e)}

    def delete_collection(self) -> bool:
        try:
            self.qdrant.delete_collection(self.collection_name)
            logger.info("Collection deleted", collection=self.collection_name)
            return True
        except Exception as e:
            logger.error("Failed to delete collection", error=str(e))
            return False

    def get_dependency_graph(self, repo_path: str) -> DependencyGraph:
        """
        Public helper: parse a repo and return the dependency graph
        without indexing — useful for analysis / visualization.
        """
        classes = self.parser.parse_directory(repo_path)
        name_map = {c.name: c for c in classes}
        fqn_map = {c.fully_qualified_name: c for c in classes}
        return self._build_dependency_graph(classes, name_map, fqn_map)

    # ------------------------------------------------------------------
    # Domain Registry Integration
    # ------------------------------------------------------------------

    def rebuild_domain_registry(
        self,
        collection_name: Optional[str] = None,
    ) -> dict:
        """Rebuild the Domain Type Registry after indexing.
        
        This should be called after index_repository() to ensure the registry
        has up-to-date construction patterns for all domain types.
        
        Returns:
            Registry statistics or error dict.
        """
        try:
            from .domain_registry import get_domain_registry
            
            effective_collection = collection_name or self.collection_name
            registry = get_domain_registry(
                qdrant_client=self.qdrant,
                default_collection=effective_collection,
            )
            
            stats = registry.build_from_collection(
                collection_name=effective_collection,
                force_rebuild=True,
            )
            
            logger.info(
                "Domain registry rebuilt",
                collection=effective_collection,
                total_types=stats.total_types,
                by_pattern=stats.by_pattern,
            )
            
            return {
                "success": True,
                "collection": effective_collection,
                "total_types": stats.total_types,
                "by_pattern": stats.by_pattern,
                "by_java_type": stats.by_java_type,
                "build_time_ms": round(stats.build_time_ms, 1),
            }
            
        except ImportError:
            logger.warning("Domain registry module not available")
            return {"success": False, "error": "Domain registry module not available"}
        except Exception as e:
            logger.error("Failed to rebuild domain registry", error=str(e))
            return {"success": False, "error": str(e)}

    def index_repository_with_registry(
        self,
        repo_path: str,
        recreate: bool = False,
        collection_name: Optional[str] = None,
    ) -> dict:
        """Index repository and rebuild domain registry in one call.
        
        This is the recommended method for full reindexing as it ensures
        the domain registry is always in sync with the index.
        
        Returns:
            Combined result with index and registry statistics.
        """
        effective_collection = collection_name or self.collection_name
        
        # Step 1: Index repository
        points_count = self.index_repository(
            repo_path=repo_path,
            recreate=recreate,
            collection_name=effective_collection,
        )
        
        # Step 2: Rebuild domain registry
        registry_result = self.rebuild_domain_registry(
            collection_name=effective_collection,
        )
        
        return {
            "index": {
                "collection": effective_collection,
                "points_count": points_count,
            },
            "registry": registry_result,
        }