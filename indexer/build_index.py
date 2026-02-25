"""
Build and manage the Qdrant vector index for Java codebase.
"""

import hashlib
import os
import re
import warnings
from typing import Optional

# Bypass SSL for corporate proxy
os.environ.setdefault('HF_HUB_DISABLE_SSL_VERIFY', '1')
warnings.filterwarnings('ignore')

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

from .parse_java import JavaParser, ClassInfo
from .summarize import CodeSummarizer

logger = structlog.get_logger()


class IndexBuilder:
    """Build and manage the vector index for Java code."""

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "java_codebase",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.embedder = SentenceTransformer(embedding_model)
        self.vector_size = self.embedder.get_sentence_embedding_dimension()
        self.parser = JavaParser()
        self.summarizer = CodeSummarizer()

    def create_collection(self, recreate: bool = False) -> None:
        """Create or recreate the Qdrant collection."""
        collections = self.qdrant.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists and recreate:
            logger.info("Deleting existing collection", collection=self.collection_name)
            self.qdrant.delete_collection(self.collection_name)
            exists = False

        if not exists:
            logger.info("Creating collection", collection=self.collection_name)
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

            # Create payload indexes for filtering
            self.qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="type",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="layer",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="class_name",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="package",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="extends",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="implements",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def index_repository(self, repo_path: str, recreate: bool = False) -> int:
        """Index all Java files in a repository."""
        logger.info("Starting repository indexing", repo_path=repo_path)

        # Create collection
        self.create_collection(recreate=recreate)

        # Parse all Java files
        classes = self.parser.parse_directory(repo_path)
        logger.info("Parsed Java files", count=len(classes))

        # Index each class
        points = []

        for class_info in classes:
            # Stable point ID from fully_qualified_name hash
            # Using int from first 8 hex chars of MD5 → avoids conflict on re-index
            class_fqn = class_info.fully_qualified_name
            class_id = int(hashlib.md5(class_fqn.encode()).hexdigest()[:15], 16)

            # Index the class itself
            class_point = self._create_class_point(class_info, class_id)
            if class_point:
                points.append(class_point)

            # Index individual methods for large classes (> 5 methods)
            if len(class_info.methods) > 5:
                for method in class_info.methods:
                    method_fqn = f"{class_fqn}.{method.name}"
                    method_id = int(hashlib.md5(method_fqn.encode()).hexdigest()[:15], 16)
                    method_point = self._create_method_point(method, class_info, method_id)
                    if method_point:
                        points.append(method_point)

        # Batch upsert to Qdrant
        if points:
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                )
                logger.info("Indexed batch", start=i, end=i + len(batch))

        logger.info("Indexing complete", total_points=len(points))
        return len(points)

    def _create_class_point(
        self, class_info: ClassInfo, point_id: int
    ) -> Optional[models.PointStruct]:
        """Create a Qdrant point for a class."""
        try:
            # Generate summary
            summary = self.summarizer.summarize_class(class_info)

            # Generate embedding
            embedding = self.embedder.encode(summary).tolist()

            # Extract service-level dependencies (fields injected as services)
            dependencies = []
            for field_type, _, _ in class_info.fields:
                if any(
                    field_type.endswith(suffix)
                    for suffix in ["Service", "Repository", "Client", "Gateway", "Handler"]
                ):
                    dependencies.append(field_type)

            # Extract all used domain types (models, DTOs, entities)
            # from fields + method params + method return types
            used_types = self._extract_used_types(class_info)

            # Create point
            return models.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "type": self.summarizer.detect_type(class_info),
                    "java_type": class_info.class_type,  # class, record, interface, enum, annotation
                    "layer": self.summarizer.detect_layer(class_info),
                    "class_name": class_info.name,
                    "package": class_info.package,
                    "file_path": class_info.file_path,
                    "fully_qualified_name": class_info.fully_qualified_name,
                    "summary": summary,
                    "dependencies": dependencies,
                    "used_types": used_types,
                    "method_count": len(class_info.methods),
                    "annotations": class_info.annotations,
                    "element_type": class_info.class_type,  # class, record, interface, enum, annotation
                    "modifiers": getattr(class_info, 'modifiers', []),
                    # Inheritance
                    "extends": class_info.extends,
                    "implements": class_info.implements,
                    # Record/Enum specifics (for filtering)
                    "enum_constants": getattr(class_info, 'enum_constants', [])[:10],
                    "record_components": [
                        {"type": c.type, "name": c.name}
                        for c in getattr(class_info, 'record_components', [])
                    ],
                    # Lombok info
                    "has_builder": getattr(class_info, 'has_builder', False),
                    "has_builder_to_builder": getattr(class_info, 'has_builder_to_builder', False),
                    "has_data": getattr(class_info, 'has_data', False),
                    "has_getter": getattr(class_info, 'has_getter', False),
                    "has_setter": getattr(class_info, 'has_setter', False),
                    "has_value": getattr(class_info, 'has_value', False),
                    "is_immutable": getattr(class_info, 'is_immutable', False),
                    "has_no_args_constructor": getattr(class_info, 'has_no_args_constructor', False),
                    "has_all_args_constructor": getattr(class_info, 'has_all_args_constructor', False),
                    "has_required_args_constructor": getattr(class_info, 'has_required_args_constructor', False),
                    # Fields info
                    "field_count": len(class_info.fields),
                    "fields": [
                        {"type": f.type, "name": f.name, "annotations": f.annotations}
                        for f in getattr(class_info, 'detailed_fields', [])
                    ][:20],  # Limit to 20 fields
                },
            )
        except Exception as e:
            logger.error(
                "Failed to create class point",
                class_name=class_info.name,
                error=str(e),
            )
            return None

    def _create_method_point(
        self, method, class_info: ClassInfo, point_id: int
    ) -> Optional[models.PointStruct]:
        """Create a Qdrant point for a method."""
        try:
            # Generate summary
            summary = self.summarizer.summarize_method(method, class_info)

            # Generate embedding
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
                    "dependencies": [],
                    "annotations": method.annotations,
                    "element_type": "method",
                    "start_line": method.start_line,
                    "end_line": method.end_line,
                },
            )
        except Exception as e:
            logger.error(
                "Failed to create method point",
                method_name=method.name,
                class_name=class_info.name,
                error=str(e),
            )
            return None

    def _extract_used_types(self, class_info: ClassInfo) -> list[str]:
        """Extract all custom domain types used by this class.

        Collects types from:
        - Field types (detailed_fields and legacy fields)
        - Method parameter types
        - Method return types

        Filters out Java builtins so only domain-specific types remain
        (e.g., OrderRequest, OrderEntity, OrderResponse).
        """
        # Java builtins and common library types to exclude
        _BUILTINS = {
            # Primitives and wrappers
            "void", "boolean", "byte", "char", "short", "int", "long", "float", "double",
            "Boolean", "Byte", "Character", "Short", "Integer", "Long", "Float", "Double",
            "Number", "Object",
            # Strings and common value types
            "String", "StringBuilder", "StringBuffer", "CharSequence",
            "UUID", "URI", "URL", "BigDecimal", "BigInteger",
            # Temporal
            "LocalDate", "LocalDateTime", "LocalTime", "ZonedDateTime", "Instant",
            "Date", "Calendar", "Timestamp",
            # Collections (generic wrappers — inner type may be domain)
            "List", "Map", "Set", "Collection", "Optional", "Stream",
            "ArrayList", "HashMap", "HashSet", "LinkedList",
            # Reactive
            "Mono", "Flux",
            # Spring / framework
            "ResponseEntity", "HttpStatus", "HttpHeaders", "Page", "Pageable",
            "Sort", "Slice", "Result",
            # Misc
            "Void", "Class", "Enum",
        }

        def _is_custom_type(raw_type: str) -> bool:
            """Return True if raw_type looks like a domain class name."""
            if not raw_type:
                return False
            inner = re.findall(r'[A-Z][A-Za-z0-9]*', raw_type)
            return any(t not in _BUILTINS for t in inner)

        def _extract_inner_types(raw_type: str) -> list[str]:
            """Extract all capitalised type names from a possibly generic type string."""
            return [
                t for t in re.findall(r'[A-Z][A-Za-z0-9]+', raw_type)
                if t not in _BUILTINS and len(t) > 2
            ]

        seen: set[str] = set()
        # Exclude self and own service-level dependencies
        seen.add(class_info.name)

        # 1. Field types (detailed)
        for f in getattr(class_info, 'detailed_fields', []):
            for t in _extract_inner_types(f.type):
                seen.add(t)  # will be added to result below

        # 2. Field types (legacy tuple)
        for field_type, _, _ in class_info.fields:
            for t in _extract_inner_types(field_type):
                seen.add(t)

        result: set[str] = set()

        # Re-run to separate out self/builtins
        def collect(raw_type: str):
            for t in _extract_inner_types(raw_type):
                if t != class_info.name:
                    result.add(t)

        for f in getattr(class_info, 'detailed_fields', []):
            collect(f.type)
        for field_type, _, _ in class_info.fields:
            collect(field_type)

        # 3. Method parameters and return types
        for method in class_info.methods:
            collect(method.return_type)
            for param_type, _ in method.parameters:
                collect(param_type)

        # Remove service-level dependencies (those are in 'dependencies' field)
        service_suffixes = {"Service", "Repository", "Client", "Gateway", "Handler"}
        result = {
            t for t in result
            if not any(t.endswith(s) for s in service_suffixes)
        }

        return sorted(result)

    def get_collection_info(self) -> dict:
        """Get information about the collection."""
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
        """Delete the collection."""
        try:
            self.qdrant.delete_collection(self.collection_name)
            logger.info("Collection deleted", collection=self.collection_name)
            return True
        except Exception as e:
            logger.error("Failed to delete collection", error=str(e))
            return False

