"""
Build and manage the Qdrant vector index for Java codebase.
"""

import os
import ssl
import warnings
from pathlib import Path
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
        point_id = 0

        for class_info in classes:
            # Index the class itself
            class_point = self._create_class_point(class_info, point_id)
            if class_point:
                points.append(class_point)
                point_id += 1

            # Index individual methods for large classes
            if len(class_info.methods) > 5:
                for method in class_info.methods:
                    method_point = self._create_method_point(method, class_info, point_id)
                    if method_point:
                        points.append(method_point)
                        point_id += 1

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

            # Extract dependencies
            dependencies = []
            for field_type, _, _ in class_info.fields:
                if any(
                    field_type.endswith(suffix)
                    for suffix in ["Service", "Repository", "Client", "Gateway", "Handler"]
                ):
                    dependencies.append(field_type)

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
                    "method_count": len(class_info.methods),
                    "annotations": class_info.annotations,
                    "element_type": class_info.class_type,  # class, record, interface, enum, annotation
                    "modifiers": getattr(class_info, 'modifiers', []),
                    # Lombok info
                    "has_builder": getattr(class_info, 'has_builder', False),
                    "has_builder_to_builder": getattr(class_info, 'has_builder_to_builder', False),
                    "has_data": getattr(class_info, 'has_data', False),
                    "has_getter": getattr(class_info, 'has_getter', False),
                    "has_setter": getattr(class_info, 'has_setter', False),
                    "has_value": getattr(class_info, 'has_value', False),
                    "is_immutable": getattr(class_info, 'is_immutable', False),
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

