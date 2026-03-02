"""
RAG client for querying the Qdrant vector index.
"""

import os
import time
import warnings
from typing import Optional

# Bypass SSL for corporate proxy
os.environ.setdefault('HF_HUB_DISABLE_SSL_VERIFY', '1')
warnings.filterwarnings('ignore')

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

from .schema import CodeChunk, SearchQuery, SearchResult, MetadataFilter, IndexStats, FieldSchema

logger = structlog.get_logger()


class RAGClient:
    """Client for RAG operations on the Java codebase index."""

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
        logger.info(
            "RAG client initialized",
            host=qdrant_host,
            port=qdrant_port,
            collection=collection_name,
        )

    def search(self, query: SearchQuery) -> SearchResult:
        """Perform semantic search with optional metadata filtering."""
        start_time = time.time()

        # Generate query embedding
        query_vector = self.embedder.encode(query.query).tolist()

        # Build filter conditions
        filter_conditions = self._build_filter(query.filters)

        # Execute search
        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=query.top_k,
            score_threshold=query.score_threshold,
            query_filter=filter_conditions,
        )

        # Convert to CodeChunk objects
        chunks = []
        for result in results:
            payload = result.payload
            # Parse fields from payload
            fields_data = payload.get("fields", [])
            fields = [
                FieldSchema(
                    type=f.get("type", ""),
                    name=f.get("name", ""),
                    annotations=f.get("annotations", [])
                )
                for f in fields_data if isinstance(f, dict)
            ]
            # Parse record_components from payload (for Java records)
            record_components_data = payload.get("record_components", [])
            record_components = [
                FieldSchema(
                    type=f.get("type", ""),
                    name=f.get("name", ""),
                    annotations=f.get("annotations", [])
                )
                for f in record_components_data if isinstance(f, dict)
            ]
            chunk = CodeChunk(
                id=result.id,
                summary=payload.get("summary", ""),
                score=result.score,
                type=payload.get("type", "unknown"),
                layer=payload.get("layer", "unknown"),
                class_name=payload.get("class_name", ""),
                package=payload.get("package", ""),
                file_path=payload.get("file_path", ""),
                fully_qualified_name=payload.get("fully_qualified_name", ""),
                dependencies=payload.get("dependencies", []),
                annotations=payload.get("annotations", []),
                element_type=payload.get("element_type", "class"),
                java_type=payload.get("java_type"),
                method_name=payload.get("method_name"),
                start_line=payload.get("start_line"),
                end_line=payload.get("end_line"),
                # Inheritance
                extends=payload.get("extends"),
                implements=payload.get("implements", []),
                # Class meta
                method_count=payload.get("method_count", 0),
                modifiers=payload.get("modifiers", []),
                # Domain types used (models, DTOs, entities)
                used_types=payload.get("used_types", []),
                # Lombok info
                has_builder=payload.get("has_builder", False),
                has_builder_to_builder=payload.get("has_builder_to_builder", False),
                has_data=payload.get("has_data", False),
                has_getter=payload.get("has_getter", False),
                has_setter=payload.get("has_setter", False),
                has_value=payload.get("has_value", False),
                is_immutable=payload.get("is_immutable", False),
                has_no_args_constructor=payload.get("has_no_args_constructor", False),
                has_all_args_constructor=payload.get("has_all_args_constructor", False),
                has_required_args_constructor=payload.get("has_required_args_constructor", False),
                # Fields info
                field_count=payload.get("field_count", 0),
                fields=fields,
                record_components=record_components,
            )
            chunks.append(chunk)

        search_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "Search completed",
            query=query.query[:50],
            results=len(chunks),
            time_ms=round(search_time_ms, 2),
        )

        return SearchResult(
            query=query.query,
            chunks=chunks,
            total_found=len(chunks),
            search_time_ms=search_time_ms,
        )


    def search_by_class(
        self, class_name: str, top_k: int = 10, include_dependencies: bool = True
    ) -> SearchResult:
        """Search for a specific class and optionally its dependencies. Fallback: match all chunks with class_name if strict filter fails."""
        # First, try strict filter
        query = SearchQuery(
            query=f"class {class_name}",
            top_k=1,
            score_threshold=0.3,
            filters=MetadataFilter(class_name=class_name),
        )
        result = self.search(query)

        # Fallback: nếu không tìm thấy, tìm mọi chunk có class_name trùng (bất kể package)
        if not result.chunks:
            fallback_query = SearchQuery(
                query=f"class {class_name}",
                top_k=top_k,
                score_threshold=0.0,
                filters=None,
            )
            fallback_result = self.search(fallback_query)
            # Ưu tiên chunk có package giống service đang xét nếu có
            if fallback_result.chunks:
                return SearchResult(
                    query=f"class {class_name} (fallback)",
                    chunks=fallback_result.chunks[:top_k],
                    total_found=len(fallback_result.chunks),
                    search_time_ms=fallback_result.search_time_ms,
                )
            # Nếu vẫn không có, thử semantic search rộng hơn
            query2 = SearchQuery(
                query=f"class {class_name} service implementation",
                top_k=top_k,
                score_threshold=0.5,
            )
            return self.search(query2)

        # If found and we want dependencies, search for them too
        if include_dependencies and result.chunks:
            main_chunk = result.chunks[0]
            all_chunks = [main_chunk]

            for dep in main_chunk.dependencies:
                dep_query = SearchQuery(
                    query=f"class {dep}",
                    top_k=1,
                    score_threshold=0.3,
                )
                dep_result = self.search(dep_query)
                all_chunks.extend(dep_result.chunks)

            return SearchResult(
                query=f"class {class_name} with dependencies",
                chunks=all_chunks[:top_k],
                total_found=len(all_chunks),
                search_time_ms=result.search_time_ms,
            )

        return result

    def search_by_layer(
        self, layer: str, query_text: str, top_k: int = 10
    ) -> SearchResult:
        """Search within a specific DDD layer."""
        query = SearchQuery(
            query=query_text,
            top_k=top_k,
            score_threshold=0.5,
            filters=MetadataFilter(layer=layer),
        )
        return self.search(query)

    def search_by_type(
        self, type_name: str, query_text: str, top_k: int = 10
    ) -> SearchResult:
        """Search for specific type (service, entity, repository, method)."""
        query = SearchQuery(
            query=query_text,
            top_k=top_k,
            score_threshold=0.5,
            filters=MetadataFilter(type=type_name),
        )
        return self.search(query)

    def get_related_code(
        self, class_name: str, context: str, top_k: int = 10
    ) -> SearchResult:
        """Get code related to a class based on context."""
        query = SearchQuery(
            query=f"{class_name} {context}",
            top_k=top_k,
            score_threshold=0.4,
        )
        return self.search(query)

    def _build_filter(
        self, filters: Optional[MetadataFilter]
    ) -> Optional[models.Filter]:
        """Build Qdrant filter from MetadataFilter."""
        if not filters:
            return None

        conditions = []

        if filters.type:
            conditions.append(
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value=filters.type),
                )
            )

        if filters.layer:
            conditions.append(
                models.FieldCondition(
                    key="layer",
                    match=models.MatchValue(value=filters.layer),
                )
            )

        if filters.class_name:
            conditions.append(
                models.FieldCondition(
                    key="class_name",
                    match=models.MatchValue(value=filters.class_name),
                )
            )

        if filters.package:
            conditions.append(
                models.FieldCondition(
                    key="package",
                    match=models.MatchValue(value=filters.package),
                )
            )

        if filters.package_prefix:
            conditions.append(
                models.FieldCondition(
                    key="package",
                    match=models.MatchText(text=filters.package_prefix),
                )
            )

        # Inheritance filters
        if filters.implements:
            conditions.append(
                models.FieldCondition(
                    key="implements",
                    match=models.MatchValue(value=filters.implements),
                )
            )

        if filters.extends:
            conditions.append(
                models.FieldCondition(
                    key="extends",
                    match=models.MatchValue(value=filters.extends),
                )
            )

        # Lombok filters
        if filters.has_builder is not None:
            conditions.append(
                models.FieldCondition(
                    key="has_builder",
                    match=models.MatchValue(value=filters.has_builder),
                )
            )

        if filters.has_data is not None:
            conditions.append(
                models.FieldCondition(
                    key="has_data",
                    match=models.MatchValue(value=filters.has_data),
                )
            )

        if filters.has_getter is not None:
            conditions.append(
                models.FieldCondition(
                    key="has_getter",
                    match=models.MatchValue(value=filters.has_getter),
                )
            )

        if filters.has_setter is not None:
            conditions.append(
                models.FieldCondition(
                    key="has_setter",
                    match=models.MatchValue(value=filters.has_setter),
                )
            )

        if filters.is_immutable is not None:
            conditions.append(
                models.FieldCondition(
                    key="is_immutable",
                    match=models.MatchValue(value=filters.is_immutable),
                )
            )

        if filters.has_no_args_constructor is not None:
            conditions.append(
                models.FieldCondition(
                    key="has_no_args_constructor",
                    match=models.MatchValue(value=filters.has_no_args_constructor),
                )
            )

        if filters.has_all_args_constructor is not None:
            conditions.append(
                models.FieldCondition(
                    key="has_all_args_constructor",
                    match=models.MatchValue(value=filters.has_all_args_constructor),
                )
            )

        if filters.uses_type:
            # Match classes that have the given type anywhere in their used_types list
            conditions.append(
                models.FieldCondition(
                    key="used_types",
                    match=models.MatchValue(value=filters.uses_type),
                )
            )

        if not conditions:
            return None

        return models.Filter(must=conditions)

    def get_stats(self) -> IndexStats:
        """Get statistics about the index."""
        try:
            info = self.qdrant.get_collection(self.collection_name)

            # Get type distribution
            type_counts = {}
            for type_name in ["service", "entity", "repository", "method"]:
                count_result = self.qdrant.count(
                    collection_name=self.collection_name,
                    count_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="type",
                                match=models.MatchValue(value=type_name),
                            )
                        ]
                    ),
                )
                type_counts[type_name] = count_result.count

            # Get layer distribution
            layer_counts = {}
            for layer in ["application", "domain", "infrastructure", "unknown"]:
                count_result = self.qdrant.count(
                    collection_name=self.collection_name,
                    count_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="layer",
                                match=models.MatchValue(value=layer),
                            )
                        ]
                    ),
                )
                layer_counts[layer] = count_result.count

            return IndexStats(
                collection_name=self.collection_name,
                total_vectors=info.vectors_count or 0,
                total_points=info.points_count or 0,
                status=info.status.value,
                type_distribution=type_counts,
                layer_distribution=layer_counts,
            )
        except Exception as e:
            logger.error("Failed to get stats", error=str(e))
            return IndexStats(
                collection_name=self.collection_name,
                total_vectors=0,
                total_points=0,
                status="error",
            )

