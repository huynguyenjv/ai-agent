"""
RAG client for querying the Qdrant vector index.
"""

import os
import time
from typing import Optional

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models

from utils.embedding import ONNXEmbedder
from .schema import CodeChunk, SearchQuery, SearchResult, MetadataFilter, IndexStats, FieldSchema

logger = structlog.get_logger()


class RAGClient:
    """Client for RAG operations on the Java codebase index."""

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
        logger.info(
            "RAG client initialized",
            host=qdrant_host,
            port=qdrant_port,
            collection=collection_name,
        )

    def search(self, query: SearchQuery, collection_name: Optional[str] = None) -> SearchResult:
        """Perform semantic search with optional metadata filtering.

        Args:
            query: Search query parameters.
            collection_name: Override collection to search in. Defaults to self.collection_name.
        """
        start_time = time.time()
        effective_collection = collection_name or self.collection_name

        # Generate query embedding
        query_vector = self.embedder.encode(query.query).tolist()

        # Build filter conditions
        filter_conditions = self._build_filter(query.filters)

        # Execute search
        results = self.qdrant.search(
            collection_name=effective_collection,
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
            collection=effective_collection,
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
        self, class_name: str, top_k: int = 10, include_dependencies: bool = True,
        collection_name: Optional[str] = None,
    ) -> SearchResult:
        """Search for a specific class and optionally its dependencies.

        Uses a three-tier lookup strategy:
          1. Semantic search WITH class_name metadata filter (fast + accurate)
          2. Qdrant scroll — pure metadata lookup, no embedding (guaranteed find)
          3. Broad semantic search without filter (last resort)

        Args:
            collection_name: Override collection to search in.
        """
        start_time = time.time()
        effective_collection = collection_name or self.collection_name

        # ── Tier 1: semantic search + metadata filter ──
        query = SearchQuery(
            query=f"class {class_name}",
            top_k=1,
            score_threshold=0.3,
            filters=MetadataFilter(class_name=class_name),
        )
        result = self.search(query, collection_name=effective_collection)

        # ── Tier 2: metadata-only scroll (guaranteed if class is indexed) ──
        if not result.chunks:
            logger.debug("Tier-1 miss, trying metadata scroll", class_name=class_name)
            chunk = self._scroll_by_class_name(class_name, collection_name=effective_collection)
            if chunk:
                result = SearchResult(
                    query=f"class {class_name} (scroll)",
                    chunks=[chunk],
                    total_found=1,
                    search_time_ms=(time.time() - start_time) * 1000,
                )

        # ── Tier 3: broad semantic search (no filter) ──
        if not result.chunks:
            logger.debug("Tier-2 miss, trying broad semantic", class_name=class_name)
            fallback_query = SearchQuery(
                query=f"class {class_name}",
                top_k=top_k,
                score_threshold=0.0,
                filters=None,
            )
            fallback_result = self.search(fallback_query, collection_name=effective_collection)
            # Filter to prefer chunks whose class_name actually matches
            matching = [c for c in fallback_result.chunks if c.class_name == class_name]
            if matching:
                result = SearchResult(
                    query=f"class {class_name} (broad-match)",
                    chunks=matching[:1],
                    total_found=len(matching),
                    search_time_ms=fallback_result.search_time_ms,
                )
            elif fallback_result.chunks:
                result = SearchResult(
                    query=f"class {class_name} (broad-fallback)",
                    chunks=fallback_result.chunks[:top_k],
                    total_found=len(fallback_result.chunks),
                    search_time_ms=fallback_result.search_time_ms,
                )

        if not result.chunks:
            logger.warning("Class not found after all tiers", class_name=class_name)
            return result

        # If found and we want dependencies, search for them too
        if include_dependencies and result.chunks:
            main_chunk = result.chunks[0]
            all_chunks = [main_chunk]

            # Collect simple names from dependencies (FQNs) + used_types (simple names)
            types_to_fetch: set[str] = set()
            for dep_fqn in (main_chunk.dependencies or []):
                simple = dep_fqn.rsplit(".", 1)[-1] if "." in dep_fqn else dep_fqn
                types_to_fetch.add(simple)
            for ut in (main_chunk.used_types or []):
                types_to_fetch.add(ut)
            # Remove self
            types_to_fetch -= {main_chunk.class_name}

            unfound_types: set[str] = set()
            for type_name in types_to_fetch:
                dep_query = SearchQuery(
                    query=f"class {type_name}",
                    top_k=1,
                    score_threshold=0.3,
                )
                dep_result = self.search(dep_query, collection_name=effective_collection)
                if not dep_result.chunks:
                    # Tier 2 fallback for deps too
                    dep_chunk = self._scroll_by_class_name(type_name, collection_name=effective_collection)
                    if dep_chunk:
                        all_chunks.append(dep_chunk)
                    else:
                        unfound_types.add(type_name)
                        logger.warning(
                            "Dependency type not found in index",
                            type_name=type_name,
                            parent_class=class_name,
                        )
                else:
                    all_chunks.extend(dep_result.chunks)

            # Attach unfound types to main_chunk so downstream can warn LLM
            if unfound_types:
                main_chunk.unfound_types = sorted(unfound_types)

            # Return ALL fetched chunks — do NOT truncate with [:top_k]
            # because domain types are critical for correct test generation.
            # The prompt builder handles token budgeting separately.
            return SearchResult(
                query=f"class {class_name} with dependencies",
                chunks=all_chunks,
                total_found=len(all_chunks),
                search_time_ms=result.search_time_ms,
            )

        return result

    def _scroll_by_class_name(
        self, class_name: str, prefer_type: str = "class",
        collection_name: Optional[str] = None,
    ) -> Optional[CodeChunk]:
        """Pure metadata lookup via Qdrant scroll — no embedding similarity needed.

        This guarantees finding a class if it's in the index, regardless of
        how well the embedding of 'class X' matches the summary embedding.
        Prefers class-level chunks over method-level ones.
        """
        try:
            effective_collection = collection_name or self.collection_name
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="class_name",
                        match=models.MatchValue(value=class_name),
                    )
                ]
            )
            points, _ = self.qdrant.scroll(
                collection_name=effective_collection,
                scroll_filter=scroll_filter,
                limit=10,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                return None

            # Prefer class-level chunks over method chunks
            class_points = [p for p in points if p.payload.get("element_type") == "class"]
            chosen = class_points[0] if class_points else points[0]
            payload = chosen.payload

            # Build CodeChunk
            fields_data = payload.get("fields", [])
            fields = [
                FieldSchema(type=f.get("type", ""), name=f.get("name", ""), annotations=f.get("annotations", []))
                for f in fields_data if isinstance(f, dict)
            ]
            record_components_data = payload.get("record_components", [])
            record_components = [
                FieldSchema(type=f.get("type", ""), name=f.get("name", ""), annotations=f.get("annotations", []))
                for f in record_components_data if isinstance(f, dict)
            ]

            chunk = CodeChunk(
                id=chosen.id,
                summary=payload.get("summary", ""),
                score=1.0,  # metadata-exact match
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
                extends=payload.get("extends"),
                implements=payload.get("implements", []),
                method_count=payload.get("method_count", 0),
                modifiers=payload.get("modifiers", []),
                used_types=payload.get("used_types", []),
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
                field_count=payload.get("field_count", 0),
                fields=fields,
                record_components=record_components,
            )
            logger.info(
                "Found class via metadata scroll",
                class_name=class_name,
                fqn=chunk.fully_qualified_name,
                element_type=chunk.element_type,
            )
            return chunk
        except Exception as e:
            logger.warning("Scroll lookup failed", class_name=class_name, error=str(e))
            return None

    def search_by_layer(
        self, layer: str, query_text: str, top_k: int = 10,
        collection_name: Optional[str] = None,
    ) -> SearchResult:
        """Search within a specific DDD layer."""
        query = SearchQuery(
            query=query_text,
            top_k=top_k,
            score_threshold=0.5,
            filters=MetadataFilter(layer=layer),
        )
        return self.search(query, collection_name=collection_name)

    def search_by_type(
        self, type_name: str, query_text: str, top_k: int = 10,
        collection_name: Optional[str] = None,
    ) -> SearchResult:
        """Search for specific type (service, entity, repository, method)."""
        query = SearchQuery(
            query=query_text,
            top_k=top_k,
            score_threshold=0.5,
            filters=MetadataFilter(type=type_name),
        )
        return self.search(query, collection_name=collection_name)

    def get_related_code(
        self, class_name: str, context: str, top_k: int = 10,
        collection_name: Optional[str] = None,
    ) -> SearchResult:
        """Get code related to a class based on context."""
        query = SearchQuery(
            query=f"{class_name} {context}",
            top_k=top_k,
            score_threshold=0.4,
        )
        return self.search(query, collection_name=collection_name)

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

    def get_stats(self, collection_name: Optional[str] = None) -> IndexStats:
        """Get statistics about the index."""
        effective_collection = collection_name or self.collection_name
        try:
            info = self.qdrant.get_collection(effective_collection)

            # Get type distribution
            type_counts = {}
            for type_name in ["service", "entity", "repository", "method"]:
                count_result = self.qdrant.count(
                    collection_name=effective_collection,
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
                    collection_name=effective_collection,
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
                collection_name=effective_collection,
                total_vectors=info.vectors_count or 0,
                total_points=info.points_count or 0,
                status=info.status.value,
                type_distribution=type_counts,
                layer_distribution=layer_counts,
            )
        except Exception as e:
            logger.error("Failed to get stats", error=str(e))
            return IndexStats(
                collection_name=effective_collection,
                total_vectors=0,
                total_points=0,
                status="error",
            )

    def list_collections(self) -> list[str]:
        """List all collection names in Qdrant."""
        try:
            collections = self.qdrant.get_collections().collections
            return [c.name for c in collections]
        except Exception as e:
            logger.error("Failed to list collections", error=str(e))
            return []

