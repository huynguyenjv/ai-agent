"""
Schema definitions for RAG operations.
"""

from typing import Optional
from pydantic import BaseModel, Field


class FieldSchema(BaseModel):
    """Field information in a class."""
    type: str
    name: str
    annotations: list[str] = Field(default_factory=list)


class CodeChunk(BaseModel):
    """Represents a code chunk retrieved from the index."""

    id: int
    summary: str
    score: float
    # type: service | entity | method | repository | record | enum | interface | annotation | controller | adapter | gateway
    type: str
    layer: str  # application | domain | infrastructure
    class_name: str
    package: str
    file_path: str
    fully_qualified_name: str
    dependencies: list[str] = Field(default_factory=list)
    annotations: list[str] = Field(default_factory=list)
    # used_types: domain types used via fields, method params, return types
    # e.g. OrderService uses [OrderRequest, OrderEntity, OrderResponse]
    used_types: list[str] = Field(default_factory=list)
    # element_type: class | record | interface | enum | annotation | method
    element_type: str = "class"
    # java_type: class | record | interface | enum | annotation (raw Java type)
    java_type: Optional[str] = None
    method_name: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    # Inheritance / interfaces
    extends: Optional[str] = None
    implements: list[str] = Field(default_factory=list)
    # Class meta
    method_count: int = 0
    modifiers: list[str] = Field(default_factory=list)
    # Lombok info
    has_builder: bool = False
    has_builder_to_builder: bool = False
    has_data: bool = False
    has_getter: bool = False
    has_setter: bool = False
    has_value: bool = False
    is_immutable: bool = False
    has_no_args_constructor: bool = False
    has_all_args_constructor: bool = False
    has_required_args_constructor: bool = False
    # Fields info
    field_count: int = 0
    fields: list[FieldSchema] = Field(default_factory=list)


class MetadataFilter(BaseModel):
    """Filter criteria for metadata-based search."""

    # type: service | entity | method | repository | record | enum | interface | annotation
    type: Optional[str] = None
    # java_type: class | record | interface | enum | annotation
    java_type: Optional[str] = None
    layer: Optional[str] = None  # application | domain | infrastructure
    class_name: Optional[str] = None
    package: Optional[str] = None
    package_prefix: Optional[str] = None  # For package hierarchy filtering
    # Inheritance
    implements: Optional[str] = None   # Filter by implemented interface name
    extends: Optional[str] = None     # Filter by superclass name
    # Lombok filters
    has_builder: Optional[bool] = None
    has_data: Optional[bool] = None
    has_getter: Optional[bool] = None
    has_setter: Optional[bool] = None
    is_immutable: Optional[bool] = None
    has_no_args_constructor: Optional[bool] = None
    has_all_args_constructor: Optional[bool] = None
    # Filter by domain type usage (e.g. find all classes that use OrderRequest)
    uses_type: Optional[str] = None


class SearchQuery(BaseModel):
    """Query parameters for RAG search."""

    query: str = Field(..., description="Semantic search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    score_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum similarity score"
    )
    filters: Optional[MetadataFilter] = None


class SearchResult(BaseModel):
    """Result of a RAG search operation."""

    query: str
    chunks: list[CodeChunk]
    total_found: int
    search_time_ms: float


class IndexStats(BaseModel):
    """Statistics about the vector index."""

    collection_name: str
    total_vectors: int
    total_points: int
    status: str
    type_distribution: dict[str, int] = Field(default_factory=dict)
    layer_distribution: dict[str, int] = Field(default_factory=dict)
