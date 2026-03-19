"""
Pydantic request/response models for the AI Agent API.

Extracted from api.py for modularity.
"""

from typing import Optional, Literal

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# OpenAI-Compatible Models (for Tabby / Continue integration)
# ============================================================================

# ── Tool-calling types (OpenAI function-calling protocol) ────────────

class FunctionDefinition(BaseModel):
    """Function schema for tool calling."""
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None  # JSON Schema


class ToolDefinition(BaseModel):
    """Tool wrapper (OpenAI format)."""
    type: str = "function"
    function: FunctionDefinition


class FunctionCall(BaseModel):
    """A function call made by the model."""
    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """Tool call in assistant message."""
    id: str
    type: str = "function"
    function: FunctionCall


# ── Chat messages & request/response ────────────────────────────────

class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str | list] = None
    tool_calls: Optional[list[ToolCall]] = None    # assistant → tool calls
    tool_call_id: Optional[str] = None             # tool → result
    name: Optional[str] = None                     # tool function name

    @field_validator("content", mode="before")
    @classmethod
    def _coerce_content(cls, v):
        """OpenAI allows content as list of parts — flatten to str."""
        if isinstance(v, list):
            texts = []
            for part in v:
                if isinstance(part, str):
                    texts.append(part)
                elif isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
            return "\n".join(texts)
        return v


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = "ai-agent"
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False
    # Tool calling (Continue IDE sends these)
    tools: Optional[list[ToolDefinition]] = None
    tool_choice: Optional[str | dict] = None  # "auto" | "none" | {"type":"function","function":{"name":...}}
    # Custom fields for RAG
    file_path: Optional[str] = None
    workspace_path: Optional[str] = None
    # Multi-collection: explicit Qdrant collection name
    # (set via Continue's extraBodyProperties or sent by CI/CD pipelines)
    collection: Optional[str] = None
    # Two-Phase Strategy options (set via Continue's extraBodyProperties)
    force_two_phase: Optional[bool] = False
    force_single_pass: Optional[bool] = False
    complexity_threshold: Optional[int] = 10
    
    @field_validator("force_two_phase", "force_single_pass", mode="before")
    @classmethod
    def _coerce_bool(cls, v):
        """Handle malformed booleans like 'true,' from IDE config."""
        if isinstance(v, str):
            v = v.strip().rstrip(",").strip().lower()
            return v in ("true", "1", "yes")
        return v

    @field_validator("complexity_threshold", mode="before")
    @classmethod
    def _coerce_int(cls, v):
        """Handle malformed ints like '10,' from IDE config."""
        if isinstance(v, str):
            v = v.strip().rstrip(",").strip()
            return int(v) if v else None
        return v


class ChatCompletionChoice(BaseModel):
    """OpenAI-compatible chat completion choice."""
    index: int = 0
    message: Optional[ChatMessage] = None
    delta: Optional[dict] = None  # For streaming (loose dict for flexibility)
    finish_reason: Optional[str] = "stop"


class ChatCompletionUsage(BaseModel):
    """OpenAI-compatible usage info."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Optional[ChatCompletionUsage] = None


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "ai-agent"


class ModelsResponse(BaseModel):
    """List models response."""
    object: str = "list"
    data: list[ModelInfo]


# ============================================================================
# Original Request/Response Models
# ============================================================================

class GenerateTestRequest(BaseModel):
    """Request model for test generation.

    Example::

        {
            "file_path": "C:\\path\\to\\MyService.java",
            "task_description": "Generate comprehensive unit tests covering all public methods"
        }
    """

    file_path: str = Field(..., description="Path to the Java source file")
    task_description: Optional[str] = Field(
        "Generate comprehensive unit tests covering all public methods",
        description="What to generate / additional task description",
    )


class GenerateTestResponse(BaseModel):
    """Response model for test generation."""

    success: bool
    test_code: Optional[str] = None
    class_name: str = ""
    session_id: Optional[str] = None
    validation_passed: bool = True
    validation_issues: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    rag_chunks_used: int = 0
    tokens_used: int = 0


class RefineTestRequest(BaseModel):
    """Request model for test refinement."""

    session_id: str = Field(..., description="Session ID from previous generation")
    feedback: str = Field(..., description="Feedback for refinement")


class ReindexRequest(BaseModel):
    """Request model for reindexing."""

    repo_path: str = Field(..., description="Path to the Java repository")
    recreate: bool = Field(False, description="Whether to recreate the collection")
    collection: Optional[str] = Field(
        None,
        description=(
            "Qdrant collection name. If omitted, auto-derived from repo folder name "
            "(e.g. 'vtrip.core.iam' → 'vtrip_core_iam')."
        ),
    )   


class ReindexResponse(BaseModel):
    """Response model for reindexing."""

    success: bool
    message: str
    collection: str = ""
    points_indexed: int = 0
    error: Optional[str] = None


class IndexFileRequest(BaseModel):
    """Request to index a single Java file into Qdrant."""

    file_path: str = Field(..., description="Original file path (stored as metadata)")
    content: str = Field(..., description="Java source code content")
    collection: Optional[str] = Field(
        None,
        description="Qdrant collection name. Defaults to QDRANT_COLLECTION env.",
    )


class IndexFileResponse(BaseModel):
    """Response from single-file indexing."""

    success: bool
    file_path: str = ""
    collection: str = ""
    classes_indexed: int = 0
    points_created: int = 0
    error: Optional[str] = None


# ── OpenAI-Compatible Embeddings ─────────────────────────────────────

class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""

    input: str | list[str] = Field(..., description="Text(s) to embed")
    model: str = Field("all-MiniLM-L6-v2-onnx", description="Model name (informational)")
    encoding_format: Optional[str] = Field("float", description="'float' or 'base64'")


class EmbeddingObject(BaseModel):
    """Single embedding in the response."""

    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingUsage(BaseModel):
    """Token usage info for embeddings."""

    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response."""

    object: str = "list"
    data: list[EmbeddingObject]
    model: str
    usage: EmbeddingUsage


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    vllm_healthy: bool
    qdrant_healthy: bool
    redis_healthy: bool = False
    cache_backend: str = "memory"
    index_stats: Optional[dict] = None


# ============================================================================
# Pipeline API Models
# ============================================================================

class PipelineGenerateRequest(BaseModel):
    """Request for pipeline-driven test generation.

    The CI script handles:
      - git diff to find changed files
      - detecting existing test files
      - reading source code from disk

    The agent only receives explicit inputs — no auto-detection.
    """

    file_path: str = Field(..., description="Path to the Java source file")
    class_name: Optional[str] = Field(
        None, description="Class name (auto-extracted from file_path if absent)"
    )
    task_description: Optional[str] = Field(
        "Generate comprehensive unit tests covering all public methods",
        description="What to generate",
    )
    mode: Literal["full", "incremental"] = Field(
        "full",
        description=(
            "'full' = generate complete test class from scratch; "
            "'incremental' = add tests for new/changed methods only"
        ),
    )
    existing_test_code: Optional[str] = Field(
        None,
        description="Content of the existing test file (required for mode='incremental')",
    )
    changed_methods: Optional[list[str]] = Field(
        None,
        description="List of changed/added method names (optional, for incremental mode)",
    )
    collection: Optional[str] = Field(
        None,
        description=(
            "Qdrant collection name. If omitted, auto-resolved from file_path "
            "via the registry, or falls back to default."
        ),
    )
    source_code: Optional[str] = Field(
        None,
        description=(
            "Full Java source code of the class. When provided the LLM sees "
            "the real source instead of only the RAG summary, significantly "
            "improving test quality. CI script should read the file and send "
            "its content here."
        ),
    )
    # Two-Phase Strategy options
    force_two_phase: bool = Field(
        False,
        description="Force two-phase generation even for simple services",
    )
    force_single_pass: bool = Field(
        False,
        description="Force single-pass generation even for complex services",
    )
    complexity_threshold: int = Field(
        10,
        description="Complexity threshold for auto-routing to two-phase",
    )


class PipelineBatchRequest(BaseModel):
    """Batch request for generating tests for multiple files."""

    files: list[PipelineGenerateRequest] = Field(
        ..., description="List of files to generate tests for", min_length=1
    )


class PipelineBatchItemResult(BaseModel):
    """Result for a single file in a batch."""

    file_path: str
    class_name: str = ""
    success: bool
    test_code: Optional[str] = None
    mode: str = "full"
    validation_passed: bool = True
    validation_issues: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    tokens_used: int = 0
    repair_attempts: int = 0


class PipelineGenerateResponse(BaseModel):
    """Response from single test generation."""

    success: bool
    test_code: str | None = None
    class_name: str | None = None
    file_path: str
    mode: str = "full"
    collection: str = ""
    validation_passed: bool = False
    validation_issues: list[str] = []
    error: str | None = None
    rag_chunks_used: int = 0
    tokens_used: int = 0
    repair_attempts: int = 0
    # Two-Phase Strategy metadata
    strategy_used: str = "single_pass"
    complexity_score: int = 0
    analysis_result: dict | None = None


class PipelineBatchResponse(BaseModel):
    """Response from batch test generation."""

    total: int
    succeeded: int
    failed: int
    results: list[PipelineBatchItemResult]


# ============================================================================
# LangGraph Models
# ============================================================================

class ReviewRequest(BaseModel):
    """Request to approve/reject a generated test."""
    approved: bool
    feedback: str = ""


class RunStatusResponse(BaseModel):
    """Run status for polling."""
    run_id: str
    status: str  # "running" | "interrupted" | "completed" | "failed"
    class_name: str = ""
    validation_passed: Optional[bool] = None
    test_code: Optional[str] = None
    validation_issues: list[str] = Field(default_factory=list)
