"""
FastAPI server for the AI coding agent.
Compatible with Tabby IDE via OpenAI-compatible API.
"""

import asyncio
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional, Literal

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from agent.orchestrator import AgentOrchestrator, GenerationRequest, StreamEvent, StreamPhase
from agent.graph_adapter import GraphOrchestrator, create_graph_orchestrator
from agent.events import get_event_bus, Event, EventType
from agent.metrics import MetricsCollector
from indexer.build_index import IndexBuilder
from rag.client import RAGClient
from rag.schema import SearchQuery
from vllm.client import VLLMClient
from utils.rate_limiter import RateLimiter
from utils.tokenizer import count_tokens
from .session import SessionManager, SessionInfo

# Load environment variables
load_dotenv()

logger = structlog.get_logger()

# Global instances
orchestrator: Optional[AgentOrchestrator] = None
graph_orchestrator: Optional[GraphOrchestrator] = None  # LangGraph backend
index_builder: Optional[IndexBuilder] = None
metrics_collector: Optional[MetricsCollector] = None
session_manager: SessionManager = SessionManager()
rate_limiter: Optional[RateLimiter] = None

# Toggle: set USE_LEGACY_ORCHESTRATOR=true to use the old orchestrator
USE_LEGACY = os.getenv("USE_LEGACY_ORCHESTRATOR", "false").lower() in ("true", "1", "yes")

# Thread pool for running blocking operations
# Use limited workers to prevent resource exhaustion
_executor: Optional[ThreadPoolExecutor] = None
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))

# Rate limiting configuration
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))  # requests per window
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # window in seconds
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))  # max request time in seconds


# ============================================================================
# OpenAI-Compatible Models (for Tabby integration)
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


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    vllm_healthy: bool
    qdrant_healthy: bool
    redis_healthy: bool = False
    cache_backend: str = "memory"
    index_stats: Optional[dict] = None


def create_orchestrator() -> AgentOrchestrator:
    """Create and configure the agent orchestrator."""
    rag_client = RAGClient(
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        collection_name=os.getenv("QDRANT_COLLECTION", "java_codebase"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2-onnx"),
    )

    vllm_base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    vllm_model = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ")
    
    logger.info(
        "Creating vLLM client",
        base_url=vllm_base_url,
        model=vllm_model,
    )
    
    vllm_client = VLLMClient(
        base_url=vllm_base_url,
        api_key=os.getenv("VLLM_API_KEY", "token-abc123"),
        model=vllm_model,
    )

    return AgentOrchestrator(
        rag_client=rag_client,
        vllm_client=vllm_client,
        repo_path=os.getenv("JAVA_REPO_PATH") or None,
    )


def create_index_builder() -> IndexBuilder:
    """Create and configure the index builder."""
    return IndexBuilder(
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        collection_name=os.getenv("QDRANT_COLLECTION", "java_codebase"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2-onnx"),
    )


# ── Collection helpers ──────────────────────────────────────────────

import re as _re


def _derive_collection_name(repo_path: str) -> str:
    """Derive a Qdrant-safe collection name from a repository path.

    Examples:
        C:\\Users\\...\\vtrip.core.iam      → vtrip_core_iam
        /home/ci/repos/my-awesome-service → my_awesome_service
    """
    folder = os.path.basename(os.path.normpath(repo_path))
    # Replace dots, hyphens, spaces with underscores; lowercase; strip non-alnum
    safe = _re.sub(r"[^a-zA-Z0-9_]", "_", folder).strip("_").lower()
    return safe or "java_codebase"


def _resolve_collection(
    explicit: Optional[str] = None,
    file_path: Optional[str] = None,
    workspace_path: Optional[str] = None,
) -> str:
    """Resolve which Qdrant collection to use (3-tier fallback).

    1. Explicit collection name from request (Continue extraBodyProperties / CI param)
    2. Auto-detect from file_path via Redis registry
    3. Default from env var
    """
    # Tier 1: explicit
    if explicit:
        return explicit

    # Tier 2: auto-detect from file_path (or workspace_path) via registry
    lookup_path = file_path or workspace_path
    if lookup_path:
        from utils.cache_service import get_cache_service
        cache = get_cache_service()
        matched = cache.resolve_collection_by_path(lookup_path)
        if matched:
            return matched

    # Tier 3: default
    return os.getenv("QDRANT_COLLECTION", "java_codebase")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global orchestrator, graph_orchestrator, index_builder, _executor, metrics_collector, rate_limiter

    logger.info("Starting AI Agent server...", backend="legacy" if USE_LEGACY else "langgraph")

    # Initialize thread pool for blocking operations
    _executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    logger.info("Thread pool initialized", max_workers=MAX_WORKERS)

    # Initialize rate limiter
    rate_limiter = RateLimiter(
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        redis_port=int(os.getenv("REDIS_PORT", "6379")),
        redis_password=os.getenv("REDIS_PASSWORD") or None,
        redis_db=int(os.getenv("REDIS_DB", "0")),
        requests_per_window=RATE_LIMIT_REQUESTS,
        window_seconds=RATE_LIMIT_WINDOW,
    )

    # Initialize components — legacy orchestrator always created for fallback
    orchestrator = create_orchestrator()
    index_builder = create_index_builder()

    # Initialize LangGraph orchestrator (new backend)
    if not USE_LEGACY:
        try:
            graph_orchestrator = create_graph_orchestrator(
                rag_client=orchestrator.rag,
                vllm_client=orchestrator.vllm,
            )
            logger.info("LangGraph orchestrator initialized")
        except Exception as e:
            logger.error("LangGraph init failed, using legacy", error=str(e))
            graph_orchestrator = None

    # Phase 4: metrics collector is wired via the orchestrator's event bus
    metrics_collector = orchestrator.metrics if orchestrator else None

    # Start background task for session cleanup
    cleanup_task = asyncio.create_task(_periodic_session_cleanup())

    logger.info("Server initialized successfully", backend="langgraph" if graph_orchestrator else "legacy")

    yield

    # Cleanup
    logger.info("Shutting down server...")
    
    # Cancel cleanup task
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    # Close vLLM client connection
    if orchestrator and orchestrator.vllm:
        orchestrator.vllm.close()
        logger.info("vLLM client closed")

    # Close Qdrant connections
    if orchestrator and orchestrator.rag:
        try:
            orchestrator.rag.qdrant.close()
            logger.info("Qdrant client (RAG) closed")
        except Exception as e:
            logger.warning("Failed to close RAG Qdrant client", error=str(e))
    if index_builder:
        try:
            index_builder.qdrant.close()
            logger.info("Qdrant client (indexer) closed")
        except Exception as e:
            logger.warning("Failed to close indexer Qdrant client", error=str(e))

    # Close Redis connections
    if rate_limiter and rate_limiter.redis_client:
        try:
            rate_limiter.redis_client.close()
            logger.info("Redis client (rate limiter) closed")
        except Exception as e:
            logger.warning("Failed to close rate limiter Redis", error=str(e))

    # Shutdown thread pool
    if _executor:
        _executor.shutdown(wait=True, cancel_futures=True)
        logger.info("Thread pool shutdown complete")


async def _periodic_session_cleanup():
    """Background task to cleanup expired sessions periodically."""
    while True:
        try:
            await asyncio.sleep(300)  # Every 5 minutes
            loop = asyncio.get_running_loop()
            count = await loop.run_in_executor(_executor, session_manager.cleanup_expired)
            if count > 0:
                logger.info("Cleaned up expired sessions", count=count)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Session cleanup failed", error=str(e))


# Track active executor tasks for observability
_active_tasks: int = 0
_active_tasks_lock = __import__("threading").Lock()


async def run_in_executor(func, *args):
    """Run a blocking function in the thread pool with timeout."""
    global _active_tasks
    loop = asyncio.get_running_loop()

    with _active_tasks_lock:
        _active_tasks += 1
        current = _active_tasks

    logger.debug(
        "Submitting task to executor",
        func=func.__name__,
        active_tasks=current,
        max_workers=MAX_WORKERS,
    )

    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_executor, func, *args),
            timeout=REQUEST_TIMEOUT
        )
    except asyncio.TimeoutError:
        logger.error(
            "Request timed out",
            func=func.__name__,
            timeout=REQUEST_TIMEOUT,
            active_tasks=_active_tasks,
        )
        raise HTTPException(
            status_code=504,
            detail=f"Request timed out after {REQUEST_TIMEOUT} seconds"
        )
    finally:
        with _active_tasks_lock:
            _active_tasks -= 1


def check_rate_limit(client_ip: str) -> tuple[bool, int]:
    """Check if client has exceeded rate limit.

    Returns:
        (allowed: bool, remaining: int) - Whether request is allowed and remaining requests

    NOTE: This is a sync function that may do Redis I/O.
    Must be called via run_in_executor() from async context.
    """
    if not rate_limiter:
        return True, RATE_LIMIT_REQUESTS  # Allow if rate limiter not initialized
    
    allowed = rate_limiter.check_rate_limit(client_ip)
    remaining = rate_limiter.get_remaining_requests(client_ip)
    
    return allowed, remaining


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AI Coding Agent",
        description="Self-hosted AI agent for generating JUnit5 + Mockito unit tests",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware - configurable for production
    # In production, set CORS_ORIGINS env var to comma-separated list of allowed origins
    cors_origins_str = os.getenv("CORS_ORIGINS", "*")
    if cors_origins_str == "*":
        # Development mode - allow all
        allow_origins = ["*"]
        allow_credentials = False  # Cannot use credentials with wildcard origin
    else:
        # Production mode - specific origins only
        allow_origins = [origin.strip() for origin in cors_origins_str.split(",")]
        allow_credentials = True

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    # Rate limiting middleware
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/v1/health", "/v1/models", "/metrics"]:
            return await call_next(request)
        
        # Get client IP (handle proxy headers)
        client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        if not client_ip:
            client_ip = request.client.host if request.client else "unknown"
        
        # Check rate limit — run in executor to avoid blocking event loop
        loop = asyncio.get_running_loop()
        allowed, remaining = await loop.run_in_executor(
            _executor, check_rate_limit, client_ip
        )
        if not allowed:
            logger.warning("Rate limit exceeded", client_ip=client_ip, remaining=remaining)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Max {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds.",
                    "retry_after": RATE_LIMIT_WINDOW
                },
                headers={
                    "Retry-After": str(RATE_LIMIT_WINDOW),
                    "X-RateLimit-Limit": str(RATE_LIMIT_REQUESTS),
                    "X-RateLimit-Remaining": str(remaining),
                    "X-RateLimit-Reset": str(int(time.time() + RATE_LIMIT_WINDOW)),
                }
            )
        
        response = await call_next(request)
        
        # Add rate limit headers to successful responses
        response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_REQUESTS)
        response.headers["X-RateLimit-Remaining"] = str(remaining - 1)  # Account for current request 
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + RATE_LIMIT_WINDOW))
        
        return response

    # ── 422 validation error handler — log details for debugging ─────
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.error(
            "Request validation failed (422)",
            path=request.url.path,
            errors=exc.errors(),
        )
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors()},
        )

    return app


app = create_app()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of all services (non-blocking)."""
    vllm_healthy = False
    qdrant_healthy = False
    index_stats = None

    if orchestrator:
        # Run sync health checks in thread pool to avoid blocking the event loop
        loop = asyncio.get_running_loop()

        vllm_healthy = await loop.run_in_executor(None, orchestrator.vllm.health_check)

        try:
            stats = await loop.run_in_executor(None, orchestrator.rag.get_stats)
            # Qdrant status có thể là "green", "yellow", hoặc tên khác tùy version
            qdrant_healthy = stats.status in ("green", "yellow", "Green", "Yellow") or stats.total_points >= 0
            index_stats = {
                "collection": stats.collection_name,
                "total_points": stats.total_points,
                "status": stats.status,
                "type_distribution": stats.type_distribution,
            }
        except Exception as e:
            logger.warning("Qdrant health check failed", error=str(e))
            # Try direct connection check
            try:
                collections = await loop.run_in_executor(
                    None, orchestrator.rag.qdrant.get_collections
                )
                qdrant_healthy = True
                index_stats = {"collection": "java_codebase", "total_points": 0, "status": "connected"}
            except Exception as e:
                logger.warning("Qdrant direct connection check failed", error=str(e))

    status = "healthy" if (vllm_healthy and qdrant_healthy) else "degraded"

    # Check Redis/cache health
    from utils.cache_service import get_cache_service
    cache = get_cache_service()
    redis_healthy = cache.is_redis_connected
    cache_backend = cache.backend_name

    return HealthResponse(
        status=status,
        vllm_healthy=vllm_healthy,
        qdrant_healthy=qdrant_healthy,
        redis_healthy=redis_healthy,
        cache_backend=cache_backend,
        index_stats=index_stats,
    )


@app.get("/cache/stats")
async def cache_stats():
    """Get cache service statistics."""
    from utils.cache_service import get_cache_service
    cache = get_cache_service()
    loop = asyncio.get_running_loop()
    stats = await loop.run_in_executor(_executor, cache.get_cache_stats)
    
    # Add rate limiter info
    if rate_limiter:
        stats["rate_limiter"] = {
            "backend": "redis" if rate_limiter.redis_client else "memory",
            "requests_per_window": rate_limiter.requests_per_window,
            "window_seconds": rate_limiter.window_seconds,
        }
    
    return stats


@app.post("/generate-test", response_model=GenerateTestResponse, deprecated=True)
async def generate_test(request: GenerateTestRequest):
    """Generate unit tests for a Java class.

    .. deprecated::
        Use ``POST /pipeline/generate`` instead for CI/CD integration.
        This endpoint is kept for backward compatibility.

    Request body::

        {
            "file_path": "C:\\\\path\\\\to\\\\MyService.java",
            "task_description": "Generate comprehensive unit tests covering all public methods"
        }

    Returns generated test code with validation results.
    """
    active = _get_orchestrator()
    if not active:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Read source file from disk and embed it as an inline code fence so the
    # LLM receives the actual Java source code. build_test_generation_prompt
    # already knows how to extract it from task_description.
    base_description = (
        request.task_description
        or "Generate comprehensive unit tests covering all public methods"
    )
    task_description_with_source = base_description
    if os.path.isfile(request.file_path):
        try:
            loop = asyncio.get_running_loop()
            source_code = await loop.run_in_executor(
                _executor, lambda: open(request.file_path, encoding="utf-8").read()
            )
            task_description_with_source = (
                f"{base_description}\n\n"
                f"```{request.file_path}\n{source_code}\n```"
            )
            logger.info(
                "Source code embedded in task_description",
                file_path=request.file_path,
                source_len=len(source_code),
            )
        except Exception as e:
            logger.warning(
                "Could not read source file, proceeding without inline source",
                file_path=request.file_path,
                error=str(e),
            )

    gen_request = GenerationRequest(
        file_path=request.file_path,
        task_description=task_description_with_source,
    )

    result = await run_in_executor(active.generate_test, gen_request)

    return GenerateTestResponse(
        success=result.success,
        test_code=result.test_code,
        class_name=result.class_name,
        validation_passed=result.validation_passed,
        validation_issues=result.validation_issues,
        error=result.error,
        rag_chunks_used=result.rag_chunks_used,
        tokens_used=result.tokens_used,
    )


@app.post("/refine-test", response_model=GenerateTestResponse)
async def refine_test(request: RefineTestRequest):
    """Refine a previously generated test based on feedback."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Verify session exists
    loop = asyncio.get_running_loop()
    session = await loop.run_in_executor(
        _executor, session_manager.get_session, request.session_id
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Run blocking operation in thread pool
    result = await run_in_executor(
        orchestrator.refine_test,
        request.session_id,
        request.feedback,
    )

    if result.success:
        await loop.run_in_executor(
            _executor,
            lambda: session_manager.update_session(
                session_id=request.session_id,
                increment_tests=True,
            ),
        )

    return GenerateTestResponse(
        success=result.success,
        test_code=result.test_code,
        class_name=result.class_name,
        validation_passed=result.validation_passed,
        validation_issues=result.validation_issues,
        error=result.error,
        session_id=request.session_id,
        tokens_used=result.tokens_used,
    )


@app.post("/reindex", response_model=ReindexResponse)
async def reindex(request: ReindexRequest, background_tasks: BackgroundTasks):
    """Reindex a Java repository into a dedicated Qdrant collection.

    If ``collection`` is not provided, it is auto-derived from the repo
    folder name (e.g. ``vtrip.core.iam`` → ``vtrip_core_iam``).
    The mapping is stored in Redis so ``chat/completions`` can resolve it
    automatically from ``file_path``.
    """
    if not index_builder:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Validate path exists
    if not os.path.isdir(request.repo_path):
        raise HTTPException(status_code=400, detail=f"Directory not found: {request.repo_path}")

    # Derive collection name
    collection_name = request.collection or _derive_collection_name(request.repo_path)

    try:
        # Run indexing in thread pool to avoid blocking event loop
        points_indexed = await run_in_executor(
            index_builder.index_repository,
            request.repo_path,
            request.recreate,
            collection_name,
        )

        # Register collection → repo_path mapping in Redis
        from utils.cache_service import get_cache_service
        cache = get_cache_service()
        cache.register_collection(collection_name, request.repo_path, points_indexed)

        return ReindexResponse(
            success=True,
            message=f"Successfully indexed {points_indexed} code elements into collection '{collection_name}'",
            collection=collection_name,
            points_indexed=points_indexed,
        )

    except Exception as e:
        logger.error("Reindexing failed", error=str(e), collection=collection_name)
        return ReindexResponse(
            success=False,
            message="Reindexing failed",
            error=str(e),
        )


@app.get("/index/stats")
async def get_index_stats(collection: Optional[str] = None):
    """Get statistics about the vector index.

    Query param ``collection`` lets you check a specific collection.
    If omitted, uses the default collection.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    stats = await run_in_executor(orchestrator.rag.get_stats, collection)
    return stats.model_dump()


@app.get("/collections")
async def list_collections():
    """List all indexed Qdrant collections with their metadata.

    Returns both Qdrant-side collections and the registry (repo_path mapping).
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Get collections from Qdrant
    qdrant_collections = await run_in_executor(orchestrator.rag.list_collections)

    # Get registry from Redis/memory
    from utils.cache_service import get_cache_service
    cache = get_cache_service()
    registry = cache.get_collection_registry()

    # Merge info
    result = []
    for coll_name in qdrant_collections:
        entry = {"name": coll_name}
        if coll_name in registry:
            entry.update(registry[coll_name])
        result.append(entry)

    # Add registry-only entries (indexed but collection might have been deleted)
    for coll_name, info in registry.items():
        if coll_name not in qdrant_collections:
            entry = {"name": coll_name, "status": "registry_only (no Qdrant collection)"}
            entry.update(info)
            result.append(entry)

    return {"collections": result, "total": len(result)}


@app.get("/index/lookup/{class_name}")
async def lookup_class(class_name: str, collection: Optional[str] = None):
    """Diagnostic: check if a class is indexed in Qdrant and what info is stored.

    Usage: GET /index/lookup/UserProfile?collection=vtrip_core_iam
    Returns the full payload stored in Qdrant for that class, including
    fields, record_components, usage_hint, used_types, has_builder, etc.
    Useful for debugging why the LLM generates wrong construction patterns.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    rag = orchestrator.rag

    # Tier 1: metadata scroll (guaranteed find)
    chunk = await run_in_executor(
        rag._scroll_by_class_name, class_name, "class", collection,
    )
    lookup_method = "scroll"

    # Tier 2: semantic search fallback
    if not chunk:
        result = await run_in_executor(
            rag.search_by_class, class_name, 1, False, collection,
        )
        chunk = result.chunks[0] if result.chunks else None
        lookup_method = "semantic"

    if not chunk:
        return {
            "found": False,
            "class_name": class_name,
            "message": f"'{class_name}' NOT found in Qdrant index. Re-index the codebase.",
        }

    return {
        "found": True,
        "lookup_method": lookup_method,
        "class_name": chunk.class_name,
        "fully_qualified_name": chunk.fully_qualified_name,
        "package": chunk.package,
        "element_type": chunk.element_type,
        "java_type": chunk.java_type,
        "is_record": chunk.java_type == "record",
        "has_builder": chunk.has_builder,
        "has_data": chunk.has_data,
        "has_value": chunk.has_value,
        "record_components": [
            {"type": rc.type, "name": rc.name} for rc in (chunk.record_components or [])
        ],
        "fields": [
            {"type": f.type, "name": f.name} for f in (chunk.fields or [])
        ],
        "used_types": chunk.used_types,
        "dependencies": chunk.dependencies,
        "annotations": chunk.annotations,
        "summary": chunk.summary,
    }


@app.post("/session", response_model=SessionInfo)
async def create_session():
    """Create a new session."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, session_manager.create_session)


@app.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get session information."""
    loop = asyncio.get_running_loop()
    session = await loop.run_in_executor(_executor, session_manager.get_session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    loop = asyncio.get_running_loop()
    deleted = await loop.run_in_executor(_executor, session_manager.delete_session, session_id)
    if deleted:
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/sessions", response_model=list[SessionInfo])
async def list_sessions():
    """List all active sessions."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, session_manager.list_sessions)


# ============================================================================
# Pipeline API — CI/CD Integration (GitLab, Jenkins, etc.)
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


@app.post("/pipeline/generate", response_model=PipelineGenerateResponse)
async def pipeline_generate(request: PipelineGenerateRequest):
    """Generate unit tests for a single Java class (CI/CD pipeline).

    Supports two modes:
      - ``full``: Generate a complete test class from scratch.
      - ``incremental``: Add tests for new/changed methods to an existing test file.
        Requires ``existing_test_code``.

    Example (full)::

        POST /pipeline/generate
        {
            "file_path": "src/main/java/com/example/UserService.java",
            "mode": "full"
        }

    Example (incremental)::

        POST /pipeline/generate
        {
            "file_path": "src/main/java/com/example/UserService.java",
            "mode": "incremental",
            "existing_test_code": "package com.example; ... existing test class ...",
            "changed_methods": ["createUser", "updateEmail"]
        }
    """
    active = _get_orchestrator()
    if not active:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Validate incremental mode requirements
    if request.mode == "incremental" and not request.existing_test_code:
        raise HTTPException(
            status_code=400,
            detail="existing_test_code is required when mode='incremental'",
        )

    # Read source file from disk and embed it as an inline code fence so the
    # LLM receives the actual Java source code. build_test_generation_prompt
    # already knows how to extract it from task_description (same format as
    # the Continue IDE sends via chat/completions).
    base_description = (
        request.task_description
        or "Generate comprehensive unit tests covering all public methods"
    )
    task_description_with_source = base_description
    if request.mode == "full" and os.path.isfile(request.file_path):
        try:
            loop = asyncio.get_running_loop()
            source_code = await loop.run_in_executor(
                _executor, lambda: open(request.file_path, encoding="utf-8").read()
            )
            task_description_with_source = (
                f"{base_description}\n\n"
                f"```{request.file_path}\n{source_code}\n```"
            )
            logger.info(
                "Source code embedded in task_description",
                file_path=request.file_path,
                source_len=len(source_code),
                mode=request.mode,
            )
        except Exception as e:
            logger.warning(
                "Could not read source file, proceeding without inline source",
                file_path=request.file_path,
                error=str(e),
            )
    elif request.mode == "incremental" and request.existing_test_code and os.path.isfile(request.file_path):
        # For incremental mode: also read source so the LLM can see what changed
        try:
            loop = asyncio.get_running_loop()
            source_code = await loop.run_in_executor(
                _executor, lambda: open(request.file_path, encoding="utf-8").read()
            )
            task_description_with_source = (
                f"{base_description}\n\n"
                f"```{request.file_path}\n{source_code}\n```"
            )
            logger.info(
                "Source code embedded in task_description (incremental)",
                file_path=request.file_path,
                source_len=len(source_code),
            )
        except Exception as e:
            logger.warning(
                "Could not read source file for incremental mode",
                file_path=request.file_path,
                error=str(e),
            )

    gen_request = GenerationRequest(
        file_path=request.file_path,
        class_name=request.class_name,
        task_description=task_description_with_source,
        existing_test_code=request.existing_test_code if request.mode == "incremental" else None,
        changed_methods=request.changed_methods if request.mode == "incremental" else None,
        collection_name=_resolve_collection(
            explicit=request.collection,
            file_path=request.file_path,
        ),
        source_code=request.source_code,
        force_two_phase=request.force_two_phase,
        force_single_pass=request.force_single_pass,
        complexity_threshold=request.complexity_threshold,
    )

    result = await run_in_executor(active.generate_test, gen_request)

    return PipelineGenerateResponse(
        success=result.success,
        test_code=result.test_code,
        class_name=result.class_name,
        file_path=request.file_path,
        mode=request.mode,
        collection=gen_request.collection_name or "",
        validation_passed=result.validation_passed,
        validation_issues=result.validation_issues,
        error=result.error,
        rag_chunks_used=result.rag_chunks_used,
        tokens_used=result.tokens_used,
        repair_attempts=result.repair_attempts,
        strategy_used=result.strategy_used,
        complexity_score=result.complexity_score,
        analysis_result=result.analysis_result,
    )


@app.post("/pipeline/generate-batch", response_model=PipelineBatchResponse)
async def pipeline_generate_batch(request: PipelineBatchRequest):
    """Generate unit tests for multiple Java classes in a single request.

    Designed for CI/CD pipelines processing an MR with multiple changed files.
    Each file is processed sequentially (to avoid overloading the LLM).

    Example::

        POST /pipeline/generate-batch
        {
            "files": [
                {
                    "file_path": "src/main/java/com/example/UserService.java",
                    "mode": "full"
                },
                {
                    "file_path": "src/main/java/com/example/OrderService.java",
                    "mode": "incremental",
                    "existing_test_code": "... existing test ...",
                    "changed_methods": ["placeOrder"]
                }
            ]
        }
    """
    active = _get_orchestrator()
    if not active:
        raise HTTPException(status_code=503, detail="Service not initialized")

    results: list[PipelineBatchItemResult] = []
    succeeded = 0
    failed = 0

    for file_req in request.files:
        try:
            # Validate incremental mode requirements
            if file_req.mode == "incremental" and not file_req.existing_test_code:
                results.append(PipelineBatchItemResult(
                    file_path=file_req.file_path,
                    success=False,
                    error="existing_test_code is required when mode='incremental'",
                    mode=file_req.mode,
                ))
                failed += 1
                continue

            gen_request = GenerationRequest(
                file_path=file_req.file_path,
                class_name=file_req.class_name,
                task_description=file_req.task_description
                    or "Generate comprehensive unit tests covering all public methods",
                existing_test_code=file_req.existing_test_code if file_req.mode == "incremental" else None,
                changed_methods=file_req.changed_methods if file_req.mode == "incremental" else None,
                collection_name=_resolve_collection(
                    explicit=file_req.collection,
                    file_path=file_req.file_path,
                ),
                source_code=file_req.source_code,
            )

            result = await run_in_executor(active.generate_test, gen_request)

            results.append(PipelineBatchItemResult(
                file_path=file_req.file_path,
                class_name=result.class_name,
                success=result.success,
                test_code=result.test_code,
                mode=file_req.mode,
                validation_passed=result.validation_passed,
                validation_issues=result.validation_issues,
                error=result.error,
                tokens_used=result.tokens_used,
                repair_attempts=result.repair_attempts,
            ))

            if result.success:
                succeeded += 1
            else:
                failed += 1

        except Exception as e:
            logger.error(
                "Batch item failed",
                file_path=file_req.file_path,
                error=str(e),
            )
            results.append(PipelineBatchItemResult(
                file_path=file_req.file_path,
                success=False,
                error=str(e),
                mode=file_req.mode,
            ))
            failed += 1

    return PipelineBatchResponse(
        total=len(request.files),
        succeeded=succeeded,
        failed=failed,
        results=results,
    )


# ============================================================================
# OpenAI-Compatible Endpoints (for Tabby integration)
# ============================================================================

@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models (OpenAI-compatible)."""
    return ModelsResponse(
        data=[
            ModelInfo(id="ai-agent", created=int(time.time())),
            ModelInfo(id="ai-agent-test-generator", created=int(time.time())),
        ]
    )



def _extract_java_file_path_from_message(content: str) -> Optional[str]:
    """Extract Java file path from inline code blocks sent by Continue IDE.

    Continue IDE format when user uses @ClassName:
        ```src/main/java/com/example/ClassName.java
        package com.example;
        ...
        ```

    Returns the file path if found, None otherwise.
    """
    import re

    # Pattern 1: code block with file path as info string
    # ```src/main/java/.../ClassName.java  or  ```java src/.../ClassName.java
    code_block_pattern = r'```(?:java\s+)?([^\n]*?\.java)\s*\n'
    match = re.search(code_block_pattern, content)
    if match:
        return match.group(1).strip()

    # Pattern 2: look for class declaration to infer class name as fallback
    class_pattern = r'(?:public\s+)?class\s+(\w+)'
    class_match = re.search(class_pattern, content)
    if class_match:
        class_name = class_match.group(1)
        # Try to find package declaration for a better path
        pkg_match = re.search(r'package\s+([\w.]+)\s*;', content)
        if pkg_match:
            package_path = pkg_match.group(1).replace('.', '/')
            return f"src/main/java/{package_path}/{class_name}.java"
        return f"{class_name}.java"

    return None


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint.

    This is the main endpoint for Continue IDE / Tabby integration.
    Supports:
      - Non-streaming and real SSE streaming (piped from vLLM)
      - Tool calling (tools in request, tool_calls in response)
      - Test generation pipeline (with RAG + validation + repair)
      - General chat with RAG context enhancement
      - Multi-collection: resolves Qdrant collection from ``collection``
        field, ``file_path``, or falls back to default.
    """
    active = _get_orchestrator()
    if not active:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # ── Parse messages ───────────────────────────────────────────────
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided")

    user_content = user_messages[-1].content or ""
    system_messages = [m for m in request.messages if m.role == "system"]
    system_content = system_messages[0].content if system_messages else ""

    # Filter overly-long IDE agent system prompts
    if system_content and any(marker in system_content for marker in [
        "<important_rules>", "You are in agent mode",
        "TOOL_NAME:", "read_file tool", "create_new_file tool",
    ]):
        system_content = ""

    # ── Handle tool calls from Continue ──────────────────────────────
    # If Continue sent tool results, include them in the message history
    # and forward to vLLM for the next response.
    tool_messages = [m for m in request.messages if m.role == "tool"]

    # ── Detect request intent ────────────────────────────────────────
    is_test_request = any(
        kw in user_content.lower()
        for kw in ["test", "junit", "unit test", "mockito", "generate test"]
    )

    file_path = request.file_path
    if not file_path:
        file_path = _extract_java_file_path_from_message(user_content)

    # ── Resolve Qdrant collection (3-tier) ───────────────────────────
    resolved_collection = _resolve_collection(
        explicit=request.collection,
        file_path=file_path,
        workspace_path=request.workspace_path,
    )
    logger.debug("Resolved collection", collection=resolved_collection)

    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created_time = int(time.time())

    # ═════════════════════════════════════════════════════════════════
    # PATH 1: Test generation pipeline (supports Two-Phase Strategy)
    # ═════════════════════════════════════════════════════════════════
    if is_test_request and file_path:
        logger.info(
            "Using test generation pipeline",
            file_path=file_path,
            collection=resolved_collection,
            force_two_phase=request.force_two_phase,
        )
        
        # Extract source code from user message if present
        source_code = None
        code_match = _re.search(r'```(?:[^\n]*\.java)?\s*\n(.*?)```', user_content, _re.DOTALL)
        if code_match:
            source_code = code_match.group(1).strip()
        
        gen_request = GenerationRequest(
            file_path=file_path,
            task_description=user_content,
            collection_name=resolved_collection,
            source_code=source_code,
            # Two-Phase Strategy options
            force_two_phase=request.force_two_phase or False,
            force_single_pass=request.force_single_pass or False,
            complexity_threshold=request.complexity_threshold or 10,
        )

        # ── REAL STREAMING: progress phases + token-by-token code ────
        if request.stream:
            return _stream_test_generation(
                response_id, created_time, request.model, gen_request,
            )

        # ── Non-streaming: run full pipeline, return complete result ─
        result = await run_in_executor(active.generate_test, gen_request)

        if result.success:
            response_content = result.test_code or ""
            tokens_used = result.tokens_used
        else:
            response_content = f"Error generating test: {result.error}"
            tokens_used = 0

        meta_block = _build_metadata_block(result)
        full_content = f"{response_content}\n\n{meta_block}" if meta_block else response_content

        return _non_streaming_response(
            response_id, created_time, request.model,
            full_content, user_content, tokens_used,
        )

    # ═════════════════════════════════════════════════════════════════
    # PATH 2: General chat — supports real streaming from vLLM
    # ═════════════════════════════════════════════════════════════════

    # Build the enhanced prompt (with optional RAG context)
    enhanced_prompt = user_content
    if file_path:
        enhanced_prompt = await _enrich_with_rag(user_content, file_path, resolved_collection)

    effective_system = system_content or orchestrator.prompt_builder.build_system_prompt()

    # ── REAL STREAMING (Fix A) ───────────────────────────────────────
    if request.stream:
        return _stream_from_vllm(
            response_id, created_time, request.model,
            effective_system, enhanced_prompt,
            request.temperature, request.max_tokens,
        )

    # ── Non-streaming ────────────────────────────────────────────────
    vllm_response = await run_in_executor(
        lambda: orchestrator.vllm.generate(
            system_prompt=effective_system,
            user_prompt=enhanced_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    )

    if vllm_response.success:
        response_content = vllm_response.content
        tokens_used = vllm_response.tokens_used
    else:
        response_content = f"Error: {vllm_response.error}"
        tokens_used = 0

    return _non_streaming_response(
        response_id, created_time, request.model,
        response_content, user_content, tokens_used,
    )


# ── Helpers ──────────────────────────────────────────────────────────

def _non_streaming_response(
    response_id: str,
    created_time: int,
    model: str,
    content: str,
    user_content: str,
    tokens_used: int,
) -> ChatCompletionResponse:
    """Build a standard non-streaming ChatCompletionResponse."""
    return ChatCompletionResponse(
        id=response_id,
        created=created_time,
        model=model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=count_tokens(user_content),
            completion_tokens=count_tokens(content),
            total_tokens=tokens_used or (count_tokens(user_content) + count_tokens(content)),
        ),
    )


def _stream_buffered(
    response_id: str,
    created_time: int,
    model: str,
    content: str,
) -> StreamingResponse:
    """Stream already-buffered content as SSE chunks (for test-gen)."""

    async def _generate():
        # Role chunk
        yield _sse_chunk(response_id, created_time, model,
                         delta={"role": "assistant", "content": ""})
        # Content in ~80-char pieces
        chunk_size = 80
        for i in range(0, len(content), chunk_size):
            yield _sse_chunk(response_id, created_time, model,
                             delta={"content": content[i:i + chunk_size]})
        # Finish
        yield _sse_chunk(response_id, created_time, model,
                         delta={}, finish_reason="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def _stream_test_generation(
    response_id: str,
    created_time: int,
    model: str,
    gen_request: GenerationRequest,
) -> StreamingResponse:
    """Stream test generation with progress phases + token-by-token code.

    Pipes ``orchestrator.generate_test_streaming()`` events directly as SSE:
    - Phase status messages (planning, retrieving, validating) → full text deltas
    - Code tokens (generating) → individual token deltas
    - Done/Error → finish reason

    The client sees output appear progressively, just like Claude or Copilot.
    """

    async def _generate():
        # Role chunk
        yield _sse_chunk(response_id, created_time, model,
                         delta={"role": "assistant", "content": ""})

        # Run the sync generator in a thread, ferry events via asyncio.Queue
        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _run_sync():
            try:
                active = _get_orchestrator()
                for event in active.generate_test_streaming(gen_request):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception as exc:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    StreamEvent(phase=StreamPhase.ERROR, content=f"Error: {exc}"),
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        loop.run_in_executor(None, _run_sync)

        while True:
            event = await queue.get()
            if event is None:
                break

            if event.phase == StreamPhase.ERROR:
                yield _sse_chunk(response_id, created_time, model,
                                 delta={"content": f"\n\n❌ {event.content}"})
                break

            if event.phase == StreamPhase.DONE:
                # Append metadata block
                meta = event.metadata
                meta_lines = ["\n\n---"]
                meta_lines.append(
                    f"**Validation:** {'✅ passed' if meta.get('validation_passed') else '❌ failed'}"
                )
                issues = meta.get("validation_issues", [])
                if issues:
                    meta_lines.append(f"**Issues:** {', '.join(issues[:5])}")
                meta_lines.append(f"**RAG context chunks:** {meta.get('rag_chunks_used', 0)}")
                meta_lines.append(f"**Tokens used:** {meta.get('tokens_used', 0)}")
                meta_block = "\n".join(meta_lines)
                yield _sse_chunk(response_id, created_time, model,
                                 delta={"content": meta_block})
                break

            if event.delta:
                # Token-by-token code delta — forward directly
                yield _sse_chunk(response_id, created_time, model,
                                 delta={"content": event.content})
            else:
                # Phase status message — send as full content chunk
                yield _sse_chunk(response_id, created_time, model,
                                 delta={"content": event.content + "\n"})

        # Finish
        yield _sse_chunk(response_id, created_time, model,
                         delta={}, finish_reason="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def _stream_from_vllm(
    response_id: str,
    created_time: int,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> StreamingResponse:
    """Pipe vLLM streaming directly to the client — real TTFT."""

    async def _generate():
        # Role chunk
        yield _sse_chunk(response_id, created_time, model,
                         delta={"role": "assistant", "content": ""})
        try:
            # stream_generate() is a sync generator (httpx sync client),
            # so we run it in a thread and ferry deltas via an asyncio.Queue
            # to avoid blocking the event loop.
            queue: asyncio.Queue[str | None] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def _run_sync_gen():
                try:
                    for delta_text in orchestrator.vllm.stream_generate(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ):
                        loop.call_soon_threadsafe(queue.put_nowait, delta_text)
                except Exception as exc:
                    loop.call_soon_threadsafe(
                        queue.put_nowait, f"\n\n[Error: {exc}]"
                    )
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

            loop.run_in_executor(None, _run_sync_gen)

            while True:
                item = await queue.get()
                if item is None:
                    break
                yield _sse_chunk(response_id, created_time, model,
                                 delta={"content": item})

        except Exception as e:
            logger.error("Streaming generation error", error=str(e))
            yield _sse_chunk(response_id, created_time, model,
                             delta={"content": f"\n\n[Error: {e}]"})
        # Finish
        yield _sse_chunk(response_id, created_time, model,
                         delta={}, finish_reason="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def _sse_chunk(
    response_id: str,
    created_time: int,
    model: str,
    delta: dict,
    finish_reason: Optional[str] = None,
) -> str:
    """Format a single SSE data line."""
    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _build_metadata_block(result) -> str:
    """Build a human-readable metadata block for test generation results."""
    parts = []
    parts.append("---")
    parts.append(f"**Validation:** {'\u2705 passed' if getattr(result, 'validation_passed', False) else '\u274c failed'}")

    issues = getattr(result, 'validation_issues', [])
    if issues:
        parts.append(f"**Issues:** {', '.join(issues[:5])}")

    repair = getattr(result, 'repair_attempts', 0)
    if repair:
        parts.append(f"**Repair attempts:** {repair}")

    rag = getattr(result, 'rag_chunks_used', 0)
    if rag:
        parts.append(f"**RAG context chunks:** {rag}")

    vs = getattr(result, 'validation_summary', None)
    if vs:
        parts.append(
            f"**Details:** {vs.get('errors', 0)} errors, "
            f"{vs.get('warnings', 0)} warnings, "
            f"{vs.get('test_count', '?')} test(s)"
        )

    parts.append(f"**Tokens used:** {getattr(result, 'tokens_used', 0)}")
    
    # Two-Phase Strategy metadata
    strategy = getattr(result, 'strategy_used', 'single_pass')
    if strategy == "two_phase":
        parts.append(f"**Strategy:** \U0001f504 Two-Phase Generation")
        complexity = getattr(result, 'complexity_score', 0)
        if complexity:
            parts.append(f"**Complexity score:** {complexity}")
    else:
        parts.append(f"**Strategy:** \u26a1 Single-Pass")
    
    return "\n".join(parts)


async def _enrich_with_rag(
    user_content: str, file_path: str,
    collection_name: Optional[str] = None,
) -> str:
    """Add RAG context to the prompt for general chat."""
    class_name = file_path.replace("\\", "/").split("/")[-1].replace(".java", "")
    try:
        search_result = await run_in_executor(
            lambda: orchestrator.rag.search(
                SearchQuery(
                    query=f"{class_name} service methods dependencies",
                    top_k=5,
                    score_threshold=0.4,
                ),
                collection_name=collection_name,
            )
        )
        if search_result.chunks:
            rag_context = "\n\n## Codebase Context:\n"
            for chunk in search_result.chunks:
                rag_context += f"\n### {chunk.class_name} ({chunk.type}, {chunk.layer})\n"
                rag_context += f"```\n{chunk.summary[:500]}\n```\n"
            return f"{user_content}\n{rag_context}"
    except Exception as e:
        logger.warning("RAG search failed", error=str(e))
    return user_content


@app.post("/v1/completions")
async def completions(request: dict):
    """OpenAI-compatible completions endpoint (legacy). Redirects to chat."""
    prompt = request.get("prompt", "")
    messages = [ChatMessage(role="user", content=prompt)]

    chat_request = ChatCompletionRequest(
        model=request.get("model", "ai-agent"),
        messages=messages,
        temperature=request.get("temperature", 0.2),
        max_tokens=request.get("max_tokens", 4096),
    )

    response = await chat_completions(chat_request)

    # If it's a streaming response, return as-is
    if isinstance(response, StreamingResponse):
        return response

    return {
        "id": response.id,
        "object": "text_completion",
        "created": response.created,
        "model": response.model,
        "choices": [
            {
                "text": response.choices[0].message.content if response.choices[0].message else "",
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "usage": response.usage.model_dump() if response.usage else {},
    }


# ============================================================================
# RAG Context Inspection
# ============================================================================

@app.get("/v1/rag-context")
async def get_rag_context(
    class_name: str,
    file_path: str = "",
    session_id: str = "",
):
    """Return RAG chunks used to build the prompt for a given class.

    Useful for debugging and transparency — lets callers see
    exactly what context the agent retrieves from the vector DB.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    session = None
    if session_id:
        session = await run_in_executor(
            orchestrator.memory_manager.get_session, session_id
        )

    chunks = await run_in_executor(
        orchestrator._get_rag_context, class_name, file_path, session
    )
    return chunks


# ============================================================================
# Phase 4: Agent Status & Metrics Endpoints
# ============================================================================

@app.get("/v1/agent/status")
async def agent_status():
    """Get agent pipeline status and metrics (Phase 4 observability)."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    metrics_data = orchestrator.metrics.get_metrics() if orchestrator.metrics else {}
    event_bus = orchestrator.event_bus

    return {
        "status": "running",
        "event_bus": {
            "total_events": event_bus.event_count,
            "subscribers": event_bus.subscriber_count,
        },
        "metrics": metrics_data,
    }


@app.get("/v1/agent/metrics")
async def agent_metrics():
    """Get detailed agent metrics (Phase 4 observability)."""
    if not orchestrator or not orchestrator.metrics:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return orchestrator.metrics.get_metrics()


@app.get("/v1/agent/events/stream")
async def agent_event_stream():
    """SSE stream of real-time agent events (Phase 4).

    Clients connect via EventSource / SSE and receive live events
    as the agent processes requests.
    """
    import asyncio
    import queue as queue_mod

    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    event_queue: queue_mod.Queue = queue_mod.Queue(maxsize=100)
    keepalive_counter = 0  # Track iterations for keepalive spacing

    def _forward_to_queue(event: Event) -> None:
        try:
            event_queue.put_nowait({
                "type": event.type.value,
                "data": event.data,
                "timestamp": event.timestamp,
                "source": event.source,
                "plan_id": event.plan_id,
            })
        except queue_mod.Full:
            logger.debug("Event queue full, dropping event")

    # Subscribe to all events
    orchestrator.event_bus.subscribe_all(_forward_to_queue)

    async def _event_generator():
        nonlocal keepalive_counter
        try:
            while True:
                try:
                    evt = event_queue.get_nowait()
                    keepalive_counter = 0
                    yield f"data: {json.dumps(evt)}\n\n"
                except queue_mod.Empty:
                    keepalive_counter += 1
                    # Send keepalive every ~15s (15 iterations × 1s sleep)
                    if keepalive_counter >= 15:
                        yield ": keepalive\n\n"
                        keepalive_counter = 0
                    await asyncio.sleep(1)
        finally:
            # Cleanup: thread-safe unsubscribe via EventBus lock
            try:
                with orchestrator.event_bus._lock:
                    handlers = orchestrator.event_bus._wildcard_handlers
                    if _forward_to_queue in handlers:
                        handlers.remove(_forward_to_queue)
            except Exception:
                logger.debug("Event handler cleanup failed (non-critical)")

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ============================================================================
# Tabby-specific endpoints
# ============================================================================

@app.get("/v1/health")
async def tabby_health():
    """Health check endpoint for Tabby."""
    return {"status": "ok"}


@app.post("/v1/events")
async def tabby_events(request: dict):
    """Receive events from Tabby (telemetry, etc.)."""
    logger.debug("Received Tabby event", event=request)
    return {"status": "ok"}


# ============================================================================
# Two-Phase Strategy & Domain Registry Endpoints
# ============================================================================

@app.get("/v1/two-phase/status")
async def two_phase_status():
    """Check if Two-Phase Strategy is enabled and get configuration."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "enabled": orchestrator.is_two_phase_enabled(),
        "domain_registry_available": orchestrator.domain_registry is not None,
        "complexity_calculator_available": orchestrator.complexity_calculator is not None,
    }


@app.get("/v1/complexity/{class_name}")
async def get_complexity(class_name: str, collection: Optional[str] = None):
    """Calculate complexity score for a class.
    
    Returns complexity score and level (simple/medium/complex).
    Used to determine if two-phase strategy should be used.
    
    Example: GET /v1/complexity/OrderService?collection=my_project
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    score, level = await run_in_executor(
        orchestrator.get_complexity_score, class_name, collection
    )
    
    return {
        "class_name": class_name,
        "complexity_score": score,
        "complexity_level": level,
        "recommended_strategy": "two_phase" if score >= 10 else "single_pass",
    }


@app.get("/v1/registry/stats")
async def registry_stats(collection: Optional[str] = None):
    """Get Domain Type Registry statistics.
    
    Shows how many domain types are indexed and their construction patterns.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if not orchestrator.domain_registry:
        return {"error": "Domain registry not available"}
    
    stats = orchestrator.domain_registry.get_stats(collection)
    return {
        "total_types": stats.total_types,
        "by_pattern": stats.by_pattern,
        "by_java_type": stats.by_java_type,
        "build_time_ms": round(stats.build_time_ms, 1),
        "cached": orchestrator.domain_registry.is_cached(collection),
    }


@app.post("/v1/registry/rebuild")
async def rebuild_registry(collection: Optional[str] = None):
    """Rebuild the Domain Type Registry.
    
    Call this after reindexing to ensure the registry is up-to-date.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    result = await run_in_executor(
        orchestrator.rebuild_domain_registry, collection
    )
    return result


@app.get("/v1/registry/lookup/{class_name}")
async def registry_lookup(class_name: str, collection: Optional[str] = None):
    """Look up construction info for a domain type.
    
    Returns the pre-computed construction pattern and example code.
    
    Example: GET /v1/registry/lookup/OrderRequest?collection=my_project
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    result = await run_in_executor(
        orchestrator.get_domain_type_info, class_name, collection
    )
    return result


@app.get("/v1/registry/prompt-section")
async def registry_prompt_section(
    class_names: str,
    collection: Optional[str] = None,
):
    """Generate a prompt section with construction examples for multiple types.
    
    Args:
        class_names: Comma-separated list of class names.
        collection: Optional Qdrant collection.
    
    Example: GET /v1/registry/prompt-section?class_names=OrderRequest,User,Order
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if not orchestrator.domain_registry:
        return {"error": "Domain registry not available"}
    
    names = [n.strip() for n in class_names.split(",") if n.strip()]
    
    prompt_section = await run_in_executor(
        orchestrator.domain_registry.get_prompt_section, names, collection
    )
    
    return {
        "class_names": names,
        "prompt_section": prompt_section,
    }


@app.post("/pipeline/generate-two-phase", response_model=PipelineGenerateResponse)
async def pipeline_generate_two_phase(request: PipelineGenerateRequest):
    """Generate unit tests using Two-Phase Strategy (explicit).
    
    This endpoint always uses the two-phase strategy regardless of complexity.
    Use this when you know the service is complex and want better accuracy.
    
    Two-Phase Strategy:
    1. Phase 1 (Analysis): LLM analyzes the service and outputs a JSON plan
    2. Phase 2 (Generation): For each method, generate focused tests with
       exact construction patterns from the Domain Registry
    
    Example::
    
        POST /pipeline/generate-two-phase
        {
            "file_path": "src/main/java/com/example/ComplexService.java",
            "source_code": "package com.example; ...",
            "collection": "my_project"
        }
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if not orchestrator.is_two_phase_enabled():
        raise HTTPException(
            status_code=400,
            detail="Two-Phase Strategy is not enabled. Check server configuration."
        )
    
    # Force two-phase
    request.force_two_phase = True
    request.force_single_pass = False
    
    # Delegate to the standard pipeline endpoint
    return await pipeline_generate(request)


# ============================================================================
# LangGraph endpoints (human review + run status)
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


def _get_orchestrator():
    """Get the active orchestrator (graph or legacy)."""
    if graph_orchestrator and not USE_LEGACY:
        return graph_orchestrator
    return orchestrator


@app.post("/review/{run_id}")
async def submit_review(run_id: str, request: ReviewRequest):
    """Resume a paused graph after human review.

    Only available when using LangGraph backend.
    """
    if not graph_orchestrator:
        raise HTTPException(
            status_code=400,
            detail="Human review requires LangGraph backend. Set USE_LEGACY_ORCHESTRATOR=false.",
        )

    def _do_review():
        return graph_orchestrator.submit_review(
            run_id=run_id,
            approved=request.approved,
            feedback=request.feedback,
        )

    result = await run_in_executor(_do_review)
    return {
        "success": result.success,
        "test_code": result.test_code,
        "class_name": result.class_name,
        "validation_passed": result.validation_passed,
        "run_id": run_id,
    }


@app.get("/runs/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: str):
    """Get the status of a graph run (for polling)."""
    if not graph_orchestrator:
        raise HTTPException(
            status_code=400,
            detail="Run tracking requires LangGraph backend.",
        )

    state = graph_orchestrator.get_run_state(run_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    # Determine status
    human_approved = state.get("human_approved")
    final_code = state.get("final_test_code")
    error = state.get("error")

    if error:
        status = "failed"
    elif final_code:
        status = "completed"
    elif human_approved is None and state.get("require_human_review"):
        status = "interrupted"
    else:
        status = "running"

    return RunStatusResponse(
        run_id=run_id,
        status=status,
        class_name=state.get("class_name", ""),
        validation_passed=state.get("validation_passed"),
        test_code=final_code,
        validation_issues=state.get("validation_issues", []),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server.api:app",
        host=os.getenv("SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("SERVER_PORT", "8080")),
        reload=True,
    )

