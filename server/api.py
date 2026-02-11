"""
FastAPI server for the AI coding agent.
Compatible with Tabby IDE via OpenAI-compatible API.
"""

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, Literal

import structlog
import logging

# Cấu hình log level DEBUG cho toàn bộ app
logging.basicConfig(level=logging.DEBUG)
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
)
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.orchestrator import AgentOrchestrator, GenerationRequest
from indexer.build_index import IndexBuilder
from rag.client import RAGClient
from rag.schema import SearchQuery
from vllm.client import VLLMClient
from .session import SessionManager, SessionInfo

# Load environment variables
load_dotenv()

logger = structlog.get_logger()

# Global instances
orchestrator: Optional[AgentOrchestrator] = None
index_builder: Optional[IndexBuilder] = None
session_manager: SessionManager = SessionManager()


# ============================================================================
# OpenAI-Compatible Models (for Tabby integration)
# ============================================================================

class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = "ai-agent"
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False
    # Custom fields for RAG
    file_path: Optional[str] = None  # Current file being edited
    workspace_path: Optional[str] = None  # Workspace root


class ChatCompletionChoice(BaseModel):
    """OpenAI-compatible chat completion choice."""
    index: int = 0
    message: Optional[ChatMessage] = None
    delta: Optional[ChatMessage] = None  # For streaming
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
    """Request model for test generation."""

    file_path: str = Field(..., description="Path to the Java source file")
    class_name: Optional[str] = Field(None, description="Class name (auto-detected if not provided)")
    task_description: Optional[str] = Field(None, description="Additional task description")
    session_id: Optional[str] = Field(None, description="Session ID for context continuity")


class GenerateTestResponse(BaseModel):
    """Response model for test generation."""

    success: bool
    test_code: Optional[str] = None
    class_name: str = ""
    validation_passed: bool = True
    validation_issues: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    session_id: Optional[str] = None
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


class ReindexResponse(BaseModel):
    """Response model for reindexing."""

    success: bool
    message: str
    points_indexed: int = 0
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    vllm_healthy: bool
    qdrant_healthy: bool
    index_stats: Optional[dict] = None


def create_orchestrator() -> AgentOrchestrator:
    """Create and configure the agent orchestrator."""
    rag_client = RAGClient(
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        collection_name=os.getenv("QDRANT_COLLECTION", "java_codebase"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    )

    vllm_client = VLLMClient(
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("VLLM_API_KEY", "token-abc123"),
        model=os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"),
    )

    return AgentOrchestrator(
        rag_client=rag_client,
        vllm_client=vllm_client,
    )


def create_index_builder() -> IndexBuilder:
    """Create and configure the index builder."""
    return IndexBuilder(
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        collection_name=os.getenv("QDRANT_COLLECTION", "java_codebase"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global orchestrator, index_builder

    logger.info("Starting AI Agent server...")

    # Initialize components
    orchestrator = create_orchestrator()
    index_builder = create_index_builder()

    logger.info("Server initialized successfully")

    yield

    # Cleanup
    logger.info("Shutting down server...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AI Coding Agent",
        description="Self-hosted AI agent for generating JUnit5 + Mockito unit tests",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of all services."""
    vllm_healthy = False
    qdrant_healthy = False
    index_stats = None

    if orchestrator:
        vllm_healthy = orchestrator.vllm.health_check()
        try:
            stats = orchestrator.rag.get_stats()
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
                collections = orchestrator.rag.qdrant.get_collections()
                qdrant_healthy = True
                index_stats = {"collection": "java_codebase", "total_points": 0, "status": "connected"}
            except Exception:
                pass

    status = "healthy" if (vllm_healthy and qdrant_healthy) else "degraded"

    return HealthResponse(
        status=status,
        vllm_healthy=vllm_healthy,
        qdrant_healthy=qdrant_healthy,
        index_stats=index_stats,
    )


@app.post("/generate-test", response_model=GenerateTestResponse)
async def generate_test(request: GenerateTestRequest):
    """Generate unit tests for a Java class."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Create session if not provided
    session_id = request.session_id
    if not session_id:
        session = session_manager.create_session()
        session_id = session.session_id

    # Generate test
    gen_request = GenerationRequest(
        file_path=request.file_path,
        class_name=request.class_name,
        task_description=request.task_description,
        session_id=session_id,
    )

    result = orchestrator.generate_test(gen_request)

    # Update session
    if result.success:
        session_manager.update_session(
            session_id=session_id,
            current_class=result.class_name,
            current_file=request.file_path,
            increment_tests=True,
        )

    return GenerateTestResponse(
        success=result.success,
        test_code=result.test_code,
        class_name=result.class_name,
        validation_passed=result.validation_passed,
        validation_issues=result.validation_issues,
        error=result.error,
        session_id=session_id,
        rag_chunks_used=result.rag_chunks_used,
        tokens_used=result.tokens_used,
    )


@app.post("/refine-test", response_model=GenerateTestResponse)
async def refine_test(request: RefineTestRequest):
    """Refine a previously generated test based on feedback."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Verify session exists
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    result = orchestrator.refine_test(
        session_id=request.session_id,
        feedback=request.feedback,
    )

    if result.success:
        session_manager.update_session(
            session_id=request.session_id,
            increment_tests=True,
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
    """Reindex a Java repository."""
    if not index_builder:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Validate path exists
    if not os.path.isdir(request.repo_path):
        raise HTTPException(status_code=400, detail=f"Directory not found: {request.repo_path}")

    try:
        # Run indexing (could be moved to background for large repos)
        points_indexed = index_builder.index_repository(
            repo_path=request.repo_path,
            recreate=request.recreate,
        )

        return ReindexResponse(
            success=True,
            message=f"Successfully indexed {points_indexed} code elements",
            points_indexed=points_indexed,
        )

    except Exception as e:
        logger.error("Reindexing failed", error=str(e))
        return ReindexResponse(
            success=False,
            message="Reindexing failed",
            error=str(e),
        )


@app.get("/index/stats")
async def get_index_stats():
    """Get statistics about the vector index."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    stats = orchestrator.rag.get_stats()
    return stats.model_dump()


@app.post("/session", response_model=SessionInfo)
async def create_session():
    """Create a new session."""
    return session_manager.create_session()


@app.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get session information."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_manager.delete_session(session_id):
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/sessions", response_model=list[SessionInfo])
async def list_sessions():
    """List all active sessions."""
    return session_manager.list_sessions()


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



@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    This is the main endpoint for Tabby integration.
    """
    logger.info("/v1/chat/completions called", request=request.model_dump())
    if not orchestrator:
        logger.error("Orchestrator not initialized")
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Extract user message
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        logger.error("No user message provided", messages=request.messages)
        raise HTTPException(status_code=400, detail="No user message provided")

    user_content = user_messages[-1].content
    system_messages = [m for m in request.messages if m.role == "system"]
    system_content = system_messages[0].content if system_messages else ""

    # Filter out Continue/IDE agent system prompts (they are too long and not useful)
    # These prompts contain tool instructions that we don't need
    if system_content and any(marker in system_content for marker in [
        "<important_rules>",
        "You are in agent mode",
        "TOOL_NAME:",
        "read_file tool",
        "create_new_file tool",
    ]):
        logger.info("Filtered out IDE agent system prompt (too long/not relevant)")
        system_content = ""  # Use our own system prompt instead

    logger.debug("Parsed messages", user_content=user_content[:100], system_content=system_content[:100] if system_content else "")

    # Check if this is a test generation request
    is_test_request = any(
        keyword in user_content.lower()
        for keyword in ["test", "junit", "unit test", "mockito", "generate test"]
    )
    logger.debug("Is test request", is_test_request=is_test_request)

    # Get RAG context if file_path is provided
    rag_context = ""
    if request.file_path:
        class_name = request.file_path.replace("\\", "/").split("/")[-1].replace(".java", "")
        try:
            logger.debug("Searching RAG context", class_name=class_name)
            search_result = orchestrator.rag.search(SearchQuery(
                query=f"{class_name} service methods dependencies",
                top_k=5,
                score_threshold=0.4,
            ))
            if search_result.chunks:
                rag_context = "\n\n## Codebase Context:\n"
                for chunk in search_result.chunks:
                    rag_context += f"\n### {chunk.class_name} ({chunk.type}, {chunk.layer})\n"
                    rag_context += f"```\n{chunk.summary[:500]}\n```\n"
            logger.debug("RAG context result", rag_context=rag_context)
        except Exception as e:
            logger.warning("RAG search failed", error=str(e))

    # Build the prompt
    if is_test_request and request.file_path:
        logger.info("Using test generation logic", file_path=request.file_path)
        gen_request = GenerationRequest(
            file_path=request.file_path,
            task_description=user_content,
        )
        result = orchestrator.generate_test(gen_request)
        logger.debug("Test generation result", result=result)

        if result.success:
            response_content = result.test_code or ""
            tokens_used = result.tokens_used
        else:
            response_content = f"Error generating test: {result.error}"
            tokens_used = 0
    else:
        # General chat with RAG context
        enhanced_prompt = user_content
        if rag_context:
            enhanced_prompt = f"{user_content}\n{rag_context}"

        logger.info("Calling vLLM for generation", prompt=enhanced_prompt[:200])
        vllm_response = orchestrator.vllm.generate(
            system_prompt=system_content or orchestrator.prompt_builder.build_system_prompt(),
            user_prompt=enhanced_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        logger.debug("vLLM response", vllm_response=vllm_response)

        if vllm_response.success:
            response_content = vllm_response.content
            tokens_used = vllm_response.tokens_used
        else:
            response_content = f"Error: {vllm_response.error}"
            tokens_used = 0

    logger.info("Returning chat completion response", response_preview=response_content[:200] if response_content else "empty")
    
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created_time = int(time.time())
    
    # Handle streaming response
    if request.stream:
        async def generate_stream():
            # Send content in chunks for streaming
            # First chunk with role
            first_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(first_chunk)}\n\n"
            
            # Send content in smaller chunks
            chunk_size = 50
            for i in range(0, len(response_content), chunk_size):
                chunk_text = response_content[i:i+chunk_size]
                chunk_data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk_text},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
            
            # Final chunk
            final_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    # Non-streaming response
    return ChatCompletionResponse(
        id=response_id,
        created=created_time,
        model=request.model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=response_content),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=len(user_content) // 4,
            completion_tokens=len(response_content) // 4,
            total_tokens=tokens_used or (len(user_content) + len(response_content)) // 4,
        ),
    )


@app.post("/v1/completions")
async def completions(request: dict):
    """
    OpenAI-compatible completions endpoint (legacy).
    Redirects to chat completions.
    """
    prompt = request.get("prompt", "")
    messages = [ChatMessage(role="user", content=prompt)]

    chat_request = ChatCompletionRequest(
        model=request.get("model", "ai-agent"),
        messages=messages,
        temperature=request.get("temperature", 0.2),
        max_tokens=request.get("max_tokens", 4096),
    )

    response = await chat_completions(chat_request)

    # Convert to completions format
    return {
        "id": response.id,
        "object": "text_completion",
        "created": response.created,
        "model": response.model,
        "choices": [
            {
                "text": response.choices[0].message.content,
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "usage": response.usage.model_dump(),
    }


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server.api:app",
        host=os.getenv("SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("SERVER_PORT", "8080")),
        reload=True,
    )

