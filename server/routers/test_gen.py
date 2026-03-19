"""
Test generation, pipeline, session, and LangGraph endpoints.

Includes: /generate-test, /refine-test, /pipeline/*, /session*,
          /review/*, /runs/*, /v1/two-phase/*, /v1/complexity/*,
          /v1/registry/*
"""

import asyncio
import os
from typing import Optional

import structlog
from fastapi import APIRouter, HTTPException

from agent.models import GenerationRequest
from ..dependencies import (
    graph_orchestrator, session_manager, _executor,
    _get_orchestrator, run_in_executor,
)
from ..schemas import (
    GenerateTestRequest, GenerateTestResponse,
    RefineTestRequest,
    PipelineGenerateRequest, PipelineGenerateResponse,
    PipelineBatchRequest, PipelineBatchResponse, PipelineBatchItemResult,
    ReviewRequest, RunStatusResponse,
)
from ..session import SessionInfo
from ..services.rag_resolver import _resolve_collection

logger = structlog.get_logger()

router = APIRouter()


# ============================================================================
# Legacy Test Generation
# ============================================================================

@router.post("/generate-test", response_model=GenerateTestResponse, deprecated=True)
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
        force_two_phase=True,
        force_single_pass=False,
    )

    result = await active.generate_test(gen_request)

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


@router.post("/refine-test", response_model=GenerateTestResponse)
async def refine_test(request: RefineTestRequest):
    """Refine a previously generated test based on feedback."""
    if not graph_orchestrator:
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
        graph_orchestrator.refine_test,
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


# ============================================================================
# Session Management
# ============================================================================

@router.post("/session", response_model=SessionInfo)
async def create_session():
    """Create a new session."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, session_manager.create_session)


@router.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get session information."""
    loop = asyncio.get_running_loop()
    session = await loop.run_in_executor(_executor, session_manager.get_session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    loop = asyncio.get_running_loop()
    deleted = await loop.run_in_executor(_executor, session_manager.delete_session, session_id)
    if deleted:
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions():
    """List all active sessions."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, session_manager.list_sessions)


# ============================================================================
# Pipeline API — CI/CD Integration (GitLab, Jenkins, etc.)
# ============================================================================

@router.post("/pipeline/generate", response_model=PipelineGenerateResponse)
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
        force_two_phase=True,
        force_single_pass=False,
        complexity_threshold=request.complexity_threshold,
    )

    result = await active.generate_test(gen_request)

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


@router.post("/pipeline/generate-batch", response_model=PipelineBatchResponse)
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
                force_two_phase=True,
                force_single_pass=False,
            )

            result = await active.generate_test(gen_request)

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


@router.post("/pipeline/generate-two-phase", response_model=PipelineGenerateResponse)
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
    if not graph_orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if not graph_orchestrator.is_two_phase_enabled():
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
# Two-Phase Strategy & Domain Registry Endpoints
# ============================================================================

@router.get("/v1/two-phase/status")
async def two_phase_status():
    """Check if Two-Phase Strategy is enabled and get configuration."""
    if not graph_orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "enabled": graph_orchestrator.is_two_phase_enabled(),
        "domain_registry_available": graph_orchestrator.domain_registry is not None,
        "complexity_calculator_available": graph_orchestrator.complexity_calculator is not None,
    }


@router.get("/v1/complexity/{class_name}")
async def get_complexity(class_name: str, collection: Optional[str] = None):
    """Calculate complexity score for a class.
    
    Returns complexity score and level (simple/medium/complex).
    Used to determine if two-phase strategy should be used.
    
    Example: GET /v1/complexity/OrderService?collection=my_project
    """
    if not graph_orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    score, level = await run_in_executor(
        graph_orchestrator.get_complexity_score, class_name, collection
    )
    
    return {
        "class_name": class_name,
        "complexity_score": score,
        "complexity_level": level,
        "recommended_strategy": "two_phase" if score >= 10 else "single_pass",
    }


@router.get("/v1/registry/stats")
async def registry_stats(collection: Optional[str] = None):
    """Get Domain Type Registry statistics.
    
    Shows how many domain types are indexed and their construction patterns.
    """
    if not graph_orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if not graph_orchestrator.domain_registry:
        return {"error": "Domain registry not available"}
    
    stats = graph_orchestrator.domain_registry.get_stats(collection)
    return {
        "total_types": stats.total_types,
        "by_pattern": stats.by_pattern,
        "by_java_type": stats.by_java_type,
        "build_time_ms": round(stats.build_time_ms, 1),
        "cached": graph_orchestrator.domain_registry.is_cached(collection),
    }


@router.post("/v1/registry/rebuild")
async def rebuild_registry(collection: Optional[str] = None):
    """Rebuild the Domain Type Registry.
    
    Call this after reindexing to ensure the registry is up-to-date.
    """
    if not graph_orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    result = await run_in_executor(
        graph_orchestrator.rebuild_domain_registry, collection
    )
    return result


@router.get("/v1/registry/lookup/{class_name}")
async def registry_lookup(class_name: str, collection: Optional[str] = None):
    """Look up construction info for a domain type.
    
    Returns the pre-computed construction pattern and example code.
    
    Example: GET /v1/registry/lookup/OrderRequest?collection=my_project
    """
    if not graph_orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    result = await run_in_executor(
        graph_orchestrator.get_domain_type_info, class_name, collection
    )
    return result


@router.get("/v1/registry/prompt-section")
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
    if not graph_orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if not graph_orchestrator.domain_registry:
        return {"error": "Domain registry not available"}
    
    names = [n.strip() for n in class_names.split(",") if n.strip()]
    
    prompt_section = await run_in_executor(
        graph_orchestrator.domain_registry.get_prompt_section, names, collection
    )
    
    return {
        "class_names": names,
        "prompt_section": prompt_section,
    }


# ============================================================================
# LangGraph endpoints (human review + run status)
# ============================================================================

@router.post("/review/{run_id}")
async def submit_review(run_id: str, request: ReviewRequest):
    """Resume a paused graph after human review.

    Only available when using LangGraph backend.
    """
    if not graph_orchestrator:
        raise HTTPException(
            status_code=400,
            detail="Human review requires LangGraph backend. Set USE_LEGACY_ORCHESTRATOR=false.",
        )

    result = await graph_orchestrator.submit_review(
        run_id=run_id,
        approved=request.approved,
        feedback=request.feedback,
    )
    return {
        "success": result.success,
        "test_code": result.test_code,
        "class_name": result.class_name,
        "validation_passed": result.validation_passed,
        "run_id": run_id,
    }


@router.get("/runs/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: str):
    """Get the status of a graph run (for polling)."""
    if not graph_orchestrator:
        raise HTTPException(
            status_code=400,
            detail="Run tracking requires LangGraph backend.",
        )

    state = await graph_orchestrator.get_run_state(run_id)
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
