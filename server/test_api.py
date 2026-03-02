from fastapi import Request
from agent.orchestrator import AgentOrchestrator
import os
# Helper to get orchestrator instance (reuse from main app if possible)
def get_orchestrator():
    # In production, import from server.api, here fallback to new instance
    try:
        from .api import orchestrator
        if orchestrator:
            return orchestrator
    except Exception:
        pass
    # Fallback: create new (may not share memory/session)
    from rag.client import RAGClient
    from vllm.client import VLLMClient
    return AgentOrchestrator(
        rag_client=RAGClient(),
        vllm_client=VLLMClient(
            base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("VLLM_API_KEY", "token-abc123"),
            model=os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"),
        ),
    )
# API: get RAG context for a class/filew
@router.get("/rag-context")
def get_rag_context(
    class_name: str,
    file_path: str = "",
    session_id: str = ""
):
    """
    Trả về context (các chunk) mà agent thực sự dùng để build prompt cho model.
    """
    orchestrator = get_orchestrator()
    session = None
    if session_id:
        session = orchestrator.memory_manager.get_session(session_id)
    chunks = orchestrator._get_rag_context(class_name, file_path, session)
    # return {
    #     "class_name": class_name,
    #     "file_path": file_path,
    #     "chunks": [
    #         {
    #             "class_name": c.class_name,
    #             "type": c.type,
    #             "java_type": c.java_type,
    #             "fields": [f"{f.type} {f.name}" for f in getattr(c, "fields", [])],
    #             "record_components": [f"{f.type} {f.name}" for f in getattr(c, "record_components", [])],
    #             "dependencies": getattr(c, "dependencies", []),
    #             "used_types": getattr(c, "used_types", []),
    #             "summary": c.summary,
    #         }
    #         for c in chunks
    #     ]
    # }
    return chunks
