"""Speculative Execution — Phase 19.3.

Best-effort pre-warming of the vLLM pipeline so the first real request doesn't
pay cold-start cost. Gated by SPECULATIVE_PREWARM (off by default). Kept small
and side-effect-free on failure.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("server.speculative")


def prewarm_enabled() -> bool:
    return os.environ.get("SPECULATIVE_PREWARM", "false").lower() in ("1", "true", "yes")


async def prewarm_vllm(client, model: str) -> bool:
    """Send a tiny completion to warm the connection pool + model. Best-effort."""
    try:
        await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            stream=False,
        )
        logger.info("vLLM prewarm ok (model=%s)", model)
        return True
    except Exception as e:
        logger.warning("vLLM prewarm failed (non-fatal): %s", e)
        return False
