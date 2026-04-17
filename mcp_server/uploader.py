"""Uploader — Section 5, MCP Server to Cloud VM.

Async HTTP POST to SERVER_URL/index.
Retry policy: 3 retries with exponential backoff starting at 500ms on 5xx errors.
Timeout: 30 seconds per request.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from mcp_server.models import CodeChunk

logger = logging.getLogger("mcp_server.uploader")


class Uploader:
    """Uploads chunks to the Cloud VM /index endpoint."""

    def __init__(self, server_url: str, api_key: str) -> None:
        self._url = f"{server_url.rstrip('/')}/index"
        self._api_key = api_key
        self._max_retries = 3
        self._initial_backoff = 0.5  # 500ms
        self._timeout = 30.0

    async def upload(
        self,
        chunks: list[CodeChunk],
        deleted_ids: list[str] | None = None,
    ) -> dict:
        """Upload chunks to the server.

        Returns {"indexed": N, "deleted": M} on success.
        Returns {"indexed": 0, "error": "..."} on failure.

        Section 5: If all retries fail, the hash store is NOT updated.
        """
        payload = {
            "chunks": [c.to_payload() for c in chunks],
            "deleted_ids": deleted_ids or [],
        }

        headers = {
            "X-Api-Key": self._api_key,
            "Content-Type": "application/json",
        }

        last_error = None
        backoff = self._initial_backoff

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for attempt in range(self._max_retries):
                try:
                    response = await client.post(
                        self._url,
                        json=payload,
                        headers=headers,
                    )
                    if response.status_code == 200:
                        return response.json()
                    elif response.status_code >= 500:
                        last_error = f"Server error {response.status_code}"
                        logger.warning(
                            "Upload attempt %d failed: %s",
                            attempt + 1, last_error,
                        )
                    else:
                        return {
                            "indexed": 0,
                            "error": f"HTTP {response.status_code}: {response.text}",
                        }
                except httpx.TimeoutException:
                    last_error = "Upload timeout"
                    logger.warning("Upload attempt %d timed out", attempt + 1)
                except httpx.HTTPError as e:
                    last_error = str(e)
                    logger.warning("Upload attempt %d error: %s", attempt + 1, e)

                if attempt < self._max_retries - 1:
                    await asyncio.sleep(backoff)
                    backoff *= 2

        return {"indexed": 0, "error": last_error or "Upload failed"}
