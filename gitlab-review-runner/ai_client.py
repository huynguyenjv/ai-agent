"""Client to the AI server's /review/analyze endpoint."""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger("runner.ai")


class AIClient:
    def __init__(self, base_url: str, api_key: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self._headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
        self._timeout = timeout

    def analyze(self, payload: dict) -> dict:
        url = f"{self.base_url}/review/analyze"
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, headers=self._headers, json=payload)
            if resp.status_code != 200:
                log.error(
                    "AI server %s: HTTP %d body=%s",
                    url, resp.status_code, resp.text[:500],
                )
            resp.raise_for_status()
            return resp.json()
