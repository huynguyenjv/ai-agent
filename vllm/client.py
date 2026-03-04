"""
vLLM client for OpenAI-compatible API.
"""

import json
import time as _time
from dataclasses import dataclass
from typing import Optional, Generator

import httpx
import structlog

logger = structlog.get_logger()


@dataclass
class GenerationResponse:
    """Response from vLLM generation."""

    success: bool
    content: str = ""
    tokens_used: int = 0
    finish_reason: str = ""
    error: Optional[str] = None


class VLLMClient:
    """Client for vLLM OpenAI-compatible API with connection pooling."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "token-abc123",
        model: str = "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        timeout: int = 600,  # Increased timeout for long generations
        max_connections: int = 10,  # Connection pool size
        max_keepalive_connections: int = 5,  # Keep-alive connections
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout
        self._closed = False

        # Configure connection pool — DISABLE keepalive to prevent
        # stale/half-open TCP connections from blocking subsequent requests.
        # Each LLM generation is heavy (seconds to minutes), so the cost
        # of a fresh TCP connection is negligible.
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=0,   # ← No keepalive: avoids stale conn reuse
        )

        # Use longer timeouts for connect and read
        self.client = httpx.Client(
            timeout=httpx.Timeout(
                connect=30.0,
                read=timeout,
                write=30.0,
                pool=60.0,   # Wait up to 60s for a connection slot
            ),
            limits=limits,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            http2=False,  # Disable HTTP/2 for better compatibility with vLLM
        )

        logger.info(
            "vLLM client initialized",
            base_url=base_url,
            model=model,
            max_connections=max_connections,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = True,  # Default to streaming to avoid KV cache blocking
        max_retries: int = 2,
    ) -> GenerationResponse:
        """Generate completion using vLLM with retry on connection errors.

        Retries only on transient network/connection errors (httpx.RequestError).
        HTTP 4xx/5xx and other errors are returned immediately.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "top_p": self.top_p,
            "stream": stream,
        }

        last_error: Optional[str] = None

        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(
                    "Sending generation request",
                    model=self.model,
                    prompt_length=len(user_prompt),
                    stream=stream,
                    attempt=attempt,
                )

                if stream:
                    return self._generate_streaming(payload)
                else:
                    return self._generate_non_streaming(payload)

            except httpx.RequestError as e:
                # Transient network error — retry
                last_error = f"Request error (attempt {attempt}/{max_retries}): {e}"
                logger.warning("vLLM request failed, retrying", error=last_error, attempt=attempt)
                if attempt < max_retries:
                    _time.sleep(min(2 ** attempt, 10))  # exponential back-off
                continue

            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP error: {e.response.status_code}"
                logger.error("vLLM request failed", error=error_msg)
                return GenerationResponse(success=False, error=error_msg)

            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                logger.error("vLLM request failed", error=error_msg)
                return GenerationResponse(success=False, error=error_msg)

        # All retries exhausted
        logger.error("vLLM generation failed after all retries", error=last_error)
        return GenerationResponse(success=False, error=last_error)

    def _generate_streaming(self, payload: dict) -> GenerationResponse:
        """Generate with streaming - releases KV cache incrementally.

        After receiving ``[DONE]``, the remaining response body is drained
        explicitly so that the underlying connection is released cleanly.
        """
        content_parts = []
        finish_reason = ""

        with self.client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line or line.startswith(":"):
                    continue

                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix

                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        choice = data.get("choices", [{}])[0]
                        delta = choice.get("delta", {})

                        if "content" in delta:
                            content_parts.append(delta["content"])

                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]

                    except json.JSONDecodeError:
                        continue

            # Drain any remaining bytes so httpx can close the connection
            # cleanly instead of leaving it in a half-open state.
            try:
                response.read()  # consumes remaining body
            except Exception:
                pass  # already closed / empty — that's fine

        content = "".join(content_parts)
        # Estimate tokens (streaming doesn't always return usage)
        tokens_used = len(content) // 4  # Rough estimate

        logger.info(
            "Streaming generation complete",
            content_length=len(content),
            finish_reason=finish_reason,
        )

        return GenerationResponse(
            success=True,
            content=content,
            tokens_used=tokens_used,
            finish_reason=finish_reason,
        )

    def _generate_non_streaming(self, payload: dict) -> GenerationResponse:
        """Generate without streaming - may block KV cache longer."""
        payload["stream"] = False
        
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]

        tokens_used = data.get("usage", {}).get("total_tokens", 0)

        logger.info(
            "Generation complete",
            tokens_used=tokens_used,
            finish_reason=choice.get("finish_reason", ""),
        )

        return GenerationResponse(
            success=True,
            content=message["content"],
            tokens_used=tokens_used,
            finish_reason=choice.get("finish_reason", ""),
        )

    def generate_with_history(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = True,  # Default to streaming
    ) -> GenerationResponse:
        """Generate completion with conversation history."""
        try:
            all_messages = [{"role": "system", "content": system_prompt}]
            all_messages.extend(messages)

            payload = {
                "model": self.model,
                "messages": all_messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
                "top_p": self.top_p,
                "stream": stream,
            }

            if stream:
                return self._generate_streaming(payload)
            else:
                return self._generate_non_streaming(payload)

        except Exception as e:
            logger.error("vLLM request failed", error=str(e))
            return GenerationResponse(success=False, error=str(e))

    # -----------------------------------------------------------------
    # Streaming iterator — yields delta strings for real SSE passthrough
    # -----------------------------------------------------------------

    def stream_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """Yield content delta strings as they arrive from vLLM.

        Unlike ``generate()`` this does **not** buffer the full response.
        Each ``yield`` is a small text delta that can be forwarded
        immediately to an SSE client (Continue / Tabby / etc.).

        Usage::

            for chunk in vllm.stream_generate(sys, usr):
                send_sse(chunk)
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "top_p": self.top_p,
            "stream": True,
        }

        with self.client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line or line.startswith(":"):
                    continue

                if line.startswith("data: "):
                    data_str = line[6:]

                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        choice = data.get("choices", [{}])[0]
                        delta = choice.get("delta", {})

                        if "content" in delta and delta["content"]:
                            yield delta["content"]

                    except json.JSONDecodeError:
                        continue

            # Drain remaining bytes to release the connection cleanly
            try:
                response.read()
            except Exception:
                pass

    def health_check(self, timeout: float = 5.0) -> bool:
        """Check if vLLM server is healthy (with short timeout)."""
        try:
            response = self.client.get(f"{self.base_url}/models", timeout=timeout)
            return response.status_code == 200
        except Exception:
            return False

    def get_model_info(self) -> Optional[dict]:
        """Get information about the loaded model."""
        try:
            response = self.client.get(f"{self.base_url}/models")
            response.raise_for_status()
            data = response.json()
            models = data.get("data", [])
            for model in models:
                if model.get("id") == self.model:
                    return model
            return models[0] if models else None
        except Exception as e:
            logger.error("Failed to get model info", error=str(e))
            return None

    def close(self):
        """Close the HTTP client and release all connections."""
        if not self._closed:
            self._closed = True
            self.client.close()
            logger.info("vLLM client closed")

    def is_closed(self) -> bool:
        """Check if the client is closed."""
        return self._closed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        """Ensure client is closed on garbage collection."""
        if not self._closed:
            self.close()

