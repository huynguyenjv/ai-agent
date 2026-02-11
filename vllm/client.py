"""
vLLM client for OpenAI-compatible API.
"""

from dataclasses import dataclass
from typing import Optional

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

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
    """Client for vLLM OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "token-abc123",
        model: str = "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        timeout: int = 600,  # Increased timeout for long generations
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout

        # Use longer timeouts for connect and read
        self.client = httpx.Client(
            timeout=httpx.Timeout(
                connect=30.0,
                read=timeout,
                write=30.0,
                pool=30.0,
            ),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        logger.info(
            "vLLM client initialized",
            base_url=base_url,
            model=model,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerationResponse:
        """Generate completion using vLLM."""
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
                "top_p": self.top_p,
            }

            logger.debug(
                "Sending generation request",
                model=self.model,
                prompt_length=len(user_prompt),
            )

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

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error: {e.response.status_code}"
            logger.error("vLLM request failed", error=error_msg)
            return GenerationResponse(success=False, error=error_msg)

        except httpx.RequestError as e:
            error_msg = f"Request error: {str(e)}"
            logger.error("vLLM request failed", error=error_msg)
            return GenerationResponse(success=False, error=error_msg)

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error("vLLM request failed", error=error_msg)
            return GenerationResponse(success=False, error=error_msg)

    def generate_with_history(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
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
            }

            response = self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()

            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]

            return GenerationResponse(
                success=True,
                content=message["content"],
                tokens_used=data.get("usage", {}).get("total_tokens", 0),
                finish_reason=choice.get("finish_reason", ""),
            )

        except Exception as e:
            logger.error("vLLM request failed", error=str(e))
            return GenerationResponse(success=False, error=str(e))

    def health_check(self) -> bool:
        """Check if vLLM server is healthy."""
        try:
            response = self.client.get(f"{self.base_url}/models")
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
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

