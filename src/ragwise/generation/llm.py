"""LLMProtocol, StreamingLLMProtocol, resolve_llm(), and concrete LLM adapters."""
from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMProtocol(Protocol):
    async def complete(self, prompt: str) -> str:
        ...


@runtime_checkable
class StreamingLLMProtocol(LLMProtocol, Protocol):
    """Extended protocol for LLMs that support token streaming."""

    def stream_complete(self, prompt: str) -> AsyncGenerator[str, None]:
        ...


def resolve_llm(spec: str | Any) -> LLMProtocol:
    """Return an LLM instance from a string shorthand or pass-through.

    Supported prefixes:
      - ``"openai/*"``     → OpenAILLM
      - ``"anthropic/*"``  → AnthropicLLM
      - ``"ollama/*"``     → OllamaLLM

    Objects with a ``.complete()`` method are returned as-is.
    """
    if not isinstance(spec, str):
        if hasattr(spec, "complete"):
            return spec  # type: ignore[no-any-return]
        raise ValueError(f"Expected a string spec or LLM object, got {type(spec)!r}")

    parts = spec.split("/", 1)
    prefix = parts[0].lower()
    model = parts[1] if len(parts) > 1 else ""

    if prefix == "openai":
        return OpenAILLM(model=model or "gpt-4o-mini")
    if prefix == "anthropic":
        return AnthropicLLM(model=model or "claude-haiku-4-5-20251001")
    if prefix == "ollama":
        return OllamaLLM(model=model or "llama3")

    raise ValueError(f"Unknown LLM spec: {spec!r}. Expected 'openai/*', 'anthropic/*', or 'ollama/*'.")


class OpenAILLM:
    """LLM adapter for OpenAI chat completions (lazy client init)."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self.model = model
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def complete(self, prompt: str) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        return content if content is not None else ""

    async def stream_complete(self, prompt: str) -> AsyncGenerator[str, None]:
        client = self._get_client()
        stream = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token


class AnthropicLLM:
    """LLM adapter for Anthropic messages API (lazy client init)."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None) -> None:
        self.model = model
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def complete(self, prompt: str) -> str:
        client = self._get_client()
        response = await client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text  # type: ignore[no-any-return]

    async def stream_complete(self, prompt: str) -> AsyncGenerator[str, None]:
        client = self._get_client()
        async with client.messages.stream(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text


class OllamaLLM:
    """LLM adapter for Ollama local inference via HTTP."""

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def complete(self, prompt: str) -> str:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=120.0,
            )
            response.raise_for_status()
            return response.json()["response"]  # type: ignore[no-any-return]

    async def stream_complete(self, prompt: str) -> AsyncGenerator[str, None]:
        import json

        import httpx

        async with httpx.AsyncClient() as client, client.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": True},
            timeout=120.0,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
