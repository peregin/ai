"""
Multi‑LLM Demo Service (FastAPI)
"""

from __future__ import annotations
import os
import asyncio
from typing import List, Literal, Optional, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Provider Interfaces ----------------------------------------------------
class LLMError(Exception):
    pass

class BaseLLM:
    provider: str

    async def acomplete(self, model: str, prompt: str, **kwargs) -> str:
        raise NotImplementedError

# Echo (mock) provider — useful for local testing without keys
class EchoLLM(BaseLLM):
    provider = "echo"

    async def acomplete(self, model: str, prompt: str, **kwargs) -> str:
        await asyncio.sleep(0)  # yield control
        return f"[echo:{model}] " + prompt[::-1]

# OpenAI provider
class OpenAILLM(BaseLLM):
    provider = "openai"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None
        if self.api_key:
            try:
                from openai import OpenAI  # type: ignore
                self._client = OpenAI(api_key=self.api_key)
            except Exception:
                self._client = None

    def available(self) -> bool:
        return bool(self.api_key and self._client)

    async def acomplete(self, model: str, prompt: str, **kwargs) -> str:
        if not self.available():
            raise LLMError("OpenAI not configured or package missing.")
        try:
            def _run() -> str:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", 0.7),
                )
                return resp.choices[0].message.content or ""

            return await asyncio.to_thread(_run)
        except Exception as e:
            raise LLMError(f"OpenAI error: {e}") from e

# Anthropic provider
class AnthropicLLM(BaseLLM):
    provider = "anthropic"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None
        if self.api_key:
            try:
                import anthropic  # type: ignore
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except Exception:
                self._client = None

    def available(self) -> bool:
        return bool(self.api_key and self._client)

    async def acomplete(self, model: str, prompt: str, **kwargs) -> str:
        if not self.available():
            raise LLMError("Anthropic not configured or package missing.")
        try:
            def _run() -> str:
                resp = self._client.messages.create(
                    model=model,
                    max_tokens=kwargs.get("max_tokens", 512),
                    temperature=kwargs.get("temperature", 0.7),
                    messages=[{"role": "user", "content": prompt}],
                )
                parts = []
                for block in getattr(resp, "content", []) or []:
                    text = getattr(block, "text", None)
                    if text:
                        parts.append(text)
                return "".join(parts)

            return await asyncio.to_thread(_run)
        except Exception as e:
            raise LLMError(f"Anthropic error: {e}") from e

# Ollama provider (local inference, no API key)
class OllamaLLM(BaseLLM):
    provider = "ollama"

    def __init__(self, base_url: Optional[str] = None, timeout: int = 60):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.timeout = timeout

    def available(self) -> bool:
        try:
            with httpx.Client(base_url=self.base_url, timeout=2.5) as c:
                r = c.get("/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    async def acomplete(self, model: str, prompt: str, **kwargs) -> str:
        # model may be of form "ollama/mistral" → pass "mistral" to Ollama
        if "/" in model:
            _, model = model.split("/", 1)
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": kwargs.get("temperature", 0.7)},
        }
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as c:
                r = await c.post("/api/generate", json=payload)
                r.raise_for_status()
                data = r.json()
                return data.get("response", "")
        except Exception as e:
            raise LLMError(f"Ollama error: {e}") from e

# --- Model Catalog ----------------------------------------------------------
# Map model id → provider and optional metadata
MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    # Always available mock
    "echo": {"provider": "echo", "display_name": "Echo (mock)", "description": "Reverses your prompt for testing."},

    # Example OpenAI chat models (adjust to what your account supports)
    "gpt-4o-mini": {"provider": "openai", "display_name": "OpenAI GPT-4o Mini"},
    "gpt-4o": {"provider": "openai", "display_name": "OpenAI GPT-4o"},

    # Example Anthropic models
    "claude-3-5-sonnet-latest": {"provider": "anthropic", "display_name": "Anthropic Claude 3.5 Sonnet"},
    "claude-3-5-haiku-latest": {"provider": "anthropic", "display_name": "Anthropic Claude 3.5 Haiku"},

    # Local models via Ollama (no API key)
    # Upstream weights correspond to mistralai/Mistral-7B; Ollama tag is "mistral".
    "ollama/mistral": {"provider": "ollama", "display_name": "Mistral 7B (Ollama)", "description": "Runs locally via Ollama; run 'ollama pull mistral' first."},
    "ollama/llama3.2:3b": {"provider": "ollama", "display_name": "Llama 3.2B (Ollama)", "description": "Runs locally via Ollama; run 'ollama pull ollama3.2:3b' first."},
}

# Instantiate providers (some may be disabled if no API keys)
PROVIDERS: Dict[str, BaseLLM] = {
    "echo": EchoLLM(),
    "openai": OpenAILLM(),
    "anthropic": AnthropicLLM(),
    "ollama": OllamaLLM(),
}

# --- FastAPI Schemas --------------------------------------------------------
class ModelInfo(BaseModel):
    id: str
    provider: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    available: bool = True

class ChatRequest(BaseModel):
    model: str = Field(..., description="Model id from /models")
    prompt: str = Field(..., description="User prompt")
    temperature: Optional[float] = Field(0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(512, gt=0, description="Used by some providers")

class ChatResponse(BaseModel):
    model: str
    provider: str
    output: str

# --- FastAPI App ------------------------------------------------------------
app = FastAPI(title="Multi‑LLM Demo Service", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/models", response_model=List[ModelInfo])
async def list_models() -> List[ModelInfo]:
    items: List[ModelInfo] = []
    for mid, meta in MODEL_CATALOG.items():
        provider_key = meta["provider"]
        provider = PROVIDERS.get(provider_key)
        available = True
        if hasattr(provider, "available"):
            try:
                available = provider.available()  # type: ignore[attr-defined]
            except Exception:
                available = False
        items.append(
            ModelInfo(
                id=mid,
                provider=provider_key,
                display_name=meta.get("display_name"),
                description=meta.get("description"),
                available=available,
            )
        )
    return items

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    meta = MODEL_CATALOG.get(req.model)
    if not meta:
        raise HTTPException(status_code=400, detail=f"Unknown model '{req.model}'. Call GET /models.")

    provider_key = meta["provider"]
    provider = PROVIDERS.get(provider_key)
    if provider is None:
        raise HTTPException(status_code=500, detail=f"Provider '{provider_key}' not available.")

    try:
        output = await provider.acomplete(
            model=req.model,
            prompt=req.prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    return ChatResponse(model=req.model, provider=provider_key, output=output)

# Health check
@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

# Allow `python multi_llm_service.py` to run a dev server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("multi_llm_service:app", host="0.0.0.0", port=8000, reload=True)

