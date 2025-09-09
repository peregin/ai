Simple REST service that provides multiple models for chat
---

Features
- GET /models               → list available models
- POST /chat {model,prompt} → run a single-turn chat on a chosen model

Providers supported out of the box
- OpenAI (chat.completions)
- Anthropic (Messages API)
- Ollama (local, no API key; e.g., mistral-7B)
- Echo (mock, always available)

Setup
1) Python 3.10+
2) pip install -U fastapi uvicorn pydantic openai anthropic httpx python-dotenv
3) Export any keys you have (optional):
   export OPENAI_API_KEY=... 
   export ANTHROPIC_API_KEY=...
4) (Optional) For local models with **Ollama**: install from https://ollama.ai and run:
   ollama pull mistral    # downloads Mistral 7B locally
5) Run
```bash
    poetry run uvicorn service:app --app-dir src --reload --port 8000
```

Example
GET  http://localhost:8000/models
POST http://localhost:8000/chat
Body: {"model":"echo","prompt":"Hello there"}
Also works locally with Ollama:
Body: {"model":"ollama/mistral","prompt":"Explain diffusion models simply"}

```bash
    curl -X POST 'http://localhost:8000/chat' -H 'Content-Type: application/json' -d '{"model":"ollama/mistral","prompt":"Explain diffusion models simply"}'
```

Notes
- This is minimal and synchronous-per-request (no streaming). Add streaming as needed.
- Edit MODEL_CATALOG below to reflect the models you actually want to expose.