# CAG Demo — Cache-Augmented Generation over an HR Policy Document

A small, runnable project for understanding **Cache-Augmented Generation (CAG)**:
instead of retrieving relevant chunks per-query like RAG, you load an entire
document into the model's context **once**, cache it server-side, and then ask
many questions against that cache — cheaper and faster than resending the
whole document on every call.

The demo uses `data/HR_Leave_Policy_and_Encashment_Guidelines.docx` as the
knowledge source and answers questions about the leave policy.

## How CAG differs from RAG (in this project)

| | RAG | CAG (this project) |
|---|---|---|
| Knowledge source | Chunked, embedded, retrieved per query | Whole document, loaded once |
| What changes per query | Which chunks get retrieved | Nothing — same cached context every time |
| Best for | Large corpora that don't fit in context | Small/medium docs that fit in one context window |

## Project layout

```
CAG/
├── data/HR_Leave_Policy_and_Encashment_Guidelines.docx
├── src/
│   ├── core/          # config.py (.env -> Settings), document_loader.py (docling)
│   ├── providers/     # CacheProvider interface + Gemini / OpenAI implementations
│   ├── services/      # cag_service.py: load doc once -> build cache once -> ask()
│   └── api/           # FastAPI app, singleton dependency, chat controller, schemas
├── ui/streamlit_app.py  # Streamlit chat UI (talks to the API over HTTP)
├── tests/             # API tests with a fake provider
├── docs/ARCHITECTURE.md # full architecture + usage guide
├── main.py            # starts the API server (or --ask for one-shot CLI)
└── requirements.txt, .env.example
```

**See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the architecture,
diagrams, API reference and design notes.**

### Quick start

```bash
python main.py                        # terminal 1: FastAPI on http://127.0.0.1:8000 (docs at /docs)
streamlit run ui/streamlit_app.py     # terminal 2: chat UI on http://localhost:8501
```

## How the two providers actually cache (this is the point of the demo)

- **Gemini — explicit caching.** The app calls `client.caches.create(...)` once
  with the document, gets back a cache handle, and every subsequent
  `generate_content` call passes `cached_content=<handle>` instead of the
  document text. The response's `usage_metadata.cached_content_token_count`
  tells you how many tokens were served from cache.
- **OpenAI — implicit caching.** There's no cache object. OpenAI automatically
  caches the longest matching prefix of a prompt across calls once it's long
  enough. The app just puts the document in the system prompt as a static
  prefix that never changes between questions, and reads
  `usage.prompt_tokens_details.cached_tokens` off the response to see the effect.

Both are exposed through the same `CacheProvider.create_cache()` /
`CacheProvider.ask()` interface so the rest of the app doesn't care which one
is active.

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate   # Windows Git Bash / WSL
   # or: .venv\Scripts\Activate.ps1   (PowerShell)
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Edit `.env` and fill in **one** API key:
   ```
   GEMINI_API_KEY=your-key-here
   # or
   OPENAI_API_KEY=your-key-here
   ```

### Which provider gets used?

`src/config.py` decides automatically:

1. If `LLM_PROVIDER` is explicitly set to `gemini` or `openai`, that wins.
2. Otherwise, if `GEMINI_API_KEY` is non-empty, Gemini is used.
3. Otherwise, if `OPENAI_API_KEY` is non-empty, OpenAI is used.
4. If neither key is set, the app exits with a clear error telling you to set one.

So switching providers is just editing `.env` — no code changes needed.

Optional overrides in `.env`:
```
GEMINI_MODEL=gemini-3.6-flash   # default
OPENAI_MODEL=gpt-4o-mini        # default
```

## Running it

1. Start the API (loads the document and builds the cache once at startup):
   ```bash
   python main.py
   ```
   ```
   [INFO] src.services.cag_service: Provider: openai | Model: gpt-4o-mini
   [INFO] src.services.cag_service: Document loaded (~1425 words). Building cache...
   [INFO] src.services.cag_service: Cache ready (mode: implicit).
   INFO:     Uvicorn running on http://127.0.0.1:8000
   ```
2. Start the UI in a second terminal and chat in the browser:
   ```bash
   streamlit run ui/streamlit_app.py
   ```
   Each answer shows latency and input/cached/total tokens. The sidebar shows
   the provider, cache mode, and session totals.
3. Or call the API directly:
   ```bash
   curl -X POST http://127.0.0.1:8000/api/v1/chat -H "Content-Type: application/json" \
        -d '{"question": "How many days of casual leave per year?"}'
   ```
   Endpoints: `GET /api/v1/health`, `POST /api/v1/chat`, `GET /api/v1/stats`.
   Swagger UI is at `/docs`.

One-shot mode (useful for scripting/testing):
```bash
python main.py --ask "How is unused earned leave encashed?"
```

## Reading the stats line

- `input_tokens` — total tokens sent as input for that call.
- `cached_tokens` — how many of those input tokens were served from the cache
  instead of being freshly processed. Higher is better; this is the actual
  cost/latency saving CAG gives you.
- `total_tokens` — input + output tokens for that call.

Ask several different questions in one session and watch `cached_tokens` stay
high on every call after the first — the document is only "read" once.

## Troubleshooting

- **Gemini: `TotalCachedContentStorageTokensPerModelFreeTier limit=0`** — the
  Gemini free tier does not allow explicit context caching for some models.
  The app detects this and any other cache-creation failure automatically and
  falls back to sending the document directly on every call (still works,
  just without the cache-hit token savings). You'll see a `[CAG] Warning:
  Gemini explicit caching unavailable...` line when this happens.
- **Gemini: model 404 / "no longer available"** — Google occasionally retires
  model names. Update `GEMINI_MODEL` in `.env` to whatever current model name
  the error message suggests.
- **`docling` install is slow / large** — it pulls in ML dependencies for
  parsing PDFs/scanned docs. That's expected; it only runs once per document
  load here since we only load one `.docx`.
