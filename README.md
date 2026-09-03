---
title: Tube AI RAG API
emoji: 🎥
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# 🎥 → 🧠 → 💬 Tube AI RAG API

Ask questions about any YouTube video without watching the whole thing. Paste a link, and the API pulls the transcript, indexes it, and answers your questions with **citations back to the exact timestamp** — plus auto-generated quizzes and summaries.

Built to demonstrate a production-minded **Retrieval-Augmented Generation (RAG)** pipeline evaluated with RAGAS.

> **Status: v1 complete.** All pipeline stages built, tested, evaluated, and containerised.

---

## Demo

```bash
# Ingest a video
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=TgF-uMvhNmg"}'

# Ask a question — get a grounded answer with timestamp citations
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=TgF-uMvhNmg", "question": "What is the main topic?"}'
```

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI + Pydantic | Async, typed, auto OpenAPI docs |
| Transcripts | youtube-transcript-api (+ pasted-text fallback) | Survives IP blocks from cloud hosts |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Local, free, no API key needed |
| Vector DB | Chroma (persistent) | Zero-config local storage |
| Generation | Groq (Llama 3, OpenAI-compatible) | Free tier, very fast inference |
| Evaluation | RAGAS + Gemini judge | Faithfulness + answer relevancy |
| Packaging | Docker | One-command reproducible deployment |

---

## Evaluation Results

Scored with [RAGAS](https://docs.ragas.io/) using Gemini as the judge over 4 questions:

| Metric | Score | What it means |
|---|---|---|
| **Faithfulness** | **0.95 / 1.0** | Answers are grounded in the transcript, not hallucinated |
| **Answer Relevancy** | **0.59 / 1.0** | Answers address what was actually asked |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check + config status |
| `POST` | `/ingest` | Fetch transcript → chunk → embed → store |
| `POST` | `/ask` | Retrieve relevant chunks → grounded answer with citations |
| `POST` | `/summary` | Timestamped bullet-point summary of the whole video |
| `POST` | `/quiz` | Auto-generated multiple-choice quiz as structured JSON |

Full interactive docs at **`/docs`** (Swagger UI) when running locally.

---

## Quickstart

### Local (Python)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env   # add your GROQ_API_KEY (free at console.groq.com)

# 3. Run
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000/docs** for the interactive API.

### Docker

```bash
# Build once
docker build -t yt-ai-rag .

# Run (pass your .env file for API keys)
docker run -p 8000:8000 --env-file .env yt-ai-rag
```

---

## Project Structure

```text
yt-ai-rag/
├── app/
│   ├── config.py         # settings loaded from .env via pydantic-settings
│   ├── ingestion.py      # YouTube transcript fetch + pasted-text fallback
│   ├── chunking.py       # overlapping word-window chunker with timestamps
│   ├── embeddings.py     # sentence-transformers wrapper
│   ├── vectorstore.py    # Chroma persistence + search
│   ├── generation.py     # Groq LLM calls (answer / summary / quiz)
│   └── main.py           # FastAPI app + all endpoints
├── eval/
│   ├── dataset.json      # evaluation Q&A pairs
│   ├── evaluate.py       # RAGAS scoring harness
│   └── results.md        # latest evaluation scores
├── tests/
│   ├── test_ingestion.py # unit tests: URL parsing, timestamps, cleaning
│   ├── test_chunking.py  # unit tests: chunking logic and edge cases
│   └── test_api.py       # integration tests: all endpoints (mocked)
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── .env.example
```

---

## RAG Pipeline

```text
YouTube URL
    │
    ▼
[1] Ingestion    youtube-transcript-api → clean segments with timestamps
    │
    ▼
[2] Chunking     120-word overlapping windows, timestamp preserved per chunk
    │
    ▼
[3] Embedding    all-MiniLM-L6-v2 → 384-dim vectors (local, no API key)
    │
    ▼
[4] Storage      Chroma persistent vector store (keyed by video_id)
    │
    ▼
[5] Retrieval    cosine similarity search → top-k chunks
    │
    ▼
[6] Generation   Groq LLM — grounded prompt → answer with [M:SS] citations
```

---

## Running Tests

```bash
pytest tests/ -v
# 50 tests, ~1 second, no network or API keys required (all external calls mocked)
```

---

## v1 Roadmap

- [x] Runnable API skeleton (`/health`, `/docs`)
- [x] Transcript ingestion + pasted-text fallback
- [x] Chunking + local embeddings (all-MiniLM-L6-v2)
- [x] Chroma vector store + `/ingest` pipeline
- [x] Retrieval + grounded `/ask` with timestamp citations
- [x] `/quiz` + `/summary` endpoints
- [x] RAGAS evaluation (faithfulness: 0.95, answer relevancy: 0.59)
- [x] 50 pytest tests (unit + integration, fully mocked)
- [x] Docker containerisation

_v2: React UI + streaming responses + live public demo._

---

## License

MIT
