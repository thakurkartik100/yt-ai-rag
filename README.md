# 🎥 → 🧠 → 💬  Tube AI RAG API

Ask questions about any YouTube video without watching the whole thing. Paste a
link, and the API pulls the transcript, indexes it, and answers your questions —
with **citations back to the exact timestamp** — plus auto-generated quizzes and
summaries.

Built to demonstrate a production-minded **Retrieval-Augmented Generation (RAG)**
pipeline: transcript ingestion → chunking → embeddings → vector search →
grounded generation → evaluation.

> **Status:** v1 in progress. This README's roadmap is checked off as features land.

---

## Why this exists

Long videos hide their value. This turns a video into searchable, cited,
question-answerable knowledge — and does it on **free-tier infrastructure** with
a **live public demo** so anyone can try it.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI + Pydantic | Async, typed, auto OpenAPI docs |
| Transcripts | youtube-transcript-api (+ fallback) | Simplest ingestion; fallback survives IP blocks |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Local, free, no API key or rate limits |
| Vector DB | Chroma (persistent) | Zero-config local; easy to host |
| Generation | Groq (Llama 3.x, OpenAI-compatible) | Free tier, very fast |
| Evaluation | RAGAS | Faithfulness / relevance / context precision |
| Packaging | Docker | Reproducible, deploy-anywhere |

## Roadmap (v1)

- [x] Runnable API skeleton (`/health`, `/docs`)
- [x] Transcript ingestion + fallback
- [ ] Chunking + local embeddings
- [ ] Chroma vector store + `/ingest`
- [ ] Retrieval + grounded `/ask` (streaming, timestamp citations)
- [ ] `/quiz` + `/summary`
- [ ] RAGAS evaluation + results table
- [ ] Docker + live demo + tests/CI

_v2 (later): a polished React UI._

## Quickstart

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your environment
cp .env.example .env             # then open .env and add your free GROQ_API_KEY

# 4. Run the API
uvicorn app.main:app --reload
```

Now open **http://127.0.0.1:8000/docs** — you should see the interactive API,
and `GET /health` should return `{"status": "ok", ...}`.

## Project structure

```
tube-ai-rag/
├── app/
│   ├── __init__.py
│   ├── config.py       # settings loaded from .env
│   └── main.py         # FastAPI app + endpoints
├── tests/              # pytest tests (added as features land)
├── requirements.txt
├── .env.example
└── README.md
```

## License

MIT (add a LICENSE file before publishing).
