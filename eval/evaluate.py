"""Offline evaluation harness — Step 1: collect the pipeline's outputs.

For each test question we run our own RAG pipeline (retrieve -> generate) and record
three things RAGAS will score later:
  - user_input          the question
  - retrieved_contexts  the chunk texts we fed the LLM (the "evidence")
  - response            the answer the LLM produced

We use RAGAS's field names now so Step 2 can hand these records straight to it.

Run from the project root (with .venv active and GROQ_API_KEY set), after you've
POSTed your test video to /ingest:

    python -m eval.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import settings
from app.embeddings import embed_query
from app.generation import generate_answer
from app.ingestion import extract_video_id
from app.vectorstore import has_video, search

DATASET_PATH = Path(__file__).parent / "dataset.json"
RECORDS_PATH = Path(__file__).parent / "records.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def collect_records(url: str, questions: list[str], k: int = 5) -> list[dict]:
    """Run the pipeline over each question and gather what RAGAS needs to score it."""
    video_id = extract_video_id(url)
    if not has_video(video_id):
        raise SystemExit(f"Video {video_id} isn't indexed yet — POST it to /ingest first.")

    records: list[dict] = []
    for question in questions:
        hits = search(video_id, embed_query(question), k=k)
        records.append(
            {
                "user_input": question,
                "retrieved_contexts": [hit.text for hit in hits],
                "response": generate_answer(question, hits),
            }
        )
    return records


def format_results_md(scores: dict, n: int) -> str:
    """Render the aggregate metric scores as a small Markdown table for the README."""
    lines = [
        "# RAG evaluation (RAGAS)",
        "",
        f"Judged by Gemini over {n} question(s). Scores are 0-1; higher is better.",
        "",
        "| Metric | Average score |",
        "| --- | --- |",
    ]
    for name, value in scores.items():
        lines.append(f"| {name} | {value:.3f} |")
    return "\n".join(lines) + "\n"


def score_records(records: list[dict]) -> dict:
    """Score the collected records with RAGAS, using Gemini as the judge + embeddings.

    Metrics:
      - faithfulness      is the answer supported by the retrieved chunks? (LLM judge)
      - answer relevancy  does the answer actually address the question? (LLM + embeddings)
    Neither needs a "correct" reference answer, so our dataset is just questions.
    """
    if not settings.gemini_api_key:
        raise SystemExit("GEMINI_API_KEY is not set in your .env — RAGAS needs it as the judge.")

    import sys
    import types
    if 'langchain_community.chat_models.vertexai' not in sys.modules:
        dummy = types.ModuleType('langchain_community.chat_models.vertexai')
        dummy.ChatVertexAI = None
        sys.modules['langchain_community.chat_models.vertexai'] = dummy

    # Imported lazily so Step-1 collection still works before these are installed.
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy

    from ragas.run_config import RunConfig
    judge = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite", 
            google_api_key=settings.gemini_api_key,
            max_retries=10,
            timeout=120
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", google_api_key=settings.gemini_api_key
        )
    )

    dataset = EvaluationDataset.from_list(records)
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=judge),
            ResponseRelevancy(llm=judge, embeddings=judge_embeddings, strictness=1),
        ],
        raise_exceptions=True,
        run_config=RunConfig(max_workers=1, timeout=180, max_retries=10)
    )
    # to_pandas() gives one row per question; the numeric columns are the metric scores.
    df = result.to_pandas()
    return df.select_dtypes("number").mean().round(3).to_dict()


def main() -> None:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    records = collect_records(data["url"], data["questions"])
    RECORDS_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    for i, rec in enumerate(records, 1):
        print(f"Q{i}: {rec['user_input']}  ({len(rec['retrieved_contexts'])} chunks retrieved)")

    print("\nScoring with RAGAS (Gemini judge) — this makes several API calls, please wait...")
    scores = score_records(records)
    RESULTS_PATH.write_text(format_results_md(scores, len(records)), encoding="utf-8")

    print("\nScores:", scores)
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
