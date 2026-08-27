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

from app.embeddings import embed_query
from app.generation import generate_answer
from app.ingestion import extract_video_id
from app.vectorstore import has_video, search

DATASET_PATH = Path(__file__).parent / "dataset.json"
RECORDS_PATH = Path(__file__).parent / "records.json"


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


def main() -> None:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    records = collect_records(data["url"], data["questions"])
    RECORDS_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")

    for i, rec in enumerate(records, 1):
        print(f"\nQ{i}: {rec['user_input']}")
        print(f"  retrieved {len(rec['retrieved_contexts'])} context chunks")
        print(f"  answer: {rec['response'][:200]}")
    print(f"\nSaved {len(records)} records to {RECORDS_PATH}")


if __name__ == "__main__":
    main()
