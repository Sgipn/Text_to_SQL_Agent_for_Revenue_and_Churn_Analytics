"""Retrieves top-k schema/metric context for a natural-language prompt.

Grounds the text-to-SQL agent (Phase 5) in the approved semantic views and
metric definitions instead of letting it guess table/column names or
reimplement metric formulas -- see the "Retrieval and grounding layer"
phase of IMPLEMENTATION_PLAN.md.
"""
from __future__ import annotations

from app.services.vector_store import get_collection

DEFAULT_TOP_K = 3


def retrieve_context(question: str, top_k: int = DEFAULT_TOP_K, collection=None) -> list[dict]:
    """Returns up to top_k context documents most relevant to `question`.

    Each result is {"id", "text", "metadata", "distance"}, ordered by
    ascending distance (most relevant first).
    """
    collection = collection or get_collection()
    if collection.count() == 0:
        raise RuntimeError(
            "Vector store is empty. Run `python -m app.services.vector_store` to build the index."
        )

    results = collection.query(query_texts=[question], n_results=min(top_k, collection.count()))

    return [
        {"id": id_, "text": text, "metadata": metadata, "distance": distance}
        for id_, text, metadata, distance in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def format_context_for_prompt(results: list[dict]) -> str:
    """Renders retrieved documents as a single block for injection into an LLM system prompt."""
    return "\n\n---\n\n".join(r["text"] for r in results)
