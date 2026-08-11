"""Best-effort natural-language summary of a query result set.

Fully separate from SQL generation: its own prompt, its own LLM call, and it
never touches SQL generation, validation, or execution -- it only describes
data that has already been safely fetched. Optional (only runs if the
caller asks for it) and non-blocking: a failure here degrades to None
rather than breaking the primary answer, the same contract as the
confidence-interval enrichment in app.agents.confidence_interval.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from app.agents.llm_client import LLMClient

MAX_SUMMARIZED_ROWS = 20

SUMMARY_SYSTEM_PROMPT = """You summarize query results for a business audience in one or two sentences.

Rules, no exceptions:
- Base the summary only on the data shown below. Never state a number, trend, or comparison that isn't directly visible in this data.
- Do not perform new calculations beyond what's already in the table (e.g. don't compute a percent change unless both values needed for it are shown).
- Do not speculate about causes, context, or anything outside the table.
- Plain prose, one or two sentences. No markdown, no code fence, no preamble."""


def _format_result_for_prompt(df: pd.DataFrame) -> str:
    shown = df.head(MAX_SUMMARIZED_ROWS)
    table = shown.to_string(index=False)
    if len(df) > MAX_SUMMARIZED_ROWS:
        table += f"\n(showing first {MAX_SUMMARIZED_ROWS} of {len(df)} rows)"
    return table


def summarize_result(question: str, df: pd.DataFrame, llm_client: LLMClient) -> Optional[str]:
    """Returns a 1-2 sentence natural-language summary of `df`, or None on failure."""
    if df.empty:
        return None

    try:
        user_prompt = f"Question: {question}\n\nResult:\n{_format_result_for_prompt(df)}"
        summary = llm_client.generate(SUMMARY_SYSTEM_PROMPT, user_prompt)
        return summary.strip() or None
    except Exception:
        return None
