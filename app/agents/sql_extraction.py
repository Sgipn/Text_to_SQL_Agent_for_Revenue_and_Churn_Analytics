"""Extracts a single SQL statement -- or an explicit decline -- from raw LLM output."""
from __future__ import annotations

import re
from typing import Optional, Tuple

_SQL_FENCE_RE = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
NO_QUERY_PREFIX = "NO_QUERY:"


def extract_sql(text: str) -> str:
    """Pulls SQL out of the last ```sql fenced code block in `text`.

    Falls back to the raw text (stripped) if the model didn't fence its
    response at all.
    """
    matches = _SQL_FENCE_RE.findall(text)
    candidate = matches[-1] if matches else text
    return candidate.strip()


def parse_llm_response(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Returns (sql, decline_reason); exactly one of the two is None.

    The prompt instructs the model to prefix a response with NO_QUERY: and
    skip the code fence when a question can't be answered from the approved
    views/metrics, rather than inventing a workaround query -- see
    prompt_builder.SYSTEM_PROMPT_TEMPLATE.
    """
    stripped = text.strip()
    if stripped.startswith(NO_QUERY_PREFIX):
        return None, stripped[len(NO_QUERY_PREFIX) :].strip()
    return extract_sql(text), None
