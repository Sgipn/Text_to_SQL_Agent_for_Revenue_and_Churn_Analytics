"""Builds the system/retry prompts for the text-to-SQL agent, grounded in
retrieved semantic-view and metric context.
"""
from __future__ import annotations

from app.services.semantic_view_registry import ALLOWED_VIEWS

SYSTEM_PROMPT_TEMPLATE = """You are a text-to-SQL agent for a subscription analytics semantic layer.

Rules, no exceptions:
- Write exactly one SQL SELECT statement. Never write DROP, INSERT, UPDATE, DELETE, or any other statement type.
- Only reference these approved semantic views, using their exact "schema.table" name: {allowed_views}
- Ratio metrics (like Average Revenue per Membership) are non-additive. Always compute them as SUM(numerator) / SUM(denominator) over the relevant rows -- never AVG() a per-row ratio, and never average monthly ratio values to get a quarterly one.
- If the question cannot be answered from the views and metrics listed below (e.g. it asks for a metric that isn't defined here, like LTV or Net Promoter Score), do not invent a workaround query. Respond with exactly: NO_QUERY: <one sentence explaining what's missing>. Do not use a code fence in that case.
- Otherwise, respond with only the SQL query, inside a single ```sql code fence. No explanation before or after.

Relevant schema and metric context for this question:

{context_block}
"""

RETRY_PROMPT_TEMPLATE = """Question: {question}

Your previous SQL was rejected by the query validator:
{invalid_sql}

Validation error: {error}

Fix the query and respond again with only a single ```sql code fence."""


def build_system_prompt(context_block: str) -> str:
    allowed_views = ", ".join(f"{schema}.{table}" for table, schema in ALLOWED_VIEWS.items())
    return SYSTEM_PROMPT_TEMPLATE.format(allowed_views=allowed_views, context_block=context_block)


def build_retry_prompt(question: str, invalid_sql: str, error: str) -> str:
    return RETRY_PROMPT_TEMPLATE.format(question=question, invalid_sql=invalid_sql, error=error)
