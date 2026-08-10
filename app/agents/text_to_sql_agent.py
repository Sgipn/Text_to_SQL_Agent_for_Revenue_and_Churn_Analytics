"""Natural-language-to-SQL orchestration: retrieval -> LLM generation ->
validation -> execution, with a validation-guided retry loop.

See the "Agent orchestration" phase of IMPLEMENTATION_PLAN.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from app.agents.confidence_interval import compute_ratio_confidence_interval
from app.agents.llm_client import ClaudeLLMClient, LLMClient
from app.agents.prompt_builder import build_retry_prompt, build_system_prompt
from app.agents.sql_extraction import parse_llm_response
from app.services.metric_statistics import RatioConfidenceInterval
from app.services.query_execution import execute_safe_query
from app.services.retrieval import DEFAULT_TOP_K, format_context_for_prompt, retrieve_context
from app.services.sql_validation import UnsafeQueryError, parse_safe_select

DEFAULT_MAX_ATTEMPTS = 2


@dataclass
class AgentResult:
    question: str
    sql: Optional[str]
    result: Optional[pd.DataFrame]
    attempts: int
    error: Optional[str] = None
    confidence_interval: Optional[RatioConfidenceInterval] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


def answer_question(
    question: str,
    llm_client: Optional[LLMClient] = None,
    top_k: int = DEFAULT_TOP_K,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> AgentResult:
    """Answers a natural-language business question end to end.

    Retrieves grounding context, asks the LLM for SQL, and validates +
    executes it. If validation rejects the SQL, the error is fed back to
    the LLM and it gets another attempt (up to max_attempts) rather than
    failing on the first hallucinated join or disallowed table.
    """
    llm_client = llm_client or ClaudeLLMClient()

    context = retrieve_context(question, top_k=top_k)
    system_prompt = build_system_prompt(format_context_for_prompt(context))

    user_prompt = question
    last_sql: Optional[str] = None
    last_error: Optional[str] = None

    for attempt in range(1, max_attempts + 1):
        raw_response = llm_client.generate(system_prompt, user_prompt)
        sql, decline_reason = parse_llm_response(raw_response)

        if decline_reason is not None:
            return AgentResult(question=question, sql=None, result=None, attempts=attempt, error=decline_reason)

        last_sql = sql

        try:
            result = execute_safe_query(sql)
            statement = parse_safe_select(sql)  # already validated by execute_safe_query above
            confidence_interval = compute_ratio_confidence_interval(statement)
            return AgentResult(
                question=question,
                sql=sql,
                result=result,
                attempts=attempt,
                confidence_interval=confidence_interval,
            )
        except UnsafeQueryError as exc:
            last_error = str(exc)
            user_prompt = build_retry_prompt(question, sql, last_error)

    return AgentResult(question=question, sql=last_sql, result=None, attempts=max_attempts, error=last_error)
