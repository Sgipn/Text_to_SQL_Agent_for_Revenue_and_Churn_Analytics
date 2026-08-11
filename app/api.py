"""FastAPI layer over the text-to-SQL agent.

A thin wrapper -- request validation, response shaping, and error mapping
only. All business logic lives in app.agents.text_to_sql_agent.

Run locally with:
    uvicorn app.api:app --reload
"""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from app.agents.llm_client import ClaudeLLMClient, LLMClient
from app.agents.text_to_sql_agent import answer_question

# Loaded eagerly at import time, not lazily inside the first LLM call (as
# ClaudeLLMClient does for ANTHROPIC_API_KEY) -- ASK_API_KEY must be in
# os.environ before the *first* request is handled, or a deployer who only
# sets it in .env (not a true OS/host env var) would have abuse protection
# silently disabled until after the first successful LLM call happened to
# trigger the lazy load elsewhere.
load_dotenv()

app = FastAPI(
    title="Semantic Metric Repository -- Text-to-SQL Agent",
    description="Ask a natural-language business question and get back validated, executed SQL.",
)

# --- Abuse protection for /ask ---------------------------------------------
# Every request triggers a real, billed Claude API call, so an internet-facing
# deployment needs *some* guardrail. Two layers, both free and dependency-free:
#
# 1. Optional shared API key (ASK_API_KEY env var). Unset by default so local
#    dev/tests are unaffected; set it before deploying anywhere public.
# 2. An in-memory sliding-window rate limit per client (the API key if one is
#    configured, else the caller's IP). In-memory is fine for a single-instance
#    deployment (e.g. Render's free tier); it resets on restart, which is an
#    acceptable tradeoff for a demo, not a production multi-instance service.
API_KEY_ENV_VAR = "ASK_API_KEY"
RATE_LIMIT_MAX_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60

_request_log: Dict[str, "deque[float]"] = defaultdict(deque)


def _check_rate_limit(client_id: str) -> None:
    now = time.monotonic()
    log = _request_log[client_id]
    while log and now - log[0] > RATE_LIMIT_WINDOW_SECONDS:
        log.popleft()
    if len(log) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {RATE_LIMIT_MAX_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECONDS}s.",
        )
    log.append(now)


def enforce_abuse_protection(request: Request, x_api_key: Optional[str] = Header(default=None)) -> None:
    """FastAPI dependency: require ASK_API_KEY (if configured), then rate-limit."""
    expected_key = os.environ.get(API_KEY_ENV_VAR)
    if expected_key:
        if x_api_key != expected_key:
            raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")
        client_id = x_api_key
    else:
        client_id = request.client.host if request.client else "unknown"

    _check_rate_limit(client_id)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    max_attempts: Optional[int] = Field(default=None, ge=1, le=5)
    summarize: bool = Field(
        default=False, description="Also generate a one-sentence summary of the result (extra LLM call)."
    )


class ConfidenceIntervalResponse(BaseModel):
    estimate: float
    standard_error: float
    lower: float
    upper: float
    n_units: int
    confidence_level: float


class AskResponse(BaseModel):
    question: str
    succeeded: bool
    attempts: int
    sql: Optional[str] = None
    error: Optional[str] = None
    rows: Optional[List[Dict[str, Any]]] = None
    row_count: Optional[int] = None
    confidence_interval: Optional[ConfidenceIntervalResponse] = None
    summary: Optional[str] = None


def get_llm_client() -> LLMClient:
    """FastAPI dependency, overridden in tests with a fake client."""
    return ClaudeLLMClient()


def _dataframe_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Converts a DataFrame to plain-Python-typed records, safe for JSON encoding."""
    records = []
    for row in df.to_dict(orient="records"):
        record = {}
        for key, value in row.items():
            if pd.isna(value):
                record[key] = None
            elif hasattr(value, "isoformat"):
                record[key] = value.isoformat()
            elif hasattr(value, "item"):
                record[key] = value.item()
            else:
                record[key] = value
        records.append(record)
    return records


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(enforce_abuse_protection)])
def ask(request: AskRequest, llm_client: LLMClient = Depends(get_llm_client)) -> AskResponse:
    kwargs: Dict[str, Any] = {"llm_client": llm_client, "summarize": request.summarize}
    if request.top_k is not None:
        kwargs["top_k"] = request.top_k
    if request.max_attempts is not None:
        kwargs["max_attempts"] = request.max_attempts

    try:
        result = answer_question(request.question, **kwargs)
    except (RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    rows = None
    row_count = None
    if result.result is not None:
        rows = _dataframe_to_records(result.result)
        row_count = len(result.result)

    confidence_interval = None
    if result.confidence_interval is not None:
        ci = result.confidence_interval
        confidence_interval = ConfidenceIntervalResponse(
            estimate=ci.estimate,
            standard_error=ci.standard_error,
            lower=ci.lower,
            upper=ci.upper,
            n_units=ci.n_units,
            confidence_level=ci.confidence_level,
        )

    return AskResponse(
        question=result.question,
        succeeded=result.succeeded,
        attempts=result.attempts,
        sql=result.sql,
        error=result.error,
        rows=rows,
        row_count=row_count,
        confidence_interval=confidence_interval,
        summary=result.summary,
    )
