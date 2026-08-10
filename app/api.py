"""FastAPI layer over the text-to-SQL agent.

A thin wrapper -- request validation, response shaping, and error mapping
only. All business logic lives in app.agents.text_to_sql_agent.

Run locally with:
    uvicorn app.api:app --reload
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agents.llm_client import ClaudeLLMClient, LLMClient
from app.agents.text_to_sql_agent import answer_question

app = FastAPI(
    title="Semantic Metric Repository -- Text-to-SQL Agent",
    description="Ask a natural-language business question and get back validated, executed SQL.",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    max_attempts: Optional[int] = Field(default=None, ge=1, le=5)


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


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, llm_client: LLMClient = Depends(get_llm_client)) -> AskResponse:
    kwargs: Dict[str, Any] = {"llm_client": llm_client}
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
    )
