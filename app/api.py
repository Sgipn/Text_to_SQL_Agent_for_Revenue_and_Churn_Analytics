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
from fastapi.responses import HTMLResponse
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


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Semantic Metric Repository</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, "Segoe UI", sans-serif; max-width: 760px; margin: 40px auto; padding: 0 16px; line-height: 1.5; }
  h1 { font-size: 1.4rem; margin-bottom: 4px; }
  .muted { color: #6b7280; font-size: 0.9rem; }
  label { display: block; margin-top: 16px; font-weight: 600; font-size: 0.9rem; }
  input[type=text], input[type=password], textarea {
    width: 100%; box-sizing: border-box; padding: 8px; font-size: 1rem;
    border: 1px solid #9ca3af; border-radius: 6px; margin-top: 4px; font-family: inherit;
  }
  textarea { min-height: 60px; resize: vertical; }
  .row { display: flex; align-items: center; gap: 8px; margin-top: 12px; }
  .row label { margin: 0; font-weight: normal; }
  button { margin-top: 16px; padding: 10px 20px; font-size: 1rem; border-radius: 6px; border: none; background: #2563eb; color: white; cursor: pointer; }
  button:disabled { opacity: 0.5; cursor: default; }
  pre { background: rgba(128,128,128,0.12); padding: 12px; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; word-break: break-word; }
  table { border-collapse: collapse; width: 100%; margin-top: 8px; display: block; overflow-x: auto; }
  th, td { border: 1px solid rgba(128,128,128,0.4); padding: 6px 10px; text-align: left; font-size: 0.9rem; white-space: nowrap; }
  .error { color: #dc2626; font-weight: 600; }
  #result { margin-top: 24px; }
  #result h3 { margin-bottom: 4px; font-size: 1rem; }
</style>
</head>
<body>
<h1>Semantic Metric Repository</h1>
<p class="muted">Ask a natural-language question about revenue or subscriber activity. Answers are grounded in an approved semantic layer -- generated SQL is validated before it ever touches the database.</p>

<label for="apiKey">API key</label>
<input type="password" id="apiKey" placeholder="X-API-Key" autocomplete="off">

<label for="question">Question</label>
<textarea id="question" placeholder="What is our Average Revenue per Membership?"></textarea>

<div class="row">
  <input type="checkbox" id="summarize">
  <label for="summarize">Also generate a natural-language summary</label>
</div>

<button id="askButton">Ask</button>

<div id="result"></div>

<script>
const apiKeyInput = document.getElementById('apiKey');
apiKeyInput.value = localStorage.getItem('askApiKey') || '';
apiKeyInput.addEventListener('input', () => localStorage.setItem('askApiKey', apiKeyInput.value));

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

document.getElementById('askButton').addEventListener('click', async () => {
  const question = document.getElementById('question').value.trim();
  const summarize = document.getElementById('summarize').checked;
  const resultEl = document.getElementById('result');
  const button = document.getElementById('askButton');

  if (!question) {
    resultEl.innerHTML = '<p class="error">Enter a question first.</p>';
    return;
  }

  button.disabled = true;
  resultEl.innerHTML = '<p class="muted">Asking... (first request after idle can take up to a minute)</p>';

  try {
    const response = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-API-Key': apiKeyInput.value },
      body: JSON.stringify({ question: question, summarize: summarize }),
    });

    if (response.status === 401) {
      resultEl.innerHTML = '<p class="error">Invalid or missing API key.</p>';
      return;
    }
    if (response.status === 429) {
      resultEl.innerHTML = '<p class="error">Rate limit exceeded. Try again shortly.</p>';
      return;
    }
    if (!response.ok) {
      resultEl.innerHTML = '<p class="error">Request failed (HTTP ' + response.status + ').</p>';
      return;
    }

    const data = await response.json();

    if (!data.succeeded) {
      resultEl.innerHTML = '<p class="error">Could not answer: ' + escapeHtml(data.error || 'unknown error') + '</p>';
      return;
    }

    let html = '';
    html += '<p class="muted">' + data.attempts + ' attempt(s)</p>';
    html += '<h3>SQL</h3><pre>' + escapeHtml(data.sql) + '</pre>';

    if (data.rows && data.rows.length) {
      const columns = Object.keys(data.rows[0]);
      const shown = data.rows.slice(0, 50);
      html += '<h3>Result (' + data.row_count + ' rows)</h3>';
      html += '<table><thead><tr>' + columns.map(function (c) { return '<th>' + escapeHtml(c) + '</th>'; }).join('') + '</tr></thead><tbody>';
      shown.forEach(function (row) {
        html += '<tr>' + columns.map(function (c) { return '<td>' + escapeHtml(String(row[c])) + '</td>'; }).join('') + '</tr>';
      });
      html += '</tbody></table>';
      if (data.row_count > 50) {
        html += '<p class="muted">Showing first 50 of ' + data.row_count + ' rows.</p>';
      }
    }

    if (data.confidence_interval) {
      const ci = data.confidence_interval;
      html += '<h3>' + Math.round(ci.confidence_level * 100) + '% Confidence Interval</h3><p>['
        + ci.lower.toFixed(4) + ', ' + ci.upper.toFixed(4) + '] (estimate=' + ci.estimate.toFixed(4)
        + ', se=' + ci.standard_error.toFixed(4) + ', n=' + ci.n_units + ')</p>';
    }

    if (data.summary) {
      html += '<h3>Summary</h3><p>' + escapeHtml(data.summary) + '</p>';
    }

    resultEl.innerHTML = html;
  } catch (err) {
    resultEl.innerHTML = '<p class="error">Network error: ' + escapeHtml(String(err)) + '</p>';
  } finally {
    button.disabled = false;
  }
});
</script>
</body>
</html>
"""


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """A minimal browser UI over /ask -- plain HTML/CSS/vanilla JS, no build
    step, no new dependency. Inlined as a string (not a separate static
    file) so it's guaranteed to be found regardless of whether the app is
    imported from the source tree or an installed package -- a static asset
    file would need its own packaging config to be reliably included."""
    return HTMLResponse(content=_INDEX_HTML)


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
