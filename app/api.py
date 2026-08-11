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
from app.services.ratio_metric_registry import RATIO_METRICS
from app.services.semantic_view_registry import ALLOWED_VIEWS
from app.utils.generate_synthetic_data import PLAN_NAMES, REGIONS

# Loaded eagerly at import time, not lazily inside the first LLM call (as
# ClaudeLLMClient does for ANTHROPIC_API_KEY) -- ASK_API_KEY must be in
# os.environ before the *first* request is handled, or a deployer who only
# sets it in .env (not a true OS/host env var) would have abuse protection
# silently disabled until after the first successful LLM call happened to
# trigger the lazy load elsewhere.
load_dotenv()

app = FastAPI(
    title="Text-to-SQL Agent for Revenue and Churn Analytics",
    description="Ask a natural-language business question and get back validated, executed SQL.",
)

# --- Abuse protection for /ask ---------------------------------------------
# Every request triggers a real, billed Claude API call, so an internet-facing
# deployment needs *some* guardrail. Three layers, all free and dependency-free:
#
# 1. Optional shared API key (ASK_API_KEY env var). Unset by default (and by
#    the deployed public UI, which has no field for one) so anyone can use the
#    site; set it if you ever want to gate access behind a shared secret again.
# 2. An in-memory sliding-window rate limit per client (the API key if one is
#    configured, else the caller's IP) -- bounds how fast any single visitor
#    can spend.
# 3. A global sliding-window cap shared across every client -- bounds total
#    spend for the whole deployment regardless of how requests are
#    distributed across IPs, which per-client limiting alone can't do once the
#    site has no per-user gate. Tune GLOBAL_RATE_LIMIT_MAX_REQUESTS to taste.
# All in-memory, so fine for a single-instance deployment (e.g. Render's free
# tier); state resets on restart, an acceptable tradeoff for a demo, not a
# production multi-instance service.
API_KEY_ENV_VAR = "ASK_API_KEY"
RATE_LIMIT_MAX_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60
GLOBAL_RATE_LIMIT_MAX_REQUESTS = 100
GLOBAL_RATE_LIMIT_WINDOW_SECONDS = 24 * 60 * 60

_request_log: Dict[str, "deque[float]"] = defaultdict(deque)
_global_request_log: "deque[float]" = deque()


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


def _check_global_rate_limit() -> None:
    now = time.monotonic()
    while _global_request_log and now - _global_request_log[0] > GLOBAL_RATE_LIMIT_WINDOW_SECONDS:
        _global_request_log.popleft()
    if len(_global_request_log) >= GLOBAL_RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=(
                f"This deployment has reached its cap of {GLOBAL_RATE_LIMIT_MAX_REQUESTS} "
                f"requests per {GLOBAL_RATE_LIMIT_WINDOW_SECONDS}s. Please try again later."
            ),
        )
    _global_request_log.append(now)


def enforce_abuse_protection(request: Request, x_api_key: Optional[str] = Header(default=None)) -> None:
    """FastAPI dependency: enforce the global spend cap, then ASK_API_KEY (if configured), then per-client rate limit."""
    _check_global_rate_limit()

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


# --- "What data is available" panel on the / page ---------------------------
# Built at request time straight from the same registries that drive query
# validation (semantic_view_registry.ALLOWED_VIEWS, ratio_metric_registry.
# RATIO_METRICS), not hand-typed prose -- so it can't drift out of sync with
# what the agent actually supports as fields or marts are added. Only the
# display labels below are hand-curated (for readability); anything not
# explicitly labeled falls back to a mechanical underscore-to-space transform.
#
# Fields are split into dimensions (what you filter/group by) and measures
# (what you actually ask about), shown as separate cards -- region and month
# used to be duplicated across both domain cards with no way to tell "how do
# I slice this" from "what can I measure" apart. Called "dimensions", not
# "identifiers": user_id is a true identifier in the raw data but is never
# exposed in either approved view (users are always aggregated away), so
# "identifier" would wrongly imply user-level lookup is possible.
_DOMAIN_LABELS = {
    "fct_monthly_subscriber_revenue": "Revenue",
    "fct_monthly_subscriber_activity": "Growth",
}
_DIMENSION_FIELDS = {"metric_month", "region_id", "plan_type"}
_DIMENSION_VALUES: Dict[str, List[str]] = {"region_id": REGIONS, "plan_type": PLAN_NAMES}
_FIELD_LABELS = {
    "metric_month": "month",
    "region_id": "region",
    "plan_type": "subscription plan",
    "active_paid_subscribers": "active paid subscribers",
    "total_net_revenue": "total net revenue",
    "active_subscribers": "active subscribers",
    "new_subscribers": "new subscribers",
    "churned_subscribers": "churned subscribers",
}
_METRIC_LABELS = {
    "average_revenue_per_membership": "ARM (Average Revenue per Membership)",
    "monthly_churn_rate": "monthly churn rate",
}


def _build_scope_html() -> str:
    views_by_dimension: Dict[str, List[str]] = {}
    for view_name, view in ALLOWED_VIEWS.items():
        for column in view.columns:
            if column in _DIMENSION_FIELDS:
                views_by_dimension.setdefault(column, []).append(view_name)

    dimension_terms = []
    for column in sorted(views_by_dimension):
        label = _FIELD_LABELS.get(column, column.replace("_", " "))
        extras = []
        if column in _DIMENSION_VALUES:
            extras.append(", ".join(_DIMENSION_VALUES[column]))
        views_with_column = views_by_dimension[column]
        if len(views_with_column) < len(ALLOWED_VIEWS):
            only = ", ".join(_DOMAIN_LABELS.get(v, v) for v in views_with_column)
            extras.append(only + " only")
        if extras:
            label += " (" + "; ".join(extras) + ")"
        dimension_terms.append(label)

    cards = [
        '<div class="scope-card">'
        '<p class="scope-domain">Filter &amp; group by</p>'
        '<p class="scope-fields">' + ", ".join(dimension_terms) + "</p>"
        "</div>"
    ]

    metrics_by_view: Dict[str, List[str]] = {}
    for metric_name, ratio_metric in RATIO_METRICS.items():
        view_name = ratio_metric.table.split(".")[-1]
        metrics_by_view.setdefault(view_name, []).append(metric_name)

    for view_name, view in ALLOWED_VIEWS.items():
        domain = _DOMAIN_LABELS.get(view_name, view_name)
        measures = sorted(c for c in view.columns if c not in _DIMENSION_FIELDS)
        fields = ", ".join(_FIELD_LABELS.get(c, c.replace("_", " ")) for c in measures)

        metric_line = ""
        metric_names = metrics_by_view.get(view_name)
        if metric_names:
            labels = ", ".join(_METRIC_LABELS.get(m, m.replace("_", " ")) for m in metric_names)
            metric_line = '<p class="scope-metric">Metric: ' + labels + "</p>"

        cards.append(
            '<div class="scope-card">'
            '<p class="scope-domain">' + domain + "</p>"
            '<p class="scope-fields">' + fields + "</p>" + metric_line + "</div>"
        )
    return "".join(cards)


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Text-to-SQL Agent for Revenue and Churn Analytics</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #D3D9D4; --surface: #FFFFFF; --text: #212A31; --text-dim: #4F6268;
    --accent: #124E66; --accent-soft: #DCE7EA; --rule: #C4CCC7; --good: #2E9E5B; --error: #D14343;
    --chart-1: #124E66; --chart-2: #3FA0C2; --chart-3: #748D92; --chart-4: #2E3944; --chart-5: #8FBFD1;
    --sans: "Inter", -apple-system, "Segoe UI", sans-serif;
    --display: "Space Grotesk", var(--sans);
    --mono: ui-monospace, "SF Mono", Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #212A31; --surface: #2E3944; --text: #D3D9D4; --text-dim: #93A2A8;
      --accent: #3FA0C2; --accent-soft: #1B3C47; --rule: #3A454E; --good: #4CC479; --error: #E5726B;
      --chart-1: #3FA0C2; --chart-2: #124E66; --chart-3: #93A2A8; --chart-4: #D3D9D4; --chart-5: #5B8A9E;
    }
  }
  * { box-sizing: border-box; }
  body {
    background: var(--bg); color: var(--text); font-family: var(--sans);
    max-width: 740px; margin: 0 auto; padding: 40px 20px 64px; line-height: 1.5;
  }
  .wordmark { font-family: var(--display); font-size: 1.3rem; font-weight: 700; margin: 0 0 4px; letter-spacing: -0.01em; }
  .lede { font-family: var(--sans); color: var(--text-dim); font-size: 0.88rem; margin: 0 0 26px; max-width: 58ch; }
  .card { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 18px 20px; }
  label {
    display: block; font-size: 0.78rem; font-weight: 600;
    letter-spacing: 0.01em; color: var(--text-dim); margin: 14px 0 6px;
  }
  label:first-child { margin-top: 0; }
  input[type=password], textarea {
    width: 100%; background: var(--bg); border: 1px solid var(--rule); border-radius: 4px;
    color: var(--text); font-family: var(--sans); font-size: 0.9rem; padding: 8px 10px;
  }
  textarea { min-height: 52px; resize: vertical; }
  input:focus, textarea:focus { outline: 2px solid var(--accent); outline-offset: 1px; border-color: transparent; }
  button {
    margin-top: 18px; font-family: var(--sans); font-size: 0.85rem; font-weight: 700;
    background: var(--accent); color: #ffffff; border: none; border-radius: 5px; padding: 9px 18px; cursor: pointer;
  }
  button:hover { filter: brightness(1.08); }
  button:disabled { opacity: 0.5; cursor: default; }
  .readout {
    margin-top: 28px; display: flex; flex-direction: column; gap: 1px;
    background: var(--rule); border: 1px solid var(--rule); border-radius: 6px; overflow: hidden;
  }
  .readout-row { background: var(--surface); padding: 14px 16px; }
  .row-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 8px; }
  .row-label { font-size: 0.76rem; font-weight: 600; letter-spacing: 0.01em; color: var(--text-dim); }
  .chip {
    font-size: 0.66rem; font-weight: 700; letter-spacing: 0.04em; white-space: nowrap;
    color: var(--good); background: color-mix(in srgb, var(--good) 15%, transparent);
    padding: 2px 8px; border-radius: 20px;
  }
  .chip.error-chip { color: var(--error); background: color-mix(in srgb, var(--error) 15%, transparent); }
  pre {
    font-family: var(--mono); font-size: 0.82rem; color: var(--text); margin: 0;
    white-space: pre-wrap; word-break: break-word; line-height: 1.55;
  }
  .chart-wrap { position: relative; height: 260px; margin-top: 2px; }
  table { border-collapse: collapse; width: 100%; display: block; overflow-x: auto; font-family: var(--mono); font-size: 0.85rem; }
  th, td { text-align: left; padding: 5px 8px 5px 0; font-variant-numeric: tabular-nums; white-space: nowrap; }
  th { color: var(--text-dim); font-family: var(--sans); font-weight: 600; font-size: 0.72rem; letter-spacing: 0.01em; }
  .ci-bar-wrap { display: flex; align-items: center; gap: 10px; }
  .ci-bar { position: relative; flex: 1; height: 5px; background: var(--rule); border-radius: 3px; }
  .ci-bar-fill { position: absolute; top: 0; bottom: 0; background: var(--accent); border-radius: 3px; }
  .ci-bar-mark { position: absolute; top: -3px; width: 2px; height: 11px; background: var(--text); border-radius: 1px; }
  .ci-num { font-family: var(--mono); font-size: 0.8rem; color: var(--text-dim); white-space: nowrap; font-variant-numeric: tabular-nums; }
  .ci-detail { margin: 8px 0 0; font-size: 0.78rem; color: var(--text-dim); font-family: var(--mono); }
  .summary-text { margin: 0; font-size: 0.92rem; line-height: 1.6; color: var(--text); }
  .status-text { color: var(--text-dim); font-size: 0.88rem; margin-top: 20px; }
  .status-text.error { color: var(--error); font-weight: 600; }
  .scope-panel { display: flex; gap: 12px; margin-bottom: 26px; flex-wrap: wrap; }
  .scope-card { flex: 1 1 220px; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 12px 14px; }
  .scope-domain { font-family: var(--display); font-size: 0.86rem; font-weight: 600; letter-spacing: 0; color: var(--accent); margin: 0 0 6px; }
  .scope-fields { font-size: 0.82rem; color: var(--text-dim); margin: 0; line-height: 1.5; }
  .scope-metric { font-size: 0.78rem; color: var(--text); margin: 8px 0 0; font-weight: 600; }
  .examples { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 0; }
  .example-chip {
    font-family: var(--sans); font-size: 0.78rem; font-weight: 600; color: var(--accent);
    background: transparent; border: 1px solid var(--accent); border-radius: 20px;
    padding: 5px 12px; cursor: pointer; text-align: left; margin: 0;
    transition: background-color 0.12s ease, color 0.12s ease;
  }
  .example-chip:hover, .example-chip:focus-visible { background: var(--accent); color: #ffffff; outline: none; }
</style>
</head>
<body>
<p class="wordmark">Text-to-SQL Agent for Revenue and Churn Analytics</p>
<p class="lede">Ask about <strong>revenue</strong> (by region, plan, or month) or <strong>subscriber growth and churn</strong> (by region or month). The exact fields available are listed below.</p>

<div class="scope-panel">__SCOPE_PANEL__</div>

<div class="card">
  <label for="question">Question</label>
  <textarea id="question" placeholder="What is the ARM for Basic plans in the US?"></textarea>

  <label>Try an example</label>
  <div class="examples">
    <button type="button" class="example-chip" data-question="What is our ARM by region?">ARM by region</button>
    <button type="button" class="example-chip" data-question="What was our monthly churn rate in APAC?">Churn rate in APAC</button>
    <button type="button" class="example-chip" data-question="Show me revenue by plan type in Q2 2024">Revenue by plan, Q2 2024</button>
    <button type="button" class="example-chip" data-question="Which region had the most new subscribers last quarter?">New subscribers by region</button>
  </div>

  <button id="askButton">Run query</button>
</div>

<div id="result"></div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script>
function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

document.querySelectorAll('.example-chip').forEach(function (chip) {
  chip.addEventListener('click', function () {
    const questionField = document.getElementById('question');
    questionField.value = chip.dataset.question;
    questionField.focus();
  });
});

let activeChart = null;

const PCT_NAME_PATTERN = /pct|percent|proportion|share/i;

function isNumericColumn(col, rows) {
  return rows.every(function (r) { return typeof r[col] === 'number'; });
}

// A column "sums to a whole" if its values across every returned row add up
// to ~100 (a percentage already scaled 0-100) or ~1 (a 0-1 fraction) --
// either shape means the rows are parts of one whole, which is what a pie
// chart communicates and a bar/line chart doesn't.
function sumsToWhole(col, rows) {
  const total = rows.reduce(function (sum, r) { return sum + r[col]; }, 0);
  return (total > 0.9 && total < 1.1) || (total > 90 && total < 110);
}

function detectChartSpec(columns, rows) {
  // Exactly one non-numeric column (the dimension) is required to chart at
  // all -- multi-dimension breakdowns stay table-only rather than guessing
  // which column to plot against. Up to two numeric columns are tolerated:
  // the model consistently returns both a raw aggregate and a percentage
  // column for "percentage of X" questions (e.g. active_subscribers *and*
  // pct_of_active_subscribers), not the percentage alone.
  if (rows.length < 2) return null;
  const numericCols = columns.filter(function (c) { return isNumericColumn(c, rows); });
  const dimCols = columns.filter(function (c) { return numericCols.indexOf(c) === -1; });
  if (dimCols.length !== 1 || numericCols.length === 0 || numericCols.length > 2) return null;

  const dimCol = dimCols[0];
  const pctCol = numericCols.find(function (c) { return PCT_NAME_PATTERN.test(c); })
    || numericCols.find(function (c) { return sumsToWhole(c, rows); });

  if (dimCol === 'metric_month') {
    return { type: 'line', dimCol: dimCol, measureCol: pctCol || numericCols[0] };
  }
  if (pctCol) {
    return { type: 'pie', dimCol: dimCol, measureCol: pctCol };
  }
  if (numericCols.length === 1) {
    return { type: 'bar', dimCol: dimCol, measureCol: numericCols[0] };
  }
  return null;
}

function renderChart(canvas, spec, rows) {
  const styles = getComputedStyle(document.body);
  const accent = styles.getPropertyValue('--accent').trim();
  const textDim = styles.getPropertyValue('--text-dim').trim();
  const text = styles.getPropertyValue('--text').trim();
  const rule = styles.getPropertyValue('--rule').trim();
  const sansFont = styles.getPropertyValue('--sans').trim();
  const slicePalette = [1, 2, 3, 4, 5].map(function (n) { return styles.getPropertyValue('--chart-' + n).trim(); });

  const sorted = rows.slice();
  if (spec.type === 'line') {
    sorted.sort(function (a, b) { return String(a[spec.dimCol]).localeCompare(String(b[spec.dimCol])); });
  } else {
    sorted.sort(function (a, b) { return b[spec.measureCol] - a[spec.measureCol]; });
  }

  const labels = sorted.map(function (r) { return String(r[spec.dimCol]); });
  const values = sorted.map(function (r) { return r[spec.measureCol]; });

  if (activeChart) { activeChart.destroy(); }

  if (spec.type === 'pie') {
    activeChart = new Chart(canvas, {
      type: 'pie',
      data: {
        labels: labels,
        datasets: [{
          label: spec.measureCol,
          data: values,
          backgroundColor: labels.map(function (_, i) { return slicePalette[i % slicePalette.length]; }),
          borderColor: styles.getPropertyValue('--surface').trim(),
          borderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'right', labels: { color: text, font: { family: sansFont } } },
        },
      },
    });
    return;
  }

  const isLine = spec.type === 'line';
  activeChart = new Chart(canvas, {
    type: isLine ? 'line' : 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: spec.measureCol,
        data: values,
        backgroundColor: isLine ? 'transparent' : accent,
        borderColor: accent,
        borderWidth: isLine ? 2 : 0,
        borderRadius: isLine ? 0 : 4,
        tension: 0.25,
        pointBackgroundColor: accent,
        pointRadius: isLine ? 3 : 0,
      }],
    },
    options: {
      indexAxis: isLine ? 'x' : 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: rule }, ticks: { color: textDim, font: { family: sansFont } } },
        y: { grid: { color: rule }, ticks: { color: textDim, font: { family: sansFont } } },
      },
    },
  });
}

function renderCiBar(ci) {
  const pad = (ci.upper - ci.lower) * 0.5 || Math.abs(ci.estimate) * 0.05 || 1;
  const domainMin = ci.lower - pad;
  const domainMax = ci.upper + pad;
  const span = domainMax - domainMin || 1;
  const leftPct = ((ci.lower - domainMin) / span) * 100;
  const rightPct = 100 - ((ci.upper - domainMin) / span) * 100;
  const markPct = ((ci.estimate - domainMin) / span) * 100;
  return '<div class="ci-bar-wrap">'
    + '<span class="ci-num">' + ci.lower.toFixed(4) + '</span>'
    + '<div class="ci-bar">'
    + '<div class="ci-bar-fill" style="left:' + leftPct.toFixed(2) + '%;right:' + rightPct.toFixed(2) + '%;"></div>'
    + '<div class="ci-bar-mark" style="left:' + markPct.toFixed(2) + '%;"></div>'
    + '</div>'
    + '<span class="ci-num">' + ci.upper.toFixed(4) + '</span>'
    + '</div>';
}

document.getElementById('askButton').addEventListener('click', async () => {
  const question = document.getElementById('question').value.trim();
  const resultEl = document.getElementById('result');
  const button = document.getElementById('askButton');

  if (!question) {
    resultEl.innerHTML = '<p class="status-text error">Enter a question first.</p>';
    return;
  }

  button.disabled = true;
  resultEl.innerHTML = '<p class="status-text">Running... (first request after idle can take up to a minute)</p>';

  if (activeChart) { activeChart.destroy(); activeChart = null; }

  try {
    const response = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: question, summarize: true }),
    });

    if (response.status === 401) {
      resultEl.innerHTML = '<p class="status-text error">This deployment requires an API key that this page has no way to provide. Ask the site owner to unset ASK_API_KEY.</p>';
      return;
    }
    if (response.status === 429) {
      const body = await response.json().catch(function () { return {}; });
      resultEl.innerHTML = '<p class="status-text error">' + escapeHtml(body.detail || 'Rate limit exceeded. Try again shortly.') + '</p>';
      return;
    }
    if (!response.ok) {
      resultEl.innerHTML = '<p class="status-text error">Request failed (HTTP ' + response.status + ').</p>';
      return;
    }

    const data = await response.json();

    if (!data.succeeded) {
      resultEl.innerHTML = '<div class="readout"><div class="readout-row">'
        + '<div class="row-head"><span class="row-label">Not answered</span><span class="chip error-chip">Declined</span></div>'
        + '<p class="summary-text">' + escapeHtml(data.error || 'Unknown error') + '</p>'
        + '</div></div>';
      return;
    }

    let rows = '';

    rows += '<div class="readout-row">'
      + '<div class="row-head"><span class="row-label">Generated SQL</span><span class="chip">&#10003; Validated &middot; ' + data.attempts + ' attempt(s)</span></div>'
      + '<pre>' + escapeHtml(data.sql) + '</pre>'
      + '</div>';

    let chartSpec = null;

    if (data.rows && data.rows.length) {
      const columns = Object.keys(data.rows[0]);
      chartSpec = detectChartSpec(columns, data.rows);

      if (chartSpec) {
        rows += '<div class="readout-row">'
          + '<div class="row-head"><span class="row-label">Chart</span></div>'
          + '<div class="chart-wrap"><canvas id="resultChart"></canvas></div>'
          + '</div>';
      }

      const shown = data.rows.slice(0, 50);
      rows += '<div class="readout-row">'
        + '<div class="row-head"><span class="row-label">Result</span><span class="chip">' + data.row_count + ' row(s)</span></div>'
        + '<table><thead><tr>' + columns.map(function (c) { return '<th>' + escapeHtml(c) + '</th>'; }).join('') + '</tr></thead><tbody>';
      shown.forEach(function (row) {
        rows += '<tr>' + columns.map(function (c) { return '<td>' + escapeHtml(String(row[c])) + '</td>'; }).join('') + '</tr>';
      });
      rows += '</tbody></table>';
      if (data.row_count > 50) {
        rows += '<p class="ci-detail">Showing first 50 of ' + data.row_count + ' rows.</p>';
      }
      rows += '</div>';
    }

    if (data.confidence_interval) {
      const ci = data.confidence_interval;
      rows += '<div class="readout-row">'
        + '<div class="row-head"><span class="row-label">' + Math.round(ci.confidence_level * 100) + '% Confidence Interval</span>'
        + '<span class="chip">n = ' + ci.n_units + '</span></div>'
        + renderCiBar(ci)
        + '<p class="ci-detail">estimate ' + ci.estimate.toFixed(4) + ' &middot; se ' + ci.standard_error.toFixed(4) + '</p>'
        + '</div>';
    }

    if (data.summary) {
      rows += '<div class="readout-row">'
        + '<div class="row-head"><span class="row-label">Summary</span></div>'
        + '<p class="summary-text">' + escapeHtml(data.summary) + '</p>'
        + '</div>';
    }

    resultEl.innerHTML = '<div class="readout">' + rows + '</div>';

    if (chartSpec) {
      renderChart(document.getElementById('resultChart'), chartSpec, data.rows);
    }
  } catch (err) {
    resultEl.innerHTML = '<p class="status-text error">Network error: ' + escapeHtml(String(err)) + '</p>';
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
    html = _INDEX_HTML.replace("__SCOPE_PANEL__", _build_scope_html())
    return HTMLResponse(content=html)


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
