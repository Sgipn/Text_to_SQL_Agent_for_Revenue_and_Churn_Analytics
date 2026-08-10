"""Best-effort Delta Method confidence interval for ratio-metric questions.

When the validated query references both the numerator and denominator
columns of a defined ratio metric (e.g. ARM's total_net_revenue and
active_paid_subscribers), this builds a companion query -- grouped by the
metric's time column, reusing the original query's WHERE clause -- executes
it safely, and computes a confidence interval. The WHERE clause is reused
verbatim rather than inspecting the LLM's SELECT/GROUP BY shape, so this
works regardless of how the model phrased its query.

Detection is column-based rather than retrieval-based: an early version
gated on retrieval's top result being the ratio metric's own doc, but live
testing showed that doc doesn't reliably rank #1 (a query naming specific
dimension values, e.g. "Premium plans in EMEA", can rank the schema doc
higher, since those values appear in its column descriptions). Checking
what the validated query actually references is ground truth instead of a
proxy for it. The check only looks at the SELECT list, not the whole
statement -- a query that merely filters on both columns (e.g. WHERE
active_paid_subscribers > 10 AND total_net_revenue > 100) without
displaying or computing anything from them isn't "about" the ratio metric.

Failure anywhere in this path (too few periods of history, an incompatible
filter, etc.) is deliberately non-fatal: callers get None back and the
primary answer is unaffected. That's why the broad except below is
intentional, not an oversight -- this is enrichment, not the safety-critical
path.
"""
from __future__ import annotations

from typing import Optional

from sqlglot import exp

from app.services.metric_statistics import RatioConfidenceInterval, ratio_confidence_interval
from app.services.query_execution import execute_safe_query
from app.services.ratio_metric_registry import RATIO_METRICS, RatioMetric


def _matching_ratio_metric_name(statement: exp.Select) -> Optional[str]:
    selected_columns = set()
    for projection in statement.expressions:
        selected_columns.update(column.name for column in projection.find_all(exp.Column))

    for name, metric in RATIO_METRICS.items():
        if metric.numerator in selected_columns and metric.denominator in selected_columns:
            return name
    return None


def _build_breakdown_sql(where: Optional[exp.Where], metric: RatioMetric) -> str:
    query = exp.select(
        metric.time_column,
        f"SUM({metric.numerator}) AS {metric.numerator}",
        f"SUM({metric.denominator}) AS {metric.denominator}",
    ).from_(metric.table)

    if where is not None:
        query = query.where(where.this)

    query = query.group_by(metric.time_column).order_by(metric.time_column)
    return query.sql(dialect="duckdb")


def compute_ratio_confidence_interval(validated_statement: exp.Select) -> Optional[RatioConfidenceInterval]:
    """Returns a Delta Method CI for the query's ratio metric, or None."""
    metric_name = _matching_ratio_metric_name(validated_statement)
    if metric_name is None:
        return None

    metric = RATIO_METRICS[metric_name]
    try:
        breakdown_sql = _build_breakdown_sql(validated_statement.args.get("where"), metric)
        breakdown = execute_safe_query(breakdown_sql)
        return ratio_confidence_interval(breakdown[metric.numerator], breakdown[metric.denominator])
    except Exception:
        return None
