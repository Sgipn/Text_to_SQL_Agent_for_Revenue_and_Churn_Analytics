"""Registry of ratio metrics eligible for a Delta Method confidence interval.

Manually curated (not read from dbt's build artifacts at request time), for
the same reason as app.services.semantic_view_registry: confidence-interval
computation shouldn't depend on a fresh `dbt build` having run before it can
serve a request. tests/test_ratio_metric_registry.py cross-checks it against
the dbt manifest's ratio metric definitions to catch drift.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RatioMetric:
    numerator: str
    denominator: str
    table: str
    time_column: str


RATIO_METRICS: dict[str, RatioMetric] = {
    "average_revenue_per_membership": RatioMetric(
        numerator="total_net_revenue",
        denominator="active_paid_subscribers",
        table="semantic_views.fct_monthly_subscriber_revenue",
        time_column="metric_month",
    ),
    "monthly_churn_rate": RatioMetric(
        numerator="churned_subscribers",
        denominator="active_subscribers",
        table="semantic_views.fct_monthly_subscriber_activity",
        time_column="metric_month",
    ),
}
