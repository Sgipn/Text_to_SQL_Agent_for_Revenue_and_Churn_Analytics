"""Registry of approved semantic views -- and their approved columns --
that generated SQL may query.

This is the single source of truth used by sql_validation for both
table-level and column-level scope checks. It is manually curated (not
read from dbt's build artifacts at request time) so query validation never
depends on a fresh `dbt build` having run before it can serve a request.
tests/test_semantic_view_registry.py cross-checks it against the dbt
manifest's tags and documented columns to catch drift when marts change.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticView:
    schema: str
    columns: frozenset[str]


ALLOWED_VIEWS: dict[str, SemanticView] = {
    "fct_monthly_subscriber_revenue": SemanticView(
        schema="semantic_views",
        columns=frozenset(
            {
                "metric_month",
                "region_id",
                "plan_type",
                "active_paid_subscribers",
                "total_net_revenue",
            }
        ),
    ),
    "fct_monthly_subscriber_activity": SemanticView(
        schema="semantic_views",
        columns=frozenset(
            {
                "metric_month",
                "region_id",
                "active_subscribers",
                "new_subscribers",
                "churned_subscribers",
            }
        ),
    ),
}
