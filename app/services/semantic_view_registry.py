"""Registry of approved semantic views that generated SQL may query.

This is the single source of truth used by sql_validation. It is manually
curated (not read from dbt's build artifacts at request time) so query
validation never depends on a fresh `dbt build` having run before it can
serve a request. tests/test_semantic_view_registry.py cross-checks it
against the dbt manifest to catch drift when marts change.
"""
from __future__ import annotations

ALLOWED_VIEWS: dict[str, str] = {
    "fct_monthly_subscriber_revenue": "semantic_views",
}
