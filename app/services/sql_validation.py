"""SQL safety validation via AST parsing (sqlglot), not string matching.

Generated queries must be single, read-only SELECT statements scoped to the
approved semantic views -- see the "Query safety and validation" phase of
IMPLEMENTATION_PLAN.md.
"""
from __future__ import annotations

import sqlglot
from sqlglot import exp

DIALECT = "duckdb"

ALLOWED_SCHEMAS = {"semantic_views"}
ALLOWED_TABLES = {"fct_monthly_subscriber_revenue"}

_DISALLOWED_NODE_TYPES = (
    exp.Drop,
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Create,
    exp.Alter,
    exp.TruncateTable,
    exp.Command,
)


def validate_sql_statement(sql: str) -> bool:
    """Returns True only for a single SELECT statement scoped to approved views."""
    try:
        statements = [s for s in sqlglot.parse(sql, read=DIALECT) if s is not None]
    except sqlglot.errors.SqlglotError:
        return False

    if len(statements) != 1:
        return False

    statement = statements[0]
    if not isinstance(statement, exp.Select):
        return False

    if statement.find(*_DISALLOWED_NODE_TYPES) is not None:
        return False

    tables = list(statement.find_all(exp.Table))
    if not tables:
        return False

    for table in tables:
        if table.name not in ALLOWED_TABLES:
            return False
        if table.db and table.db not in ALLOWED_SCHEMAS:
            return False

    return True
