"""SQL safety validation via AST parsing (sqlglot), not string matching.

Generated queries must be single, read-only SELECT statements scoped to the
approved semantic views -- see the "Query safety and validation" phase of
IMPLEMENTATION_PLAN.md.
"""
from __future__ import annotations

import sqlglot
from sqlglot import exp

from app.services.semantic_view_registry import ALLOWED_VIEWS

DIALECT = "duckdb"

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


class UnsafeQueryError(ValueError):
    """Raised when a candidate SQL statement fails safety validation."""


def parse_safe_select(sql: str) -> exp.Select:
    """Parses `sql` and returns it as a validated Select AST.

    Raises UnsafeQueryError if the statement is anything other than a single
    SELECT scoped to the approved semantic views in ALLOWED_VIEWS.
    """
    try:
        statements = [s for s in sqlglot.parse(sql, read=DIALECT) if s is not None]
    except sqlglot.errors.SqlglotError as exc:
        raise UnsafeQueryError(f"Could not parse SQL: {exc}") from exc

    if len(statements) != 1:
        raise UnsafeQueryError("Only a single SQL statement is permitted per request.")

    statement = statements[0]
    if not isinstance(statement, exp.Select):
        raise UnsafeQueryError(f"Only SELECT statements are permitted, got {type(statement).__name__}.")

    if statement.find(*_DISALLOWED_NODE_TYPES) is not None:
        raise UnsafeQueryError("Statement contains a disallowed operation.")

    tables = list(statement.find_all(exp.Table))
    if not tables:
        raise UnsafeQueryError("Statement does not reference any table.")

    for table in tables:
        allowed_schema = ALLOWED_VIEWS.get(table.name)
        if allowed_schema is None:
            raise UnsafeQueryError(f"'{table.name}' is not an approved semantic view.")
        if table.db and table.db != allowed_schema:
            raise UnsafeQueryError(
                f"'{table.name}' must be referenced as '{allowed_schema}.{table.name}'."
            )

    return statement


def validate_sql_statement(sql: str) -> bool:
    """Returns True only for a single SELECT statement scoped to approved semantic views."""
    try:
        parse_safe_select(sql)
        return True
    except UnsafeQueryError:
        return False
