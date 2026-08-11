"""SQL safety validation via AST parsing (sqlglot), not string matching.

Generated queries must be single, read-only SELECT statements scoped to the
approved semantic views -- and their approved columns -- see the "Query
safety and validation" phase of IMPLEMENTATION_PLAN.md.
"""
from __future__ import annotations

import sqlglot
from sqlglot import exp

from app.services.semantic_view_registry import ALLOWED_VIEWS

DIALECT = "duckdb"

# DuckDB folds unquoted identifiers case-insensitively, but sqlglot preserves
# whatever case the model actually wrote. Matching ALLOWED_VIEWS case-sensitively
# would reject a perfectly valid, semantically correct query -- e.g. `Total_Net_Revenue`
# -- purely because of incidental capitalization, burning a retry for no real
# safety benefit (lenience here only reduces false positives; it can't expand what's
# actually queryable, since DuckDB still resolves the real column at execution time).
_ALLOWED_VIEWS_BY_LOWER_NAME = {name.lower(): (name, view) for name, view in ALLOWED_VIEWS.items()}

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


def _known_aliases(statement: exp.Select) -> set[str]:
    """Output aliases defined anywhere in the statement (including subqueries).

    A SELECT-list alias (e.g. `SUM(x) AS arm`) can legitimately be
    referenced elsewhere in the query, e.g. `ORDER BY arm` -- sqlglot parses
    that as an exp.Column named "arm" indistinguishable from a real table
    column reference. Without excluding known aliases, that would be
    rejected as an unapproved column despite never touching unapproved data.
    """
    aliases: set[str] = set()
    for select in statement.find_all(exp.Select):
        aliases.update(projection.alias for projection in select.expressions if projection.alias)
    return aliases


def parse_safe_select(sql: str) -> exp.Select:
    """Parses `sql` and returns it as a validated Select AST.

    Raises UnsafeQueryError if the statement is anything other than a single
    SELECT scoped to the approved semantic views in ALLOWED_VIEWS, or if it
    references a column not documented on any of those views (and not a
    known output alias -- see _known_aliases).
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

    allowed_columns: set[str] = set()
    for table in tables:
        match = _ALLOWED_VIEWS_BY_LOWER_NAME.get(table.name.lower())
        if match is None:
            raise UnsafeQueryError(f"'{table.name}' is not an approved semantic view.")
        canonical_name, view = match
        if table.db and table.db.lower() != view.schema.lower():
            raise UnsafeQueryError(f"'{table.name}' must be referenced as '{view.schema}.{canonical_name}'.")
        allowed_columns.update(view.columns)

    allowed_columns.update(_known_aliases(statement))
    allowed_columns_lower = {name.lower() for name in allowed_columns}

    for column in statement.find_all(exp.Column):
        if column.name.lower() not in allowed_columns_lower:
            raise UnsafeQueryError(f"'{column.name}' is not a column on any approved semantic view.")

    return statement


def validate_sql_statement(sql: str) -> bool:
    """Returns True only for a single SELECT statement scoped to approved semantic views."""
    try:
        parse_safe_select(sql)
        return True
    except UnsafeQueryError:
        return False
