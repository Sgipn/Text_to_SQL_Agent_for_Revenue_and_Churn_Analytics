"""Executes validated SQL against the DuckDB warehouse.

Every query passes through sql_validation before touching the database. As a
second layer, the connection itself is opened read_only so a gap in
validation still can't mutate data -- see the "Query safety and validation"
phase of IMPLEMENTATION_PLAN.md.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from app.services.sql_validation import UnsafeQueryError, parse_safe_select

DB_PATH = Path(__file__).resolve().parents[2] / "dbt" / "semantic_metric_repository.duckdb"
DEFAULT_ROW_LIMIT = 1_000


def _cap_row_limit(statement, row_limit: int):
    existing_limit = statement.args.get("limit")
    if existing_limit is None:
        return statement.limit(row_limit)

    requested = int(existing_limit.expression.this)
    return statement.limit(min(requested, row_limit)) if requested > row_limit else statement


def execute_safe_query(
    sql: str,
    db_path: Path = DB_PATH,
    row_limit: int = DEFAULT_ROW_LIMIT,
) -> pd.DataFrame:
    """Validates `sql`, caps its row limit, and executes it read-only.

    Raises UnsafeQueryError if the statement fails validation.
    """
    statement = parse_safe_select(sql)
    statement = _cap_row_limit(statement, row_limit)
    safe_sql = statement.sql(dialect="duckdb")

    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB database not found at {db_path}. Run `dbt build` first.")

    with duckdb.connect(str(db_path), read_only=True) as conn:
        return conn.execute(safe_sql).fetchdf()
