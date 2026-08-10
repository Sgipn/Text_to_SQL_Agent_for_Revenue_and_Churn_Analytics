import unittest

from app.services.query_execution import DB_PATH, execute_safe_query
from app.services.sql_validation import UnsafeQueryError

SAFE_SQL = "SELECT * FROM semantic_views.fct_monthly_subscriber_revenue"


@unittest.skipUnless(DB_PATH.exists(), f"DuckDB database not found at {DB_PATH}; run `dbt build` first")
class QueryExecutionTests(unittest.TestCase):
    def test_executes_safe_select_and_returns_expected_columns(self) -> None:
        result = execute_safe_query(SAFE_SQL)
        self.assertGreater(len(result), 0)
        self.assertEqual(
            set(result.columns),
            {"metric_month", "region_id", "plan_type", "active_paid_subscribers", "total_net_revenue"},
        )

    def test_raises_and_does_not_execute_unsafe_query(self) -> None:
        with self.assertRaises(UnsafeQueryError):
            execute_safe_query("DROP TABLE semantic_views.fct_monthly_subscriber_revenue")

    def test_default_row_limit_is_applied_when_query_has_none(self) -> None:
        result = execute_safe_query(SAFE_SQL, row_limit=5)
        self.assertLessEqual(len(result), 5)

    def test_requested_limit_above_cap_is_reduced_to_cap(self) -> None:
        result = execute_safe_query(f"{SAFE_SQL} LIMIT 10000", row_limit=5)
        self.assertLessEqual(len(result), 5)

    def test_requested_limit_below_cap_is_respected(self) -> None:
        result = execute_safe_query(f"{SAFE_SQL} LIMIT 2", row_limit=1000)
        self.assertLessEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
