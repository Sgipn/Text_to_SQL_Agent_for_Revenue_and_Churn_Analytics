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

    def test_executes_safe_select_against_growth_view(self) -> None:
        result = execute_safe_query("SELECT * FROM semantic_views.fct_monthly_subscriber_activity")
        self.assertGreater(len(result), 0)
        self.assertEqual(
            set(result.columns),
            {"metric_month", "region_id", "active_subscribers", "new_subscribers", "churned_subscribers"},
        )

    def test_churn_never_counted_in_the_final_month_of_data(self) -> None:
        # The final period is right-censored -- a still-active user isn't a
        # churn event just because the dataset ends.
        result = execute_safe_query(
            "SELECT SUM(churned_subscribers) AS total_churned "
            "FROM semantic_views.fct_monthly_subscriber_activity "
            "WHERE metric_month = (SELECT MAX(metric_month) FROM semantic_views.fct_monthly_subscriber_activity)"
        )
        self.assertEqual(result["total_churned"].iloc[0], 0)

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
