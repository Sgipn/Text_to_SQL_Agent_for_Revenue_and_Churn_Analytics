import unittest

import sqlglot

from app.agents.confidence_interval import compute_ratio_confidence_interval
from app.services.query_execution import DB_PATH


def _parse(sql: str):
    return sqlglot.parse_one(sql, read="duckdb")


@unittest.skipUnless(DB_PATH.exists(), f"DuckDB database not found at {DB_PATH}; run `dbt build` first")
class ComputeRatioConfidenceIntervalTests(unittest.TestCase):
    def test_computes_ci_for_arm_shaped_query(self) -> None:
        statement = _parse(
            "SELECT SUM(total_net_revenue) / SUM(active_paid_subscribers) AS arm "
            "FROM semantic_views.fct_monthly_subscriber_revenue"
        )

        ci = compute_ratio_confidence_interval(statement)

        self.assertIsNotNone(ci)
        self.assertLess(ci.lower, ci.estimate)
        self.assertGreater(ci.upper, ci.estimate)

    def test_reuses_where_clause_so_filtered_slice_differs_from_overall(self) -> None:
        overall = compute_ratio_confidence_interval(
            _parse(
                "SELECT SUM(total_net_revenue) / SUM(active_paid_subscribers) AS arm "
                "FROM semantic_views.fct_monthly_subscriber_revenue"
            )
        )
        filtered = compute_ratio_confidence_interval(
            _parse(
                "SELECT SUM(total_net_revenue) / SUM(active_paid_subscribers) AS arm "
                "FROM semantic_views.fct_monthly_subscriber_revenue "
                "WHERE plan_type = 'Premium' AND region_id = 'EMEA'"
            )
        )

        self.assertIsNotNone(overall)
        self.assertIsNotNone(filtered)
        # Different slices of the data should produce different estimates --
        # if the WHERE clause weren't actually reused, these would match.
        self.assertNotAlmostEqual(overall.estimate, filtered.estimate, places=2)

    def test_returns_none_when_ratio_columns_only_appear_in_where_clause(self) -> None:
        # Filtering on both columns isn't the same as computing/displaying
        # the ratio metric -- this must not trigger CI computation.
        statement = _parse(
            "SELECT plan_type FROM semantic_views.fct_monthly_subscriber_revenue "
            "WHERE active_paid_subscribers > 10 AND total_net_revenue > 100"
        )

        self.assertIsNone(compute_ratio_confidence_interval(statement))

    def test_returns_none_for_query_not_referencing_a_ratio_metric(self) -> None:
        statement = _parse(
            "SELECT region_id, SUM(total_net_revenue) AS total_net_revenue "
            "FROM semantic_views.fct_monthly_subscriber_revenue GROUP BY region_id"
        )

        self.assertIsNone(compute_ratio_confidence_interval(statement))

    def test_returns_none_rather_than_raising_on_execution_failure(self) -> None:
        # References a nonexistent column in the WHERE clause reused for the
        # breakdown query -- this must degrade to None, not propagate.
        statement = _parse(
            "SELECT SUM(total_net_revenue) / SUM(active_paid_subscribers) AS arm "
            "FROM semantic_views.fct_monthly_subscriber_revenue "
            "WHERE nonexistent_column = 'x'"
        )

        self.assertIsNone(compute_ratio_confidence_interval(statement))


if __name__ == "__main__":
    unittest.main()
