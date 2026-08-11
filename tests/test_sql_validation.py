import unittest

from app.services.sql_validation import validate_sql_statement


class SqlValidationTests(unittest.TestCase):
    def test_allows_safe_select_query(self) -> None:
        sql = "SELECT * FROM semantic_views.fct_monthly_subscriber_revenue"
        self.assertTrue(validate_sql_statement(sql))

    def test_rejects_dangerous_statement(self) -> None:
        sql = "DROP TABLE users"
        self.assertFalse(validate_sql_statement(sql))

    def test_rejects_unapproved_table(self) -> None:
        sql = "SELECT * FROM other_secret_table"
        self.assertFalse(validate_sql_statement(sql))

    def test_rejects_stacked_statement_injection(self) -> None:
        sql = (
            "SELECT * FROM semantic_views.fct_monthly_subscriber_revenue; "
            "DROP TABLE semantic_views.fct_monthly_subscriber_revenue"
        )
        self.assertFalse(validate_sql_statement(sql))

    def test_allows_subquery_over_approved_view(self) -> None:
        sql = (
            "SELECT * FROM ("
            "SELECT 1 FROM semantic_views.fct_monthly_subscriber_revenue"
            ") AS sub"
        )
        self.assertTrue(validate_sql_statement(sql))

    def test_rejects_nonexistent_column(self) -> None:
        sql = "SELECT hallucinated_column FROM semantic_views.fct_monthly_subscriber_revenue"
        self.assertFalse(validate_sql_statement(sql))

    def test_rejects_nonexistent_column_in_where_clause(self) -> None:
        sql = (
            "SELECT total_net_revenue FROM semantic_views.fct_monthly_subscriber_revenue "
            "WHERE hallucinated_column = 'x'"
        )
        self.assertFalse(validate_sql_statement(sql))

    def test_rejects_column_that_only_exists_on_a_different_approved_view(self) -> None:
        # churned_subscribers is real -- just not on this view. A column
        # allowlist scoped to "any approved view" rather than "the views
        # actually referenced in this query" would wrongly let this through.
        sql = "SELECT churned_subscribers FROM semantic_views.fct_monthly_subscriber_revenue"
        self.assertFalse(validate_sql_statement(sql))

    def test_allows_order_by_referencing_a_select_alias(self) -> None:
        # SUM(total_net_revenue) / SUM(active_paid_subscribers) AS arm, then
        # ORDER BY arm -- sqlglot parses "arm" there as a column reference,
        # indistinguishable from a real one without alias-awareness.
        sql = (
            "SELECT region_id, SUM(total_net_revenue) / SUM(active_paid_subscribers) AS arm "
            "FROM semantic_views.fct_monthly_subscriber_revenue "
            "GROUP BY region_id ORDER BY arm DESC"
        )
        self.assertTrue(validate_sql_statement(sql))

    def test_allows_differently_cased_column_reference(self) -> None:
        # DuckDB folds unquoted identifiers case-insensitively; sqlglot
        # preserves whatever case was written. A capitalized-but-real
        # column must not be rejected just because of casing.
        sql = "SELECT Total_Net_Revenue FROM semantic_views.fct_monthly_subscriber_revenue"
        self.assertTrue(validate_sql_statement(sql))

    def test_allows_differently_cased_table_and_schema_reference(self) -> None:
        sql = "SELECT * FROM Semantic_Views.Fct_Monthly_Subscriber_Revenue"
        self.assertTrue(validate_sql_statement(sql))

    def test_still_rejects_nonexistent_column_regardless_of_case(self) -> None:
        sql = "SELECT Hallucinated_Column FROM semantic_views.fct_monthly_subscriber_revenue"
        self.assertFalse(validate_sql_statement(sql))


if __name__ == "__main__":
    unittest.main()
