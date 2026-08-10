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


if __name__ == "__main__":
    unittest.main()
