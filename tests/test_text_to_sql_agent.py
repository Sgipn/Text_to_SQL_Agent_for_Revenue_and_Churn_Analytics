import unittest

from app.agents.text_to_sql_agent import answer_question
from app.services.query_execution import DB_PATH, execute_safe_query
from app.services.vector_store import get_collection

VALID_ARM_SQL = (
    "```sql\n"
    "SELECT sum(total_net_revenue) / sum(active_paid_subscribers) as arm\n"
    "FROM semantic_views.fct_monthly_subscriber_revenue\n"
    "```"
)
REVENUE_BY_REGION_SQL = (
    "```sql\n"
    "SELECT region_id, sum(total_net_revenue) as total_net_revenue\n"
    "FROM semantic_views.fct_monthly_subscriber_revenue\n"
    "GROUP BY region_id\n"
    "```"
)
UNAPPROVED_TABLE_SQL = "```sql\nSELECT * FROM users\n```"
DESTRUCTIVE_SQL = "```sql\nDROP TABLE semantic_views.fct_monthly_subscriber_revenue\n```"
NO_QUERY_RESPONSE = "NO_QUERY: Churn rate is not a metric defined in the available semantic view."


class FakeLLMClient:
    """Returns canned responses in order; records every prompt it was called with."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self._responses.pop(0)


def _retrieval_index_ready() -> bool:
    try:
        return get_collection().count() > 0
    except Exception:
        return False


@unittest.skipUnless(DB_PATH.exists(), f"DuckDB database not found at {DB_PATH}; run `dbt build` first")
@unittest.skipUnless(
    _retrieval_index_ready(), "Vector store not indexed; run `python -m app.services.vector_store` first"
)
class TextToSqlAgentTests(unittest.TestCase):
    def test_succeeds_on_first_valid_attempt(self) -> None:
        client = FakeLLMClient([VALID_ARM_SQL])
        result = answer_question("What is our Average Revenue per Membership?", llm_client=client)

        self.assertTrue(result.succeeded)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(len(client.calls), 1)
        self.assertIsNotNone(result.result)
        self.assertGreater(len(result.result), 0)

    def test_attaches_confidence_interval_for_ratio_metric_query(self) -> None:
        client = FakeLLMClient([VALID_ARM_SQL])
        result = answer_question("What is our Average Revenue per Membership?", llm_client=client)

        self.assertTrue(result.succeeded)
        ci = result.confidence_interval
        self.assertIsNotNone(ci)
        self.assertLess(ci.lower, ci.estimate)
        self.assertGreater(ci.upper, ci.estimate)
        self.assertGreater(ci.n_units, 1)

    def test_no_confidence_interval_for_non_ratio_query(self) -> None:
        client = FakeLLMClient([REVENUE_BY_REGION_SQL])
        result = answer_question("What was total revenue by region?", llm_client=client)

        self.assertTrue(result.succeeded)
        self.assertIsNone(result.confidence_interval)

    def test_retries_after_invalid_sql_then_succeeds(self) -> None:
        client = FakeLLMClient([UNAPPROVED_TABLE_SQL, VALID_ARM_SQL])
        result = answer_question("What is our ARM?", llm_client=client, max_attempts=2)

        self.assertTrue(result.succeeded)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(client.calls), 2)
        # the retry prompt should tell the model what went wrong
        self.assertIn("rejected", client.calls[1][1].lower())

    def test_exhausts_retries_and_reports_error_without_executing(self) -> None:
        client = FakeLLMClient([UNAPPROVED_TABLE_SQL, UNAPPROVED_TABLE_SQL])
        result = answer_question("Show me everything", llm_client=client, max_attempts=2)

        self.assertFalse(result.succeeded)
        self.assertIsNone(result.result)
        self.assertEqual(result.attempts, 2)
        self.assertIsNotNone(result.error)

    def test_declines_immediately_for_unsupported_metric_without_retrying(self) -> None:
        client = FakeLLMClient([NO_QUERY_RESPONSE])
        result = answer_question("What is our churn rate?", llm_client=client, max_attempts=2)

        self.assertFalse(result.succeeded)
        self.assertIsNone(result.sql)
        self.assertIsNone(result.result)
        self.assertEqual(result.attempts, 1)
        self.assertIn("churn rate", result.error.lower())
        self.assertEqual(len(client.calls), 1)  # no wasted retry on a deliberate decline

    def test_blocks_destructive_generated_sql_and_never_executes_it(self) -> None:
        client = FakeLLMClient([DESTRUCTIVE_SQL, DESTRUCTIVE_SQL])
        result = answer_question("Delete all the revenue data", llm_client=client, max_attempts=2)

        self.assertFalse(result.succeeded)
        # the mart must be untouched -- prove the DROP never reached the database
        df = execute_safe_query("SELECT * FROM semantic_views.fct_monthly_subscriber_revenue")
        self.assertGreater(len(df), 0)


if __name__ == "__main__":
    unittest.main()
