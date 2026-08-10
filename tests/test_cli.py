import io
import unittest
from contextlib import redirect_stdout

from app.cli import main
from app.services.query_execution import DB_PATH
from app.services.vector_store import get_collection

VALID_ARM_SQL = (
    "```sql\n"
    "SELECT sum(total_net_revenue) / sum(active_paid_subscribers) as arm\n"
    "FROM semantic_views.fct_monthly_subscriber_revenue\n"
    "```"
)
NO_QUERY_RESPONSE = "NO_QUERY: Churn rate is not a metric defined in the available semantic view."


class FakeLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
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
class CliTests(unittest.TestCase):
    def test_prints_sql_and_result_and_returns_zero_on_success(self) -> None:
        client = FakeLLMClient([VALID_ARM_SQL])
        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = main(["What is our ARM?"], llm_client=client)

        output = buf.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("SQL:", output)
        self.assertIn("semantic_views.fct_monthly_subscriber_revenue", output)
        self.assertIn("Result (", output)

    def test_prints_decline_reason_and_returns_nonzero(self) -> None:
        client = FakeLLMClient([NO_QUERY_RESPONSE])
        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = main(["What is our churn rate?"], llm_client=client)

        output = buf.getvalue()
        self.assertEqual(exit_code, 1)
        self.assertIn("Could not answer this question", output)
        self.assertIn("Churn rate", output)

    def test_truncates_large_result_sets_for_display(self) -> None:
        client = FakeLLMClient(
            ["```sql\nSELECT * FROM semantic_views.fct_monthly_subscriber_revenue\n```"]
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["Show me everything"], llm_client=client)

        self.assertIn("showing first 20", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
