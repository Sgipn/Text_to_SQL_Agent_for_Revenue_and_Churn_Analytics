import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api import API_KEY_ENV_VAR, RATE_LIMIT_MAX_REQUESTS, _build_scope_html, _request_log, app, get_llm_client
from app.services.query_execution import DB_PATH
from app.services.semantic_view_registry import ALLOWED_VIEWS
from app.services.vector_store import get_collection

VALID_ARM_SQL = (
    "```sql\n"
    "SELECT sum(total_net_revenue) / sum(active_paid_subscribers) as arm\n"
    "FROM semantic_views.fct_monthly_subscriber_revenue\n"
    "```"
)
NO_QUERY_RESPONSE = "NO_QUERY: Net Promoter Score is not a metric defined in the available semantic view."
DESTRUCTIVE_SQL = "```sql\nDROP TABLE semantic_views.fct_monthly_subscriber_revenue\n```"


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
class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        _request_log.clear()

    def tearDown(self) -> None:
        app.dependency_overrides.clear()
        _request_log.clear()

    def test_health_check(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_index_page_serves_html_ui(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("askButton", response.text)
        self.assertIn("/ask", response.text)

    def test_index_page_is_never_rate_limited(self) -> None:
        for _ in range(RATE_LIMIT_MAX_REQUESTS + 5):
            response = self.client.get("/")
            self.assertEqual(response.status_code, 200)

    def test_index_page_scope_panel_lists_every_approved_view_and_ratio_metric(self) -> None:
        # Sourced from the same registries that drive query validation --
        # this fails if a future mart/metric is added without the page
        # reflecting it, instead of silently going stale.
        response = self.client.get("/")

        self.assertNotIn("__SCOPE_PANEL__", response.text)
        self.assertIn(_build_scope_html(), response.text)
        self.assertEqual(response.text.count('class="scope-card"'), len(ALLOWED_VIEWS))

    def test_ask_returns_sql_and_json_safe_rows_on_success(self) -> None:
        app.dependency_overrides[get_llm_client] = lambda: FakeLLMClient([VALID_ARM_SQL])

        response = self.client.post("/ask", json={"question": "What is our ARM?"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["succeeded"])
        self.assertEqual(body["attempts"], 1)
        self.assertIn("semantic_views.fct_monthly_subscriber_revenue", body["sql"])
        self.assertEqual(body["row_count"], 1)
        self.assertIsInstance(body["rows"], list)

    def test_ask_summarize_true_includes_summary(self) -> None:
        app.dependency_overrides[get_llm_client] = lambda: FakeLLMClient(
            [VALID_ARM_SQL, "ARM is roughly 13 dollars per membership."]
        )

        response = self.client.post("/ask", json={"question": "What is our ARM?", "summarize": True})

        self.assertEqual(response.json()["summary"], "ARM is roughly 13 dollars per membership.")

    def test_ask_summarize_false_by_default_omits_summary(self) -> None:
        app.dependency_overrides[get_llm_client] = lambda: FakeLLMClient([VALID_ARM_SQL])

        response = self.client.post("/ask", json={"question": "What is our ARM?"})

        self.assertIsNone(response.json()["summary"])

    def test_ask_includes_confidence_interval_for_ratio_metric(self) -> None:
        app.dependency_overrides[get_llm_client] = lambda: FakeLLMClient([VALID_ARM_SQL])

        response = self.client.post("/ask", json={"question": "What is our ARM?"})

        body = response.json()
        ci = body["confidence_interval"]
        self.assertIsNotNone(ci)
        self.assertLess(ci["lower"], ci["estimate"])
        self.assertGreater(ci["upper"], ci["estimate"])
        self.assertEqual(ci["confidence_level"], 0.95)

    def test_ask_omits_confidence_interval_for_non_ratio_query(self) -> None:
        app.dependency_overrides[get_llm_client] = lambda: FakeLLMClient(
            [
                "```sql\nSELECT region_id, sum(total_net_revenue) as total_net_revenue "
                "FROM semantic_views.fct_monthly_subscriber_revenue GROUP BY region_id\n```"
            ]
        )

        response = self.client.post("/ask", json={"question": "Total revenue by region?"})

        self.assertIsNone(response.json()["confidence_interval"])

    def test_ask_returns_decline_reason_without_sql(self) -> None:
        app.dependency_overrides[get_llm_client] = lambda: FakeLLMClient([NO_QUERY_RESPONSE])

        response = self.client.post("/ask", json={"question": "What is our Net Promoter Score?"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["succeeded"])
        self.assertIsNone(body["sql"])
        self.assertIsNone(body["rows"])
        self.assertIn("Net Promoter Score", body["error"])

    def test_ask_blocks_destructive_generated_sql(self) -> None:
        app.dependency_overrides[get_llm_client] = lambda: FakeLLMClient([DESTRUCTIVE_SQL, DESTRUCTIVE_SQL])

        response = self.client.post("/ask", json={"question": "Delete everything", "max_attempts": 2})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["succeeded"])
        self.assertEqual(body["attempts"], 2)

    def test_ask_rejects_empty_question_with_422(self) -> None:
        response = self.client.post("/ask", json={"question": ""})
        self.assertEqual(response.status_code, 422)

    def test_ask_rejects_missing_question_with_422(self) -> None:
        response = self.client.post("/ask", json={})
        self.assertEqual(response.status_code, 422)

    def test_ask_works_without_a_key_when_none_is_configured(self) -> None:
        # ASK_API_KEY unset (the default) -- no auth required, matching
        # every other test in this file that never sends X-API-Key.
        app.dependency_overrides[get_llm_client] = lambda: FakeLLMClient([VALID_ARM_SQL])

        response = self.client.post("/ask", json={"question": "What is our ARM?"})

        self.assertEqual(response.status_code, 200)

    def test_ask_rejects_missing_api_key_when_configured(self) -> None:
        with patch.dict("os.environ", {API_KEY_ENV_VAR: "secret123"}):
            response = self.client.post("/ask", json={"question": "What is our ARM?"})

        self.assertEqual(response.status_code, 401)

    def test_ask_rejects_wrong_api_key_when_configured(self) -> None:
        with patch.dict("os.environ", {API_KEY_ENV_VAR: "secret123"}):
            response = self.client.post(
                "/ask", json={"question": "What is our ARM?"}, headers={"X-API-Key": "wrong"}
            )

        self.assertEqual(response.status_code, 401)

    def test_ask_accepts_correct_api_key_when_configured(self) -> None:
        app.dependency_overrides[get_llm_client] = lambda: FakeLLMClient([VALID_ARM_SQL])

        with patch.dict("os.environ", {API_KEY_ENV_VAR: "secret123"}):
            response = self.client.post(
                "/ask", json={"question": "What is our ARM?"}, headers={"X-API-Key": "secret123"}
            )

        self.assertEqual(response.status_code, 200)

    def test_ask_rate_limits_after_max_requests_per_window(self) -> None:
        app.dependency_overrides[get_llm_client] = lambda: FakeLLMClient([VALID_ARM_SQL] * RATE_LIMIT_MAX_REQUESTS)

        for _ in range(RATE_LIMIT_MAX_REQUESTS):
            response = self.client.post("/ask", json={"question": "What is our ARM?"})
            self.assertEqual(response.status_code, 200)

        one_too_many = self.client.post("/ask", json={"question": "What is our ARM?"})
        self.assertEqual(one_too_many.status_code, 429)

    def test_health_check_is_never_rate_limited(self) -> None:
        for _ in range(RATE_LIMIT_MAX_REQUESTS + 5):
            response = self.client.get("/health")
            self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
