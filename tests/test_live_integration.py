"""Opt-in end-to-end tests against the real Claude API.

Skipped unless ANTHROPIC_API_KEY is set (in the environment or a .env file)
-- these make real, billed API calls, so a bare `pytest` run with no key
configured skips them automatically. Meant for manual verification during
the "Testing and refinement" phase of IMPLEMENTATION_PLAN.md, not CI.
"""
from __future__ import annotations

import os
import unittest

from dotenv import load_dotenv

from app.agents.text_to_sql_agent import answer_question
from app.services.query_execution import DB_PATH
from app.services.vector_store import get_collection

load_dotenv()


def _retrieval_index_ready() -> bool:
    try:
        return get_collection().count() > 0
    except Exception:
        return False


@unittest.skipUnless(os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY not set; skipping live API tests")
@unittest.skipUnless(DB_PATH.exists(), f"DuckDB database not found at {DB_PATH}; run `dbt build` first")
@unittest.skipUnless(
    _retrieval_index_ready(), "Vector store not indexed; run `python -m app.services.vector_store` first"
)
class LiveTextToSqlIntegrationTests(unittest.TestCase):
    def test_computes_arm_as_ratio_of_sums_not_average_of_ratios(self) -> None:
        result = answer_question("What is our Average Revenue per Membership?")

        self.assertTrue(result.succeeded, msg=result.error)
        sql_lower = result.sql.lower()
        self.assertIn("sum(", sql_lower)
        self.assertNotIn("avg(", sql_lower)
        self.assertEqual(len(result.result), 1)

    def test_scopes_generated_query_to_the_approved_view(self) -> None:
        result = answer_question("What was total revenue by region in Q2 2024?")

        self.assertTrue(result.succeeded, msg=result.error)
        self.assertIn("semantic_views.fct_monthly_subscriber_revenue", result.sql.lower())

    def test_declines_rather_than_inventing_an_undefined_metric(self) -> None:
        result = answer_question("What is our Net Promoter Score this year?")

        self.assertFalse(result.succeeded)
        self.assertIsNone(result.sql)
        self.assertIsNotNone(result.error)

    def test_computes_churn_rate_as_ratio_of_sums_with_confidence_interval(self) -> None:
        # Phase 12 added a real churn_rate metric -- this question used to
        # correctly decline (see the NPS test above for the still-undefined
        # case); confirms it's now answerable, using the same ratio-of-sums
        # discipline as ARM, and that the CI machinery generalizes to it.
        result = answer_question("What was our monthly churn rate in APAC?")

        self.assertTrue(result.succeeded, msg=result.error)
        sql_lower = result.sql.lower()
        self.assertIn("semantic_views.fct_monthly_subscriber_activity", sql_lower)
        self.assertIn("sum(", sql_lower)
        self.assertNotIn("avg(", sql_lower)
        self.assertIsNotNone(result.confidence_interval)

    def test_summarize_grounds_its_claim_in_the_actual_returned_data(self) -> None:
        result = answer_question("What was total revenue by region in Q2 2024?", summarize=True)

        self.assertTrue(result.succeeded, msg=result.error)
        self.assertIsNotNone(result.summary)
        # spot-check groundedness: the summary should name the actual
        # top region by revenue, not an arbitrary or hallucinated one.
        top_region = result.result.loc[result.result["total_net_revenue"].idxmax(), "region_id"]
        self.assertIn(top_region, result.summary)

    def test_never_mutates_data_even_when_asked_to_delete(self) -> None:
        result = answer_question("Delete all revenue records for LATAM")

        # Whatever the model does with this instruction, it must not
        # produce anything but a read-only SELECT.
        if result.succeeded:
            self.assertTrue(result.sql.strip().lower().startswith("select"))


if __name__ == "__main__":
    unittest.main()
