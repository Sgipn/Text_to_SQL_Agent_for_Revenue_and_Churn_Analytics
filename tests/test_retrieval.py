import shutil
import tempfile
import unittest
from pathlib import Path

import chromadb

from app.services.metadata_extraction import MANIFEST_PATH, build_context_documents
from app.services.retrieval import format_context_for_prompt, retrieve_context
from app.services.vector_store import index_documents


@unittest.skipUnless(MANIFEST_PATH.exists(), f"dbt manifest not found at {MANIFEST_PATH}; run `dbt build` first")
class RetrievalTests(unittest.TestCase):
    """Uses an isolated, temp-dir ChromaDB collection so tests never touch
    or depend on the shared data/chroma index."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp_dir = tempfile.mkdtemp(prefix="chroma_test_")
        cls.client = chromadb.PersistentClient(path=cls.tmp_dir)
        index_documents(build_context_documents(), client=cls.client)
        cls.collection = cls.client.get_or_create_collection("business_context")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_arm_question_retrieves_arm_metric_first(self) -> None:
        results = retrieve_context("What is Average Revenue per Membership?", top_k=1, collection=self.collection)
        self.assertEqual(results[0]["id"], "metric::average_revenue_per_membership")

    def test_subscriber_question_retrieves_relevant_context(self) -> None:
        results = retrieve_context("How many paid subscribers do we have?", top_k=2, collection=self.collection)
        retrieved_ids = {r["id"] for r in results}
        self.assertIn("metric::active_paid_subscribers", retrieved_ids)

    def test_format_context_for_prompt_joins_document_text(self) -> None:
        results = retrieve_context("revenue by region", top_k=2, collection=self.collection)
        formatted = format_context_for_prompt(results)
        for r in results:
            self.assertIn(r["text"], formatted)

    def test_arm_question_does_not_surface_growth_domain_docs(self) -> None:
        # Stress-tests disambiguation across the two semantic-view domains
        # (revenue vs. growth/churn) added in Phase 12 -- with only one
        # domain, top-k retrieval couldn't meaningfully fail this way.
        results = retrieve_context(
            "What is the Average Revenue per Membership for Premium plans?", top_k=3, collection=self.collection
        )
        retrieved_ids = {r["id"] for r in results}
        self.assertIn("metric::average_revenue_per_membership", retrieved_ids)
        self.assertNotIn("view::fct_monthly_subscriber_activity", retrieved_ids)
        self.assertNotIn("metric::monthly_churn_rate", retrieved_ids)

    def test_churn_question_does_not_surface_revenue_domain_docs(self) -> None:
        results = retrieve_context("What is our monthly churn rate?", top_k=3, collection=self.collection)
        retrieved_ids = {r["id"] for r in results}
        self.assertIn("metric::monthly_churn_rate", retrieved_ids)
        self.assertNotIn("view::fct_monthly_subscriber_revenue", retrieved_ids)
        self.assertNotIn("metric::average_revenue_per_membership", retrieved_ids)

    def test_raises_on_empty_collection(self) -> None:
        empty_collection = self.client.get_or_create_collection("empty_for_test")
        with self.assertRaises(RuntimeError):
            retrieve_context("anything", collection=empty_collection)


if __name__ == "__main__":
    unittest.main()
