import unittest

from app.services.metadata_extraction import MANIFEST_PATH, build_context_documents


@unittest.skipUnless(MANIFEST_PATH.exists(), f"dbt manifest not found at {MANIFEST_PATH}; run `dbt build` first")
class MetadataExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.docs_by_id = {doc.id: doc for doc in build_context_documents()}

    def test_extracts_one_document_per_approved_semantic_view(self) -> None:
        self.assertIn("view::fct_monthly_subscriber_revenue", self.docs_by_id)
        doc = self.docs_by_id["view::fct_monthly_subscriber_revenue"]
        self.assertEqual(doc.metadata["doc_type"], "semantic_view")
        self.assertEqual(doc.metadata["schema"], "semantic_views")
        self.assertIn("active_paid_subscribers", doc.text)
        self.assertIn("total_net_revenue", doc.text)

    def test_extracts_ratio_metric_with_formula(self) -> None:
        doc = self.docs_by_id["metric::average_revenue_per_membership"]
        self.assertEqual(doc.metadata["metric_type"], "ratio")
        self.assertIn("total_net_revenue / active_paid_subscribers", doc.text)
        self.assertIn("ratio of sums, not average of ratios", doc.text)

    def test_extracts_simple_metrics(self) -> None:
        self.assertIn("metric::total_net_revenue", self.docs_by_id)
        self.assertIn("metric::active_paid_subscribers", self.docs_by_id)


if __name__ == "__main__":
    unittest.main()
