import json
import unittest
from pathlib import Path

from app.services.ratio_metric_registry import RATIO_METRICS

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "dbt" / "target" / "manifest.json"


class RatioMetricRegistryDriftTests(unittest.TestCase):
    """Guards against the registry silently drifting from the dbt ratio
    metrics and semantic models it describes.

    Skipped if the manifest hasn't been built yet -- run `dbt build` in
    dbt/ before this test can check for drift.
    """

    def test_registry_matches_dbt_ratio_metric_definitions(self) -> None:
        if not MANIFEST_PATH.exists():
            self.skipTest(f"dbt manifest not found at {MANIFEST_PATH}; run `dbt build` first")

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        ratio_metrics = {
            metric["name"]: metric
            for metric in manifest.get("metrics", {}).values()
            if metric.get("type") == "ratio"
        }
        semantic_models = list(manifest.get("semantic_models", {}).values())

        self.assertEqual(set(RATIO_METRICS), set(ratio_metrics), "ratio_metric_registry.RATIO_METRICS is out of sync with dbt's ratio metrics.")

        for name, registered in RATIO_METRICS.items():
            dbt_metric = ratio_metrics[name]
            type_params = dbt_metric["type_params"]
            expected_numerator = type_params["numerator"]["name"]
            expected_denominator = type_params["denominator"]["name"]

            self.assertEqual(registered.numerator, expected_numerator, f"{name}: numerator drifted from dbt")
            self.assertEqual(registered.denominator, expected_denominator, f"{name}: denominator drifted from dbt")

            owning_model = next(
                sm
                for sm in semantic_models
                if any(m["name"] == expected_numerator for m in sm.get("measures", []))
            )
            relation = owning_model["node_relation"]
            expected_table = f"{relation['schema_name']}.{relation['alias']}"
            expected_time_column = owning_model["defaults"]["agg_time_dimension"]

            self.assertEqual(registered.table, expected_table, f"{name}: table drifted from dbt")
            self.assertEqual(registered.time_column, expected_time_column, f"{name}: time_column drifted from dbt")


if __name__ == "__main__":
    unittest.main()
