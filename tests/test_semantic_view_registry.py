import json
import unittest
from pathlib import Path

from app.services.semantic_view_registry import ALLOWED_VIEWS

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "dbt" / "target" / "manifest.json"


class SemanticViewRegistryDriftTests(unittest.TestCase):
    """Guards against the registry silently drifting from the dbt models it describes.

    Skipped if the manifest hasn't been built yet -- run `dbt build` in dbt/
    before this test can check for drift.
    """

    def test_registry_matches_dbt_semantic_view_tags(self) -> None:
        if not MANIFEST_PATH.exists():
            self.skipTest(f"dbt manifest not found at {MANIFEST_PATH}; run `dbt build` first")

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        tagged_models = {
            node["alias"]: node["schema"]
            for node in manifest["nodes"].values()
            if node.get("resource_type") == "model" and "semantic_view" in node.get("tags", [])
        }

        self.assertEqual(
            tagged_models,
            ALLOWED_VIEWS,
            "app/services/semantic_view_registry.py is out of sync with the dbt "
            "models tagged 'semantic_view' -- update ALLOWED_VIEWS to match.",
        )


if __name__ == "__main__":
    unittest.main()
