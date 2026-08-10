"""Extracts schema, metric, and column metadata from the dbt manifest into
retrieval-ready Markdown documents for the vector store.

Reads dbt/target/manifest.json rather than the raw YAML source so the
documents always reflect what was actually built -- descriptions,
resolved measure references, and the semantic_view tag used to scope
retrieval to approved views.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "dbt" / "target" / "manifest.json"


@dataclass
class ContextDocument:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _load_manifest(manifest_path: Path = MANIFEST_PATH) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(f"dbt manifest not found at {manifest_path}. Run `dbt build` first.")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _semantic_view_documents(manifest: dict) -> list[ContextDocument]:
    docs = []
    for node in manifest["nodes"].values():
        if node.get("resource_type") != "model" or "semantic_view" not in node.get("tags", []):
            continue

        columns = node.get("columns", {})
        column_lines = "\n".join(
            f"- {name}: {col.get('description') or 'no description'}" for name, col in columns.items()
        )
        text = (
            f"# Semantic view: {node['schema']}.{node['alias']}\n\n"
            f"{node.get('description') or ''}\n\n"
            f"## Columns\n{column_lines}"
        ).strip()

        docs.append(
            ContextDocument(
                id=f"view::{node['alias']}",
                text=text,
                metadata={"doc_type": "semantic_view", "table": node["alias"], "schema": node["schema"]},
            )
        )
    return docs


def _metric_documents(manifest: dict) -> list[ContextDocument]:
    docs = []
    for metric in manifest.get("metrics", {}).values():
        type_params = metric.get("type_params") or {}
        metric_type = metric.get("type")

        if metric_type == "ratio":
            numerator = (type_params.get("numerator") or {}).get("name")
            denominator = (type_params.get("denominator") or {}).get("name")
            formula = f"\n\nFormula: {numerator} / {denominator} (ratio of sums, not average of ratios)"
        elif metric_type == "simple":
            measure = (type_params.get("measure") or {}).get("name")
            formula = f"\n\nMeasure: {measure}"
        else:
            formula = ""

        text = (
            f"# Metric: {metric.get('label') or metric['name']} ({metric['name']})\n\n"
            f"Type: {metric_type}\n\n"
            f"{metric.get('description') or ''}"
            f"{formula}"
        ).strip()

        docs.append(
            ContextDocument(
                id=f"metric::{metric['name']}",
                text=text,
                metadata={"doc_type": "metric", "name": metric["name"], "metric_type": metric_type},
            )
        )
    return docs


def build_context_documents(manifest_path: Path = MANIFEST_PATH) -> list[ContextDocument]:
    manifest = _load_manifest(manifest_path)
    return _semantic_view_documents(manifest) + _metric_documents(manifest)
