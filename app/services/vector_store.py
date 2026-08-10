"""Local ChromaDB vector store for schema/metric grounding.

Uses ChromaDB's default embedding function (a local ONNX MiniLM model, no
API key or per-call cost) so the retrieval layer stays consistent with the
rest of this project's fully in-process, offline-friendly design.
"""
from __future__ import annotations

from pathlib import Path

import chromadb

from app.services.metadata_extraction import ContextDocument, build_context_documents

PERSIST_DIR = Path(__file__).resolve().parents[2] / "data" / "chroma"
COLLECTION_NAME = "business_context"


def get_client(persist_dir: Path = PERSIST_DIR) -> chromadb.ClientAPI:
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def get_collection(client: chromadb.ClientAPI | None = None):
    client = client or get_client()
    return client.get_or_create_collection(COLLECTION_NAME)


def index_documents(documents: list[ContextDocument], client: chromadb.ClientAPI | None = None) -> None:
    """Replaces the collection's contents with `documents` (full re-index)."""
    client = client or get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        pass
    collection = client.get_or_create_collection(COLLECTION_NAME)

    if not documents:
        return

    collection.add(
        ids=[doc.id for doc in documents],
        documents=[doc.text for doc in documents],
        metadatas=[doc.metadata for doc in documents],
    )


def rebuild_index(client: chromadb.ClientAPI | None = None) -> int:
    """Rebuilds the vector store from the current dbt manifest. Returns document count."""
    documents = build_context_documents()
    index_documents(documents, client=client)
    return len(documents)


if __name__ == "__main__":
    count = rebuild_index()
    print(f"Indexed {count} context documents into '{COLLECTION_NAME}' at {PERSIST_DIR}")
