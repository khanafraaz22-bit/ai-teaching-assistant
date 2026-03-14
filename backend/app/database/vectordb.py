"""
ChromaDB vector store — persisted locally for development.
Swap this module for Pinecone / Weaviate in production without touching any
other part of the codebase.
"""

import logging
import chromadb
from typing import Optional
from chromadb.config import Settings as ChromaSettings
from app.config.settings import settings

logger = logging.getLogger(__name__)

_chroma_client: Optional[chromadb.PersistentClient] = None
_collection = None


def init_vector_db() -> None:
    global _chroma_client, _collection
    logger.info("Initializing ChromaDB...")
    _chroma_client = chromadb.PersistentClient(
        path=settings.CHROMA_PERSIST_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    _collection = _chroma_client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # Cosine similarity for text
    )
    logger.info(f"✅ ChromaDB ready. Collection: '{settings.CHROMA_COLLECTION_NAME}'")


def get_collection():
    """Return the active Chroma collection. Used across RAG services."""
    if _collection is None:
        raise RuntimeError("ChromaDB is not initialized. Call init_vector_db() first.")
    return _collection


def get_chroma_client():
    if _chroma_client is None:
        raise RuntimeError("ChromaDB client is not initialized.")
    return _chroma_client