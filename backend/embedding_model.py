"""
backend/embedding_model.py

Shared ML models used by the KG-RAG pipeline.

Models:

1. Entity embedding model
   BAAI/bge-base-en-v1.5

2. Cross encoder
   BAAI/bge-reranker-base

3. RAG/vector embedding model
   all-MiniLM-L6-v2

All models are loaded lazily and only once per
Python process.
"""

from sentence_transformers import SentenceTransformer, CrossEncoder


# =========================================================
# Configuration
# =========================================================

ENTITY_EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

CROSS_ENCODER_MODEL_NAME = "BAAI/bge-reranker-base"

RAG_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# =========================================================
# Singleton instances
# =========================================================

_entity_embedding_model = None
_cross_encoder = None
_rag_embedding_model = None


# =========================================================
# Entity embedding model
# =========================================================

def get_embedding_model() -> SentenceTransformer:
    """
    Return the shared BGE embedding model.

    Used by Entity Resolution.
    Loaded only on first use.
    """

    global _entity_embedding_model

    if _entity_embedding_model is None:

        print(
            f"Loading entity embedding model: "
            f"{ENTITY_EMBEDDING_MODEL_NAME}"
        )

        _entity_embedding_model = SentenceTransformer(
            ENTITY_EMBEDDING_MODEL_NAME
        )

        print(
            "Entity embedding model loaded."
        )

    return _entity_embedding_model


# =========================================================
# Cross encoder
# =========================================================

def get_cross_encoder() -> CrossEncoder:
    """
    Return the shared BGE reranker.

    Used by Entity Resolution for borderline pairs.
    Loaded only on first use.
    """

    global _cross_encoder

    if _cross_encoder is None:

        print(
            f"Loading cross-encoder: "
            f"{CROSS_ENCODER_MODEL_NAME}"
        )

        _cross_encoder = CrossEncoder(
            CROSS_ENCODER_MODEL_NAME
        )

        print(
            "Cross-encoder loaded."
        )

    return _cross_encoder


# =========================================================
# RAG embedding model
# =========================================================

def get_rag_embedding_model() -> SentenceTransformer:
    """
    Return the shared MiniLM embedding model.

    Used for:

    - Chunk vector embeddings
    - Entity vector embeddings
    - Query embeddings
    - Graph-path semantic ranking

    Loaded only on first use.
    """

    global _rag_embedding_model

    if _rag_embedding_model is None:

        print(
            f"Loading RAG embedding model: "
            f"{RAG_EMBEDDING_MODEL_NAME}"
        )

        _rag_embedding_model = SentenceTransformer(
            RAG_EMBEDDING_MODEL_NAME
        )

        print(
            "RAG embedding model loaded."
        )

    return _rag_embedding_model