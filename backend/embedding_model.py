"""
backend/embedding_model.py

Shared ML models used by the KG pipeline.

Models:

1. Entity embedding model
   BAAI/bge-base-en-v1.5

2. Cross encoder
   BAAI/bge-reranker-base

The models are loaded lazily and only once.
"""

from sentence_transformers import SentenceTransformer, CrossEncoder


# =========================================================
# Configuration
# =========================================================

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

CROSS_ENCODER_MODEL_NAME = "BAAI/bge-reranker-base"


# =========================================================
# Singleton instances
# =========================================================

_embedding_model = None
_cross_encoder = None


# =========================================================
# Embedding model
# =========================================================

def get_embedding_model() -> SentenceTransformer:
    """
    Return the shared BGE embedding model.

    The model is downloaded/loaded only on first use.
    """

    global _embedding_model

    if _embedding_model is None:

        print(
            f"Loading embedding model: "
            f"{EMBEDDING_MODEL_NAME}"
        )

        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        print(
            "Embedding model loaded."
        )

    return _embedding_model


# =========================================================
# Cross encoder
# =========================================================

def get_cross_encoder() -> CrossEncoder:
    """
    Return the shared BGE reranker.

    Loaded only when borderline entity pairs
    require additional verification.
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