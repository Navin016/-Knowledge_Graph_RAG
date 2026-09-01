"""
backend/embedding_model.py

Shared embedding model for the project.

Currently:
    all-MiniLM-L6-v2

The model is loaded only once and reused by:

    relation_normalizer.py
    entity_resolution.py
    retrieval.py

This avoids loading the same model multiple times.
"""

from sentence_transformers import SentenceTransformer


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = "all-MiniLM-L6-v2"


# =========================================================
# Singleton model
# =========================================================

_model = None


def get_embedding_model() -> SentenceTransformer:
    """
    Return the shared SentenceTransformer model.

    The model is loaded only on the first call.

    Returns:
        Shared SentenceTransformer instance.
    """

    global _model

    if _model is None:

        print(
            f"Loading embedding model: {MODEL_NAME}"
        )

        _model = SentenceTransformer(
            MODEL_NAME
        )

        print(
            "Embedding model loaded."
        )

    return _model