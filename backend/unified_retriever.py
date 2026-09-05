from typing import Any

from backend.vector_retriever import VectorRetriever
from backend.graph_retriever import GraphRetriever


# ============================================================
# Retrieval configuration
# ============================================================

VECTOR_K = 5
ENTITY_K = 10
GRAPH_K = 5

# Expand only the strongest vector results.
NEIGHBOR_EXPAND_TOP_K = 3

# Include one chunk before and one chunk after.
NEIGHBOR_RADIUS = 1


class UnifiedRetriever:

    def __init__(
        self,
        vector_k: int = VECTOR_K,
        entity_k: int = ENTITY_K,
        graph_k: int = GRAPH_K,
    ):

        self.vector_retriever = VectorRetriever(
            top_k=vector_k
        )

        self.graph_retriever = GraphRetriever(
            entity_k=entity_k,
            graph_k=graph_k,
        )

    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(
        self,
        query: str,
    ) -> dict[str, Any]:

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        # -----------------------------------------------------
        # Vector retrieval
        # -----------------------------------------------------

        vector_results = (
            self.vector_retriever.search(
                query
            )
        )

        # -----------------------------------------------------
        # Neighbor expansion
        # -----------------------------------------------------

        expanded_vector_results = (
            self.vector_retriever.expand_neighbors(
                vector_results,
                radius=NEIGHBOR_RADIUS,
                expand_top_k=NEIGHBOR_EXPAND_TOP_K,
            )
        )

        # -----------------------------------------------------
        # Graph retrieval
        # -----------------------------------------------------

        graph_results = (
            self.graph_retriever.search(
                query
            )
        )

        # -----------------------------------------------------
        # Unified result
        # -----------------------------------------------------

        return {
            "query": query,

            # Keep original results available.
            "vector_results": vector_results,

            # Expanded results used for RAG context.
            "expanded_vector_results":
                expanded_vector_results,

            "graph_results": graph_results,

            "context": {
                "source_text":
                    expanded_vector_results,

                "graph_paths":
                    graph_results.get(
                        "graph_paths",
                        [],
                    ),
            },
        }

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        self.vector_retriever.close()

        self.graph_retriever.close()


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    result: dict[str, Any]
):

    print()
    print("=" * 90)
    print("UNIFIED RETRIEVAL")
    print("=" * 90)

    print()
    print("QUESTION")
    print("-" * 90)

    print(
        result["query"]
    )

    # ========================================================
    # ORIGINAL VECTOR RESULTS
    # ========================================================

    print()
    print("ORIGINAL VECTOR RESULTS")
    print("-" * 90)

    vector_results = (
        result["vector_results"]
    )

    for i, item in enumerate(
        vector_results,
        start=1,
    ):

        print(
            f"\n{i}. "
            f"Chunk={item['chunk_id']} "
            f"Score={item['score']:.4f}"
        )

        print(
            item["text"][:700]
        )

    # ========================================================
    # EXPANDED VECTOR RESULTS
    # ========================================================

    print()
    print("EXPANDED VECTOR CONTEXT")
    print("-" * 90)

    expanded_results = (
        result["expanded_vector_results"]
    )

    for i, item in enumerate(
        expanded_results,
        start=1,
    ):

        result_type = (
            "NEIGHBOR"
            if item.get(
                "is_neighbor",
                False,
            )
            else "VECTOR"
        )

        print(
            f"\n{i}. "
            f"Chunk={item['chunk_id']} "
            f"Type={result_type} "
            f"Score={item['score']:.4f}"
        )

        print(
            item["text"][:700]
        )

    # ========================================================
    # GRAPH RESULTS
    # ========================================================

    print()
    print("GRAPH EVIDENCE")
    print("-" * 90)

    graph_paths = (
        result["graph_results"].get(
            "graph_paths",
            [],
        )
    )

    for i, path in enumerate(
        graph_paths,
        start=1,
    ):

        print(
            f"\n{i}. "
            f"Score={path['score']:.4f}"
        )

        print(
            path["text"]
        )

    if not graph_paths:

        print(
            "No graph evidence found."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    retriever = UnifiedRetriever(
        vector_k=5,
        entity_k=10,
        graph_k=5,
    )

    try:

        query = (
            "How did the 5.9 nM and "
            "13.2 nM detection limits compare?"
        )

        result = retriever.retrieve(
            query
        )

        print_results(
            result
        )

    finally:

        retriever.close()


if __name__ == "__main__":
    main()