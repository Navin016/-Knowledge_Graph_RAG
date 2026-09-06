from typing import Any

from backend.vector_retriever import VectorRetriever
from backend.graph_retriever import GraphRetriever


# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

VECTOR_K = 5
ENTITY_K = 12
GRAPH_K = 5

# Expand only the strongest original vector hits.
NEIGHBOR_EXPAND_TOP_K = 3

# Include one chunk before and after each selected chunk.
NEIGHBOR_RADIUS = 1


class UnifiedRetriever:

    def __init__(
        self,
        vector_k: int = VECTOR_K,
        entity_k: int = ENTITY_K,
        graph_k: int = GRAPH_K,
    ) -> None:

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
        # 1. Vector retrieval
        # -----------------------------------------------------

        vector_results = (
            self.vector_retriever.search(
                query
            )
        )

        # -----------------------------------------------------
        # 2. Neighbor expansion
        # -----------------------------------------------------

        expanded_vector_results = (
            self.vector_retriever.expand_neighbors(
                vector_results,
                radius=NEIGHBOR_RADIUS,
                expand_top_k=NEIGHBOR_EXPAND_TOP_K,
            )
        )

        # -----------------------------------------------------
        # 3. Graph retrieval
        # -----------------------------------------------------

        graph_results = (
            self.graph_retriever.search(
                query
            )
        )

        graph_entities = graph_results.get(
            "entities",
            [],
        )

        graph_paths = graph_results.get(
            "graph_paths",
            [],
        )

        # -----------------------------------------------------
        # 4. Unified result
        # -----------------------------------------------------

        return {
            "query": query,

            # Original semantic chunk hits.
            "vector_results": vector_results,

            # Neighbor-expanded chunk context.
            "expanded_vector_results":
                expanded_vector_results,

            # Graph retrieval.
            "graph_results": graph_results,

            # Convenience accessors.
            "graph_entities": graph_entities,
            "graph_paths": graph_paths,

            # This is the context passed to the final LLM.
            "context": {
                "source_text":
                    expanded_vector_results,

                "graph_entities":
                    graph_entities,

                "graph_paths":
                    graph_paths,
            },
        }

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self) -> None:

        self.vector_retriever.close()
        self.graph_retriever.close()


# ============================================================
# DEBUG OUTPUT
# ============================================================

def print_results(
    result: dict[str, Any],
) -> None:

    print()
    print("=" * 100)
    print("UNIFIED RETRIEVAL")
    print("=" * 100)

    print()
    print("QUESTION")
    print("-" * 100)
    print(result["query"])

    # ========================================================
    # ORIGINAL VECTOR RESULTS
    # ========================================================

    print()
    print("ORIGINAL VECTOR RESULTS")
    print("-" * 100)

    vector_results = result.get(
        "vector_results",
        [],
    )

    if not vector_results:
        print("No vector results found.")

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
            item.get("text", "")[:1000]
        )

    # ========================================================
    # EXPANDED VECTOR CONTEXT
    # ========================================================

    print()
    print("EXPANDED VECTOR CONTEXT")
    print("-" * 100)

    expanded_results = result.get(
        "expanded_vector_results",
        [],
    )

    if not expanded_results:
        print("No expanded vector results found.")

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
            item.get("text", "")[:1000]
        )

    # ========================================================
    # GRAPH ENTITIES
    # ========================================================

    print()
    print("GRAPH ENTITY CANDIDATES")
    print("-" * 100)

    graph_entities = result.get(
        "graph_entities",
        [],
    )

    if not graph_entities:
        print("No graph entities found.")

    for i, entity in enumerate(
        graph_entities,
        start=1,
    ):

        print(
            f"{i}. {entity['name']} "
            f"(score={entity.get('score', 0.0):.4f}, "
            f"semantic={entity.get('semantic_score', 0.0):.4f}, "
            f"lexical={entity.get('lexical_score', 0.0):.4f}, "
            f"matched={entity.get('matched_token_count', 0)}, "
            f"exact={entity.get('exact_phrase_match', False)})"
        )

    # ========================================================
    # GRAPH EVIDENCE
    # ========================================================

    print()
    print("GRAPH EVIDENCE")
    print("-" * 100)

    graph_paths = result.get(
        "graph_paths",
        [],
    )

    if not graph_paths:
        print("No graph evidence found.")

    for i, path in enumerate(
        graph_paths,
        start=1,
    ):

        print(
            f"\n{i}. "
            f"Final={path.get('score', 0.0):.4f} "
            f"Selection={path.get('selection_score', 0.0):.4f} "
            f"Semantic={path.get('semantic_score', 0.0):.4f}"
        )

        print(
            path.get("text", "")
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    retriever = UnifiedRetriever(
        vector_k=5,
        entity_k=12,
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
