from typing import Any

from backend.vector_retriever import VectorRetriever
from backend.graph_retriever import GraphRetriever


class UnifiedRetriever:

    def __init__(
        self,
        vector_k: int = 5,
        entity_k: int = 10,
        graph_k: int = 5,
    ):
        self.vector_retriever = VectorRetriever(
            top_k=vector_k
        )

        self.graph_retriever = GraphRetriever(
            entity_k=entity_k,
            graph_k=graph_k,
        )

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

        vector_results = self.vector_retriever.search(
            query
        )

        # -----------------------------------------------------
        # Graph retrieval
        # -----------------------------------------------------

        graph_results = self.graph_retriever.search(
            query
        )

        # -----------------------------------------------------
        # Unified result
        # -----------------------------------------------------

        return {
            "query": query,

            "vector_results": vector_results,

            "graph_results": graph_results,

            "context": {
                "source_text": vector_results,
                "graph_paths": graph_results.get(
                    "graph_paths",
                    []
                ),
            },
        }

    def close(self):
        self.vector_retriever.close()
        self.graph_retriever.close()


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
    print(result["query"])

    # ========================================================
    # VECTOR RESULTS
    # ========================================================

    print()
    print("VECTOR EVIDENCE")
    print("-" * 90)

    vector_results = result["vector_results"]

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
    # GRAPH RESULTS
    # ========================================================

    print()
    print("\nGRAPH EVIDENCE")
    print("-" * 90)

    graph_paths = result[
        "graph_results"
    ].get(
        "graph_paths",
        []
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
        print("No graph evidence found.")


def main():

    retriever = UnifiedRetriever(
        vector_k=5,
        entity_k=10,
        graph_k=5,
    )

    try:

        query = (
            "How does fluorescence imaging "
            "reduce motion artifacts?"
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