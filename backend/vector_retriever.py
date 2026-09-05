from typing import Any

from backend.embedding_model import get_rag_embedding_model
from backend.neo4j_client import Neo4jClient


MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_INDEX_NAME = "chunk_text_vector"

# ------------------------------------------------------------
# Neighbor expansion settings
# ------------------------------------------------------------

DEFAULT_NEIGHBOR_RADIUS = 1
DEFAULT_EXPAND_TOP_K = 3


class VectorRetriever:

    def __init__(
        self,
        top_k: int = 5,
    ):
        self.top_k = top_k
        self.client = Neo4jClient()

        # Shared MiniLM singleton
        self.model = get_rag_embedding_model()

    # ========================================================
    # VECTOR SEARCH
    # ========================================================

    def search(
        self,
        query: str,
    ) -> list[dict[str, Any]]:
        """
        Embed the user query and perform vector search in Neo4j.
        """

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        # -----------------------------------------------------
        # Create query embedding
        # -----------------------------------------------------

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        # -----------------------------------------------------
        # Neo4j vector search
        # -----------------------------------------------------

        cypher = """
        MATCH (node:Chunk)
        SEARCH node IN (
            VECTOR INDEX chunk_text_vector
            FOR $query_embedding
            LIMIT $top_k
        )
        SCORE AS score

        RETURN
            node.chunk_id AS chunk_id,
            node.source_file AS source_file,
            node.text AS text,
            score

        ORDER BY score DESC
        """

        with self.client.driver.session(
            database=self.client.database
        ) as session:

            result = session.run(
                cypher,
                index_name=VECTOR_INDEX_NAME,
                top_k=self.top_k,
                query_embedding=query_embedding,
            )

            return [
                record.data()
                for record in result
            ]

    # ========================================================
    # NEIGHBOR EXPANSION
    # ========================================================

    def expand_neighbors(
        self,
        results: list[dict[str, Any]],
        radius: int = DEFAULT_NEIGHBOR_RADIUS,
        expand_top_k: int = DEFAULT_EXPAND_TOP_K,
    ) -> list[dict[str, Any]]:
        """
        Expand the strongest vector results with nearby chunks.

        Example:

            Retrieved:
                chunk 9

            radius=1 gives:
                chunk 8
                chunk 9
                chunk 10

        Only the top `expand_top_k` vector results are expanded.

        This improves context completeness when an answer is split
        across chunk boundaries.
        """

        if not results:
            return []

        if radius < 0:
            raise ValueError(
                "radius cannot be negative."
            )

        if expand_top_k <= 0:
            return results

        # -----------------------------------------------------
        # Select strongest retrieved chunks to expand
        # -----------------------------------------------------

        selected_results = results[
            :expand_top_k
        ]

        # -----------------------------------------------------
        # Build requested chunk IDs grouped by source file
        # -----------------------------------------------------

        requested_by_source: dict[str, set[int]] = {}

        for item in selected_results:

            chunk_id = item.get("chunk_id")
            source_file = item.get("source_file")

            if chunk_id is None or source_file is None:
                continue

            try:
                chunk_id = int(chunk_id)
            except (TypeError, ValueError):
                continue

            requested_by_source.setdefault(
                source_file,
                set(),
            )

            for offset in range(
                -radius,
                radius + 1,
            ):
                neighbor_id = chunk_id + offset

                if neighbor_id > 0:
                    requested_by_source[
                        source_file
                    ].add(neighbor_id)

        if not requested_by_source:
            return results

        # -----------------------------------------------------
        # Query all neighbors in ONE Neo4j query
        # -----------------------------------------------------

        source_names = list(
            requested_by_source.keys()
        )

        pairs = []

        for source_file, chunk_ids in (
            requested_by_source.items()
        ):
            for chunk_id in chunk_ids:
                pairs.append(
                    {
                        "source_file": source_file,
                        "chunk_id": chunk_id,
                    }
                )

        cypher = """
        UNWIND $pairs AS pair

        MATCH (node:Chunk)
        WHERE node.source_file = pair.source_file
          AND node.chunk_id = pair.chunk_id

        RETURN
            node.chunk_id AS chunk_id,
            node.source_file AS source_file,
            node.text AS text
        """

        with self.client.driver.session(
            database=self.client.database
        ) as session:

            result = session.run(
                cypher,
                pairs=pairs,
            )

            neighbor_rows = [
                record.data()
                for record in result
            ]

        # -----------------------------------------------------
        # Create lookup table
        # -----------------------------------------------------

        neighbor_lookup: dict[
            tuple[str, int],
            dict[str, Any],
        ] = {}

        for item in neighbor_rows:

            source_file = item.get(
                "source_file"
            )

            chunk_id = item.get(
                "chunk_id"
            )

            if (
                source_file is None
                or chunk_id is None
            ):
                continue

            try:
                chunk_id = int(chunk_id)
            except (TypeError, ValueError):
                continue

            neighbor_lookup[
                (
                    source_file,
                    chunk_id,
                )
            ] = {
                "chunk_id": chunk_id,
                "source_file": source_file,
                "text": item.get(
                    "text",
                    "",
                ),
            }

        # -----------------------------------------------------
        # Preserve original vector scores
        # -----------------------------------------------------

        score_lookup: dict[
            tuple[str, int],
            float,
        ] = {}

        for item in results:

            source_file = item.get(
                "source_file"
            )

            chunk_id = item.get(
                "chunk_id"
            )

            if (
                source_file is None
                or chunk_id is None
            ):
                continue

            try:
                chunk_id = int(chunk_id)
            except (TypeError, ValueError):
                continue

            score_lookup[
                (
                    source_file,
                    chunk_id,
                )
            ] = float(
                item.get(
                    "score",
                    0.0,
                )
            )

        # -----------------------------------------------------
        # Merge original + neighboring chunks
        # -----------------------------------------------------

        merged: dict[
            tuple[str, int],
            dict[str, Any],
        ] = {}

        # Original results first
        for item in results:

            source_file = item.get(
                "source_file"
            )

            chunk_id = item.get(
                "chunk_id"
            )

            if (
                source_file is None
                or chunk_id is None
            ):
                continue

            try:
                chunk_id = int(chunk_id)
            except (TypeError, ValueError):
                continue

            key = (
                source_file,
                chunk_id,
            )

            merged[key] = {
                "chunk_id": chunk_id,
                "source_file": source_file,
                "text": item.get(
                    "text",
                    "",
                ),
                "score": float(
                    item.get(
                        "score",
                        0.0,
                    )
                ),
                "is_neighbor": False,
            }

        # Neighbor chunks
        for key, item in neighbor_lookup.items():

            if key in merged:
                continue

            merged[key] = {
                "chunk_id": item["chunk_id"],
                "source_file": item["source_file"],
                "text": item["text"],
                "score": score_lookup.get(
                    key,
                    0.0,
                ),
                "is_neighbor": True,
            }

        # -----------------------------------------------------
        # Sort by source file + chunk ID
        #
        # This makes the final context readable:
        #
        # Chunk 8
        # Chunk 9
        # Chunk 10
        # -----------------------------------------------------

        expanded_results = sorted(
            merged.values(),
            key=lambda item: (
                item["source_file"],
                item["chunk_id"],
            ),
        )

        return expanded_results

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):
        self.client.close()


# ============================================================
# TEST
# ============================================================

def main():

    retriever = VectorRetriever(
        top_k=5
    )

    try:

        query = (
            "How did the 5.9 nM and "
            "13.2 nM detection limits compare?"
        )

        print()
        print("=" * 80)
        print("VECTOR SEARCH + NEIGHBOR EXPANSION")
        print("=" * 80)

        print(
            f"Query: {query}"
        )

        # ----------------------------------------------------
        # Original vector retrieval
        # ----------------------------------------------------

        results = retriever.search(
            query
        )

        print()
        print(
            f"Original retrieved chunks: "
            f"{len(results)}"
        )

        # ----------------------------------------------------
        # Neighbor expansion
        # ----------------------------------------------------

        expanded_results = (
            retriever.expand_neighbors(
                results,
                radius=1,
                expand_top_k=3,
            )
        )

        print(
            f"Expanded chunks: "
            f"{len(expanded_results)}"
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        for i, result in enumerate(
            expanded_results,
            start=1,
        ):

            marker = (
                "NEIGHBOR"
                if result.get(
                    "is_neighbor",
                    False,
                )
                else "VECTOR"
            )

            print()
            print("-" * 80)

            print(
                f"{i}. "
                f"Chunk={result['chunk_id']} "
                f"Type={marker} "
                f"Score={result['score']:.4f}"
            )

            print(
                f"Source={result['source_file']}"
            )

            print()

            print(
                result["text"][:1000]
            )

    finally:

        retriever.close()


if __name__ == "__main__":
    main()