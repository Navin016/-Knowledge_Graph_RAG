import re
from typing import Any

from sentence_transformers import SentenceTransformer

from backend.neo4j_client import Neo4jClient


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_ENTITY_K = 10
DEFAULT_GRAPH_K = 5

MAX_GRAPH_PATHS = 150


# ============================================================
# GENERIC METADATA RELATIONSHIPS
# These are domain-independent bibliographic/admin relations.
# ============================================================

METADATA_RELATIONS = {
    "PUBLISHED_BY",
    "PUBLISHED_IN",
    "AUTHORED_BY",
    "AUTHORED",
    "WRITTEN_BY",
    "CREATED_BY",
}


# ============================================================
# GENERIC ENTITY PENALTIES
# ============================================================

def looks_like_metadata_entity(name: str) -> bool:
    """
    Detect obvious bibliographic/title-like entities.

    This is intentionally generic and does not contain
    domain-specific vocabulary.
    """

    text = str(name).strip().lower()

    metadata_patterns = [
        "doi:",
        "volume ",
        "vol.",
        "journal",
        "conference",
        "proceedings",
        "paper",
        "article",
        "published",
        "isbn",
    ]

    return any(
        pattern in text
        for pattern in metadata_patterns
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# ENTITY SEARCH
# ============================================================

def vector_search_entities(
    client: Neo4jClient,
    query_embedding: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Semantic retrieval of Entity nodes.
    """

    query = """
    CYPHER 25

    MATCH (e:Entity)

    SEARCH e IN (
        VECTOR INDEX entity_name_vector
        FOR $query_embedding
        LIMIT $top_k
    )

    SCORE AS score

    RETURN
        e.name AS name,
        e.source_file AS source_file,
        score

    ORDER BY score DESC
    """

    with client.driver.session(
        database=client.database
    ) as session:

        result = session.run(
            query,
            query_embedding=query_embedding,
            top_k=top_k,
        )

        return [
            record.data()
            for record in result
        ]


# ============================================================
# GRAPH PATH RETRIEVAL
# ============================================================

def retrieve_graph_paths(
    client: Neo4jClient,
    entity_names: list[str],
    max_paths: int = MAX_GRAPH_PATHS,
) -> list[dict[str, Any]]:
    """
    Retrieve directed 1-2 hop paths around semantically
    relevant entities.
    """

    if not entity_names:
        return []

    query = """
    MATCH p = (a:Entity)-[*1..2]->(b:Entity)

    WHERE
        a.name IN $entity_names
        OR b.name IN $entity_names

    WITH
        p,
        nodes(p) AS path_nodes,
        relationships(p) AS path_relationships

    WHERE
        ALL(
            n IN path_nodes
            WHERE single(
                m IN path_nodes
                WHERE m = n
            )
        )

    RETURN
        [n IN path_nodes | n.name] AS entities,
        [r IN path_relationships | type(r)]
            AS relationships

    LIMIT $max_paths
    """

    with client.driver.session(
        database=client.database
    ) as session:

        result = session.run(
            query,
            entity_names=entity_names,
            max_paths=max_paths,
        )

        return [
            record.data()
            for record in result
        ]


# ============================================================
# PATH TEXT
# ============================================================

def path_to_text(
    path: dict[str, Any]
) -> str:
    """
    Convert a graph path into text for semantic ranking.
    """

    entities = path.get(
        "entities",
        []
    )

    relationships = path.get(
        "relationships",
        []
    )

    if not entities:
        return ""

    parts = []

    for i, relationship in enumerate(
        relationships
    ):

        if i >= len(entities) - 1:
            break

        subject = entities[i]
        object_name = entities[i + 1]

        parts.append(
            f"{subject} "
            f"{relationship} "
            f"{object_name}"
        )

    return " ; ".join(parts)


# ============================================================
# PATH SCORING
# ============================================================

def score_path(
    path: dict[str, Any],
    semantic_score: float,
    relevant_entity_names: set[str],
) -> float:
    """
    Combine:

    - semantic similarity
    - relevant endpoint count
    - direct-path preference
    - metadata penalty
    - metadata entity penalty

    All rules are domain-independent.
    """

    entities = path.get(
        "entities",
        []
    )

    relationships = path.get(
        "relationships",
        []
    )

    # --------------------------------------------------------
    # Number of relevant entities from semantic search
    # --------------------------------------------------------

    endpoint_matches = sum(
        1
        for entity in entities
        if entity in relevant_entity_names
    )

    # --------------------------------------------------------
    # Path length
    # --------------------------------------------------------

    hops = len(relationships)

    # Prefer shorter explanations.
    if hops == 1:
        path_length_bonus = 0.08
    elif hops == 2:
        path_length_bonus = 0.03
    else:
        path_length_bonus = 0.0

    # --------------------------------------------------------
    # Relevant endpoint bonus
    # --------------------------------------------------------

    endpoint_bonus = min(
        endpoint_matches * 0.06,
        0.12
    )

    # --------------------------------------------------------
    # Metadata relationship penalty
    # --------------------------------------------------------

    metadata_count = sum(
        1
        for relation in relationships
        if relation in METADATA_RELATIONS
    )

    metadata_penalty = min(
        metadata_count * 0.20,
        0.40
    )

    # --------------------------------------------------------
    # Metadata entity penalty
    # --------------------------------------------------------

    metadata_entity_count = sum(
        1
        for entity in entities
        if looks_like_metadata_entity(entity)
    )

    metadata_entity_penalty = min(
        metadata_entity_count * 0.10,
        0.20
    )

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    final_score = (
        float(semantic_score)
        + endpoint_bonus
        + path_length_bonus
        - metadata_penalty
        - metadata_entity_penalty
    )

    return final_score


# ============================================================
# PATH RANKING
# ============================================================

def rank_paths(
    paths: list[dict[str, Any]],
    query_embedding: list[float],
    model: SentenceTransformer,
    relevant_entity_names: set[str],
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Rank graph paths using semantic similarity plus
    structure/relevance heuristics.
    """

    if not paths:
        return []

    texts = [
        path_to_text(path)
        for path in paths
    ]

    valid_pairs = [
        (path, text)
        for path, text in zip(
            paths,
            texts
        )
        if text
    ]

    if not valid_pairs:
        return []

    valid_paths = [
        pair[0]
        for pair in valid_pairs
    ]

    valid_texts = [
        pair[1]
        for pair in valid_pairs
    ]

    # --------------------------------------------------------
    # Embed all graph-path texts
    # --------------------------------------------------------

    embeddings = model.encode(
        valid_texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # --------------------------------------------------------
    # Semantic similarity
    # --------------------------------------------------------

    semantic_scores = embeddings @ query_embedding

    ranked = []

    for path, text, semantic_score in zip(
        valid_paths,
        valid_texts,
        semantic_scores,
    ):

        final_score = score_path(
            path=path,
            semantic_score=float(semantic_score),
            relevant_entity_names=relevant_entity_names,
        )

        item = dict(path)

        item["text"] = text
        item["semantic_score"] = float(
            semantic_score
        )
        item["score"] = final_score

        ranked.append(item)

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return ranked[:top_k]


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_paths(
    paths: list[dict[str, Any]]
) -> list[dict[str, Any]]:

    seen = set()
    unique = []

    for path in paths:

        key = (
            tuple(
                path.get(
                    "entities",
                    []
                )
            ),
            tuple(
                path.get(
                    "relationships",
                    []
                )
            ),
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(path)

    return unique


# ============================================================
# GRAPH RETRIEVER
# ============================================================

class GraphRetriever:

    def __init__(
        self,
        entity_k: int = DEFAULT_ENTITY_K,
        graph_k: int = DEFAULT_GRAPH_K,
    ):

        self.entity_k = entity_k
        self.graph_k = graph_k

        self.client = Neo4jClient()

        self.client.driver.verify_connectivity()

        print(
            f"Loading graph retrieval model: {MODEL_NAME}"
        )

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        print(
            "Graph retrieval model loaded."
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
    ) -> dict[str, Any]:

        query = clean_text(query)

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        # -----------------------------------------------------
        # Query embedding
        # -----------------------------------------------------

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        query_embedding = [
            float(value)
            for value in query_embedding
        ]

        # -----------------------------------------------------
        # Semantic entity retrieval
        # -----------------------------------------------------

        entity_results = vector_search_entities(
            self.client,
            query_embedding,
            self.entity_k,
        )

        entity_names = [
            item["name"]
            for item in entity_results
        ]

        relevant_entity_names = set(
            entity_names
        )

        # -----------------------------------------------------
        # Graph path retrieval
        # -----------------------------------------------------

        graph_paths = retrieve_graph_paths(
            self.client,
            entity_names,
        )

        # -----------------------------------------------------
        # Deduplicate
        # -----------------------------------------------------

        graph_paths = deduplicate_paths(
            graph_paths
        )

        # -----------------------------------------------------
        # Rank
        # -----------------------------------------------------

        ranked_paths = rank_paths(
            paths=graph_paths,
            query_embedding=query_embedding,
            model=self.model,
            relevant_entity_names=relevant_entity_names,
            top_k=self.graph_k,
        )

        return {
            "query": query,
            "entities": entity_results,
            "graph_paths": ranked_paths,
        }

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):
        self.client.close()


# ============================================================
# OUTPUT
# ============================================================

def print_results(
    result: dict[str, Any]
):

    print()
    print("=" * 90)
    print("GENERIC SEMANTIC GRAPH RETRIEVAL")
    print("=" * 90)

    print()
    print("Question:")
    print("-" * 90)
    print(result["query"])

    # --------------------------------------------------------
    # Entity candidates
    # --------------------------------------------------------

    print()
    print("Semantic Entity Candidates:")
    print("-" * 90)

    for i, entity in enumerate(
        result["entities"],
        start=1,
    ):

        print(
            f"{i}. "
            f"{entity['name']} "
            f"(score={entity['score']:.4f})"
        )

    # --------------------------------------------------------
    # Ranked graph paths
    # --------------------------------------------------------

    print()
    print("Ranked Graph Paths:")
    print("-" * 90)

    paths = result["graph_paths"]

    if not paths:

        print("No graph paths found.")

    else:

        for i, path in enumerate(
            paths,
            start=1,
        ):

            print(
                f"{i}. "
                f"final={path['score']:.4f} "
                f"semantic={path['semantic_score']:.4f}"
            )

            print(
                f"   {path['text']}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    retriever = GraphRetriever(
        entity_k=10,
        graph_k=5,
    )

    try:

        query = (
            "How does fluorescence imaging "
            "reduce motion artifacts?"
        )

        result = retriever.search(
            query
        )

        print_results(
            result
        )

    finally:

        retriever.close()


if __name__ == "__main__":
    main()