import re
from typing import Any

from backend.embedding_model import get_rag_embedding_model
from backend.neo4j_client import Neo4jClient


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_ENTITY_K = 10
DEFAULT_GRAPH_K = 5

# Keep separate semantic and lexical graph seeds.
SEMANTIC_SEED_K = 6
LEXICAL_SEED_K = 6

# Broad graph candidate pool.
MAX_GRAPH_PATHS = 1000

# Controlled maximum traversal depth.
MAX_GRAPH_HOPS = 3

# Number of lexical entity candidates collected.
LEXICAL_ENTITY_K = 40


# ============================================================
# METADATA RELATIONSHIPS
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
# LOW-INFORMATION RELATIONSHIPS
# ============================================================

GENERIC_RELATIONS = {
    "RELATED_TO",
    "ASSOCIATED_WITH",
    "CONNECTED_TO",
}


# ============================================================
# GENERIC QUERY INTENT TERMS
# ============================================================

MEASUREMENT_TERMS = {
    "minimum",
    "maximum",
    "concentration",
    "amount",
    "value",
    "rate",
    "limit",
    "threshold",
    "measurement",
    "detected",
    "detection",
    "sensitivity",
    "percentage",
    "percent",
    "ratio",
    "quantity",
    "level",
    "intensity",
    "temperature",
    "pressure",
    "time",
    "duration",
    "power",
    "frequency",
    "size",
}


# ============================================================
# METADATA ENTITY DETECTION
# ============================================================

def looks_like_metadata_entity(
    name: str,
) -> bool:
    """
    Detect obvious bibliographic/admin entities.

    Domain-independent.
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
# VALUE ENTITY DETECTION
# ============================================================

def looks_like_value_entity(
    name: str,
) -> bool:
    """
    Detect generic numerical/scientific value entities.

    Examples:
        13.2 nM
        227.8 nM in DC mode
        45%
        4 mW
        20 ms
    """

    text = str(name).strip().lower()

    patterns = [
        r"\b\d+(?:\.\d+)?\s*%",
        r"\b\d+(?:\.\d+)?\s*(?:nm|um|mm|cm|m)\b",
        r"\b\d+(?:\.\d+)?\s*(?:mw|kw|w)\b",
        r"\b\d+(?:\.\d+)?\s*(?:hz|khz|mhz|ghz)\b",
        r"\b\d+(?:\.\d+)?\s*(?:ms|us|s|min|sec)\b",
    ]

    return any(
        re.search(
            pattern,
            text,
        )
        for pattern in patterns
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(
    text: str,
) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip()


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(
    text: str,
) -> set[str]:
    """
    Lightweight lexical tokenization.
    """

    text = str(text).lower()

    tokens = re.findall(
        r"[a-z0-9]+(?:\.[a-z0-9]+)?",
        text,
    )

    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "what",
        "how",
        "why",
        "when",
        "where",
        "which",
        "does",
        "do",
        "did",
        "can",
        "could",
        "would",
        "should",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "from",
        "by",
        "and",
        "or",
        "than",
        "this",
        "that",
        "these",
        "those",
        "under",
        "over",
        "into",
        "through",
        "using",
    }

    return {
        token
        for token in tokens
        if token not in stopwords
        and len(token) > 1
    }


# ============================================================
# LEXICAL OVERLAP
# ============================================================

def lexical_overlap_score(
    query_tokens: set[str],
    text: str,
) -> float:

    if not query_tokens:
        return 0.0

    text_tokens = tokenize(
        text
    )

    if not text_tokens:
        return 0.0

    overlap = query_tokens.intersection(
        text_tokens
    )

    return len(overlap) / max(
        len(query_tokens),
        1,
    )


# ============================================================
# ENTITY VECTOR SEARCH
# ============================================================

def vector_search_entities(
    client: Neo4jClient,
    query_embedding: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Semantic entity retrieval from Neo4j.
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
# ENTITY LEXICAL SEARCH
# ============================================================

def lexical_search_entities(
    client: Neo4jClient,
    query_tokens: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Retrieve entities whose names contain explicit query
    tokens.
    """

    if not query_tokens:
        return []

    query = """
    MATCH (e:Entity)

    WHERE
        ANY(
            token IN $query_tokens
            WHERE toLower(e.name) CONTAINS token
        )

    RETURN DISTINCT
        e.name AS name,
        e.source_file AS source_file

    LIMIT $top_k
    """

    with client.driver.session(
        database=client.database
    ) as session:

        result = session.run(
            query,
            query_tokens=[
                token.lower()
                for token in query_tokens
            ],
            top_k=top_k,
        )

        return [
            record.data()
            for record in result
        ]


# ============================================================
# SEMANTIC SEEDS
# ============================================================

def rank_semantic_entities(
    vector_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Preserve the best semantic candidates independently.
    """

    ranked = []

    for item in vector_results:

        name = item.get(
            "name",
            "",
        )

        if not name:
            continue

        if looks_like_metadata_entity(
            name
        ):
            continue

        score = float(
            item.get(
                "score",
                0.0,
            )
        )

        ranked.append(
            {
                "name": name,
                "source_file": item.get(
                    "source_file",
                    "",
                ),
                "score": score,
                "semantic_score": score,
                "lexical_score": 0.0,
                "from_vector": True,
                "from_lexical": False,
            }
        )

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return ranked[
        :SEMANTIC_SEED_K
    ]


# ============================================================
# LEXICAL SEEDS
# ============================================================

def rank_lexical_entities(
    lexical_results: list[dict[str, Any]],
    query_tokens: set[str],
    vector_entity_names: set[str],
) -> list[dict[str, Any]]:
    """
    Preserve explicit lexical entities independently of
    semantic candidates.
    """

    ranked = []

    for item in lexical_results:

        name = item.get(
            "name",
            "",
        )

        if not name:
            continue

        if name in vector_entity_names:
            continue

        if looks_like_metadata_entity(
            name
        ):
            continue

        lexical_score = lexical_overlap_score(
            query_tokens=query_tokens,
            text=name,
        )

        if lexical_score <= 0.0:
            continue

        score = lexical_score

        # Numerical entities are particularly useful for
        # measurement/comparison questions.
        if looks_like_value_entity(
            name
        ):
            score += 0.05

        ranked.append(
            {
                "name": name,
                "source_file": item.get(
                    "source_file",
                    "",
                ),
                "score": float(
                    score
                ),
                "semantic_score": 0.0,
                "lexical_score": float(
                    lexical_score
                ),
                "from_vector": False,
                "from_lexical": True,
            }
        )

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return ranked[
        :LEXICAL_SEED_K
    ]


# ============================================================
# HYBRID SEED RETRIEVAL
# ============================================================

def hybrid_entity_search(
    client: Neo4jClient,
    query: str,
    query_embedding: list[float],
) -> list[dict[str, Any]]:
    """
    Produce:

        6 semantic seeds
        +
        6 lexical seeds
    """

    query_tokens = tokenize(
        query
    )

    # --------------------------------------------------------
    # Semantic candidates
    # --------------------------------------------------------

    vector_results = vector_search_entities(
        client=client,
        query_embedding=query_embedding,
        top_k=DEFAULT_ENTITY_K,
    )

    semantic_seeds = rank_semantic_entities(
        vector_results=vector_results,
    )

    semantic_names = {
        item["name"]
        for item in semantic_seeds
    }

    # --------------------------------------------------------
    # Lexical candidates
    # --------------------------------------------------------

    lexical_results = lexical_search_entities(
        client=client,
        query_tokens=list(
            query_tokens
        ),
        top_k=LEXICAL_ENTITY_K,
    )

    lexical_seeds = rank_lexical_entities(
        lexical_results=lexical_results,
        query_tokens=query_tokens,
        vector_entity_names=semantic_names,
    )

    # --------------------------------------------------------
    # Preserve both pools
    # --------------------------------------------------------

    combined = []

    combined.extend(
        semantic_seeds
    )

    combined.extend(
        lexical_seeds
    )

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    seen = set()

    unique = []

    for item in combined:

        name = item["name"]

        if name in seen:
            continue

        seen.add(name)
        unique.append(item)

    return unique


# ============================================================
# GRAPH PATH RETRIEVAL
# ============================================================

def retrieve_graph_paths(
    client: Neo4jClient,
    entity_names: list[str],
    max_paths: int = MAX_GRAPH_PATHS,
) -> list[dict[str, Any]]:
    """
    Retrieve a broad 1-3 hop graph candidate pool.
    """

    if not entity_names:
        return []

    query = f"""
    MATCH p = (a:Entity)-[*1..{MAX_GRAPH_HOPS}]->(b:Entity)

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
# PATH TO TEXT
# ============================================================

def path_to_text(
    path: dict[str, Any],
) -> str:

    entities = path.get(
        "entities",
        [],
    )

    relationships = path.get(
        "relationships",
        [],
    )

    if not entities:
        return ""

    parts = []

    for i, relationship in enumerate(
        relationships
    ):

        if i >= len(entities) - 1:
            break

        parts.append(
            f"{entities[i]} "
            f"{relationship} "
            f"{entities[i + 1]}"
        )

    return " ; ".join(
        parts
    )


# ============================================================
# VALUE SCORE
# ============================================================

def value_relevance_score(
    entities: list[str],
) -> float:

    value_count = sum(
        1
        for entity in entities
        if looks_like_value_entity(
            entity
        )
    )

    return min(
        value_count * 0.04,
        0.08,
    )


# ============================================================
# RELATION QUALITY
# ============================================================

def relation_specificity_score(
    relationships: list[str],
) -> float:

    if not relationships:
        return 0.0

    generic_count = sum(
        1
        for relation in relationships
        if relation in GENERIC_RELATIONS
    )

    informative_count = (
        len(relationships)
        - generic_count
    )

    informative_bonus = min(
        informative_count * 0.03,
        0.06,
    )

    generic_penalty = min(
        generic_count * 0.035,
        0.07,
    )

    return (
        informative_bonus
        - generic_penalty
    )


# ============================================================
# PATH METADATA PENALTY
# ============================================================

def path_metadata_penalty(
    entities: list[str],
    relationships: list[str],
) -> float:

    relation_count = sum(
        1
        for relation in relationships
        if relation in METADATA_RELATIONS
    )

    entity_count = sum(
        1
        for entity in entities
        if looks_like_metadata_entity(
            entity
        )
    )

    return (
        min(
            relation_count * 0.20,
            0.40,
        )
        +
        min(
            entity_count * 0.10,
            0.20,
        )
    )


# ============================================================
# BASE PATH SCORE
# ============================================================

def score_path(
    path: dict[str, Any],
    semantic_score: float,
    relevant_entity_names: set[str],
    query_tokens: set[str],
) -> float:

    entities = path.get(
        "entities",
        [],
    )

    relationships = path.get(
        "relationships",
        [],
    )

    text = path_to_text(
        path
    )

    # --------------------------------------------------------
    # Seed relevance
    # --------------------------------------------------------

    seed_matches = sum(
        1
        for entity in entities
        if entity in relevant_entity_names
    )

    seed_bonus = min(
        seed_matches * 0.06,
        0.12,
    )

    # --------------------------------------------------------
    # Path length
    # --------------------------------------------------------

    hops = len(
        relationships
    )

    if hops == 1:
        hop_bonus = 0.08

    elif hops == 2:
        hop_bonus = 0.03

    else:
        hop_bonus = 0.0

    # --------------------------------------------------------
    # Lexical relevance
    # --------------------------------------------------------

    lexical_score = lexical_overlap_score(
        query_tokens=query_tokens,
        text=text,
    )

    lexical_bonus = min(
        lexical_score * 0.10,
        0.10,
    )

    # --------------------------------------------------------
    # Value relevance
    # --------------------------------------------------------

    value_bonus = (
        value_relevance_score(
            entities
        )
    )

    # --------------------------------------------------------
    # Relation quality
    # --------------------------------------------------------

    specificity_bonus = (
        relation_specificity_score(
            relationships
        )
    )

    # --------------------------------------------------------
    # Metadata penalty
    # --------------------------------------------------------

    metadata_penalty = (
        path_metadata_penalty(
            entities=entities,
            relationships=relationships,
        )
    )

    # --------------------------------------------------------
    # Final base score
    # --------------------------------------------------------

    return (
        float(semantic_score)
        + seed_bonus
        + hop_bonus
        + lexical_bonus
        + value_bonus
        + specificity_bonus
        - metadata_penalty
    )


# ============================================================
# QUERY COVERAGE
# ============================================================

def get_coverage_tokens(
    query_tokens: set[str],
) -> set[str]:
    """
    Keep meaningful query concepts for coverage.

    Measurement terms are retained because they are important
    to numerical/comparison questions.
    """

    return set(
        query_tokens
    )


def path_coverage_tokens(
    path: dict[str, Any],
    query_tokens: set[str],
) -> set[str]:

    text_tokens = tokenize(
        path_to_text(path)
    )

    return text_tokens.intersection(
        get_coverage_tokens(
            query_tokens
        )
    )


# ============================================================
# QUERY INTENT
# ============================================================

def is_measurement_query(
    query_tokens: set[str],
) -> bool:

    return bool(
        query_tokens.intersection(
            MEASUREMENT_TERMS
        )
    )


# ============================================================
# VALUE PATH CHECK
# ============================================================

def contains_value(
    path: dict[str, Any],
) -> bool:

    return any(
        looks_like_value_entity(
            entity
        )
        for entity in path.get(
            "entities",
            [],
        )
    )


# ============================================================
# PATH DUPLICATE KEY
# ============================================================

def path_key(
    path: dict[str, Any],
) -> tuple:

    return (
        tuple(
            path.get(
                "entities",
                [],
            )
        ),
        tuple(
            path.get(
                "relationships",
                [],
            )
        ),
    )


# ============================================================
# FINAL PATH SELECTION
# ============================================================

def select_final_paths(
    candidates: list[dict[str, Any]],
    query_tokens: set[str],
    model,
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Greedy evidence selection.

    The first path is the strongest individual result.

    Later paths receive a bonus for contributing query terms
    that have not already been covered.

    For measurement questions, strong value-bearing paths
    receive an additional modest bonus.

    This prevents several near-identical paths from consuming
    all five result slots.
    """

    if not candidates:
        return []

    selected = []

    remaining = list(
        candidates
    )

    covered_tokens: set[str] = set()

    measurement_query = is_measurement_query(
        query_tokens
    )

    while (
        remaining
        and len(selected) < top_k
    ):

        best_index = None
        best_selection_score = float(
            "-inf"
        )

        for index, candidate in enumerate(
            remaining
        ):

            base_score = float(
                candidate.get(
                    "score",
                    0.0,
                )
            )

            candidate_tokens = (
                path_coverage_tokens(
                    path=candidate,
                    query_tokens=query_tokens,
                )
            )

            # ------------------------------------------------
            # Reward new query concepts.
            # ------------------------------------------------

            new_tokens = (
                candidate_tokens
                - covered_tokens
            )

            coverage_bonus = min(
                len(new_tokens) * 0.06,
                0.18,
            )

            # ------------------------------------------------
            # Measurement/value bonus.
            # ------------------------------------------------

            value_bonus = 0.0

            if (
                measurement_query
                and contains_value(
                    candidate
                )
            ):
                value_bonus = 0.08

            # ------------------------------------------------
            # Redundancy penalty.
            # ------------------------------------------------

            redundancy_penalty = 0.0

            if selected:

                candidate_entities = set(
                    candidate.get(
                        "entities",
                        [],
                    )
                )

                for previous in selected:

                    previous_entities = set(
                        previous.get(
                            "entities",
                            [],
                        )
                    )

                    intersection = (
                        candidate_entities
                        & previous_entities
                    )

                    union = (
                        candidate_entities
                        | previous_entities
                    )

                    if union:

                        entity_overlap = (
                            len(intersection)
                            / len(union)
                        )

                        if (
                            entity_overlap
                            >= 0.80
                        ):
                            redundancy_penalty = max(
                                redundancy_penalty,
                                0.07,
                            )

            selection_score = (
                base_score
                + coverage_bonus
                + value_bonus
                - redundancy_penalty
            )

            if (
                selection_score
                > best_selection_score
            ):

                best_selection_score = (
                    selection_score
                )

                best_index = index

        if best_index is None:
            break

        chosen = remaining.pop(
            best_index
        )

        chosen = dict(
            chosen
        )

        chosen[
            "selection_score"
        ] = float(
            best_selection_score
        )

        chosen[
            "new_query_tokens"
        ] = sorted(
            path_coverage_tokens(
                path=chosen,
                query_tokens=query_tokens,
            )
            - covered_tokens
        )

        covered_tokens.update(
            path_coverage_tokens(
                path=chosen,
                query_tokens=query_tokens,
            )
        )

        selected.append(
            chosen
        )

    return selected


# ============================================================
# PATH RANKING
# ============================================================

def rank_paths(
    paths: list[dict[str, Any]],
    query_embedding: list[float],
    model,
    relevant_entity_names: set[str],
    query_tokens: set[str],
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Semantic ranking followed by query-coverage-aware
    evidence selection.
    """

    if not paths:
        return []

    # --------------------------------------------------------
    # Convert paths to text.
    # --------------------------------------------------------

    texts = [
        path_to_text(
            path
        )
        for path in paths
    ]

    valid_pairs = [
        (path, text)
        for path, text in zip(
            paths,
            texts,
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
    # Encode paths once.
    # --------------------------------------------------------

    embeddings = model.encode(
        valid_texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    semantic_scores = (
        embeddings @ query_embedding
    )

    ranked = []

    for path, text, semantic_score in zip(
        valid_paths,
        valid_texts,
        semantic_scores,
    ):

        score = score_path(
            path=path,
            semantic_score=float(
                semantic_score
            ),
            relevant_entity_names=(
                relevant_entity_names
            ),
            query_tokens=query_tokens,
        )

        item = dict(
            path
        )

        item["text"] = text

        item["semantic_score"] = float(
            semantic_score
        )

        item["score"] = float(
            score
        )

        item["query_coverage"] = (
            lexical_overlap_score(
                query_tokens=query_tokens,
                text=text,
            )
        )

        item["direct_fact"] = (
            len(
                path.get(
                    "relationships",
                    [],
                )
            ) == 1
        )

        ranked.append(
            item
        )

    # --------------------------------------------------------
    # Base ranking.
    # --------------------------------------------------------

    ranked.sort(
        key=lambda item: item[
            "score"
        ],
        reverse=True,
    )

    # Keep enough candidates for coverage-aware selection.
    candidate_pool_size = max(
        top_k * 8,
        40,
    )

    ranked = ranked[
        :candidate_pool_size
    ]

    # --------------------------------------------------------
    # Coverage-aware final selection.
    # --------------------------------------------------------

    return select_final_paths(
        candidates=ranked,
        query_tokens=query_tokens,
        model=model,
        top_k=top_k,
    )


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_paths(
    paths: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    seen = set()

    unique = []

    for path in paths:

        key = path_key(
            path
        )

        if key in seen:
            continue

        seen.add(key)

        unique.append(
            path
        )

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

        # Existing retrieval embedding model.
        self.model = get_rag_embedding_model()

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
    ) -> dict[str, Any]:

        query = clean_text(
            query
        )

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        # -----------------------------------------------------
        # Query tokens.
        # -----------------------------------------------------

        query_tokens = tokenize(
            query
        )

        # -----------------------------------------------------
        # Query embedding.
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
        # Hybrid seeds.
        # -----------------------------------------------------

        entity_results = hybrid_entity_search(
            client=self.client,
            query=query,
            query_embedding=query_embedding,
        )

        entity_names = [
            item["name"]
            for item in entity_results
            if item.get("name")
        ]

        relevant_entity_names = set(
            entity_names
        )

        if not entity_names:

            return {
                "query": query,
                "entities": [],
                "graph_paths": [],
            }

        # -----------------------------------------------------
        # 1-3 hop graph retrieval.
        # -----------------------------------------------------

        graph_paths = retrieve_graph_paths(
            client=self.client,
            entity_names=entity_names,
            max_paths=MAX_GRAPH_PATHS,
        )

        # -----------------------------------------------------
        # Deduplicate.
        # -----------------------------------------------------

        graph_paths = deduplicate_paths(
            graph_paths
        )

        # -----------------------------------------------------
        # Rank + query coverage.
        # -----------------------------------------------------

        ranked_paths = rank_paths(
            paths=graph_paths,
            query_embedding=query_embedding,
            model=self.model,
            relevant_entity_names=(
                relevant_entity_names
            ),
            query_tokens=query_tokens,
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

    def close(
        self,
    ):
        self.client.close()


# ============================================================
# OUTPUT
# ============================================================

def print_results(
    result: dict[str, Any],
):

    print()

    print("=" * 90)
    print(
        "GENERIC SEMANTIC GRAPH RETRIEVAL"
    )
    print("=" * 90)

    print()

    print("Question:")
    print("-" * 90)

    print(
        result["query"]
    )

    # --------------------------------------------------------
    # Seeds.
    # --------------------------------------------------------

    print()

    print("Graph Seed Entities:")
    print("-" * 90)

    for i, entity in enumerate(
        result["entities"],
        start=1,
    ):

        source = (
            "semantic"
            if entity.get(
                "from_vector",
                False,
            )
            else "lexical"
        )

        print(
            f"{i}. "
            f"{entity['name']} "
            f"[{source}] "
            f"(score={entity['score']:.4f}, "
            f"semantic={entity['semantic_score']:.4f}, "
            f"lexical={entity['lexical_score']:.4f})"
        )

    # --------------------------------------------------------
    # Paths.
    # --------------------------------------------------------

    print()

    print("Ranked Graph Paths:")
    print("-" * 90)

    paths = result[
        "graph_paths"
    ]

    if not paths:

        print(
            "No graph paths found."
        )

    else:

        for i, path in enumerate(
            paths,
            start=1,
        ):

            print(
                f"{i}. "
                f"score={path.get('score', 0.0):.4f} "
                f"selection={path.get('selection_score', path.get('score', 0.0)):.4f} "
                f"semantic={path.get('semantic_score', 0.0):.4f} "
                f"coverage={path.get('query_coverage', 0.0):.4f} "
                f"direct={path.get('direct_fact', False)}"
            )

            new_tokens = path.get(
                "new_query_tokens",
                [],
            )

            if new_tokens:
                print(
                    f"   new query concepts: "
                    f"{', '.join(new_tokens)}"
                )

            print(
                f"   {path['text']}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    retriever = GraphRetriever(
        entity_k=DEFAULT_ENTITY_K,
        graph_k=DEFAULT_GRAPH_K,
    )

    try:

        query = (
            "What was the minimum detectable "
            "PpIX concentration on pig skin "
            "under DC and pulsed illumination?"
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