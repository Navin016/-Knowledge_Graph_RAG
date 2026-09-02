"""
backend/entity_resolution.py

Entity Resolution V6
====================

General-purpose entity resolution for LLM-extracted
knowledge-graph triples.

Design goals
------------

1. High precision
2. Low false merges
3. Good recall
4. Lower cross-encoder workload
5. No dependency on Gemini for normal ER
6. Preserve the existing downstream JSON format

Pipeline:

    normalized triples
            |
            v
    entity feature construction
            |
            v
    exact normalization
            |
            v
    explicit alias detection
            |
            v
    BGE contextual embeddings
            |
            v
    Top-K semantic candidate blocking
            |
            v
    HARD identity blocking
            |
            v
    cheap multi-signal scoring
            |
       +----+----------------+
       |                     |
    safe match           plausible pair
       |                     |
       v                     v
    merge              cross-encoder
                             |
                       +-----+------+
                       |            |
                    strong      ambiguous
                       |            |
                       v            v
                     merge       reject
                                  / optional
                                 Gemini
                                      |
                                      v
                                 Union-Find
                                      |
                                      v
                              canonical entities
                                      |
                                      v
                                resolved triples
                                      |
                                      v
                                  deduplication
"""


# =========================================================
# Imports
# =========================================================

import json
import os
import re
import time
from pathlib import Path

import numpy as np
from rapidfuzz.fuzz import ratio

from backend.embedding_model import (
    get_embedding_model,
    get_cross_encoder,
)

from backend.entity_features import (
    build_entity_features,
)

from backend.extraction import Triple


# =========================================================
# Paths
# =========================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

PROCESSED_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)


# =========================================================
# Configuration
# =========================================================

# ---------------------------------------------------------
# Embeddings
# ---------------------------------------------------------

EMBEDDING_BATCH_SIZE = 32


# ---------------------------------------------------------
# Candidate blocking
# ---------------------------------------------------------

TOP_K = 6

# Minimum BGE similarity for a semantic candidate.
SEMANTIC_CANDIDATE_MIN = 0.72


# ---------------------------------------------------------
# Automatic identity merge
# ---------------------------------------------------------

# Very strict.
SEMANTIC_STRONG = 0.93
LEXICAL_STRONG = 92.0
TOKEN_STRONG = 0.75


# ---------------------------------------------------------
# Plausible candidate blocking
#
# A semantic candidate reaches the cross encoder only when
# at least one additional identity signal supports it.
# ---------------------------------------------------------

BLOCK_SEMANTIC = 0.80
BLOCK_LEXICAL = 72.0
BLOCK_TOKEN = 0.50
BLOCK_CONTEXT = 0.35


# ---------------------------------------------------------
# Cross encoder
# ---------------------------------------------------------

CROSS_ENCODER_BATCH_SIZE = 32

# BGE reranker output is treated as a model score,
# not a calibrated probability.
CROSS_ENCODER_ACCEPT = 0.88

# Below this we simply reject.
CROSS_ENCODER_REVIEW = 0.72


# ---------------------------------------------------------
# Optional Gemini verification
# ---------------------------------------------------------

# Disabled by default.
MAX_GEMINI_VERIFICATIONS = 2

GEMINI_CONFIDENCE_THRESHOLD = 0.90


# =========================================================
# Generic normalization
# =========================================================

def normalize_entity_name(
    value: str,
) -> str:
    """
    Normalize whitespace/punctuation without changing the
    actual semantic content of an entity mention.
    """

    if not value:
        return ""

    value = str(value).strip()

    value = value.strip(
        "\"'`.,;:"
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value


def comparison_form(
    value: str,
) -> str:
    """
    Normalized lower-case comparison representation.
    """

    return normalize_entity_name(
        value
    ).lower()


# =========================================================
# Tokens
# =========================================================

def get_tokens(
    value: str,
) -> set[str]:
    """
    Return normalized alphanumeric tokens.
    """

    return set(
        re.findall(
            r"[a-z0-9]+",
            comparison_form(value),
        )
    )


def token_jaccard(
    a: str,
    b: str,
) -> float:
    """
    Token-set Jaccard similarity.
    """

    a_tokens = get_tokens(a)
    b_tokens = get_tokens(b)

    if not a_tokens or not b_tokens:
        return 0.0

    union = (
        a_tokens
        |
        b_tokens
    )

    if not union:
        return 0.0

    return (
        len(
            a_tokens
            &
            b_tokens
        )
        /
        len(union)
    )


# =========================================================
# Acronym detection
# =========================================================

def build_acronym(
    value: str,
) -> str:
    """
    Build an acronym from a multi-word entity.

    Example:

        Retrieval Augmented Generation
        ->
        rag
    """

    words = re.findall(
        r"[A-Za-z]+",
        normalize_entity_name(value),
    )

    if len(words) < 2:
        return ""

    return "".join(
        word[0]
        for word in words
    ).lower()


def acronym_match(
    a: str,
    b: str,
) -> bool:
    """
    Detect acronym/expanded-name equivalence.
    """

    a_norm = comparison_form(a)
    b_norm = comparison_form(b)

    if not a_norm or not b_norm:
        return False

    a_acronym = build_acronym(a)
    b_acronym = build_acronym(b)

    return (
        bool(a_acronym)
        and a_acronym == b_norm
    ) or (
        bool(b_acronym)
        and b_acronym == a_norm
    )


# =========================================================
# Explicit aliases
# =========================================================

def extract_alias_pairs(
    texts: list[str],
) -> set[frozenset[str]]:
    """
    Detect explicit aliases from source text.

    Examples:

        Retrieval-Augmented Generation (RAG)

        Retrieval-Augmented Generation,
        abbreviated as RAG
    """

    aliases = set()

    parenthesis_pattern = re.compile(
        r"""
        ([A-Za-z][A-Za-z0-9 .+\-/]{3,120}?)
        \s*
        \(
        ([A-Z][A-Z0-9\-]{1,20})
        \)
        """,
        re.VERBOSE,
    )

    abbreviated_pattern = re.compile(
        r"""
        ([A-Za-z][A-Za-z0-9 .+\-/]{3,120}?)
        \s*,?\s*
        (?:commonly\s+)?
        abbreviated\s+as
        \s+
        ([A-Za-z][A-Za-z0-9.+\-/]{1,20})
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    for text in texts:

        if not text:
            continue

        # -------------------------------------------------
        # Long Name (ABC)
        # -------------------------------------------------

        for match in parenthesis_pattern.finditer(
            text
        ):

            long_name = normalize_entity_name(
                match.group(1)
            )

            short_name = normalize_entity_name(
                match.group(2)
            )

            if (
                build_acronym(long_name)
                ==
                short_name.lower()
            ):

                aliases.add(
                    frozenset(
                        {
                            comparison_form(
                                long_name
                            ),
                            comparison_form(
                                short_name
                            ),
                        }
                    )
                )

        # -------------------------------------------------
        # abbreviated as
        # -------------------------------------------------

        for match in abbreviated_pattern.finditer(
            text
        ):

            long_name = normalize_entity_name(
                match.group(1)
            )

            short_name = normalize_entity_name(
                match.group(2)
            )

            aliases.add(
                frozenset(
                    {
                        comparison_form(
                            long_name
                        ),
                        comparison_form(
                            short_name
                        ),
                    }
                )
            )

    return aliases


def aliases_match(
    a: str,
    b: str,
    aliases: set[frozenset[str]],
) -> bool:
    """
    Test explicit alias membership.
    """

    return frozenset(
        {
            comparison_form(a),
            comparison_form(b),
        }
    ) in aliases


# =========================================================
# Lexical similarity
# =========================================================

def lexical_similarity(
    a: str,
    b: str,
) -> float:
    """
    Character-level fuzzy similarity.
    """

    return float(
        ratio(
            comparison_form(a),
            comparison_form(b),
        )
    )


# =========================================================
# Context
# =========================================================

def relation_set(
    feature,
) -> set[str]:
    return {
        comparison_form(relation)
        for relation, _ in (
            feature.outgoing
            +
            feature.incoming
        )
    }


def neighbor_set(
    feature,
) -> set[str]:
    return {
        comparison_form(neighbor)
        for neighbor in feature.neighbors
    }


def type_set(
    feature,
) -> set[str]:
    return {
        comparison_form(value)
        for value in (
            feature.entity_type_candidates
        )
    }


def jaccard(
    a: set[str],
    b: set[str],
) -> float:

    if not a and not b:
        return 0.0

    union = a | b

    if not union:
        return 0.0

    return (
        len(a & b)
        /
        len(union)
    )


def context_similarity(
    feature_a,
    feature_b,
) -> float:
    """
    Compare local graph neighborhoods.
    """

    relation_score = jaccard(
        relation_set(feature_a),
        relation_set(feature_b),
    )

    neighbor_score = jaccard(
        neighbor_set(feature_a),
        neighbor_set(feature_b),
    )

    type_score = jaccard(
        type_set(feature_a),
        type_set(feature_b),
    )

    return (
        0.35 * relation_score
        +
        0.40 * neighbor_score
        +
        0.25 * type_score
    )


# =========================================================
# Generic / specificity protection
# =========================================================

GENERIC_TERMS = {
    "model",
    "system",
    "framework",
    "library",
    "tool",
    "method",
    "technology",
    "platform",
    "database",
    "language",
    "service",
    "application",
    "algorithm",
    "process",
    "software",
    "hardware",
    "device",
    "program",
    "module",
    "component",
    "technique",
    "approach",
}


def phrase_is_generic(
    value: str,
) -> bool:
    """
    Detect generic conceptual phrases.

    Examples:

        model
        software
        framework
        software program
        imaging system
        retrieval framework

    This is intentionally conservative.
    """

    tokens = get_tokens(
        value
    )

    if not tokens:
        return True

    # Direct generic singleton.
    if len(tokens) == 1:
        return next(
            iter(tokens)
        ) in GENERIC_TERMS

    # Entire phrase consists of generic terms.
    generic_ratio = (
        len(
            tokens & GENERIC_TERMS
        )
        /
        len(tokens)
    )

    return (
        generic_ratio >= 0.70
    )


def contains_generic_head(
    value: str,
) -> bool:
    """
    Detect phrases whose final token is a generic head.

    Example:

        software
        software program
        Python framework

    We do NOT automatically reject such entities; this is
    used only when comparing two otherwise similar mentions.
    """

    tokens = list(
        get_tokens(value)
    )

    if not tokens:
        return False

    return (
        tokens[-1]
        in GENERIC_TERMS
    )


# =========================================================
# Explicit parent/type protection
# =========================================================

def has_explicit_parent_child_relation(
    feature_a,
    feature_b,
) -> bool:
    """
    Prevent merging when the document explicitly says one
    entity is a type/parent of the other.
    """

    a_name = comparison_form(
        feature_a.name
    )

    b_name = comparison_form(
        feature_b.name
    )

    # A is_a B
    for relation, obj in feature_a.outgoing:

        relation_norm = comparison_form(
            relation
        )

        obj_norm = comparison_form(
            obj
        )

        if relation_norm in {
            "is_a",
            "type_of",
            "isa",
        }:

            if obj_norm == b_name:
                return True

    # B is_a A
    for relation, obj in feature_b.outgoing:

        relation_norm = comparison_form(
            relation
        )

        obj_norm = comparison_form(
            obj
        )

        if relation_norm in {
            "is_a",
            "type_of",
            "isa",
        }:

            if obj_norm == a_name:
                return True

    return False


# =========================================================
# Specificity conflict
# =========================================================

def has_specificity_conflict(
    a: str,
    b: str,
) -> bool:
    """
    Prevent generic/specific false merges.

    Examples:

        software
            vs
        software program

        framework
            vs
        Python framework

        library
            vs
        Python library

    Exact names are safe and return False.
    """

    a_norm = comparison_form(a)
    b_norm = comparison_form(b)

    if a_norm == b_norm:
        return False

    a_tokens = get_tokens(a)
    b_tokens = get_tokens(b)

    if not a_tokens or not b_tokens:
        return False

    # One phrase is a strict token subset.
    if a_tokens < b_tokens:

        # Shorter phrase is generic.
        if phrase_is_generic(a):
            return True

        # The shorter phrase is itself a generic head.
        if (
            len(a_tokens) == 1
            and contains_generic_head(a)
        ):
            return True

    if b_tokens < a_tokens:

        if phrase_is_generic(b):
            return True

        if (
            len(b_tokens) == 1
            and contains_generic_head(b)
        ):
            return True

    return False


# =========================================================
# Identity conflict
# =========================================================

def identity_conflict(
    entity_a: str,
    entity_b: str,
    feature_a,
    feature_b,
) -> tuple[bool, str]:
    """
    Determine whether there is evidence that two mentions
    should NOT be merged.
    """

    # -----------------------------------------------------
    # Exact normalized names are always safe.
    # -----------------------------------------------------

    if (
        comparison_form(entity_a)
        ==
        comparison_form(entity_b)
    ):
        return (
            False,
            "",
        )

    # -----------------------------------------------------
    # Explicit parent/type relation.
    # -----------------------------------------------------

    if has_explicit_parent_child_relation(
        feature_a,
        feature_b,
    ):

        return (
            True,
            "explicit_parent_child_relation",
        )

    # -----------------------------------------------------
    # Generic vs specific.
    # -----------------------------------------------------

    if has_specificity_conflict(
        entity_a,
        entity_b,
    ):

        return (
            True,
            "generic_specific_conflict",
        )

    return (
        False,
        "",
    )


# =========================================================
# Cheap identity compatibility
# =========================================================

def identity_compatibility(
    entity_a: str,
    entity_b: str,
    semantic: float,
    lexical: float,
    token: float,
    context: float,
) -> bool:
    """
    Determine whether a pair deserves expensive
    cross-encoder evaluation.

    This is the main V6 performance optimization.

    A candidate must have semantic similarity AND at least
    one additional identity signal.
    """

    # High semantic + high lexical.
    if (
        semantic >= BLOCK_SEMANTIC
        and lexical >= BLOCK_LEXICAL
    ):
        return True

    # Strong token overlap.
    if (
        semantic >= 0.78
        and token >= BLOCK_TOKEN
    ):
        return True

    # Strong shared graph context.
    if (
        semantic >= 0.76
        and context >= BLOCK_CONTEXT
    ):
        return True

    # Very strong semantic similarity can still pass even
    # when lexical overlap is weak.
    if semantic >= 0.88:
        return True

    return False


# =========================================================
# Identity score
# =========================================================

def identity_score(
    semantic: float,
    lexical: float,
    token: float,
    context: float,
) -> float:
    """
    Combine cheap signals.

    Semantic similarity remains important, but identity
    cannot be established from semantic similarity alone.
    """

    lexical_normalized = (
        lexical / 100.0
    )

    return (
        0.45 * semantic
        +
        0.25 * lexical_normalized
        +
        0.15 * token
        +
        0.15 * context
    )


# =========================================================
# Candidate generation
# =========================================================

def generate_candidates(
    entities: list[str],
    embeddings: np.ndarray,
) -> tuple[
    dict[str, list[str]],
    np.ndarray,
]:
    """
    Generate Top-K semantic candidates using a single
    matrix multiplication.
    """

    similarity_matrix = (
        embeddings
        @
        embeddings.T
    )

    n = len(
        entities
    )

    candidates = {
        entity: []
        for entity in entities
    }

    if n <= 1:

        return (
            candidates,
            similarity_matrix,
        )

    k = min(
        TOP_K,
        n - 1,
    )

    for i, entity in enumerate(
        entities
    ):

        scores = similarity_matrix[
            i
        ]

        # Only inspect top-k indices.
        indices = np.argpartition(
            scores,
            -k,
        )[-k:]

        indices = indices[
            np.argsort(
                scores[indices]
            )[::-1]
        ]

        candidates[
            entity
        ] = [
            entities[j]
            for j in indices
            if (
                j != i
                and scores[j]
                >= SEMANTIC_CANDIDATE_MIN
            )
        ]

    return (
        candidates,
        similarity_matrix,
    )


# =========================================================
# Union-Find
# =========================================================

class UnionFind:

    def __init__(
        self,
        items,
    ):

        self.parent = {
            item: item
            for item in items
        }

        self.rank = {
            item: 0
            for item in items
        }

    def find(
        self,
        item,
    ):

        if (
            self.parent[item]
            != item
        ):

            self.parent[item] = (
                self.find(
                    self.parent[item]
                )
            )

        return self.parent[item]

    def union(
        self,
        a,
        b,
    ):

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if (
            self.rank[root_a]
            <
            self.rank[root_b]
        ):

            root_a, root_b = (
                root_b,
                root_a,
            )

        self.parent[
            root_b
        ] = root_a

        if (
            self.rank[root_a]
            ==
            self.rank[root_b]
        ):

            self.rank[root_a] += 1


# =========================================================
# Optional Gemini verifier
# =========================================================

def verify_with_gemini(
    entity_a: str,
    context_a: str,
    entity_b: str,
    context_b: str,
) -> tuple[
    bool,
    float,
    str,
]:
    """
    Strict optional final verifier.

    This should only be used for very small numbers of
    difficult cases.
    """

    from dotenv import load_dotenv
    from google import genai
    from pydantic import BaseModel, Field

    load_dotenv(
        BASE_DIR / ".env"
    )

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        return (
            False,
            0.0,
            "Gemini API key unavailable.",
        )

    class VerificationResult(
        BaseModel
    ):

        is_same_entity: bool = Field(
            description=(
                "True only when both mentions "
                "refer to exactly the same "
                "entity or concept."
            )
        )

        confidence: float = Field(
            ge=0.0,
            le=1.0,
        )

        reason: str

    client = genai.Client(
        api_key=api_key
    )

    prompt = f"""
You are a strict entity-resolution verifier.

Determine whether Entity A and Entity B refer to the
EXACT SAME entity or concept.

Do NOT merge entities merely because:

- they are related
- they have the same type
- one is a type/category of the other
- one is a parent/child of the other
- one is a component of the other
- one is a variant of the other
- one uses the other
- one is more specific than the other
- their names overlap
- they are semantically similar

Use ONLY the supplied document context.

ENTITY A:
{entity_a}

CONTEXT A:
{context_a}

ENTITY B:
{entity_b}

CONTEXT B:
{context_b}

Return true only if identity equivalence is strongly
supported by the document.
"""

    try:

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": VerificationResult,
            },
        )

        result = (
            VerificationResult
            .model_validate_json(
                response.text
            )
        )

        return (
            result.is_same_entity,
            result.confidence,
            result.reason,
        )

    except Exception as exc:

        print(
            f"    Gemini verification failed: "
            f"{exc}"
        )

        return (
            False,
            0.0,
            "Gemini verification failed.",
        )


# =========================================================
# Resolver
# =========================================================

def resolve_entities(
    triples: list[Triple],
    source_texts: list[str] | None = None,
    use_gemini_verifier: bool = False,
):
    """
    Resolve entity mentions across one document.

    V6 prioritizes:

        precision
        +
        efficient blocking
        +
        contextual evidence

    Gemini is disabled by default.
    """

    if not triples:

        return (
            [],
            [],
            {},
            {},
        )

    # =====================================================
    # 1. Build features
    # =====================================================

    print(
        "\nBuilding entity features..."
    )

    features = (
        build_entity_features(
            triples
        )
    )

    entities = list(
        features.keys()
    )

    print(
        f"Unique entity mentions: "
        f"{len(entities)}"
    )

    entity_index = {
        entity: index
        for index, entity in enumerate(
            entities
        )
    }

    # =====================================================
    # 2. Explicit aliases
    # =====================================================

    aliases = set()

    if source_texts:

        aliases = (
            extract_alias_pairs(
                source_texts
            )
        )

    print(
        f"Explicit alias pairs found: "
        f"{len(aliases)}"
    )

    # =====================================================
    # 3. Normalized names
    # =====================================================

    normalized_names = {
        entity: comparison_form(
            entity
        )
        for entity in entities
    }

    # =====================================================
    # 4. Embeddings
    # =====================================================

    print(
        "\nGenerating entity embeddings..."
    )

    embedding_model = (
        get_embedding_model()
    )

    representations = [
        feature.build_representation()
        for feature in (
            features[entity]
            for entity in entities
        )
    ]

    embeddings = (
        embedding_model.encode(
            representations,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    # =====================================================
    # 5. Semantic candidates
    # =====================================================

    print(
        "\nGenerating semantic candidates..."
    )

    (
        candidates,
        similarity_matrix,
    ) = generate_candidates(
        entities,
        embeddings,
    )

    # =====================================================
    # 6. Deduplicate unordered pairs
    # =====================================================

    candidate_pairs = {}

    for entity in entities:

        i = entity_index[
            entity
        ]

        for candidate in candidates[
            entity
        ]:

            j = entity_index[
                candidate
            ]

            if i == j:
                continue

            key = (
                min(i, j),
                max(i, j),
            )

            if key in candidate_pairs:
                continue

            candidate_pairs[
                key
            ] = (
                entity,
                candidate,
            )

    print(
        f"Semantic candidate pairs: "
        f"{len(candidate_pairs)}"
    )

    # =====================================================
    # 7. Union-Find
    # =====================================================

    uf = UnionFind(
        entities
    )

    decisions = []

    borderline_pairs = []

    blocking_rejections = 0

    # =====================================================
    # 8. Cheap identity filtering
    # =====================================================

    for (
        pair_key,
        pair,
    ) in candidate_pairs.items():

        entity_a, entity_b = pair

        i, j = pair_key

        semantic = float(
            similarity_matrix[
                i,
                j,
            ]
        )

        # -------------------------------------------------
        # Exact normalized match
        # -------------------------------------------------

        if (
            normalized_names[
                entity_a
            ]
            ==
            normalized_names[
                entity_b
            ]
        ):

            uf.union(
                entity_a,
                entity_b,
            )

            decisions.append(
                {
                    "entity": entity_a,
                    "candidate": entity_b,
                    "method": "exact",
                    "semantic": 1.0,
                    "lexical": 100.0,
                    "token": 1.0,
                    "context": 1.0,
                    "merge": True,
                }
            )

            continue

        # -------------------------------------------------
        # Explicit alias
        # -------------------------------------------------

        if aliases_match(
            entity_a,
            entity_b,
            aliases,
        ):

            uf.union(
                entity_a,
                entity_b,
            )

            decisions.append(
                {
                    "entity": entity_a,
                    "candidate": entity_b,
                    "method": "explicit_alias",
                    "semantic": round(
                        semantic,
                        4,
                    ),
                    "lexical": round(
                        lexical_similarity(
                            entity_a,
                            entity_b,
                        ),
                        2,
                    ),
                    "token": round(
                        token_jaccard(
                            entity_a,
                            entity_b,
                        ),
                        4,
                    ),
                    "context": 1.0,
                    "merge": True,
                }
            )

            continue

        # -------------------------------------------------
        # Cheap lexical signals
        # -------------------------------------------------

        lexical = lexical_similarity(
            entity_a,
            entity_b,
        )

        token = token_jaccard(
            entity_a,
            entity_b,
        )

        context = context_similarity(
            features[entity_a],
            features[entity_b],
        )

        # -------------------------------------------------
        # Identity conflict
        # -------------------------------------------------

        conflict, reason = (
            identity_conflict(
                entity_a,
                entity_b,
                features[entity_a],
                features[entity_b],
            )
        )

        if conflict:

            decisions.append(
                {
                    "entity": entity_a,
                    "candidate": entity_b,
                    "method": "identity_conflict",
                    "reason": reason,
                    "semantic": round(
                        semantic,
                        4,
                    ),
                    "lexical": round(
                        lexical,
                        2,
                    ),
                    "token": round(
                        token,
                        4,
                    ),
                    "context": round(
                        context,
                        4,
                    ),
                    "merge": False,
                }
            )

            continue

        # -------------------------------------------------
        # Acronym match
        # -------------------------------------------------

        if (
            acronym_match(
                entity_a,
                entity_b,
            )
            and semantic >= 0.70
        ):

            uf.union(
                entity_a,
                entity_b,
            )

            decisions.append(
                {
                    "entity": entity_a,
                    "candidate": entity_b,
                    "method": "acronym",
                    "semantic": round(
                        semantic,
                        4,
                    ),
                    "lexical": round(
                        lexical,
                        2,
                    ),
                    "token": round(
                        token,
                        4,
                    ),
                    "context": round(
                        context,
                        4,
                    ),
                    "merge": True,
                }
            )

            continue

        # -------------------------------------------------
        # Very strong identity
        # -------------------------------------------------

        if (
            semantic >= SEMANTIC_STRONG
            and lexical >= LEXICAL_STRONG
            and token >= TOKEN_STRONG
        ):

            uf.union(
                entity_a,
                entity_b,
            )

            decisions.append(
                {
                    "entity": entity_a,
                    "candidate": entity_b,
                    "method": "strong_identity",
                    "semantic": round(
                        semantic,
                        4,
                    ),
                    "lexical": round(
                        lexical,
                        2,
                    ),
                    "token": round(
                        token,
                        4,
                    ),
                    "context": round(
                        context,
                        4,
                    ),
                    "merge": True,
                }
            )

            continue

        # -------------------------------------------------
        # V6 hard candidate blocking
        #
        # Only plausible identity candidates proceed to
        # cross-encoder.
        # -------------------------------------------------

        if not identity_compatibility(
            entity_a,
            entity_b,
            semantic,
            lexical,
            token,
            context,
        ):

            blocking_rejections += 1

            decisions.append(
                {
                    "entity": entity_a,
                    "candidate": entity_b,
                    "method": "candidate_blocked",
                    "semantic": round(
                        semantic,
                        4,
                    ),
                    "lexical": round(
                        lexical,
                        2,
                    ),
                    "token": round(
                        token,
                        4,
                    ),
                    "context": round(
                        context,
                        4,
                    ),
                    "merge": False,
                }
            )

            continue

        # -------------------------------------------------
        # Plausible candidate
        # -------------------------------------------------

        borderline_pairs.append(
            {
                "entity": entity_a,
                "candidate": entity_b,
                "semantic": semantic,
                "lexical": lexical,
                "token": token,
                "context": context,
            }
        )

    # =====================================================
    # 9. Cross encoder
    # =====================================================

    print(
        f"\nCross-encoder candidates: "
        f"{len(borderline_pairs)}"
    )

    gemini_candidates = []

    if borderline_pairs:

        print(
            "\nRunning cross-encoder..."
        )

        cross_encoder = (
            get_cross_encoder()
        )

        # -------------------------------------------------
        # Build cross encoder inputs.
        # -------------------------------------------------

        cross_inputs = []

        for pair in borderline_pairs:

            context_a = (
                features[
                    pair["entity"]
                ].build_representation()
            )

            context_b = (
                features[
                    pair["candidate"]
                ].build_representation()
            )

            cross_inputs.append(
                [
                    context_a,
                    context_b,
                ]
            )

        # -------------------------------------------------
        # Batch prediction
        # -------------------------------------------------

        cross_scores = (
            cross_encoder.predict(
                cross_inputs,
                batch_size=CROSS_ENCODER_BATCH_SIZE,
                show_progress_bar=True,
            )
        )

        # -------------------------------------------------
        # Process scores
        # -------------------------------------------------

        for pair, cross_score in zip(
            borderline_pairs,
            cross_scores,
        ):

            cross_score = float(
                cross_score
            )

            entity_a = pair[
                "entity"
            ]

            entity_b = pair[
                "candidate"
            ]

            semantic = pair[
                "semantic"
            ]

            lexical = pair[
                "lexical"
            ]

            token = pair[
                "token"
            ]

            context = pair[
                "context"
            ]

            combined = identity_score(
                semantic,
                lexical,
                token,
                context,
            )

            # -------------------------------------------------
            # Strong cross encoder identity
            # -------------------------------------------------

            if (
                cross_score
                >= CROSS_ENCODER_ACCEPT
                and semantic >= 0.82
                and (
                    lexical >= 72
                    or token >= 0.50
                    or context >= 0.45
                )
                and combined >= 0.72
            ):

                uf.union(
                    entity_a,
                    entity_b,
                )

                decisions.append(
                    {
                        "entity": entity_a,
                        "candidate": entity_b,
                        "method": "cross_encoder_identity",
                        "semantic": round(
                            semantic,
                            4,
                        ),
                        "lexical": round(
                            lexical,
                            2,
                        ),
                        "token": round(
                            token,
                            4,
                        ),
                        "context": round(
                            context,
                            4,
                        ),
                        "combined": round(
                            combined,
                            4,
                        ),
                        "cross_encoder": round(
                            cross_score,
                            4,
                        ),
                        "merge": True,
                    }
                )

                continue

            # -------------------------------------------------
            # Optional Gemini candidate.
            #
            # ONLY extremely difficult cases.
            # -------------------------------------------------

            if (
                use_gemini_verifier
                and cross_score
                >= CROSS_ENCODER_REVIEW
                and semantic >= 0.80
                and (
                    lexical >= 55
                    or token >= 0.35
                    or context >= 0.35
                )
            ):

                gemini_candidates.append(
                    {
                        "pair": pair,
                        "cross_encoder": cross_score,
                        "combined": combined,
                    }
                )

                continue

            # -------------------------------------------------
            # Cross encoder rejection
            # -------------------------------------------------

            decisions.append(
                {
                    "entity": entity_a,
                    "candidate": entity_b,
                    "method": "cross_encoder_rejected",
                    "semantic": round(
                        semantic,
                        4,
                    ),
                    "lexical": round(
                        lexical,
                        2,
                    ),
                    "token": round(
                        token,
                        4,
                    ),
                    "context": round(
                        context,
                        4,
                    ),
                    "combined": round(
                        combined,
                        4,
                    ),
                    "cross_encoder": round(
                        cross_score,
                        4,
                    ),
                    "merge": False,
                }
            )

    # =====================================================
    # 10. Optional Gemini verification
    # =====================================================

    gemini_calls = 0

    if (
        use_gemini_verifier
        and gemini_candidates
    ):

        gemini_candidates.sort(
            key=lambda item: (
                item["combined"],
                item["cross_encoder"],
                item["pair"]["semantic"],
            ),
            reverse=True,
        )

        selected = gemini_candidates[
            :MAX_GEMINI_VERIFICATIONS
        ]

        print(
            "\nGemini verification candidates: "
            f"{len(gemini_candidates)}"
        )

        for item in selected:

            pair = item[
                "pair"
            ]

            entity_a = pair[
                "entity"
            ]

            entity_b = pair[
                "candidate"
            ]

            print(
                "\n    Gemini verification:"
            )

            print(
                f"    {entity_a} <-> {entity_b}"
            )

            context_a = (
                features[
                    entity_a
                ].build_representation()
            )

            context_b = (
                features[
                    entity_b
                ].build_representation()
            )

            (
                same_entity,
                confidence,
                reason,
            ) = verify_with_gemini(
                entity_a,
                context_a,
                entity_b,
                context_b,
            )

            gemini_calls += 1

            if (
                same_entity
                and confidence
                >= GEMINI_CONFIDENCE_THRESHOLD
            ):

                uf.union(
                    entity_a,
                    entity_b,
                )

                method = (
                    "gemini_verified"
                )

                merge = True

            else:

                method = (
                    "gemini_rejected"
                )

                merge = False

            decisions.append(
                {
                    "entity": entity_a,
                    "candidate": entity_b,
                    "method": method,
                    "semantic": round(
                        pair["semantic"],
                        4,
                    ),
                    "lexical": round(
                        pair["lexical"],
                        2,
                    ),
                    "token": round(
                        pair["token"],
                        4,
                    ),
                    "context": round(
                        pair["context"],
                        4,
                    ),
                    "combined": round(
                        item["combined"],
                        4,
                    ),
                    "cross_encoder": round(
                        item["cross_encoder"],
                        4,
                    ),
                    "gemini_confidence": round(
                        confidence,
                        4,
                    ),
                    "reason": reason,
                    "merge": merge,
                }
            )

    # =====================================================
    # 11. Build clusters
    # =====================================================

    clusters = {}

    for entity in entities:

        root = uf.find(
            entity
        )

        clusters.setdefault(
            root,
            [],
        ).append(
            entity
        )

    # =====================================================
    # 12. Canonical entity selection
    # =====================================================

    entity_to_canonical = {}

    for members in clusters.values():

        def canonical_score(
            name: str,
        ):

            feature = features[
                name
            ]

            normalized = (
                comparison_form(
                    name
                )
            )

            score = 0.0

            # Mention frequency.
            score += (
                feature.mentions
                * 10
            )

            # Prefer informative names.
            score += min(
                len(normalized),
                80,
            )

            # Avoid tiny labels.
            if len(normalized) <= 2:

                score -= 20

            # Prefer non-generic names when possible.
            if phrase_is_generic(
                name
            ):

                score -= 15

            return score

        canonical = max(
            members,
            key=canonical_score,
        )

        for member in members:

            entity_to_canonical[
                member
            ] = canonical

    # =====================================================
    # 13. Resolve triples
    # =====================================================

    resolved = []

    for triple in triples:

        subject = (
            entity_to_canonical.get(
                triple.subject,
                triple.subject,
            )
        )

        object_name = (
            entity_to_canonical.get(
                triple.object,
                triple.object,
            )
        )

        resolved.append(
            Triple(
                subject=subject,
                relation=triple.relation,
                object=object_name,
            )
        )

    # =====================================================
    # 14. Deduplicate triples
    # =====================================================

    unique_triples = []

    seen = set()

    duplicate_count = 0

    for triple in resolved:

        key = (
            comparison_form(
                triple.subject
            ),
            comparison_form(
                triple.relation
            ),
            comparison_form(
                triple.object
            ),
        )

        if key in seen:

            duplicate_count += 1

            continue

        seen.add(
            key
        )

        unique_triples.append(
            triple
        )

    print(
        f"\nDuplicate triples removed: "
        f"{duplicate_count}"
    )

    print(
        f"Canonical entities: "
        f"{len(clusters)}"
    )

    return (
        unique_triples,
        decisions,
        clusters,
        entity_to_canonical,
        {
            "candidate_pairs": len(
                candidate_pairs
            ),
            "blocking_rejections": (
                blocking_rejections
            ),
            "cross_encoder_candidates": (
                len(borderline_pairs)
            ),
            "gemini_candidates": (
                len(gemini_candidates)
            ),
            "gemini_calls": gemini_calls,
        },
    )


# =========================================================
# Load document
# =========================================================

def load_document(
    path: str | Path,
):
    """
    Load normalized document JSON.

    Supports:

        chunks[].triples

    and:

        triples[]
    """

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    triples = []

    texts = []

    # -----------------------------------------------------
    # Chunk format
    # -----------------------------------------------------

    for chunk in data.get(
        "chunks",
        [],
    ):

        text = chunk.get(
            "text",
            "",
        )

        if text:

            texts.append(
                text
            )

        for triple_data in chunk.get(
            "triples",
            [],
        ):

            triples.append(
                Triple.model_validate(
                    triple_data
                )
            )

    # -----------------------------------------------------
    # Flat format
    # -----------------------------------------------------

    if not triples:

        for triple_data in data.get(
            "triples",
            [],
        ):

            triples.append(
                Triple.model_validate(
                    triple_data
                )
            )

    return (
        data,
        triples,
        texts,
    )


# =========================================================
# Save result
# =========================================================

def save_result(
    original_data,
    triples,
    decisions,
    clusters,
    entity_to_canonical,
    output_path,
):
    """
    Save resolved document.

    The main downstream field remains:

        triples

    so Neo4j ingestion remains compatible.
    """

    output = {

        "source_file": (
            original_data.get(
                "source_file"
            )
        ),

        "total_chunks": (
            original_data.get(
                "total_chunks"
            )
        ),

        "input_triples": (
            sum(
                len(
                    chunk.get(
                        "triples",
                        [],
                    )
                )
                for chunk in original_data.get(
                    "chunks",
                    [],
                )
            )
        ),

        "resolved_unique_triples": len(
            triples
        ),

        "triples": [
            triple.model_dump()
            for triple in triples
        ],

        "entities": [],

        "entity_clusters": [],

        "resolution_decisions": decisions,
    }

    # -----------------------------------------------------
    # Entity clusters
    # -----------------------------------------------------

    for members in clusters.values():

        canonical = (
            entity_to_canonical[
                members[0]
            ]
        )

        aliases = [
            member
            for member in members
            if member != canonical
        ]

        output[
            "entities"
        ].append(
            {
                "canonical_name": canonical,
                "aliases": aliases,
            }
        )

        output[
            "entity_clusters"
        ].append(
            {
                "canonical": canonical,
                "members": members,
            }
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )


# =========================================================
# Print decisions
# =========================================================

def print_decisions(
    decisions,
):
    """
    Print only actual merges.

    This keeps console output manageable.
    """

    print(
        "\n"
        + "=" * 125
    )

    print(
        "ENTITY RESOLUTION V6 DECISIONS"
    )

    print(
        "=" * 125
    )

    print(
        f"{'ENTITY':30} "
        f"{'CANDIDATE':30} "
        f"{'METHOD':28} "
        f"{'SEM':7} "
        f"{'LEX':7} "
        f"{'CTX':7}"
    )

    print(
        "-" * 125
    )

    for decision in decisions:

        if not decision.get(
            "merge",
            False,
        ):
            continue

        print(
            f"{decision['entity'][:30]:30} "
            f"{decision['candidate'][:30]:30} "
            f"{decision['method'][:28]:28} "
            f"{str(decision.get('semantic', '-')):7} "
            f"{str(decision.get('lexical', '-')):7} "
            f"{str(decision.get('context', '-')):7}"
        )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    import sys

    start_time = time.perf_counter()

    # -----------------------------------------------------
    # Input
    # -----------------------------------------------------

    if len(sys.argv) > 1:

        input_path = Path(
            sys.argv[1]
        )

    else:

        input_path = (
            PROCESSED_DIR
            / "sample_normalized.json"
        )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    output_path = (
        input_path.parent
        /
        f"{input_path.stem}_resolved.json"
    )

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "ENTITY RESOLUTION V6"
    )

    print(
        "=" * 100
    )

    print(
        f"\nInput:"
    )

    print(
        input_path
    )

    # =====================================================
    # Load
    # =====================================================

    (
        original_data,
        triples,
        texts,
    ) = load_document(
        input_path
    )

    print(
        f"\nNormalized triples loaded: "
        f"{len(triples)}"
    )

    print(
        f"Source chunks loaded: "
        f"{len(texts)}"
    )

    # =====================================================
    # Resolve
    # =====================================================

    (
        resolved,
        decisions,
        clusters,
        entity_to_canonical,
        stats,
    ) = resolve_entities(
        triples,
        source_texts=texts,
        use_gemini_verifier=False,
    )

    # =====================================================
    # Print merge decisions
    # =====================================================

    print_decisions(
        decisions
    )

    # =====================================================
    # Statistics
    # =====================================================

    method_counts = {}

    for decision in decisions:

        method = decision[
            "method"
        ]

        method_counts[
            method
        ] = (
            method_counts.get(
                method,
                0,
            )
            + 1
        )

    elapsed = (
        time.perf_counter()
        -
        start_time
    )

    # =====================================================
    # Summary
    # =====================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "ENTITY RESOLUTION V6 SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"Input triples:             "
        f"{len(triples)}"
    )

    print(
        f"Output unique triples:     "
        f"{len(resolved)}"
    )

    print(
        f"Entity mentions:           "
        f"{len(entity_to_canonical)}"
    )

    print(
        f"Canonical entities:        "
        f"{len(clusters)}"
    )

    print(
        f"Semantic candidate pairs:  "
        f"{stats['candidate_pairs']}"
    )

    print(
        f"Blocked before reranker:   "
        f"{stats['blocking_rejections']}"
    )

    print(
        f"Cross-encoder candidates:  "
        f"{stats['cross_encoder_candidates']}"
    )

    print(
        f"Gemini candidates:         "
        f"{stats['gemini_candidates']}"
    )

    print(
        f"Gemini calls:              "
        f"{stats['gemini_calls']}"
    )

    print(
        f"Exact merges:              "
        f"{method_counts.get('exact', 0)}"
    )

    print(
        f"Explicit alias merges:     "
        f"{method_counts.get('explicit_alias', 0)}"
    )

    print(
        f"Acronym merges:            "
        f"{method_counts.get('acronym', 0)}"
    )

    print(
        f"Strong identity merges:    "
        f"{method_counts.get('strong_identity', 0)}"
    )

    print(
        f"Cross-encoder merges:      "
        f"{method_counts.get('cross_encoder_identity', 0)}"
    )

    print(
        f"Gemini verified merges:    "
        f"{method_counts.get('gemini_verified', 0)}"
    )

    print(
        f"Identity conflicts:        "
        f"{method_counts.get('identity_conflict', 0)}"
    )

    print(
        f"Candidate blocked:         "
        f"{method_counts.get('candidate_blocked', 0)}"
    )

    print(
        f"Cross-encoder rejected:    "
        f"{method_counts.get('cross_encoder_rejected', 0)}"
    )

    print(
        f"Duplicate triples removed: "
        f"{len(triples) - len(resolved)}"
    )

    print(
        f"Processing time:           "
        f"{elapsed:.2f} seconds"
    )

    # =====================================================
    # Save
    # =====================================================

    print(
        "\nSaving..."
    )

    save_result(
        original_data,
        resolved,
        decisions,
        clusters,
        entity_to_canonical,
        output_path,
    )

    print(
        f"\nSaved to:"
    )

    print(
        output_path
    )

    print(
        "\nEntity Resolution V6 complete."
    )