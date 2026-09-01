"""
backend/relation_normalizer.py

Phase 4A:
Relation normalization for the Knowledge Graph pipeline.

Pipeline:

    Raw Gemini triples
            |
            v
    Clean relation text
            |
            v
    Exact mapping?
       /           \
     YES            NO
      |              |
      v              v
  Canonical      MiniLM semantic
  relation       similarity
                      |
                 ┌────┴────┐
                 |         |
               MATCH     NEW RELATION
                 |         |
                 v         v
             Existing    Quality
             canonical    check
                            |
                       ┌────┴────┐
                       |         |
                     valid     invalid
                       |         |
                       v         v
                     keep     related_to

IMPORTANT:

This module normalizes RELATIONS.

It does NOT resolve entities.

Entity resolution will be handled by:

    backend/entity_resolution.py


The relation normalizer operates on the complete set of
triples from one document, not independently per chunk.
"""


import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from backend.embedding_model import get_embedding_model
from backend.extraction import Triple


# =========================================================
# Configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)


# =========================================================
# Thresholds
# =========================================================

# If similarity is this high, consider the relation
# semantically equivalent to an existing canonical relation.

MATCH_THRESHOLD = 0.82


# =========================================================
# Exact mapping
# =========================================================
#
# These mappings are high-confidence transformations.
#
# Example:
#
#     purchased
#     bought
#
#         ↓
#
#     acquired
#
# We deliberately avoid aggressive semantic assumptions.
# =========================================================

EXACT_MAPPING = {

    # -----------------------------------------------------
    # Classification
    # -----------------------------------------------------

    "is": "is_a",
    "is_a": "is_a",
    "is_a_type_of": "is_a",
    "type_of": "is_a",
    "kind_of": "is_a",
    "instance_of": "is_a",
    "classified_as": "is_a",

    # -----------------------------------------------------
    # Has
    # -----------------------------------------------------

    "has": "has",
    "have": "has",
    "having": "has",
    "possesses": "has",

    # -----------------------------------------------------
    # Part / containment
    # -----------------------------------------------------

    "part_of": "part_of",
    "is_part_of": "part_of",
    "belongs_to": "part_of",
    "component_of": "part_of",

    "contains": "contains",
    "includes": "contains",
    "consists_of": "contains",
    "comprises": "contains",

    # -----------------------------------------------------
    # Uses
    # -----------------------------------------------------

    "uses": "uses",
    "use": "uses",
    "using": "uses",
    "utilizes": "uses",
    "utilises": "uses",
    "employs": "uses",

    # -----------------------------------------------------
    # Purpose
    # -----------------------------------------------------

    "used_for": "used_for",
    "used_to": "used_for",
    "serves_as": "used_for",
    "serves_for": "used_for",

    # -----------------------------------------------------
    # Provides
    # -----------------------------------------------------

    "provides": "provides",
    "provide": "provides",
    "offers": "provides",
    "supplies": "provides",
    "delivers": "provides",

    # -----------------------------------------------------
    # Supports
    # -----------------------------------------------------

    "supports": "supports",
    "support": "supports",
    "helps": "supports",
    "assists": "supports",
    "enables": "supports",

    # -----------------------------------------------------
    # Dependencies
    # -----------------------------------------------------

    "depends_on": "depends_on",
    "depends_upon": "depends_on",
    "relies_on": "depends_on",
    "reliant_on": "depends_on",
    "requires": "depends_on",

    # -----------------------------------------------------
    # Integration
    # -----------------------------------------------------

    "integrates_with": "integrates_with",
    "integrates": "integrates_with",

    # -----------------------------------------------------
    # Development
    # -----------------------------------------------------

    "developed_by": "developed_by",
    "built_by": "developed_by",
    "engineered_by": "developed_by",

    # -----------------------------------------------------
    # Creation
    # -----------------------------------------------------

    "created_by": "created_by",
    "designed_by": "created_by",
    "authored_by": "created_by",
    "written_by": "created_by",

    # -----------------------------------------------------
    # Ownership
    # -----------------------------------------------------

    "owned_by": "owned_by",
    "property_of": "owned_by",

    # -----------------------------------------------------
    # Employment
    # -----------------------------------------------------

    "works_for": "works_for",
    "works_at": "works_for",
    "employed_by": "works_for",
    "employee_of": "works_for",

    # -----------------------------------------------------
    # Location
    # -----------------------------------------------------

    "located_in": "located_in",
    "located_at": "located_in",
    "situated_in": "located_in",
    "based_in": "located_in",
    "found_in": "located_in",

    # -----------------------------------------------------
    # Founding
    # -----------------------------------------------------

    "founded_by": "founded_by",
    "established_by": "founded_by",
    "started_by": "founded_by",

    "founded_in": "founded_in",
    "established_in": "founded_in",
    "started_in": "founded_in",

    # -----------------------------------------------------
    # Acquisition
    # -----------------------------------------------------

    "acquired": "acquired",
    "acquire": "acquired",
    "acquires": "acquired",
    "bought": "acquired",
    "buy": "acquired",
    "purchased": "acquired",
    "purchase": "acquired",

    # -----------------------------------------------------
    # Causation
    # -----------------------------------------------------

    "causes": "causes",
    "cause": "causes",
    "leads_to": "causes",
    "results_in": "causes",

    # -----------------------------------------------------
    # Production
    # -----------------------------------------------------
    #
    # IMPORTANT:
    #
    # produces != causes
    #
    # We preserve "produces" as its own relationship.

    "produces": "produces",

    # -----------------------------------------------------
    # Prevention
    # -----------------------------------------------------

    "prevents": "prevents",
    "prevent": "prevents",
    "avoids": "prevents",
    "protects_against": "prevents",

    # -----------------------------------------------------
    # Treatment
    # -----------------------------------------------------

    "treats": "treats",
    "treat": "treats",
    "used_to_treat": "treats",
    "helps_treat": "treats",

    # -----------------------------------------------------
    # Manufacturing
    # -----------------------------------------------------

    "manufactures": "manufactures",
    "manufacture": "manufactures",
    "makes": "manufactures",

    # -----------------------------------------------------
    # Storage
    # -----------------------------------------------------

    "stores": "stores",

    # -----------------------------------------------------
    # Retrieval
    # -----------------------------------------------------

    "retrieves_from": "retrieves_from",

    # -----------------------------------------------------
    # Rendering / visualization
    # -----------------------------------------------------

    "renders": "renders",
    "displays": "displays",

    # -----------------------------------------------------
    # Extraction
    # -----------------------------------------------------

    "extracts": "extracts",

    # -----------------------------------------------------
    # Preservation
    # -----------------------------------------------------

    "preserves": "preserves",

    # -----------------------------------------------------
    # Extension
    # -----------------------------------------------------

    "extends": "extends",

    # -----------------------------------------------------
    # Combination
    # -----------------------------------------------------

    "combine": "combines",
    "combines": "combines",
    "combined_with": "combines",

    # -----------------------------------------------------
    # Association
    # -----------------------------------------------------

    "associated_with": "associated_with",
    "associated": "associated_with",
    "linked_to": "associated_with",
    "connected_with": "associated_with",

    # -----------------------------------------------------
    # Generic relationship
    # -----------------------------------------------------

    "related_to": "related_to",
    "relates_to": "related_to",
    "related": "related_to",
    "relevant_to": "related_to",
}


# =========================================================
# Canonical relation registry
# =========================================================

INITIAL_CANONICAL_RELATIONS = sorted(
    set(
        EXACT_MAPPING.values()
    )
)


# =========================================================
# Relation quality validation
# =========================================================

def is_valid_relation(
    relation: str,
) -> bool:
    """
    Determine whether a cleaned relation looks like a
    reasonable knowledge-graph relationship.

    This prevents long or sentence-like text from becoming
    new relation types.

    Examples of valid relations:

        stores
        retrieves_from
        works_for
        located_in
        renders

    Examples of invalid relations:

        the_system_that_was_used
        information_about_the_document
        because_the_system_uses

    Args:
        relation:
            Cleaned relation string.

    Returns:
        True if the relation is reasonable.
    """

    if not relation:
        return False

    # -----------------------------------------------------
    # Length check
    # -----------------------------------------------------

    if len(relation) > 50:
        return False

    # -----------------------------------------------------
    # Word count check
    # -----------------------------------------------------

    parts = relation.split("_")

    if len(parts) > 5:
        return False

    # -----------------------------------------------------
    # Avoid obviously sentence-like relations.
    # -----------------------------------------------------

    forbidden_words = {
        "the",
        "this",
        "that",
        "which",
        "because",
        "when",
        "where",
        "while",
        "and",
        "or",
        "with",
        "from",
    }

    if any(
        part in forbidden_words
        for part in parts
    ):
        return False

    # -----------------------------------------------------
    # Must contain alphabetic content.
    # -----------------------------------------------------

    if not re.search(
        r"[a-z]",
        relation,
    ):
        return False

    return True


# =========================================================
# Clean relation
# =========================================================

def clean_relation(
    relation: str,
) -> str:
    """
    Convert raw relation text into lowercase snake_case.

    Example:

        " Works At "
            ↓
        "works_at"

        "located-in"
            ↓
        "located_in"
    """

    if not relation:
        return ""

    cleaned = relation.strip().lower()

    cleaned = re.sub(
        r"[\s\-]+",
        "_",
        cleaned,
    )

    cleaned = re.sub(
        r"[^a-z0-9_]",
        "",
        cleaned,
    )

    cleaned = re.sub(
        r"_+",
        "_",
        cleaned,
    )

    return cleaned.strip("_")


# =========================================================
# Normalize relations
# =========================================================

def normalize_relations(
    triples: list[Triple],
    return_decisions: bool = False,
):
    """
    Normalize relations for a complete document.

    IMPORTANT:

    Call this once after all chunks have been extracted.

    Do not call this independently for each chunk.

    Args:
        triples:
            All raw triples from one document.

        return_decisions:
            If True, return detailed normalization decisions.

    Returns:

        If False:

            list[Triple]

        If True:

            (normalized_triples, decisions)
    """

    if not triples:

        if return_decisions:
            return [], []

        return []

    # -----------------------------------------------------
    # Shared MiniLM model
    # -----------------------------------------------------

    model = get_embedding_model()

    # -----------------------------------------------------
    # Create growing canonical registry.
    # -----------------------------------------------------

    canonical_registry = list(
        INITIAL_CANONICAL_RELATIONS
    )

    canonical_embeddings = model.encode(
        canonical_registry,
        normalize_embeddings=True,
    )

    normalized = []

    decisions = []

    # =====================================================
    # Process every triple
    # =====================================================

    for triple in triples:

        raw_relation = triple.relation

        cleaned = clean_relation(
            raw_relation
        )

        # -------------------------------------------------
        # Empty relation
        # -------------------------------------------------

        if not cleaned:

            canonical = "related_to"

            decision = {
                "raw_relation": raw_relation,
                "cleaned_relation": cleaned,
                "canonical_relation": canonical,
                "method": "fallback_empty",
                "similarity": None,
            }

        # -------------------------------------------------
        # Exact mapping
        # -------------------------------------------------

        elif cleaned in EXACT_MAPPING:

            canonical = EXACT_MAPPING[
                cleaned
            ]

            decision = {
                "raw_relation": raw_relation,
                "cleaned_relation": cleaned,
                "canonical_relation": canonical,
                "method": "exact_mapping",
                "similarity": 1.0,
            }

        # -------------------------------------------------
        # Unknown relation
        # -------------------------------------------------

        else:

            # ---------------------------------------------
            # Generate embedding
            # ---------------------------------------------

            raw_embedding = model.encode(
                [cleaned],
                normalize_embeddings=True,
            )

            similarities = cosine_similarity(
                raw_embedding,
                canonical_embeddings,
            )[0]

            best_index = int(
                np.argmax(similarities)
            )

            best_score = float(
                similarities[best_index]
            )

            best_relation = (
                canonical_registry[
                    best_index
                ]
            )

            # ---------------------------------------------
            # Semantic MATCH
            # ---------------------------------------------

            if best_score >= MATCH_THRESHOLD:

                canonical = best_relation

                decision = {
                    "raw_relation": raw_relation,
                    "cleaned_relation": cleaned,
                    "canonical_relation": canonical,
                    "method": "semantic_match",
                    "similarity": round(
                        best_score,
                        4,
                    ),
                    "matched_with": best_relation,
                }

            # ---------------------------------------------
            # New relation
            # ---------------------------------------------

            elif is_valid_relation(cleaned):

                canonical = cleaned

                canonical_registry.append(
                    canonical
                )

                canonical_embeddings = np.vstack(
                    [
                        canonical_embeddings,
                        raw_embedding[0],
                    ]
                )

                decision = {
                    "raw_relation": raw_relation,
                    "cleaned_relation": cleaned,
                    "canonical_relation": canonical,
                    "method": "new_canonical",
                    "similarity": round(
                        best_score,
                        4,
                    ),
                    "closest_existing_relation": (
                        best_relation
                    ),
                }

            # ---------------------------------------------
            # Invalid relation
            # ---------------------------------------------

            else:

                canonical = "related_to"

                decision = {
                    "raw_relation": raw_relation,
                    "cleaned_relation": cleaned,
                    "canonical_relation": canonical,
                    "method": "fallback_invalid",
                    "similarity": round(
                        best_score,
                        4,
                    ),
                    "closest_existing_relation": (
                        best_relation
                    ),
                }

        # -------------------------------------------------
        # Create normalized triple.
        # -------------------------------------------------

        normalized.append(
            Triple(
                subject=triple.subject.strip(),
                relation=canonical,
                object=triple.object.strip(),
            )
        )

        decisions.append(
            decision
        )

    # =====================================================
    # Return
    # =====================================================

    if return_decisions:

        return normalized, decisions

    return normalized


# =========================================================
# Load processed document
# =========================================================

def load_processed_document(
    json_path: str | Path,
) -> tuple[dict, list[Triple]]:
    """
    Load the processed document and flatten all triples
    from all chunks into one list.
    """

    json_path = Path(
        json_path
    )

    if not json_path.exists():

        raise FileNotFoundError(
            f"Processed JSON not found: "
            f"{json_path}"
        )

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    triples = []

    for chunk in data.get(
        "chunks",
        [],
    ):

        for triple_data in chunk.get(
            "triples",
            [],
        ):

            triples.append(
                Triple.model_validate(
                    triple_data
                )
            )

    return data, triples


# =========================================================
# Save normalized document
# =========================================================

def save_normalized_document(
    original_data: dict,
    normalized_triples: list[Triple],
    decisions: list[dict],
    output_path: str | Path,
) -> None:
    """
    Save normalized triples while preserving the original
    chunk structure.
    """

    output_path = Path(
        output_path
    )

    # -----------------------------------------------------
    # Copy document metadata.
    # -----------------------------------------------------

    output_data = {
        "source_file": original_data.get(
            "source_file"
        ),

        "total_characters": original_data.get(
            "total_characters"
        ),

        "chunk_size": original_data.get(
            "chunk_size"
        ),

        "overlap": original_data.get(
            "overlap"
        ),

        "total_chunks": original_data.get(
            "total_chunks"
        ),

        "successful_chunks": original_data.get(
            "successful_chunks"
        ),

        "failed_chunks": original_data.get(
            "failed_chunks"
        ),

        "raw_total_triples": len(
            [
                triple
                for chunk in original_data.get(
                    "chunks",
                    []
                )
                for triple in chunk.get(
                    "triples",
                    []
                )
            ]
        ),

        "normalized_total_triples": len(
            normalized_triples
        ),

        "chunks": [],
    }

    # -----------------------------------------------------
    # Rebuild chunks.
    # -----------------------------------------------------

    triple_index = 0

    for chunk in original_data.get(
        "chunks",
        [],
    ):

        raw_chunk_triples = chunk.get(
            "triples",
            []
        )

        count = len(
            raw_chunk_triples
        )

        normalized_chunk_triples = (
            normalized_triples[
                triple_index:
                triple_index + count
            ]
        )

        triple_index += count

        new_chunk = {
            "chunk_id": chunk.get(
                "chunk_id"
            ),

            "text": chunk.get(
                "text",
                ""
            ),

            "status": chunk.get(
                "status",
                "unknown"
            ),

            "raw_triple_count": count,

            "normalized_triple_count": len(
                normalized_chunk_triples
            ),

            "triples": [
                triple.model_dump()
                for triple
                in normalized_chunk_triples
            ],
        }

        if "error" in chunk:

            new_chunk["error"] = (
                chunk["error"]
            )

        output_data[
            "chunks"
        ].append(
            new_chunk
        )

    # -----------------------------------------------------
    # Normalization metadata.
    # -----------------------------------------------------

    output_data[
        "normalization"
    ] = {

        "method": (
            "exact mapping + MiniLM semantic "
            "matching + new relation quality validation"
        ),

        "embedding_model": (
            "all-MiniLM-L6-v2"
        ),

        "match_threshold": (
            MATCH_THRESHOLD
        ),

        "initial_canonical_relation_count": (
            len(
                INITIAL_CANONICAL_RELATIONS
            )
        ),
    }

    # -----------------------------------------------------
    # Save.
    # -----------------------------------------------------

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output_data,
            file,
            indent=2,
            ensure_ascii=False,
        )


# =========================================================
# Print decisions
# =========================================================

def print_decisions(
    decisions: list[dict],
) -> None:
    """
    Print every relation normalization decision.
    """

    print(
        "\n" + "=" * 100
    )

    print(
        "RELATION NORMALIZATION DECISIONS"
    )

    print(
        "=" * 100
    )

    print(
        f"{'RAW':25} "
        f"{'CANONICAL':25} "
        f"{'METHOD':22} "
        f"{'SCORE':8}"
    )

    print(
        "-" * 100
    )

    for decision in decisions:

        raw = str(
            decision.get(
                "raw_relation",
                ""
            )
        )

        canonical = str(
            decision.get(
                "canonical_relation",
                ""
            )
        )

        method = str(
            decision.get(
                "method",
                ""
            )
        )

        score = decision.get(
            "similarity"
        )

        score_text = (
            "-"
            if score is None
            else str(score)
        )

        print(
            f"{raw[:25]:25} "
            f"{canonical[:25]:25} "
            f"{method[:22]:22} "
            f"{score_text:8}"
        )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    import sys

    # -----------------------------------------------------
    # Determine input.
    # -----------------------------------------------------

    if len(sys.argv) > 1:

        input_path = Path(
            sys.argv[1]
        )

    else:

        input_path = (
            PROCESSED_DIR
            / "sample.json"
        )

    # -----------------------------------------------------
    # Output.
    # -----------------------------------------------------

    output_path = (
        input_path.parent
        / f"{input_path.stem}_normalized.json"
    )

    # -----------------------------------------------------
    # Header.
    # -----------------------------------------------------

    print(
        "\n" + "=" * 100
    )

    print(
        "RELATION NORMALIZATION"
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

    # -----------------------------------------------------
    # Load.
    # -----------------------------------------------------

    print(
        "\nLoading raw triples..."
    )

    original_data, triples = (
        load_processed_document(
            input_path
        )
    )

    print(
        f"Raw triples loaded: {len(triples)}"
    )

    # -----------------------------------------------------
    # Normalize.
    # -----------------------------------------------------

    print(
        "\nNormalizing relations..."
    )

    normalized, decisions = (
        normalize_relations(
            triples,
            return_decisions=True,
        )
    )

    # -----------------------------------------------------
    # Save.
    # -----------------------------------------------------

    save_normalized_document(
        original_data,
        normalized,
        decisions,
        output_path,
    )

    # -----------------------------------------------------
    # Print decisions.
    # -----------------------------------------------------

    print_decisions(
        decisions
    )

    # -----------------------------------------------------
    # Summary.
    # -----------------------------------------------------

    exact_count = sum(
        d["method"] == "exact_mapping"
        for d in decisions
    )

    semantic_count = sum(
        d["method"] == "semantic_match"
        for d in decisions
    )

    new_count = sum(
        d["method"] == "new_canonical"
        for d in decisions
    )

    fallback_count = sum(
        d["method"].startswith(
            "fallback"
        )
        for d in decisions
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "NORMALIZATION SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"Raw triples:              {len(triples)}"
    )

    print(
        f"Normalized triples:       {len(normalized)}"
    )

    print(
        f"Exact mappings:           {exact_count}"
    )

    print(
        f"Semantic matches:         {semantic_count}"
    )

    print(
        f"New canonical relations:  {new_count}"
    )

    print(
        f"Fallback relations:       {fallback_count}"
    )

    print(
        f"\nSaved to:"
    )

    print(
        output_path
    )

    print(
        "\nRelation normalization complete."
    )