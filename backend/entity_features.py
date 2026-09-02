"""
backend/entity_features.py

Build contextual representations for entities.

The feature object stores both:
    1. Human-readable relationship information
    2. Precomputed normalized sets for fast similarity comparison

This avoids repeatedly rebuilding sets during entity-pair comparison.
"""

from backend.extraction import Triple


# =========================================================
# Entity Feature
# =========================================================

class EntityFeature:
    """
    Contextual representation of one entity mention.
    """

    def __init__(self, name: str):

        self.name = name

        # Possible semantic/entity types
        self.entity_type_candidates = set()

        # Human-readable relationships
        self.outgoing = []
        self.incoming = []

        # Unique neighboring entities
        self.neighbors = set()

        # Number of times entity appears
        self.mentions = 0

        # -------------------------------------------------
        # Precomputed comparison features
        # -------------------------------------------------

        self.relation_set = set()
        self.neighbor_set = set()
        self.type_set = set()

    # =====================================================
    # Add outgoing relationship
    # =====================================================

    def add_outgoing(
        self,
        relation: str,
        object_name: str,
    ):

        self.outgoing.append(
            (
                relation,
                object_name,
            )
        )

        self.neighbors.add(
            object_name
        )

        self.relation_set.add(
            relation.strip().lower()
        )

        self.neighbor_set.add(
            object_name.strip().lower()
        )

    # =====================================================
    # Add incoming relationship
    # =====================================================

    def add_incoming(
        self,
        relation: str,
        subject_name: str,
    ):

        self.incoming.append(
            (
                relation,
                subject_name,
            )
        )

        self.neighbors.add(
            subject_name
        )

        self.relation_set.add(
            relation.strip().lower()
        )

        self.neighbor_set.add(
            subject_name.strip().lower()
        )

    # =====================================================
    # Add entity type
    # =====================================================

    def add_type(
        self,
        entity_type: str,
    ):

        if not entity_type:
            return

        value = entity_type.strip().lower()

        self.entity_type_candidates.add(
            entity_type
        )

        self.type_set.add(
            value
        )

    # =====================================================
    # Build representation
    # =====================================================

    def build_representation(
        self,
    ) -> str:

        lines = [
            f"Entity: {self.name}"
        ]

        if self.entity_type_candidates:

            types = sorted(
                self.entity_type_candidates
            )

            lines.append(
                "Possible types: "
                + ", ".join(types)
            )

        if self.outgoing:

            lines.append(
                "Outgoing relations:"
            )

            for relation, obj in self.outgoing[:15]:

                lines.append(
                    f"- {relation} -> {obj}"
                )

        if self.incoming:

            lines.append(
                "Incoming relations:"
            )

            for relation, subject in self.incoming[:15]:

                lines.append(
                    f"- {relation} <- {subject}"
                )

        return "\n".join(lines)


# =========================================================
# Build Features
# =========================================================

def build_entity_features(
    triples: list[Triple],
) -> dict[str, EntityFeature]:
    """
    Construct contextual features for all entities.

    Each entity is created once and reused.
    """

    features: dict[str, EntityFeature] = {}

    # -----------------------------------------------------
    # Get/create entity
    # -----------------------------------------------------

    def get(name: str) -> EntityFeature:

        if name not in features:

            features[name] = EntityFeature(
                name
            )

        features[name].mentions += 1

        return features[name]

    # -----------------------------------------------------
    # Process triples
    # -----------------------------------------------------

    for triple in triples:

        subject_name = triple.subject
        object_name = triple.object
        relation = triple.relation

        subject = get(
            subject_name
        )

        object_entity = get(
            object_name
        )

        # Subject -> Object
        subject.add_outgoing(
            relation,
            object_name,
        )

        # Object <- Subject
        object_entity.add_incoming(
            relation,
            subject_name,
        )

        # -------------------------------------------------
        # Infer type information
        # -------------------------------------------------

        if relation.lower() in {
            "is_a",
            "isa",
            "type_of",
        }:

            subject.add_type(
                object_name
            )

    return features