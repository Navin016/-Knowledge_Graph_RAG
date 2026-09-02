import json
import re
import sys
from pathlib import Path

from backend.embedding_model import get_rag_embedding_model

from backend.neo4j_client import Neo4jClient


# ============================================================
# CONFIGURATION
# ============================================================


DEFAULT_RESOLVED_FILE = (
    "data/processed/demo_normalized_resolved.json"
)

DEFAULT_NORMALIZED_FILE = (
    "data/processed/demo_normalized.json"
)


# ============================================================
# RELATION TYPE SAFETY
# ============================================================

def normalize_relationship_type(relation: str) -> str:
    """
    Convert a relation name into a safe Neo4j relationship type.

    Example:
        used_for        -> USED_FOR
        related to      -> RELATED_TO
        implemented-by  -> IMPLEMENTED_BY
    """

    relation = str(relation).strip()

    if not relation:
        return "RELATED_TO"

    relation = relation.upper()

    # Replace anything other than letters, numbers, or _
    relation = re.sub(r"[^A-Z0-9_]+", "_", relation)

    # Remove repeated underscores
    relation = re.sub(r"_+", "_", relation)

    # Remove leading/trailing underscores
    relation = relation.strip("_")

    if not relation:
        return "RELATED_TO"

    return relation


# ============================================================
# FILE LOADING
# ============================================================

def load_json(path: str) -> dict:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# EMBEDDING MODEL
# ============================================================
def load_vector_model():
    return get_rag_embedding_model()


# ============================================================
# DATABASE CLEANUP
# ============================================================

def clear_database(client: Neo4jClient):
    """
    Delete all nodes and relationships.

    USE ONLY when intentionally rebuilding the complete database.
    """

    with client.driver.session(database=client.database) as session:
        session.run("MATCH (n) DETACH DELETE n")

    print("Neo4j database cleared.")


# ============================================================
# INGEST ENTITY GRAPH
# ============================================================

def ingest_graph(
    client: Neo4jClient,
    resolved_data: dict
):
    triples = resolved_data.get("triples", [])

    if not triples:
        print("No resolved triples found.")
        return

    source_file = resolved_data.get(
        "source_file",
        "unknown"
    )

    print()
    print("=" * 70)
    print("INGESTING ENTITY GRAPH")
    print("=" * 70)

    print(f"Resolved triples: {len(triples)}")
    print(f"Source file:      {source_file}")

    entity_names = set()

    with client.driver.session(database=client.database) as session:

        for triple in triples:

            subject = str(
                triple.get("subject", "")
            ).strip()

            relation = str(
                triple.get("relation", "")
            ).strip()

            object_name = str(
                triple.get("object", "")
            ).strip()

            if not subject or not object_name:
                continue

            relationship_type = normalize_relationship_type(
                relation
            )

            entity_names.add(subject)
            entity_names.add(object_name)

            # ----------------------------------------------------
            # MERGE subject + object
            # ----------------------------------------------------

            query = f"""
                MERGE (s:Entity {{name: $subject}})
                ON CREATE SET
                    s.source_file = $source_file

                MERGE (o:Entity {{name: $object_name}})
                ON CREATE SET
                    o.source_file = $source_file

                MERGE (s)-[:`{relationship_type}`]->(o)
            """

            session.run(
                query,
                subject=subject,
                object_name=object_name,
                source_file=source_file
            )

    print(f"Unique entity names processed: {len(entity_names)}")


# ============================================================
# INGEST CHUNKS + EMBEDDINGS
# ============================================================

def ingest_chunks(
    client: Neo4jClient,
    normalized_data: dict,
    model
):
    chunks = normalized_data.get("chunks", [])

    if not chunks:
        print("No chunks found.")
        return

    source_file = normalized_data.get(
        "source_file",
        "unknown"
    )

    print()
    print("=" * 70)
    print("INGESTING CHUNKS + EMBEDDINGS")
    print("=" * 70)

    print(f"Chunks:      {len(chunks)}")
    print(f"Source file: {source_file}")

    texts = []

    valid_chunks = []

    for chunk in chunks:

        chunk_id = chunk.get("chunk_id")

        text = str(
            chunk.get("text", "")
        ).strip()

        if chunk_id is None or not text:
            continue

        valid_chunks.append(
            (chunk_id, text)
        )

        texts.append(text)

    if not texts:
        print("No valid chunk text found.")
        return

    print()
    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    print("Embeddings generated.")

    with client.driver.session(database=client.database) as session:

        for (chunk_id, text), embedding in zip(
            valid_chunks,
            embeddings
        ):

            chunk_key = (
                f"{source_file}::{chunk_id}"
            )

            session.run(
                """
                MERGE (c:Chunk {
                    chunk_key: $chunk_key
                })

                SET
                    c.chunk_id = $chunk_id,
                    c.source_file = $source_file,
                    c.text = $text,
                    c.embedding = $embedding
                """,
                chunk_key=chunk_key,
                chunk_id=chunk_id,
                source_file=source_file,
                text=text,
                embedding=embedding.tolist()
            )

    print(
        f"Inserted/updated {len(valid_chunks)} chunk nodes."
    )


# ============================================================
# VERIFY DATABASE
# ============================================================

def verify_database(client: Neo4jClient):

    print()
    print("=" * 70)
    print("VERIFYING NEO4J DATABASE")
    print("=" * 70)

    with client.driver.session(database=client.database) as session:

        entity_result = session.run(
            "MATCH (e:Entity) RETURN count(e) AS count"
        )

        entity_count = entity_result.single()["count"]

        chunk_result = session.run(
            "MATCH (c:Chunk) RETURN count(c) AS count"
        )

        chunk_count = chunk_result.single()["count"]

        relationship_result = session.run(
            """
            MATCH ()-[r]->()
            RETURN count(r) AS count
            """
        )

        relationship_count = (
            relationship_result.single()["count"]
        )

        embedded_result = session.run(
            """
            MATCH (c:Chunk)
            WHERE c.embedding IS NOT NULL
            RETURN count(c) AS count
            """
        )

        embedded_count = (
            embedded_result.single()["count"]
        )

    print(f"Entity nodes:       {entity_count}")
    print(f"Chunk nodes:        {chunk_count}")
    print(f"Relationships:      {relationship_count}")
    print(f"Embedded chunks:    {embedded_count}")


# ============================================================
# MAIN INGESTION PIPELINE
# ============================================================

def ingest(
    resolved_path: str,
    normalized_path: str,
    clear_existing: bool = False
):

    print()
    print("=" * 80)
    print("NEO4J KG-RAG INGESTION")
    print("=" * 80)

    print(f"Resolved file:   {resolved_path}")
    print(f"Normalized file: {normalized_path}")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    resolved_data = load_json(resolved_path)

    normalized_data = load_json(normalized_path)

    resolved_triples = resolved_data.get(
        "triples",
        []
    )

    chunks = normalized_data.get(
        "chunks",
        []
    )

    print()
    print(f"Resolved triples loaded: {len(resolved_triples)}")
    print(f"Chunks loaded:           {len(chunks)}")

    # --------------------------------------------------------
    # Neo4j client
    # --------------------------------------------------------

    client = Neo4jClient()

    try:

        client.driver.verify_connectivity()

        print()
        print("Neo4j connection verified.")

        # ----------------------------------------------------
        # Optional database cleanup
        # ----------------------------------------------------

        if clear_existing:
            clear_database(client)

        # ----------------------------------------------------
        # Load embedding model
        # ----------------------------------------------------

        embedding_model = load_vector_model()

        # ----------------------------------------------------
        # Graph ingestion
        # ----------------------------------------------------

        ingest_graph(
            client,
            resolved_data
        )

        # ----------------------------------------------------
        # Chunk + vector ingestion
        # ----------------------------------------------------

        ingest_chunks(
            client,
            normalized_data,
            embedding_model
        )

        # ----------------------------------------------------
        # Verification
        # ----------------------------------------------------

        verify_database(client)

        print()
        print("=" * 80)
        print("NEO4J INGESTION COMPLETE")
        print("=" * 80)

    finally:
        client.close()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    resolved_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else DEFAULT_RESOLVED_FILE
    )

    normalized_file = (
        sys.argv[2]
        if len(sys.argv) > 2
        else DEFAULT_NORMALIZED_FILE
    )

    # By default, DO NOT clear Neo4j.
    ingest(
        resolved_path=resolved_file,
        normalized_path=normalized_file,
        clear_existing=False
    )