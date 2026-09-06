"""
backend/pipeline.py

End-to-end KG-RAG document ingestion pipeline.

Flow:
    PDF
      ↓
    Document processing / Gemini extraction
      ↓
    Relation normalization
      ↓
    Entity resolution
      ↓
    Neo4j setup
      ↓
    Neo4j graph + chunk/vector ingestion
      ↓
    Entity vector index

Run from project root:

    python -m backend.pipeline data/pdfs/demo.pdf

Optional full Neo4j rebuild:

    python -m backend.pipeline data/pdfs/demo.pdf --clear
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable

from backend.document_processor import process_document
from backend.relation_normalizer import (
    load_processed_document,
    normalize_relations,
    save_normalized_document,
)
from backend.entity_resolution import (
    load_document,
    resolve_entities,
    save_result,
)
from backend.neo4j_client import Neo4jClient
from backend.neo4j_ingestion import ingest as ingest_neo4j
from backend.embedding_model import get_rag_embedding_model


# ============================================================
# Pipeline phase reporting
# ============================================================

PIPELINE_PHASES = [
    ("upload", "PDF uploaded"),
    ("processing", "PDF processing + Gemini extraction"),
    ("normalization", "Relation normalization"),
    ("resolution", "Entity resolution"),
    ("neo4j_setup", "Neo4j setup"),
    ("neo4j_ingestion", "Neo4j graph + chunk ingestion"),
    ("embeddings", "Entity/vector embeddings"),
    ("verification", "Final verification"),
    ("ready", "Ready for questions"),
]


ProgressCallback = Callable[
    [str, str, int, str],
    None,
]


def _report_phase(
    callback: ProgressCallback | None,
    phase_id: str,
    status: str,
    progress: int,
    message: str,
) -> None:
    """Report one pipeline phase without making the pipeline depend on FastAPI."""
    if callback is None:
        return

    callback(
        phase_id,
        status,
        progress,
        message,
    )



# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"


# ============================================================
# Neo4j setup
# ============================================================

def setup_neo4j() -> None:
    """Create required constraints and vector indexes idempotently."""
    client = Neo4jClient()

    queries = [
        """
        CREATE VECTOR INDEX chunk_text_vector IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding)
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
            }
        }
        """,
        """
        CREATE VECTOR INDEX entity_name_vector IF NOT EXISTS
        FOR (e:Entity) ON (e.embedding)
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
            }
        }
        """,    
    ]

    try:
        client.driver.verify_connectivity()
        with client.driver.session(database=client.database) as session:
            for query in queries:
                session.run(query).consume()

        print("Neo4j constraints and vector indexes are ready.")

    finally:
        client.close()


# ============================================================
# Entity vector index population
# ============================================================

def build_entity_vector_index() -> None:
    """Embed Entity.name with the shared all-MiniLM-L6-v2 model."""
    print()
    print("=" * 80)
    print("BUILDING ENTITY VECTOR INDEX")
    print("=" * 80)

    client = Neo4jClient()

    try:
        client.driver.verify_connectivity()

        with client.driver.session(database=client.database) as session:
            records = list(
                session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.name IS NOT NULL AND trim(e.name) <> ''
                    RETURN e.name AS name
                    ORDER BY e.name
                    """
                )
            )

        names = [record["name"] for record in records]
        print(f"Entities found: {len(names)}")

        if not names:
            print("No entities found. Skipping entity embeddings.")
            return

        model = get_rag_embedding_model()

        print("Generating entity embeddings...")
        embeddings = model.encode(
            names,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        with client.driver.session(database=client.database) as session:
            for name, embedding in zip(names, embeddings):
                session.run(
                    """
                    MATCH (e:Entity {name: $name})
                    SET e.embedding = $embedding
                    """,
                    name=name,
                    embedding=embedding.tolist(),
                )

        print(f"Entity embeddings written: {len(names)}")

    finally:
        client.close()


# ============================================================
# Verification
# ============================================================

def verify_pipeline() -> None:
    """Print final Neo4j counts after ingestion and indexing."""
    client = Neo4jClient()

    try:
        client.driver.verify_connectivity()

        with client.driver.session(database=client.database) as session:
            entity_count = session.run(
                "MATCH (e:Entity) RETURN count(e) AS count"
            ).single()["count"]

            chunk_count = session.run(
                "MATCH (c:Chunk) RETURN count(c) AS count"
            ).single()["count"]

            relationship_count = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS count"
            ).single()["count"]

            embedded_chunks = session.run(
                """
                MATCH (c:Chunk)
                WHERE c.embedding IS NOT NULL
                RETURN count(c) AS count
                """
            ).single()["count"]

            embedded_entities = session.run(
                """
                MATCH (e:Entity)
                WHERE e.embedding IS NOT NULL
                RETURN count(e) AS count
                """
            ).single()["count"]

        print()
        print("=" * 80)
        print("FINAL PIPELINE VERIFICATION")
        print("=" * 80)
        print(f"Entity nodes:          {entity_count}")
        print(f"Chunk nodes:           {chunk_count}")
        print(f"Relationships:         {relationship_count}")
        print(f"Embedded chunks:       {embedded_chunks}")
        print(f"Embedded entities:     {embedded_entities}")

    finally:
        client.close()


# ============================================================
# Main pipeline
# ============================================================

def run_pipeline(
    pdf_path: str | Path,
    clear_existing: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, str]:
    """Run the complete PDF -> Neo4j ingestion pipeline."""
    start_time = time.perf_counter()

    pdf_path = Path(pdf_path).resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path.name}")

    _report_phase(
        progress_callback,
        "upload",
        "completed",
        8,
        "PDF upload completed. Starting ingestion pipeline.",
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 90)
    print("KG-RAG END-TO-END PIPELINE")
    print("=" * 90)
    print(f"PDF:            {pdf_path}")
    print(f"Clear Neo4j:    {clear_existing}")

    # --------------------------------------------------------
    # 1. PDF -> extracted triples JSON
    # --------------------------------------------------------
    print()
    print("[1/7] PROCESSING PDF")
    _report_phase(
        progress_callback,
        "processing",
        "running",
        12,
        "Extracting PDF text, chunking the document, and extracting triples with Gemini...",
    )

    processed_path = Path(
        process_document(pdf_path)
    ).resolve()

    _report_phase(
        progress_callback,
        "processing",
        "completed",
        28,
        "PDF processing and Gemini extraction completed.",
    )

    # --------------------------------------------------------
    # 2. Relation normalization
    # --------------------------------------------------------
    print()
    print("[2/7] NORMALIZING RELATIONS")

    _report_phase(
        progress_callback,
        "normalization",
        "running",
        32,
        "Normalizing extracted relationship types...",
    )

    normalized_path = (
        processed_path.parent
        / f"{processed_path.stem}_normalized.json"
    )

    original_data, raw_triples = load_processed_document(processed_path)

    normalized_triples, normalization_decisions = normalize_relations(
        raw_triples,
        return_decisions=True,
    )

    save_normalized_document(
        original_data=original_data,
        normalized_triples=normalized_triples,
        decisions=normalization_decisions,
        output_path=normalized_path,
    )

    print(f"Normalized triples: {len(normalized_triples)}")
    print(f"Saved to: {normalized_path}")

    _report_phase(
        progress_callback,
        "normalization",
        "completed",
        40,
        f"Relation normalization completed: {len(normalized_triples)} triples.",
    )

    # --------------------------------------------------------
    # 3. Entity resolution
    # --------------------------------------------------------
    print()
    print("[3/7] RESOLVING ENTITIES")

    _report_phase(
        progress_callback,
        "resolution",
        "running",
        43,
        "Resolving entity mentions into canonical entities...",
    )

    resolved_path = (
        normalized_path.parent
        / f"{normalized_path.stem}_resolved.json"
    )

    resolved_original_data, normalized_for_resolution, texts = load_document(
        normalized_path
    )

    (
        resolved_triples,
        resolution_decisions,
        clusters,
        entity_to_canonical,
        _,
    ) = resolve_entities(
        normalized_for_resolution,
        source_texts=texts,
        use_gemini_verifier=False,
    )

    save_result(
        resolved_original_data,
        resolved_triples,
        resolution_decisions,
        clusters,
        entity_to_canonical,
        resolved_path,
    )

    print(f"Resolved unique triples: {len(resolved_triples)}")
    print(f"Canonical entities:      {len(clusters)}")
    print(f"Saved to: {resolved_path}")

    _report_phase(
        progress_callback,
        "resolution",
        "completed",
        56,
        f"Entity resolution completed: {len(clusters)} canonical entities.",
    )

    # --------------------------------------------------------
    # 4. Neo4j setup
    # --------------------------------------------------------
    print()
    print("[4/7] PREPARING NEO4J")

    _report_phase(
        progress_callback,
        "neo4j_setup",
        "running",
        60,
        "Preparing Neo4j constraints and vector indexes...",
    )

    setup_neo4j()

    _report_phase(
        progress_callback,
        "neo4j_setup",
        "completed",
        65,
        "Neo4j constraints and vector indexes are ready.",
    )

    # --------------------------------------------------------
    # 5. Graph + chunk/vector ingestion
    # --------------------------------------------------------
    print()
    print("[5/7] INGESTING GRAPH + CHUNKS")

    _report_phase(
        progress_callback,
        "neo4j_ingestion",
        "running",
        68,
        "Writing entities, relationships, chunks, and chunk embeddings to Neo4j...",
    )

    ingest_neo4j(
        resolved_path=str(resolved_path),
        normalized_path=str(normalized_path),
        clear_existing=clear_existing,
    )

    _report_phase(
        progress_callback,
        "neo4j_ingestion",
        "completed",
        80,
        "Neo4j graph and chunk ingestion completed.",
    )

    # --------------------------------------------------------
    # 6. Entity vector embeddings
    # --------------------------------------------------------
    print()
    print("[6/7] INDEXING ENTITY EMBEDDINGS")

    _report_phase(
        progress_callback,
        "embeddings",
        "running",
        83,
        "Generating and writing entity vector embeddings...",
    )

    build_entity_vector_index()

    _report_phase(
        progress_callback,
        "embeddings",
        "completed",
        92,
        "Entity/vector embeddings completed.",
    )

    # --------------------------------------------------------
    # 7. Final verification
    # --------------------------------------------------------
    print()
    print("[7/7] VERIFYING")

    _report_phase(
        progress_callback,
        "verification",
        "running",
        94,
        "Verifying Neo4j nodes, relationships, and embeddings...",
    )

    verify_pipeline()

    _report_phase(
        progress_callback,
        "verification",
        "completed",
        98,
        "Final Neo4j verification completed.",
    )

    _report_phase(
        progress_callback,
        "ready",
        "completed",
        100,
        "Ingestion completed. The PDF is ready for questions.",
    )

    elapsed = time.perf_counter() - start_time

    print()
    print("=" * 90)
    print("KG-RAG PIPELINE COMPLETE")
    print("=" * 90)
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Processed JSON:  {processed_path}")
    print(f"Normalized JSON: {normalized_path}")
    print(f"Resolved JSON:   {resolved_path}")

    return {
        "processed": str(processed_path),
        "normalized": str(normalized_path),
        "resolved": str(resolved_path),
    }


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete KG-RAG PDF ingestion pipeline."
    )

    parser.add_argument(
        "pdf",
        help="Path to the PDF file.",
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all existing Neo4j nodes/relationships before ingestion.",
    )

    args = parser.parse_args()

    run_pipeline(
        pdf_path=args.pdf,
        clear_existing=args.clear,
    )


if __name__ == "__main__":
    main()
