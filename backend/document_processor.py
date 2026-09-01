"""
backend/document_processor.py

Document ingestion pipeline:

    PDF
     ↓
    PDF text extraction
     ↓
    Smart chunking
     ↓
    Gemini triple extraction
     ↓
    Pydantic validation
     ↓
    JSON cache

Features:

    - Resumable processing
    - Per-chunk caching
    - Retry handling for Gemini failures
    - Saves after every processed chunk
    - Reuses successful cached chunks
    - Retries failed chunks
    - Detects changes in chunking configuration
"""

import json
import time
from pathlib import Path

from backend.pdf_processor import extract_text
from backend.chunker import (
    chunk_text_smart,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
)
from backend.extraction import extract_triples


# =========================================================
# Configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

PDF_DIR = DATA_DIR / "pdfs"

PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# Gemini retry configuration
# =========================================================

MAX_RETRIES = 3

INITIAL_RETRY_DELAY = 2


# =========================================================
# Cache version
# =========================================================

CACHE_VERSION = 1


# =========================================================
# Gemini extraction with retry
# =========================================================

def extract_triples_with_retry(
    chunk: str,
) -> tuple[list, str | None]:
    """
    Extract triples from a chunk.

    Gemini/API failures are retried up to MAX_RETRIES times.

    Returns:

        (triples, None)
            if successful

        ([], error_message)
            if all attempts fail
    """

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            print(
                f"    Gemini attempt "
                f"{attempt}/{MAX_RETRIES}"
            )

            triples = extract_triples(
                chunk
            )

            return triples, None

        except Exception as exc:

            last_error = str(exc)

            print(
                f"    Attempt {attempt} failed:"
                f" {last_error}"
            )

            # Don't wait after the final attempt.
            if attempt == MAX_RETRIES:
                break

            delay = (
                INITIAL_RETRY_DELAY
                * (2 ** (attempt - 1))
            )

            print(
                f"    Retrying in "
                f"{delay} seconds..."
            )

            time.sleep(delay)

    return [], last_error


# =========================================================
# Cache helpers
# =========================================================

def get_output_path(
    pdf_path: Path,
) -> Path:
    """
    Return the JSON cache path for a PDF.

    Example:

        sample.pdf
            ↓
        sample.json
    """

    return (
        PROCESSED_DIR
        / f"{pdf_path.stem}.json"
    )


def load_cache(
    output_path: Path,
) -> dict | None:
    """
    Load an existing processed JSON file.

    Returns:
        Dictionary if cache exists and is valid.

        None if cache doesn't exist or cannot be read.
    """

    if not output_path.exists():
        return None

    try:

        with open(
            output_path,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        if not isinstance(data, dict):
            return None

        return data

    except (
        json.JSONDecodeError,
        OSError,
    ) as exc:

        print(
            f"\nWarning: could not load cache:"
            f" {exc}"
        )

        return None


def save_cache(
    output_path: Path,
    data: dict,
) -> None:
    """
    Save processing data to JSON.

    The file is rewritten after every chunk so that
    progress is not lost if processing stops.
    """

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


# =========================================================
# Check whether existing cache can be reused
# =========================================================

def cache_matches_document(
    cache: dict,
    pdf_path: Path,
    total_chunks: int,
) -> bool:
    """
    Check whether the existing cache belongs to the
    same document and chunking configuration.

    If the chunk size or overlap changes, old chunks
    should not be reused.
    """

    if cache.get("cache_version") != CACHE_VERSION:
        return False

    if cache.get("source_file") != pdf_path.name:
        return False

    if cache.get("chunk_size") != DEFAULT_CHUNK_SIZE:
        return False

    if cache.get("overlap") != DEFAULT_OVERLAP:
        return False

    if cache.get("total_chunks") != total_chunks:
        return False

    return True


# =========================================================
# Create fresh cache structure
# =========================================================

def create_cache(
    pdf_path: Path,
    text: str,
    chunks: list[str],
) -> dict:
    """
    Create a new cache structure.
    """

    return {
        "cache_version": CACHE_VERSION,

        "source_file": pdf_path.name,

        "total_characters": len(text),

        "chunk_size": DEFAULT_CHUNK_SIZE,

        "overlap": DEFAULT_OVERLAP,

        "total_chunks": len(chunks),

        "successful_chunks": 0,

        "failed_chunks": 0,

        "total_triples": 0,

        "chunks": [],
    }


# =========================================================
# Process one PDF
# =========================================================

def process_document(
    pdf_path: str | Path,
) -> Path:
    """
    Process a PDF using a resumable cache.

    Existing successful chunks are reused.

    Failed or missing chunks are sent to Gemini again.
    """

    pdf_path = Path(pdf_path)

    # -----------------------------------------------------
    # Validate PDF
    # -----------------------------------------------------

    if not pdf_path.exists():

        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":

        raise ValueError(
            f"Expected a PDF file, "
            f"got: {pdf_path.name}"
        )

    print("\n" + "=" * 70)
    print("DOCUMENT PROCESSING")
    print("=" * 70)

    print(
        f"\nSource PDF: {pdf_path.name}"
    )

    # -----------------------------------------------------
    # Step 1: PDF → Text
    # -----------------------------------------------------

    print(
        "\n[1/4] Extracting text..."
    )

    text = extract_text(
        pdf_path
    )

    print(
        f"Extracted {len(text)} characters."
    )

    # -----------------------------------------------------
    # Step 2: Text → Chunks
    # -----------------------------------------------------

    print(
        "\n[2/4] Creating chunks..."
    )

    chunks = chunk_text_smart(
        text
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    # -----------------------------------------------------
    # Cache path
    # -----------------------------------------------------

    output_path = get_output_path(
        pdf_path
    )

    # -----------------------------------------------------
    # Try existing cache
    # -----------------------------------------------------

    existing_cache = load_cache(
        output_path
    )

    if existing_cache is not None:

        if cache_matches_document(
            existing_cache,
            pdf_path,
            len(chunks),
        ):

            print(
                "\nExisting cache found."
            )

            print(
                "Checking individual chunks..."
            )

            cache = existing_cache

        else:

            print(
                "\nExisting cache is not compatible "
                "with the current document/chunking "
                "configuration."
            )

            print(
                "Creating a fresh cache."
            )

            cache = create_cache(
                pdf_path,
                text,
                chunks,
            )

    else:

        print(
            "\nNo existing cache found."
        )

        print(
            "Creating a new cache."
        )

        cache = create_cache(
            pdf_path,
            text,
            chunks,
        )

    # -----------------------------------------------------
    # Make sure chunks list exists
    # -----------------------------------------------------

    if "chunks" not in cache:

        cache["chunks"] = []

    # -----------------------------------------------------
    # Build lookup of existing chunks
    # -----------------------------------------------------

    cached_chunks = {}

    for item in cache["chunks"]:

        chunk_id = item.get(
            "chunk_id"
        )

        if chunk_id is not None:

            cached_chunks[chunk_id] = item

    # -----------------------------------------------------
    # Step 3: Process chunks
    # -----------------------------------------------------

    print(
        "\n[3/4] Processing chunks..."
    )

    processed_chunks = []

    for chunk_id, chunk in enumerate(
        chunks,
        start=1
    ):

        print(
            f"\n  Chunk "
            f"{chunk_id}/{len(chunks)}"
        )

        # -------------------------------------------------
        # Check cache
        # -------------------------------------------------

        cached = cached_chunks.get(
            chunk_id
        )

        if cached is not None:

            cached_text = cached.get(
                "text"
            )

            cached_status = cached.get(
                "status"
            )

            # Reuse ONLY if both the text and status
            # match what we currently have.
            if (
                cached_text == chunk
                and cached_status == "success"
            ):

                print(
                    "    ✓ Using cached result"
                )

                processed_chunks.append(
                    cached
                )

                continue

        # -------------------------------------------------
        # Gemini extraction
        # -------------------------------------------------

        print(
            "    → Sending chunk to Gemini..."
        )

        triples, error = extract_triples_with_retry(
            chunk
        )

        # -------------------------------------------------
        # Convert Pydantic models → dictionaries
        # -------------------------------------------------

        triple_data = [
            triple.model_dump()
            for triple in triples
        ]

        # -------------------------------------------------
        # Create chunk result
        # -------------------------------------------------

        if error is None:

            chunk_data = {
                "chunk_id": chunk_id,
                "text": chunk,
                "status": "success",
                "triples": triple_data,
            }

            print(
                f"    → {len(triple_data)} triples"
            )

            print(
                "    → Status: success"
            )

        else:

            chunk_data = {
                "chunk_id": chunk_id,
                "text": chunk,
                "status": "failed",
                "triples": [],
                "error": error,
            }

            print(
                "    → Status: failed"
            )

        processed_chunks.append(
            chunk_data
        )

        # -------------------------------------------------
        # Save progress immediately
        # -------------------------------------------------

        cache["chunks"] = processed_chunks

        cache["successful_chunks"] = sum(
            1
            for item in processed_chunks
            if item["status"] == "success"
        )

        cache["failed_chunks"] = sum(
            1
            for item in processed_chunks
            if item["status"] == "failed"
        )

        cache["total_triples"] = sum(
            len(item["triples"])
            for item in processed_chunks
        )

        save_cache(
            output_path,
            cache
        )

        print(
            "    ✓ Progress saved"
        )

    # -----------------------------------------------------
    # Step 4: Final save
    # -----------------------------------------------------

    print(
        "\n[4/4] Finalizing processed document..."
    )

    cache["chunks"] = processed_chunks

    cache["successful_chunks"] = sum(
        1
        for item in processed_chunks
        if item["status"] == "success"
    )

    cache["failed_chunks"] = sum(
        1
        for item in processed_chunks
        if item["status"] == "failed"
    )

    cache["total_triples"] = sum(
        len(item["triples"])
        for item in processed_chunks
    )

    save_cache(
        output_path,
        cache
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "PROCESSING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Total chunks:      "
        f"{len(processed_chunks)}"
    )

    print(
        f"Successful chunks: "
        f"{cache['successful_chunks']}"
    )

    print(
        f"Failed chunks:     "
        f"{cache['failed_chunks']}"
    )

    print(
        f"Total triples:     "
        f"{cache['total_triples']}"
    )

    print(
        f"\nSaved to:"
    )

    print(
        output_path
    )

    return output_path


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python -m backend.document_processor "
            "data/pdfs/sample.pdf"
        )

        sys.exit(1)

    pdf_path = Path(
        sys.argv[1]
    )

    process_document(
        pdf_path
    )