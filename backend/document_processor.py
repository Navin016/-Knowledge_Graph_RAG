"""
backend/document_processor.py

Phase 1-4:
Document ingestion pipeline.

Pipeline:

    PDF
     ↓
    PDF text extraction
     ↓
    Smart chunking
     ↓
    Gemini 3.5 Flash-Lite extraction
     ↓
    Pydantic validation
     ↓
    JSON cache


Features:

    - Resumable processing
    - Per-chunk caching
    - RPM rate limiting
    - Exponential backoff
    - 429 handling
    - Saves after every processed chunk
    - Reuses successful cached chunks
    - Retries failed chunks
    - Detects changes in chunking configuration
    - Safe interruption / restart


Gemini 3.5 Flash-Lite limits configured for this project:

    RPM = 15
    TPM = 250,000
    RPD = 500

The processor does not attempt to consume the entire quota.

Requests are spaced according to the RPM limit.
"""


# =========================================================
# Imports
# =========================================================

import json
import os
import time
from pathlib import Path


from dotenv import load_dotenv


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
# Environment
# =========================================================

load_dotenv(
    BASE_DIR / ".env"
)


# =========================================================
# Gemini configuration
# =========================================================

GEMINI_MODEL = os.getenv(
    "GEMINI_EXTRACTION_MODEL",
    "gemini-3.5-flash-lite"
)


# =========================================================
# Gemini quota configuration
# =========================================================
#
# Current model limits:
#
#     RPM = 15
#     TPM = 250,000
#     RPD = 500
#
# These values are configuration/reference values.
# The processor actively enforces RPM.
#
# TPM depends on actual request token usage and is not
# estimated purely from Python character counts.
# =========================================================

GEMINI_RPM = int(
    os.getenv(
        "GEMINI_RPM",
        "15"
    )
)


GEMINI_TPM = int(
    os.getenv(
        "GEMINI_TPM",
        "250000"
    )
)


GEMINI_RPD = int(
    os.getenv(
        "GEMINI_RPD",
        "500"
    )
)


# =========================================================
# Rate limiting
# =========================================================

if GEMINI_RPM <= 0:
    raise ValueError(
        "GEMINI_RPM must be greater than 0."
    )


# For 15 RPM:
#
#     60 / 15 = 4 seconds
#
# Therefore requests are spaced approximately
# 4 seconds apart.

MIN_REQUEST_INTERVAL = (
    60.0 / GEMINI_RPM
)


_last_request_time = 0.0


# =========================================================
# Retry configuration
# =========================================================

MAX_RETRIES = 3


# First retry waits 4 seconds.
#
# Exponential backoff:
#
#     4 seconds
#     8 seconds
#     16 seconds

INITIAL_RETRY_DELAY = 4


# =========================================================
# Cache version
# =========================================================
#
# Increment whenever the cache structure or chunking
# behavior changes significantly.
# =========================================================

CACHE_VERSION = 2


# =========================================================
# Rate limiter
# =========================================================

def wait_for_rate_limit() -> None:
    """
    Ensure Gemini requests respect the configured RPM.

    Example:

        RPM = 15

        60 / 15 = 4 seconds/request

    The function does not limit local processing.
    It only controls Gemini API calls.
    """

    global _last_request_time

    now = time.monotonic()

    elapsed = (
        now - _last_request_time
    )

    if elapsed < MIN_REQUEST_INTERVAL:

        wait_time = (
            MIN_REQUEST_INTERVAL
            - elapsed
        )

        print(
            f"    Rate limiter: "
            f"waiting {wait_time:.2f}s..."
        )

        time.sleep(
            wait_time
        )

    _last_request_time = (
        time.monotonic()
    )


# =========================================================
# Detect rate-limit errors
# =========================================================

def is_rate_limit_error(
    error_message: str,
) -> bool:
    """
    Detect Gemini rate-limit / quota errors.

    Returns:
        True if the error appears to be related
        to rate limiting or quota exhaustion.
    """

    message = (
        error_message
        .lower()
    )

    indicators = [
        "429",
        "resource_exhausted",
        "rate limit",
        "rate_limit",
        "quota",
        "too many requests",
    ]

    return any(
        indicator in message
        for indicator in indicators
    )


# =========================================================
# Gemini extraction with retry
# =========================================================

def extract_triples_with_retry(
    chunk: str,
) -> tuple[list, str | None]:
    """
    Extract triples from one chunk.

    Features:

        - RPM limiting
        - Retry handling
        - Exponential backoff
        - 429 detection
        - Same chunk is retried
        - No successful data is discarded

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

            # -------------------------------------------------
            # Respect RPM
            # -------------------------------------------------

            wait_for_rate_limit()


            print(
                f"    Gemini attempt "
                f"{attempt}/{MAX_RETRIES}"
            )


            # -------------------------------------------------
            # Gemini extraction
            # -------------------------------------------------

            triples = extract_triples(
                chunk
            )


            # -------------------------------------------------
            # Success
            # -------------------------------------------------

            return triples, None


        except Exception as exc:

            last_error = str(
                exc
            )


            print(
                f"    Attempt {attempt} failed:"
                f" {last_error}"
            )


            # -------------------------------------------------
            # Last attempt
            # -------------------------------------------------

            if attempt == MAX_RETRIES:

                break


            # -------------------------------------------------
            # Determine error type
            # -------------------------------------------------

            rate_limit = (
                is_rate_limit_error(
                    last_error
                )
            )


            # -------------------------------------------------
            # Exponential backoff
            # -------------------------------------------------

            delay = (
                INITIAL_RETRY_DELAY
                * (2 ** (attempt - 1))
            )


            if rate_limit:

                print(
                    f"    Rate limit detected."
                    f" Retrying SAME chunk in "
                    f"{delay} seconds..."
                )

            else:

                print(
                    f"    Temporary API error."
                    f" Retrying in "
                    f"{delay} seconds..."
                )


            time.sleep(
                delay
            )


    # ---------------------------------------------------------
    # All attempts failed
    # ---------------------------------------------------------

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


# =========================================================
# Load cache
# =========================================================

def load_cache(
    output_path: Path,
) -> dict | None:
    """
    Load an existing processed JSON file.

    Returns:

        Dictionary if cache exists and is valid.

        None if cache does not exist or
        cannot be read.
    """

    if not output_path.exists():

        return None


    try:

        with open(
            output_path,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )


        if not isinstance(
            data,
            dict
        ):

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


# =========================================================
# Save cache
# =========================================================

def save_cache(
    output_path: Path,
    data: dict,
) -> None:
    """
    Save processing data to JSON.

    Progress is written after every chunk.

    Therefore:

        successful chunks
        +
        failed chunks

    are preserved if processing stops.
    """

    # -----------------------------------------------------
    # Write to temporary file first
    #
    # This reduces the chance of leaving a corrupted
    # JSON file if the program is interrupted during write.
    # -----------------------------------------------------

    temp_path = output_path.with_suffix(
        ".tmp"
    )


    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


    # -----------------------------------------------------
    # Replace old cache atomically
    # -----------------------------------------------------

    temp_path.replace(
        output_path
    )


# =========================================================
# Cache compatibility
# =========================================================

def cache_matches_document(
    cache: dict,
    pdf_path: Path,
    total_chunks: int,
) -> bool:
    """
    Check whether an existing cache belongs to
    the same document and chunking configuration.

    Cache is reusable only when:

        - cache version matches
        - source PDF matches
        - chunk size matches
        - overlap matches
        - total chunk count matches
    """

    if cache.get(
        "cache_version"
    ) != CACHE_VERSION:

        return False


    if cache.get(
        "source_file"
    ) != pdf_path.name:

        return False


    if cache.get(
        "chunk_size"
    ) != DEFAULT_CHUNK_SIZE:

        return False


    if cache.get(
        "overlap"
    ) != DEFAULT_OVERLAP:

        return False


    if cache.get(
        "total_chunks"
    ) != total_chunks:

        return False


    return True


# =========================================================
# Create fresh cache
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

        "total_characters": len(
            text
        ),

        "chunk_size": DEFAULT_CHUNK_SIZE,

        "overlap": DEFAULT_OVERLAP,

        "total_chunks": len(
            chunks
        ),

        "successful_chunks": 0,

        "failed_chunks": 0,

        "total_triples": 0,

        "chunks": [],
    }


# =========================================================
# Recalculate cache statistics
# =========================================================

def update_cache_statistics(
    cache: dict,
) -> None:
    """
    Recalculate:

        successful_chunks
        failed_chunks
        total_triples
    """

    chunks = cache.get(
        "chunks",
        []
    )


    cache["successful_chunks"] = sum(
        1
        for item in chunks
        if item.get("status")
        == "success"
    )


    cache["failed_chunks"] = sum(
        1
        for item in chunks
        if item.get("status")
        == "failed"
    )


    cache["total_triples"] = sum(
        len(
            item.get(
                "triples",
                []
            )
        )
        for item in chunks
    )


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

    The downstream output format remains:

        {
            "chunk_id": ...,
            "text": ...,
            "status": ...,
            "triples": [...]
        }
    """

    pdf_path = Path(
        pdf_path
    )


    # =====================================================
    # Validate PDF
    # =====================================================

    if not pdf_path.exists():

        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )


    if pdf_path.suffix.lower() != ".pdf":

        raise ValueError(
            f"Expected a PDF file, "
            f"got: {pdf_path.name}"
        )


    # =====================================================
    # Header
    # =====================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "DOCUMENT PROCESSING"
    )

    print(
        "=" * 80
    )


    print(
        f"\nSource PDF: "
        f"{pdf_path.name}"
    )


    print(
        f"Gemini model: "
        f"{GEMINI_MODEL}"
    )


    print(
        f"Gemini RPM: "
        f"{GEMINI_RPM}"
    )


    print(
        f"Gemini TPM: "
        f"{GEMINI_TPM:,}"
    )


    print(
        f"Gemini RPD: "
        f"{GEMINI_RPD}"
    )


    print(
        f"Chunk size: "
        f"{DEFAULT_CHUNK_SIZE}"
    )


    print(
        f"Chunk overlap: "
        f"{DEFAULT_OVERLAP}"
    )


    # =====================================================
    # Step 1
    # PDF → Text
    # =====================================================

    print(
        "\n[1/4] Extracting text..."
    )


    text = extract_text(
        pdf_path
    )


    print(
        f"Extracted "
        f"{len(text)} characters."
    )


    if not text.strip():

        raise ValueError(
            "No text could be extracted "
            "from the PDF."
        )


    # =====================================================
    # Step 2
    # Text → Chunks
    # =====================================================

    print(
        "\n[2/4] Creating chunks..."
    )


    chunks = chunk_text_smart(
        text
    )


    print(
        f"Created "
        f"{len(chunks)} chunks."
    )


    if not chunks:

        raise ValueError(
            "No chunks were created "
            "from the extracted text."
        )


    # =====================================================
    # Cache path
    # =====================================================

    output_path = get_output_path(
        pdf_path
    )


    # =====================================================
    # Existing cache
    # =====================================================

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
                "\nExisting compatible "
                "cache found."
            )

            print(
                "Successful chunks will "
                "be reused."
            )

            cache = existing_cache


        else:

            print(
                "\nExisting cache is not "
                "compatible with the "
                "current document/chunking "
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


    # =====================================================
    # Ensure chunks list exists
    # =====================================================

    if "chunks" not in cache:

        cache["chunks"] = []


    # =====================================================
    # Build cached chunk lookup
    # =====================================================

    cached_chunks = {}


    for item in cache[
        "chunks"
    ]:

        chunk_id = item.get(
            "chunk_id"
        )


        if chunk_id is not None:

            cached_chunks[
                chunk_id
            ] = item


    # =====================================================
    # Step 3
    # Process chunks
    # =====================================================

    print(
        "\n[3/4] Processing chunks..."
    )


    processed_chunks = []


    for chunk_id, chunk in enumerate(
        chunks,
        start=1,
    ):

        print(
            "\n"
            + "-" * 70
        )


        print(
            f"Chunk "
            f"{chunk_id}/{len(chunks)}"
        )


        print(
            f"Characters: "
            f"{len(chunk)}"
        )


        # =================================================
        # Check existing cache
        # =================================================

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


            # ---------------------------------------------
            # Reuse ONLY successful chunks whose text
            # is identical to the current chunk.
            # ---------------------------------------------

            if (
                cached_text == chunk
                and cached_status
                == "success"
            ):

                print(
                    "    ✓ Using cached result"
                )


                print(
                    f"    → "
                    f"{len(cached.get('triples', []))}"
                    f" triples"
                )


                processed_chunks.append(
                    cached
                )


                continue


            # ---------------------------------------------
            # Failed chunks are intentionally NOT reused.
            #
            # They will be sent to Gemini again.
            # ---------------------------------------------

            if (
                cached_text == chunk
                and cached_status
                == "failed"
            ):

                print(
                    "    Previous attempt "
                    "failed."
                )

                print(
                    "    → Retrying chunk..."
                )


        # =================================================
        # Gemini extraction
        # =================================================

        print(
            "    → Sending chunk "
            "to Gemini..."
        )


        triples, error = (
            extract_triples_with_retry(
                chunk
            )
        )


        # =================================================
        # Convert Pydantic → dictionaries
        # =================================================

        triple_data = [

            triple.model_dump()

            for triple in triples

        ]


        # =================================================
        # Create successful result
        # =================================================

        if error is None:

            chunk_data = {

                "chunk_id": chunk_id,

                "text": chunk,

                "status": "success",

                "triples": triple_data,

            }


            print(
                f"    → "
                f"{len(triple_data)} "
                f"triples"
            )


            print(
                "    → Status: success"
            )


        # =================================================
        # Create failed result
        # =================================================

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


            print(
                f"    → Error: "
                f"{error}"
            )


        # =================================================
        # Replace any previous result for this chunk
        # =================================================

        processed_chunks.append(
            chunk_data
        )


        # =================================================
        # Save progress immediately
        # =================================================

        cache["chunks"] = (
            processed_chunks
        )


        update_cache_statistics(
            cache
        )


        save_cache(
            output_path,
            cache
        )


        print(
            "    ✓ Progress saved"
        )


    # =====================================================
    # Step 4
    # Finalize
    # =====================================================

    print(
        "\n[4/4] Finalizing "
        "processed document..."
    )


    cache["chunks"] = (
        processed_chunks
    )


    update_cache_statistics(
        cache
    )


    save_cache(
        output_path,
        cache
    )


    # =====================================================
    # Summary
    # =====================================================

    print(
        "\n"
        + "=" * 80
    )


    print(
        "PROCESSING COMPLETE"
    )


    print(
        "=" * 80
    )


    print(
        f"Total chunks:       "
        f"{len(processed_chunks)}"
    )


    print(
        f"Successful chunks:  "
        f"{cache['successful_chunks']}"
    )


    print(
        f"Failed chunks:      "
        f"{cache['failed_chunks']}"
    )


    print(
        f"Total triples:      "
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
            "python -m "
            "backend.document_processor "
            "data/pdfs/sample.pdf"
        )


        sys.exit(1)


    pdf_path = Path(
        sys.argv[1]
    )


    process_document(
        pdf_path
    )