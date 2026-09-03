"""
FastAPI layer for the KG-RAG application.

Version 1 intentionally keeps one active PDF in Neo4j at a time.

Optimization:
    - Uploaded PDFs are identified by SHA-256 content hash.
    - If the same PDF is uploaded again while it is already the
      active document, ingestion is skipped completely.
    - The user can go directly to Q/A.
    - If the cache exists but the PDF is not the active Neo4j
      document, the existing pipeline is still executed. The
      document processor will reuse the cached Gemini extraction.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.pipeline import run_pipeline
from backend.rag_service import RAGService

from backend.document_processor import (
    calculate_file_hash,
    get_output_path,
    load_cache,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

UPLOAD_DIR = (
    DATA_DIR
    / "pdfs"
    / "uploads"
)

PROCESSED_DIR = DATA_DIR / "processed"

ACTIVE_DOCUMENT_STATE = (
    PROCESSED_DIR
    / "active_document.json"
)


UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(
    BASE_DIR / ".env"
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_UPLOAD_BYTES = int(
    os.getenv(
        "MAX_UPLOAD_BYTES",
        str(50 * 1024 * 1024),
    )
)


origins_raw = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
)


ALLOWED_ORIGINS = [
    item.strip()
    for item in origins_raw.split(",")
    if item.strip()
]


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="KG-RAG API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# RUNTIME STATE
# ============================================================

jobs: dict[str, dict[str, Any]] = {}

jobs_lock = Lock()


runtime: dict[str, Any] = {
    "rag_service": None,
    "active_document_hash": None,
    "active_document_filename": None,
    "active_job_id": None,
}


runtime_lock = Lock()


# ============================================================
# MODELS
# ============================================================

class QueryRequest(BaseModel):

    question: str = Field(
        min_length=1,
        max_length=2000,
    )

    job_id: str = Field(
        min_length=1,
    )


# ============================================================
# JOB HELPERS
# ============================================================

def _update_job(
    job_id: str,
    **updates: Any,
) -> None:

    with jobs_lock:

        if job_id in jobs:

            jobs[job_id].update(
                updates
            )


# ============================================================
# RAG SERVICE
# ============================================================

def _get_rag_service() -> RAGService:

    with runtime_lock:

        service = runtime.get(
            "rag_service"
        )

        if service is None:

            service = RAGService()

            runtime[
                "rag_service"
            ] = service

        return service


# ============================================================
# FILENAME
# ============================================================

def _sanitize_filename(
    filename: str,
) -> str:

    stem = Path(
        filename
    ).stem

    stem = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        stem,
    ).strip(
        "._"
    )

    return stem or "document"


# ============================================================
# ACTIVE DOCUMENT STATE
# ============================================================

def _load_active_document_state() -> dict[str, Any] | None:
    """
    Load the last active document state written by the FastAPI
    application.

    This allows repeated uploads of the same PDF to skip the
    pipeline even after a FastAPI reload/restart, assuming the
    Neo4j database has not been cleared or changed externally.
    """

    if not ACTIVE_DOCUMENT_STATE.exists():

        return None

    try:

        with open(
            ACTIVE_DOCUMENT_STATE,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

        if not isinstance(
            data,
            dict,
        ):

            return None

        return data

    except (
        OSError,
        json.JSONDecodeError,
    ):

        return None


def _save_active_document_state(
    *,
    file_hash: str,
    filename: str,
    job_id: str,
) -> None:
    """
    Persist the active document hash.

    The hash is the identity of the PDF.
    The filename is metadata only.
    """

    state = {

        "file_hash": file_hash,

        "filename": filename,

        "job_id": job_id,
    }


    temp_path = (
        ACTIVE_DOCUMENT_STATE
        .with_suffix(".tmp")
    )


    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            state,
            file,
            indent=2,
        )


    temp_path.replace(
        ACTIVE_DOCUMENT_STATE
    )


# ============================================================
# CACHE HELPERS
# ============================================================

def _get_complete_cache(
    file_path: Path,
) -> dict[str, Any] | None:
    """
    Return cache information when a complete successful cache
    exists for the exact PDF content.

    Returns:

        {
            "file_hash": ...,
            "cache_path": ...,
            "cache": ...
        }

    or None when no complete cache exists.
    """

    # --------------------------------------------------------
    # Calculate content hash
    # --------------------------------------------------------

    file_hash = calculate_file_hash(
        file_path
    )


    # --------------------------------------------------------
    # Get content-hash cache path
    # --------------------------------------------------------

    cache_path = get_output_path(
        file_hash
    )


    # --------------------------------------------------------
    # Load cache
    # --------------------------------------------------------

    cache = load_cache(
        cache_path
    )


    if cache is None:

        return None


    # --------------------------------------------------------
    # Validate cache identity
    # --------------------------------------------------------

    if cache.get(
        "file_hash"
    ) != file_hash:

        return None


    # --------------------------------------------------------
    # Validate total chunk count
    # --------------------------------------------------------

    total_chunks = cache.get(
        "total_chunks"
    )

    cached_chunks = cache.get(
        "chunks",
        [],
    )


    if not isinstance(
        total_chunks,
        int,
    ):

        return None


    if len(cached_chunks) != total_chunks:

        return None


    # --------------------------------------------------------
    # Every chunk must be successful
    # --------------------------------------------------------

    for chunk in cached_chunks:

        if chunk.get(
            "status"
        ) != "success":

            return None

        if not chunk.get(
            "text"
        ):

            return None

        if not isinstance(
            chunk.get(
                "triples",
                [],
            ),
            list,
        ):

            return None


    return {

        "file_hash": file_hash,

        "cache_path": cache_path,

        "cache": cache,

    }


# ============================================================
# ACTIVE DOCUMENT CHECK
# ============================================================

def _is_active_document(
    file_hash: str,
) -> bool:
    """
    Determine whether this PDF is already the active document.

    First checks in-memory runtime state.

    Then checks the persisted active-document state.
    """

    # --------------------------------------------------------
    # Runtime check
    # --------------------------------------------------------

    with runtime_lock:

        active_hash = runtime.get(
            "active_document_hash"
        )

        if active_hash == file_hash:

            return True


    # --------------------------------------------------------
    # Persisted state check
    # --------------------------------------------------------

    state = _load_active_document_state()


    if state is None:

        return False


    persisted_hash = state.get(
        "file_hash"
    )


    if persisted_hash != file_hash:

        return False


    # --------------------------------------------------------
    # Restore runtime state
    # --------------------------------------------------------

    with runtime_lock:

        runtime[
            "active_document_hash"
        ] = file_hash

        runtime[
            "active_document_filename"
        ] = state.get(
            "filename"
        )

        runtime[
            "active_job_id"
        ] = state.get(
            "job_id"
        )


    return True


# ============================================================
# MARK ACTIVE DOCUMENT
# ============================================================

def _mark_active_document(
    *,
    file_hash: str,
    filename: str,
    job_id: str,
) -> None:

    with runtime_lock:

        runtime[
            "active_document_hash"
        ] = file_hash

        runtime[
            "active_document_filename"
        ] = filename

        runtime[
            "active_job_id"
        ] = job_id


    _save_active_document_state(
        file_hash=file_hash,
        filename=filename,
        job_id=job_id,
    )


# ============================================================
# NORMAL PIPELINE JOB
# ============================================================

def _ingest_job(
    job_id: str,
    file_path: Path,
    original_filename: str,
    file_hash: str,
) -> None:

    _update_job(
        job_id,
        status="running",
        progress=10,
        message=(
            "PDF saved. Building chunks, extracting triples, "
            "resolving entities, and updating Neo4j..."
        ),
    )


    try:

        result = run_pipeline(
            pdf_path=file_path,
            clear_existing=True,
        )


        # ----------------------------------------------------
        # Mark this document as active
        # ----------------------------------------------------

        _mark_active_document(
            file_hash=file_hash,
            filename=original_filename,
            job_id=job_id,
        )


        _update_job(
            job_id,
            status="completed",
            progress=100,
            message=(
                "Ingestion completed. "
                "The PDF is ready for questions."
            ),
            file_hash=file_hash,
            cache_hit=(
                _get_complete_cache(
                    file_path
                )
                is not None
            ),
            result={
                "filename": original_filename,
                "processed": result.get(
                    "processed"
                ),
                "normalized": result.get(
                    "normalized"
                ),
                "resolved": result.get(
                    "resolved"
                ),
            },
        )


    except Exception as exc:

        _update_job(
            job_id,
            status="failed",
            progress=100,
            message="PDF ingestion failed.",
            error=str(exc),
        )


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/api/health"
)
def health() -> dict[str, str]:

    return {

        "status": "ok",

        "service": "kg-rag",
    }


# ============================================================
# UPLOAD
# ============================================================

@app.post(
    "/api/upload"
)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict[str, str]:

    # --------------------------------------------------------
    # Original filename
    # --------------------------------------------------------

    filename = (
        file.filename
        or "document.pdf"
    )


    # --------------------------------------------------------
    # Validate extension
    # --------------------------------------------------------

    if not filename.lower().endswith(
        ".pdf"
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF files are supported."
            ),
        )


    # --------------------------------------------------------
    # Job ID
    # --------------------------------------------------------

    job_id = uuid4().hex


    # --------------------------------------------------------
    # Safe filename
    # --------------------------------------------------------

    safe_stem = _sanitize_filename(
        filename
    )


    saved_path = (
        UPLOAD_DIR
        / f"{job_id}_{safe_stem}.pdf"
    )


    # --------------------------------------------------------
    # Save upload
    # --------------------------------------------------------

    total_bytes = 0


    try:

        with saved_path.open(
            "wb"
        ) as output:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )


                if not chunk:

                    break


                total_bytes += len(
                    chunk
                )


                if total_bytes > MAX_UPLOAD_BYTES:

                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"PDF is larger than the "
                            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} "
                            f"MB limit."
                        ),
                    )


                output.write(
                    chunk
                )


    except HTTPException:

        saved_path.unlink(
            missing_ok=True
        )

        raise


    except Exception as exc:

        saved_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not save PDF: {exc}"
            ),
        ) from exc


    finally:

        await file.close()


    # --------------------------------------------------------
    # Calculate PDF hash
    # --------------------------------------------------------

    try:

        file_hash = calculate_file_hash(
            saved_path
        )

    except Exception as exc:

        saved_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not calculate PDF hash: {exc}"
            ),
        ) from exc


    # --------------------------------------------------------
    # Create initial job
    # --------------------------------------------------------

    with jobs_lock:

        jobs[job_id] = {

            "job_id": job_id,

            "filename": filename,

            "status": "queued",

            "progress": 0,

            "message": (
                "Upload received. "
                "Checking document cache..."
            ),

            "file_hash": file_hash,

        }


    # ========================================================
    # CACHE CHECK
    # ========================================================

    complete_cache = _get_complete_cache(
        saved_path
    )


    # --------------------------------------------------------
    # Same PDF already active
    # --------------------------------------------------------

    if (
        complete_cache is not None
        and _is_active_document(
            file_hash
        )
    ):

        print()
        print("=" * 80)
        print("CACHE HIT — ACTIVE DOCUMENT")
        print("=" * 80)
        print(
            f"Filename:   {filename}"
        )
        print(
            f"File hash:  {file_hash}"
        )
        print(
            "Existing Neo4j document is already active."
        )
        print(
            "Skipping PDF processing, extraction, "
            "entity resolution, and Neo4j ingestion."
        )
        print(
            "Going directly to Q/A."
        )


        _mark_active_document(
            file_hash=file_hash,
            filename=filename,
            job_id=job_id,
        )


        _update_job(
            job_id,
            status="completed",
            progress=100,
            message=(
                "Document already processed and "
                "active. Ready for questions."
            ),
            cache_hit=True,
            reused_active_document=True,
            result={
                "filename": filename,
                "file_hash": file_hash,
                "cache_path": str(
                    complete_cache[
                        "cache_path"
                    ]
                ),
            },
        )


        return {

            "job_id": job_id,

            "filename": filename,

            "status": "completed",

        }


    # --------------------------------------------------------
    # Cache exists but document is not active
    # --------------------------------------------------------

    if complete_cache is not None:

        print()
        print("=" * 80)
        print("CACHE HIT — DOCUMENT NOT CURRENTLY ACTIVE")
        print("=" * 80)
        print(
            f"Filename:   {filename}"
        )
        print(
            f"File hash:  {file_hash}"
        )
        print(
            "Cached Gemini extraction exists."
        )
        print(
            "The existing pipeline will run, but "
            "document extraction will reuse the cache."
        )


        _update_job(
            job_id,
            message=(
                "Complete extraction cache found. "
                "Preparing the document for Neo4j..."
            ),
            cache_hit=True,
        )


    # --------------------------------------------------------
    # No cache
    # --------------------------------------------------------

    else:

        print()
        print("=" * 80)
        print("CACHE MISS")
        print("=" * 80)
        print(
            f"Filename:   {filename}"
        )
        print(
            f"File hash:  {file_hash}"
        )
        print(
            "Full ingestion pipeline will run."
        )


        _update_job(
            job_id,
            message=(
                "No extraction cache found. "
                "Starting full ingestion..."
            ),
            cache_hit=False,
        )


    # --------------------------------------------------------
    # Run existing pipeline asynchronously
    # --------------------------------------------------------

    background_tasks.add_task(
        _ingest_job,
        job_id,
        saved_path,
        filename,
        file_hash,
    )


    return {

        "job_id": job_id,

        "filename": filename,

        "status": "queued",

    }


# ============================================================
# JOB STATUS
# ============================================================

@app.get(
    "/api/jobs/{job_id}"
)
def job_status(
    job_id: str,
) -> dict[str, Any]:

    with jobs_lock:

        job = jobs.get(
            job_id
        )


    if job is None:

        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )


    return job.copy()


# ============================================================
# QUERY
# ============================================================

@app.post(
    "/api/query"
)
def query(
    request: QueryRequest,
) -> dict[str, Any]:

    # --------------------------------------------------------
    # Find ingestion job
    # --------------------------------------------------------

    with jobs_lock:

        job = jobs.get(
            request.job_id
        )


    if job is None:

        raise HTTPException(
            status_code=404,
            detail="Ingestion job not found.",
        )


    # --------------------------------------------------------
    # Must be completed
    # --------------------------------------------------------

    if job.get(
        "status"
    ) != "completed":

        raise HTTPException(
            status_code=409,
            detail=(
                "The selected PDF is not ready yet."
            ),
        )


    # --------------------------------------------------------
    # Ask RAG
    # --------------------------------------------------------

    try:

        result = _get_rag_service().answer(
            request.question
        )

        # Add active document metadata to response.
        # This is useful for the frontend and debugging.

        if isinstance(
            result,
            dict,
        ):

            result["job_id"] = (
                request.job_id
            )

            result["document"] = {

                "filename": job.get(
                    "filename"
                ),

                "file_hash": job.get(
                    "file_hash"
                ),

            }


        return result


    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Question answering failed: {exc}"
            ),
        ) from exc


# ============================================================
# CURRENT DOCUMENT
# ============================================================

@app.get(
    "/api/current-document"
)
def current_document() -> dict[str, Any]:
    """
    Return the most recently completed active upload.

    Useful after browser refresh.
    """

    # --------------------------------------------------------
    # First try runtime state
    # --------------------------------------------------------

    with runtime_lock:

        active_hash = runtime.get(
            "active_document_hash"
        )

        active_filename = runtime.get(
            "active_document_filename"
        )

        active_job_id = runtime.get(
            "active_job_id"
        )


    # --------------------------------------------------------
    # Then persisted state
    # --------------------------------------------------------

    if not active_hash:

        state = (
            _load_active_document_state()
        )


        if state is not None:

            active_hash = state.get(
                "file_hash"
            )

            active_filename = state.get(
                "filename"
            )

            active_job_id = state.get(
                "job_id"
            )


    if not active_hash:

        return {
            "ready": False
        }


    # --------------------------------------------------------
    # Return active document
    # --------------------------------------------------------

    return {

        "ready": True,

        "job_id": active_job_id,

        "filename": active_filename,

        "file_hash": active_hash,

        "status": "completed",

    }


# ============================================================
# SHUTDOWN
# ============================================================

@app.on_event(
    "shutdown"
)
def shutdown_event() -> None:

    service = runtime.get(
        "rag_service"
    )


    if service is not None:

        service.close()

        runtime[
            "rag_service"
        ] = None