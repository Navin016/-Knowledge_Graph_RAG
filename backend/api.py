"""FastAPI layer for the KG-RAG application.

Version 1 intentionally keeps one active PDF in Neo4j at a time.
Uploading a new PDF replaces the current graph with that document.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.pipeline import run_pipeline
from backend.rag_service import RAGService


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "pdfs" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))

origins_raw = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
)
ALLOWED_ORIGINS = [
    item.strip()
    for item in origins_raw.split(",")
    if item.strip()
]

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


jobs: dict[str, dict[str, Any]] = {}
jobs_lock = Lock()
runtime: dict[str, RAGService | None] = {"rag_service": None}
runtime_lock = Lock()


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    job_id: str = Field(min_length=1)


def _update_job(job_id: str, **updates: Any) -> None:
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(updates)


def _get_rag_service() -> RAGService:
    with runtime_lock:
        service = runtime.get("rag_service")
        if service is None:
            service = RAGService()
            runtime["rag_service"] = service
        return service


def _sanitize_filename(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return stem or "document"


def _ingest_job(job_id: str, file_path: Path, original_filename: str) -> None:
    _update_job(
        job_id,
        status="running",
        progress=10,
        message="PDF saved. Building chunks, extracting triples, resolving entities, and updating Neo4j...",
    )

    try:
        result = run_pipeline(
            pdf_path=file_path,
            clear_existing=True,
        )

        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Ingestion completed. The PDF is ready for questions.",
            result={
                "filename": original_filename,
                "processed": result.get("processed"),
                "normalized": result.get("normalized"),
                "resolved": result.get("resolved"),
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


@app.get("/api/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "kg-rag",
    }


@app.post("/api/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict[str, str]:
    filename = file.filename or "document.pdf"

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    job_id = uuid4().hex
    safe_stem = _sanitize_filename(filename)
    saved_path = UPLOAD_DIR / f"{job_id}_{safe_stem}.pdf"

    total_bytes = 0

    try:
        with saved_path.open("wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"PDF is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
                    )

                output.write(chunk)
    except HTTPException:
        saved_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Could not save PDF: {exc}") from exc
    finally:
        await file.close()

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "filename": filename,
            "status": "queued",
            "progress": 0,
            "message": "Upload received. Waiting for ingestion to start...",
        }

    background_tasks.add_task(_ingest_job, job_id, saved_path, filename)

    return {
        "job_id": job_id,
        "filename": filename,
        "status": "queued",
    }


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    return job.copy()


@app.post("/api/query")
def query(request: QueryRequest) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(request.job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found.")

    if job.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail="The selected PDF is not ready yet.",
        )

    try:
        return _get_rag_service().answer(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Question answering failed: {exc}") from exc


@app.get("/api/current-document")
def current_document() -> dict[str, Any]:
    """Return the most recently completed upload, useful after browser refresh."""
    with jobs_lock:
        completed = [
            job for job in jobs.values()
            if job.get("status") == "completed"
        ]

    if not completed:
        return {"ready": False}

    latest = completed[-1]
    return {
        "ready": True,
        "job_id": latest["job_id"],
        "filename": latest["filename"],
        "status": latest["status"],
    }


@app.on_event("shutdown")
def shutdown_event() -> None:
    service = runtime.get("rag_service")
    if service is not None:
        service.close()
        runtime["rag_service"] = None
