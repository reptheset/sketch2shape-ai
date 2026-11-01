"""HTTP endpoints. Handlers stay thin: validate input, talk to the job store, schedule work."""

import asyncio

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile, status

from app.config import settings
from app.jobs import Job, job_store
from app.pipeline import run_pipeline
from app.schemas import GenerateResponse, JobStatusResponse
from app.services.image_generation import model_status
from app.services.reconstruction import provider_status
from app.services.image_preprocessing import InvalidImageError, decode_image

router = APIRouter(prefix="/api")

ALLOWED_TYPES = {"image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg"}
READ_CHUNK = 1024 * 1024


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    """Reads the upload in chunks and stops as soon as it exceeds `limit` bytes."""
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(READ_CHUNK):
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds the {limit // (1024 * 1024)} MB limit.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _to_response(job: Job, request: Request) -> JobStatusResponse:
    artifacts = {
        name: str(request.url_for("files", path=f"jobs/{job.id}/{filename}"))
        for name, filename in job.artifacts.items()
    }
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        stage=job.stage,
        progress=job.progress,
        message=job.message,
        error=job.error,
        prompt=job.prompt,
        created_at=job.created_at,
        updated_at=job.updated_at,
        elapsed_seconds=job.elapsed_seconds,
        artifacts=artifacts,
        details=job.details,
    )


@router.get("/health")
async def health() -> dict[str, object]:
    """Server status, local image model state (loading/ready/error) and 3D provider setup."""
    return {"status": "ok", "image_model": model_status(), "reconstruction": provider_status()}


@router.post("/generate", response_model=GenerateResponse, status_code=status.HTTP_202_ACCEPTED)
async def generate(
    background_tasks: BackgroundTasks,
    sketch: UploadFile = File(..., description="Sketch image (PNG or JPG)"),
    prompt: str | None = Form(None, description="Optional description of the object"),
) -> GenerateResponse:
    extension = ALLOWED_TYPES.get((sketch.content_type or "").lower())
    if extension is None:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{sketch.content_type}'. Please upload a PNG or JPG image.",
        )

    data = await _read_limited(sketch, settings.max_upload_bytes)
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The uploaded file is empty.")

    # Decode now so a broken or tiny image is rejected immediately, not minutes later.
    try:
        await asyncio.to_thread(decode_image, data)
    except InvalidImageError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    clean_prompt = (prompt or "").strip() or None
    if clean_prompt and len(clean_prompt) > settings.max_prompt_chars:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The description is too long (maximum {settings.max_prompt_chars} characters).",
        )

    job = job_store.create(prompt=clean_prompt)
    input_name = f"input{extension}"
    (job.dir / input_name).write_bytes(data)
    job_store.add_artifact(job.id, "input", input_name)

    # Runs after the response is sent, so the client gets its job_id right away.
    background_tasks.add_task(run_pipeline, job.id)
    return GenerateResponse(job_id=job.id, status=job.status)


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str, request: Request) -> JobStatusResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found. Jobs are kept in memory, so they are lost if the server restarts.",
        )
    return _to_response(job, request)
