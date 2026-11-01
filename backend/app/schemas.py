"""Request/response models and job enums shared by the routes and the pipeline."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class JobStage(str, Enum):
    """The pipeline step a job is currently in. New stages are added as the pipeline grows."""

    received = "received"
    preprocessing = "preprocessing"
    understanding = "understanding"
    generating_image = "generating_image"
    removing_background = "removing_background"
    reconstructing = "reconstructing"
    validating = "validating"
    done = "done"


class GenerateResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    stage: JobStage
    progress: int
    message: str
    error: str | None
    prompt: str | None
    created_at: datetime
    updated_at: datetime
    elapsed_seconds: float
    artifacts: dict[str, str]
    details: dict[str, Any]
