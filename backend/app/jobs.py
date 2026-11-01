"""
In-memory job store.

Each job gets its own folder (data/jobs/<job_id>/) that holds every file the pipeline
produces. A job.json snapshot is written on every update so you can inspect a job's
state on disk while debugging.

Jobs live in memory, so they are forgotten when the server restarts. That is fine for a
single-machine demo and keeps the project free of a database.
"""

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.schemas import JobStage, JobStatus


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Job:
    id: str
    dir: Path
    prompt: str | None
    status: JobStatus = JobStatus.queued
    stage: JobStage = JobStage.received
    progress: int = 0
    message: str = "Waiting to start"
    error: str | None = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    finished_at: datetime | None = None
    # artifact name -> file name inside the job folder
    artifacts: dict[str, str] = field(default_factory=dict)
    # extra information produced by pipeline stages (sizes, scores, model used, ...)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_seconds(self) -> float:
        end = self.finished_at or _now()
        return round((end - self.created_at).total_seconds(), 1)


class JobStore:
    def __init__(self, jobs_dir: Path) -> None:
        self._jobs_dir = jobs_dir
        self._jobs: dict[str, Job] = {}
        # Pipeline steps run in worker threads, so updates must be thread-safe.
        self._lock = threading.Lock()

    def create(self, prompt: str | None) -> Job:
        job_id = uuid.uuid4().hex
        job_dir = self._jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        job = Job(id=job_id, dir=job_dir, prompt=prompt)
        with self._lock:
            self._jobs[job_id] = job
            self._write_snapshot(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **changes: Any) -> Job:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in changes.items():
                if not hasattr(job, key):
                    raise AttributeError(f"Job has no field '{key}'")
                setattr(job, key, value)
            job.updated_at = _now()
            if job.status in (JobStatus.completed, JobStatus.failed) and job.finished_at is None:
                job.finished_at = job.updated_at
            self._write_snapshot(job)
            return job

    def add_artifact(self, job_id: str, name: str, filename: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.artifacts[name] = filename
            job.updated_at = _now()
            self._write_snapshot(job)

    def add_details(self, job_id: str, **details: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.details.update(details)
            job.updated_at = _now()
            self._write_snapshot(job)

    @staticmethod
    def _write_snapshot(job: Job) -> None:
        data = asdict(job)
        data["dir"] = str(job.dir)
        data["elapsed_seconds"] = job.elapsed_seconds
        (job.dir / "job.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


job_store = JobStore(settings.jobs_dir)
