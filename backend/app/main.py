"""
FastAPI application entry point.

Run from the backend/ folder:
    uvicorn app.main:app --port 8000
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.errors import PipelineError
from app.routes import router
from app.services.image_generation import load_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

settings.jobs_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("sketch2shape")


async def _preload_models() -> None:
    try:
        await asyncio.to_thread(load_pipeline)
    except PipelineError as exc:
        logger.error("Model preload failed: %s", exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load SDXL in the background so the server is usable immediately; the first job
    # waits for loading only if it arrives before loading finishes.
    task = asyncio.create_task(_preload_models()) if settings.preload_models else None
    yield
    if task is not None and not task.done():
        task.cancel()


app = FastAPI(
    title="Sketch2Shape AI API",
    description="Turns a single-view 2D sketch into a textured 3D asset (.glb).",
    version="0.7.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Every file a job produces (cleaned sketch, generated image, GLB, ...) is served from here,
# e.g. http://localhost:8000/files/jobs/<job_id>/sketch_clean.png
app.mount("/files", StaticFiles(directory=settings.data_dir), name="files")

app.include_router(router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"name": "Sketch2Shape AI API", "docs": "/docs", "health": "/api/health"}
