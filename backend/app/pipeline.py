"""
The generation pipeline that runs in the background for each job.

POST /api/generate stores the upload, creates a job and schedules run_pipeline(job_id).
The HTTP response returns immediately; this coroutine then works through the stages and
records progress in the job store, which GET /api/status/{job_id} reads.

Stages implemented so far:
  2. preprocessing:  OpenCV clean-up of the sketch          (services/image_preprocessing.py)
  3. understanding:  vision-LLM analysis + SDXL prompt      (services/semantic_enrichment.py)
  4. generating_image: local SDXL + ControlNet Scribble       (services/image_generation.py)
  5. removing_background: rembg U²-Net + mask clean-up        (services/background_removal.py)
  6. reconstructing:  TRELLIS (free HF Space) -> TripoSR local fallback (services/reconstruction.py)
  7. validating:      trimesh checks + material-safe clean-up  (services/mesh_validation.py)
"""

import asyncio
import json
import logging
import math
import time

from app.config import settings
from app.errors import PipelineError
from app.jobs import job_store
from app.schemas import JobStage, JobStatus
from app.services.background_removal import remove_background
from app.services.image_generation import generate_image
from app.services.image_preprocessing import InvalidImageError, preprocess_sketch
from app.services.mesh_validation import validate_mesh
from app.services.reconstruction import reconstruct_3d
from app.services.semantic_enrichment import enrich_sketch

logger = logging.getLogger("sketch2shape.pipeline")


async def run_pipeline(job_id: str) -> None:
    job = job_store.get(job_id)
    if job is None:
        logger.error("Job %s vanished before the pipeline started", job_id)
        return

    started = time.perf_counter()
    timings: dict[str, float] = {}
    logger.info("[%s] pipeline started", job_id)

    try:
        # ---- Stage 2: preprocessing (CPU-bound, so it runs in a worker thread and the
        # server stays responsive to status polls).
        t = time.perf_counter()
        job_store.update(
            job_id,
            status=JobStatus.processing,
            stage=JobStage.preprocessing,
            progress=10,
            message="Cleaning up your sketch",
        )
        input_path = job.dir / job.artifacts["input"]
        clean_path = job.dir / "sketch_clean.png"
        try:
            prep = await asyncio.to_thread(preprocess_sketch, input_path, clean_path, settings.sketch_size)
        except InvalidImageError as exc:
            raise PipelineError(str(exc)) from exc
        job_store.add_artifact(job_id, "sketch_clean", clean_path.name)
        job_store.add_details(
            job_id,
            preprocessing={
                "original_size": [prep.original_width, prep.original_height],
                "output_size": [prep.width, prep.height],
                "ink_ratio": prep.ink_ratio,
                "stroke_components": prep.stroke_components,
                "inverted": prep.inverted,
            },
        )
        timings["preprocessing"] = round(time.perf_counter() - t, 2)
        logger.info("[%s] preprocessing done (ink ratio %.3f)", job_id, prep.ink_ratio)

        # ---- Stage 3: semantic enrichment (network I/O, awaited directly).
        t = time.perf_counter()
        job_store.update(job_id, stage=JobStage.understanding, progress=20, message="Understanding your sketch")
        enrichment = await enrich_sketch(input_path, clean_path, job.prompt, prep.ink_ratio)
        enrichment_path = job.dir / "enrichment.json"
        enrichment_path.write_text(json.dumps(enrichment.to_dict(), indent=2), encoding="utf-8")
        job_store.add_artifact(job_id, "enrichment", enrichment_path.name)
        job_store.add_details(job_id, enrichment=enrichment.to_dict())
        if enrichment.warnings:
            job_store.add_details(job_id, warnings=list(enrichment.warnings))
        timings["understanding"] = round(time.perf_counter() - t, 2)
        logger.info(
            "[%s] enrichment done via %s: %s (detail=%s, controlnet_scale=%.2f)",
            job_id,
            enrichment.source,
            enrichment.object_category,
            enrichment.sketch_detail,
            enrichment.controlnet_scale,
        )

        # ---- Stage 4: sketch-guided image generation (local SDXL, runs in a worker thread).
        t = time.perf_counter()
        job_store.update(job_id, stage=JobStage.generating_image, progress=30, message="Generating a reference image")

        def on_progress(step: int, total: int) -> None:
            # Called from the generation thread after every denoising step (job_store is thread-safe).
            job_store.update(
                job_id,
                progress=30 + int(25 * step / total),
                message=f"Generating a reference image (step {step}/{total})",
            )

        generated_path = job.dir / "generated.png"
        generation = await generate_image(
            clean_path,
            generated_path,
            enrichment.image_prompt,
            enrichment.negative_prompt,
            enrichment.controlnet_scale,
            on_progress=on_progress,
        )
        job_store.add_artifact(job_id, "control_image", "control.png")
        job_store.add_artifact(job_id, "generated_image", generated_path.name)
        job_store.add_details(job_id, generation=generation.to_dict())
        timings["generating_image"] = round(time.perf_counter() - t, 2)
        logger.info(
            "[%s] image generated on %s (seed %s, %.1fs)",
            job_id,
            generation.device,
            generation.seed,
            generation.inference_seconds,
        )

        # ---- Stage 5: background removal (CPU-bound, runs in a worker thread).
        t = time.perf_counter()
        job_store.update(job_id, stage=JobStage.removing_background, progress=58, message="Isolating the object")
        foreground = await remove_background(
            generated_path, job.dir / "foreground.png", job.dir / "foreground_white.png"
        )
        job_store.add_artifact(job_id, "foreground", foreground.rgba_path.name)
        job_store.add_artifact(job_id, "foreground_white", foreground.white_path.name)
        job_store.add_details(job_id, background_removal=foreground.to_dict())
        timings["removing_background"] = round(time.perf_counter() - t, 2)
        logger.info(
            "[%s] background removed (coverage %.2f, kept %d of %d regions)",
            job_id,
            foreground.object_coverage,
            foreground.regions_kept,
            foreground.regions_found,
        )

        # ---- Stage 6: 3D reconstruction (network or local GPU work, runs in a worker thread).
        t = time.perf_counter()
        job_store.update(job_id, stage=JobStage.reconstructing, progress=62, message="Building the 3D model")
        def on_recon_status(message: str) -> None:
            job_store.update(job_id, message=message)

        async def tick_progress() -> None:
            # Providers cannot report exact progress, so the bar eases from 62% towards 90%
            # (about two thirds of the way after a minute) until reconstruction returns.
            while True:
                await asyncio.sleep(2)
                waited = time.perf_counter() - t
                job_store.update(job_id, progress=62 + int(28 * (1 - math.exp(-waited / 60))))

        ticker = asyncio.create_task(tick_progress())
        try:
            reconstruction = await reconstruct_3d(
                job.dir / "foreground.png", job.dir / "model_raw.glb", generation.seed, on_recon_status
            )
        finally:
            ticker.cancel()
        job_store.add_artifact(job_id, "model_raw", reconstruction.glb_path.name)
        if reconstruction.preview_video:
            job_store.add_artifact(job_id, "preview_video", reconstruction.preview_video)
        job_store.add_details(job_id, reconstruction=reconstruction.to_dict())
        if len(reconstruction.attempts) > 1:
            job_store.add_details(
                job_id,
                warnings=list(job.details.get("warnings", []))
                + [f"3D model made with {reconstruction.provider_label} (the first choice was unavailable)."],
            )
        timings["reconstructing"] = round(time.perf_counter() - t, 2)
        logger.info(
            "[%s] 3D model built with %s in %.1fs (%d bytes)",
            job_id,
            reconstruction.provider,
            reconstruction.seconds,
            reconstruction.file_size_bytes,
        )

        # ---- Stage 7: mesh validation and clean-up (CPU-bound, runs in a worker thread).
        t = time.perf_counter()
        job_store.update(job_id, stage=JobStage.validating, progress=92, message="Checking and optimising the 3D model")
        validation = await validate_mesh(
            reconstruction.glb_path, job.dir / "model.glb", job.dir / "validation.json"
        )
        job_store.add_artifact(job_id, "model_glb", validation.glb_path.name)
        job_store.add_artifact(job_id, "validation", "validation.json")
        job_store.add_details(job_id, validation=validation.to_dict())
        timings["validating"] = round(time.perf_counter() - t, 2)
        logger.info(
            "[%s] mesh valid: %d vertices, %d faces, watertight=%s, textured=%d, vertex-coloured=%d%s",
            job_id,
            validation.after.vertices,
            validation.after.faces,
            validation.watertight,
            validation.after.textured_meshes,
            validation.after.vertex_colour_meshes,
            " (original file kept)" if validation.used_original else "",
        )

        elapsed = time.perf_counter() - started
        job_store.add_details(job_id, timings_seconds=timings)
        job_store.update(
            job_id,
            status=JobStatus.completed,
            stage=JobStage.done,
            progress=100,
            message=f"Finished in {elapsed:.1f}s",
        )
        logger.info("[%s] pipeline completed in %.1fs", job_id, elapsed)

    except PipelineError as exc:
        logger.warning("[%s] pipeline failed: %s", job_id, exc)
        job_store.add_details(job_id, timings_seconds=timings)
        job_store.update(job_id, status=JobStatus.failed, error=str(exc), message="Generation failed")
    except Exception:
        logger.exception("[%s] unexpected pipeline error", job_id)
        job_store.update(
            job_id,
            status=JobStatus.failed,
            error="An unexpected server error occurred. Check the backend logs for details.",
            message="Generation failed",
        )
