# Sketch2Shape AI: HTTP API

Base URL (local): `http://localhost:8000`. Interactive docs (Swagger UI): `http://localhost:8000/docs`.
All responses are JSON. Errors use FastAPI's format: `{"detail": "human-readable message"}`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Server, image-model and 3D-provider status |
| `POST` | `/api/generate` | Upload a sketch (+ optional prompt) and start a job |
| `GET` | `/api/status/{job_id}` | Job progress, result files and details (poll every 3 s) |
| `GET` | `/files/jobs/{job_id}/{file}` | Download any file a job produced (e.g. `model.glb`) |

---

## `GET /api/health`

```json
{
  "status": "ok",
  "image_model": { "status": "ready", "device": "mps", "error": null },
  "reconstruction": {
    "order": ["trellis", "triposr"],
    "trellis": { "space": "trellis-community/TRELLIS", "hf_token_set": true, "quota_cooldown_seconds": 0 },
    "triposr": { "installed": true, "loaded": false, "device": null }
  }
}
```

`image_model.status` is one of `not_loaded`, `loading`, `ready`, `error`. Jobs submitted while it is `loading`
simply wait. `quota_cooldown_seconds > 0` means the Space ran out of GPU quota recently and is being skipped.

## `POST /api/generate`

`multipart/form-data` fields:

| Field | Required | Rules |
|---|---|---|
| `sketch` | yes | PNG or JPEG, ≤ 10 MB (`MAX_UPLOAD_MB`), must decode as an image |
| `prompt` | no | ≤ 500 characters (`MAX_PROMPT_CHARS`) |

**202 Accepted**

```json
{ "job_id": "2765173cc485416b9f1451c1d0ce55b2", "status": "queued" }
```

| Error | When |
|---|---|
| `400` | Empty file, file is not a readable image, prompt too long |
| `413` | File larger than the limit |
| `415` | Not PNG/JPEG |
| `422` | `sketch` field missing |

```bash
curl -F "sketch=@my_sketch.png" -F "prompt=a white ceramic mug" http://localhost:8000/api/generate
```

## `GET /api/status/{job_id}`

Returns `404` (`"Job not found. Jobs are kept in memory, so they are lost if the server restarts."`) for unknown
IDs, including every job from before a server restart.

| Field | Type | Meaning |
|---|---|---|
| `status` | `queued` · `processing` · `completed` · `failed` | Overall state. Stop polling on `completed` or `failed`. |
| `stage` | `received` · `preprocessing` · `understanding` · `generating_image` · `removing_background` · `reconstructing` · `validating` · `done` | Current pipeline step |
| `progress` | 0–100 | Rough percentage (10 / 20 / 30–55 / 58 / 62–90 / 92 / 100) |
| `message` | string | Live human-readable message (e.g. `Generating a reference image (step 12/30)`, `Waiting for a free GPU on Hugging Face (position 2 of 4)`) |
| `error` | string or null | User-facing reason when `status` is `failed` |
| `prompt` | string or null | The user's description |
| `created_at`, `updated_at` | ISO date-times | |
| `elapsed_seconds` | number | Seconds since the job was created (frozen when it ends) |
| `artifacts` | object name → absolute URL | Files available so far (see below) |
| `details` | object | Per-stage results: `preprocessing`, `enrichment`, `generation`, `background_removal`, `reconstruction`, `validation`, `warnings`, `timings_seconds` |

Artifact names: `input`, `sketch_clean`, `enrichment`, `control_image`, `generated_image`, `foreground`,
`foreground_white`, `model_raw`, `preview_video` (TRELLIS only), **`model_glb`** (final asset), `validation`.

**Example (completed job, shortened)**

```json
{
  "job_id": "2765173cc485416b9f1451c1d0ce55b2",
  "status": "completed",
  "stage": "done",
  "progress": 100,
  "message": "Finished in 163.3s",
  "error": null,
  "prompt": null,
  "elapsed_seconds": 163.3,
  "artifacts": {
    "input": "http://localhost:8000/files/jobs/2765…55b2/input.png",
    "generated_image": "http://localhost:8000/files/jobs/2765…55b2/generated.png",
    "foreground": "http://localhost:8000/files/jobs/2765…55b2/foreground.png",
    "model_glb": "http://localhost:8000/files/jobs/2765…55b2/model.glb",
    "validation": "http://localhost:8000/files/jobs/2765…55b2/validation.json"
  },
  "details": {
    "enrichment": { "source": "llm", "object_category": "mug", "sketch_detail": "moderate", "controlnet_scale": 0.7 },
    "generation": { "device": "mps", "steps": 30, "seed": 1949861380 },
    "reconstruction": {
      "provider": "trellis",
      "provider_label": "TRELLIS (Hugging Face Space)",
      "seconds": 42.82,
      "attempts": [{ "provider": "trellis", "ok": true, "seconds": 42.82 }]
    },
    "validation": {
      "valid": true, "used_original": false, "watertight": false,
      "after": { "vertices": 10159, "faces": 17198, "textured_meshes": 1, "texture_sizes": [[1024, 1024]] },
      "extents": [1.0, 0.7688, 0.6938], "file_size_bytes": 1531596,
      "warnings": ["Mesh is not watertight (normal for single-view models; fine for viewing and most engines)."]
    },
    "timings_seconds": {
      "preprocessing": 0.04, "understanding": 1.28, "generating_image": 116.13,
      "removing_background": 2.19, "reconstructing": 42.82, "validating": 0.88
    }
  }
}
```

**Failure example**

```json
{
  "status": "failed",
  "stage": "understanding",
  "message": "Generation failed",
  "error": "The input is a photograph of a person, not a sketch or drawing of an object. Please draw or upload a sketch of a single object."
}
```

When the 3D step falls back, `details.reconstruction.attempts` lists each provider tried, with its error (for
example a ZeroGPU quota message), and a line is added to `details.warnings`.

## `GET /files/jobs/{job_id}/{file}`

Static file serving for everything in `backend/data/jobs/<job_id>/`, with CORS enabled for the frontend origin.
The frontend downloads `model.glb` through this route and saves it as `sketch2shape-<object>.glb`.
