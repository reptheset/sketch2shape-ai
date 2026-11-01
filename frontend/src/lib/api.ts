// All communication with the FastAPI backend lives in this file:
//   POST /api/generate          -> submitSketch()
//   GET  /api/status/{job_id}   -> getJobStatus()   (polled every 3 s by useJobStatus)

export const API_BASE_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/+$/, "");

/** Shape of the response from POST /api/generate (the contract Stage 2 implements). */
export type GenerateResponse = {
  job_id: string;
  status: string;
};

export class ApiError extends Error {
  status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/** FastAPI returns errors as {"detail": "..."} or, for validation errors, {"detail": [{msg: ...}]}. */
async function readErrorDetail(res: Response): Promise<string> {
  const fallback = `The server responded with status ${res.status}.`;
  try {
    const body: unknown = await res.json();
    if (body && typeof body === "object" && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string") return detail;
      if (Array.isArray(detail)) {
        const messages = detail
          .map((item) => (item && typeof item === "object" && "msg" in item ? String(item.msg) : null))
          .filter(Boolean);
        if (messages.length > 0) return messages.join("; ");
      }
    }
  } catch {
    // Response body was not JSON; use the fallback message.
  }
  return fallback;
}

/**
 * Sends the sketch (and optional prompt) to POST /api/generate as multipart/form-data.
 * The backend replies immediately with a job_id; the AI work happens in the background.
 */
export async function submitSketch(
  sketch: Blob,
  filename: string,
  prompt: string,
  signal?: AbortSignal,
): Promise<GenerateResponse> {
  const form = new FormData();
  form.append("sketch", sketch, filename);

  const trimmedPrompt = prompt.trim();
  if (trimmedPrompt) {
    form.append("prompt", trimmedPrompt);
  }

  let res: Response;
  try {
    // Do NOT set a Content-Type header: the browser adds multipart/form-data with the correct boundary.
    res = await fetch(`${API_BASE_URL}/api/generate`, {
      method: "POST",
      body: form,
      signal,
    });
  } catch (err) {
    if (err instanceof DOMException && (err.name === "TimeoutError" || err.name === "AbortError")) {
      throw new ApiError("The backend took too long to respond. Please try again.");
    }
    throw new ApiError(
      `Cannot reach the backend at ${API_BASE_URL}. Make sure the FastAPI server is running and CORS allows this page.`,
    );
  }

  if (!res.ok) {
    throw new ApiError(await readErrorDetail(res), res.status);
  }

  const data: unknown = await res.json().catch(() => null);
  if (!data || typeof data !== "object" || typeof (data as GenerateResponse).job_id !== "string") {
    throw new ApiError("The backend response did not include a job_id.");
  }
  return data as GenerateResponse;
}

// ---------------------------------------------------------------- job status (Stage 8)

export type JobState = "queued" | "processing" | "completed" | "failed";

export type JobStage =
  | "received"
  | "preprocessing"
  | "understanding"
  | "generating_image"
  | "removing_background"
  | "reconstructing"
  | "validating"
  | "done";

/** Mirrors JobStatusResponse in backend/app/schemas.py. Artifact values are absolute URLs. */
export type JobStatusResponse = {
  job_id: string;
  status: JobState;
  stage: JobStage;
  progress: number;
  message: string;
  error: string | null;
  prompt: string | null;
  created_at: string;
  updated_at: string;
  elapsed_seconds: number;
  artifacts: Partial<
    Record<
      | "input"
      | "sketch_clean"
      | "enrichment"
      | "control_image"
      | "generated_image"
      | "foreground"
      | "foreground_white"
      | "model_raw"
      | "preview_video"
      | "model_glb"
      | "validation",
      string
    >
  >;
  details: JobDetails;
};

export type JobDetails = {
  warnings?: string[];
  enrichment?: { object_category?: string; object_description?: string; source?: string };
  generation?: { device?: string; seed?: number; inference_seconds?: number };
  reconstruction?: {
    provider?: string;
    provider_label?: string;
    seconds?: number;
    attempts?: { provider: string; ok: boolean; error?: string; skipped?: boolean }[];
  };
  validation?: {
    watertight?: boolean;
    used_original?: boolean;
    extents?: number[];
    file_size_bytes?: number;
    after?: { vertices?: number; faces?: number; textured_meshes?: number; vertex_colour_meshes?: number; texture_sizes?: number[][] };
  };
  timings_seconds?: Record<string, number>;
  [key: string]: unknown;
};

export async function getJobStatus(jobId: string, signal?: AbortSignal): Promise<JobStatusResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/api/status/${encodeURIComponent(jobId)}`, { signal, cache: "no-store" });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError(`Cannot reach the backend at ${API_BASE_URL}.`);
  }
  if (!res.ok) {
    throw new ApiError(await readErrorDetail(res), res.status);
  }
  return (await res.json()) as JobStatusResponse;
}

/**
 * Downloads a file from the backend and saves it with a friendly name. A plain
 * <a download> is ignored for cross-origin URLs (frontend :3000, backend :8000), so the
 * file is fetched as a blob first.
 */
export async function downloadFile(url: string, filename: string): Promise<void> {
  const res = await fetch(url);
  if (!res.ok) throw new ApiError(`Download failed (status ${res.status}).`, res.status);
  const blobUrl = URL.createObjectURL(await res.blob());
  const link = document.createElement("a");
  link.href = blobUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(blobUrl), 10_000);
}
