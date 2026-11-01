"use client";

// Live view of a running job: status badge, progress bar, the six pipeline steps (done /
// active / failed / waiting), the backend's live message (e.g. "step 12/30" or the GPU
// queue position) and thumbnails of each intermediate image as soon as it exists.

import { useEffect, useRef, useState } from "react";
import { AlertIcon, CheckIcon, ClockIcon, RotateIcon, SpinnerIcon, XIcon } from "@/components/icons";
import type { JobStage, JobStatusResponse } from "@/lib/api";
import { cx } from "@/lib/cx";

const STEPS: { stage: JobStage; label: string }[] = [
  { stage: "preprocessing", label: "Cleaning up the sketch" },
  { stage: "understanding", label: "Understanding the sketch" },
  { stage: "generating_image", label: "Generating a reference image" },
  { stage: "removing_background", label: "Isolating the object" },
  { stage: "reconstructing", label: "Reconstructing 3D geometry" },
  { stage: "validating", label: "Validating the mesh" },
];

export const PREVIEW_IMAGES = [
  { key: "sketch_clean", label: "Clean sketch" },
  { key: "generated_image", label: "AI reference" },
  { key: "foreground", label: "Isolated object" },
] as const;

export function formatSeconds(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

/** Index of the step the job is in (-1 before the first step, STEPS.length when done). */
function currentStepIndex(stage: JobStage): number {
  if (stage === "done") return STEPS.length;
  return STEPS.findIndex((s) => s.stage === stage);
}

export function PreviewStrip({ status }: { status: JobStatusResponse }) {
  const available = PREVIEW_IMAGES.filter((p) => status.artifacts[p.key]);
  if (available.length === 0) return null;
  return (
    <div className="grid grid-cols-3 gap-3">
      {PREVIEW_IMAGES.map((p) => {
        const url = status.artifacts[p.key];
        return (
          <figure key={p.key} className="min-w-0">
            <div
              className={cx(
                "aspect-square overflow-hidden rounded-2xl border border-slate-100",
                p.key === "foreground" ? "checkerboard" : "bg-white",
              )}
            >
              {url ? (
                // eslint-disable-next-line @next/next/no-img-element
                <a href={url} target="_blank" rel="noreferrer">
                  <img src={url} alt={p.label} className="animate-fade-in h-full w-full object-contain" />
                </a>
              ) : (
                <div className="h-full w-full animate-pulse bg-slate-100/70" />
              )}
            </div>
            <figcaption className="mt-1.5 truncate text-center text-xs text-slate-500">{p.label}</figcaption>
          </figure>
        );
      })}
    </div>
  );
}

type Props = {
  jobId: string;
  status: JobStatusResponse | null;
  pollError: string | null;
  onRetry: () => void;
  onDismiss: () => void;
};

export default function ProgressCard({ jobId, status, pollError, onRetry, onDismiss }: Props) {
  const failed = status?.status === "failed" || Boolean(pollError);
  const active = !failed && status?.status !== "completed";

  // Elapsed time: the server's number at the last poll + seconds since that poll.
  const receivedAt = useRef(Date.now());
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    receivedAt.current = Date.now();
  }, [status]);
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [active]);
  const elapsed = status ? status.elapsed_seconds + (active ? (now - receivedAt.current) / 1000 : 0) : 0;

  const stepIndex = status ? currentStepIndex(status.stage) : -1;
  const progress = status?.progress ?? 0;
  const badge = pollError ? "offline" : (status?.status ?? "connecting");

  return (
    <section className="animate-fade-up mt-8 rounded-[28px] border border-white bg-white/85 p-6 shadow-[0_20px_60px_-20px_rgba(30,27,75,0.25)] backdrop-blur-xl">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="font-display text-xl font-semibold tracking-tight">
            {failed ? "Generation stopped" : "Building your 3D asset"}
          </h2>
          <p className="mt-1 truncate font-mono text-xs text-slate-400">job {jobId}</p>
        </div>
        <div className="flex items-center gap-2">
          <span className="flex items-center gap-1 rounded-full bg-slate-100 px-3 py-1 text-xs font-medium tabular-nums text-slate-500">
            <ClockIcon className="h-3.5 w-3.5" />
            {formatSeconds(elapsed)}
          </span>
          <span
            className={cx(
              "rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wider",
              failed ? "bg-red-50 text-red-600" : "bg-indigo-50 text-indigo-600",
            )}
          >
            {badge}
          </span>
        </div>
      </div>

      {/* Progress bar */}
      <div className="mt-5">
        <div className="h-2 overflow-hidden rounded-full bg-slate-100">
          <div
            className={cx(
              "relative h-full rounded-full transition-[width] duration-700 ease-out",
              failed ? "bg-red-400" : "bg-linear-to-r from-blue-500 via-indigo-500 to-purple-500",
            )}
            style={{ width: `${Math.max(progress, 3)}%` }}
          >
            {active && <div className="progress-shimmer absolute inset-0" />}
          </div>
        </div>
        <div className="mt-2 flex items-center justify-between gap-4 text-sm">
          <p className={cx("min-w-0 truncate", failed ? "text-red-600" : "text-slate-600")}>
            {pollError ? "Lost contact with the backend" : (status?.message ?? "Connecting to the backend…")}
          </p>
          <span className="shrink-0 font-medium tabular-nums text-slate-400">{progress}%</span>
        </div>
      </div>

      <ol className="mt-6 grid gap-3 sm:grid-cols-2">
        {STEPS.map((step, i) => {
          const isDone = i < stepIndex;
          const isCurrent = i === stepIndex;
          const isFailed = isCurrent && failed;
          const isActive = isCurrent && active;
          return (
            <li
              key={step.stage}
              className={cx(
                "flex items-center gap-3 rounded-2xl border px-4 py-3 text-sm transition-colors duration-500",
                isDone && "border-emerald-100 bg-emerald-50/70 text-emerald-800",
                isActive && "border-indigo-200 bg-indigo-50/80 text-indigo-900 shadow-[0_8px_24px_-12px_rgba(79,70,229,0.45)]",
                isFailed && "border-red-100 bg-red-50/80 text-red-700",
                !isDone && !isCurrent && "border-slate-100 bg-slate-50/70 text-slate-400",
              )}
            >
              <span
                className={cx(
                  "flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-semibold",
                  isDone && "bg-emerald-500 text-white",
                  isActive && "bg-indigo-600 text-white",
                  isFailed && "bg-red-500 text-white",
                  !isDone && !isCurrent && "bg-white text-slate-400 ring-1 ring-slate-200",
                )}
              >
                {isDone ? (
                  <CheckIcon className="h-3.5 w-3.5" />
                ) : isActive ? (
                  <SpinnerIcon className="h-3.5 w-3.5" />
                ) : isFailed ? (
                  <XIcon className="h-3.5 w-3.5" />
                ) : (
                  i + 1
                )}
              </span>
              <span className="min-w-0 truncate">{step.label}</span>
            </li>
          );
        })}
      </ol>

      {status && (
        <div className="mt-6">
          <PreviewStrip status={status} />
        </div>
      )}

      {failed && (
        <div role="alert" className="animate-fade-in mt-6 rounded-2xl border border-red-100 bg-red-50/90 p-4 text-sm text-red-700">
          <div className="flex items-start gap-3">
            <AlertIcon className="mt-0.5 h-5 w-5 shrink-0" />
            <p className="min-w-0 break-words">{pollError ?? status?.error ?? "Something went wrong."}</p>
          </div>
          <div className="mt-4 flex flex-wrap gap-2 pl-8">
            <button
              type="button"
              onClick={onRetry}
              className="flex items-center gap-2 rounded-full bg-red-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-red-700"
            >
              <RotateIcon className="h-4 w-4" /> Try again
            </button>
            <button
              type="button"
              onClick={onDismiss}
              className="rounded-full bg-white px-4 py-2 text-sm font-medium text-red-700 ring-1 ring-red-200 transition hover:bg-red-100"
            >
              Edit sketch or prompt
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
