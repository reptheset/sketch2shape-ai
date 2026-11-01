"use client";

// The finished asset: interactive 3D viewer, viewer controls, download button, model facts
// and the intermediate images that led to it.

import dynamic from "next/dynamic";
import { useState, type ReactNode } from "react";
import { DownloadIcon, FocusIcon, GridIcon, RotateIcon, SparkleIcon, SpinnerIcon } from "@/components/icons";
import { PreviewStrip, formatSeconds } from "@/components/ProgressCard";
import { downloadFile, type JobStatusResponse } from "@/lib/api";
import { cx } from "@/lib/cx";

const ModelViewer = dynamic(() => import("@/components/ModelViewer"), {
  ssr: false,
  loading: () => <div className="h-full w-full animate-pulse bg-slate-100/60" />,
});

function slugify(text: string): string {
  return (
    text
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 40) || "model"
  );
}

function formatBytes(bytes: number): string {
  return bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.round(bytes / 1024)} KB`;
}

function ToggleButton({
  active,
  onClick,
  label,
  children,
}: {
  active?: boolean;
  onClick: () => void;
  label: string;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      aria-pressed={active}
      className={cx(
        "flex h-9 w-9 items-center justify-center rounded-full border shadow-sm backdrop-blur transition",
        active
          ? "border-indigo-200 bg-indigo-600 text-white"
          : "border-white bg-white/80 text-slate-600 hover:bg-white hover:text-indigo-600",
      )}
    >
      {children}
    </button>
  );
}

type Props = {
  status: JobStatusResponse;
  onNewSketch: () => void;
};

export default function ResultCard({ status, onNewSketch }: Props) {
  const [autoRotate, setAutoRotate] = useState(true);
  const [wireframe, setWireframe] = useState(false);
  const [resetSignal, setResetSignal] = useState(0);
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  const modelUrl = status.artifacts.model_glb;
  const { enrichment, reconstruction, validation, warnings = [] } = status.details;
  const category = enrichment?.object_category || "3D model";
  const mesh = validation?.after;
  const texture = mesh?.textured_meshes
    ? `PBR texture ${mesh.texture_sizes?.[0]?.join("×") ?? ""}`.trim()
    : mesh?.vertex_colour_meshes
      ? "Vertex colours"
      : "Untextured";
  const extents = validation?.extents;

  const facts: [string, string][] = [
    ["3D engine", reconstruction?.provider_label ?? "—"],
    ["Triangles", mesh?.faces != null ? mesh.faces.toLocaleString() : "—"],
    ["Vertices", mesh?.vertices != null ? mesh.vertices.toLocaleString() : "—"],
    ["Material", texture],
    ["Size (W×H×D)", extents ? extents.map((x) => `${x.toFixed(2)}`).join(" × ") + " m" : "—"],
    ["File", validation?.file_size_bytes ? `GLB · ${formatBytes(validation.file_size_bytes)}` : "GLB"],
    ["Total time", formatSeconds(status.elapsed_seconds)],
  ];

  async function handleDownload() {
    if (!modelUrl) return;
    setDownloading(true);
    setDownloadError(null);
    try {
      await downloadFile(modelUrl, `sketch2shape-${slugify(category)}.glb`);
    } catch (err) {
      setDownloadError(err instanceof Error ? err.message : "Download failed.");
    } finally {
      setDownloading(false);
    }
  }

  return (
    <section className="animate-fade-up mt-8 overflow-hidden rounded-[28px] border border-white bg-white/85 shadow-[0_20px_60px_-20px_rgba(30,27,75,0.25)] backdrop-blur-xl">
      <div className="grid lg:grid-cols-[1.45fr_1fr]">
        {/* Viewer */}
        <div className="relative h-[420px] bg-linear-to-b from-slate-50 via-white to-indigo-50/60 sm:h-[500px]">
          {modelUrl ? (
            <ModelViewer url={modelUrl} autoRotate={autoRotate} wireframe={wireframe} resetSignal={resetSignal} />
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-slate-500">No model file was produced.</div>
          )}
          <div className="absolute left-4 top-4 flex items-center gap-2 rounded-full bg-emerald-500/95 px-3 py-1 text-xs font-semibold text-white shadow">
            <span className="h-1.5 w-1.5 rounded-full bg-white" /> Ready
          </div>
          <div className="absolute right-4 top-4 flex gap-2">
            <ToggleButton active={autoRotate} onClick={() => setAutoRotate((v) => !v)} label="Auto-rotate">
              <RotateIcon className="h-4 w-4" />
            </ToggleButton>
            <ToggleButton active={wireframe} onClick={() => setWireframe((v) => !v)} label="Wireframe">
              <GridIcon className="h-4 w-4" />
            </ToggleButton>
            <ToggleButton onClick={() => setResetSignal((n) => n + 1)} label="Reset view">
              <FocusIcon className="h-4 w-4" />
            </ToggleButton>
          </div>
          <p className="pointer-events-none absolute bottom-3 left-0 right-0 text-center text-xs text-slate-400">
            Drag to orbit · scroll to zoom · right-drag to pan
          </p>
        </div>

        {/* Details */}
        <div className="flex flex-col gap-5 border-t border-slate-100 p-6 lg:border-l lg:border-t-0">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-indigo-600">Your 3D asset</p>
            <h2 className="mt-1 font-display text-2xl font-semibold capitalize tracking-tight">{category}</h2>
            {enrichment?.object_description && (
              <p className="mt-1.5 line-clamp-3 text-sm leading-relaxed text-slate-500">{enrichment.object_description}</p>
            )}
          </div>

          <button
            type="button"
            onClick={handleDownload}
            disabled={!modelUrl || downloading}
            className="flex items-center justify-center gap-2 rounded-full bg-linear-to-r from-blue-600 to-indigo-600 px-6 py-3 text-[15px] font-semibold text-white shadow-lg shadow-indigo-500/30 transition hover:-translate-y-0.5 hover:shadow-xl hover:brightness-110 disabled:translate-y-0 disabled:opacity-60"
          >
            {downloading ? <SpinnerIcon className="h-4 w-4" /> : <DownloadIcon className="h-4 w-4" />}
            Download .glb
          </button>
          {downloadError && <p className="-mt-3 text-xs text-red-600">{downloadError}</p>}

          <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
            {facts.map(([label, value]) => (
              <div key={label} className="min-w-0">
                <dt className="text-xs text-slate-400">{label}</dt>
                <dd className="truncate font-medium text-slate-700" title={value}>
                  {value}
                </dd>
              </div>
            ))}
          </dl>

          {warnings.length > 0 && (
            <ul className="space-y-1 rounded-2xl bg-amber-50/80 p-3 text-xs leading-relaxed text-amber-800">
              {warnings.map((w) => (
                <li key={w}>• {w}</li>
              ))}
            </ul>
          )}

          <div className="mt-auto flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
            <button
              type="button"
              onClick={onNewSketch}
              className="flex items-center gap-2 rounded-full bg-slate-100 px-4 py-2 font-medium text-slate-700 transition hover:bg-indigo-50 hover:text-indigo-700"
            >
              <SparkleIcon className="h-4 w-4" /> New sketch
            </button>
            {status.artifacts.validation && (
              <a href={status.artifacts.validation} target="_blank" rel="noreferrer" className="text-xs font-medium text-indigo-600 hover:underline">
                Mesh report
              </a>
            )}
            {status.artifacts.preview_video && (
              <a href={status.artifacts.preview_video} target="_blank" rel="noreferrer" className="text-xs font-medium text-indigo-600 hover:underline">
                Turntable video
              </a>
            )}
          </div>
        </div>
      </div>

      <div className="border-t border-slate-100 p-6">
        <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">How it was made</p>
        <PreviewStrip status={status} />
      </div>
    </section>
  );
}
