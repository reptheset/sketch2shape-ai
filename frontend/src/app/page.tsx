"use client";

import { useCallback, useEffect, useRef, useState, type ChangeEvent, type FormEvent, type KeyboardEvent } from "react";
import ProgressCard from "@/components/ProgressCard";
import ResultCard from "@/components/ResultCard";
import Whiteboard from "@/components/Whiteboard";
import {
  AlertIcon,
  CheckIcon,
  CubeIcon,
  PencilIcon,
  SparkleIcon,
  SpinnerIcon,
  UploadIcon,
  XIcon,
} from "@/components/icons";
import { useJobStatus } from "@/hooks/useJobStatus";
import { submitSketch } from "@/lib/api";
import { cx } from "@/lib/cx";
import { ACCEPT_ATTRIBUTE, MAX_PROMPT_LENGTH, validateSketchFile } from "@/lib/sketch";
import { exportStrokesToPng, type Stroke } from "@/lib/whiteboard";

type AttachedSketch = {
  blob: Blob;
  filename: string;
  previewUrl: string;
  source: "upload" | "whiteboard";
};

const SUBMIT_TIMEOUT_MS = 30_000;

const SUGGESTIONS = [
  { emoji: "🪑", text: "Mid-century oak armchair with curved legs" },
  { emoji: "🧰", text: "Vintage wooden treasure chest with brass iron rims" },
  { emoji: "🏎️", text: "Futuristic low-poly cyberpunk concept car with a matte metallic finish", wide: true },
];

// The current job id is mirrored in the URL (?job=...), so a refresh or a shared link
// reopens the same job instead of losing it.
function setJobInUrl(jobId: string | null) {
  const url = new URL(window.location.href);
  if (jobId) url.searchParams.set("job", jobId);
  else url.searchParams.delete("job");
  window.history.replaceState(null, "", url);
}

export default function HomePage() {
  const [sketch, setSketch] = useState<AttachedSketch | null>(null);
  const [strokes, setStrokes] = useState<Stroke[]>([]);
  const [whiteboardOpen, setWhiteboardOpen] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobIdState] = useState<string | null>(null);
  const [isDraggingFile, setIsDraggingFile] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const formRef = useRef<HTMLFormElement>(null);
  const sketchRef = useRef<AttachedSketch | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  const { status: jobStatus, pollError } = useJobStatus(jobId);
  const jobRunning = Boolean(jobId) && !pollError && jobStatus?.status !== "completed" && jobStatus?.status !== "failed";
  const busy = isSubmitting || jobRunning;

  const setJobId = useCallback((next: string | null) => {
    setJobIdState(next);
    setJobInUrl(next);
  }, []);

  // Reopen a job from the URL on first load.
  useEffect(() => {
    const fromUrl = new URLSearchParams(window.location.search).get("job");
    if (fromUrl && /^[a-f0-9]{32}$/i.test(fromUrl)) setJobIdState(fromUrl);
  }, []);

  // Bring the progress card into view when a new job starts.
  useEffect(() => {
    if (jobId) resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [jobId]);

  // Swap the attached sketch and free the old preview URL.
  const replaceSketch = useCallback((next: AttachedSketch | null) => {
    if (sketchRef.current) URL.revokeObjectURL(sketchRef.current.previewUrl);
    sketchRef.current = next;
    setSketch(next);
  }, []);

  const attachFile = useCallback(
    (file: File) => {
      const validationError = validateSketchFile(file);
      if (validationError) {
        setError(validationError);
        return;
      }
      setError(null);
      replaceSketch({ blob: file, filename: file.name, previewUrl: URL.createObjectURL(file), source: "upload" });
    },
    [replaceSketch],
  );

  // Drop an image anywhere on the page, or paste one with ⌘V.
  useEffect(() => {
    let dragDepth = 0;
    const hasFiles = (e: DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes("Files");

    function onDragEnter(e: DragEvent) {
      if (!hasFiles(e)) return;
      dragDepth += 1;
      setIsDraggingFile(true);
    }
    function onDragOver(e: DragEvent) {
      if (hasFiles(e)) e.preventDefault();
    }
    function onDragLeave(e: DragEvent) {
      if (!hasFiles(e)) return;
      dragDepth = Math.max(0, dragDepth - 1);
      if (dragDepth === 0) setIsDraggingFile(false);
    }
    function onDrop(e: DragEvent) {
      if (!hasFiles(e)) return;
      e.preventDefault();
      dragDepth = 0;
      setIsDraggingFile(false);
      const file = e.dataTransfer?.files?.[0];
      if (file) attachFile(file);
    }
    function onPaste(e: ClipboardEvent) {
      const file = Array.from(e.clipboardData?.files ?? []).find((f) => f.type.startsWith("image/"));
      if (file) {
        e.preventDefault();
        attachFile(file);
      }
    }

    window.addEventListener("dragenter", onDragEnter);
    window.addEventListener("dragover", onDragOver);
    window.addEventListener("dragleave", onDragLeave);
    window.addEventListener("drop", onDrop);
    window.addEventListener("paste", onPaste);
    return () => {
      window.removeEventListener("dragenter", onDragEnter);
      window.removeEventListener("dragover", onDragOver);
      window.removeEventListener("dragleave", onDragLeave);
      window.removeEventListener("drop", onDrop);
      window.removeEventListener("paste", onPaste);
    };
  }, [attachFile]);

  // Auto-grow the prompt box.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }, [prompt]);

  function handleFileInput(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) attachFile(file);
    e.target.value = "";
  }

  async function handleUseWhiteboard(nextStrokes: Stroke[]) {
    setStrokes(nextStrokes);
    setWhiteboardOpen(false);
    const blob = await exportStrokesToPng(nextStrokes);
    if (!blob) {
      setError("Could not export the drawing. Please try again.");
      return;
    }
    setError(null);
    replaceSketch({ blob, filename: "whiteboard-sketch.png", previewUrl: URL.createObjectURL(blob), source: "whiteboard" });
  }

  function handleSuggestion(text: string) {
    setPrompt(text);
    textareaRef.current?.focus();
  }

  function handlePromptKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      formRef.current?.requestSubmit();
    }
  }

  async function generate() {
    if (busy) return;
    if (!sketch) {
      setError("Add a sketch first: open the whiteboard or upload an image.");
      return;
    }
    setError(null);
    setJobId(null);
    setIsSubmitting(true);
    try {
      const result = await submitSketch(sketch.blob, sketch.filename, prompt, AbortSignal.timeout(SUBMIT_TIMEOUT_MS));
      setJobId(result.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred.");
    } finally {
      setIsSubmitting(false);
    }
  }

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    generate();
  }

  function handleDismissJob() {
    setJobId(null);
    window.scrollTo({ top: 0, behavior: "smooth" });
    textareaRef.current?.focus();
  }

  function handleNewSketch() {
    setJobId(null);
    replaceSketch(null);
    setStrokes([]);
    setPrompt("");
    setError(null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-[#f7f7fb] font-sans text-slate-900">
      {/* Soft colour wash */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden="true">
        <div className="animate-drift absolute -left-40 -top-40 h-[520px] w-[520px] rounded-full bg-blue-300/30 blur-3xl" />
        <div className="animate-drift-slow absolute -right-32 top-10 h-[460px] w-[460px] rounded-full bg-purple-300/25 blur-3xl" />
        <div className="animate-drift absolute bottom-[-200px] left-1/3 h-[480px] w-[480px] rounded-full bg-pink-200/30 blur-3xl" />
      </div>

      <header className="relative z-10 mx-auto flex max-w-5xl items-center justify-between px-6 py-6">
        <div className="flex items-center gap-2.5">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-linear-to-br from-blue-600 to-purple-600 text-white shadow-md shadow-indigo-500/30">
            <CubeIcon className="h-5 w-5" />
          </span>
          <span className="font-display text-lg font-semibold tracking-tight">Sketch2Shape</span>
        </div>
      </header>

      <main className="relative z-10 mx-auto max-w-5xl px-6 pb-24 pt-10 sm:pt-16">
        <h1 className="animate-fade-up bg-linear-to-r from-blue-600 via-indigo-500 to-purple-500 bg-clip-text pb-2 font-display text-6xl font-semibold leading-[1.05] tracking-tight text-transparent sm:text-7xl">
          Hello, Creator.
        </h1>
        <p
          className="animate-fade-up mt-2 font-display text-3xl font-medium tracking-tight text-slate-400 sm:text-[2.6rem] sm:leading-tight"
          style={{ animationDelay: "80ms" }}
        >
          What 3D asset shall we sketch today?
        </p>

        {/* Suggestions */}
        <div className="animate-fade-up mt-12 grid gap-3 sm:grid-cols-2" style={{ animationDelay: "160ms" }}>
          {SUGGESTIONS.map((s) => (
            <button
              key={s.text}
              type="button"
              onClick={() => handleSuggestion(s.text)}
              className={cx(
                "group rounded-2xl border border-white bg-white/70 px-5 py-4 text-center text-[15px] text-slate-600 shadow-[0_2px_12px_-4px_rgba(30,27,75,0.12)] backdrop-blur transition hover:-translate-y-0.5 hover:border-indigo-100 hover:bg-white hover:text-slate-900 hover:shadow-[0_10px_30px_-10px_rgba(79,70,229,0.35)]",
                s.wide && "sm:col-span-2",
              )}
            >
              <span className="mr-2 inline-block transition group-hover:scale-110">{s.emoji}</span>
              {s.text}
            </button>
          ))}
        </div>

        {/* Composer */}
        <form
          ref={formRef}
          onSubmit={handleSubmit}
          className="animate-fade-up mt-8 rounded-[28px] border border-white bg-white/85 p-5 shadow-[0_20px_60px_-20px_rgba(30,27,75,0.25)] backdrop-blur-xl transition focus-within:shadow-[0_24px_70px_-20px_rgba(79,70,229,0.35)] sm:p-6"
          style={{ animationDelay: "240ms" }}
        >
          {sketch && (
            <div className="mb-4 flex items-center gap-4 animate-fade-in">
              <div className="relative">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={sketch.previewUrl}
                  alt="Attached sketch"
                  className="h-20 w-20 rounded-2xl border border-slate-200 bg-white object-contain p-1 shadow-sm"
                />
                <button
                  type="button"
                  onClick={() => replaceSketch(null)}
                  disabled={busy}
                  aria-label="Remove sketch"
                  className="absolute -right-2 -top-2 flex h-6 w-6 items-center justify-center rounded-full bg-slate-900 text-white shadow transition hover:bg-red-600 disabled:opacity-50"
                >
                  <XIcon className="h-3.5 w-3.5" />
                </button>
              </div>
              <div className="min-w-0">
                <p className="truncate text-sm font-medium text-slate-800">
                  {sketch.source === "whiteboard" ? "Whiteboard sketch" : sketch.filename}
                </p>
                <p className="mt-0.5 flex items-center gap-1 text-xs text-emerald-600">
                  <CheckIcon className="h-3.5 w-3.5" /> Ready to generate
                </p>
                {sketch.source === "whiteboard" && (
                  <button
                    type="button"
                    onClick={() => setWhiteboardOpen(true)}
                    className="mt-1 text-xs font-medium text-indigo-600 hover:underline"
                  >
                    Edit drawing
                  </button>
                )}
              </div>
            </div>
          )}

          <label htmlFor="prompt" className="sr-only">
            Describe your object (optional)
          </label>
          <textarea
            id="prompt"
            ref={textareaRef}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value.slice(0, MAX_PROMPT_LENGTH))}
            onKeyDown={handlePromptKeyDown}
            disabled={busy}
            rows={2}
            placeholder="Describe your object — materials, colours, style… (optional)"
            className="block w-full resize-none bg-transparent px-1 text-lg leading-relaxed text-slate-800 placeholder:text-slate-400 focus:outline-none disabled:opacity-60"
          />

          <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-slate-100 pt-4">
            <button
              type="button"
              onClick={() => setWhiteboardOpen(true)}
              disabled={busy}
              className="flex items-center gap-2 rounded-full bg-slate-100 px-4 py-2.5 text-sm font-medium text-slate-700 transition hover:bg-indigo-50 hover:text-indigo-700 disabled:opacity-50"
            >
              <PencilIcon className="h-4 w-4" />
              {strokes.length > 0 ? "Continue drawing" : "Open Whiteboard"}
            </button>
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={busy}
              className="flex items-center gap-2 rounded-full bg-slate-100 px-4 py-2.5 text-sm font-medium text-slate-700 transition hover:bg-indigo-50 hover:text-indigo-700 disabled:opacity-50"
            >
              <UploadIcon className="h-4 w-4" />
              Upload Sketch Image
            </button>
            <input ref={fileInputRef} type="file" accept={ACCEPT_ATTRIBUTE} onChange={handleFileInput} className="hidden" />

            <div className="ml-auto flex items-center gap-3">
              {prompt.length > MAX_PROMPT_LENGTH * 0.8 && (
                <span className="text-xs text-slate-400">
                  {prompt.length}/{MAX_PROMPT_LENGTH}
                </span>
              )}
              <button
                type="submit"
                disabled={!sketch || busy}
                title={!sketch ? "Add a sketch first" : jobRunning ? "Wait for the current model to finish" : "Generate (⌘ + Enter)"}
                className="flex items-center gap-2 rounded-full bg-linear-to-r from-blue-600 to-indigo-600 px-6 py-3 text-[15px] font-semibold text-white shadow-lg shadow-indigo-500/30 transition hover:-translate-y-0.5 hover:shadow-xl hover:shadow-indigo-500/40 hover:brightness-110 disabled:translate-y-0 disabled:cursor-not-allowed disabled:from-slate-300 disabled:to-slate-300 disabled:shadow-none"
              >
                {busy ? <SpinnerIcon className="h-4 w-4" /> : <SparkleIcon className="h-4 w-4" />}
                {isSubmitting ? "Sending…" : jobRunning ? "Generating…" : "Generate 3D Preview"}
              </button>
            </div>
          </div>
        </form>

        <p className="mt-4 text-center text-xs text-slate-400">
          Draw on the whiteboard, upload a PNG/JPG, or just drop or paste an image anywhere · <span className="font-medium">⌘ + Enter</span> to generate
        </p>

        {error && (
          <div role="alert" className="animate-fade-in mt-6 flex items-start gap-3 rounded-2xl border border-red-100 bg-red-50/90 p-4 text-sm text-red-700">
            <AlertIcon className="mt-0.5 h-5 w-5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <div ref={resultRef} className="scroll-mt-6">
          {jobId &&
            (jobStatus?.status === "completed" ? (
              <ResultCard status={jobStatus} onNewSketch={handleNewSketch} />
            ) : (
              <ProgressCard
                jobId={jobId}
                status={jobStatus}
                pollError={pollError}
                onRetry={generate}
                onDismiss={handleDismissJob}
              />
            ))}
        </div>
      </main>

      {isDraggingFile && (
        <div className="pointer-events-none fixed inset-0 z-40 flex items-center justify-center bg-indigo-600/10 p-6 backdrop-blur-sm">
          <div className="flex flex-col items-center rounded-[32px] border-2 border-dashed border-indigo-400 bg-white/90 px-16 py-12 text-center shadow-2xl">
            <UploadIcon className="h-10 w-10 text-indigo-500" />
            <p className="mt-4 font-display text-2xl font-semibold">Drop your sketch</p>
            <p className="mt-1 text-sm text-slate-500">PNG or JPG, up to 10 MB</p>
          </div>
        </div>
      )}

      {whiteboardOpen && (
        <Whiteboard
          initialStrokes={strokes}
          onClose={(s) => {
            setStrokes(s);
            setWhiteboardOpen(false);
          }}
          onUse={handleUseWhiteboard}
        />
      )}
    </div>
  );
}
