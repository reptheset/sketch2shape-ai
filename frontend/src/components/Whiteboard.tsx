"use client";

import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";
import {
  ArrowLeftIcon,
  CheckIcon,
  EraserIcon,
  MoveIcon,
  PencilIcon,
  RedoIcon,
  TrashIcon,
  UndoIcon,
} from "@/components/icons";
import { cx } from "@/lib/cx";
import { STROKE_SIZES, drawStroke, newStrokeId, strokeHit, type Point, type Stroke, type Tool } from "@/lib/whiteboard";

type WhiteboardProps = {
  initialStrokes: Stroke[];
  /** Leave the whiteboard but keep the drawing for later. */
  onClose: (strokes: Stroke[]) => void;
  /** Attach the drawing to the prompt. */
  onUse: (strokes: Stroke[]) => void;
};

const ERASER_RADIUS = 10;

export default function Whiteboard({ initialStrokes, onClose, onUse }: WhiteboardProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Refs hold the live drawing data so pointer events never wait for a React re-render.
  const strokesRef = useRef<Stroke[]>(initialStrokes);
  const undoRef = useRef<Stroke[][]>([]);
  const redoRef = useRef<Stroke[][]>([]);
  const currentStrokeRef = useRef<Stroke | null>(null);
  const eraseStartRef = useRef<Stroke[] | null>(null);
  const panStartRef = useRef<{ x: number; y: number; ox: number; oy: number } | null>(null);
  const offsetRef = useRef<Point>({ x: 0, y: 0 });
  const onCloseRef = useRef(onClose);

  const [tool, setTool] = useState<Tool>("pen");
  const [size, setSize] = useState<number>(STROKE_SIZES[1].value);
  const [spaceHeld, setSpaceHeld] = useState(false);
  const [isPanning, setIsPanning] = useState(false);
  const [isDrawing, setIsDrawing] = useState(false);
  const [offset, setOffset] = useState<Point>({ x: 0, y: 0 });
  const [history, setHistory] = useState({ count: initialStrokes.length, canUndo: false, canRedo: false });

  useEffect(() => {
    onCloseRef.current = onClose;
  });

  const activeTool: Tool = spaceHeld ? "hand" : tool;

  const syncHistory = useCallback(() => {
    setHistory({
      count: strokesRef.current.length,
      canUndo: undoRef.current.length > 0,
      canRedo: redoRef.current.length > 0,
    });
  }, []);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const { x, y } = offsetRef.current;
    ctx.setTransform(dpr, 0, 0, dpr, x * dpr, y * dpr);
    for (const stroke of strokesRef.current) drawStroke(ctx, stroke);
    if (currentStrokeRef.current) drawStroke(ctx, currentStrokeRef.current);
  }, []);

  const commit = useCallback(
    (next: Stroke[]) => {
      undoRef.current.push(strokesRef.current);
      redoRef.current = [];
      strokesRef.current = next;
      syncHistory();
    },
    [syncHistory],
  );

  const undo = useCallback(() => {
    const previous = undoRef.current.pop();
    if (!previous) return;
    redoRef.current.push(strokesRef.current);
    strokesRef.current = previous;
    syncHistory();
  }, [syncHistory]);

  const redo = useCallback(() => {
    const next = redoRef.current.pop();
    if (!next) return;
    undoRef.current.push(strokesRef.current);
    strokesRef.current = next;
    syncHistory();
  }, [syncHistory]);

  const clearAll = useCallback(() => {
    if (strokesRef.current.length > 0) commit([]);
  }, [commit]);

  // Keep the canvas pixel-perfect on any screen size / retina display.
  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;
    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(container.clientWidth * dpr);
      canvas.height = Math.round(container.clientHeight * dpr);
      redraw();
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    return () => observer.disconnect();
  }, [redraw]);

  useEffect(() => {
    redraw();
  }, [history, offset, redraw]);

  // Lock page scroll while the full-screen board is open.
  useEffect(() => {
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, []);

  // Keyboard shortcuts (same spirit as Excalidraw).
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      const mod = e.metaKey || e.ctrlKey;
      if (e.key === "Escape") {
        onCloseRef.current(strokesRef.current);
        return;
      }
      if (mod && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (e.shiftKey) redo();
        else undo();
        return;
      }
      if (mod && e.key.toLowerCase() === "y") {
        e.preventDefault();
        redo();
        return;
      }
      if (mod || e.altKey) return;
      if (e.code === "Space") {
        e.preventDefault();
        setSpaceHeld(true);
        return;
      }
      const key = e.key.toLowerCase();
      if (key === "p" || key === "b") setTool("pen");
      else if (key === "e") setTool("eraser");
      else if (key === "h") setTool("hand");
      else if (key === "1" || key === "2" || key === "3") setSize(STROKE_SIZES[Number(key) - 1].value);
    }
    function onKeyUp(e: KeyboardEvent) {
      if (e.code === "Space") setSpaceHeld(false);
    }
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [undo, redo]);

  function toWorld(e: { clientX: number; clientY: number }): Point {
    const rect = canvasRef.current?.getBoundingClientRect();
    const left = rect?.left ?? 0;
    const top = rect?.top ?? 0;
    return { x: e.clientX - left - offsetRef.current.x, y: e.clientY - top - offsetRef.current.y };
  }

  function eraseAt(p: Point) {
    const remaining = strokesRef.current.filter((s) => !strokeHit(s, p, ERASER_RADIUS));
    if (remaining.length !== strokesRef.current.length) {
      strokesRef.current = remaining;
      redraw();
    }
  }

  function handlePointerDown(e: ReactPointerEvent<HTMLCanvasElement>) {
    if (e.pointerType === "mouse" && e.button !== 0 && e.button !== 1) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    const mode: Tool = e.button === 1 ? "hand" : activeTool;

    if (mode === "hand") {
      panStartRef.current = { x: e.clientX, y: e.clientY, ox: offsetRef.current.x, oy: offsetRef.current.y };
      setIsPanning(true);
      return;
    }

    const p = toWorld(e);
    if (mode === "pen") {
      currentStrokeRef.current = { id: newStrokeId(), size, points: [p] };
      setIsDrawing(true);
      redraw();
      return;
    }

    eraseStartRef.current = strokesRef.current;
    eraseAt(p);
  }

  function handlePointerMove(e: ReactPointerEvent<HTMLCanvasElement>) {
    const pan = panStartRef.current;
    if (pan) {
      const next = { x: pan.ox + e.clientX - pan.x, y: pan.oy + e.clientY - pan.y };
      offsetRef.current = next;
      setOffset(next);
      return;
    }

    // Coalesced events give smoother lines on fast strokes (Chrome/Edge; Safari falls back).
    const events = e.nativeEvent.getCoalescedEvents?.() ?? [e.nativeEvent];
    const stroke = currentStrokeRef.current;
    if (stroke) {
      for (const ev of events.length > 0 ? events : [e.nativeEvent]) {
        const p = toWorld(ev);
        const last = stroke.points[stroke.points.length - 1];
        if (Math.hypot(p.x - last.x, p.y - last.y) >= 1) stroke.points.push(p);
      }
      redraw();
      return;
    }

    if (eraseStartRef.current) eraseAt(toWorld(e));
  }

  function handlePointerUp() {
    if (panStartRef.current) {
      panStartRef.current = null;
      setIsPanning(false);
      return;
    }
    const stroke = currentStrokeRef.current;
    if (stroke) {
      currentStrokeRef.current = null;
      setIsDrawing(false);
      commit([...strokesRef.current, stroke]);
      return;
    }
    const before = eraseStartRef.current;
    if (before) {
      eraseStartRef.current = null;
      if (before !== strokesRef.current) {
        // One eraser drag = one undo step.
        undoRef.current.push(before);
        redoRef.current = [];
        syncHistory();
      }
    }
  }

  const cursorClass =
    activeTool === "hand" ? (isPanning ? "cursor-grabbing" : "cursor-grab") : activeTool === "eraser" ? "cursor-cell" : "cursor-crosshair";

  const isEmpty = history.count === 0 && !isDrawing;

  return (
    <div className="fixed inset-0 z-50 flex animate-fade-in flex-col bg-[#fbfbfd]" role="dialog" aria-modal="true" aria-label="Sketch whiteboard">
      <div
        ref={containerRef}
        className="dot-grid relative flex-1 overflow-hidden"
        style={{ backgroundPosition: `${offset.x}px ${offset.y}px` }}
      >
        <canvas
          ref={canvasRef}
          className={cx("absolute inset-0 h-full w-full touch-none", cursorClass)}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
        />

        {isEmpty && (
          <div className="pointer-events-none absolute inset-0 flex select-none flex-col items-center justify-center px-6 text-center">
            <PencilIcon className="mb-4 h-10 w-10 text-indigo-200" />
            <p className="font-display text-3xl font-medium text-slate-300 sm:text-4xl">Sketch your object here</p>
            <p className="mt-3 max-w-md text-sm text-slate-400">
              One object, one view. Clean, closed outlines give the best 3D results.
            </p>
          </div>
        )}

        {/* Top bar: back + primary action */}
        <div className="pointer-events-none absolute inset-x-0 top-0 flex items-start justify-between gap-3 p-4">
          <button
            type="button"
            onClick={() => onClose(strokesRef.current)}
            className="pointer-events-auto flex items-center gap-2 rounded-full border border-slate-200/80 bg-white/90 px-4 py-2.5 text-sm font-medium text-slate-700 shadow-sm backdrop-blur transition hover:bg-white hover:shadow"
          >
            <ArrowLeftIcon className="h-4 w-4" />
            Back
          </button>

          <button
            type="button"
            disabled={history.count === 0}
            onClick={() => onUse(strokesRef.current)}
            className="pointer-events-auto flex items-center gap-2 rounded-full bg-linear-to-r from-blue-600 to-indigo-600 px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-indigo-500/30 transition hover:shadow-xl hover:shadow-indigo-500/40 hover:brightness-110 disabled:cursor-not-allowed disabled:from-slate-300 disabled:to-slate-300 disabled:shadow-none"
          >
            <CheckIcon className="h-4 w-4" />
            Use this sketch
          </button>
        </div>

        {/* Floating toolbar */}
        <div className="absolute bottom-6 left-1/2 flex -translate-x-1/2 items-center gap-1 rounded-2xl border border-slate-200/80 bg-white/95 p-1.5 shadow-[0_8px_30px_-8px_rgba(30,27,75,0.25)] backdrop-blur md:bottom-auto md:top-4">
          <ToolButton label="Pen" shortcut="P" active={activeTool === "pen"} onClick={() => setTool("pen")}>
            <PencilIcon className="h-[18px] w-[18px]" />
          </ToolButton>
          <ToolButton label="Eraser" shortcut="E" active={activeTool === "eraser"} onClick={() => setTool("eraser")}>
            <EraserIcon className="h-[18px] w-[18px]" />
          </ToolButton>
          <ToolButton label="Pan (or hold Space)" shortcut="H" active={activeTool === "hand"} onClick={() => setTool("hand")}>
            <MoveIcon className="h-[18px] w-[18px]" />
          </ToolButton>

          <Divider />

          {STROKE_SIZES.map((s, i) => (
            <ToolButton
              key={s.value}
              label={`${s.label} stroke`}
              shortcut={String(i + 1)}
              active={size === s.value}
              onClick={() => {
                setSize(s.value);
                if (tool !== "pen") setTool("pen");
              }}
            >
              <span className="rounded-full bg-current" style={{ width: s.value + 3, height: s.value + 3 }} />
            </ToolButton>
          ))}

          <Divider />

          <ToolButton label="Undo" shortcut="⌘Z" disabled={!history.canUndo} onClick={undo}>
            <UndoIcon className="h-[18px] w-[18px]" />
          </ToolButton>
          <ToolButton label="Redo" shortcut="⌘⇧Z" disabled={!history.canRedo} onClick={redo}>
            <RedoIcon className="h-[18px] w-[18px]" />
          </ToolButton>
          <ToolButton label="Clear canvas" disabled={history.count === 0} onClick={clearAll} danger>
            <TrashIcon className="h-[18px] w-[18px]" />
          </ToolButton>
        </div>

        <p className="pointer-events-none absolute bottom-6 right-6 hidden select-none text-xs text-slate-400 md:block">
          Hold <Kbd>Space</Kbd> to pan · <Kbd>⌘Z</Kbd> undo · <Kbd>Esc</Kbd> back
        </p>
      </div>
    </div>
  );
}

function ToolButton({
  label,
  shortcut,
  active = false,
  disabled = false,
  danger = false,
  onClick,
  children,
}: {
  label: string;
  shortcut?: string;
  active?: boolean;
  disabled?: boolean;
  danger?: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      title={shortcut ? `${label} — ${shortcut}` : label}
      aria-label={label}
      aria-pressed={active}
      disabled={disabled}
      onClick={onClick}
      className={cx(
        "flex h-9 w-9 items-center justify-center rounded-xl transition",
        active ? "bg-indigo-100 text-indigo-700" : "text-slate-600 hover:bg-slate-100",
        danger && !disabled && "hover:bg-red-50 hover:text-red-600",
        disabled && "cursor-not-allowed opacity-35 hover:bg-transparent",
      )}
    >
      {children}
    </button>
  );
}

function Divider() {
  return <span className="mx-1 h-6 w-px bg-slate-200" aria-hidden="true" />;
}

function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="rounded-md border border-slate-200 bg-white px-1.5 py-0.5 font-sans text-[11px] text-slate-500 shadow-sm">
      {children}
    </kbd>
  );
}
