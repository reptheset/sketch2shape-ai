// Vector model + drawing helpers for the full-screen whiteboard.
// Strokes are stored as point lists (not pixels), which makes undo/redo, panning,
// whole-stroke erasing and a clean cropped export easy.

export type Tool = "pen" | "eraser" | "hand";
export type Point = { x: number; y: number };
export type Stroke = { id: string; size: number; points: Point[] };

export const STROKE_SIZES = [
  { label: "Thin", value: 2 },
  { label: "Medium", value: 4 },
  { label: "Bold", value: 7 },
] as const;

export const INK = "#1e1b4b";
const EXPORT_INK = "#000000";

let strokeCounter = 0;
export function newStrokeId(): string {
  strokeCounter += 1;
  return `${Date.now().toString(36)}-${strokeCounter}`;
}

/** Draws a smoothed stroke (quadratic curves through segment midpoints). */
export function drawStroke(ctx: CanvasRenderingContext2D, stroke: Stroke, color: string = INK, minWidth = 0) {
  const pts = stroke.points;
  if (pts.length === 0) return;
  const width = Math.max(stroke.size, minWidth);

  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  if (pts.length === 1) {
    ctx.beginPath();
    ctx.arc(pts[0].x, pts[0].y, width / 2, 0, Math.PI * 2);
    ctx.fill();
    return;
  }

  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length - 1; i++) {
    const midX = (pts[i].x + pts[i + 1].x) / 2;
    const midY = (pts[i].y + pts[i + 1].y) / 2;
    ctx.quadraticCurveTo(pts[i].x, pts[i].y, midX, midY);
  }
  const last = pts[pts.length - 1];
  ctx.lineTo(last.x, last.y);
  ctx.stroke();
}

function distanceToSegment(p: Point, a: Point, b: Point): number {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const lengthSq = dx * dx + dy * dy;
  if (lengthSq === 0) return Math.hypot(p.x - a.x, p.y - a.y);
  const t = Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / lengthSq));
  return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}

/** True if point p is within `radius` of the stroke (used by the eraser). */
export function strokeHit(stroke: Stroke, p: Point, radius: number): boolean {
  const reach = radius + stroke.size / 2;
  const pts = stroke.points;
  if (pts.length === 1) return Math.hypot(p.x - pts[0].x, p.y - pts[0].y) <= reach;
  for (let i = 0; i < pts.length - 1; i++) {
    if (distanceToSegment(p, pts[i], pts[i + 1]) <= reach) return true;
  }
  return false;
}

export function getBounds(strokes: Stroke[]) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const s of strokes) {
    const half = s.size / 2;
    for (const p of s.points) {
      minX = Math.min(minX, p.x - half);
      minY = Math.min(minY, p.y - half);
      maxX = Math.max(maxX, p.x + half);
      maxY = Math.max(maxY, p.y + half);
    }
  }
  if (!Number.isFinite(minX)) return null;
  return { minX, minY, maxX, maxY };
}

/**
 * Exports the drawing as a square, centred, black-on-white PNG.
 * Cropping to the drawing (instead of saving the whole screen) gives ControlNet a
 * well-framed object, which noticeably improves the generated image.
 */
export function exportStrokesToPng(strokes: Stroke[], outputSize = 1024): Promise<Blob | null> {
  const bounds = getBounds(strokes);
  if (!bounds) return Promise.resolve(null);

  const width = bounds.maxX - bounds.minX;
  const height = bounds.maxY - bounds.minY;
  const side = Math.max(width, height, 1) * 1.3; // ~15% padding on each side
  const scale = outputSize / side;

  const canvas = document.createElement("canvas");
  canvas.width = outputSize;
  canvas.height = outputSize;
  const ctx = canvas.getContext("2d");
  if (!ctx) return Promise.resolve(null);

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, outputSize, outputSize);
  ctx.setTransform(
    scale,
    0,
    0,
    scale,
    (outputSize - width * scale) / 2 - bounds.minX * scale,
    (outputSize - height * scale) / 2 - bounds.minY * scale,
  );
  // Keep lines at least ~3px thick in the output, even for large drawings.
  for (const stroke of strokes) drawStroke(ctx, stroke, EXPORT_INK, 3 / scale);

  return new Promise((resolve) => canvas.toBlob((blob) => resolve(blob), "image/png"));
}
