// Input rules shared by the upload widget, the drawing canvas and the submit logic.
// The FastAPI backend (Stage 2) enforces the same limits, so keep both in sync.

export const ACCEPTED_MIME_TYPES = ["image/png", "image/jpeg"] as const;
export const ACCEPT_ATTRIBUTE = ".png,.jpg,.jpeg,image/png,image/jpeg";
export const MAX_SKETCH_BYTES = 10 * 1024 * 1024; // 10 MB
export const MAX_PROMPT_LENGTH = 500;

/** Returns a human-readable error message, or null if the file is acceptable. */
export function validateSketchFile(file: File): string | null {
  const isAccepted = (ACCEPTED_MIME_TYPES as readonly string[]).includes(file.type);
  if (!isAccepted) {
    return `Unsupported file type "${file.type || "unknown"}". Please upload a PNG or JPG image.`;
  }
  if (file.size === 0) {
    return "The selected file is empty.";
  }
  if (file.size > MAX_SKETCH_BYTES) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    return `The file is ${sizeMb} MB. The maximum allowed size is 10 MB.`;
  }
  return null;
}
