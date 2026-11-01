"""
Stage 3: semantic enrichment with a vision LLM.

The default provider is Groq running Qwen 3.8 27B (vision + strict structured outputs);
OpenAI GPT-4o-mini is supported as an alternative. Both expose the OpenAI Chat Completions
API, so the same `openai` SDK client is used and only the base URL, key and model change
(LLM_PROVIDER in backend/.env).

The model looks at the uploaded sketch (and the cleaned version from Stage 2) together
with the user's optional description and returns structured data:
  - what the object is (category, description, key features)
  - plausible materials and colours
  - how detailed the sketch is (sparse / moderate / detailed)
  - an SDXL subject description, which code combines with a fixed product-photo style
    (white background, studio light) into the final prompt for Stage 4

The reply is constrained with a strict JSON schema (response_format type "json_schema",
generated from the Pydantic model below) and then validated with Pydantic, so the rest of
the pipeline always receives a well-formed, typed object.

The ControlNet conditioning scale is then chosen in code from the model's detail rating
plus the ink ratio measured by OpenCV in Stage 2. Keeping that number deterministic makes
runs reproducible and easy to tune:
    sparse   -> 0.55  (let SDXL invent detail the sketch does not show)
    moderate -> 0.70
    detailed -> 0.85+ (follow the drawing closely)

If the LLM is unavailable (no key, no credit, network error, outage) the job does not
fail: a fallback built from the user's prompt and the ink ratio is used and a warning is
recorded.
"""

import base64
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from pydantic import BaseModel, Field, ValidationError

from app.config import settings
from app.errors import PipelineError

logger = logging.getLogger("sketch2shape.enrichment")

VISION_MAX_SIDE = 768  # images are downscaled before upload to keep requests small and fast
MAX_SUBJECT_WORDS = 40  # subject + STYLE_SUFFIX stays within SDXL's ~77-token text window

# The image-to-3D stage works best with a solid object, fully visible, evenly lit, on a clean white
# background. Colours named by the user must stay ON the object (SDXL otherwise tints the whole scene).
STYLE_SUFFIX = (
    "product photography, single solid object, centered, entire object in frame, "
    "isolated on seamless pure white background, soft even studio lighting, sharp focus, highly detailed"
)
DEFAULT_NEGATIVE = (
    "blurry, low quality, multiple objects, cropped, cut off, text, letters, watermark, logo, "
    "colored background, gradient background, vignette, busy background, floor reflection, harsh shadows, "
    "glow, sparkles, distorted, deformed, sketch, line art, drawing, flat 2d"
)


# ---------------------------------------------------------------- LLM response schema


class SketchDetail(str, Enum):
    sparse = "sparse"
    moderate = "moderate"
    detailed = "detailed"


class SketchAnalysis(BaseModel):
    """The exact JSON structure the vision model must return (enforced by a strict JSON schema)."""

    is_object_sketch: bool = Field(
        description="True if the image is a drawing or sketch of a physical object. False only if it is clearly "
        "not a drawing (e.g. a photograph of a person or a scene), or is blank or illegible."
    )
    rejection_reason: str = Field(description="Short, friendly reason if is_object_sketch is false, otherwise an empty string.")
    object_category: str = Field(description="One or two words, e.g. 'armchair', 'coffee mug', 'sports car'.")
    object_description: str = Field(description="One sentence describing the object as drawn, including its shape and parts.")
    sketch_detail: SketchDetail = Field(
        description="sparse = few rough strokes or a simple outline; moderate = clear outline with some parts; "
        "detailed = many parts, internal lines or shading."
    )
    key_features: list[str] = Field(description="3 to 6 visible structural features that must be preserved, e.g. 'curved armrests'.")
    materials: list[str] = Field(description="1 to 3 plausible materials. Use the user's description if it names any.")
    colors: list[str] = Field(description="1 to 3 colours. Use the user's description if it names any.")
    subject_prompt: str = Field(description="SDXL subject description following the rules in the system message.")
    negative_extras: str = Field(description="Comma-separated object-specific things to avoid (e.g. 'extra legs'), at most 12 words, or empty.")


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema in the form strict structured outputs require (OpenAI and Groq):
    every property listed in `required` and `additionalProperties: false` on objects."""
    schema = model.model_json_schema()

    def harden(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            for value in node.values():
                harden(value)
        elif isinstance(node, list):
            for value in node:
                harden(value)

    harden(schema)
    return schema


RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "sketch_analysis", "strict": True, "schema": strict_json_schema(SketchAnalysis)},
}


SYSTEM_PROMPT = f"""You are the vision-and-prompting stage of a pipeline that turns a hand-drawn sketch into a 3D model.
The next stage uses SDXL with ControlNet Scribble: the sketch fixes the shape, your prompt decides what it looks like.
After that, the background is removed and an image-to-3D model reconstructs the object.

You receive the user's original image, a cleaned black-on-white version of it, and an optional user description.

Rules:
- Identify the single main object. Preserve its structure exactly as drawn; never add or remove major parts.
- The user's description is authoritative for object type, materials, colours and style. If it conflicts with
  your guess, follow the user. If there is no description, choose realistic, attractive materials and colours.
- subject_prompt: describe only the object for SDXL. Start with the object (e.g. "a mid-century oak armchair
  with curved legs"), then its materials, colours and key features, as comma-separated phrases, at most
  {MAX_SUBJECT_WORDS} words. Do not describe background, lighting or camera; a fixed product-photo style is appended.
  Never mention sketches, drawings, lines or pencils.
- Turn instructions into attributes: "make it red" or "let the cat be yellow and white" becomes
  "a red plastic pencil sharpener" or "a cat with yellow and white fur". Colours belong to the object only,
  never to the background or scene.
- Prefer solid, physically plausible objects with clear 3D volume and realistic materials (plastic, metal,
  wood, fabric, ceramic...). Avoid glowing, transparent or glittery looks unless the user asks for them.
- Ignore watermarks, captions, stock-photo labels, signatures and stray text in the upload.
- Mark is_object_sketch false only when the input is clearly not a drawing of an object (a photo of a person,
  a landscape photo, a screenshot of text) or is blank. Rough, messy or childlike drawings ARE valid sketches."""


# ---------------------------------------------------------------- result type


@dataclass
class Enrichment:
    source: str  # "llm" or "fallback"
    model: str | None
    object_category: str
    object_description: str
    sketch_detail: str
    key_features: list[str]
    materials: list[str]
    colors: list[str]
    image_prompt: str
    negative_prompt: str
    controlnet_scale: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------- helpers


def detail_from_ink_ratio(ink_ratio: float) -> SketchDetail:
    """Rough detail estimate from Stage 2's ink ratio, used when the LLM is unavailable."""
    if ink_ratio < 0.025:
        return SketchDetail.sparse
    if ink_ratio < 0.06:
        return SketchDetail.moderate
    return SketchDetail.detailed


def choose_controlnet_scale(detail: SketchDetail, ink_ratio: float) -> float:
    """Sparse sketches get a looser guide (~0.55), detailed ones a strict one (0.85+)."""
    scale = {SketchDetail.sparse: 0.55, SketchDetail.moderate: 0.70, SketchDetail.detailed: 0.85}[detail]
    if ink_ratio >= 0.08:  # very dense drawing: follow it closely
        scale = max(scale, 0.90)
    elif ink_ratio < 0.012:  # only a few thin strokes: never over-constrain
        scale = min(scale, 0.70)
    return round(scale, 2)


def _clip_words(text: str, limit: int) -> str:
    words = text.split()
    return " ".join(words[:limit]).rstrip(",") if len(words) > limit else text.strip()


def compose_prompts(subject: str, negative_extras: str = "") -> tuple[str, str]:
    """Builds the final SDXL prompt pair: subject + fixed product-photo style; default negatives + extras."""
    subject = _clip_words(subject.strip().rstrip(".,"), MAX_SUBJECT_WORDS)
    extras = _clip_words(negative_extras.strip().strip(","), 12)
    negative = f"{DEFAULT_NEGATIVE}, {extras}" if extras else DEFAULT_NEGATIVE
    return f"{subject}, {STYLE_SUFFIX}", negative


def _image_data_url(path: Path) -> str:
    """Downscales an image, flattens transparency onto white and returns a base64 JPEG data URL."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise PipelineError("The sketch could not be read for analysis.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype("float32") / 255.0
        image = (image[:, :, :3].astype("float32") * alpha + 255.0 * (1.0 - alpha)).astype("uint8")
    height, width = image.shape[:2]
    scale = VISION_MAX_SIDE / max(height, width)
    if scale < 1:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise PipelineError("The sketch could not be encoded for analysis.")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


# Instruction phrasing that confuses SDXL ("make this laptop of yellow colour" -> "laptop of yellow colour").
_INSTRUCTION_PREFIX = re.compile(
    r"^\s*(please\s+)?((make|turn|render|let|draw|create|generate|give)\s+(me\s+)?(it|this|that|the|a|an)?\s*(be\s+)?|"
    r"(it|this)\s+is\s+(a|an)?\s*)",
    re.IGNORECASE,
)


def clean_user_prompt(user_prompt: str) -> str:
    """Strips instruction phrasing so the description reads like an object, for the no-LLM fallback."""
    text = _INSTRUCTION_PREFIX.sub("", user_prompt.strip()).strip(" .,")
    text = re.sub(r"\b(and\s+)?(make|let)\s+(it|this|them)\s+(be\s+)?", ", ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(colou?r(ed)?)\b|\bbe\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*(,\s*)*", ", ", text).strip(" .,")
    if len(text.split()) == 1:
        text = f"{text} object"  # e.g. "red" -> "red object"
    return text


def build_fallback(user_prompt: str | None, ink_ratio: float, warning: str) -> Enrichment:
    """Enrichment without the LLM: uses the user's description (if any) and the measured ink ratio."""
    detail = detail_from_ink_ratio(ink_ratio)
    subject = f"a {clean_user_prompt(user_prompt)}" if user_prompt else "a well-designed everyday object"
    image_prompt, negative_prompt = compose_prompts(subject)
    return Enrichment(
        source="fallback",
        model=None,
        object_category="object",
        object_description=user_prompt or "Object described only by the sketch outline.",
        sketch_detail=detail.value,
        key_features=[],
        materials=[],
        colors=[],
        image_prompt=image_prompt,
        negative_prompt=negative_prompt,
        controlnet_scale=choose_controlnet_scale(detail, ink_ratio),
        warnings=[warning],
    )


# ---------------------------------------------------------------- LLM call


def _parse_reply(content: str | None) -> SketchAnalysis:
    """Validates the model's JSON. Tolerates text around the JSON (e.g. a reasoning preamble)."""
    if not content:
        raise ValueError("The model returned an empty reply.")
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("The model reply did not contain JSON.")
    try:
        return SketchAnalysis.model_validate(json.loads(content[start : end + 1]))
    except json.JSONDecodeError as exc:
        raise ValueError(f"The model reply was not valid JSON: {exc.msg}") from exc
    except ValidationError as exc:
        fields = ", ".join(".".join(str(p) for p in err["loc"]) for err in exc.errors()[:5])
        raise ValueError(f"The model reply did not match the schema (problem fields: {fields})") from exc


async def _analyse_with_llm(original_path: Path, clean_path: Path, user_prompt: str | None) -> SketchAnalysis:
    description = f'User description: "{user_prompt}"' if user_prompt else "User description: none provided."

    def image_part(path: Path) -> dict[str, Any]:
        image_url: dict[str, Any] = {"url": _image_data_url(path)}
        if settings.llm_provider == "openai":
            image_url["detail"] = "low"  # OpenAI-specific option; cheaper and plenty for line art
        return {"type": "image_url", "image_url": image_url}

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": f"{description}\nImage 1 is the user's original upload:"},
        image_part(original_path),
        {"type": "text", "text": "Image 2 is the cleaned black-on-white sketch that ControlNet will follow:"},
        image_part(clean_path),
    ]

    client = AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,  # None means api.openai.com
        timeout=settings.llm_timeout_seconds,
        max_retries=4,  # the SDK retries connection errors, 429s (honouring Retry-After) and 5xx with backoff
    )
    try:
        completion = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format=RESPONSE_FORMAT,
            temperature=0.2,
        )
    finally:
        await client.close()

    message = completion.choices[0].message
    refusal = getattr(message, "refusal", None)
    if refusal:
        raise ValueError(f"The model refused to analyse the sketch: {refusal}")
    return _parse_reply(message.content)


async def enrich_sketch(
    original_path: Path,
    clean_path: Path,
    user_prompt: str | None,
    ink_ratio: float,
) -> Enrichment:
    """Runs Stage 3. Raises PipelineError only when the input is clearly not a sketch."""
    provider = settings.llm_provider.capitalize() if settings.llm_provider != "openai" else "OpenAI"
    key_var = "GROQ_API_KEY" if settings.llm_provider == "groq" else "OPENAI_API_KEY"

    if not settings.llm_api_key:
        logger.warning("%s is not set; using fallback enrichment", key_var)
        return build_fallback(user_prompt, ink_ratio, f"{key_var} is not set, so the sketch was not analysed by the vision model.")

    try:
        analysis = await _analyse_with_llm(original_path, clean_path, user_prompt)
    except AuthenticationError:
        logger.error("%s rejected the API key", provider)
        return build_fallback(user_prompt, ink_ratio, f"{provider} rejected the API key; check {key_var} in backend/.env.")
    except RateLimitError:
        logger.warning("%s rate limit or quota exceeded", provider)
        return build_fallback(user_prompt, ink_ratio, f"{provider} rate limit or quota exceeded; used a basic prompt instead.")
    except APITimeoutError:
        logger.warning("%s request timed out", provider)
        return build_fallback(user_prompt, ink_ratio, f"{provider} timed out; used a basic prompt instead.")
    except APIConnectionError:
        logger.warning("Could not connect to %s", provider)
        return build_fallback(user_prompt, ink_ratio, f"Could not reach {provider}; used a basic prompt instead.")
    except APIStatusError as exc:
        logger.error("%s API error %s: %s", provider, exc.status_code, exc.message)
        return build_fallback(user_prompt, ink_ratio, f"{provider} returned an error ({exc.status_code}); used a basic prompt instead.")
    except (OpenAIError, ValueError) as exc:
        logger.error("Sketch analysis failed: %s", exc)
        return build_fallback(user_prompt, ink_ratio, "Sketch analysis failed; used a basic prompt instead.")

    if not analysis.is_object_sketch and settings.reject_non_sketches:
        reason = analysis.rejection_reason.strip() or "The image does not look like a sketch of an object."
        raise PipelineError(f"{reason} Please draw or upload a sketch of a single object.")

    detail = analysis.sketch_detail
    image_prompt, negative_prompt = compose_prompts(analysis.subject_prompt, analysis.negative_extras)
    return Enrichment(
        source="llm",
        model=f"{settings.llm_provider}/{settings.llm_model}",
        object_category=analysis.object_category.strip(),
        object_description=analysis.object_description.strip(),
        sketch_detail=detail.value,
        key_features=[f.strip() for f in analysis.key_features if f.strip()][:6],
        materials=[m.strip() for m in analysis.materials if m.strip()][:3],
        colors=[c.strip() for c in analysis.colors if c.strip()][:3],
        image_prompt=image_prompt,
        negative_prompt=negative_prompt,
        controlnet_scale=choose_controlnet_scale(detail, ink_ratio),
    )
