"""Application settings, read once from environment variables (and backend/.env)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")


def _csv(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


RECON_PROVIDERS = ("trellis", "triposr")


def _providers() -> list[str]:
    providers = _csv("RECON_PROVIDERS", "trellis,triposr")
    providers = [p.lower() for p in providers]
    unknown = [p for p in providers if p not in RECON_PROVIDERS]
    if unknown or not providers:
        raise ValueError(f"RECON_PROVIDERS must be a comma-separated list of {RECON_PROVIDERS}, got {providers}")
    return providers


def _path(name: str, default: str) -> Path:
    """Reads a path setting; relative paths are resolved against backend/."""
    path = Path(os.getenv(name, default)).expanduser()
    return (path if path.is_absolute() else BACKEND_DIR / path).resolve()


# Provider -> (API key variable, model variable, default model, base URL).
# Both speak the OpenAI Chat Completions API, so one client works for either.
LLM_PROVIDERS = {
    "groq": ("GROQ_API_KEY", "GROQ_MODEL", "qwen/qwen3.8-27b", "https://api.groq.com/openai/v1"),
    "openai": ("OPENAI_API_KEY", "OPENAI_MODEL", "gpt-4o-mini", None),
}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    cors_origins: list[str]
    max_upload_bytes: int
    max_prompt_chars: int
    sketch_size: int
    # Stage 3: semantic enrichment (any OpenAI-compatible vision LLM: Groq or OpenAI)
    llm_provider: str
    llm_api_key: str | None
    llm_model: str
    llm_base_url: str | None
    llm_timeout_seconds: float
    reject_non_sketches: bool
    # Stage 4: sketch-guided image generation (local SDXL + ControlNet Scribble)
    sd_base_model: str
    sd_controlnet_model: str
    sd_vae_model: str
    sd_device: str
    sd_steps: int
    sd_guidance_scale: float
    sd_attention_slicing: bool
    image_size: int
    preload_models: bool
    # Stage 5: background removal (rembg)
    rembg_model: str
    foreground_size: int
    foreground_margin: float
    # Stage 6: 3D reconstruction (free TRELLIS Space first, local TripoSR as fallback)
    recon_providers: list[str]
    hf_token: str | None
    trellis_space: str
    trellis_ss_steps: int
    trellis_ss_guidance: float
    trellis_slat_steps: int
    trellis_slat_guidance: float
    trellis_mesh_simplify: float
    trellis_texture_size: int
    trellis_timeout_seconds: float
    trellis_quota_cooldown_seconds: float
    triposr_dir: Path
    triposr_model: str
    triposr_device: str
    triposr_mc_resolution: int
    triposr_chunk_size: int
    # Stage 7: mesh validation
    mesh_normalize: bool
    mesh_target_size: float

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"


def load_settings() -> Settings:
    # Relative paths are resolved against backend/, so it works from any working directory.
    data_dir = _path("DATA_DIR", "data")

    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    if provider not in LLM_PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {sorted(LLM_PROVIDERS)}, got '{provider}'")
    key_var, model_var, default_model, base_url = LLM_PROVIDERS[provider]

    return Settings(
        data_dir=data_dir,
        cors_origins=_csv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"),
        max_upload_bytes=int(float(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024),
        max_prompt_chars=int(os.getenv("MAX_PROMPT_CHARS", "500")),
        sketch_size=int(os.getenv("SKETCH_SIZE", "1024")),
        llm_provider=provider,
        llm_api_key=os.getenv(key_var) or None,
        llm_model=os.getenv(model_var, default_model),
        llm_base_url=base_url,
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
        reject_non_sketches=_bool("REJECT_NON_SKETCHES", True),
        sd_base_model=os.getenv("SD_BASE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"),
        sd_controlnet_model=os.getenv("SD_CONTROLNET_MODEL", "xinsir/controlnet-scribble-sdxl-1.0"),
        sd_vae_model=os.getenv("SD_VAE_MODEL", "madebyollin/sdxl-vae-fp16-fix"),
        sd_device=os.getenv("SD_DEVICE", "auto").strip().lower(),
        sd_steps=int(os.getenv("SD_STEPS", "30")),
        sd_guidance_scale=float(os.getenv("SD_GUIDANCE_SCALE", "7.0")),
        sd_attention_slicing=_bool("SD_ATTENTION_SLICING", True),
        image_size=int(os.getenv("IMAGE_SIZE", "1024")),
        preload_models=_bool("PRELOAD_MODELS", True),
        rembg_model=os.getenv("REMBG_MODEL", "u2net"),
        foreground_size=int(os.getenv("FOREGROUND_SIZE", "1024")),
        foreground_margin=float(os.getenv("FOREGROUND_MARGIN", "0.08")),
        recon_providers=_providers(),
        hf_token=os.getenv("HF_TOKEN") or None,
        trellis_space=os.getenv("TRELLIS_SPACE", "trellis-community/TRELLIS"),
        trellis_ss_steps=int(os.getenv("TRELLIS_SS_STEPS", "20")),
        trellis_ss_guidance=float(os.getenv("TRELLIS_SS_GUIDANCE", "7.5")),
        trellis_slat_steps=int(os.getenv("TRELLIS_SLAT_STEPS", "20")),
        trellis_slat_guidance=float(os.getenv("TRELLIS_SLAT_GUIDANCE", "3.0")),
        trellis_mesh_simplify=float(os.getenv("TRELLIS_MESH_SIMPLIFY", "0.95")),
        trellis_texture_size=int(os.getenv("TRELLIS_TEXTURE_SIZE", "1024")),
        trellis_timeout_seconds=float(os.getenv("TRELLIS_TIMEOUT_SECONDS", "300")),
        trellis_quota_cooldown_seconds=float(os.getenv("TRELLIS_QUOTA_COOLDOWN_MINUTES", "30")) * 60,
        triposr_dir=_path("TRIPOSR_DIR", "vendor/TripoSR"),
        triposr_model=os.getenv("TRIPOSR_MODEL", "stabilityai/TripoSR"),
        triposr_device=os.getenv("TRIPOSR_DEVICE", "auto").strip().lower(),
        triposr_mc_resolution=int(os.getenv("TRIPOSR_MC_RESOLUTION", "256")),
        triposr_chunk_size=int(os.getenv("TRIPOSR_CHUNK_SIZE", "8192")),
        mesh_normalize=_bool("MESH_NORMALIZE", True),
        mesh_target_size=float(os.getenv("MESH_TARGET_SIZE", "1.0")),
    )


settings = load_settings()
