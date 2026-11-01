"""
Downloads (and test-loads) all open-source models used by the backend, so the first
generation does not have to wait for ~10 GB of downloads. Run once after installing:

    cd backend
    source .venv/bin/activate
    python -m scripts.download_models

Files are cached in ~/.cache/huggingface and reused by the server.
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.services.background_removal import get_session  # noqa: E402
from app.services.image_generation import load_pipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    print("Downloading models (first time: ~10 GB, this can take a while)...")
    for name in (settings.sd_base_model, settings.sd_controlnet_model, settings.sd_vae_model):
        print("  -", name)
    started = time.perf_counter()
    load_pipeline(move_to_device=False)
    print("  -", f"rembg {settings.rembg_model} (background removal, ~170 MB)")
    get_session()
    print(f"Done in {time.perf_counter() - started:.0f}s. Models are cached and ready.")
    print("For the local 3D fallback (Stage 6) also run: python -m scripts.setup_triposr")
