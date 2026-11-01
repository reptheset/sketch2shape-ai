"""
Sets up the local TripoSR fallback for Stage 6 (run once, from backend/ with the venv active):

    python -m scripts.setup_triposr

1. Clones github.com/VAST-AI-Research/TripoSR (MIT) into backend/vendor/TripoSR.
   Only its `tsr` Python package is used; its requirements are already covered by
   backend/requirements.txt (torchmcubes is replaced by a scikit-image shim, so nothing
   has to be compiled).
2. Downloads the stabilityai/TripoSR weights (~1.7 GB) into ~/.cache/huggingface.
3. Runs a tiny test reconstruction to prove everything works.
"""

import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402

REPO_URL = "https://github.com/VAST-AI-Research/TripoSR.git"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def clone_repo() -> None:
    target = settings.triposr_dir
    if (target / "tsr" / "system.py").is_file():
        print(f"TripoSR code already present in {target}")
        return
    if shutil.which("git") is None:
        sys.exit("git is not installed. On macOS run: xcode-select --install")
    if target.exists() and any(target.iterdir()):
        sys.exit(f"{target} exists but does not contain TripoSR. Delete it and run this again.")
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {REPO_URL} into {target} ...")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(target)], check=True)


def smoke_test() -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    from app.services.reconstruction_triposr import load_model, run_triposr

    started = time.perf_counter()
    load_model()
    print(f"Model loaded in {time.perf_counter() - started:.0f}s")

    work = settings.data_dir / "triposr_selftest"
    work.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    ImageDraw.Draw(image).ellipse((96, 96, 416, 416), fill=(200, 60, 60, 255))  # a red ball
    image.save(work / "ball.png")
    details = run_triposr(work / "ball.png", work / "ball.glb", lambda msg: print("  ", msg))
    print(f"Test model written to {work / 'ball.glb'}: {details['vertices']} vertices, "
          f"{details['faces']} faces, {details['seconds']}s on {details['device']}")
    assert np.isfinite(details["seconds"])


if __name__ == "__main__":
    clone_repo()
    print(f"Downloading/loading TripoSR weights ({settings.triposr_model}, ~1.7 GB the first time)...")
    smoke_test()
    print("TripoSR fallback is ready.")
