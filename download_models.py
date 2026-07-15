import os
import argparse
from huggingface_hub import hf_hub_download, snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

LTX_REPO = "Lightricks/LTX-2.3"
GEMMA_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"

LTX_FILES = [
    "ltx-2.3-22b-dev.safetensors",
    "ltx-2.3-22b-distilled-1.1.safetensors",
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
]


def ensure_models(target_dir):
    """Download LTX-2.3 + Gemma-3 weights into target_dir, skipping any that
    already exist.

    When target_dir is a mounted network volume, files persist across workers,
    so the first cold start populates the volume and every later start reuses
    it instead of re-downloading ~120 GB from HuggingFace.
    """
    models_dir = os.path.abspath(target_dir)
    ltx_dir = os.path.join(models_dir, "ltx-2.3")
    gemma_dir = os.path.join(models_dir, "gemma-3-12b")
    os.makedirs(ltx_dir, exist_ok=True)
    os.makedirs(gemma_dir, exist_ok=True)

    token = os.getenv("HF_TOKEN")

    for filename in LTX_FILES:
        target_path = os.path.join(ltx_dir, filename)
        if os.path.exists(target_path):
            print(f"[models] {filename} already present, skipping")
            continue
        print(f"[models] downloading {filename}...")
        hf_hub_download(repo_id=LTX_REPO, filename=filename, local_dir=ltx_dir, token=token)
        print(f"[models] downloaded {filename}")

    # A populated config.json is a reliable marker that the Gemma snapshot
    # completed; snapshot_download also re-verifies and skips existing files.
    if os.path.exists(os.path.join(gemma_dir, "config.json")):
        print("[models] gemma-3 already present, skipping")
    else:
        print("[models] downloading gemma-3...")
        snapshot_download(
            repo_id=GEMMA_REPO,
            local_dir=gemma_dir,
            token=token,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        )
        print("[models] downloaded gemma-3")

    return ltx_dir, gemma_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=os.getenv("MODELS_ROOT", "/workspace/models"))
    args = parser.parse_args()
    ensure_models(args.dir)
