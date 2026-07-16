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


COMPLETION_MARKER = ".download-complete"


def _is_complete(directory):
    return os.path.exists(os.path.join(directory, COMPLETION_MARKER))


def _mark_complete(directory):
    with open(os.path.join(directory, COMPLETION_MARKER), "w") as marker:
        marker.write("ok\n")


def ensure_models(target_dir):
    """Download LTX-2.3 + Gemma-3 weights into target_dir.

    When target_dir is a mounted network volume, files persist across workers,
    so the first cold start populates the volume and every later start reuses
    it instead of re-downloading ~120 GB from HuggingFace.

    Completion is tracked with a marker file written only after a download
    returns successfully. Never infer completion from the presence of an
    individual file: snapshot_download fetches in parallel, so a 1 KB
    config.json lands in milliseconds while the 24 GB of shards take minutes.
    Any interruption in that window leaves the small files on disk and the
    large ones missing, and a per-file check then reports "already present"
    forever while the encoder is silently unusable.

    Until the marker exists, the download helpers are re-invoked on every
    start. That is cheap when the files are already there -- both verify
    etags and skip complete files -- and it repairs a partial volume rather
    than wedging on it.
    """
    models_dir = os.path.abspath(target_dir)
    ltx_dir = os.path.join(models_dir, "ltx-2.3")
    gemma_dir = os.path.join(models_dir, "gemma-3-12b")
    os.makedirs(ltx_dir, exist_ok=True)
    os.makedirs(gemma_dir, exist_ok=True)

    token = os.getenv("HF_TOKEN")

    if _is_complete(ltx_dir):
        print("[models] ltx-2.3 complete, skipping")
    else:
        for filename in LTX_FILES:
            print(f"[models] ensuring {filename}...")
            hf_hub_download(repo_id=LTX_REPO, filename=filename, local_dir=ltx_dir, token=token)
        _mark_complete(ltx_dir)
        print("[models] ltx-2.3 complete")

    if _is_complete(gemma_dir):
        print("[models] gemma-3 complete, skipping")
    else:
        print("[models] ensuring gemma-3...")
        snapshot_download(
            repo_id=GEMMA_REPO,
            local_dir=gemma_dir,
            token=token,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        )
        _mark_complete(gemma_dir)
        print("[models] gemma-3 complete")

    return ltx_dir, gemma_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=os.getenv("MODELS_ROOT", "/workspace/models"))
    args = parser.parse_args()
    ensure_models(args.dir)
