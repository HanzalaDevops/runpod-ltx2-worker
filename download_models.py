import os
import argparse
import shutil
from huggingface_hub import hf_hub_download, snapshot_download

LTX_REPO = "Lightricks/LTX-2.3"
GEMMA_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"

# This deployment serves ONLY the standalone distilled checkpoint. The
# distilled file is self-contained (transformer + VAE + text-encoder
# projections); the spatial upscaler is still required by DistilledPipeline
# for its second-stage x2 upscale.
LTX_FILES = [
    "ltx-2.3-22b-distilled-1.1.safetensors",
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
]

# Downloaded by earlier versions and left behind on any volume they touched.
# The previous revision served distilled as dev checkpoint + distillation
# LoRA; this deployment loads the standalone distilled checkpoint instead,
# so dev (~44 GB) and the LoRA are dead weight. Pruning them is also what
# makes room for the 46 GB distilled file on an already-populated volume --
# without it the download fails mid-way with EDQUOT.
OBSOLETE_LTX_FILES = [
    "ltx-2.3-22b-dev.safetensors",
    "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
]


# Versioned per file-set: a volume marked complete by the dev+LoRA revision
# holds none of the files this revision needs, so its old ".download-complete"
# marker must not short-circuit the download. Bump the suffix whenever
# LTX_FILES changes.
LTX_COMPLETION_MARKER = ".download-complete-distilled-v2"
GEMMA_COMPLETION_MARKER = ".download-complete"

# Markers written by earlier revisions, superseded by the versioned one.
OBSOLETE_LTX_MARKERS = [
    ".download-complete",
]


def _is_complete(directory, marker_name):
    return os.path.exists(os.path.join(directory, marker_name))


def _mark_complete(directory, marker_name):
    with open(os.path.join(directory, marker_name), "w") as marker:
        marker.write("ok\n")


def _prune(directory, filenames):
    """Delete named files we no longer need, freeing volume space.

    Runs outside the completion-marker check on purpose: a volume marked
    complete by an older version still holds whatever that version fetched,
    and nothing else would ever reclaim it.
    """
    for name in filenames:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            size_gb = os.path.getsize(path) / 1e9
            os.remove(path)
            print(f"[models] removed obsolete {name} ({size_gb:.1f} GB reclaimed)")


def _reset(directory):
    """Delete a directory's contents so a download starts from a clean slate.

    An interrupted snapshot_download leaves scratch files behind that nothing
    ever reclaims. Each crash-and-restart adds another set, and on a network
    volume they accumulate silently until it fills and every write fails with
    EDQUOT -- which reads like "the model is too big" rather than "we leaked
    100 GB of temp files". Re-downloading is bounded and cheap; leaking is not.
    """
    shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(directory, exist_ok=True)


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
    start, so a partial volume repairs itself rather than wedging.

    Gemma is reset before each attempt instead of resumed. Resuming looks
    cheaper but leaks: every interrupted snapshot_download strands scratch
    files, and a crash loop piles up one set per restart until the volume is
    full and writes fail with EDQUOT. Bounded re-downloads beat unbounded
    leaks. LTX is not reset -- its ~48 GB of files are fetched one at a time
    via hf_hub_download and guarded by the versioned marker.
    """
    models_dir = os.path.abspath(target_dir)
    ltx_dir = os.path.join(models_dir, "ltx-2.3")
    gemma_dir = os.path.join(models_dir, "gemma-3-12b")
    os.makedirs(ltx_dir, exist_ok=True)
    os.makedirs(gemma_dir, exist_ok=True)

    token = os.getenv("HF_TOKEN")

    # Prune before the completion check so an old volume both reclaims the
    # space the distilled checkpoint needs and drops the stale marker that
    # would otherwise skip downloading it.
    _prune(ltx_dir, OBSOLETE_LTX_FILES + OBSOLETE_LTX_MARKERS)

    if _is_complete(ltx_dir, LTX_COMPLETION_MARKER):
        print("[models] ltx-2.3 distilled complete, skipping")
    else:
        for filename in LTX_FILES:
            print(f"[models] ensuring {filename}...")
            hf_hub_download(repo_id=LTX_REPO, filename=filename, local_dir=ltx_dir, token=token)
        _mark_complete(ltx_dir, LTX_COMPLETION_MARKER)
        print("[models] ltx-2.3 distilled complete")

    if _is_complete(gemma_dir, GEMMA_COMPLETION_MARKER):
        print("[models] gemma-3 complete, skipping")
    else:
        print("[models] resetting partial gemma-3 download...")
        _reset(gemma_dir)
        print("[models] ensuring gemma-3...")
        snapshot_download(
            repo_id=GEMMA_REPO,
            local_dir=gemma_dir,
            token=token,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        )
        _mark_complete(gemma_dir, GEMMA_COMPLETION_MARKER)
        print("[models] gemma-3 complete")

    return ltx_dir, gemma_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=os.getenv("MODELS_ROOT", "/workspace/models"))
    args = parser.parse_args()
    ensure_models(args.dir)
