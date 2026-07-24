"""Copy model weights from the network volume onto local container disk.

ltx-pipelines rebuilds each component from its safetensors on every call
because it has nowhere to cache them: the checkpoints total ~70 GB against
this endpoint's 48 GB of VRAM and 50 GB of container RAM. The distilled
transformer alone is therefore re-read twice per job, and /runpod-volume
reads at roughly 270 MB/s, so each of those reads costs ~170 s -- far more
than the ~40 s of denoising they feed.

Local NVMe turns those reads into seconds. The one-time copy costs about as
much as a single transformer build and is paid once per worker lifetime
rather than twice per job, which FlashBoot then amortises across every job
that worker ever serves.
"""

import os
import shutil
import time

# Headroom left on the container disk after staging. /tmp holds the decoded
# mp4 and any conditioning images while a job runs, and a staged copy must
# never be the thing that fills the disk out from under a live job.
DEFAULT_RESERVE_BYTES = 5 * 1024 ** 3


def _free_bytes(path):
    stats = os.statvfs(path)
    return stats.f_bavail * stats.f_frsize


def _is_memory_backed(path):
    """True when path lives on tmpfs/ramfs.

    Guards against the worst possible outcome of this module: staging 46 GB
    into a RAM-backed filesystem on a box with 50 GB of RAM would OOM-kill the
    worker at startup, and it would look like a model-loading bug rather than
    a caching one. Cheaper to check than to debug.
    """
    try:
        with open("/proc/mounts") as mounts:
            entries = [line.split() for line in mounts]
    except OSError:
        return False

    target = os.path.abspath(path)
    best_match = ""
    best_type = ""
    for entry in entries:
        if len(entry) < 3:
            continue
        mount_point, fs_type = entry[1], entry[2]
        if target == mount_point or target.startswith(mount_point.rstrip("/") + "/"):
            if len(mount_point) >= len(best_match):
                best_match, best_type = mount_point, fs_type

    return best_type in ("tmpfs", "ramfs")


def _copy(source, target, size):
    """Copy one file, falling back to the source path if anything goes wrong.

    A partial copy is worse than no copy -- it would load as a truncated
    checkpoint and fail deep inside the pipeline -- so the temp name is only
    promoted to the real one after the copy returns successfully.
    """
    temp = f"{target}.partial"
    started = time.perf_counter()
    try:
        shutil.copyfile(source, temp)
        os.replace(temp, target)
    except OSError as error:
        print(f"[staging] failed to stage {os.path.basename(source)}: {error}")
        if os.path.exists(temp):
            os.remove(temp)
        return source

    elapsed = max(time.perf_counter() - started, 1e-6)
    print(
        f"[staging] staged {os.path.basename(source)} "
        f"({size / 1e9:.1f} GB in {elapsed:.0f}s, {size / 1e6 / elapsed:.0f} MB/s)"
    )
    return target


def stage_files(paths, local_root, reserve_bytes=DEFAULT_RESERVE_BYTES):
    """Copy weights onto local disk, in the caller's priority order.

    Returns {original_path: path_to_use}. Anything that does not fit maps to
    itself, so the caller transparently falls back to the network volume and
    the endpoint keeps working on a container disk too small to hold the
    checkpoint. Pass the highest-value file first: benefit is (bytes read per
    job), not file size, and the distilled checkpoint is re-read twice while
    the upsampler is read once.
    """
    mapping = {path: path for path in paths}

    if not paths:
        return mapping

    if _is_memory_backed(local_root):
        print(f"[staging] {local_root} is memory-backed, staging disabled")
        return mapping

    try:
        os.makedirs(local_root, exist_ok=True)
    except OSError as error:
        print(f"[staging] cannot create {local_root}: {error}, staging disabled")
        return mapping

    for source in paths:
        name = os.path.basename(source)

        if not os.path.exists(source):
            print(f"[staging] {name} not found at {source}, leaving as-is")
            continue

        size = os.path.getsize(source)
        target = os.path.join(local_root, name)

        # A worker resumed by FlashBoot, or restarted onto a warm container
        # disk, already has the file. Size-match rather than trusting presence:
        # a copy interrupted before os.replace leaves only a .partial, but a
        # truncated real file would otherwise be reused forever.
        if os.path.exists(target) and os.path.getsize(target) == size:
            print(f"[staging] {name} already staged")
            mapping[source] = target
            continue

        free = _free_bytes(local_root)
        if size + reserve_bytes > free:
            print(
                f"[staging] skipping {name}: needs {size / 1e9:.1f} GB plus "
                f"{reserve_bytes / 1e9:.1f} GB reserve, only {free / 1e9:.1f} GB free "
                f"-- serving it from the network volume instead"
            )
            continue

        mapping[source] = _copy(source, target, size)

    return mapping
