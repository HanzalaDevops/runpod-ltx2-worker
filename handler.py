import os
import gc
import json
import time
import uuid
import urllib.parse
import urllib.request
import torch
import runpod
from runpod.serverless.utils import rp_upload

from download_models import ensure_models
from staging import stage_files

# Import LTX-2 pipeline components
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.types import OffloadMode
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.args import ImageConditioningInput

# Global pipeline cache
current_pipeline = None
current_pipeline_name = None
current_quantization = None
current_offload_mode = None

MODELS_ROOT = os.getenv("MODELS_ROOT", "/workspace/models")
LTX_DIR = os.path.join(MODELS_ROOT, "ltx-2.3")
GEMMA_DIR = os.path.join(MODELS_ROOT, "gemma-3-12b")

DISTILLED_CHECKPOINT = os.path.join(LTX_DIR, "ltx-2.3-22b-distilled-1.1.safetensors")
SPATIAL_UPSAMPLER = os.path.join(LTX_DIR, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")

# Where staged copies of the weights live. Must be on the container's local
# disk, not the network volume -- the whole point is to stop reading the
# checkpoint over the network. Staging is skipped automatically when the disk
# is too small, so this is safe to leave on.
LOCAL_CACHE_DIR = os.getenv("LOCAL_CACHE_DIR", "/local-cache")
STAGING_ENABLED = os.getenv("DISABLE_LOCAL_STAGING", "0") != "1"
STAGING_RESERVE_BYTES = int(float(os.getenv("STAGING_RESERVE_GB", "5")) * 1024 ** 3)

# Resolved at startup: {original_path: path_to_actually_load_from}. Falls back
# to identity so the handler works unchanged when staging is off or skipped.
STAGED_PATHS = {}

# DiffusionStage.__call__ is documented upstream as "Build transformer -> run
# denoising loop -> free transformer", and PromptEncoder frees Gemma the same
# way. Every component is therefore re-read from its checkpoint on every job,
# whatever the offload mode -- which is why the read path matters far more
# than the pipeline cache. On a Pod (MODELS_ROOT on local disk) these reads are
# fast; on serverless they come off /runpod-volume at ~270 MB/s, and that alone
# is the 2-3 minute gap between the two.
#
# quantization/offload_mode are deployment properties, not request properties:
# changing either rebuilds the pipeline, so honouring them per request lets one
# caller evict the warm pipeline for everyone queued behind it. Set
# ALLOW_REQUEST_PIPELINE_OVERRIDE=1 to accept them from the payload for an A/B.
#
# Both default to what an A40 can actually run, and the two are coupled.
#
# QuantizationKind offers only FP8_CAST and FP8_SCALED_MM -- there is no int8
# or bf16-compatible backend. The A40 is Ampere (SM 8.6) and FP8 tensor cores
# only arrive with Ada (SM 8.9), so *no* quantization is available here; asking
# for fp8-cast fails outright with an unsupported-hardware error.
#
# Unquantized, the 22B transformer is bf16: ~44 GB against 48 GB of VRAM, with
# nothing left for activations. OffloadMode.NONE needs the full model resident,
# so it cannot fit either. CPU offload is therefore not a tuning choice on this
# GPU, it is the only mode that runs -- which is also why VRAM sits near idle
# while weights stream through a ~5 GB buffer.
#
# Escaping that needs different hardware, not different settings: an FP8-capable
# GPU (Ada/Hopper) makes fp8-cast work and drops the transformer to ~22 GB, or
# a >=64 GB card fits bf16 resident. On the A40 the lever that remains is where
# the weights are read from -- see the staging notes above.
DEFAULT_QUANTIZATION = os.getenv("LTX_QUANTIZATION", "none")
DEFAULT_OFFLOAD_MODE = os.getenv("LTX_OFFLOAD_MODE", "cpu")
ALLOW_REQUEST_OVERRIDE = os.getenv("ALLOW_REQUEST_PIPELINE_OVERRIDE", "0") == "1"

S3_CRED_KEYS = ("endpointUrl", "accessId", "accessSecret")
S3_REQUIRED_KEYS = S3_CRED_KEYS + ("bucketName",)


def resolve_s3_config(job_input):
    """Resolve the destination bucket for this job's video.

    RunPod hands the handler only the job id and input, so per-request bucket
    config has to travel in input.s3_config -- HTTP headers sent to /run are
    consumed by RunPod's gateway and never reach here. Endpoint BUCKET_* env
    vars act as a fallback so a single-bucket deployment can keep its secrets
    out of request payloads entirely.

    Returns None when neither source is configured. Raises ValueError naming
    only the missing keys, never their values.
    """
    config = job_input.get("s3_config") or {}

    if not config:
        from_env = {
            "endpointUrl": os.getenv("BUCKET_ENDPOINT_URL"),
            "accessId": os.getenv("BUCKET_ACCESS_KEY_ID"),
            "accessSecret": os.getenv("BUCKET_SECRET_ACCESS_KEY"),
            "bucketName": os.getenv("BUCKET_NAME"),
        }
        config = from_env if all(from_env.values()) else {}

    if not config:
        return None

    missing = [key for key in S3_REQUIRED_KEYS if not config.get(key)]
    if missing:
        raise ValueError(f"s3_config is missing required keys: {', '.join(missing)}")

    return config


def upload_video(output_path, job_input):
    """Upload the rendered video and return a presigned URL valid for 7 days."""
    s3_config = resolve_s3_config(job_input)
    if s3_config is None:
        raise ValueError(
            "No bucket configured. Pass input.s3_config with endpointUrl, "
            "accessId, accessSecret and bucketName, or set the BUCKET_* "
            "environment variables on the endpoint."
        )

    return rp_upload.upload_file_to_bucket(
        file_name=f"{uuid.uuid4()}.mp4",
        file_location=output_path,
        bucket_creds={key: s3_config[key] for key in S3_CRED_KEYS},
        bucket_name=s3_config["bucketName"],
        prefix=job_input.get("s3_prefix"),
    )


def redact_url(url):
    """Strip the query string from a URL before logging it.

    Conditioning images arrive as presigned URLs, so the query string carries
    the access key id and signature. Printing it whole puts live credentials
    into RunPod's log storage, where they outlive the job by a long way.
    """
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return "<unparseable url>"
    redacted = "?<redacted>" if parts.query else ""
    return f"{parts.scheme}://{parts.netloc}{parts.path}{redacted}"


def download_file(url, target_path):
    print(f"Downloading input file from {redact_url(url)} to {target_path}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
        out_file.write(response.read())
    return target_path


class StageTimer:
    """Collect per-stage wall-clock timings for one job and emit them as JSON.

    The endpoint's cost is dominated by work that is invisible in the response:
    on the A40 the per-call model rebuilds run roughly 11x longer than the
    denoising they feed. Without per-stage numbers there is no way to tell a
    config change that helped from one that did not, so every job now reports
    where its seconds went and how close it came to filling VRAM.
    """

    def __init__(self, job_id):
        self.job_id = job_id
        self.stages = {}
        self._started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def stage(self, name):
        return _StageContext(self, name)

    def record(self, name, seconds):
        self.stages[name] = round(seconds, 2)

    def emit(self, **extra):
        payload = {
            "event": "job_timing",
            "job_id": self.job_id,
            "total_s": round(time.perf_counter() - self._started, 2),
            "stages_s": self.stages,
            "quantization": DEFAULT_QUANTIZATION,
            "offload_mode": DEFAULT_OFFLOAD_MODE,
            "weights_staged_locally": any(
                source != target for source, target in STAGED_PATHS.items()
            ),
            **extra,
        }
        if torch.cuda.is_available():
            payload["peak_vram_gb"] = round(
                torch.cuda.max_memory_allocated() / 1024 ** 3, 2
            )
        print(json.dumps(payload), flush=True)


class _StageContext:
    def __init__(self, timer, name):
        self._timer = timer
        self._name = name

    def __enter__(self):
        self._started = time.perf_counter()
        return self

    def __exit__(self, *_):
        # Record on failure too -- knowing a job died 300s into generation is
        # more useful than knowing only that it died. Returning False keeps the
        # exception propagating; this only observes it.
        self._timer.record(self._name, time.perf_counter() - self._started)
        return False


def resolve_pipeline_config(job_input):
    """Return (quantization, offload_mode) for this job.

    Reads the endpoint environment by default so a single request cannot force
    a multi-minute pipeline rebuild on everyone else. See the notes on the
    DEFAULT_* constants for why these belong to the deployment.
    """
    if ALLOW_REQUEST_OVERRIDE:
        quantization = job_input.get("quantization", DEFAULT_QUANTIZATION)
        offload_mode = job_input.get("offload_mode", DEFAULT_OFFLOAD_MODE)
    else:
        quantization, offload_mode = DEFAULT_QUANTIZATION, DEFAULT_OFFLOAD_MODE

    return (None if quantization in ("none", "", None) else quantization), offload_mode


def stage_weights():
    """Copy the weights onto local disk, highest-value file first.

    Ordering is by bytes read per job, not file size: the distilled checkpoint
    backs the transformer, which upstream builds and frees twice per job, so it
    is worth far more than the upsampler even though both are read.
    """
    global STAGED_PATHS

    candidates = [DISTILLED_CHECKPOINT, SPATIAL_UPSAMPLER]
    if not STAGING_ENABLED:
        print("[staging] disabled via DISABLE_LOCAL_STAGING")
        STAGED_PATHS = {path: path for path in candidates}
        return

    started = time.perf_counter()
    STAGED_PATHS = stage_files(
        candidates, LOCAL_CACHE_DIR, reserve_bytes=STAGING_RESERVE_BYTES
    )
    staged = [os.path.basename(p) for p, t in STAGED_PATHS.items() if p != t]
    print(
        f"[staging] done in {time.perf_counter() - started:.0f}s; "
        f"local: {staged or 'none (serving from network volume)'}"
    )


def resolve_weight_path(path):
    return STAGED_PATHS.get(path, path)


def get_pipeline(pipeline_name, quantization_str, offload_str):
    global current_pipeline, current_pipeline_name, current_quantization, current_offload_mode
    
    quantization_kind = QuantizationKind(quantization_str) if quantization_str else None
    offload_mode = OffloadMode(offload_str) if offload_str else OffloadMode.NONE
    
    if (current_pipeline is not None and 
        current_pipeline_name == pipeline_name and 
        current_quantization == quantization_str and 
        current_offload_mode == offload_str):
        return current_pipeline
    
    if current_pipeline is not None:
        print("Cleaning up old pipeline instance...")
        del current_pipeline
        current_pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print(f"Initializing pipeline: {pipeline_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if pipeline_name == "distilled":
        # This deployment ships only the standalone distilled checkpoint; the
        # file carries VAE and text-encoder projection weights alongside the
        # transformer, so no LoRA and no dev checkpoint are involved.
        # Resolved through the staging map: local disk when it fit, the
        # network volume otherwise. Upstream re-reads these on every job, so
        # which one wins here is the single biggest lever on job duration.
        checkpoint_path = resolve_weight_path(DISTILLED_CHECKPOINT)
        spatial_upsampler_path = resolve_weight_path(SPATIAL_UPSAMPLER)

        quantization_policy = quantization_kind.to_policy(checkpoint_path) if quantization_kind else None
        current_pipeline = DistilledPipeline(
            distilled_checkpoint_path=checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root=GEMMA_DIR,
            loras=[],
            device=device,
            quantization=quantization_policy,
            offload_mode=offload_mode,
        )
    else:
        raise ValueError(
            f"Unknown pipeline: {pipeline_name}. This deployment serves only "
            "the distilled model; omit 'pipeline' or pass 'distilled'."
        )
        
    current_pipeline_name = pipeline_name
    current_quantization = quantization_str
    current_offload_mode = offload_str
    return current_pipeline

# Upstream guards its CLI entrypoints with this, not the pipeline classes, so
# calling DistilledPipeline directly leaves autograd on. The weights load as
# inference tensors, autograd then tries to save them for a backward pass that
# will never happen, and the first F.linear of the first denoising step dies
# with "Inference tensors cannot be saved for backward". Covers construction
# as well as the call, exactly as upstream's main() does.
@torch.inference_mode()
def handler(event):
    job_input = event.get("input", {})
    if not job_input:
        return {"error": "No input configuration provided."}
        
    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "A prompt must be provided."}
        
    # negative_prompt is intentionally not read: the distilled pipeline runs
    # guidance-free on its fixed sigma schedule and has no negative-prompt input.
    seed = job_input.get("seed", 42)
    height = job_input.get("height", 512)
    width = job_input.get("width", 768)
    num_frames = job_input.get("num_frames", 49) # Default shorter for fast dev cycles
    frame_rate = job_input.get("frame_rate", 25.0)
    
    # Only the distilled pipeline exists in this deployment. It runs a fixed
    # 8-step (+3 refine) sigma schedule, so num_inference_steps and the
    # cfg/stg guidance knobs of the two-stage pipeline do not apply here.
    pipeline_name = job_input.get("pipeline", "distilled")
    quantization, offload_mode = resolve_pipeline_config(job_input)

    image_conditioning = job_input.get("image_conditioning", [])
    temp_files = []
    timer = StageTimer(event.get("id"))

    try:
        with timer.stage("pipeline_init"):
            pipe = get_pipeline(
                pipeline_name=pipeline_name,
                quantization_str=quantization,
                offload_str=offload_mode,
            )

        with timer.stage("input_download"):
            images_input = []
            for idx, img_cond in enumerate(image_conditioning):
                url = img_cond.get("url")
                frame_idx = img_cond.get("frame_idx", 0)
                strength = img_cond.get("strength", 1.0)
                if url:
                    local_path = f"/tmp/{uuid.uuid4()}_{idx}.png"
                    download_file(url, local_path)
                    temp_files.append(local_path)
                    images_input.append(ImageConditioningInput(
                        path=local_path, frame_idx=frame_idx, strength=strength
                    ))

        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

        # Covers everything upstream does per job: rebuilding the text encoder,
        # both transformer builds, the upsampler and the decoders, plus the
        # ~40s of denoising they exist to feed. The ltx-pipelines log lines
        # break this down further; this is the number to watch across configs.
        with timer.stage("generate"):
            video_gen, audio_gen = pipe(
                prompt=prompt, seed=seed, height=height, width=width,
                num_frames=num_frames, frame_rate=frame_rate, images=images_input,
                tiling_config=tiling_config
            )

        output_filename = f"/tmp/{uuid.uuid4()}.mp4"
        temp_files.append(output_filename)

        with timer.stage("encode"):
            encode_video(
                video=video_gen, fps=frame_rate, audio=audio_gen,
                output_path=output_filename, video_chunks_number=video_chunks_number
            )

        with timer.stage("upload"):
            video_url = upload_video(output_filename, job_input)

        timer.emit(
            outcome="success",
            height=height,
            width=width,
            num_frames=num_frames,
        )
        return {"status": "success", "video_url": video_url, "seed": seed}

    except Exception as e:
        import traceback
        # Log the traceback rather than only returning it: the job record is
        # not somewhere you can grep, and a failure that leaves no trace in the
        # container logs is a failure you cannot debug after the fact.
        traceback.print_exc()
        timer.emit(outcome="failed", error_type=type(e).__name__)
        return {"status": "failed", "error": str(e), "traceback": traceback.format_exc()}
    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Populate the model weights before serving. On a mounted network volume
    # (MODELS_ROOT=/runpod-volume/models) this downloads once and every later
    # cold start reuses the cached files instead of re-pulling from HuggingFace.
    ensure_models(MODELS_ROOT)

    # Then copy them onto local disk, because the network volume is the
    # bottleneck rather than the GPU. Paid once per worker, and FlashBoot
    # amortises it across every job that worker goes on to serve.
    stage_weights()

    # Construct the pipeline before accepting traffic. The object graph is
    # cheap to build (upstream defers the expensive weight loads to call time),
    # but doing it here keeps it off the first request's clock and turns a bad
    # checkpoint path into a startup failure instead of a failed job.
    get_pipeline("distilled", *resolve_pipeline_config({}))

    runpod.serverless.start({"handler": handler})
