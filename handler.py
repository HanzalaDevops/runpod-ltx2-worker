import os
import gc
import uuid
import urllib.request
import torch
import runpod
from runpod.serverless.utils import rp_upload

from download_models import ensure_models

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


def download_file(url, target_path):
    print(f"Downloading input file from {url} to {target_path}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
        out_file.write(response.read())
    return target_path

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
        checkpoint_path = os.path.join(LTX_DIR, "ltx-2.3-22b-distilled-1.1.safetensors")
        spatial_upsampler_path = os.path.join(LTX_DIR, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")

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
    quantization = job_input.get("quantization", "fp8-cast")
    offload_mode = job_input.get("offload_mode", "cpu")

    image_conditioning = job_input.get("image_conditioning", [])
    temp_files = []
    
    try:
        pipe = get_pipeline(
            pipeline_name=pipeline_name,
            quantization_str=quantization if quantization != "none" else None,
            offload_str=offload_mode
        )
        
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

        video_gen, audio_gen = pipe(
            prompt=prompt, seed=seed, height=height, width=width,
            num_frames=num_frames, frame_rate=frame_rate, images=images_input, tiling_config=tiling_config
        )

        output_filename = f"/tmp/{uuid.uuid4()}.mp4"
        temp_files.append(output_filename)
        
        encode_video(
            video=video_gen, fps=frame_rate, audio=audio_gen,
            output_path=output_filename, video_chunks_number=video_chunks_number
        )
        
        video_url = upload_video(output_filename, job_input)
        return {"status": "success", "video_url": video_url, "seed": seed}
        
    except Exception as e:
        import traceback
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
    runpod.serverless.start({"handler": handler})
