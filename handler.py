import os
import gc
import uuid
import urllib.request
import torch
import runpod

from download_models import ensure_models

# Import LTX-2 pipeline components
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
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
    
    if pipeline_name == "two_stage":
        checkpoint_path = os.path.join(LTX_DIR, "ltx-2.3-22b-dev.safetensors")
        distilled_lora_path = os.path.join(LTX_DIR, "ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
        spatial_upsampler_path = os.path.join(LTX_DIR, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
        
        quantization_policy = quantization_kind.to_policy(checkpoint_path) if quantization_kind else None
        distilled_lora = [LoraPathStrengthAndSDOps(distilled_lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)]
        
        current_pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root=GEMMA_DIR,
            loras=[],
            device=device,
            quantization=quantization_policy,
            offload_mode=offload_mode,
        )
    elif pipeline_name == "distilled":
        distilled_checkpoint_path = os.path.join(LTX_DIR, "ltx-2.3-22b-distilled-1.1.safetensors")
        spatial_upsampler_path = os.path.join(LTX_DIR, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
        
        quantization_policy = quantization_kind.to_policy(distilled_checkpoint_path) if quantization_kind else None
        current_pipeline = DistilledPipeline(
            distilled_checkpoint_path=distilled_checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root=GEMMA_DIR,
            loras=[],
            device=device,
            quantization=quantization_policy,
            offload_mode=offload_mode,
        )
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")
        
    current_pipeline_name = pipeline_name
    current_quantization = quantization_str
    current_offload_mode = offload_str
    return current_pipeline

def handler(event):
    job_input = event.get("input", {})
    if not job_input:
        return {"error": "No input configuration provided."}
        
    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "A prompt must be provided."}
        
    negative_prompt = job_input.get("negative_prompt", "blurry, low quality, static, deformed, noisy")
    seed = job_input.get("seed", 42)
    height = job_input.get("height", 512)
    width = job_input.get("width", 768)
    num_frames = job_input.get("num_frames", 49) # Default shorter for fast dev cycles
    frame_rate = job_input.get("frame_rate", 25.0)
    
    pipeline_name = job_input.get("pipeline", "two_stage")
    quantization = job_input.get("quantization", "fp8-cast")
    offload_mode = job_input.get("offload_mode", "cpu")
    
    num_inference_steps = job_input.get("num_inference_steps", 30)
    video_cfg = job_input.get("video_cfg", 4.0)
    video_stg = job_input.get("video_stg", 2.0)
    audio_cfg = job_input.get("audio_cfg", 4.0)
    audio_stg = job_input.get("audio_stg", 2.0)
    
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
        
        if pipeline_name == "two_stage":
            video_guider = MultiModalGuiderParams(cfg_scale=video_cfg, stg_scale=video_stg)
            audio_guider = MultiModalGuiderParams(cfg_scale=audio_cfg, stg_scale=audio_stg)
            
            video_gen, audio_gen = pipe(
                prompt=prompt, negative_prompt=negative_prompt, seed=seed,
                height=height, width=width, num_frames=num_frames, frame_rate=frame_rate,
                num_inference_steps=num_inference_steps, video_guider_params=video_guider,
                audio_guider_params=audio_guider, images=images_input, tiling_config=tiling_config
            )
        else:
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
        
        video_url = runpod.upload_file(f"{uuid.uuid4()}.mp4", output_filename)
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
