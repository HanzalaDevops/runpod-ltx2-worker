import os
import argparse
from huggingface_hub import hf_hub_download, snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def download_models(target_dir):
    models_dir = os.path.abspath(target_dir)
    os.makedirs(models_dir, exist_ok=True)
    ltx_dir = os.path.join(models_dir, "ltx-2.3")
    gemma_dir = os.path.join(models_dir, "gemma-3-12b")
    
    ltx_files = [
        "ltx-2.3-22b-dev.safetensors",
        "ltx-2.3-22b-distilled-1.1.safetensors",
        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
    ]
    
    token = os.getenv("HF_TOKEN")
    for filename in ltx_files:
        print(f"Downloading {filename}...")
        try:
            hf_hub_download(repo_id="Lightricks/LTX-2.3", filename=filename, local_dir=ltx_dir, token=token)
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            
    print(f"Downloading Gemma-3...")
    try:
        snapshot_download(
            repo_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
            local_dir=gemma_dir,
            token=token,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
        )
        print("Successfully downloaded Gemma-3")
    except Exception as e:
        print(f"Failed to download Gemma-3: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="/workspace/models")
    args = parser.parse_args()
    download_models(args.dir)
