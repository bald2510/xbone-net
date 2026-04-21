from huggingface_hub import hf_hub_download
import os

def download_biomed_clip():
    repo_id = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    checkpoint_path = "checkpoints/biomedclip"
    
    os.makedirs(checkpoint_path, exist_ok=True)

    # Tải trọng số (Weights)
    hf_hub_download(
        repo_id=repo_id,
        filename="open_clip_pytorch_model.bin",
        local_dir=checkpoint_path
    )
    
    # Tải cấu hình (Config)
    hf_hub_download(
        repo_id=repo_id,
        filename="open_clip_config.json",
        local_dir=checkpoint_path
    )
    print(f"Model downloaded to {checkpoint_path}")

if __name__ == "__main__":
    download_biomed_clip()