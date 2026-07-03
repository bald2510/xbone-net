"""
BiomedCLIP Model Weights Downloader.
===============================================================================
Downloads pre-trained BiomedCLIP model weights and configurations:
  - Fetches open_clip_pytorch_model.bin and open_clip_config.json from Hugging Face.
  - Saves weight files to checkpoints/biomedclip directory for offline model building.
"""

import os
from huggingface_hub import hf_hub_download


# ============================================================
# Weight Downloading Core
# ============================================================

def download_biomed_clip():
    """Download BiomedCLIP binary weights and config JSON from Hugging Face Hub.

    Returns:
        None
    """
    repo_id = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    checkpoint_path = "checkpoints/biomedclip"
    
    os.makedirs(checkpoint_path, exist_ok=True)

    hf_hub_download(
        repo_id=repo_id,
        filename="open_clip_pytorch_model.bin",
        local_dir=checkpoint_path
    )
    
    hf_hub_download(
        repo_id=repo_id,
        filename="open_clip_config.json",
        local_dir=checkpoint_path
    )
    print(f"Model downloaded to {checkpoint_path}")


if __name__ == "__main__":
    download_biomed_clip()