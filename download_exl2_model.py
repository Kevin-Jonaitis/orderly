#!/usr/bin/env python3
"""
Download ExLlamaV2 compatible model for performance comparison
"""

from huggingface_hub import snapshot_download
import os
from pathlib import Path

def download_exl2_model():
    """Download Phi-3-Medium EXL2 model"""
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Model repository that has EXL2 format
    # Using a well-known quantized version of Phi-3-Medium
    model_repo = "turboderp/Phi-3-medium-4k-instruct-exl2"
    
    print(f"🔧 Downloading EXL2 model: {model_repo}")
    print("📦 This may take a while (several GB)...")
    
    try:
        # Download the model
        model_path = snapshot_download(
            repo_id=model_repo,
            local_dir=str(models_dir / "Phi-3-medium-4k-instruct-exl2"),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print(f"✅ Model downloaded successfully to: {model_path}")
        
        # List downloaded files
        files = list(Path(model_path).glob("*"))
        print(f"📁 Downloaded files: {len(files)}")
        for file in files[:10]:  # Show first 10 files
            print(f"   - {file.name}")
        if len(files) > 10:
            print(f"   ... and {len(files) - 10} more files")
            
        return model_path
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        
        # Try alternative smaller model if main one fails
        print("🔄 Trying alternative model...")
        alt_repo = "bartowski/Phi-3-medium-4k-instruct-GGUF"
        try:
            model_path = snapshot_download(
                repo_id=alt_repo,
                local_dir=str(models_dir / "Phi-3-medium-4k-instruct-alt"),
                allow_patterns=["*Q4_K_M.gguf", "*.json"],
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"✅ Alternative model downloaded to: {model_path}")
            return model_path
        except Exception as e2:
            print(f"❌ Failed to download alternative model: {e2}")
            return None

if __name__ == "__main__":
    download_exl2_model()