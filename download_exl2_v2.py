#!/usr/bin/env python3
"""
Download a proper EXL2 model for ExLlamaV2
"""

from huggingface_hub import snapshot_download, list_repo_files
import os
from pathlib import Path

def find_exl2_models():
    """Search for available EXL2 models"""
    
    # Known EXL2 model repositories
    exl2_repos = [
        "turboderp/Llama-3-8B-Instruct-exl2",  # Test with a known working model first
        "bartowski/Phi-3-mini-4k-instruct-exl2",
        "LoneStriker/Phi-3-medium-4k-instruct-4.0bpw-h6-exl2",
        "bartowski/Phi-3-medium-4k-instruct-exl2"
    ]
    
    for repo in exl2_repos:
        print(f"🔍 Checking {repo}...")
        try:
            files = list_repo_files(repo)
            exl2_files = [f for f in files if f.endswith('.safetensors') or 'config.json' in f]
            if exl2_files:
                print(f"✅ Found EXL2 model: {repo}")
                print(f"   Files: {exl2_files[:5]}")
                return repo
        except Exception as e:
            print(f"❌ {repo}: {e}")
    
    return None

def download_exl2_model():
    """Download EXL2 model"""
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Try to find a working EXL2 model
    model_repo = find_exl2_models()
    
    if not model_repo:
        print("❌ No EXL2 models found. Let's try to create one ourselves...")
        return None
    
    print(f"🔧 Downloading EXL2 model: {model_repo}")
    
    try:
        model_path = snapshot_download(
            repo_id=model_repo,
            local_dir=str(models_dir / "exl2_model"),
            resume_download=True
        )
        
        print(f"✅ Model downloaded successfully to: {model_path}")
        
        # List downloaded files
        files = list(Path(model_path).glob("*"))
        print(f"📁 Downloaded files: {len(files)}")
        for file in files:
            print(f"   - {file.name} ({file.stat().st_size / (1024*1024):.1f}MB)")
            
        return model_path
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return None

if __name__ == "__main__":
    download_exl2_model()