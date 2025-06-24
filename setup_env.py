#!/usr/bin/env python3
"""
Setup Python virtual environment for AI Order Taker
Creates venv and installs dependencies
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and handle errors"""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def setup_venv():
    """Setup Python virtual environment"""
    print("🐍 Setting up Python virtual environment...")
    
    project_root = Path(__file__).parent
    venv_path = project_root / "venv"
    
    # Create virtual environment
    if venv_path.exists():
        print(f"Virtual environment already exists at {venv_path}")
    else:
        print("Creating virtual environment...")
        if not run_command(f"{sys.executable} -m venv venv"):
            print("❌ Failed to create virtual environment!")
            return False
    
    # Determine activation script path
    if os.name == 'nt':  # Windows
        activate_script = venv_path / "Scripts" / "activate"
        python_exe = venv_path / "Scripts" / "python.exe"
        pip_exe = venv_path / "Scripts" / "pip.exe"
    else:  # Unix/Linux/macOS
        activate_script = venv_path / "bin" / "activate"
        python_exe = venv_path / "bin" / "python"
        pip_exe = venv_path / "bin" / "pip"
    
    # Install backend dependencies
    print("\n📦 Installing backend dependencies...")
    requirements_path = project_root / "backend" / "requirements.txt"
    if not run_command(f"{pip_exe} install -r {requirements_path}"):
        print("❌ Failed to install backend dependencies!")
        return False
    
    # Install frontend dependencies
    print("\n📦 Installing frontend dependencies...")
    frontend_path = project_root / "frontend"
    if not run_command("npm install", cwd=frontend_path):
        print("❌ Failed to install frontend dependencies!")
        return False
    
    print("\n✅ Environment setup complete!")
    print("\n🚀 To activate the virtual environment:")
    
    if os.name == 'nt':  # Windows
        print(f"  venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print(f"  source venv/bin/activate")
    
    print("\n🎯 Quick start commands:")
    print("  python run_dev.py        # Development mode")
    print("  python start_production.py  # Production mode")
    
    return True

if __name__ == "__main__":
    success = setup_venv()
    exit(0 if success else 1)