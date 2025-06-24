#!/usr/bin/env python3
"""
Unified development runner for AI Order Taker
Runs both backend and frontend in development mode with hot reload
"""

import subprocess
import sys
import os
import threading
import time
from pathlib import Path

def run_backend():
    """Run the backend server"""
    project_root = Path(__file__).parent
    venv_path = project_root / "venv"
    backend_dir = project_root / "backend"
    
    # Check if venv exists
    if not venv_path.exists():
        print("❌ Virtual environment not found! Please run:")
        print("  python setup_env.py")
        return
    
    # Determine Python executable path
    if os.name == 'nt':  # Windows
        python_exe = venv_path / "Scripts" / "python.exe"
    else:  # Unix/Linux/macOS
        python_exe = venv_path / "bin" / "python"
    
    if not python_exe.exists():
        print(f"❌ Python executable not found at {python_exe}")
        print("Please run: python setup_env.py")
        return
    
    print(f"🐍 Starting backend with: {python_exe}")
    
    try:
        subprocess.run([
            str(python_exe), "main.py"
        ], cwd=backend_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Backend failed to start: {e}")
    except KeyboardInterrupt:
        pass

def run_frontend():
    """Run the frontend dev server"""
    project_root = Path(__file__).parent
    frontend_dir = project_root / "frontend"
    
    print("⚛️  Starting frontend dev server...")
    
    try:
        subprocess.run([
            "npm", "run", "dev"
        ], cwd=frontend_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Frontend failed to start: {e}")
    except KeyboardInterrupt:
        pass

def main():
    """Run both backend and frontend in parallel"""
    print("🚀 Starting AI Order Taker (Backend + Frontend)")
    print("🌐 Backend will be available at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs")
    print("⚛️  Frontend will be available at: http://localhost:5173")
    print("⏹️  Press Ctrl+C to stop both servers")
    print()
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    
    # Give backend a moment to start
    time.sleep(2)
    
    try:
        # Run frontend in main thread (so Ctrl+C works properly)
        run_frontend()
    except KeyboardInterrupt:
        print("\n👋 Servers stopped by user")

if __name__ == "__main__":
    main()