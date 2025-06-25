#!/usr/bin/env python3
"""
Process cleanup script for AI Order Taker development

Kills any stray backend processes and clears GPU memory.
Useful when multiple instances get stuck running.
"""

import os
import signal
import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run a command and capture output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.returncode
    except Exception as e:
        print(f"Error running command '{cmd}': {e}")
        return "", 1

def get_python_processes():
    """Get all Python processes running main.py"""
    output, _ = run_command("pgrep -f 'python.*main.py'")
    if output:
        return [int(pid) for pid in output.split('\n') if pid]
    return []

def kill_processes(pids):
    """Kill processes by PID"""
    killed = []
    for pid in pids:
        try:
            # Try graceful termination first
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
            print(f"Sent SIGTERM to process {pid}")
        except ProcessLookupError:
            print(f"Process {pid} already dead")
        except PermissionError:
            print(f"No permission to kill process {pid}")
    
    # Wait a bit, then force kill if needed
    import time
    time.sleep(2)
    
    for pid in killed:
        try:
            # Check if still alive, force kill
            os.kill(pid, 0)  # Test if process exists
            os.kill(pid, signal.SIGKILL)
            print(f"Force killed process {pid}")
        except ProcessLookupError:
            print(f"Process {pid} terminated gracefully")
        except PermissionError:
            print(f"No permission to force kill process {pid}")

def clear_gpu_memory():
    """Clear GPU memory cache"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU memory cache cleared")
            return True
    except ImportError:
        print("PyTorch not available, skipping GPU cleanup")
    return False

def get_gpu_memory():
    """Get current GPU memory usage"""
    output, code = run_command("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits")
    if code == 0 and output:
        lines = output.split('\n')
        if lines:
            used, total = lines[0].split(', ')
            return int(used), int(total)
    return None, None

def main():
    print("🧹 AI Order Taker Process Cleanup")
    print("=" * 40)
    
    # Show current GPU usage
    used, total = get_gpu_memory()
    if used and total:
        print(f"📊 Current GPU memory: {used}MB / {total}MB ({used/total*100:.1f}%)")
    
    # Find Python processes
    pids = get_python_processes()
    if not pids:
        print("✅ No stray Python processes found")
    else:
        print(f"🔍 Found {len(pids)} Python processes: {pids}")
        
        # Confirm before killing
        if len(sys.argv) > 1 and sys.argv[1] == "--force":
            kill_processes(pids)
        else:
            response = input("Kill these processes? (y/N): ")
            if response.lower() in ['y', 'yes']:
                kill_processes(pids)
            else:
                print("Skipping process cleanup")
    
    # Clear GPU memory
    if clear_gpu_memory():
        # Show updated GPU usage
        used, total = get_gpu_memory()
        if used and total:
            print(f"📊 Updated GPU memory: {used}MB / {total}MB ({used/total*100:.1f}%)")
    
    print("✅ Cleanup complete!")

if __name__ == "__main__":
    main()