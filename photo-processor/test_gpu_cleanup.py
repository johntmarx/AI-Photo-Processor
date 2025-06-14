#!/usr/bin/env python3
"""
Test script to verify GPU memory cleanup functionality
"""
import requests
import time
import json

API_BASE = "http://localhost:8000/api"

def check_gpu_status():
    """Check current GPU memory status"""
    response = requests.get(f"{API_BASE}/stats/gpu/status")
    if response.ok:
        status = response.json()
        print("\n📊 GPU Status:")
        if status.get("gpu_available"):
            print(f"  Device: {status['device_name']}")
            print(f"  Total Memory: {status['memory_total_gb']:.2f} GB")
            print(f"  Free Memory: {status['memory_free_gb']:.2f} GB")
            print(f"  Allocated: {status['memory_allocated_gb']:.2f} GB")
            print(f"  Reserved: {status['memory_reserved_gb']:.2f} GB")
        else:
            print("  No GPU available")
        return status
    else:
        print(f"❌ Failed to get GPU status: {response.status_code}")
        return None

def trigger_gpu_cleanup():
    """Trigger GPU memory cleanup"""
    print("\n🧹 Triggering GPU cleanup...")
    response = requests.post(f"{API_BASE}/stats/gpu/cleanup")
    if response.ok:
        result = response.json()
        print(f"✅ Cleanup task started: Task ID {result['task_id']}")
        cleanup_result = result['result']
        if cleanup_result['success']:
            print(f"✅ GPU cleanup successful!")
            print(f"  Allocated after cleanup: {cleanup_result['allocated_gb']:.2f} GB")
            print(f"  Reserved after cleanup: {cleanup_result['reserved_gb']:.2f} GB")
        else:
            print(f"❌ Cleanup failed: {cleanup_result.get('error', 'Unknown error')}")
        return result
    else:
        print(f"❌ Failed to trigger cleanup: {response.status_code}")
        return None

def main():
    print("GPU Memory Cleanup Test")
    print("=" * 60)
    
    # Check initial status
    print("\n1️⃣ Initial GPU Status:")
    initial = check_gpu_status()
    
    # Wait a moment
    time.sleep(2)
    
    # Trigger cleanup
    print("\n2️⃣ Triggering GPU Cleanup:")
    cleanup = trigger_gpu_cleanup()
    
    # Wait for cleanup to complete
    time.sleep(3)
    
    # Check final status
    print("\n3️⃣ Final GPU Status:")
    final = check_gpu_status()
    
    # Compare
    if initial and final and initial.get("gpu_available") and final.get("gpu_available"):
        freed = final['memory_free_gb'] - initial['memory_free_gb']
        print(f"\n📈 Memory freed: {freed:.2f} GB")

if __name__ == "__main__":
    main()