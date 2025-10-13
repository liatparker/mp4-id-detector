#!/usr/bin/env python3
"""
Video ID Detector - Automatic Run Script
This script automatically runs the video ID detection system
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("Starting Video ID Detection System...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("video_id_detector2_optimized.py"):
        print("ERROR: video_id_detector2_optimized.py not found!")
        print("   Please run this script from the mp4_id_detector directory")
        sys.exit(1)
    
    # Check if videos exist
    print("Checking video files...")
    videos_dir = Path("videos")
    if not videos_dir.exists():
        print("ERROR: Videos directory not found!")
        sys.exit(1)
    
    video_files = list(videos_dir.glob("*.mp4"))
    print(f"Found {len(video_files)} video files:")
    for video in video_files:
        print(f"   - {video.name}")
    
    if len(video_files) == 0:
        print("ERROR: No video files found in videos/ directory!")
        sys.exit(1)
    
    # Check if virtual environment exists
    if not os.path.exists("cvenv"):
        print("ERROR: Virtual environment (cvenv) not found!")
        print("   Please create it first: python -m venv cvenv")
        sys.exit(1)
    
    # Run the detector
    print("Running video ID detector...")
    try:
        # Use the virtual environment's Python
        python_path = "cvenv/bin/python" if os.name != 'nt' else "cvenv\\Scripts\\python.exe"
        result = subprocess.run([python_path, "video_id_detector2_optimized.py"], 
                              check=True, capture_output=False)
        
        print("Detection completed successfully!")
        print("Results saved in outputs_v3/")
        print("Check global_identity_catalogue_v3.json for results")
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Detection failed with error code {e.returncode}!")
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: Python executable not found in virtual environment!")
        sys.exit(1)
    
    print("=" * 50)
    print("All done!")

if __name__ == "__main__":
    main()
