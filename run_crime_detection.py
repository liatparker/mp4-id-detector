#!/usr/bin/env python3
"""
Crime Detection System - Automatic Run Script
This script automatically runs the crime detection system on all videos
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🚨 Starting Crime Detection System...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("crime_no_crime_zero_shot1.py"):
        print("❌ crime_no_crime_zero_shot1.py not found!")
        print("   Please run this script from the mp4_id_detector directory")
        sys.exit(1)
    
    # Check if videos exist
    print("🔍 Checking video files...")
    videos_dir = Path("videos")
    if not videos_dir.exists():
        print("❌ Videos directory not found!")
        sys.exit(1)
    
    video_files = list(videos_dir.glob("*.mp4"))
    print(f"📹 Found {len(video_files)} video files:")
    for video in video_files:
        print(f"   - {video.name}")
    
    if len(video_files) == 0:
        print("❌ No video files found in videos/ directory!")
        sys.exit(1)
    
    # Check if virtual environment exists
    if not os.path.exists("cvenv"):
        print("❌ Virtual environment (cvenv) not found!")
        print("   Please create it first: python -m venv cvenv")
        sys.exit(1)
    
    # Run the crime detector
    print("🎯 Running crime detection on all videos...")
    try:
        # Use the virtual environment's Python
        python_path = "cvenv/bin/python" if os.name != 'nt' else "cvenv\\Scripts\\python.exe"
        result = subprocess.run([
            python_path, 
            "crime_no_crime_zero_shot1.py", 
            "--batch", "./videos",
            "--sample-rate", "30",
            "--threshold", "0.5"
        ], check=True, capture_output=False)
        
        print("✅ Crime detection completed successfully!")
        print("📁 Results saved as clip*_crime_validated.json files")
        print("📊 Check the JSON files for detailed crime analysis")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Crime detection failed with error code {e.returncode}!")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Python executable not found in virtual environment!")
        sys.exit(1)
    
    print("=" * 50)
    print("🎉 All done!")

if __name__ == "__main__":
    main()
