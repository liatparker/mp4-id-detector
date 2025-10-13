#!/usr/bin/env python3
"""
Run Video Annotation Script

This script automatically runs the video annotation process.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the annotation process."""
    print("=" * 50)
    print("VIDEO ANNOTATION SCRIPT")
    print("=" * 50)
    
    # Check if results exist
    results_dir = Path("./outputs_clean/")
    if not results_dir.exists():
        print("Error: Results directory not found. Run the ID detector first.")
        sys.exit(1)
    
    # Check if videos exist
    video_dir = Path("./videos/")
    if not video_dir.exists():
        print("Error: Videos directory not found.")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path("./annotated/")
    output_dir.mkdir(exist_ok=True)
    
    print("Starting video annotation...")
    print(f"Video directory: {video_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Run the annotation script
        cmd = [
            "python", "annotate_video_advanced.py",
            "--video-dir", str(video_dir),
            "--results", str(results_dir),
            "--output-dir", str(output_dir)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("Annotation completed successfully!")
        print("\nOutput:")
        print(result.stdout)
        
        # List created files
        if output_dir.exists():
            print(f"\nAnnotated videos saved in: {output_dir}")
            print("Files created:")
            for file in output_dir.iterdir():
                if file.is_file():
                    print(f"  - {file.name}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running annotation: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
