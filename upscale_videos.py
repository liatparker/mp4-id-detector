#!/usr/bin/env python3
"""
Super-Resolution Video Upscaler
Upscale videos 2x using EDSR or bicubic interpolation.
Creates permanent high-resolution videos for all downstream tasks.

Usage:
    python3 upscale_videos.py
    
Output:
    Input:  1.mp4 (768x432)
    Output: 1_upscaled.mp4 (1536x864)
"""

import cv2
import os
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

# Input videos (original resolution)
INPUT_DIR = "/Users/liatparker/Documents/mp4_files_id_detectors"
INPUT_FILES = ["1.mp4", "2.mp4", "3.mp4", "4.mp4"]

# Output directory for upscaled videos
OUTPUT_DIR = "/Users/liatparker/Documents/mp4_files_id_detectors_upscaled"

# Super-resolution model
SR_MODEL = "EDSR_x2.pb"  # Download from: https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb
USE_SR_MODEL = True      # Set to False to use bicubic (faster but lower quality)

# Video encoding settings
OUTPUT_CODEC = 'mp4v'    # 'mp4v' for .mp4, 'XVID' for .avi
OUTPUT_FPS = None        # None = use original FPS

# ============================================================
# SUPER-RESOLUTION FUNCTIONS
# ============================================================

def load_sr_model():
    """Load EDSR super-resolution model (2x upscaling)."""
    if not USE_SR_MODEL:
        print("⚠️  SR model disabled, using bicubic interpolation")
        return None
    
    if not os.path.exists(SR_MODEL):
        print(f"⚠️  SR model not found: {SR_MODEL}")
        print("   Downloading instructions:")
        print("   wget https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb")
        print("   Using bicubic fallback...")
        return None
    
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(SR_MODEL)
        sr.setModel("edsr", 2)  # 2x upscale
        print(f"✅ Loaded SR model: {SR_MODEL}")
        return sr
    except Exception as e:
        print(f"⚠️  Failed to load SR model: {e}")
        print("   Using bicubic fallback...")
        return None

def upscale_frame(frame, sr_model=None):
    """
    Upscale a single frame 2x.
    Uses SR model if available, otherwise bicubic interpolation.
    """
    if sr_model is not None:
        try:
            return sr_model.upsample(frame)
        except Exception:
            pass
    
    # Fallback to bicubic
    h, w = frame.shape[:2]
    return cv2.resize(frame, (w*2, h*2), interpolation=cv2.INTER_CUBIC)

# ============================================================
# VIDEO PROCESSING
# ============================================================

def upscale_video(input_path, output_path, sr_model=None):
    """
    Upscale entire video 2x and save to output path.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {Path(input_path).name}")
    print(f"{'='*60}")
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Failed to open: {input_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) if OUTPUT_FPS is None else OUTPUT_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Input:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.1f}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.1f}s")
    
    # Output resolution (2x)
    out_width = width * 2
    out_height = height * 2
    
    print(f"\n🎬 Output:")
    print(f"   Resolution: {out_width}x{out_height} (2x upscale)")
    print(f"   Method: {'EDSR SR model' if sr_model else 'Bicubic interpolation'}")
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_CODEC)
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    if not out.isOpened():
        print(f"❌ Failed to create output video: {output_path}")
        cap.release()
        return False
    
    # Process frames
    print(f"\n⚙️  Processing frames...")
    pbar = tqdm(total=total_frames, desc="Upscaling", unit="frames")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Upscale frame
        upscaled_frame = upscale_frame(frame, sr_model)
        
        # Write to output
        out.write(upscaled_frame)
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"✅ Saved: {output_path}")
    print(f"   Processed: {frame_count}/{total_frames} frames")
    
    # Verify output file
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   File size: {size_mb:.1f} MB")
        return True
    else:
        print(f"❌ Output file not created!")
        return False

# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("🎬 VIDEO SUPER-RESOLUTION UPSCALER")
    print("="*60)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Files to process: {len(INPUT_FILES)}")
    print("="*60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✅ Output directory ready: {OUTPUT_DIR}\n")
    
    # Load SR model
    sr_model = load_sr_model()
    
    # Process each video
    success_count = 0
    total_time = 0
    
    for i, filename in enumerate(INPUT_FILES, 1):
        input_path = os.path.join(INPUT_DIR, filename)
        
        # Check if input exists
        if not os.path.exists(input_path):
            print(f"\n⚠️  Video {i}/{len(INPUT_FILES)}: {filename} not found, skipping...")
            continue
        
        # Create output filename
        name_without_ext = Path(filename).stem
        output_filename = f"{name_without_ext}_upscaled.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Skip if already exists
        if os.path.exists(output_path):
            print(f"\n⚠️  Video {i}/{len(INPUT_FILES)}: {output_filename} already exists, skipping...")
            continue
        
        # Upscale video
        import time
        start_time = time.time()
        
        success = upscale_video(input_path, output_path, sr_model)
        
        elapsed = time.time() - start_time
        total_time += elapsed
        
        if success:
            success_count += 1
            print(f"   Time: {elapsed:.1f}s")
    
    # Summary
    print(f"\n{'='*60}")
    print("🎉 UPSCALING COMPLETE!")
    print(f"{'='*60}")
    print(f"Processed: {success_count}/{len(INPUT_FILES)} videos")
    print(f"Total time: {total_time/60:.1f} minutes")
    if success_count > 0:
        print(f"Avg time per video: {total_time/success_count:.1f}s")
    print(f"\n📁 Upscaled videos saved to:")
    print(f"   {OUTPUT_DIR}")
    print(f"\n💡 Next steps:")
    print(f"   1. Update VIDEO_FILES in video_id_detector1.py to use upscaled videos")
    print(f"   2. Update video paths in crime_no_crime_zero_shot.py")
    print(f"   3. Re-run all tasks with improved resolution!")
    print("="*60)

if __name__ == "__main__":
    main()

