"""
Export annotated videos with global IDs from video_id_detector2_fast.py results
"""

import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuration
VIDEO_DIR = "/Users/liatparker/Documents/mp4_files_id_detectors_upscaled/"
VIDEO_FILES = ["1_upscaled.mp4", "2_upscaled.mp4", "3_upscaled.mp4", "4_upscaled.mp4"]
JSON_FILE = "./outputs_v2_fast/global_identity_catalogue_v2_fast.json"
OUTPUT_DIR = "./annotated_videos_v2_fast/"

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Load results
print("📖 Loading results...")
with open(JSON_FILE, 'r') as f:
    data = json.load(f)

print(f"✅ Found {data['summary']['total_global_identities']} global IDs")
print(f"   Total tracklets: {data['summary']['total_tracklets']}")

# Colors for global IDs (extended palette)
COLORS = [
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 128),    # Purple
    (255, 165, 0),    # Orange
    (0, 191, 255),    # Deep Sky Blue
    (50, 205, 50),    # Lime Green
    (255, 20, 147),   # Deep Pink
    (0, 206, 209),    # Dark Turquoise
    (255, 140, 0),    # Dark Orange
    (138, 43, 226),   # Blue Violet
    (34, 139, 34),    # Forest Green
    (220, 20, 60),    # Crimson
    (64, 224, 208),   # Turquoise
    (255, 105, 180),  # Hot Pink
    (107, 142, 35),   # Olive Drab
    (0, 100, 0),      # Dark Green
    (139, 0, 139),    # Dark Magenta
    (184, 134, 11),   # Dark Goldenrod
    (0, 128, 128),    # Teal
    (128, 128, 0),    # Olive
]

def get_color(global_id):
    """Get color for global ID"""
    return COLORS[global_id % len(COLORS)]

# Build frame-to-global-id mapping for each clip
print("\n📊 Building frame mappings...")
clip_mappings = {i: {} for i in range(len(VIDEO_FILES))}

for gid_key, identity_data in data['identities'].items():
    global_id = identity_data['global_id']
    
    for appearance in identity_data['appearances']:
        clip_idx = appearance['clip_idx']
        start_frame = appearance['start_frame']
        end_frame = appearance['end_frame']
        
        # Map all frames in this appearance to this global_id
        for frame_num in range(start_frame, end_frame + 1):
            if frame_num not in clip_mappings[clip_idx]:
                clip_mappings[clip_idx][frame_num] = []
            clip_mappings[clip_idx][frame_num].append(global_id)

# Process each video
print("\n📹 Annotating videos...\n")

for clip_idx, video_file in enumerate(VIDEO_FILES):
    video_path = Path(VIDEO_DIR) / video_file
    output_path = Path(OUTPUT_DIR) / f"clip{clip_idx}_annotated_v2.mp4"
    
    print(f"🎬 Processing Clip {clip_idx}: {video_file}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Cannot open {video_path}")
        continue
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc=f"Clip {clip_idx}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get global IDs present in this frame
        if frame_idx in clip_mappings[clip_idx]:
            global_ids = clip_mappings[clip_idx][frame_idx]
            unique_ids = sorted(set(global_ids))
            
            # Draw legend/labels at top
            y_offset = 30
            for gid in unique_ids:
                color = get_color(gid)
                text = f"Global ID: {gid}"
                
                # Background rectangle
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (10, y_offset - 20), (10 + text_w + 10, y_offset + 5), color, -1)
                
                # Text
                cv2.putText(frame, text, (15, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                y_offset += 35
        
        out.write(frame)
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"   ✅ Saved: {output_path}")

print("\n" + "="*60)
print("✅ ALL VIDEOS ANNOTATED!")
print("="*60)
print(f"📁 Output directory: {OUTPUT_DIR}")
print(f"📊 Total global IDs: {data['summary']['total_global_identities']}")
print("="*60 + "\n")