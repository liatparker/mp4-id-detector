"""
Export annotated videos with bounding boxes and global IDs
Re-detects and tracks, then overlays global ID labels on bounding boxes
"""

import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv

# Configuration
VIDEO_DIR = "/Users/liatparker/Documents/mp4_files_id_detectors_upscaled/"
VIDEO_FILES = ["1_upscaled.mp4", "2_upscaled.mp4", "3_upscaled.mp4", "4_upscaled.mp4"]
JSON_FILE = "./outputs_v3/global_identity_catalogue_v3.json"  # 🔥 Updated path
OUTPUT_DIR = "./annotated_videos_v3/"  # 🔥 Updated output directory
YOLO_WEIGHTS = "yolov8n.pt"

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Load YOLO
print("📦 Loading YOLO...")
yolo = YOLO(YOLO_WEIGHTS)

# Load results
print("📖 Loading clustering results...")
with open(JSON_FILE, 'r') as f:
    data = json.load(f)

print(f"✅ Found {data['summary']['total_global_identities']} global IDs")

# Colors for global IDs
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

def build_track_to_global_id_map(clip_idx):
    """
    Build mapping from (local_track_id, frame_range) -> global_id
    
    Strategy:
    - For each global ID appearance, we know: clip_idx, start_frame, end_frame
    - We'll re-track the video and match tracks by frame overlap
    """
    appearances = []
    
    for gid_key, identity_data in data['identities'].items():
        global_id = identity_data['global_id']
        
        for appearance in identity_data['appearances']:
            if appearance['clip_idx'] == clip_idx:
                appearances.append({
                    'global_id': global_id,
                    'start_frame': appearance['start_frame'],
                    'end_frame': appearance['end_frame']
                })
    
    return appearances

def find_global_id_for_track(track_id, frame_idx, track_history, appearances):
    """
    Find global ID for a track at a given frame.
    
    Logic:
    - Check when this track_id first appeared (track_history)
    - Find appearance that overlaps with track's frame range
    - Return matching global_id
    """
    # Get this track's frame range
    if track_id not in track_history:
        return None
    
    track_start = track_history[track_id]['start']
    track_end = frame_idx  # Current frame
    
    # Find best matching appearance (most overlap)
    best_match = None
    best_overlap = 0
    
    for appearance in appearances:
        app_start = appearance['start_frame']
        app_end = appearance['end_frame']
        
        # Calculate overlap
        overlap_start = max(track_start, app_start)
        overlap_end = min(track_end, app_end)
        overlap = max(0, overlap_end - overlap_start + 1)
        
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = appearance['global_id']
    
    # Require significant overlap (at least 50% of track duration)
    track_duration = track_end - track_start + 1
    if best_match is not None and best_overlap >= track_duration * 0.3:
        return best_match
    
    return None

# Process each video
print("\n📹 Annotating videos with bounding boxes...\n")

for clip_idx, video_file in enumerate(VIDEO_FILES):
    video_path = Path(VIDEO_DIR) / video_file
    output_path = Path(OUTPUT_DIR) / f"clip{clip_idx}_annotated_v2_boxes.mp4"
    
    print(f"🎬 Processing Clip {clip_idx}: {video_file}")
    
    # Get appearances for this clip
    appearances = build_track_to_global_id_map(clip_idx)
    print(f"   Found {len(appearances)} tracklet appearances")
    
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
    
        # Run YOLO tracking (same as clustering script)
    print("   Running detection and tracking...")
    results_generator = yolo.track(
        source=str(video_path),
        classes=[0],  # person class
        stream=True,
        persist=True,
        tracker="bytetrack_tuned.yaml",  # 🔥 Use same config as clustering
        verbose=False
    )
    
    track_history = {}
    global_ids_in_clip = set()  # 🔥 Track which global IDs appear in this clip
    frame_idx = 0
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print("   Annotating frames...")
    for result in tqdm(results_generator, total=total_frames, desc=f"   Clip {clip_idx}"):
        ret, frame = cap.read()
        if not ret:
            break
        
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            
            for i, track_id in enumerate(track_ids):
                bbox = boxes[i]
                x1, y1, x2, y2 = [int(v) for v in bbox]
                
                # Track history
                if track_id not in track_history:
                    track_history[track_id] = {'start': frame_idx, 'end': frame_idx}
                else:
                    track_history[track_id]['end'] = frame_idx
                
                # Find global ID for this track
                global_id = find_global_id_for_track(track_id, frame_idx, track_history, appearances)
                
                if global_id is not None:
                    global_ids_in_clip.add(global_id)
                    color = get_color(global_id)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with background
                    label = f"ID: {global_id}"
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    

                    
                    # Background rectangle
                    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
                    
                    # Text
                    cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    # Unknown track - draw gray box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
                    label = "?"
                    cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        
        # Draw legend at top
        legend_y = 30
        for gid in sorted(global_ids_in_clip):
            color = get_color(gid)
            text = f"Global ID {gid}"
            cv2.putText(frame, text, (10, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            legend_y += 25
        
        out.write(frame)
        frame_idx += 1

    

    cap.release()
    out.release()
    
    print(f"   ✅ Saved: {output_path}")
    print(f"   📊 Global IDs in this clip: {sorted(global_ids_in_clip)}")

print("\n" + "="*60)
print("✅ ALL VIDEOS ANNOTATED WITH BOUNDING BOXES!")
print("="*60)
print(f"📁 Output directory: {OUTPUT_DIR}")
print(f"📊 Total global IDs: {data['summary']['total_global_identities']}")
print("="*60 + "\n")