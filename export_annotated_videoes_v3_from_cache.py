"""
export_annotated_videos_v5_from_npz.py
Export annotated videos using the EXACT bboxes from the NPZ cache.
This guarantees perfect alignment with clustering results.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuration
VIDEO_DIR = "/Users/liatparker/Documents/mp4_files_id_detectors_upscaled/"
VIDEO_FILES = ["1_upscaled.mp4", "2_upscaled.mp4", "3_upscaled.mp4", "4_upscaled.mp4"]
OUTPUT_DIR = "./annotated_videos_v3/"
MAPPING_FILE = "./outputs_v3/tracklet_to_global_id.npz"
print("="*60)
print("📦 LOADING CLUSTERING RESULTS")
print("="*60)

cached = np.load(MAPPING_FILE, allow_pickle=True)
all_tracklets_raw = cached['tracklets']
global_ids_raw = cached['global_ids']

print(f"✅ Loaded {len(all_tracklets_raw)} tracklets with global IDs")

all_tracklets = []
for i, t in enumerate(all_tracklets_raw):
    if hasattr(t, 'item'):
        t_dict = t.item()
    else:
        t_dict = dict(t) if isinstance(t, dict) else t
    
    if 'global_id' not in t_dict:
        t_dict['global_id'] = int(global_ids_raw[i])
    
    all_tracklets.append(t_dict)

tracklet_to_global_id = {}
for t in all_tracklets:
    key = (t['clip_idx'], t['track_id'])
    tracklet_to_global_id[key] = t['global_id']

print(f"✅ Mapped {len(tracklet_to_global_id)} tracklets\n")
COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
    (0, 191, 255), (50, 205, 50), (255, 20, 147), (0, 206, 209),
    (255, 140, 0), (138, 43, 226), (34, 139, 34)
]

def get_color(global_id):
    return COLORS[global_id % len(COLORS)]

print("="*60)
print("📹 ANNOTATING VIDEOS FROM NPZ DATA")
print("="*60 + "\n")

# Process each video
for clip_idx, video_file in enumerate(VIDEO_FILES):
    video_path = Path(VIDEO_DIR) / video_file
    output_path = Path(OUTPUT_DIR) / f"clip{clip_idx}_annotated.mp4"
    
    print(f"🎬 Clip {clip_idx}: {video_file}")
    
    # Get all tracklets for this clip
    clip_tracklets = [t for t in all_tracklets if t['clip_idx'] == clip_idx]
    print(f"   Found {len(clip_tracklets)} tracklets")
    
    # Build frame-level detections from NPZ data
    print("   Building frame detections from NPZ...")
    frame_detections = {}  # frame_idx -> [(global_id, bbox), ...]
    
    for tracklet in clip_tracklets:
        track_id = tracklet['track_id']
        key = (clip_idx, track_id)
        global_id = tracklet_to_global_id.get(key)
        
        if global_id is None:
            continue  # Skip unmatched tracklets
        
        # Get frames and bboxes
        frames = tracklet['frames']
        bboxes = tracklet['bboxes']
        
        for frame_idx, bbox in zip(frames, bboxes):
            if frame_idx not in frame_detections:
                frame_detections[frame_idx] = []
            frame_detections[frame_idx].append((global_id, bbox))
    
    # Get video properties
    cap_temp = cv2.VideoCapture(str(video_path))
    fps = cap_temp.get(cv2.CAP_PROP_FPS)
    width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_temp.release()
    
    # Annotate video
    print("   Annotating frames...")
    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    global_ids_in_clip = set()
    
    for frame_idx in tqdm(range(total_frames), desc="   Annotating"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw detections for this frame
        if frame_idx in frame_detections:
            for global_id, bbox in frame_detections[frame_idx]:
                global_ids_in_clip.add(global_id)
                x1, y1, x2, y2 = [int(v) for v in bbox]
                color = get_color(global_id)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw label
                label = f"ID: {global_id}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw legend
        legend_y = 30
        for gid in sorted(global_ids_in_clip):
            color = get_color(gid)
            text = f"Global ID {gid}"
            cv2.putText(frame, text, (10, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            legend_y += 35
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    print(f"   ✅ Saved: {output_path}")
    print(f"   📊 Global IDs: {sorted(global_ids_in_clip)}\n")

print("="*60)
print("✅ ANNOTATION COMPLETE!")
print("="*60)
print("\n🎯 These annotations use the EXACT bboxes from clustering!")
print("Perfect alignment guaranteed! 🎉")