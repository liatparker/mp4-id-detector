#!/usr/bin/env python3
"""
export_annotated_videos.py

Export videos with bounding boxes showing global_id labels.
This version re-runs detection and matches to global_ids using the catalogue.

Usage:
    python export_annotated_videos.py                    # Process all clips
    python export_annotated_videos.py --clip 0           # Process only clip 0
    python export_annotated_videos.py --clip 0 --frames 1-100  # Specific frame range
"""

import os
import json
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# -----------------------
# Configuration
# -----------------------
dir_path = "/Users/liatparker/Documents/mp4_files_id_detectors/"
VIDEO_FILES = [dir_path+"1.mp4", dir_path+"2.mp4", dir_path+"3.mp4", dir_path+"4.mp4"]
JSON_FILE = "global_identity_catalogue.json"
OUTPUT_DIR = "annotated_videos"
YOLO_MODEL = "yolov8n.pt"

# Colors for different global IDs (BGR format)
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
]

def get_color(global_id):
    """Get color for a global ID"""
    if global_id == -1:
        return (128, 128, 128)  # Gray for unknown
    return COLORS[global_id % len(COLORS)]

def load_catalogue(json_file):
    """Load the global identity catalogue"""
    with open(json_file, 'r') as f:
        return json.load(f)

def build_frame_to_gid_map(data):
    """
    Build a map of {clip_id: {frame_num: [global_ids]}}
    """
    frame_map = {}
    identities = data.get("identities", {})
    
    for gid_key, person in identities.items():
        global_id = int(gid_key.split('_')[-1])
        
        for appearance in person["appearances"]:
            clip_id = appearance["clip_id"]
            start_frame, end_frame = appearance["frame_range"]
            
            if clip_id not in frame_map:
                frame_map[clip_id] = {}
            
            for frame_num in range(start_frame, end_frame + 1):
                if frame_num not in frame_map[clip_id]:
                    frame_map[clip_id][frame_num] = []
                frame_map[clip_id][frame_num].append(global_id)
    
    return frame_map

def draw_bbox(frame, bbox, global_id, conf=None):
    """Draw bounding box with label"""
    x1, y1, x2, y2 = map(int, bbox)
    color = get_color(global_id)
    
    # Draw box
    thickness = 3 if global_id != -1 else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Prepare label
    if global_id == -1:
        label = "Unknown"
    else:
        label = f"ID {global_id}"
    
    if conf is not None:
        label += f" {conf:.2f}"
    
    # Draw label background
    (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - label_h - 12), (x1 + label_w + 10, y1), color, -1)
    
    # Draw label text
    cv2.putText(frame, label, (x1 + 5, y1 - 7), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def process_clip(clip_id, video_path, frame_map, output_path, frame_range=None):
    """Process a single video clip"""
    
    print(f"\n📹 Processing Clip {clip_id}: {os.path.basename(video_path)}")
    
    # Load YOLO
    yolo = YOLO(YOLO_MODEL)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open {video_path}")
        return False
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   Resolution: {width}x{height}, FPS: {fps}")
    
    # Parse frame range
    start_frame = 1
    end_frame = total_frames
    if frame_range:
        parts = frame_range.split('-')
        if len(parts) == 2:
            start_frame = int(parts[0])
            end_frame = int(parts[1])
            end_frame = min(end_frame, total_frames)
    
    print(f"   Processing frames {start_frame}-{end_frame} (of {total_frames})")
    
    # Create output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Get frame map for this clip
    clip_frame_map = frame_map.get(clip_id, {})
    
    # Process frames
    frame_num = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
    
    pbar = tqdm(range(start_frame, end_frame + 1), desc=f"Clip {clip_id}")
    
    for _ in pbar:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        current_frame = start_frame + frame_num - 1
        
        # Run YOLO detection (person class = 0)
        results = yolo(frame, classes=[0], verbose=False, conf=0.3)
        
        # Get detections
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            detections = list(zip(xyxy, confs))
        
        # Get global_ids for this frame
        global_ids_in_frame = clip_frame_map.get(current_frame, [])
        
        # Match detections to global_ids
        # Simple strategy: assign by order (leftmost to rightmost)
        if len(detections) > 0:
            # Sort detections left to right
            detections.sort(key=lambda x: x[0][0])  # Sort by x1
            
            for i, (bbox, conf) in enumerate(detections):
                if i < len(global_ids_in_frame):
                    gid = global_ids_in_frame[i]
                else:
                    gid = -1  # Unknown
                
                draw_bbox(frame, bbox, gid, conf)
        
        # Draw frame info
        info_text = f"Clip {clip_id} | Frame {current_frame}"
        if len(global_ids_in_frame) > 0:
            info_text += f" | IDs: {global_ids_in_frame}"
        
        cv2.rectangle(frame, (5, 5), (width - 5, 45), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
    
    cap.release()
    out.release()
    pbar.close()
    
    print(f"   ✅ Saved: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Export annotated videos with global_id labels")
    parser.add_argument("--clip", type=int, default=None, help="Process specific clip (0-3)")
    parser.add_argument("--frames", type=str, default=None, help="Frame range (e.g. '1-100')")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--json", default=JSON_FILE, help="JSON catalogue file")
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎬 VIDEO ANNOTATION WITH GLOBAL IDs")
    print("=" * 70)
    
    # Load catalogue
    print(f"\n📂 Loading {args.json}...")
    try:
        data = load_catalogue(args.json)
        summary = data.get("summary", {})
        print(f"✅ Catalogue loaded:")
        print(f"   {summary.get('total_global_ids', 0)} global IDs")
        print(f"   {summary.get('total_appearances', 0)} appearances")
        if summary.get('noise_filtered', 0) > 0:
            print(f"   {summary.get('noise_filtered', 0)} noise points filtered")
    except FileNotFoundError:
        print(f"❌ Error: {args.json} not found!")
        print("   Run: python cluster_embeddings.py --eps 0.4 --min_samples 1")
        return
    
    # Build frame map
    print("\n🗺️  Building frame-to-ID mapping...")
    frame_map = build_frame_to_gid_map(data)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 Output directory: {args.output_dir}/")
    
    # Determine which clips to process
    if args.clip is not None:
        clips = [args.clip]
    else:
        clips = range(len(VIDEO_FILES))
    
    # Process each clip
    success_count = 0
    for clip_id in clips:
        if clip_id >= len(VIDEO_FILES):
            print(f"\n⚠️  Warning: Clip {clip_id} out of range (max: {len(VIDEO_FILES)-1})")
            continue
        
        video_path = VIDEO_FILES[clip_id]
        if not os.path.exists(video_path):
            print(f"\n⚠️  Warning: Video not found: {video_path}")
            continue
        
        # Output filename
        if args.frames:
            output_filename = f"clip_{clip_id}_frames_{args.frames.replace('-', '_')}_annotated.mp4"
        else:
            output_filename = f"clip_{clip_id}_annotated.mp4"
        
        output_path = os.path.join(args.output_dir, output_filename)
        
        # Process
        if process_clip(clip_id, video_path, frame_map, output_path, args.frames):
            success_count += 1
    
    # Summary
    print("\n" + "=" * 70)
    print(f"✅ Done! Processed {success_count} clip(s)")
    print(f"📁 Output: {os.path.abspath(args.output_dir)}/")
    print("=" * 70)
    
    # Color legend
    print("\n🎨 Color Legend:")
    for i in range(min(summary.get('total_global_ids', 0), len(COLORS))):
        color_name = ["Green", "Blue", "Red", "Cyan", "Magenta", "Yellow", "Purple", "Orange", "Sky Blue", "Lime"][i]
        print(f"   ID {i}: {color_name}")
    print()

if __name__ == "__main__":
    main()

