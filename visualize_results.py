#!/usr/bin/env python3
"""
visualize_results.py

Export videos with bounding boxes and global_id labels to visually verify clustering results.

Usage:
    python visualize_results.py                          # Process all videos
    python visualize_results.py --clip 0                 # Process only clip 0
    python visualize_results.py --output_dir annotated/  # Custom output directory
"""

import os
import json
import argparse
import cv2
import numpy as np
from pathlib import Path

# -----------------------
# Configuration
# -----------------------
# Video files (same as in video_id_detector1.py)
dir_path = "/Users/liatparker/Documents/mp4_files_id_detectors/"
VIDEO_FILES = [dir_path+"1.mp4", dir_path+"2.mp4", dir_path+"3.mp4", dir_path+"4.mp4"]
JSON_FILE = "global_identity_catalogue.json"
OUTPUT_DIR = "annotated_videos"

# Colors for different global IDs (BGR format)
COLORS = [
    (0, 255, 0),      # Green - ID 0
    (255, 0, 0),      # Blue - ID 1
    (0, 0, 255),      # Red - ID 2
    (255, 255, 0),    # Cyan - ID 3
    (255, 0, 255),    # Magenta - ID 4
    (0, 255, 255),    # Yellow - ID 5
    (128, 0, 128),    # Purple - ID 6
    (255, 128, 0),    # Orange - ID 7
    (0, 128, 255),    # Sky Blue - ID 8
    (128, 255, 0),    # Lime - ID 9
]

def get_color(global_id):
    """Get color for a global ID"""
    return COLORS[global_id % len(COLORS)]

def load_catalogue(json_file):
    """Load the global identity catalogue"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def parse_catalogue_to_frame_map(data):
    """
    Convert catalogue to a mapping: {clip_id: {frame_num: [global_ids]}}
    Since multiple people can appear in the same frame, we track all global_ids per frame.
    """
    frame_map = {}
    
    identities = data.get("identities", {})
    for gid_key, person in identities.items():
        global_id = int(gid_key.split('_')[-1])
        
        for appearance in person["appearances"]:
            clip_id = appearance["clip_id"]
            start_frame = appearance["frame_range"][0]
            end_frame = appearance["frame_range"][1]
            
            if clip_id not in frame_map:
                frame_map[clip_id] = {}
            
            # Mark all frames in this range with this global_id
            for frame_num in range(start_frame, end_frame + 1):
                if frame_num not in frame_map[clip_id]:
                    frame_map[clip_id][frame_num] = []
                frame_map[clip_id][frame_num].append(global_id)
    
    return frame_map

def draw_detections_on_frame(frame, detections, frame_num):
    """
    Draw bounding boxes and labels on frame.
    detections: list of (bbox, global_id) tuples
    """
    for bbox, global_id in detections:
        x1, y1, x2, y2 = map(int, bbox)
        color = get_color(global_id)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"ID {global_id}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw frame number
    cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return frame

def process_video_with_yolo(video_path, clip_id, frame_map, output_path):
    """
    Process video with YOLOv8 detection and overlay global_id labels.
    """
    from ultralytics import YOLO
    
    print(f"\nProcessing {os.path.basename(video_path)}...")
    
    # Load YOLO model
    yolo = YOLO("yolov8n.pt")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        # Run YOLO detection (person class only)
        results = yolo(frame, classes=[0], verbose=False)
        
        # Get detections for this frame
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            
            # Check if this frame has any global_ids
            if clip_id in frame_map and frame_num in frame_map[clip_id]:
                global_ids = frame_map[clip_id][frame_num]
                
                # Match detections to global_ids
                # Simple approach: assign global_ids to detections in order
                for i, bbox in enumerate(xyxy):
                    if i < len(global_ids):
                        detections.append((bbox, global_ids[i]))
                    else:
                        # If more detections than global_ids, draw without label
                        detections.append((bbox, -1))  # -1 means unknown
        
        # Draw detections on frame
        annotated_frame = draw_detections_on_frame(frame.copy(), detections, frame_num)
        
        # Write frame
        out.write(annotated_frame)
        
        # Progress indicator
        if frame_num % 100 == 0:
            print(f"  Processed {frame_num}/{total_frames} frames...", end='\r')
    
    cap.release()
    out.release()
    
    print(f"\n  ✅ Saved to: {output_path}")

def process_video_simple(video_path, clip_id, frame_map, output_path):
    """
    Simple version: Draw boxes based on catalogue data only (no re-detection).
    This is faster but requires the original detection coordinates.
    """
    print(f"\n⚠️  Simple mode not implemented yet. Use YOLO detection mode.")
    print(f"   Run with default settings to use YOLO-based visualization.")

def main():
    parser = argparse.ArgumentParser(description="Visualize clustering results on videos")
    parser.add_argument("--clip", type=int, default=None, help="Process only specific clip (0-3)")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--json", default=JSON_FILE, help="JSON catalogue file")
    parser.add_argument("--simple", action="store_true", help="Simple mode (no re-detection)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("📹 VIDEO VISUALIZATION WITH GLOBAL IDs")
    print("=" * 60)
    
    # Load catalogue
    print(f"\nLoading catalogue from {args.json}...")
    try:
        data = load_catalogue(args.json)
        summary = data.get("summary", {})
        print(f"✅ Loaded catalogue:")
        print(f"   {summary.get('total_global_ids', '?')} global IDs")
        print(f"   {summary.get('total_appearances', '?')} appearances")
    except FileNotFoundError:
        print(f"❌ Error: {args.json} not found. Run clustering first!")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: {args.json} is not valid JSON")
        return
    
    # Parse catalogue to frame map
    frame_map = parse_catalogue_to_frame_map(data)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n📁 Output directory: {args.output_dir}/")
    
    # Process videos
    clips_to_process = [args.clip] if args.clip is not None else range(len(VIDEO_FILES))
    
    for clip_id in clips_to_process:
        if clip_id >= len(VIDEO_FILES):
            print(f"\n⚠️  Warning: Clip {clip_id} not found (only {len(VIDEO_FILES)} clips available)")
            continue
        
        video_path = VIDEO_FILES[clip_id]
        if not os.path.exists(video_path):
            print(f"\n⚠️  Warning: Video file not found: {video_path}")
            continue
        
        # Output path
        output_filename = f"clip_{clip_id}_annotated.mp4"
        output_path = os.path.join(args.output_dir, output_filename)
        
        # Process video
        if args.simple:
            process_video_simple(video_path, clip_id, frame_map, output_path)
        else:
            process_video_with_yolo(video_path, clip_id, frame_map, output_path)
    
    print("\n" + "=" * 60)
    print("✅ Done! Annotated videos saved to:", args.output_dir)
    print("=" * 60)

if __name__ == "__main__":
    main()

