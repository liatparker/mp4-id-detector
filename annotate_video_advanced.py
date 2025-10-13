#!/usr/bin/env python3
"""
Advanced Video Annotation Script for ID Detection Results

This script creates annotated videos with actual bounding box data from the embeddings.
It can handle multiple clips and provides detailed visualization of identity tracking.

Usage:
    python annotate_video_advanced.py --results outputs_clean/ --output-dir ./annotated/
    python annotate_video_advanced.py --results outputs_clean/ --clip 0 --output clip_0_annotated.mp4
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
import pandas as pd

# Configuration
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
BOX_THICKNESS = 2

# Color palette for different identities (BGR format for OpenCV)
IDENTITY_COLORS = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
    (0, 128, 128),  # Teal
    (128, 128, 0),  # Olive
    (255, 192, 203), # Pink
    (0, 0, 128),    # Navy
    (128, 0, 0),    # Maroon
    (0, 128, 0),    # Dark Green
    (128, 128, 128), # Gray
]

def load_results(results_dir):
    """Load detection results from the output directory."""
    results_dir = Path(results_dir)
    
    # Load CSV results
    csv_path = results_dir / "global_identity_catalogue.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} tracklets from {csv_path}")
    
    # Load embeddings for bounding box data
    embeddings_path = results_dir / "track_embeddings.npz"
    embeddings_data = None
    if embeddings_path.exists():
        embeddings_data = np.load(embeddings_path, allow_pickle=True)
        print(f"Loaded embeddings from {embeddings_path}")
    
    return df, embeddings_data

def get_identity_color(global_id):
    """Get a consistent color for a global identity."""
    return IDENTITY_COLORS[global_id % len(IDENTITY_COLORS)]

def create_tracklet_mapping(df, embeddings_data):
    """Create a mapping from tracklets to their data."""
    tracklet_mapping = {}
    
    if embeddings_data is not None and 'tracklets' in embeddings_data:
        tracklets = embeddings_data['tracklets']
        
        for i, tracklet in enumerate(tracklets):
            clip_idx = int(tracklet['clip_idx'])
            track_id = int(tracklet['track_id'])
            key = f"clip_{clip_idx}_track_{track_id}"
            
            # Find corresponding row in CSV
            csv_row = df[(df['clip_idx'] == clip_idx) & (df['track_id'] == track_id)]
            if len(csv_row) > 0:
                global_id = csv_row.iloc[0]['global_id']
                
                tracklet_mapping[key] = {
                    'global_id': global_id,
                    'clip_idx': clip_idx,
                    'track_id': track_id,
                    'start_frame': int(tracklet['start_frame']),
                    'end_frame': int(tracklet['end_frame']),
                    'bboxes': tracklet['bboxes'],
                    'frames': tracklet['frames'],
                    'has_face': bool(tracklet['has_face'])
                }
    
    return tracklet_mapping

def get_detections_for_frame(tracklet_mapping, clip_idx, frame_num):
    """Get all detections for a specific frame."""
    detections = []
    
    for key, tracklet in tracklet_mapping.items():
        if tracklet['clip_idx'] != clip_idx:
            continue
            
        start_frame = tracklet['start_frame']
        end_frame = tracklet['end_frame']
        
        if start_frame <= frame_num <= end_frame:
            # Find the bbox for this frame
            frames = tracklet['frames']
            bboxes = tracklet['bboxes']
            
            if frame_num in frames:
                frame_idx = frames.index(frame_num)
                if frame_idx < len(bboxes):
                    bbox = bboxes[frame_idx]
                    detections.append({
                        'bbox': bbox,
                        'global_id': tracklet['global_id'],
                        'track_id': tracklet['track_id'],
                        'has_face': tracklet['has_face']
                    })
    
    return detections

def draw_identity_info(frame, bbox, track_id, global_id, has_face=False, confidence=None):
    """Draw identity information on the frame."""
    x1, y1, x2, y2 = bbox
    
    # Get color for this identity
    color = get_identity_color(global_id)
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
    
    # Prepare text labels
    labels = [
        f"ID: {global_id}",
        f"Track: {track_id}"
    ]
    
    if has_face:
        labels.append("Face: ✓")
    else:
        labels.append("Face: ✗")
    
    if confidence is not None:
        labels.append(f"Conf: {confidence:.2f}")
    
    # Draw text background
    text_height = 25
    text_y = y1 - 10
    if text_y < 0:
        text_y = y2 + 10
    
    # Calculate text width
    text_width = 0
    for label in labels:
        (w, h), _ = cv2.getTextSize(label, DEFAULT_FONT, FONT_SCALE, FONT_THICKNESS)
        text_width = max(text_width, w)
    text_width += 10
    
    # Draw background rectangle for text
    cv2.rectangle(frame, (x1, text_y - len(labels) * text_height), 
                  (x1 + text_width, text_y + 5), color, -1)
    
    # Draw text
    for i, label in enumerate(labels):
        text_pos = (x1 + 5, text_y - (len(labels) - 1 - i) * text_height)
        cv2.putText(frame, label, text_pos, DEFAULT_FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)

def annotate_video_clip(video_path, tracklet_mapping, clip_idx, output_path):
    """Annotate a specific clip of a video."""
    print(f"Annotating clip {clip_idx} from video: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    processed_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get detections for this frame
        detections = get_detections_for_frame(tracklet_mapping, clip_idx, frame_count)
        
        # Draw annotations
        for detection in detections:
            draw_identity_info(
                frame,
                detection['bbox'],
                detection['track_id'],
                detection['global_id'],
                detection['has_face']
            )
        
        # Add frame counter and clip info
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   DEFAULT_FONT, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Clip: {clip_idx}", (10, 60), 
                   DEFAULT_FONT, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Detections: {len(detections)}", (10, 90), 
                   DEFAULT_FONT, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        frame_count += 1
        
        if len(detections) > 0:
            processed_frames += 1
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({processed_frames} with detections)")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"Annotation complete! Saved to: {output_path}")
    print(f"Processed {processed_frames} frames with detections out of {total_frames} total frames")

def annotate_all_clips(video_dir, results_dir, output_dir):
    """Annotate all clips from the results."""
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load results
    df, embeddings_data = load_results(results_dir)
    
    # Create tracklet mapping
    tracklet_mapping = create_tracklet_mapping(df, embeddings_data)
    print(f"Created mapping for {len(tracklet_mapping)} tracklets")
    
    # Get unique clips
    unique_clips = sorted(df['clip_idx'].unique())
    print(f"Found {len(unique_clips)} clips in results")
    
    # Find video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Annotate each clip
    for clip_idx in unique_clips:
        if clip_idx < len(video_files):
            video_file = video_files[clip_idx]
            output_path = output_dir / f"annotated_clip_{clip_idx}_{video_file.name}"
            
            try:
                annotate_video_clip(video_file, tracklet_mapping, clip_idx, output_path)
                print(f"✓ Annotated clip {clip_idx} from {video_file.name}")
            except Exception as e:
                print(f"✗ Failed to annotate clip {clip_idx}: {e}")
        else:
            print(f"⚠ No video file for clip {clip_idx}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Advanced video annotation with ID detection results")
    parser.add_argument("--video-dir", type=str, default="./videos/", help="Directory containing video files")
    parser.add_argument("--results", type=str, required=True, help="Results directory")
    parser.add_argument("--output-dir", type=str, default="./annotated/", help="Output directory")
    parser.add_argument("--clip", type=int, help="Specific clip index to annotate")
    
    args = parser.parse_args()
    
    if not args.results:
        print("Error: --results directory is required")
        sys.exit(1)
    
    if args.clip is not None:
        # Single clip mode
        video_dir = Path(args.video_dir)
        video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
        
        if args.clip >= len(video_files):
            print(f"Error: Clip {args.clip} not found. Available clips: 0-{len(video_files)-1}")
            sys.exit(1)
        
        video_file = video_files[args.clip]
        output_path = Path(args.output_dir) / f"annotated_clip_{args.clip}_{video_file.name}"
        output_path.parent.mkdir(exist_ok=True)
        
        # Load results and create mapping
        df, embeddings_data = load_results(args.results)
        tracklet_mapping = create_tracklet_mapping(df, embeddings_data)
        
        annotate_video_clip(video_file, tracklet_mapping, args.clip, output_path)
    else:
        # All clips mode
        annotate_all_clips(args.video_dir, args.results, args.output_dir)

if __name__ == "__main__":
    main()
