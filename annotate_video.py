#!/usr/bin/env python3
"""
Video Annotation Script for ID Detection Results

This script creates annotated videos showing detected identities with bounding boxes,
track IDs, and global IDs overlaid on the original video frames.

Usage:
    python annotate_video.py --input video.mp4 --results outputs_clean/ --output annotated_video.mp4
    python annotate_video.py --batch ./videos/ --results outputs_clean/ --output-dir ./annotated/
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
    
    # Load embeddings for additional data
    embeddings_path = results_dir / "track_embeddings.npz"
    embeddings_data = None
    if embeddings_path.exists():
        embeddings_data = np.load(embeddings_path, allow_pickle=True)
        print(f"Loaded embeddings from {embeddings_path}")
    
    return df, embeddings_data

def get_identity_color(global_id):
    """Get a consistent color for a global identity."""
    return IDENTITY_COLORS[global_id % len(IDENTITY_COLORS)]

def draw_identity_info(frame, bbox, track_id, global_id, clip_idx, confidence=None):
    """Draw identity information on the frame."""
    x1, y1, x2, y2 = bbox
    
    # Get color for this identity
    color = get_identity_color(global_id)
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
    
    # Prepare text labels
    labels = [
        f"ID: {global_id}",
        f"Track: {track_id}",
        f"Clip: {clip_idx}"
    ]
    
    if confidence is not None:
        labels.append(f"Conf: {confidence:.2f}")
    
    # Draw text background
    text_height = 25
    text_y = y1 - 10
    if text_y < 0:
        text_y = y2 + 10
    
    # Draw background rectangle for text
    text_width = max([cv2.getTextSize(label, DEFAULT_FONT, FONT_SCALE, FONT_THICKNESS)[0][0] 
                     for label in labels]) + 10
    
    cv2.rectangle(frame, (x1, text_y - len(labels) * text_height), 
                  (x1 + text_width, text_y + 5), color, -1)
    
    # Draw text
    for i, label in enumerate(labels):
        text_pos = (x1 + 5, text_y - (len(labels) - 1 - i) * text_height)
        cv2.putText(frame, label, text_pos, DEFAULT_FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)

def create_frame_annotations(df, clip_idx):
    """Create annotations for a specific clip."""
    clip_data = df[df['clip_idx'] == clip_idx].copy()
    
    if len(clip_data) == 0:
        return {}
    
    # Group by frame to get all detections in each frame
    frame_annotations = {}
    
    for _, row in clip_data.iterrows():
        start_frame = row['start_frame']
        end_frame = row['end_frame']
        global_id = row['global_id']
        track_id = row['track_id']
        
        # For each frame in this tracklet, we need to get the bbox
        # This is a simplified version - in practice, you'd need to load the actual bbox data
        for frame_num in range(start_frame, end_frame + 1):
            if frame_num not in frame_annotations:
                frame_annotations[frame_num] = []
            
            # Placeholder bbox - in real implementation, load from embeddings data
            bbox = [100, 100, 200, 200]  # x1, y1, x2, y2
            frame_annotations[frame_num].append({
                'bbox': bbox,
                'global_id': global_id,
                'track_id': track_id,
                'clip_idx': clip_idx
            })
    
    return frame_annotations

def annotate_video(video_path, results_dir, output_path, clip_idx=None):
    """Annotate a single video with detection results."""
    print(f"Annotating video: {video_path}")
    
    # Load results
    df, embeddings_data = load_results(results_dir)
    
    # Filter by clip if specified
    if clip_idx is not None:
        df = df[df['clip_idx'] == clip_idx]
        print(f"Filtering to clip {clip_idx}: {len(df)} tracklets")
    
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
    
    # Create frame annotations
    frame_annotations = create_frame_annotations(df, clip_idx or 0)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add annotations for this frame
        if frame_count in frame_annotations:
            for annotation in frame_annotations[frame_count]:
                draw_identity_info(
                    frame,
                    annotation['bbox'],
                    annotation['track_id'],
                    annotation['global_id'],
                    annotation['clip_idx']
                )
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   DEFAULT_FONT, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"Annotation complete! Saved to: {output_path}")

def annotate_batch(video_dir, results_dir, output_dir):
    """Annotate multiple videos in batch."""
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Load results
    df, embeddings_data = load_results(results_dir)
    
    # Get unique clips
    unique_clips = sorted(df['clip_idx'].unique())
    print(f"Found {len(unique_clips)} clips in results")
    
    for i, video_file in enumerate(video_files):
        if i < len(unique_clips):
            clip_idx = unique_clips[i]
            output_path = output_dir / f"annotated_clip_{clip_idx}_{video_file.name}"
            
            try:
                annotate_video(video_file, results_dir, output_path, clip_idx)
                print(f"✓ Annotated {video_file.name} as clip {clip_idx}")
            except Exception as e:
                print(f"✗ Failed to annotate {video_file.name}: {e}")
        else:
            print(f"⚠ Skipping {video_file.name} - no corresponding clip data")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Annotate videos with ID detection results")
    parser.add_argument("--input", type=str, help="Input video file")
    parser.add_argument("--batch", type=str, help="Directory containing video files")
    parser.add_argument("--results", type=str, required=True, help="Results directory")
    parser.add_argument("--output", type=str, help="Output video file (single video mode)")
    parser.add_argument("--output-dir", type=str, help="Output directory (batch mode)")
    parser.add_argument("--clip", type=int, help="Specific clip index to annotate")
    
    args = parser.parse_args()
    
    if not args.results:
        print("Error: --results directory is required")
        sys.exit(1)
    
    if args.input and args.batch:
        print("Error: Cannot specify both --input and --batch")
        sys.exit(1)
    
    if args.input:
        # Single video mode
        if not args.output:
            print("Error: --output is required for single video mode")
            sys.exit(1)
        
        annotate_video(args.input, args.results, args.output, args.clip)
        
    elif args.batch:
        # Batch mode
        if not args.output_dir:
            print("Error: --output-dir is required for batch mode")
            sys.exit(1)
        
        annotate_batch(args.batch, args.results, args.output_dir)
        
    else:
        print("Error: Must specify either --input or --batch")
        sys.exit(1)

if __name__ == "__main__":
    main()
