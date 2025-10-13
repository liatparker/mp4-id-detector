#!/bin/bash

# Video Annotation Script
# This script runs the video annotation to create annotated videos with ID detection results

echo "=========================================="
echo "VIDEO ANNOTATION SCRIPT"
echo "=========================================="

# Activate virtual environment
source cvenv/bin/activate

# Create output directory
mkdir -p ./annotated/

# Run annotation for all clips
echo "Annotating all clips..."
python annotate_video_advanced.py \
    --video-dir ./videos/ \
    --results ./outputs_clean/ \
    --output-dir ./annotated/

echo "=========================================="
echo "ANNOTATION COMPLETE!"
echo "=========================================="
echo "Annotated videos saved in: ./annotated/"
echo "Files created:"
ls -la ./annotated/
