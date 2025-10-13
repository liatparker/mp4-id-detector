#!/bin/bash

# Video ID Detector - Automatic Run Script
# This script automatically runs the video ID detection system

echo "🚀 Starting Video ID Detection System..."
echo "=========================================="

# Activate virtual environment
echo "📦 Activating virtual environment..."
source cvenv/bin/activate

# Check if videos exist
echo "🔍 Checking video files..."
if [ ! -d "videos" ]; then
    echo "❌ Videos directory not found!"
    exit 1
fi

video_count=$(ls videos/*.mp4 2>/dev/null | wc -l)
echo "📹 Found $video_count video files"

if [ $video_count -eq 0 ]; then
    echo "❌ No video files found in videos/ directory!"
    exit 1
fi

# Run the detector
echo "🎯 Running video ID detector..."
python video_id_detector2_optimized.py

# Check if successful
if [ $? -eq 0 ]; then
    echo "✅ Detection completed successfully!"
    echo "📁 Results saved in outputs_v3/"
    echo "📊 Check global_identity_catalogue_v3.json for results"
else
    echo "❌ Detection failed!"
    exit 1
fi

echo "=========================================="
echo "🎉 All done!"
