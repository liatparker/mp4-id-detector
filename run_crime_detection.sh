#!/bin/bash

# Crime Detection System - Automatic Run Script
# This script automatically runs the crime detection system on all videos

echo "🚨 Starting Crime Detection System..."
echo "====================================="

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

# Run the crime detector in batch mode
echo "🎯 Running crime detection on all videos..."
python crime_no_crime_zero_shot1.py --batch ./videos --sample-rate 30 --threshold 0.5

# Check if successful
if [ $? -eq 0 ]; then
    echo "✅ Crime detection completed successfully!"
    echo "📁 Results saved as clip*_crime_validated.json files"
    echo "📊 Check the JSON files for detailed crime analysis"
else
    echo "❌ Crime detection failed!"
    exit 1
fi

echo "====================================="
echo "🎉 All done!"