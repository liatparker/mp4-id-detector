#!/bin/bash

# GitHub Repository Setup Script for MP4 ID Detector
# This script helps you create and configure a GitHub repository

echo "=========================================="
echo "MP4 ID DETECTOR - GITHUB REPOSITORY SETUP"
echo "=========================================="

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    exit 1
fi

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "🔄 Initializing Git repository..."
    git init
fi

# Add all files
echo "📁 Adding files to Git..."
git add .

# Create initial commit
echo "💾 Creating initial commit..."
git commit -m "Initial commit: MP4 ID Detector v1.0.0

- Person re-identification system with OSNet and TransReID
- Crime detection with YOLO + CLIP
- Video annotation with bounding box visualization
- Adaptive clustering with 2-stage approach
- Hybrid embeddings (40% image + 60% video)
- Comprehensive documentation and setup scripts"

echo ""
echo "🚀 Next steps to create GitHub repository:"
echo ""
echo "1. Go to GitHub.com and create a new repository:"
echo "   - Repository name: mp4-id-detector"
echo "   - Description: A comprehensive video analysis system for person re-identification and crime detection"
echo "   - Make it Public"
echo "   - Don't initialize with README (we already have one)"
echo ""
echo "2. After creating the repository, run these commands:"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/mp4-id-detector.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Update the repository URL in these files:"
echo "   - setup.py (line with url=)"
echo "   - README.md (if any GitHub links exist)"
echo ""
echo "4. Add topics/tags to your repository:"
echo "   - computer-vision"
echo "   - person-reidentification"
echo "   - object-detection"
echo "   - video-analysis"
echo "   - crime-detection"
echo "   - yolo"
echo "   - clip"
echo "   - deep-learning"
echo "   - python"
echo ""
echo "5. Enable GitHub Pages (optional):"
echo "   - Go to repository Settings > Pages"
echo "   - Source: Deploy from a branch"
echo "   - Branch: main"
echo ""
echo "✅ Repository setup complete!"
echo "📝 Don't forget to update the URLs in setup.py and README.md"
