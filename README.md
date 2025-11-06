# Video Analysis System - Person Re-identification & Crime Detection

A comprehensive video analysis system combining advanced person re-identification across multiple video clips with zero-shot crime detection capabilities.

## 🎯 Overview

This system provides two complementary capabilities:

1. **Person Re-identification**: Identifies the same person across multiple video clips, even when viewed from different angles, lighting conditions, or with occlusions.

2. **Zero-Shot Crime Detection**: Detects criminal activities (shoplifting, weapons) in video clips using hybrid YOLO + CLIP models. No training data required!

### Key Capabilities

**Person Re-identification:**
- Multi-video person matching across different clips
- Robust to view changes (front, back, side views)
- Adaptive clustering that adjusts based on scene characteristics
- Face + body recognition combining multiple feature sources
- Annotated video output with bounding boxes and global IDs

**Crime Detection:**
- Zero-shot detection (no training required)
- Shoplifting detection via behavior analysis
- Weapon detection with validation to reduce false positives
- Temporal consistency filtering
- Batch processing for multiple video clips

---

## 📦 How It Works

### **Person Re-identification System**

The system uses a two-stage approach:

**Stage 1: Within-Clip Clustering (Strict Separation)**
- When people appear in the same video clip, they are likely different people
- The system carefully separates them using face and body features
- If two people appear simultaneously, they are blocked from merging
- Similarity requirements adapt based on temporal overlap, face availability, and other factors
- Each clip gets automatic threshold adjustment based on scene characteristics

**Stage 2: Cross-Clip Merging (Lenient Matching)**
- When matching across different clips, the same person may look different
- The system uses a two-pass approach:
  - First pass: Handles cases where face information is missing (more lenient)
  - Second pass: Handles cases where both clusters have faces (stricter, recalculates distances)
- Uses best-of-cluster matching to prevent single bad tracklets from dominating
- Longer tracklets contribute more to cluster representations

**Multi-Model Ensemble:**
- Combines multiple deep learning models for robust feature extraction
- OSNet and TransReID for body appearance
- InsightFace for face recognition
- Optional pose estimation for additional characterization

**Advanced Features:**
- Test-time augmentation for view invariance
- Temporal smoothing for stability
- Re-ranking for improved accuracy
- Camera bias correction
- Caching system for fast reprocessing

### **Crime Detection System**

**Hybrid Detection Architecture:**
- Combines YOLO object detection with CLIP scene understanding
- Uses specialized weapon detection model (optional)
- CLIP validates each weapon detection to filter false positives (like phones)

**Detection Pipeline:**
1. **Frame Preprocessing**: Normalizes lighting and enhances details
2. **YOLO Detection**: Detects people and weapons
3. **CLIP Validation**: Validates weapon detections to reduce false positives
4. **CLIP Scene Analysis**: Analyzes scene behavior using natural language descriptions
5. **Temporal Filtering**: Requires weapons to appear in multiple frames
6. **Decision Making**: Priority-based crime classification

**Crime Categories:**
- **SHOPLIFTING**: Behavior-based detection via scene understanding
- **GUN**: Validated weapon detections with temporal consistency
- **URGENT_REACTION**: Panic/rushing behavior detection
- **NORMAL**: No crime detected

---

## 🚀 Usage Guide

### **Person Re-identification**

**Step 1: Run Detection**
```bash
source cvenv/bin/activate
python video_id_detector2_optimized.py
```

This processes all video clips, detects and tracks people, extracts features, performs clustering, and exports results.

**Step 2: Generate Annotated Videos**
```bash
python export_annotated_videoes_v3_from_cache.py
```

Creates individual annotated videos with color-coded bounding boxes and global ID labels.

**Step 3: Create Grid Video** (Optional)
```bash
python create_2x2_grid_video.py
```

Combines all clips into a synchronized 2x2 grid view for easy comparison.

### **Crime Detection**

**Single Video:**
```bash
source cvenv/bin/activate
python crime_no_crime_zero_shot1.py --video ./videos/1_upscaled.mp4
```

**Batch Processing:**
```bash
python crime_no_crime_zero_shot1.py --batch ./videos
```

**Output:**
- Individual clip results: JSON files with crime detection results
- Dataset summary: Overall crime labels for all clips

---

## 📊 Output Files

### **Person Re-identification**
- **Global Identity Catalogue (JSON)**: Summary and detailed appearance information
- **Tracklet Mappings (NPZ)**: ID mappings and cached embeddings
- **Annotated Videos**: Individual clips and 2x2 grid view with color-coded IDs

### **Crime Detection**
- **Per-Clip Results (JSON)**: Crime detection results with confidence scores
- **Dataset Summary (JSON)**: Overall crime labels for all clips

---

## 🎨 Visualization

**Annotated Videos:**
- Each person gets a unique color-coded bounding box
- Global ID labels displayed on each detection
- Legend showing all IDs in the clip

**2x2 Grid Video:**
- All 4 clips displayed simultaneously
- Perfect frame-by-frame synchronization
- Ideal for cross-clip validation and presentations

---

## ⚡ Performance

- **Caching System**: Pre-computed embeddings stored for fast reprocessing
- **Frame Sampling**: Processes sampled frames for efficiency
- **Optimized Operations**: Vectorized computations for speed

---

## 🔧 Troubleshooting

**Missing cache files**: Script will process from scratch (first run takes longer)

**Model loading errors**: Check model file paths and ensure models are downloaded

**Memory issues**: Reduce batch size, use CPU, or process fewer clips

**Video format issues**: Ensure MP4 format and check codec compatibility

**Performance**: Use existing cache, reduce video resolution, or adjust sampling rates

---

## 📁 File Structure

```
mp4_id_detector/
├── video_id_detector2_optimized.py    # Person re-identification
├── crime_no_crime_zero_shot1.py        # Crime detection
├── export_annotated_videoes_v3_from_cache.py  # Video annotation
├── create_2x2_grid_video.py            # Grid video creator
├── outputs_v3/                         # Output directory
│   ├── annotated_videos_v3/           # Annotated videos
│   └── grid_2x2_annotated.mp4         # Grid video
└── videos/                             # Input videos
```

---

## 🎯 Use Cases

**Research & Development:**
- Algorithm testing and evaluation
- Parameter tuning and optimization
- Feature analysis and comparison

**Production Deployment:**
- Multi-camera surveillance systems
- Person tracking across different scenes
- Real-time crime detection in retail environments
- Security monitoring systems

---

## 📚 References

- OSNet: [Torchreid](https://github.com/KaiyangZhou/deep-person-reid)
- TransReID: [TransReID Repository](https://github.com/damo-cv/TransReID)
- InsightFace: [Face Recognition](https://github.com/deepinsight/insightface)
- YOLOv8: [Ultralytics](https://github.com/ultralytics/ultralytics)
- CLIP: [OpenAI CLIP](https://github.com/openai/CLIP)

---

## 📝 License

MIT License - See LICENSE file for details.

---

**Note**: This system uses sophisticated two-stage adaptive clustering and hybrid YOLO + CLIP detection approaches tuned for accuracy across different video conditions.
