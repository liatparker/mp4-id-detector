# Video Analysis System - Person Re-identification & Crime Detection

A comprehensive video analysis system combining advanced person re-identification across multiple video clips with zero-shot crime detection capabilities using hybrid YOLO + CLIP models.

## 🎯 Overview

This system provides two complementary capabilities:

1. **Person Re-identification (`video_id_detector2_optimized.py`)**: Identifies the same person across multiple video clips using advanced computer vision and machine learning techniques, even when viewed from different angles, lighting conditions, or with occlusions.

2. **Zero-Shot Crime Detection (`crime_no_crime_zero_shot1.py`)**: Detects criminal activities (shoplifting, weapons) in video clips using hybrid YOLO + CLIP models with CLIP-based gun validation. No training data required!

### Key Capabilities

**Person Re-identification:**
- Multi-video person matching across different clips
- Robust to view changes (front, back, side views)
- Adaptive clustering that adjusts thresholds based on scene characteristics
- Face + body recognition combining multiple feature sources
- Annotated video output with bounding boxes and global IDs

**Crime Detection:**
- Zero-shot detection (no training required)
- Shoplifting detection via CLIP behavior analysis
- Weapon detection with YOLO + CLIP validation
- Temporal consistency filtering to reduce false positives
- Batch processing for multiple video clips

---

## 📦 System Components

### **1. Person Re-identification System**

**File**: `video_id_detector2_optimized.py`

#### **Core Algorithm: Two-Stage Adaptive Clustering**

The system uses a sophisticated two-stage approach that first separates different people within the same video clip (strict), then matches the same person across different clips (lenient).

##### **Stage 1: Within-Clip Clustering (Strict Separation)**

**Why Strict?** When people appear in the same video clip, they are likely different people. We need to be very careful not to merge them incorrectly.

**How it works:**
- **Face + Body Features**: Combines face recognition (40% weight) and body appearance (60% weight). Faces are reliable within the same lighting/angle conditions.
- **Temporal Overlap Analysis**: If two people appear simultaneously (high temporal overlap), they are almost certainly different people. The system blocks merging when overlap exceeds 50%.
- **Adaptive Similarity Requirements**: The required similarity between two tracklets depends on:
  - **Temporal overlap**: Higher overlap requires higher similarity (more strict)
  - **Face availability**: When faces are detected, slightly stricter requirements apply
  - **Containment**: If one tracklet is fully contained within another, requirements are slightly stricter
  - **Duration ratio**: Very different tracklet lengths get stricter requirements
- **Cohesion Check**: A tracklet must be similar to ALL members of a cluster, not just the average. This prevents weak matches from being accepted.
- **Per-Clip Adaptation**: Each clip gets automatic threshold adjustment based on:
  - Number of people (crowding analysis)
  - Face visibility rate
  - Lighting/angle diversity
  - Tracking quality and stability

##### **Stage 2: Cross-Clip Merging (Lenient Matching)**

**Why Lenient?** When matching across different clips, the same person may look different due to lighting, angle, or camera conditions. We need to be more flexible.

**How it works:**
- **Two-Pass Merging System**: Processes matches in two passes to handle different scenarios:
  1. **Pass 1: Face-to-No-Face Matches**
     - Handles cases where one or both clusters lack face information
     - Prioritizes body similarity (higher similarity processed first)
     - Uses more lenient thresholds since face information is missing
     - Locks merged clusters to prevent conflicts in Pass 2
  2. **Pass 2: Face-to-Face Matches**
     - Processes pairs where both clusters have face information
     - Recalculates distances using all tracklets from merged clusters (if Pass 1 merged any)
     - Uses stricter requirements since face information is available
     - Implements best-match logic: each target cluster gets matched with its best source candidate
     - Only processes unlocked clusters to avoid conflicts
- **Best-of-Cluster Matching**: Instead of comparing average cluster embeddings, the system finds the minimum distance among all tracklet pairs. This prevents a single bad tracklet from dominating the comparison.
- **Length-Weighted Averaging**: Longer tracklets (more frames) contribute more to cluster embeddings, as they contain more reliable information.
- **Adaptive Thresholds**: Thresholds adjust based on:
  - **Tracklet length**: Longer tracklets are more reliable and get slightly more lenient thresholds
  - **Face availability**: Face-to-face pairs use stricter thresholds than face-to-no-face pairs
  - **Same-clip blocking**: Never merges clusters from the same clip (hard constraint)
  - **Transitive blocking**: Prevents indirect merging of incompatible clusters, but can be overridden by strong spatial evidence

#### **Multi-Model Ensemble**

The system combines multiple deep learning models for robust feature extraction:

- **OSNet (IBN-Net)**: Primary body appearance model, 512D embeddings
- **TransReID (ViT)**: Vision Transformer for body features, 768D embeddings
- **Ensemble**: When enabled, combines OSNet (70% weight) and TransReID (30% weight) into 1280D embeddings
- **InsightFace**: Face recognition model, 512D embeddings (when faces detected)
- **MediaPipe**: Optional pose estimation for additional person characterization

#### **Advanced Features**

- **Test-Time Augmentation (TTA)**: Horizontal flip to improve robustness across different viewing angles
- **Temporal Smoothing**: Smooths embeddings within tracks over time for stability
- **k-Reciprocal Re-ranking**: Improves matching accuracy by considering reciprocal nearest neighbors
- **Camera Bias Correction**: Accounts for camera-specific variations in appearance
- **Quality-Weighted Embeddings**: Frames with faces and good pose detection get higher weights
- **Representative Frame Selection**: Chooses diverse, high-quality frames for multi-frame matching
- **Caching System**: Stores pre-computed embeddings in NPZ files for fast parameter tuning

#### **2x2 Grid Video Output** 🎬

Creates a synchronized 2x2 grid view combining all 4 annotated clips. Perfect for:
- Verifying cross-clip person matching
- Comparing the same person across different views
- Quality assurance and validation
- Presentations and demonstrations

**Output file**: `outputs_v3/grid_2x2_annotated.mp4`

---

### **2. Zero-Shot Crime Detection System**

**File**: `crime_no_crime_zero_shot1.py`

#### **Hybrid Detection Architecture**

The system combines three complementary approaches:
- **YOLOv8**: Fast object detection for people and objects (COCO dataset)
- **Custom Weapon YOLO**: Specialized gun detection model (optional, `weapon_yolov8_gun.pt`)
- **CLIP (OpenAI)**: Zero-shot scene understanding and gun validation

#### **Detection Pipeline**

**1. Frame Preprocessing**
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Normalizes lighting variations across clips, critical for detecting small objects like guns
- **Mild Sharpening**: Enhances edges and details to improve detection of small weapons

**2. YOLO Object Detection**
- Detects people using standard YOLOv8 COCO model
- Detects weapons using specialized gun model (if available)
- **CLIP Validation**: Each YOLO gun detection is validated using CLIP to filter false positives
  - Asks CLIP: "Is this a gun or a smartphone?"
  - Uses strict validation logic requiring strong evidence that it's a gun (not a phone)
  - Combines YOLO confidence with CLIP validation score for robust detection

**3. CLIP Scene Analysis**
- Zero-shot classification using natural language crime descriptions:
  - **Shoplifting**: Behavioral descriptions like "person secretly stealing items"
  - **Guns**: Multiple variations like "person holding a handgun"
  - **Urgent Reaction**: Panic behavior like "person rushing suddenly"
  - **Normal**: Calm, peaceful scene descriptions
- Returns probability scores for each category

**4. Temporal Consistency Filter**
- Requires weapons to appear in multiple frames (not just one) to avoid false positives
- Checks if detections are temporally consistent (within a few frames)
- Rejects isolated single-frame detections

**5. Crime Decision Logic**
- **Priority-based decision making**:
  1. **Shoplifting**: If strong shoplifting signal detected
  2. **Gun Detection**: If validated weapons found, uses combined YOLO + CLIP scores
     - High confidence: Trust validation
     - Medium confidence: Reduced filtering
     - Low confidence: Requires temporal support (multiple detections)
  3. **Urgent Reaction**: Panic behavior without weapons, combined with other signals
  4. **CLIP-only Gun**: Very strong CLIP weapon signal without YOLO confirmation

#### **Crime Categories**

1. **SHOPLIFTING**: Behavior-based detection via CLIP scene understanding
2. **GUN**: YOLO + CLIP validated weapon detections with temporal consistency
3. **URGENT_REACTION**: Panic/rushing behavior detection
4. **NORMAL**: No crime detected, calm scenes

---

## 📁 File Structure

```
mp4_id_detector/
├── video_id_detector2_optimized.py    # Main person re-identification script
├── crime_no_crime_zero_shot1.py        # Crime detection script
├── export_annotated_videoes_v3_from_cache.py  # Annotation script
├── create_2x2_grid_video.py            # 2x2 grid video creator
├── outputs_v3/                         # Output directory
│   ├── annotated_videos_v3/            # Individual annotated videos
│   ├── global_identity_catalogue_v3.json  # Global identity data
│   ├── tracklet_to_global_id.npz      # ID mappings
│   ├── track_embeddings_v3.npz        # Embedding cache
│   └── grid_2x2_annotated.mp4         # 2x2 grid video
├── videos/                            # Input videos
│   ├── 1_upscaled.mp4
│   ├── 2_upscaled.mp4
│   ├── 3_upscaled.mp4
│   └── 4_upscaled.mp4
├── clip{0,1,2,3}_crime_validated.json # Crime detection results per clip
├── scene_labels_validated.json        # Dataset-level crime summary
└── weapon_yolov8_gun.pt              # Gun detection model (optional)
```

---

## 🔧 Configuration

### **Person Re-identification Configuration**

#### **Video Input**
```python
VIDEO_DIR = "./videos/"
VIDEO_FILES = ["1_upscaled.mp4", "2_upscaled.mp4", "3_upscaled.mp4", "4_upscaled.mp4"]
```

#### **Model Configuration**
```python
YOLO_WEIGHTS = "yolov8n.pt"           # Person detection
REID_MODEL = "osnet_ibn_x1_0"         # Primary ReID model
USE_VIT_REID = True                   # Enable TransReID
VIT_MODEL = "vit_base_patch16_224"     # Vision Transformer model
FACE_MODEL = "buffalo_l"               # Face recognition model
DEVICE = "cpu"                         # CPU/GPU device
```

#### **Clustering Parameters**
```python
# Within-clip clustering
WITHIN_CLIP_FACE_WEIGHT = 0.4          # Face weight (40%)
WITHIN_CLIP_THRESHOLD = 0.15           # Base threshold
USE_PER_CLIP_THRESHOLDS = True         # Enable automatic per-clip analysis

# Cross-clip merging
CROSS_CLIP_FACE_WEIGHT = 0.45          # Face weight (45%)
CROSS_CLIP_THRESHOLD = 0.46            # Base threshold
```

#### **Feature Extraction Options**
```python
USE_TTA = True                         # Test-Time Augmentation
USE_TEMPORAL_SMOOTHING = True          # Smooth embeddings
USE_POSE_FEATURES = True               # MediaPipe pose features
USE_RERANKING = True                  # k-Reciprocal re-ranking
USE_CAMERA_BIAS = True                # Camera bias correction
USE_ENSEMBLE_REID = True              # Ensemble OSNet + TransReID
```

### **Crime Detection Configuration**

#### **Model Paths**
```python
YOLO_WEIGHTS = "yolov8n.pt"           # Person detection
WEAPON_MODEL = "weapon_yolov8_gun.pt" # Gun detection (optional)
CLIP_MODEL = "openai/clip-vit-base-patch32"
```

#### **Detection Parameters**
```python
SAMPLE_RATE = 30                      # Analyze every Nth frame
THRESHOLD = 0.5                       # Crime detection threshold
DEVICE = "cpu"                        # CPU/GPU device
```

---

## 🚀 Usage

### **Person Re-identification**

#### **Step 1: Run Main Detection Script**
```bash
# Activate virtual environment
source cvenv/bin/activate

# Run the main detection and clustering script
python video_id_detector2_optimized.py
```

**What it does:**
1. Loads models (YOLO, OSNet, TransReID, InsightFace)
2. Processes each video clip sequentially
3. Detects and tracks persons using YOLO + ByteTrack
4. Extracts features (body + face embeddings)
5. Performs Stage 1: Within-clip clustering (strict separation)
6. Performs Stage 2: Cross-clip merging (lenient matching)
7. Exports results to JSON, NPZ, and CSV formats

#### **Step 2: Generate Annotated Videos**
```bash
# Create individual annotated videos for each clip
python export_annotated_videoes_v3_from_cache.py

# Optional: Create GIF previews
python create_gif_previews.py
```

**Output:** Creates `annotated_videos_v3/clip{0,1,2,3}_annotated.mp4` with:
- Color-coded bounding boxes for each global ID
- Global ID labels on each detection
- Legend showing all global IDs in the clip

#### **Step 3: Create 2x2 Grid Video** 🎬
```bash
# Combine all 4 annotated clips into a 2x2 grid
python create_2x2_grid_video.py
```

**Output:** Creates `outputs_v3/grid_2x2_annotated.mp4` - a synchronized 2x2 grid view

---

### **Crime Detection**

#### **Single Video Mode**
```bash
# Activate virtual environment
source cvenv/bin/activate

# Analyze a single video
python crime_no_crime_zero_shot1.py --video ./videos/1_upscaled.mp4 --sample-rate 30

# Save results to JSON
python crime_no_crime_zero_shot1.py --video ./videos/1_upscaled.mp4 --output clip0_crime.json
```

#### **Batch Mode** (Process all clips)
```bash
# Process all clips in videos directory
python crime_no_crime_zero_shot1.py --batch ./videos --sample-rate 30
```

**Output:**
- Individual clip results: `clip{0,1,2,3}_crime_validated.json`
- Dataset summary: `scene_labels_validated.json`

#### **Example Output**
```json
{
  "crime_detected": true,
  "crime_type": "GUN",
  "confidence": 0.95,
  "detailed_scores": {
    "shoplifting": {"avg": 0.12, "max": 0.25},
    "weapon": {"avg": 0.45, "max": 0.78},
    "normal": {"avg": 0.35}
  },
  "validated_weapons": 4,
  "people_stats": {"avg": 2.5, "max": 4}
}
```

---

## 📊 Output Files

### **Person Re-identification Outputs**

#### **1. Global Identity Catalogue (JSON)**
Contains summary and detailed appearance information for each global identity across all clips.

#### **2. Tracklet Assignments (NPZ)**
- `tracklet_to_global_id.npz`: Maps each tracklet to its global ID
- `track_embeddings_v3.npz`: Cached embeddings for fast reprocessing

#### **3. Annotated Videos**
- Individual clips: `clip{0,1,2,3}_annotated.mp4`
- 2x2 Grid: `grid_2x2_annotated.mp4`

### **Crime Detection Outputs**

#### **1. Per-Clip Results (JSON)**
Contains crime detection results, confidence scores, and detailed frame-by-frame analysis.

#### **2. Dataset Summary (JSON)**
Contains dataset-level summary with crime labels for all clips.

---

## 🔍 How It Works

### **Person Re-identification: Distance Calculation**

#### **Within-Clip Distance**
- Uses weighted combination of body (60%) and face (40%) embeddings when both available
- Uses body-only when face information is missing
- Implements **cohesion check**: tracklet must be similar to ALL cluster members
- Applies adaptive similarity requirements based on temporal overlap and face availability
- Hard blocks merging when temporal overlap > 50% (different people)

#### **Cross-Clip Distance**
- Uses **best-of-cluster matching**: finds minimum distance among all tracklet pairs
- Prevents single bad tracklet from dominating cluster comparison
- Recalculates distances after Pass 1 merges using all tracklets from merged clusters
- Uses length-weighted averaging: longer tracklets contribute more to cluster embeddings
- Applies adaptive thresholds based on tracklet length, face availability, and other factors

### **Clustering Logic Flow**

#### **Stage 1: Within-Clip**
1. Group tracklets by clip index
2. For each clip:
   - Determine adaptive threshold (automatic or manual override)
   - Calculate pairwise distance matrix (body + face)
   - Apply temporal overlap analysis with hard blocking
   - Use adaptive similarity requirements based on overlap and face presence
   - Require cohesion: tracklet must meet threshold against ALL cluster members
   - Assign to best matching cluster or create new one
3. Result: Person Clusters (temp_global_id) within each clip

#### **Stage 2: Cross-Clip**
1. Consolidate clusters using length-weighted averaging
2. Calculate best-of-cluster distances
3. Collect merge candidates with adaptive thresholds
4. **Pass 1**: Process face-to-no-face matches (more lenient, locks clusters)
5. **Pass 2**: Process face-to-face matches (stricter, recalculates distances, best-match logic)
6. Validate and fix same-clip duplicates
7. Result: Global IDs assigned across all clips

### **Crime Detection: Decision Making**

The system uses a priority-based decision tree:
1. **Shoplifting**: Strong behavioral signal from CLIP
2. **Gun Detection**: Validated YOLO + CLIP detections with temporal consistency
3. **Urgent Reaction**: Panic behavior combined with other signals
4. **CLIP-only Gun**: Very strong scene-level weapon signal
5. **Normal**: No crime detected

Confidence scores are calculated based on:
- Combined YOLO + CLIP validation scores (for guns)
- Temporal consistency (multiple detections)
- Scene-level CLIP scores (for behavior-based crimes)

---

## 🎨 Annotated Videos

### **Color Coding**
Each global ID is assigned a unique color from a predefined palette (Green, Red, Blue, Yellow, Magenta, Cyan, etc.)

### **Visualization Elements**
- **Bounding Boxes**: Thick colored rectangles around detected persons
- **Labels**: "ID: X" text on bounding box with colored background
- **Legend**: List of all global IDs present in the clip (top-left corner)

### **2x2 Grid Video** 🎬
Combines all 4 annotated clips into a single synchronized view:
- Perfect synchronization: frame-by-frame aligned playback
- Preserves aspect ratios
- Same color coding and labels as individual videos
- Clear separation with borders and spacing
- Perfect for cross-clip validation and presentations

---

## ⚡ Performance

### **Caching System**
- NPZ cache files store pre-computed embeddings and mappings
- Fast reprocessing when cache is available
- Enables quick parameter tuning without reprocessing videos

### **Optimization Features**
- Frame sampling: Processes every 10th frame for feature extraction (person re-id)
- Temporal sampling: Processes every 30th frame for crime detection (~1 FPS)
- Length-weighted averaging: Longer tracklets contribute more
- Best-of-cluster matching: Efficient comparison of merged clusters
- Vectorized operations: NumPy operations for speed

---

## 🛠️ Customization

### **Person Re-identification**

#### **Adjusting Thresholds**
Edit configuration section in `video_id_detector2_optimized.py`:
```python
WITHIN_CLIP_THRESHOLD = 0.15           # Base within-clip threshold
CROSS_CLIP_THRESHOLD = 0.46            # Cross-clip base threshold
```

#### **Per-Clip Thresholds**
Manually override thresholds for specific clips:
```python
PER_CLIP_THRESHOLDS = {
    0: 0.75,  # Stricter for clip 0
    2: 0.65,  # More lenient for clip 2
}
```

#### **Feature Weights**
Adjust face/body weights:
```python
WITHIN_CLIP_FACE_WEIGHT = 0.4          # Face weight within clip
CROSS_CLIP_FACE_WEIGHT = 0.45          # Face weight across clips
```

### **Crime Detection**

#### **Adjusting Detection Sensitivity**
```bash
--threshold 0.5  # Crime detection threshold (0.0-1.0)
--sample-rate 30  # Analyze every Nth frame (lower = more thorough but slower)
```

#### **CLIP Validation Strictness**
Modify `validate_gun_with_clip()` function thresholds in the code for more/less strict validation.

---

## 📈 Results Analysis

### **Person Re-identification**

The system produces global identities from tracklets across multiple video clips, showing:
- Which people appear in which clips
- Cross-clip matching results
- Detailed appearance information

### **Crime Detection**

Results show:
- Crime type detected (GUN, SHOPLIFTING, etc.)
- Confidence scores
- Number of validated weapons (for gun detection)
- Frame-by-frame analysis
- Dataset-level summary

---

## 🎯 Use Cases

### **Research & Development**
- Algorithm testing and evaluation
- Parameter tuning and optimization
- Feature analysis and comparison
- Cross-clip matching validation
- Zero-shot crime detection research

### **Production Deployment**
- Multi-camera surveillance systems
- Person tracking across different scenes
- Identity verification and validation
- Quality assurance and monitoring
- Real-time crime detection in retail environments
- Security monitoring systems

---

## 🔧 Troubleshooting

### **Common Issues**

1. **Missing cache files**
   - Script will process from scratch
   - First run takes longer but creates cache

2. **Model loading errors**
   - Check model file paths
   - Ensure models are downloaded
   - Verify device compatibility (CPU/GPU)
   - For gun detection: Download `weapon_yolov8_gun.pt` or disable weapon detector

3. **Memory issues**
   - Reduce batch size
   - Use CPU instead of GPU
   - Process fewer clips at once
   - Increase sample rate for crime detection

4. **Video format issues**
   - Ensure MP4 format
   - Check video codec compatibility
   - Verify video file paths

5. **CLIP validation too strict/lenient**
   - Adjust validation thresholds in `validate_gun_with_clip()`
   - Modify combined score calculation
   - Adjust temporal consistency requirements

### **Performance Tips**
- Use existing cache for person re-id
- CPU processing: Set `DEVICE = "cpu"` for compatibility
- Reduce video resolution for faster processing
- Adjust sampling rates for speed vs. accuracy tradeoff

---

## 📝 License

MIT License - See LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Performance optimization
- New feature extraction methods
- Improved clustering algorithms
- Better visualization tools
- Enhanced crime detection accuracy
- Documentation enhancements

---

## 📚 References

### **Person Re-identification**
- OSNet (IBN-Net): [Torchreid](https://github.com/KaiyangZhou/deep-person-reid)
- TransReID: [TransReID Repository](https://github.com/damo-cv/TransReID)
- InsightFace: [Face Recognition](https://github.com/deepinsight/insightface)
- ByteTrack: [Multi-Object Tracking](https://github.com/ifzhang/ByteTrack)

### **Crime Detection**
- YOLOv8: [Ultralytics](https://github.com/ultralytics/ultralytics)
- CLIP: [OpenAI CLIP](https://github.com/openai/CLIP)
- CLAHE: [OpenCV](https://docs.opencv.org/)

---

**Note**: This system uses sophisticated two-stage adaptive clustering and hybrid YOLO + CLIP detection approaches that have been carefully tuned for accuracy across different video conditions, person appearances, and crime scenarios.
