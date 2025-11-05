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

##### **Stage 1: Within-Clip Clustering (Strict Separation)**
- **Purpose**: Ensures different people in the same scene are kept separate
- **Face Weight**: 40% (high weight since faces are reliable within same lighting/angle)
- **Body Weight**: 60%
- **Base Threshold**: 0.15 (used when automatic per-clip analysis is disabled)
- **Temporal Overlap Analysis**: 
  - **Hard block**: Overlap > 50% (different people appearing simultaneously) - cluster is skipped
  - **Adaptive similarity thresholds** based on overlap ratio:
    - **Overlap > 40%**: Required body similarity = `0.80` (very strict)
      - No face: Minimum `0.80`
      - Face-to-face: `max(0.80, 0.82)` (stricter)
      - Containment bonus: `+0.05`
      - Duration ratio ≥ 2.0: `+0.03`
    - **Overlap > 10%**: Required body similarity = `0.70` (lenient)
      - No face: Minimum `0.87` (very strict)
      - Face-to-face: `max(0.70, 0.82)` (stricter)
      - Containment bonus: `+0.05`
      - Duration ratio ≥ 2.0: `+0.03`
    - **Overlap ≤ 10%**: Required body similarity = `0.70` (very lenient)
      - No face or duration ratio ≥ 2.0: `max(0.70, 0.87)` or `0.80`
      - Face-to-face: `max(0.70, 0.83)` (stricter for back-view cases like Clip 2)
  - All thresholds have minimum floor of `0.65`
- **Cohesion Check**: Tracklet must meet threshold against ALL cluster members (not just average)
- **Adaptive Thresholds**: When `USE_PER_CLIP_THRESHOLDS = True`, automatically adjusts based on:
  - Number of people in clip (crowding analysis)
  - Face visibility (face detection rate)
  - Lighting/angle conditions (diversity analysis)
  - Tracking quality (tracking stability)
- **Per-Clip Analysis**: Each clip gets an automatically calculated threshold based on its characteristics, or manual override via `PER_CLIP_THRESHOLDS` dictionary

##### **Stage 2: Cross-Clip Merging (Lenient Matching)**
- **Purpose**: Matches the same person across different scenes using image-only embeddings (body + face)
- **Face Weight**: 45% (slightly higher than within-clip 0.4 for cross-camera matching)
- **Body Weight**: 55%
- **Base Threshold**: 0.46 (used as baseline for adaptive adjustments)
- **Two-Pass Merging System**:
  1. **Pass 1: Face-to-No-Face Matches**
     - Processes pairs where one or both clusters lack faces
     - Sorted by **body similarity** (higher = better, processed first)
     - Threshold adjustment: `-0.08` when one has face but not both (makes merging easier)
     - Body similarity requirement: `0.65` (lowered for face/no-face pairs)
     - **Locking**: Clusters merged in Pass 1 are locked to prevent conflicts
     - All clusters in a merged group are locked together
  2. **Pass 2: Face-to-Face Matches**
     - Processes pairs where both clusters have faces
     - **Distance Recalculation**: Distances are recomputed using ALL tracklets from merged clusters (if Pass 1 merged clusters)
     - Uses **best-of-cluster matching**: Finds minimum distance among all tracklet pairs in merged clusters
     - Sorted by **distance** (lower = better)
     - Body similarity requirement: `0.7` (stricter for face-to-face pairs)
     - **Best-Match Logic**: Each target cluster is matched with its BEST (lowest distance) source candidate
     - Only processes unlocked clusters (or allows locked clusters to merge with NEW unlocked clusters)
     - Skips if both clusters are locked (prevents re-merging)
- **Best-of-Cluster Matching**: Compares minimum distance among all tracklet pairs to prevent single bad tracklet from dominating
- **Cluster Representation**: Uses length-weighted averaging of tracklet embeddings (longer tracklets contribute more)
- **Adaptive Thresholds**: Based on:
  - **Tracklet length**: 
    - High confidence (longer >500 frames): `base - 0.02 = 0.44` (more lenient)
    - Medium confidence (300-500 frames): `base = 0.46`
    - Medium+ confidence (250-300 frames): `base + 0.03 = 0.49`
    - Low confidence (shorter <250 frames): `base + 0.05 = 0.51` (very lenient)
  - **Face availability**: 
    - Face-to-no-face: Threshold reduced by `0.08` (easier to merge)
    - Face-to-face: Uses stricter thresholds
  - **Same-clip blocking**: Prevents merging clusters from the same clip (hard constraint)
  - **Transitive blocking**: Prevents indirect merging of blocked clusters (can be overridden by Hausdorff distance <50 pixels)

#### **Multi-Model Ensemble**
- **OSNet (IBN-Net)**: 512D body embeddings (weight: 70%)
- **TransReID (ViT)**: 768D body embeddings (weight: 30%)
- **Combined Body**: 1280D weighted ensemble (when `USE_ENSEMBLE_REID = True`)
- **InsightFace**: 512D face embeddings (when face detected)
- **MediaPipe**: Pose keypoints (66D, enabled when `USE_POSE_FEATURES = True`)

#### **Advanced Features**
- **Test-Time Augmentation (TTA)**: Horizontal flip for view invariance (`USE_TTA = True`)
- **Temporal Smoothing**: Smooths embeddings within tracks for stability (`USE_TEMPORAL_SMOOTHING = True`)
- **k-Reciprocal Re-ranking**: Improves matching accuracy using reciprocal neighbors (`USE_RERANKING = True`, k=25)
- **Camera Bias Correction**: Accounts for camera-specific variations (`USE_CAMERA_BIAS = True`)
- **Pose Features**: MediaPipe pose estimation for additional person characterization (`USE_POSE_FEATURES = True`)
- **Quality-Weighted Embeddings**: Higher weight for frames with faces and good pose detection
- **Representative Frame Selection**: Chooses diverse high-quality frames for multi-frame matching
- **Caching System**: NPZ file caching for fast parameter tuning and development
- **Length-Weighted Averaging**: Longer tracklets contribute more to cluster embeddings

#### **2x2 Grid Video Output** 🎬
- **Combines all 4 annotated clips** into a single synchronized 2x2 grid view
- **Output file**: `outputs_v3/grid_2x2_annotated.mp4`
- **Perfect for visualization**: See the same person across different clips simultaneously
- **Synchronized playback**: All clips play frame-by-frame in perfect sync
- **Use case**: Ideal for verifying cross-clip person matching and comparing same person across different views

---

### **2. Zero-Shot Crime Detection System**

**File**: `crime_no_crime_zero_shot1.py`

#### **Hybrid Detection Architecture**

Combines multiple models for robust crime detection:
- **YOLOv8**: Person and object detection (COCO dataset)
- **Custom Weapon YOLO**: Specialized gun detection model (`weapon_yolov8_gun.pt`)
- **CLIP (OpenAI)**: Zero-shot scene understanding and gun validation

#### **Detection Pipeline**

1. **Frame Preprocessing**
   - **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Normalizes lighting across different clips
   - **Mild Sharpening**: Enhances edges and small objects (like guns)
   - Critical for detecting guns in varying lighting conditions

2. **YOLO Object Detection**
   - Detects people using YOLOv8 COCO (class 0)
   - Detects weapons using specialized gun model (if available)
   - **CLIP Validation**: Each YOLO gun detection is validated using CLIP
     - Asks CLIP: "Is this a gun or a smartphone?"
     - Filters false positives (iPhones, hands, etc.)
     - **Strict Validation Logic**:
       - `gun_prob > phone_prob + 0.25`: Very strong gun evidence (validation_score = 0.9)
       - `gun_prob > phone_prob + 0.15` AND `gun_confidence > 0.50`: Strong evidence (validation_score = 0.7)
       - `gun_prob > phone_prob + 0.05` AND `gun_confidence > 0.65`: Marginal but high YOLO confidence (validation_score = 0.6)
       - Otherwise: Rejected (validation_score = 0.2)
     - Combined score: `(YOLO_confidence + CLIP_validation_score) / 2`

3. **CLIP Scene Analysis**
   - Zero-shot classification using predefined crime labels:
     - **Shoplifting**: "person secretly stealing items and hiding them in pockets or bags", etc.
     - **Guns**: "person holding a handgun or pistol", "person with a gun or firearm", etc.
     - **Urgent Reaction**: "person rushing or running suddenly in panic", etc.
     - **Normal**: "normal shopping in a store with customers browsing", etc.
   - Returns probability scores for each category

4. **Temporal Consistency Filter**
   - Requires guns to appear in **multiple frames** (≥2 frames) to avoid false positives
   - Checks if gun detections are temporally close (within 3-frame window)
   - Isolated single-frame detections are rejected

5. **Crime Decision Logic**
   - **Priority 1: Shoplifting** (if strong signal)
     - `avg_shoplifting > 0.50` AND `max_shoplifting > 0.70`
   - **Priority 2: Gun Detection** (with CLIP validation)
     - Uses **combined scores** (YOLO + CLIP validation)
     - **High confidence** (>0.7 combined): Trust validation, confidence = 0.95
     - **Medium confidence** (0.5-0.7): Reduced filtering, confidence = 0.90
     - **Low confidence** (<0.5): Need temporal support (≥3 validated guns), confidence = 0.85
   - **Priority 3: Urgent Reaction** (without YOLO guns)
     - `avg_urgent_reaction > 0.40` OR `max_urgent_reaction > 0.65`
     - Combined with shoplifting or weapon scene scores
   - **Priority 4: CLIP-only Gun** (if very strong CLIP signal)
     - `avg_weapon > 0.60` OR `max_weapon > 0.80`

#### **Crime Categories**

1. **SHOPLIFTING**
   - Behavior-based detection via CLIP
   - Multiple specific labels for robust detection
   - Strong signal required (avg > 0.50, max > 0.70)

2. **GUN**
   - YOLO + CLIP validated detections
   - Temporal consistency required
   - Combined confidence scoring

3. **URGENT_REACTION**
   - Panic/rushing behavior detection
   - Often combined with other crime signals

4. **NORMAL**
   - No crime detected
   - Calm, peaceful scenes

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
│   │   ├── clip0_annotated.mp4
│   │   ├── clip1_annotated.mp4
│   │   ├── clip2_annotated.mp4
│   │   └── clip3_annotated.mp4
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
├── weapon_yolov8_gun.pt              # Gun detection model (optional)
└── README.md                          # This file
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
WITHIN_CLIP_FACE_WEIGHT = 0.4          # Face weight within same clip (40%)
WITHIN_CLIP_THRESHOLD = 0.15           # Base similarity threshold
USE_PER_CLIP_THRESHOLDS = True         # Enable automatic per-clip threshold analysis

# Cross-clip merging
CROSS_CLIP_FACE_WEIGHT = 0.45          # Face weight across clips (45%)
CROSS_CLIP_THRESHOLD = 0.46            # Base threshold for cross-clip matching
```

#### **Feature Extraction & Advanced Options**
```python
USE_TTA = True                         # Test-Time Augmentation (horizontal flip)
USE_TEMPORAL_SMOOTHING = True          # Smooth embeddings within tracks
USE_POSE_FEATURES = True               # MediaPipe pose features (66D)
USE_RERANKING = True                  # k-Reciprocal re-ranking (k=25)
USE_CAMERA_BIAS = True                # Camera bias correction
USE_ENSEMBLE_REID = True              # Ensemble OSNet + TransReID
ENSEMBLE_WEIGHTS = [0.7, 0.3]          # [OSNet weight, TransReID weight]
```

### **Crime Detection Configuration**

#### **Model Paths**
```python
# YOLO models
YOLO_WEIGHTS = "yolov8n.pt"           # Person detection (COCO)
WEAPON_MODEL = "weapon_yolov8_gun.pt" # Gun detection (optional, custom trained)

# CLIP model
CLIP_MODEL = "openai/clip-vit-base-patch32"
```

#### **Detection Parameters**
```python
SAMPLE_RATE = 30                      # Analyze every Nth frame (30 = ~1 FPS for 30fps video)
THRESHOLD = 0.5                       # Crime detection threshold (0.0-1.0)
DEVICE = "cpu"                        # CPU/GPU device
```

#### **CLIP Validation Thresholds**
```python
# Gun validation (strict to filter false positives)
STRONG_GUN_MARGIN = 0.25              # gun_prob > phone_prob + 0.25 → validation_score = 0.9
MEDIUM_GUN_MARGIN = 0.15              # gun_prob > phone_prob + 0.15 → validation_score = 0.7
MARGINAL_GUN_MARGIN = 0.05            # gun_prob > phone_prob + 0.05 → validation_score = 0.6
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

# Optional: Create GIF previews for README display
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
python crime_no_crime_zero_shot1.py --video ./videos/1_upscaled.mp4 --sample-rate 30 --threshold 0.5

# Save results to JSON
python crime_no_crime_zero_shot1.py --video ./videos/1_upscaled.mp4 --output clip0_crime.json
```

#### **Batch Mode** (Process all clips)
```bash
# Process all clips in videos directory
python crime_no_crime_zero_shot1.py --batch ./videos --sample-rate 30 --threshold 0.5
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
```json
{
  "summary": {
    "total_clips": 4,
    "total_tracklets": 17,
    "total_global_identities": 8
  },
  "identities": {
    "0": {
      "global_id": 0,
      "appearances": [
        {
          "clip_idx": 0,
          "frames": [0, 1, 2, ...],
          "bboxes": [[x1, y1, x2, y2], ...]
        }
      ]
    }
  }
}
```

#### **2. Tracklet Assignments (NPZ)**
- `tracklet_to_global_id.npz`: Maps each tracklet to its global ID
- `track_embeddings_v3.npz`: Cached embeddings for fast reprocessing

#### **3. Annotated Videos**
- Individual clips: `clip{0,1,2,3}_annotated.mp4`
- 2x2 Grid: `grid_2x2_annotated.mp4`

### **Crime Detection Outputs**

#### **1. Per-Clip Results (JSON)**
```json
{
  "crime_detected": true,
  "crime_type": "GUN",
  "confidence": 0.95,
  "detailed_scores": {...},
  "detections_per_frame": [...],
  "validated_weapons": 4
}
```

#### **2. Dataset Summary (JSON)**
```json
{
  "dataset": "mp4_files_id_detectors_upscaled_validated",
  "scene_labels": [
    {"clip_id": 0, "label": "crime", "category": "GUN", "confidence": 0.95},
    {"clip_id": 1, "label": "normal", "category": "NORMAL", "confidence": 0.85}
  ]
}
```

---

## 🔍 Algorithm Details

### **Distance Calculation (Person Re-identification)**

#### **Within-Clip Distance**
- **Distance Calculation**: Uses weighted combination of body and face embeddings
- When both tracklets have faces:
  ```
  distance = 0.60 × body_distance + 0.40 × face_distance
  ```
- When one or both lack faces:
  ```
  distance = body_distance (face_weight = 0)
  ```
- **Cohesion Check**: Tracklet must meet similarity threshold against ALL cluster members (not just average distance)
- **Adaptive Body Similarity Requirements** (not distance thresholds, but similarity requirements):
  - Based on temporal overlap ratio (0.80, 0.70, etc.)
  - Face availability adjustments (0.80-0.87 depending on overlap)
  - Containment bonus: `+0.05`
  - Duration ratio ≥ 2.0: `+0.03`
  - All requirements have minimum floor of `0.65`
- **Hard Block**: Overlap > 50% (skip cluster, different people)

#### **Cross-Clip Distance**
- **Best-of-Cluster Matching**: Finds minimum distance among ALL tracklet pairs between clusters
  - Prevents single bad tracklet from dominating the cluster embedding
  - When clusters merge (Pass 1), distances are recomputed using all tracklets from merged clusters
- **Distance Calculation**:
  - When both clusters have faces:
    ```
    distance = min(0.55 × body_distance + 0.45 × face_distance) across all tracklet pairs
    ```
  - When one or both lack faces:
    ```
    distance = min(body_distance) across all tracklet pairs (face_weight = 0)
    ```
- **Cluster Representation**: Uses length-weighted averaging - longer tracklets contribute more to cluster embeddings
- **Adaptive Threshold**: Based on tracklet length, face availability, and body similarity requirements

### **Clustering Logic Flow**

#### **Stage 1: Within-Clip**
1. Group tracklets by clip index
2. For each clip:
   - Determine threshold (automatic analysis or manual override)
   - Calculate pairwise distance matrix (body + face)
   - Apply temporal overlap analysis
   - Hard block: Skip if overlap > 50%
   - Adaptive threshold based on overlap ratio and face presence
   - Cohesion check: Tracklet must meet threshold against ALL cluster members
   - Assign to best matching cluster or create new one
3. Result: Person Clusters (temp_global_id) within each clip

#### **Stage 2: Cross-Clip**
1. **Cluster Consolidation**: Compute per-cluster representative embeddings using length-weighted averaging
2. **Distance Computation**: Calculate best-of-cluster distances (minimum among all tracklet pairs)
3. **Candidate Collection**: Collect all merge candidates with adaptive thresholds
   - Check same-clip overlap block (hard constraint)
   - Check transitive blocking (can be overridden by Hausdorff distance <50 pixels)
   - Calculate adaptive threshold based on tracklet length and face availability
   - Face-to-no-face pairs get -0.08 threshold adjustment
4. **Pass 1: Face-to-No-Face Merging**
   - Sort candidates by body similarity (higher = better, processed first)
   - Check if clusters are already locked (skip if locked)
   - Check same-clip overlap block (hard constraint)
   - Check for confirmed face-to-face matches (prevent conflicts)
   - Check for multiple same-source-clip merges (prevent conflicts)
   - Merge if distance < adaptive threshold AND body similarity ≥ 0.65
   - **Lock all merged clusters** (prevents conflicts in Pass 2)
5. **Pass 2: Face-to-Face Merging**
   - **Distance Recalculation**: Recompute distances using ALL tracklets from merged clusters (if Pass 1 merged any)
   - Prioritize face-to-face pairs when recomputing distances
   - **Best-Match Logic**: Find best (lowest distance) source candidate for each target cluster
   - Sort best candidates by distance (lower = better)
   - Skip if both clusters are locked (prevents re-merging)
   - Allow locked clusters to merge with NEW unlocked clusters
   - Check same-clip overlap block (hard constraint)
   - Check for confirmed face-to-face matches (prevent conflicts)
   - Merge if distance < adaptive threshold AND body similarity ≥ 0.7
   - Record confirmed face-to-face matches
6. **Validation**: Fix same-clip duplicates (prevent multiple tracklets from same clip in same global ID)
7. **Result**: Global IDs assigned across all clips

### **Adaptive Thresholds**

#### **Within-Clip Adaptive Thresholds**
- **Base threshold**: 0.15 (when auto-analysis disabled via `USE_PER_CLIP_THRESHOLDS = False`)
- **Automatic per-clip threshold**: Calculated based on clip characteristics when `USE_PER_CLIP_THRESHOLDS = True`
- **Overlap-based body similarity requirements**:
  - **Overlap > 40%**: Required body similarity = `0.80`
  - **Overlap > 10%**: Required body similarity = `0.70`
  - **Overlap ≤ 10%**: Required body similarity = `0.70`
- **All thresholds have minimum floor**: `0.65`
- **Hard block**: Overlap > 50% (different people, skip cluster)

#### **Cross-Clip Adaptive Thresholds**
- **Base threshold**: 0.46 (used as baseline for distance threshold)
- **Body similarity requirements**:
  - Face-to-face: `0.7` (stricter)
  - Face-to-no-face: `0.65` (more lenient)
- **Length-based distance threshold adjustments**:
  - **High confidence** (long tracklets >500 frames): `base - 0.02 = 0.44` (more lenient)
  - **Medium confidence** (300-500 frames): `base = 0.46`
  - **Medium+ confidence** (250-300 frames): `base + 0.03 = 0.49`
  - **Low confidence** (short tracklets <250 frames): `base + 0.05 = 0.51` (very lenient)
- **Face availability adjustments**:
  - **Face-to-no-face pairs**: Distance threshold reduced by `0.08` (easier to merge)
  - **Face-to-face pairs**: Uses stricter thresholds (no adjustment)
- **Same-clip blocking**: Hard constraint (never merge clusters from same clip)
- **Transitive blocking**: Prevents indirect merging (can be overridden by Hausdorff distance <50 pixels)

---

## 🎨 Annotated Videos

### **Color Coding**
Each global ID is assigned a unique color from a predefined palette:
- Global ID 0: Green
- Global ID 1: Red
- Global ID 2: Blue
- Global ID 3: Yellow
- Global ID 4: Magenta
- Global ID 5: Cyan
- ... (cycles through palette)

### **Visualization Elements**
- **Bounding Boxes**: Thick colored rectangles around detected persons
- **Labels**: "ID: X" text on bounding box with colored background
- **Legend**: List of all global IDs present in the clip (top-left corner)

### **2x2 Grid Video** 🎬
The grid video combines all 4 annotated clips into a single synchronized view for easy comparison.

**Output file**: `outputs_v3/grid_2x2_annotated.mp4` (created by `create_2x2_grid_video.py`)

**Key Features:**
- **Perfect Synchronization**: All clips play frame-by-frame in perfect sync
- **Aspect Ratio Preservation**: Each clip maintains its original aspect ratio
- **Visual Consistency**: Same color coding and labels as individual annotated videos
- **Clear Separation**: White borders and spacing between clips
- **Clip Identification**: Labels (Clip 0-3) clearly marked on each cell

**Perfect For:**
- ✅ **Cross-clip validation**: See the same person (same global ID) across different clips simultaneously
- ✅ **View comparison**: Compare front view, back view, side view of the same person
- ✅ **Quality assurance**: Quickly verify clustering accuracy
- ✅ **Presentations**: Professional visualization for demonstrations and reports

---

## ⚡ Performance

### **Caching System**
- **NPZ Cache Files**: Pre-computed embeddings and mappings
- **Fast Reprocessing**: Load from cache when available
- **Parameter Tuning**: Iterate quickly without reprocessing videos

### **Optimization Features**
- **Frame Sampling**: Processes every 10th frame for feature extraction (person re-id)
- **Temporal Sampling**: Processes every 30th frame for crime detection (~1 FPS)
- **Length-Weighted Averaging**: Longer tracklets contribute more to embeddings
- **Best-of-Cluster Matching**: Efficient comparison of merged clusters
- **Vectorized Operations**: NumPy operations for speed

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
```python
# In crime_no_crime_zero_shot1.py or via command line
--threshold 0.5  # Crime detection threshold (0.0-1.0)
--sample-rate 30  # Analyze every Nth frame (lower = more thorough but slower)
```

#### **CLIP Validation Strictness**
Modify `validate_gun_with_clip()` function thresholds:
```python
# More strict (fewer false positives, more false negatives)
if gun_prob > phone_prob + 0.30:  # Increased from 0.25

# More lenient (more detections, more false positives)
if gun_prob > phone_prob + 0.20:  # Decreased from 0.25
```

---

## 📈 Results Analysis

### **Person Re-identification**

The system produces **8 global identities** from **17 tracklets** across **4 video clips**:

- **Clip 0**: Tracklets → Global IDs
- **Clip 1**: Tracklets → Global IDs  
- **Clip 2**: Tracklets → Global IDs
- **Clip 3**: Tracklets → Global IDs

**Cross-Clip Matching:**
- Global ID 0: Appears in Clip 0 and Clip 2
- Global ID 1: Appears in Clip 0 and Clip 1
- Global ID 2: Appears in Clip 1 and Clip 2
- ... (see JSON output for complete mapping)

### **Crime Detection**

**Example Results:**
- Clip 0: GUN detected (confidence: 0.95, 4 validated weapons)
- Clip 1: NORMAL (confidence: 0.85)
- Clip 2: SHOPLIFTING detected (confidence: 0.78)
- Clip 3: GUN detected (confidence: 0.90, 2 validated weapons)

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
- Use existing cache: Set `USE_EXISTING_CACHE = True` (person re-id)
- CPU processing: Set `DEVICE = "cpu"` for compatibility
- Reduce video resolution: For faster processing
- Adjust sampling rate: Increase `SAMPLING_RATE` for speed (person re-id)
- Increase sample rate: Use `--sample-rate 60` for crime detection (faster but less thorough)

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

