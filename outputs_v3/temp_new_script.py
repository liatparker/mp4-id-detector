import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from ultralytics import YOLO
import torchreid
import supervision as sv
import insightface
import mediapipe as mp
import sys
import sys
sys.path.append('/Users/liatparker/vscode_agentic/mp4_id_detector/TransReID')
from model.make_model import make_model  # This is all you need

# Post-processing functions removed - let clustering handle person separation
from config.defaults import _C as cfg_default  # This is correct
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

"""
Video ID Detector 2 - Advanced Person Re-identification System

OVERVIEW:
This system performs person re-identification across multiple video clips using advanced
computer vision and machine learning techniques. It can identify the same person
appearing in different video clips, even when viewed from different angles or under
different lighting conditions.

KEY FEATURES:
1. Multi-Model Ensemble: Combines OSNet and TransReID for robust feature extraction
2. Two-Stage Clustering: Strict within-clip separation, lenient cross-clip matching
3. View Invariance: Test-Time Augmentation and temporal smoothing for robust embeddings
4. Physical Proximity Logic: Uses spatial relationships to resolve ambiguous cases
5. Hybrid Embeddings: Combines image and video features for better accuracy
6. Caching System: NPZ file caching for fast parameter tuning and development

TECHNICAL APPROACH:
- Stage 1: Within-clip clustering (strict thresholds) - separates different people in same scene
- Stage 2: Cross-clip merging (adaptive thresholds) - matches same person across scenes
- Adaptive thresholds based on scene characteristics (crowding, tracking quality, face visibility)
- Physical proximity detection to resolve ambiguous clustering cases

INPUT:
- Multiple video files (MP4 format)
- Each video represents a different scene/time period

OUTPUT:
- Global identity assignments for each detected person
- CSV and JSON files with detailed results
- Tracklet-to-global-ID mappings

USAGE:
    python video_id_detector2_optimized.py

AUTHOR: Advanced Computer Vision System
VERSION: 2.0 - Optimized with Physical Proximity Logic
"""

# ======================
# POST-PROCESSING FUNCTIONS FOR SINGLE PERSON DETECTION
# ======================

def filter_by_size(boxes, max_width=800, max_height=800):
    """Filter out bounding boxes that are too large (likely contain multiple people)"""
    filtered_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        width = x2 - x1
        height = y2 - y1
        
        if width <= max_width and height <= max_height:
            filtered_boxes.append(box)
    return filtered_boxes

def filter_by_aspect_ratio(boxes, min_ratio=0.5, max_ratio=3.0):
    """Filter out bounding boxes with extreme aspect ratios (likely multiple people)"""
    filtered_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        width = x2 - x1
        height = y2 - y1
        ratio = width / height if height > 0 else 0
        
        if min_ratio <= ratio <= max_ratio:
            filtered_boxes.append(box)
    return filtered_boxes

def filter_by_area(boxes, max_area=400000):
    """Filter out bounding boxes that are too large in area (likely multiple people)"""
    filtered_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        area = (x2 - x1) * (y2 - y1)
        
        if area <= max_area:
            filtered_boxes.append(box)
    return filtered_boxes

def analyze_density(boxes, frame):
    """Analyze edge density to detect wide back views with multiple people"""
    filtered_boxes = []
    
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Extract crop
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
            
        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Calculate edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Check if it's a wide back view with high edge density (multiple people)
        width = x2 - x1
        height = y2 - y1
        is_wide = width > height * 2.0  # Very wide aspect ratio (more restrictive)
        is_back_view = edge_density > 0.25  # Very high edge density (more restrictive)
        
        # Only filter if it's both very wide AND has very high edge density
        # This is very conservative - only filters obvious multi-person cases
        if is_wide and is_back_view:
            print(f"    Filtered very wide back view with very high edge density: {edge_density:.3f}")
            continue
            
        filtered_boxes.append(box)
    
    return filtered_boxes

def ensure_single_person_detections(boxes, frame):
    """Post-process detections to ensure one person per bounding box"""
    if len(boxes) == 0:
        return boxes
    
    # Apply size-based filtering (more lenient)
    boxes = filter_by_size(boxes, max_width=800, max_height=800)
    
    # Apply aspect ratio filtering (more lenient)
    boxes = filter_by_aspect_ratio(boxes, min_ratio=0.4, max_ratio=4.0)
    
    # Apply area-based filtering (more lenient)
    boxes = filter_by_area(boxes, max_area=400000)
    
    # Apply density analysis for wide back views
    boxes = analyze_density(boxes, frame)
    
    return boxes

# ======================
# CONFIGURATION
# ======================

# Video Input Configuration
# Directory containing input video files
VIDEO_DIR = "./videos/"
# List of video files to process (each represents a different scene/time period)
VIDEO_FILES = ["1_upscaled.mp4", "2_upscaled.mp4", "3_upscaled.mp4", "4_upscaled.mp4"]
# Full paths to video files
VIDEO_PATHS = [os.path.join(VIDEO_DIR, f) for f in VIDEO_FILES]

# Model Configuration
# YOLO model for person detection and tracking
YOLO_WEIGHTS = "yolov8n.pt"
# Primary ReID model (OSNet with Instance Batch Normalization)
REID_MODEL = "osnet_ibn_x1_0"
# Enable Vision Transformer-based ReID for enhanced feature extraction
USE_VIT_REID = True
# TransReID model architecture
VIT_MODEL = "vit_base_patch16_224"
# Face recognition model for additional features
FACE_MODEL = "buffalo_l"
# Force CPU usage to avoid macOS MPS/CUDA compatibility issues
DEVICE = "cpu"

# Ensemble Configuration
# Enable ensemble of multiple ReID models for improved accuracy
USE_ENSEMBLE_REID = True
# Weight distribution: [OSNet, TransReID] - OSNet gets higher weight
ENSEMBLE_WEIGHTS = [0.7, 0.3]

# Feature Weight Configuration
# These weights determine the importance of different features in distance calculation
FACE_WEIGHT = 0.2              # Face recognition features (20%)
MOTION_WEIGHT = 0.05           # Motion/temporal features (5% - reduced for less ambiguity)
POSE_WEIGHT = 0.0              # Pose estimation features (0% - removed, adds ambiguity)
# Body features get remaining weight: 1 - 0.2 - 0.05 - 0.0 = 0.75 (75%)

# Advanced Features Configuration
# Camera bias correction (disabled for cross-clip matching)
USE_CAMERA_BIAS = True
# k-Reciprocal re-ranking for improved matching accuracy
USE_RERANKING = True
K_RECIPROCAL = 25              # Number of reciprocal neighbors for re-ranking
LAMBDA_VALUE = 0.5             # Lambda parameter for re-ranking
# Pose estimation features for additional person characterization
USE_POSE_FEATURES = True
POSE_CONFIDENCE = 0.5          # Minimum confidence threshold for pose features
# Spatial-temporal context matching (currently disabled)
# USE_SPATIAL_TEMPORAL_CONTEXT = True


# ======================
# CLUSTERING CONFIGURATION
# ======================

# Physical Proximity Logic
# Master flag for advanced clustering with physical proximity detection
USE_ADVANCED_CLUSTERING_LOGIC = True
# DISABLED: Physical proximity tolerance and spatial threshold (commented out to prevent over-merging)
# PROXIMITY_PATTERN_TOLERANCE = 0.15  # Tolerance for physical proximity patterns
# SPATIAL_PROXIMITY_THRESHOLD = 0.3   # Threshold for cross-clip spatial proximity detection

# Robust Weight System for Ambiguous Cases
# Enhanced weights that prioritize clothing/appearance features for difficult cases
USE_ROBUST_AMBIGUOUS_WEIGHTS = False  # Currently disabled
AMBIGUOUS_DISTANCE_THRESHOLD = 0.5   # Distance threshold to trigger robust weights
ROBUST_BODY_WEIGHT = 0.8            # Higher body weight for ambiguous cases (clothing/appearance)
ROBUST_FACE_WEIGHT = 0.1            # Lower face weight for ambiguous cases
ROBUST_POSE_WEIGHT = 0.05           # Lower pose weight for ambiguous cases
ROBUST_MOTION_WEIGHT = 0.05         # Lower motion weight for ambiguous cases

# Physical proximity logic removed - only temporal overlaps within clips are needed

# Output Control
# Set to True for detailed debugging output during development

# Alternative Clustering Approaches
# Clothing/appearance focused approach (currently disabled)
USE_CLOTHING_FOCUSED_APPROACH = False  # Use clothing/appearance weights for ALL comparisons
CLOTHING_BODY_WEIGHT = 0.8            # Body weight for clothing-focused approach
CLOTHING_FACE_WEIGHT = 0.1            # Face weight for clothing-focused approach
CLOTHING_POSE_WEIGHT = 0.05           # Pose weight for clothing-focused approach
CLOTHING_MOTION_WEIGHT = 0.05         # Motion weight for clothing-focused approach

# Video-Level ReID Configuration
# Enable video sequence processing for temporal consistency
USE_VIDEO_REID = False
VIDEO_SEQUENCE_LENGTH = 4             # Number of frames per video clip
VIDEO_SEQUENCE_STRIDE = 2             # Overlap between clips (50% overlap)
# Video embeddings removed - not used in current system

# View Invariance Features
# Test-time augmentation for improved robustness across different viewing angles
USE_TTA = True                        # Horizontal flip augmentation
USE_TEMPORAL_SMOOTHING = True         # Smooth embeddings within tracks for stability
SMOOTHING_WINDOW = 5                  # Number of frames to smooth over

# ======================
# CLUSTERING ALGORITHM CONFIGURATION
# ======================

# Clustering Method Selection
CLUSTERING_METHOD = "ADAPTIVE"        # Options: "SIMILARITY" or "DBSCAN"
SIMILARITY_THRESHOLD = 0.8            # High threshold for Euclidean distance
DBSCAN_EPS = 0.35                     # DBSCAN epsilon parameter
DBSCAN_MIN_SAMPLES = 1                # Minimum samples for DBSCAN clusters

# Distance Metrics
DISTANCE_METRIC = "cosine"            # Primary distance metric
USE_CHAMFER_DISTANCE = False          # Alternative distance metric for cross-clip matching

# Adaptive Clustering System
# Two-stage approach: strict within-clip, lenient cross-clip
USE_ADAPTIVE_CLUSTERING = True

# Stage 1: Within-clip clustering (ADAPTIVE - handle back-to-back views)
# Back-to-back views: same person facing different directions
# - Front view: has face, different pose
# - Back view: no face, different body appearance
# - Need to focus on body features, not faces
WITHIN_CLIP_THRESHOLD = 0.3  # Moderate: handle overlaps and short frames of same person within clip
WITHIN_CLIP_FACE_WEIGHT = 0.6  # High face weight - faces are reliable within same clip

# Stage 2: Cross-clip merging (ENABLED - compare clustered embeddings, not tracklets)
CROSS_CLIP_THRESHOLD = 0.9  # Very conservative: only merge extremely similar clusters across clips
CROSS_CLIP_FACE_WEIGHT = 0.4  # Higher face weight - faces more reliable when available

# Minimum similarity requirements for cross-clip merging (very conservative)
CROSS_CLIP_MIN_BODY_SIMILARITY = 0.5   # Very conservative: require excellent body similarity
CROSS_CLIP_MIN_FACE_SIMILARITY = 0.4   # Very conservative: require excellent face similarity
CROSS_CLIP_MIN_POSE_SIMILARITY = 0.0   # Pose not used (set to 0)
CROSS_CLIP_MIN_MOTION_SIMILARITY = 0.0  # Motion not used (set to 0)

# Adaptive weights based on face availability and view type
# Front view (has face): Body 0.5, Face 0.5, Motion 0.0, Pose 0.0
# Back view (no face): Body 1.0, Face 0.0, Motion 0.0, Pose 0.0
# Mixed (one face): Body 0.8, Face 0.2, Motion 0.0, Pose 0.0

# Duplicate removed - using the smart face logic above

# Per-clip adaptive thresholds (AUTOMATIC ANALYSIS)
USE_PER_CLIP_THRESHOLDS = False  # Disable automatic per-clip analysis - use manual WITHIN_CLIP_THRESHOLD
PER_CLIP_THRESHOLDS = {
    # Leave empty for full automatic analysis
    # Or specify manual overrides for specific clips:
    # 0: 0.20,  # Uncomment to manually override clip 0
    # 3: 0.32,  # Uncomment to manually override clip 3
}
# Performance
BATCH_SIZE = 8

# Outputs
OUTPUT_DIR = "./outputs_v3/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dimensions
FACE_DIM = 512
OSNET_DIM = 512
TRANSREID_DIM = 768
RESNET_DIM = 2048  # ResNet50 output dimension
BODY_DIM = OSNET_DIM + TRANSREID_DIM if USE_ENSEMBLE_REID else OSNET_DIM  # 1280 or 512
MOTION_DIM = 5
POSE_DIM = 66

print("="*60)
print("ADVANCED REID SYSTEM v3 - VIEW INVARIANT")
print("="*60)
print(f"Ensemble: {USE_ENSEMBLE_REID} (OSNet + ResNet50) {ENSEMBLE_WEIGHTS if USE_ENSEMBLE_REID else ''}")
print(f"Body dim: {BODY_DIM}")
print(f"TTA (flip augmentation): {USE_TTA}")
print(f"Temporal smoothing: {USE_TEMPORAL_SMOOTHING}")
print(f"Face weight: {FACE_WEIGHT}")
print(f"Body weight: {1 - FACE_WEIGHT - MOTION_WEIGHT - POSE_WEIGHT:.2f}")
print("APPROACH: Standard Adaptive Clustering (Physical Proximity Chain DISABLED)")
print("="*60 + "\n")

# ======================
# LOAD MODELS
# ======================

print("Loading models...")
yolo = YOLO(YOLO_WEIGHTS)

# Load IBN-OSNet
print(f"  • IBN-OSNet ({REID_MODEL})...")
try:
    osnet_extractor = torchreid.models.build_model(
        name=REID_MODEL,  # osnet_ibn_x1_0
        num_classes=751,  # Market-1501 classes
        pretrained=True,
        loss='softmax'
    )
    osnet_extractor.eval()
    osnet_extractor.to(DEVICE)
    print("    IBN-OSNet loaded!")
except Exception as e:
    print(f"    Failed to load IBN-OSNet: {e}")
    print("    Falling back to standard OSNet...")
    osnet_extractor = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=751,
        pretrained=True
    )
    osnet_extractor.eval()
    osnet_extractor.to(DEVICE)

# Load TransReID model (replacing ResNet50)
transreid_extractor = None
if USE_ENSEMBLE_REID:
    print("  • TransReID (Transformer-based ReID)...")
    try:
        # Add TransReID path
        import sys
        sys.path.insert(0, '/Users/liatparker/vscode_agentic/mp4_id_detector/TransReID')
        # Config already imported at top, no need to re-import
        from model.make_model import make_model
        
        cfg = cfg_default.clone()
        cfg.merge_from_file('/Users/liatparker/vscode_agentic/mp4_id_detector/TransReID/configs/Market/vit_base.yml')
        cfg.MODEL.DEVICE = DEVICE
        cfg.MODEL.PRETRAIN_PATH = '/Users/liatparker/.cache/torch/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth'
        cfg.freeze()
        # Build TransReID model
        transreid_extractor = make_model(
            cfg, 
            num_class=751,      # Market-1501 has 751 identities
            camera_num=0,       # Not using camera info
            view_num=0          # Not using view info
        )
        transreid_extractor.eval()
        transreid_extractor.to(DEVICE)
        
        # Update embedding dimension
        TRANSREID_DIM = 768  # ViT-Base outputs 768-dim features
        BODY_DIM = OSNET_DIM + TRANSREID_DIM  # 512 + 768 = 1280
        
        print(f"    TransReID loaded! Embedding dim: {TRANSREID_DIM}")
    except Exception as e:
        print(f"    TransReID failed: {e}")
        import traceback
        traceback.print_exc()
        print("    Falling back to OSNet only...")
        USE_ENSEMBLE_REID = False
        BODY_DIM = OSNET_DIM

body_extractors = {
    'osnet': osnet_extractor,
    'transreid': transreid_extractor  # Now actually TransReID!
}



# Load face model
face_model = insightface.app.FaceAnalysis(name=FACE_MODEL)
ctx_id = 0 if DEVICE == "cuda" else -1
face_model.prepare(ctx_id=ctx_id)

# Load pose detector
if USE_POSE_FEATURES:
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=POSE_CONFIDENCE
    )




else:
    pose_detector = None

print("All models loaded!\n")

# STEP C: DEFINE the video model class FIRST
# ======================
# VIDEO REID MODEL
# ======================

# Add VID-Trans-ReID to path
import sys
sys.path.append('./VID-Trans-ReID')
from VID_Trans_model import VID_Trans

# Initialize video model variable
video_reid_model = None

# Load Video ReID Model
if USE_VIDEO_REID:
    print(f"  • VID-Trans-ReID Model...")
    try:
        # Load pretrained VID-Trans-ReID
        video_reid_model = VID_Trans(
            num_classes=625,  # MARS dataset classes
            camera_num=6,     # Number of cameras in MARS
            pretrainpath='./VID-Trans-ReID/jx_vit_base_p16_224-80ecf9dd.pth'  # ImageNet pretrained ViT
        )
        video_reid_model.eval()
        video_reid_model.to(DEVICE)
        print("    VID-Trans-ReID loaded with pretrained weights!")
    except Exception as e:
        print(f"    Failed to load VID-Trans-ReID: {e}")
        print("    Falling back to image-only mode")
        USE_VIDEO_REID = False


print(f"Video ReID enabled: {USE_VIDEO_REID}")
if USE_VIDEO_REID:
    print(f"   Sequence length: {VIDEO_SEQUENCE_LENGTH} frames")
    print(f"   Video weight: {VIDEO_WEIGHT:.0%}\n")
else:
    print()


def preprocess_crop(crop):
    """
    Preprocess person crop image using CLAHE and sharpening.
    
    This function enhances person crop images using Contrast Limited Adaptive
    Histogram Equalization (CLAHE) and sharpening to improve feature extraction
    quality for ReID models.
    
    Args:
        crop (numpy.ndarray): Raw person crop image (BGR format)
        
    Returns:
        numpy.ndarray: Enhanced crop image with improved contrast and sharpness
        
    Processing Steps:
        1. Validates input crop (not None and not empty)
        2. Converts BGR to LAB color space
        3. Applies CLAHE to L channel for adaptive contrast enhancement
        4. Applies sharpening filter to enhance edges
        5. Converts back to BGR color space
    """
    if crop is None or crop.size == 0:
        return crop
    try:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        return enhanced
    except:
        return crop

def extract_pose_features(crop):
    """
    Extract pose keypoints from person crop image.
    
    This function uses MediaPipe to detect human pose landmarks and extracts
    a normalized 66-dimensional feature vector representing the person's pose.
    Pose features help distinguish people based on their body posture and stance.
    
    Args:
        crop (numpy.ndarray): Person crop image (BGR format)
        
    Returns:
        numpy.ndarray: 66-dimensional pose feature vector (normalized)
                      Returns zero vector if pose detection fails or is disabled
                      
    Features:
        - Uses MediaPipe Pose for robust pose detection
        - Extracts 33 landmark points (x,y coordinates = 66D vector)
        - Normalizes features to unit length for consistent scaling
        - Graceful fallback to zero vector on detection failure
    """
    if not USE_POSE_FEATURES or pose_detector is None:
        return np.zeros(POSE_DIM, dtype=np.float32)
    try:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(rgb)
        if results.pose_landmarks:
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])
            pose_vec = np.array(landmarks, dtype=np.float32)
            pose_vec = pose_vec / (np.linalg.norm(pose_vec) + 1e-8)
            return pose_vec
        return np.zeros(POSE_DIM, dtype=np.float32)
    except:
        return np.zeros(POSE_DIM, dtype=np.float32)

def extract_face_batch(crops):
    """
    Extract face embeddings for a batch of person crops.
    
    This function processes multiple person crop images and extracts face
    embeddings using InsightFace. Face embeddings are crucial for person
    re-identification as they provide strong discriminative features.
    
    Args:
        crops (list): List of person crop images (numpy arrays)
        
    Returns:
        tuple: (embeddings, has_faces)
            - embeddings (list): List of 512-dimensional face embedding vectors
            - has_faces (list): Boolean list indicating which crops have detected faces
            
    Features:
        - Uses InsightFace for robust face detection and embedding extraction
        - Normalizes embeddings to unit length for consistent distance calculation
        - Handles cases where no face is detected (zero embedding)
        - Batch processing for efficiency
    """
    embeddings = []
    has_faces = []
    for crop in crops:
        try:
            faces = face_model.get(crop)
            if faces:
                emb = faces[0].embedding.astype(np.float32)
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                embeddings.append(emb)
                has_faces.append(True)
            else:
                embeddings.append(np.zeros(FACE_DIM, dtype=np.float32))
                has_faces.append(False)
        except:
            embeddings.append(np.zeros(FACE_DIM, dtype=np.float32))
            has_faces.append(False)
    return np.array(embeddings, dtype=np.float32), np.array(has_faces)
def extract_osnet_with_tta(crop):
    """Extract OSNet embedding with Test-Time Augmentation"""
    try:
        # Preprocess for OSNet: 256x128 (H x W)
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 256))
        img = img.astype(np.float32) / 255.0
        
        # Normalize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # Convert to tensor: (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        
        # Extract OSNet embedding
        with torch.no_grad():
            osnet_emb = body_extractors['osnet'](img_tensor)
            osnet_emb = osnet_emb.cpu().numpy().flatten()
        
        # TTA: horizontal flip
        if USE_TTA:
            img_flipped = np.flip(img, axis=1).copy()  # Flip horizontally
            img_flipped_tensor = torch.from_numpy(img_flipped).permute(2, 0, 1).float()
            img_flipped_tensor = img_flipped_tensor.unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                osnet_emb_flipped = body_extractors['osnet'](img_flipped_tensor)
                osnet_emb_flipped = osnet_emb_flipped.cpu().numpy().flatten()
            
            # Average original + flipped
            osnet_emb = (osnet_emb + osnet_emb_flipped) / 2.0
        
        # Normalize
        osnet_emb = osnet_emb / (np.linalg.norm(osnet_emb) + 1e-8)
        return osnet_emb
        
    except Exception as e:
        print(f"OSNet extraction failed: {e}")
        return np.zeros(OSNET_DIM, dtype=np.float32)


    # ... your existing code ...
def extract_body_batch(crops):
    """Extract body embeddings using IBN-OSNet + TransReID ensemble with TTA"""
    batch_embeddings = []
    
    for crop in crops:
        # IBN-OSNet embedding (now using appearance-invariant model)
        osnet_emb = extract_osnet_with_tta(crop)  # Now uses IBN-Net
        
        # TransReID embedding
        if USE_ENSEMBLE_REID and body_extractors['transreid'] is not None:
            # TransReID preprocessing
            transreid_img = cv2.resize(crop, (128, 256))  # TransReID size
            transreid_img = cv2.cvtColor(transreid_img, cv2.COLOR_BGR2RGB)
            transreid_img = transreid_img.astype(np.float32) / 255.0
            
            # Normalize (ImageNet stats)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            transreid_img = (transreid_img - mean) / std
            
            # Convert to tensor: (H, W, C) -> (C, H, W)
            transreid_tensor = torch.from_numpy(transreid_img).permute(2, 0, 1).float()
            transreid_tensor = transreid_tensor.unsqueeze(0).to(DEVICE)
            
            # Extract TransReID embedding
            with torch.no_grad():
                transreid_emb = body_extractors['transreid'](transreid_tensor)
                transreid_emb = transreid_emb.cpu().numpy().flatten()
            
            # NEW: Combine with IBN-heavy weights
            combined_emb = np.concatenate([
                osnet_emb * 0.7,  # IBN-Net weight increased
                transreid_emb * 0.3  # TransReID weight decreased
            ])
            batch_embeddings.append(combined_emb)
        else:
            batch_embeddings.append(osnet_emb)
    
    return batch_embeddings
def smooth_track_embeddings(embeddings_list, window_size=5):
    """
    Apply temporal smoothing to track embeddings using exponential moving average.
    
    This function reduces noise and jitter in person embeddings across frames
    within a track, making the person representation more stable and consistent.
    Temporal smoothing is crucial for robust person re-identification.
    
    Args:
        embeddings_list (list): List of embedding vectors from consecutive frames
        window_size (int): Smoothing window size (default: 5)
        
    Returns:
        list: List of smoothed embedding vectors
        
    Algorithm:
        - Uses exponential moving average (EMA) for temporal smoothing
        - Alpha parameter calculated as 2/(window_size + 1)
        - Each embedding is smoothed with previous smoothed embedding
        - Reduces frame-to-frame variations while preserving person identity
        
    Benefits:
        - Reduces noise from detection/tracking errors
        - Improves consistency across frames
        - Better clustering performance
        - More stable person representations
    """
    if len(embeddings_list) <= 1:
        return embeddings_list
    
    smoothed = []
    alpha = 2.0 / (window_size + 1)
    
    # Initialize with first embedding
    ema = embeddings_list[0].copy()
    smoothed.append(ema)
    
    # Apply EMA
    for emb in embeddings_list[1:]:
        ema = alpha * emb + (1 - alpha) * ema
        ema = ema / (np.linalg.norm(ema) + 1e-8)
        smoothed.append(ema)
    
    return smoothed
def compute_frame_quality(bbox_conf, face_detected, pose_conf):
    """
    Compute quality score for a frame based on detection confidence.
    Higher quality = more reliable for ReID embedding.
    
    Priority: Face > Pose > BBox
    """
    # Normalize inputs
    face_score = 1.0 if face_detected else 0.0
    pose_score = min(pose_conf, 1.0) if pose_conf > 0 else 0.0
    bbox_score = min(bbox_conf, 1.0) if bbox_conf > 0 else 0.5  # Default 0.5 if missing
    
    # Weighted combination (face is most important for view consistency)
    quality = (
        face_score * 0.6 +       # Face presence (binary)
        pose_score * 0.3 +       # Pose confidence
        bbox_score * 0.1         # Detection confidence
    )
    return quality

def weighted_average_embeddings(embeddings_list, qualities):
    """
    Compute weighted average of embeddings based on quality scores.
    High-quality frames (with faces) get more weight.
    """
    if len(embeddings_list) == 0:
        return None
    if len(embeddings_list) == 1:
        return embeddings_list[0]
    
    # Convert to numpy
    embeddings = np.array(embeddings_list)
    qualities = np.array(qualities)
    
    # Softmax weighting (sharpen distribution to focus on best frames)
    weights = np.exp(qualities * 3.0)  # Temperature = 3.0 (higher = sharper)
    weights = weights / (weights.sum() + 1e-8)
    
    # Weighted average
    weighted_emb = (embeddings * weights[:, np.newaxis]).sum(axis=0)
    
    # Normalize
    weighted_emb = weighted_emb / (np.linalg.norm(weighted_emb) + 1e-8)
    
    return weighted_emb

def calculate_bbox_proximity(bbox1, bbox2):
    """
    Calculate spatial proximity between two bounding boxes.
    Returns a value between 0 and 1, where 1 means perfect overlap.
    
    Args:
        bbox1: [x1, y1, x2, y2] - first bounding box
        bbox2: [x1, y1, x2, y2] - second bounding box
    
    Returns:
        float: proximity score (0-1)
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Check if there's an intersection
    if x1_i >= x2_i or y1_i >= y2_i:
        # No intersection - calculate distance-based proximity
        # Calculate center points
        cx1 = (x1_1 + x2_1) / 2
        cy1 = (y1_1 + y2_1) / 2
        cx2 = (x1_2 + x2_2) / 2
        cy2 = (y1_2 + y2_2) / 2
        
        # Calculate distance between centers
        distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        
        # Calculate average size for normalization
        size1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        size2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        avg_size = np.sqrt((size1 + size2) / 2)
        
        # Convert distance to proximity (closer = higher proximity)
        proximity = max(0, 1 - (distance / (avg_size * 2)))  # Normalize by 2x average size
        return proximity
    else:
        # There's an intersection - calculate IoU
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        if union_area > 0:
            iou = intersection_area / union_area
            return iou
        else:
            return 0.0

def chamfer_distance(frames1, frames2, metric='cosine'):
    """
    Compute normalized Chamfer distance between two sets of frame embeddings.
    Uses bidirectional matching and normalizes to [0,1] range.
    """
    if len(frames1) == 0 or len(frames2) == 0:
        return 1.0  # Max distance
    
    # Compute pairwise distances [N, M]
    dist_matrix = cdist(frames1, frames2, metric=metric)
    
    # For each frame in set1, find closest in set2
    min_dist_1to2 = dist_matrix.min(axis=1)  # [N]
    avg_1to2 = min_dist_1to2.mean()
    
    # For each frame in set2, find closest in set1
    min_dist_2to1 = dist_matrix.min(axis=0)  # [M]
    avg_2to1 = min_dist_2to1.mean()
    
    # Chamfer distance is average of both directions
    chamfer_dist = (avg_1to2 + avg_2to1) / 2
    
    # NEW: Normalize to [0,1] range
    #normalized_dist = chamfer_dist / (1 + chamfer_dist)
    
    return chamfer_dist

def select_representative_frames(frame_embeddings, qualities, max_frames=15):
    """
    Select most representative frames from a tracklet.
    
    Strategy:
    1. Always include top-quality frames (with faces)
    2. Add diverse frames to cover different views
    3. Limit to max_frames for efficiency
    
    Args:
        frame_embeddings: List of [D] embeddings
        qualities: List of quality scores
        max_frames: Maximum frames to keep
    
    Returns:
        Selected frame embeddings as [K, D] array
    """
    if len(frame_embeddings) <= max_frames:
        return np.array(frame_embeddings)
    
    frame_embeddings = np.array(frame_embeddings)
    qualities = np.array(qualities)
    n = len(frame_embeddings)
    
    # Step 1: Select top quality frames (at least 5 or 30%)
    num_top = max(5, int(n * 0.3))
    top_indices = np.argsort(qualities)[-num_top:][::-1]  # Highest quality first
    
    selected_indices = set(top_indices.tolist())
    
    # Step 2: Add diverse frames (maximize coverage of appearance space)
    # Use farthest-point sampling to get diverse views
    remaining_budget = max_frames - len(selected_indices)
    if remaining_budget > 0:
        available = [i for i in range(n) if i not in selected_indices]
        
        if available:
            # Start with a random frame from available
            current = available[0]
            selected_indices.add(current)
            remaining_budget -= 1
            
            # Greedily add frames that are farthest from selected set
            while remaining_budget > 0 and len(available) > 1:
                available = [i for i in available if i not in selected_indices]
                if not available:
                    break
                
                selected_embs = frame_embeddings[list(selected_indices)]
                
                # Find frame farthest from selected set
                max_min_dist = -1
                farthest_idx = None
                
                for idx in available:
                    distances = cdist([frame_embeddings[idx]], selected_embs, metric='cosine')[0]
                    min_dist = distances.min()  # Distance to nearest selected frame
                    
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        farthest_idx = idx
                
                if farthest_idx is not None:
                    selected_indices.add(farthest_idx)
                    remaining_budget -= 1
                else:
                    break
    
    # Return selected frames
    selected_list = sorted(list(selected_indices))
    return frame_embeddings[selected_list]


# STEP E: Video ReID Helper Functions

def group_crops_into_sequences(crops, seq_len=16, stride=8):
    """
    Group frame crops into temporal sequences
    
    Args:
        crops: List of [H, W, 3] numpy arrays
        seq_len: Number of frames per sequence
        stride: Step between sequences
    
    Returns:
        List of sequences, each [seq_len, H, W, 3]
    """
    sequences = []
    
    for start_idx in range(0, len(crops), stride):
        end_idx = start_idx + seq_len
        
        if end_idx > len(crops):
            # Pad last sequence if needed
            seq = crops[start_idx:]
            padding_needed = seq_len - len(seq)
            # Repeat last frame
            seq = seq + [seq[-1]] * padding_needed
        else:
            seq = crops[start_idx:end_idx]
        
        sequences.append(seq)
    
    return sequences

def extract_video_embedding(sequence):
    """Extract video embedding using pretrained VID-Trans-ReID (simplified)"""
    if video_reid_model is None:
        return np.zeros(VIDEO_DIM, dtype=np.float32)
    
    try:
        # VID-Trans-ReID preprocessing
        processed_frames = []
        for crop in sequence:
            # Resize to match other models
            img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 256))  # Consistent with other models
            img = img.astype(np.float32) / 255.0
            
            # Normalize (ImageNet stats)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (img - mean) / std
            
            processed_frames.append(img)
        
        # Stack to tensor: [T, H, W, C] → [1, T, C, H, W]
        sequence_array = np.stack(processed_frames, axis=0)
        sequence_tensor = torch.from_numpy(sequence_array).float()
        sequence_tensor = sequence_tensor.permute(0, 3, 1, 2)  # [T, C, H, W]
        sequence_tensor = sequence_tensor.unsqueeze(0).to(DEVICE)  # [1, T, C, H, W]
        
        # Extract embedding using VID-Trans-ReID (simplified)
        with torch.no_grad():
            # Create camera label tensor (assuming clip 0 for now)
            cam_label = torch.zeros(sequence_tensor.size(0), dtype=torch.long, device=DEVICE)
            outputs = video_reid_model(sequence_tensor, cam_label=cam_label)
            
            # Simple approach: just take the first output
            if isinstance(outputs, (tuple, list)):
                video_emb = outputs[0]  # Take first element
            else:
                video_emb = outputs
            
            # Ensure it's a tensor and flatten
            if hasattr(video_emb, 'cpu'):
                video_emb = video_emb.cpu().numpy().flatten()
            else:
                video_emb = np.array(video_emb).flatten()
        
        # Normalize
        video_emb = video_emb / (np.linalg.norm(video_emb) + 1e-8)
        return video_emb
        
    except Exception as e:
        print(f"Video embedding extraction failed: {e}")
        return np.zeros(VIDEO_DIM, dtype=np.float32)
        


def merge_overlapping_tracks_same_clip(tracklets):
    """
    Merge tracklet fragments that belong to the same person within each video clip.
    
    This function identifies and merges tracklet fragments that are likely from
    the same person but were split due to tracking interruptions or occlusions.
    It uses adaptive temporal overlap checking and embedding similarity.
    
    Args:
        tracklets (list): List of tracklet dictionaries from all video clips
        
    Returns:
        list: List of merged tracklets with combined temporal and embedding information
        
    Algorithm:
        1. Groups tracklets by video clip
        2. For each clip, identifies potential fragment pairs
        3. Applies adaptive temporal overlap checking (prevents merging simultaneous people)
        4. Uses embedding similarity for final merge decision
        5. Combines temporal ranges and averages embeddings for merged tracklets
        
    Key Features:
        - Adaptive overlap threshold based on track length
        - High embedding similarity requirement (0.8) for merging
        - Preserves temporal boundaries to prevent false merges
        - Handles multiple fragments per person
    """
    if not tracklets:
        return []
    
    # Group by clip
    clips = {}
    for t in tracklets:
        cid = t['clip_idx']
        if cid not in clips:
            clips[cid] = []
        clips[cid].append(t)
    
    merged_all = []
    for clip_idx in sorted(clips.keys()):
        clip_tracks = clips[clip_idx]
        
        
        # NO special cases - adaptive temporal overlap check handles everything
        used = set()
        
        for i in range(len(clip_tracks)):
            if i in used:
                continue
            
            current = clip_tracks[i]
            group = [current]
            
            # Find tracks that can be merged (same person fragments)
            for j in range(i + 1, len(clip_tracks)):
                if j in used:
                    continue
                
                candidate = clip_tracks[j]
                
                # Check temporal overlap (adaptive based on track length)
                overlap_start = max(current['start_frame'], candidate['start_frame'])
                overlap_end = min(current['end_frame'], candidate['end_frame'])
                overlap_frames = max(0, overlap_end - overlap_start)
                
                # Calculate overlap ratio relative to shorter track
                min_length = min(current['end_frame'] - current['start_frame'], 
                                 candidate['end_frame'] - candidate['start_frame'])
                overlap_ratio = overlap_frames / min_length if min_length > 0 else 0
                
                if overlap_ratio > 0.5:  # Block if overlap > 50% of shorter track
                    # They appear simultaneously → MUST be different people → DON'T merge
                    continue
                
                # Check gap (only consider if NOT overlapping significantly)
                gap = candidate['start_frame'] - current['end_frame']
                
                if 0 < gap <= 120:  # 3 seconds at 30fps
                    # Check embedding similarity
                    body_sim = np.dot(current['body_emb'], candidate['body_emb'])
                    
                    # Very strict similarity requirement
                    if body_sim > 0.8:
                        group.append(candidate)
                        used.add(j)
                        current = candidate  # Update current to extend the range
            
            # Merge the group by averaging embeddings
            if len(group) > 1:
                # Collect all frames and bboxes from the group
                all_frames = []
                all_bboxes = []
                for t in group:
                    if 'frames' in t and 'bboxes' in t:
                        all_frames.extend(t['frames'])
                        all_bboxes.extend(t['bboxes'])
                
                # Sort by frame number
                if all_frames:
                    sorted_indices = sorted(range(len(all_frames)), key=lambda i: all_frames[i])
                    all_frames = [all_frames[i] for i in sorted_indices]
                    all_bboxes = [all_bboxes[i] for i in sorted_indices]
                
                merged_track = {
                    'clip_idx': clip_idx,
                    'track_id': group[0].get('track_id', -1),
                    'start_frame': min(t['start_frame'] for t in group),
                    'end_frame': max(t['end_frame'] for t in group),
                    'body_emb': np.mean([t['body_emb'] for t in group], axis=0),
                    'has_face': any(t.get('has_face', False) for t in group),
                    'motion_emb': np.mean([t['motion_emb'] for t in group], axis=0),
                    'pose_emb': np.mean([t['pose_emb'] for t in group], axis=0),
                    'frames': all_frames,
                    'bboxes': all_bboxes,
                }
                
                # Normalize body embedding
                merged_track['body_emb'] = merged_track['body_emb'] / (np.linalg.norm(merged_track['body_emb']) + 1e-8)
                
                # Average face embedding if any track has face
                if merged_track['has_face']:
                    face_embs = [t['face_emb'] for t in group if t.get('has_face', False)]
                    if face_embs:
                        merged_track['face_emb'] = np.mean(face_embs, axis=0)
                        merged_track['face_emb'] = merged_track['face_emb'] / (np.linalg.norm(merged_track['face_emb']) + 1e-8)
                    else:
                        merged_track['face_emb'] = np.zeros(512, dtype=np.float32)
                else:
                    merged_track['face_emb'] = np.zeros(512, dtype=np.float32)
                
                merged_all.append(merged_track)
            else:
                # Single tracklet, keep as is
                merged_all.append(current)
    
    print(f"  Merge complete: {len(tracklets)} → {len(merged_all)} tracklets")
    return merged_all
def extract_pose_batch(crops):
    """Extract pose features for batch"""
    return np.array([extract_pose_features(c) for c in crops], dtype=np.float32)

def extract_motion_features(track_data):
    """Extract motion features from track (5D vector)"""
    if len(track_data) == 0:
        return np.zeros(MOTION_DIM, dtype=np.float32)
    
    movements = [d.get('movement', 0) for d in track_data]
    sizes = [d.get('bbox_size', 0.5) for d in track_data]
    v_positions = [d.get('v_pos', 0.5) for d in track_data]
    h_positions = [d.get('h_pos', 0.5) for d in track_data]
    
    features = np.array([
        np.mean(movements),
        np.std(movements) if len(movements) > 1 else 0.0,
        np.mean(sizes),
        np.mean(v_positions),
        np.mean(h_positions)
    ], dtype=np.float32)
    
    return features / (np.linalg.norm(features) + 1e-8)


# ======================
# VIDEO PROCESSING
# ======================

def process_video(video_path, clip_idx):
    """
    Process a single video file and extract person tracklets with embeddings.
    
    This is the main video processing function that handles the complete pipeline
    from video input to tracklet extraction with all necessary embeddings.
    
    Args:
        video_path (str): Path to the input video file
        clip_idx (int): Index of the video clip (for identification)
        
    Returns:
        list: List of tracklet dictionaries, each containing:
            - 'clip_idx': Video clip index
            - 'track_id': YOLO track ID
            - 'start_frame', 'end_frame': Temporal boundaries
            - 'body_emb': Body embedding vector
            - 'face_emb': Face embedding vector (if available)
            - 'bboxes': List of bounding boxes
            - 'body_frames': Representative body frame embeddings
            - 'face_frames': Representative face frame embeddings (if available)
            
    Processing Pipeline:
        1. Video metadata extraction (FPS, frame count)
        2. YOLO person detection and tracking
        3. Tracklet extraction and validation
        4. Multi-frame embedding extraction (body, face, pose)
        5. Temporal smoothing of embeddings
        6. Representative frame selection
        7. Quality assessment and filtering
    """
    print(f"\nProcessing: {os.path.basename(video_path)} (Clip {clip_idx})")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Detect and track
    results_generator = yolo.track(
        source=video_path,
        classes=[0],
        conf = 0.15,
        stream=True,
        persist=True,
        iou=0.35,
        tracker="bytetrack_tuned.yaml",  # Use tuned config (was working before!)    ,
        verbose=False
    )
    
    # Storage
    tracks_data = {}
    frame_idx = 0
    
    for result in tqdm(results_generator, total=total_frames, desc=f"  Tracking"):
        if result.boxes is None or len(result.boxes) == 0:
            frame_idx += 1
            continue
        
        frame = result.orig_img
        h, w = frame.shape[:2]
        
        # Post-processing for single person detection
        if len(result.boxes) > 0:
            original_count = len(result.boxes)
            result.boxes = ensure_single_person_detections(result.boxes, frame)
            filtered_count = len(result.boxes)
            if original_count != filtered_count:
                print(f"    Filtered {original_count - filtered_count} detections (kept {filtered_count}/{original_count})")
        
        for box in result.boxes:
            if box.id is None:
                continue
            
            track_id = int(box.id.item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Extract crop
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            # Preprocess
            crop = preprocess_crop(crop)
            
            # Initialize track
            if track_id not in tracks_data:
                tracks_data[track_id] = {
                    'crops': [],
                    'frames': [],
                    'bboxes': [],
                    'movements': []
                }
            
            # Store
            tracks_data[track_id]['crops'].append(crop)
            tracks_data[track_id]['frames'].append(frame_idx)
            tracks_data[track_id]['bboxes'].append([x1, y1, x2, y2])
            
            # Motion features
            bbox_size = ((x2 - x1) * (y2 - y1)) / (w * h)
            v_pos = (y1 + y2) / (2 * h)
            h_pos = (x1 + x2) / (2 * w)
            
            if len(tracks_data[track_id]['bboxes']) > 1:
                prev_bbox = tracks_data[track_id]['bboxes'][-2]
                dx = ((x1 + x2) / 2 - (prev_bbox[0] + prev_bbox[2]) / 2) / w
                dy = ((y1 + y2) / 2 - (prev_bbox[1] + prev_bbox[3]) / 2) / h
                movement = np.sqrt(dx**2 + dy**2)
            else:
                movement = 0.0
            
            tracks_data[track_id]['movements'].append({
                'movement': movement,
                'bbox_size': bbox_size,
                'v_pos': v_pos,
                'h_pos': h_pos
            })
        
        frame_idx += 1
    
    # Extract embeddings
    print(f"  Extracting embeddings for {len(tracks_data)} tracks...")
    tracklets = []
    
    for track_id, data in tracks_data.items():
        if len(data['crops']) < 3:
            continue
        
        crops = data['crops']
        
        # Extract embeddings in batches
        body_embs_list = []
        face_embs_list = []
        pose_embs_list = []
        has_faces_list = []
        qualities_list = []  # NEW: Track frame quality
        bbox_confs_list = []  # NEW: Store bbox confidences
        
        # Video embeddings removed - not used in current system
        
        for i in range(0, len(crops), BATCH_SIZE):
            batch = crops[i:i+BATCH_SIZE]
            
            body_batch = extract_body_batch(batch)
            face_batch, has_face_batch = extract_face_batch(batch)
            pose_batch = extract_pose_batch(batch)
            
            body_embs_list.extend(body_batch)
            face_embs_list.extend(face_batch)
            pose_embs_list.extend(pose_batch)
            has_faces_list.extend(has_face_batch)
            
            # NEW: Compute quality for each frame in batch
            for j in range(len(batch)):
                crop_idx = i + j
                # Get bbox confidence (default to 0.7 if not available)
                bbox_conf = 0.7  # YOLO doesn't expose confidence in track data, use default
                
                # Compute quality score
                face_detected = has_face_batch[j]
                pose_detected = (pose_batch[j].sum() > 0.01)  # Check if pose features exist
                pose_conf = 0.8 if pose_detected else 0.0
                
                quality = compute_frame_quality(bbox_conf, face_detected, pose_conf)
                qualities_list.append(quality)
        
        # Apply temporal smoothing
        if USE_TEMPORAL_SMOOTHING and len(body_embs_list) > 1:
            body_embs_list = smooth_track_embeddings(body_embs_list, SMOOTHING_WINDOW)
        if USE_TEMPORAL_SMOOTHING and len(face_embs_list) > 1:
            face_embs_list = smooth_track_embeddings(face_embs_list, SMOOTHING_WINDOW)
        
        # NEW: Use quality-weighted aggregation for body embeddings
        avg_body = weighted_average_embeddings(body_embs_list, qualities_list)
        
        # NEW: Face - weighted average among frames with faces
        face_indices = [i for i in range(len(face_embs_list)) if has_faces_list[i]]
        if face_indices:
            face_embs_with_faces = [face_embs_list[i] for i in face_indices]
            face_qualities = [qualities_list[i] for i in face_indices]
            avg_face = weighted_average_embeddings(face_embs_with_faces, face_qualities)
        else:
            avg_face = np.zeros(FACE_DIM, dtype=np.float32)
        
        # Pose - simple average (less critical for cross-clip matching)
        avg_pose = np.mean(pose_embs_list, axis=0).astype(np.float32)
        avg_pose = avg_pose / (np.linalg.norm(avg_pose) + 1e-8)
        
        motion_feat = extract_motion_features(data['movements'])
        
        # Video embeddings removed - not used in current system
        
        # NEW: Select representative frames for multi-frame distance
        representative_body_frames = select_representative_frames(
            body_embs_list, qualities_list, max_frames=15
        )
        
        representative_face_frames = None
        if face_indices:
            face_embs_with_faces = [face_embs_list[i] for i in face_indices]
            face_qualities = [qualities_list[i] for i in face_indices]
            representative_face_frames = select_representative_frames(
                face_embs_with_faces, face_qualities, max_frames=10
            )
        
        tracklets.append({
            'clip_idx': clip_idx,
            'track_id': track_id,
            'start_frame': data['frames'][0],
            'end_frame': data['frames'][-1],
            'num_detections': len(data['crops']),
            'body_emb': avg_body,  # Keep averaged for within-clip
            'face_emb': avg_face,
            'pose_emb': avg_pose,
            'motion_emb': motion_feat,
            'has_face': len(face_indices) > 0,
            'frames': data['frames'],
            'bboxes': data['bboxes'],
            # NEW: Multi-frame embeddings for cross-clip matching
            'body_frames': representative_body_frames,  # [K, D] array
            'face_frames': representative_face_frames if representative_face_frames is not None else np.array([])
        })
    
    print(f"  Extracted {len(tracklets)} tracklets")
    return tracklets

# ======================
# DISTANCE & CLUSTERING
# ======================

def compute_distance_matrix(tracklets):
    """Compute pairwise distance matrix (vectorized)"""
    n = len(tracklets)
    
    # Extract embeddings
    body_embs = np.array([t['body_emb'] for t in tracklets])
    face_embs = np.array([t['face_emb'] for t in tracklets])
    pose_embs = np.array([t['pose_emb'] for t in tracklets])
    motion_embs = np.array([t['motion_emb'] for t in tracklets])
    has_faces = np.array([t['has_face'] for t in tracklets])
    
    # Compute distances
    body_dist = cdist(body_embs, body_embs, metric='cosine')
    face_dist = cdist(face_embs, face_embs, metric='cosine')
    pose_dist = cdist(pose_embs, pose_embs, metric='cosine')
    motion_dist = cdist(motion_embs, motion_embs, metric='cosine')
    
    # Hybrid weighting
    body_weight = 1.0 - FACE_WEIGHT - MOTION_WEIGHT - POSE_WEIGHT
    
    dist_matrix = (
        body_weight * body_dist +
        FACE_WEIGHT * face_dist +
        POSE_WEIGHT * pose_dist +
        MOTION_WEIGHT * motion_dist
    )
    
    # Adjust face weight based on detection
    for i in range(n):
        for j in range(n):
            if not has_faces[i] or not has_faces[j]:
                # No face: exclude face weight entirely
                total_weight = body_weight + POSE_WEIGHT + MOTION_WEIGHT
                dist_matrix[i, j] = (
                    body_weight * body_dist[i, j] +
                    POSE_WEIGHT * pose_dist[i, j] +
                    MOTION_WEIGHT * motion_dist[i, j]
                ) / total_weight
    
    return dist_matrix

def k_reciprocal_rerank(dist_matrix):
    """k-Reciprocal re-ranking"""
    print("🔄 Applying k-reciprocal re-ranking...")
    N = dist_matrix.shape[0]
    
    # Find k-reciprocal neighbors
    k_nearest = np.argsort(dist_matrix, axis=1)[:, 1:K_RECIPROCAL+1]
    reciprocal_mask = np.zeros((N, N), dtype=bool)
    
    for i in range(N):
        for j in k_nearest[i]:
            if i in k_nearest[j]:
                reciprocal_mask[i, j] = True
    
    # Jaccard distance
    reranked_dist = np.copy(dist_matrix)
    for i in range(N):
        reciprocal_i = np.where(reciprocal_mask[i])[0]
        if len(reciprocal_i) > 0:
            for j in reciprocal_i:
                reciprocal_j = np.where(reciprocal_mask[j])[0]
                intersection = len(np.intersect1d(reciprocal_i, reciprocal_j))
                union = len(np.union1d(reciprocal_i, reciprocal_j))
                jaccard = intersection / (union + 1e-8)
                reranked_dist[i, j] = (1 - jaccard) * dist_matrix[i, j]
    
    final_dist = (1 - LAMBDA_VALUE) * dist_matrix + LAMBDA_VALUE * reranked_dist
    return final_dist

def cluster_tracklets(dist_matrix):
    """Cluster tracklets into global IDs"""
    print(f"📌 Clustering with {CLUSTERING_METHOD}...")
    
    if CLUSTERING_METHOD.upper() == "SIMILARITY":
        N = dist_matrix.shape[0]
        global_ids = [-1] * N
        next_id = 0
        
        for i in range(N):
            if global_ids[i] != -1:
                continue
            
            best_cluster = -1
            best_dist = float('inf')
            
            for cluster_id in range(next_id):
                members = [j for j in range(i) if global_ids[j] == cluster_id]
                if members:
                    avg_dist = np.mean([dist_matrix[i, j] for j in members])
                    if avg_dist < best_dist:
                        best_dist = avg_dist
                        best_cluster = cluster_id
            
            if best_cluster != -1 and best_dist < SIMILARITY_THRESHOLD:
                global_ids[i] = best_cluster
            else:
                global_ids[i] = next_id
                next_id += 1
        
        return np.array(global_ids)
    
    elif CLUSTERING_METHOD.upper() == "DBSCAN":
        clusterer = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric='precomputed')
        global_ids = clusterer.fit_predict(dist_matrix)
        
        # Remap noise
        noise_mask = (global_ids == -1)
        if noise_mask.any():
            max_id = global_ids.max()
            noise_ids = np.arange(max_id + 1, max_id + 1 + noise_mask.sum())
            global_ids[noise_mask] = noise_ids
        
        return global_ids
    
    else:
        raise ValueError(f"Unknown clustering method: {CLUSTERING_METHOD}")

def analyze_clip_characteristics(clip_tracks):
    """
    Automatically analyze clip characteristics to determine optimal threshold.
    
    NEW: Uses max simultaneous people detected to guide clustering.
    
    Returns adaptive threshold based on:
    - Max simultaneous people (YOLO detection count)
    - Number of tracklets (crowded vs simple)
    - Average tracklet length (stable tracking vs fragments)
    - Face detection rate (good lighting vs poor)
    - Embedding variance (similar people vs diverse)
    """
    from scipy.spatial.distance import cdist
    
    n = len(clip_tracks)
    
    # NEW: Determine max simultaneous people in this clip
    frame_to_people = {}
    for t in clip_tracks:
        for frame in range(t['start_frame'], t['end_frame'] + 1):
            if frame not in frame_to_people:
                frame_to_people[frame] = 0
            frame_to_people[frame] += 1
    
    max_simultaneous = max(frame_to_people.values()) if frame_to_people else n
    
    # 1. Crowding factor (more tracklets = need more lenient)
    if n <= 3:
        crowding_score = 0.0  # Simple scene
    elif n <= 10:
        crowding_score = 0.02
    else:
        crowding_score = 0.04
    
    # 2. Tracking quality (long tracks = good tracking, short = fragmented)
    avg_length = np.mean([t['end_frame'] - t['start_frame'] for t in clip_tracks])
    if avg_length > 300:  # Long tracks (>10 seconds at 30fps)
        tracking_score = 0.0  # Good tracking
    elif avg_length > 100:
        tracking_score = 0.01
    else:
        tracking_score = 0.03
    
    # 3. Face detection rate (high rate = can use strict, low = need lenient)
    face_rate = np.mean([t['has_face'] for t in clip_tracks])
    if face_rate > 0.7:
        face_score = 0.0  # Good faces, can be strict
    elif face_rate > 0.4:
        face_score = 0.01
    else:
        face_score = 0.02
    
    # 4. Embedding diversity (high variance = diverse people, low = similar)
    body_embs = np.array([t['body_emb'] for t in clip_tracks])
    pairwise_dist = cdist(body_embs, body_embs, metric='cosine')
    avg_dist = np.mean(pairwise_dist[np.triu_indices_from(pairwise_dist, k=1)])
    
    # Calculate traditional diversity score
    if n == 2:
        diversity_score = -0.20
    elif n <= 8:
        if avg_dist > 0.32:
            diversity_score = -0.20
        elif avg_dist > 0.18:
            diversity_score = -0.18
        else:
            diversity_score = -0.15
    elif n > 15:
        if avg_dist > 0.5:
            diversity_score = -0.05
        elif avg_dist > 0.4:
            diversity_score = -0.02
        elif avg_dist > 0.35:
            diversity_score = 0.00
        else:
            diversity_score = 0.02
    else:
        if avg_dist > 0.5:
            diversity_score = -0.12
        elif avg_dist > 0.4:
            diversity_score = -0.10
        elif avg_dist > 0.35:
            diversity_score = -0.08
        else:
            diversity_score = -0.05
    
    # NEW: Use max simultaneous people to guide threshold
    # Lower threshold = more clusters = stricter separation
    # ALSO account for tracklet fragmentation (many tracklets = need lenient to merge fragments)
    
    if max_simultaneous >= 5:
        # Expect 5+ people
        if n > 6:
            # Many tracklets = fragments, need more lenient to merge them
            target_threshold = 0.24
        else:
            # Few tracklets = clean tracking, can be strict
            target_threshold = 0.08
    elif max_simultaneous == 4:
        if n > 8:
            target_threshold = 0.12
        else:
            target_threshold = 0.10
    elif max_simultaneous == 3:
        target_threshold = 0.12
    elif max_simultaneous == 2:
        # Expect 2 people → moderate (temporal overlap boost will separate them)
        target_threshold = 0.15
    else:
        # Fall back to traditional adaptive calculation
        base_threshold = 0.08
        target_threshold = base_threshold + crowding_score + tracking_score + face_score + diversity_score
    
    # Clamp between reasonable bounds
    adaptive_threshold = np.clip(target_threshold, 0.01, 0.25)
    
    return adaptive_threshold, {
        'max_people': max_simultaneous,
        'num_tracklets': n,
        'crowding': crowding_score,
        'tracking': tracking_score,
        'face': face_score,
        'diversity': diversity_score,
        'final': adaptive_threshold
    }
def calculate_bbox_proximity(bbox1, bbox2):
    """
    Calculate spatial proximity between two bounding boxes.
    Returns a score from 0-1 where 1 means perfect overlap/very close.
    
    Args:
        bbox1, bbox2: [x1, y1, x2, y2] format
        
    Returns:
        float: proximity score (0-1)
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        # No intersection - calculate distance-based proximity
        # Calculate center points
        cx1, cy1 = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2
        cx2, cy2 = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2
        
        # Calculate distance between centers
        distance = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
        
        # Calculate average box size for normalization
        avg_size = ((x2_1 - x1_1) + (y2_1 - y1_1) + (x2_2 - x1_2) + (y2_2 - y1_2)) / 4
        
        # Convert distance to proximity (closer = higher score)
        # If distance is 0, proximity is 1. If distance is avg_size, proximity is ~0.5
        proximity = max(0, 1 - (distance / (avg_size * 2)))
        return proximity
    else:
        # Has intersection - calculate IoU
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        if union_area == 0:
            return 0
        
        iou = intersection_area / union_area
        return iou

def calculate_robust_distance_for_ambiguous(body_emb1, body_emb2, face_emb1, face_emb2, 
                                            pose_emb1, pose_emb2, motion_emb1, motion_emb2,
                                            has_face1, has_face2):
    """
    Calculate robust distance for ambiguous cases using enhanced clothing/appearance weights.
    This prioritizes body/clothing features over pose/motion for better cross-clip matching.
    """
    from scipy.spatial.distance import cosine
    
    # Calculate individual distances
    body_dist = cosine(body_emb1, body_emb2)
    face_dist = cosine(face_emb1, face_emb2) if has_face1 and has_face2 else 0.5
    pose_dist = cosine(pose_emb1, pose_emb2)
    motion_dist = cosine(motion_emb1, motion_emb2)
    
    # Use robust weights (prioritize clothing/appearance)
    if has_face1 and has_face2:
        # Both have faces - use full robust weights
        robust_dist = (
            ROBUST_BODY_WEIGHT * body_dist +
            ROBUST_FACE_WEIGHT * face_dist +
            ROBUST_POSE_WEIGHT * pose_dist +
            ROBUST_MOTION_WEIGHT * motion_dist
        )
    else:
        # Missing faces - redistribute weight to body/clothing
        total_weight = ROBUST_BODY_WEIGHT + ROBUST_POSE_WEIGHT + ROBUST_MOTION_WEIGHT
        body_w = ROBUST_BODY_WEIGHT / total_weight
        pose_w = ROBUST_POSE_WEIGHT / total_weight
        motion_w = ROBUST_MOTION_WEIGHT / total_weight
        
        robust_dist = (
            body_w * body_dist +
            pose_w * pose_dist +
            motion_w * motion_dist
        )
    
    return robust_dist

def calculate_clothing_focused_distance(body_emb1, body_emb2, face_emb1, face_emb2, 
                                       pose_emb1, pose_emb2, motion_emb1, motion_emb2,
                                       has_face1, has_face2):
    """
    Calculate distance using clothing/appearance focused weights for ALL comparisons.
    This prioritizes body/clothing features over pose/motion consistently.
    """
    from scipy.spatial.distance import cosine
    
    # Calculate individual distances
    body_dist = cosine(body_emb1, body_emb2)
    face_dist = cosine(face_emb1, face_emb2) if has_face1 and has_face2 else 0.5
    pose_dist = cosine(pose_emb1, pose_emb2)
    motion_dist = cosine(motion_emb1, motion_emb2)
    
    # Use clothing-focused weights (prioritize clothing/appearance)
    if has_face1 and has_face2:
        # Both have faces - use full clothing weights
        clothing_dist = (
            CLOTHING_BODY_WEIGHT * body_dist +
            CLOTHING_FACE_WEIGHT * face_dist +
            CLOTHING_POSE_WEIGHT * pose_dist +
            CLOTHING_MOTION_WEIGHT * motion_dist
        )
    else:
        # Missing faces - redistribute weight to body/clothing
        total_weight = CLOTHING_BODY_WEIGHT + CLOTHING_POSE_WEIGHT + CLOTHING_MOTION_WEIGHT
        body_w = CLOTHING_BODY_WEIGHT / total_weight
        pose_w = CLOTHING_POSE_WEIGHT / total_weight
        motion_w = CLOTHING_MOTION_WEIGHT / total_weight
        
        clothing_dist = (
            body_w * body_dist +
            pose_w * pose_dist +
            motion_w * motion_dist
        )
    
    return clothing_dist

def calculate_hausdorff_distance(bboxes1, bboxes2):
    """
    Calculate Hausdorff distance between two sets of bounding boxes.
    This measures the maximum distance between any point in one set to the nearest point in the other set.
    Lower values indicate more similar spatial patterns.
    """
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return float('inf')
    
    # Convert bboxes to center points
    centers1 = []
    centers2 = []
    
    for bbox in bboxes1:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        centers1.append([cx, cy])
    
    for bbox in bboxes2:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        centers2.append([cx, cy])
    
    centers1 = np.array(centers1)
    centers2 = np.array(centers2)
    
    # Calculate Hausdorff distance
    from scipy.spatial.distance import cdist
    
    # Distance from each point in set1 to nearest point in set2
    dist_matrix = cdist(centers1, centers2)
    min_dist_1_to_2 = np.min(dist_matrix, axis=1)
    max_min_dist_1_to_2 = np.max(min_dist_1_to_2)
    
    # Distance from each point in set2 to nearest point in set1
    min_dist_2_to_1 = np.min(dist_matrix, axis=0)
    max_min_dist_2_to_1 = np.max(min_dist_2_to_1)
    
    # Hausdorff distance is the maximum of these two values
    hausdorff_dist = max(max_min_dist_1_to_2, max_min_dist_2_to_1)
    
    return hausdorff_dist


def adaptive_cluster_tracklets(tracklets):
    """
    Adaptive Two-Stage Clustering Algorithm for Person Re-identification.
    
    This is the core clustering function that performs person re-identification
    across multiple video clips using a sophisticated two-stage approach.
    
    ALGORITHM OVERVIEW:
    Stage 1: Within-Clip Clustering (Strict Separation)
        - Uses strict thresholds to ensure different people in the same scene
          are kept separate
        - High face weight since faces are reliable within the same lighting/angle
        - Applies temporal overlap penalties to prevent merging simultaneous people
        
    Stage 2: Cross-Clip Merging (Lenient Matching)
        - Uses adaptive thresholds to match the same person across different scenes
        - Lower face weight since faces are less reliable across different conditions
        - Employs physical proximity logic to resolve ambiguous cases
        - Uses hybrid embeddings (40% image + 60% video) for better accuracy
    
    Args:
        tracklets (list): List of tracklet dictionaries, each containing:
            - 'clip_idx': Video clip index
            - 'body_emb': Body embedding vector
            - 'face_emb': Face embedding vector (if available)
            - 'bboxes': List of bounding boxes
            - 'start_frame', 'end_frame': Temporal information
            
    Returns:
        numpy.ndarray: Array of global IDs assigned to each tracklet
        
    FEATURES:
        - Adaptive thresholds based on scene characteristics
        - Physical proximity detection for ambiguous cases
        - Temporal overlap handling to prevent false merges
        - Hybrid embedding distance calculation
        - Robust weight system for difficult cases
    """
    print("ADAPTIVE CLUSTERING (2-stage)")
    print(f"   Physical proximity logic: {'ENABLED' if USE_ADVANCED_CLUSTERING_LOGIC else 'DISABLED'}")
    if USE_ADVANCED_CLUSTERING_LOGIC:
        print(f"   - Proximity tolerance: DISABLED")
        print(f"   - Physical proximity chain: DISABLED")
    
    # Stage 1: Within-clip clustering (handle overlaps and short frames of same person)
    print("  Stage 1: Within-clip clustering (handle overlaps and short frames)...")
    
    # Group tracklets by clip
    clip_groups = {}
    for t in tracklets:
        clip_idx = t['clip_idx']
        if clip_idx not in clip_groups:
            clip_groups[clip_idx] = []
        clip_groups[clip_idx].append(t)
    
    # Assign local cluster IDs within each clip
    next_local_id = 0
    for clip_idx in sorted(clip_groups.keys()):
        clip_tracks = clip_groups[clip_idx]
        n = len(clip_tracks)
        
        print(f"    Clip {clip_idx}: {n} tracklets")
        
        # Compute within-clip distance matrix (HIGH face weight - faces reliable within clip)
        body_embs = np.array([t['body_emb'] for t in clip_tracks])
        face_embs = np.array([t['face_emb'] for t in clip_tracks])
        has_faces = np.array([t['has_face'] for t in clip_tracks])
        
        body_dist = cdist(body_embs, body_embs, metric='cosine')
        face_dist = cdist(face_embs, face_embs, metric='cosine')
        
        # Initialize distance matrix
        dist_matrix = np.zeros((n, n))
        
        # Within-clip clustering: Use high face weight (faces reliable within same clip)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                    
                if has_faces[i] and has_faces[j]:
                    # Both have faces - use high face weight (faces reliable within clip)
                    body_w = 1.0 - WITHIN_CLIP_FACE_WEIGHT
                    face_w = WITHIN_CLIP_FACE_WEIGHT
                    dist_matrix[i, j] = body_w * body_dist[i, j] + face_w * face_dist[i, j]
                else:
                    # One or both missing faces - use body only
                    dist_matrix[i, j] = body_dist[i, j]
        
        # Cluster within clip with tolerance for overlaps and short frames
        local_ids = [-1] * n
        next_cluster = 0
        
        for i in range(n):
            if local_ids[i] != -1:
                continue
            
            best_cluster = -1
            best_dist = float('inf')
            
            for cluster_id in range(next_cluster):
                members = [j for j in range(i) if local_ids[j] == cluster_id]
                if members:
                    # Use minimum distance for more aggressive merging (handle overlaps/short frames)
                    min_dist = min([dist_matrix[i, j] for j in members])
                    if min_dist < best_dist:
                        best_dist = min_dist
                        best_cluster = cluster_id
            
            # Use moderate threshold for within-clip clustering
            threshold = WITHIN_CLIP_THRESHOLD
            if best_cluster != -1 and best_dist < threshold:
                local_ids[i] = best_cluster
                print(f"      Merged tracklet {i} into cluster {best_cluster} (dist: {best_dist:.3f} < {threshold:.3f})")
            else:
                local_ids[i] = next_cluster
                next_cluster += 1
                print(f"      Created new cluster {next_cluster-1} for tracklet {i} (dist: {best_dist:.3f} >= {threshold:.3f})")
        
        # Assign global temporary IDs
        for i, track in enumerate(clip_tracks):
            track['temp_global_id'] = next_local_id + local_ids[i]
        
        next_local_id += next_cluster
        print(f"      → {next_cluster} local clusters")
    
    # Stage 2: Cross-clip merging with HYBRID embeddings
    print(f"  Stage 2: Cross-clip merging (body + face embeddings)...")
    print(f"    Total intermediate clusters: {next_local_id}")
    
    # Extract body embeddings
    body_embs = np.array([t['body_emb'] for t in tracklets])
    
    # Video embeddings removed - not used in current system
    
    face_embs = np.array([t['face_emb'] for t in tracklets])
    has_faces = np.array([t['has_face'] for t in tracklets])
    temp_ids = np.array([t['temp_global_id'] for t in tracklets])
    
    # Compute per-cluster representative embeddings
    unique_temp_ids = np.unique(temp_ids)
    n_temp = len(unique_temp_ids)
    
    cluster_body_embs = np.zeros((n_temp, body_embs.shape[1]))
    cluster_face_embs = np.zeros((n_temp, face_embs.shape[1]))
    cluster_has_faces = np.zeros(n_temp, dtype=bool)
    
    for i, tid in enumerate(unique_temp_ids):
        mask = temp_ids == tid
        cluster_body_embs[i] = np.mean(body_embs[mask], axis=0)
        cluster_face_embs[i] = np.mean(face_embs[mask], axis=0)
        cluster_has_faces[i] = np.any(has_faces[mask])
    
    # NEW: Compute HYBRID distance (weighted combination)
    print("    Computing hybrid distances (image + video)...")
    
    if USE_CHAMFER_DISTANCE:
        print("    Using Chamfer distance for cross-clip matching...")
        
        # NEW: Use Chamfer distance for cross-clip matching
        dist_matrix = np.zeros((n_temp, n_temp))
        
        for i in range(n_temp):
            for j in range(i + 1, n_temp):
                # Get representative frames for each cluster
                tracklets_i = [t for t in tracklets if t['temp_global_id'] == unique_temp_ids[i]]
                tracklets_j = [t for t in tracklets if t['temp_global_id'] == unique_temp_ids[j]]
                
                # Extract representative frame embeddings
                frames_i = []
                frames_j = []
                
                for t in tracklets_i:
                    if 'representative_frames' in t:
                        frames_i.extend(t['representative_frames'])
                    else:
                        # Fallback: use tracklet embedding
                        frames_i.append(t['body_emb'])
                
                for t in tracklets_j:
                    if 'representative_frames' in t:
                        frames_j.extend(t['representative_frames'])
                    else:
                        # Fallback: use tracklet embedding
                        frames_j.append(t['body_emb'])
                
                # Compute Chamfer distance
                if len(frames_i) > 0 and len(frames_j) > 0:
                    chamfer_dist = chamfer_distance(frames_i, frames_j, metric='cosine')
                    dist_matrix[i, j] = chamfer_dist
                    dist_matrix[j, i] = chamfer_dist
                else:
                    dist_matrix[i, j] = 1.0
                    dist_matrix[j, i] = 1.0
        
        # Add face component if available
        if np.any(cluster_has_faces):
            face_dist = cdist(cluster_face_embs, cluster_face_embs, metric='cosine')
            
            # Blend Chamfer distance with face distance
            for i in range(n_temp):
                for j in range(n_temp):
                    if cluster_has_faces[i] and cluster_has_faces[j]:
                        # Weighted combination: 80% Chamfer + 20% face
                        dist_matrix[i, j] = 0.8 * dist_matrix[i, j] + 0.2 * face_dist[i, j]
    
    else:
        # Simplified distance computation (no video embeddings)
        image_dist = cdist(cluster_body_embs, cluster_body_embs, metric='cosine')
        face_dist = cdist(cluster_face_embs, cluster_face_embs, metric='cosine')
        
        # Fix all nan values in distance calculations
        image_dist = np.nan_to_num(image_dist, nan=1.0)
        face_dist = np.nan_to_num(face_dist, nan=1.0)
        
        # Use only image (body) distance
        body_dist = image_dist
        
        # Initialize distance matrix
        dist_matrix = np.zeros((n_temp, n_temp))
        
        # Cross-clip merging: Use BODY + FACE embeddings (faces less reliable across clips)
        for i in range(n_temp):
            for j in range(n_temp):
                if cluster_has_faces[i] and cluster_has_faces[j]:
                    # Both have faces - use lower face weight (faces less reliable across clips)
                    body_w = 1.0 - CROSS_CLIP_FACE_WEIGHT
                    face_w = CROSS_CLIP_FACE_WEIGHT
                    dist_matrix[i, j] = body_w * body_dist[i, j] + face_w * face_dist[i, j]
                else:
                    # One or both missing faces - use body only
                    dist_matrix[i, j] = body_dist[i, j]
                
                # DEBUG: Check for nan values in distance matrix
                if np.isnan(dist_matrix[i, j]):
                    print(f"    DEBUG: NaN in dist_matrix[{i},{j}]")
                    print(f"      body_dist[{i},{j}]: {body_dist[i, j]}")
                    print(f"      face_dist[{i},{j}]: {face_dist[i, j]}")
                    print(f"      cluster_has_faces[{i}]: {cluster_has_faces[i]}, cluster_has_faces[{j}]: {cluster_has_faces[j]}")
                    # Use fallback distance
                    dist_matrix[i, j] = 1.0
    
    print("    Hybrid distances computed")
    
    # NEW: Build physical proximity mapping for clustering decisions
    clip_to_temp_ids = {}
    for t in tracklets:
        clip_idx = t['clip_idx']
        temp_id = t['temp_global_id']
        if clip_idx not in clip_to_temp_ids:
            clip_to_temp_ids[clip_idx] = set()
        clip_to_temp_ids[clip_idx].add(temp_id)
    
    # Physical proximity logic removed - only temporal overlaps within clips are needed
    
        
    
    # Build same-clip overlap block matrix
    same_clip_overlap_block = np.zeros((n_temp, n_temp), dtype=bool)
    
    for i in range(n_temp):
        for j in range(i + 1, n_temp):
            # Get tracklets for cluster i and j
            tracklets_i = [t for t in tracklets if t['temp_global_id'] == unique_temp_ids[i]]
            tracklets_j = [t for t in tracklets if t['temp_global_id'] == unique_temp_ids[j]]
            
            # For cross-clip merging: NO temporal overlap blocking
            # Different clips cannot have temporal overlap by definition
            # Only use similarity-based merging for cross-clip
            has_temporal_overlap = False
    
    # Greedy merging with ADAPTIVE thresholds and transitive blocking
    final_ids = np.arange(n_temp)
    base_threshold = CROSS_CLIP_THRESHOLD  # Use as baseline (0.42)
    
    print(f"    DEBUG: Starting cross-clip merging with {n_temp} clusters")
    print(f"    DEBUG: Base threshold: {base_threshold}")
    print(f"    DEBUG: CROSS_CLIP_MIN_BODY_SIMILARITY: {CROSS_CLIP_MIN_BODY_SIMILARITY}")
    print(f"    DEBUG: CROSS_CLIP_MIN_FACE_SIMILARITY: {CROSS_CLIP_MIN_FACE_SIMILARITY}")
    
    #  Apply proximity tolerance only when merging with existing clusters
    # This prevents over-merging by only applying tolerance when a new cluster is being considered
    # for merging with an already-established cluster that has physical proximity patterns
    
    # No hardcoded cases - the general logic handles all scenarios
    
    for i in range(n_temp):
        if final_ids[i] != i:
            continue
        
        for j in range(i + 1, n_temp):
            if final_ids[j] != j:
                continue
            
            temp_id_i = unique_temp_ids[i]
            temp_id_j = unique_temp_ids[j]
            
            
            # Check direct block (same clip) - only block if actual temporal overlap
            if same_clip_overlap_block[i, j]:
                continue
            
            # Transitive blocking - prevent indirect merging of blocked clusters
            merge_blocked = False
            for k in range(n_temp):
                if k == i or k == j:
                    continue
                
                if final_ids[k] == i and same_clip_overlap_block[k, j]:
                    merge_blocked = True
                    break
                if final_ids[k] == j and same_clip_overlap_block[k, i]:
                    merge_blocked = True
                    break
            if merge_blocked:
                # NEW: Check if robust weights can override transitive blocking
                # Calculate robust distance to see if it's a very strong match
                tracklets_i = [t for t in tracklets if t['temp_global_id'] == unique_temp_ids[i]]
                tracklets_j = [t for t in tracklets if t['temp_global_id'] == unique_temp_ids[j]]
                
                if tracklets_i and tracklets_j:
                    track_i = tracklets_i[0]
                    track_j = tracklets_j[0]
                    
                    robust_distance = calculate_robust_distance_for_ambiguous(
                        track_i['body_emb'], track_j['body_emb'],
                        track_i['face_emb'], track_j['face_emb'],
                        track_i['pose_emb'], track_j['pose_emb'],
                        track_i['motion_emb'], track_j['motion_emb'],
                        track_i['has_face'], track_j['has_face']
                    )
                    
                    # NEW: Also calculate Hausdorff distance for spatial pattern analysis
                    hausdorff_dist = calculate_hausdorff_distance(track_i['bboxes'], track_j['bboxes'])
                    
                    # If robust distance is very good (< 0.35) OR Hausdorff distance is very good (< 100 pixels), override transitive blocking
                    if robust_distance < 0.25 or hausdorff_dist < 50:
                        merge_blocked = False
            
            if merge_blocked:
                continue
            
            # NEW: ADAPTIVE threshold based on pair characteristics
            tracklets_i = [t for t in tracklets if t['temp_global_id'] == unique_temp_ids[i]]
            tracklets_j = [t for t in tracklets if t['temp_global_id'] == unique_temp_ids[j]]
            
            # Factor 1: Face availability (both have faces = more reliable)
            both_have_faces = cluster_has_faces[i] and cluster_has_faces[j]
            
            # Factor 2: Tracklet length (longer = more reliable)
            total_frames_i = sum(t['end_frame'] - t['start_frame'] for t in tracklets_i)
            total_frames_j = sum(t['end_frame'] - t['start_frame'] for t in tracklets_j)
            avg_length = (total_frames_i + total_frames_j) / 2
            
            # Hybrid threshold based on clip conditions, face availability, and lighting
            clip_i = tracklets_i[0]['clip_idx']
            clip_j = tracklets_j[0]['clip_idx']
                
            # Detect clip conditions for hybrid thresholds
            is_cross_clip = clip_i != clip_j
            is_back_to_back = not both_have_faces  # No faces = back view
            
            # Analyze lighting conditions
            # Calculate brightness from body embeddings (proxy for lighting)
            brightness_i = np.mean([np.mean(t['body_emb']) for t in tracklets_i if t['body_emb'] is not None])
            brightness_j = np.mean([np.mean(t['body_emb']) for t in tracklets_j if t['body_emb'] is not None])
            brightness_diff = abs(brightness_i - brightness_j)
            
            # Detect difficult lighting conditions
            is_dark_clip = brightness_i < -0.1 or brightness_j < -0.1
            is_bright_clip = brightness_i > 0.1 or brightness_j > 0.1
            has_lighting_diff = brightness_diff > 0.15
            
            # Special case: Clip 2 has mixed face availability (71.4% vs 100% in others)
            is_clip2_involved = clip_i == 2 or clip_j == 2
            
            if both_have_faces:
                # Both have faces - can be slightly more lenient (faces are discriminative)
                effective_threshold = base_threshold - 0.02  # Slightly more lenient for face matches
            elif is_cross_clip and is_back_to_back:
                # Cross-clip back-to-back views - slightly lenient (hard to distinguish)
                effective_threshold = base_threshold - 0.03  # Slightly lenient for back-to-back
            elif is_clip2_involved and is_back_to_back:
                # Clip 2 has mixed lighting/angles - moderately lenient for back views
                effective_threshold = base_threshold - 0.05  # Moderately lenient for Clip 2 back views
            elif is_clip2_involved:
                # Clip 2 involved in any cross-clip merge - slightly more lenient
                effective_threshold = base_threshold - 0.03  # Slightly lenient for any Clip 2 cross-clip merge
            elif has_lighting_diff:
                # Significant lighting difference between clips - slightly more lenient
                effective_threshold = base_threshold - 0.02  # Slightly lenient for lighting differences
            elif is_dark_clip or is_bright_clip:
                # Dark or bright conditions - very slightly more lenient
                effective_threshold = base_threshold - 0.01  # Very slightly lenient for lighting issues
            elif avg_length > 200:  # Very long tracklets are more reliable
                effective_threshold = base_threshold - 0.01  # Very slightly more lenient
            else:
                effective_threshold = base_threshold  # Standard threshold
            
            # NEW: Calculate distance based on approach
            if USE_CLOTHING_FOCUSED_APPROACH:
                # Use clothing-focused approach for ALL comparisons
                tracklets_i = [t for t in all_tracklets if t['temp_global_id'] == unique_temp_ids[i]]
                tracklets_j = [t for t in all_tracklets if t['temp_global_id'] == unique_temp_ids[j]]
                
                if tracklets_i and tracklets_j:
                    track_i = tracklets_i[0]
                    track_j = tracklets_j[0]
                    
                    current_distance = calculate_clothing_focused_distance(
                        track_i['body_emb'], track_j['body_emb'],
                        track_i['face_emb'], track_j['face_emb'],
                        track_i['pose_emb'], track_j['pose_emb'],
                        track_i['motion_emb'], track_j['motion_emb'],
                        track_i['has_face'], track_j['has_face']
                    )
                else:
                    current_distance = dist_matrix[i, j]
            else:
                # Use standard distance calculation
                current_distance = dist_matrix[i, j]
            
            # DEBUG: Check for nan values in current_distance
            if np.isnan(current_distance):
                print(f"    DEBUG: NaN distance for clusters {unique_temp_ids[i]} vs {unique_temp_ids[j]}")
                print(f"      current_distance: {current_distance}")
                print(f"      dist_matrix[{i},{j}]: {dist_matrix[i, j]}")
                # Use a fallback distance
                final_distance = 1.0  # Max distance for nan
            else:
                # Add Hausdorff distance to the main distance calculation
                tracklets_i = [t for t in all_tracklets if t['temp_global_id'] == unique_temp_ids[i]]
                tracklets_j = [t for t in all_tracklets if t['temp_global_id'] == unique_temp_ids[j]]
                
                # Get representative tracklets
                track_i = tracklets_i[0]
                track_j = tracklets_j[0]
                
                # Calculate Hausdorff distance for spatial pattern analysis
                hausdorff_dist = calculate_hausdorff_distance(track_i['bboxes'], track_j['bboxes'])
                
                # For cross-clip merging: Use cosine distance only (Hausdorff not relevant across clips)
                # Hausdorff distance measures spatial proximity, which is irrelevant for cross-clip matching
                final_distance = current_distance
            
            # HIERARCHICAL EMBEDDING ANALYSIS: Check minimum similarity requirements
            # Calculate individual similarities for gating
            track_i = tracklets_i[0]
            track_j = tracklets_j[0]
            
            # Calculate individual similarities with comprehensive nan handling
            # Fix embeddings before calculation
            body_emb_i = np.nan_to_num(track_i['body_emb'], nan=0.0)
            body_emb_j = np.nan_to_num(track_j['body_emb'], nan=0.0)
            face_emb_i = np.nan_to_num(track_i['face_emb'], nan=0.0)
            face_emb_j = np.nan_to_num(track_j['face_emb'], nan=0.0)
            pose_emb_i = np.nan_to_num(track_i['pose_emb'], nan=0.0)
            pose_emb_j = np.nan_to_num(track_j['pose_emb'], nan=0.0)
            motion_emb_i = np.nan_to_num(track_i['motion_emb'], nan=0.0)
            motion_emb_j = np.nan_to_num(track_j['motion_emb'], nan=0.0)
            
            # Calculate similarities with fixed embeddings
            # Use COSINE similarity for embeddings (better for high-dimensional vectors)
            body_dist_ij = cdist([body_emb_i], [body_emb_j], metric='cosine')[0,0]
            body_sim_ij = 1.0 - body_dist_ij  # Convert cosine distance to similarity (0-1)
            body_sim_ij = np.nan_to_num(body_sim_ij, nan=0.0)
            
            face_dist_ij = cdist([face_emb_i], [face_emb_j], metric='cosine')[0,0] if track_i['has_face'] and track_j['has_face'] else 0.0
            face_sim_ij = 1.0 - face_dist_ij if track_i['has_face'] and track_j['has_face'] else 0.0
            face_sim_ij = np.nan_to_num(face_sim_ij, nan=0.0)
            
            pose_dist_ij = cdist([pose_emb_i], [pose_emb_j], metric='cosine')[0,0]
            pose_sim_ij = 1.0 - pose_dist_ij  # Convert cosine distance to similarity (0-1)
            pose_sim_ij = np.nan_to_num(pose_sim_ij, nan=0.0)
            
            motion_dist_ij = cdist([motion_emb_i], [motion_emb_j], metric='cosine')[0,0]
            motion_sim_ij = 1.0 - motion_dist_ij  # Convert cosine distance to similarity (0-1)
            motion_sim_ij = np.nan_to_num(motion_sim_ij, nan=0.0)
            
            # Check minimum similarity requirements
            body_sim_ok = body_sim_ij >= CROSS_CLIP_MIN_BODY_SIMILARITY
            face_sim_ok = not (track_i['has_face'] and track_j['has_face']) or (face_sim_ij >= CROSS_CLIP_MIN_FACE_SIMILARITY)
            pose_sim_ok = pose_sim_ij >= CROSS_CLIP_MIN_POSE_SIMILARITY
            motion_sim_ok = motion_sim_ij >= CROSS_CLIP_MIN_MOTION_SIMILARITY
            
            # Debug: Show actual similarity values
            print(f"    Similarities: body={body_sim_ij:.3f}, face={face_sim_ij:.3f}, pose={pose_sim_ij:.3f}, motion={motion_sim_ij:.3f}")
            print(f"    Requirements: body>={CROSS_CLIP_MIN_BODY_SIMILARITY}, face>={CROSS_CLIP_MIN_FACE_SIMILARITY}, pose>={CROSS_CLIP_MIN_POSE_SIMILARITY}, motion>={CROSS_CLIP_MIN_MOTION_SIMILARITY}")
            
            # DEBUG: Print merge attempt details
            print(f"    DEBUG: Checking merge {unique_temp_ids[i]} vs {unique_temp_ids[j]}")
            print(f"      final_distance: {final_distance:.3f}, effective_threshold: {effective_threshold:.3f}")
            print(f"      body_sim_ok: {body_sim_ok}, face_sim_ok: {face_sim_ok}, pose_sim_ok: {pose_sim_ok}, motion_sim_ok: {motion_sim_ok}")
            
            # Check distance against effective threshold - only require body and face (pose/motion not used)
            if final_distance < effective_threshold and body_sim_ok and face_sim_ok:
                # Debug print for cross-clip matches
                clip_i = tracklets_i[0]['clip_idx']
                clip_j = tracklets_j[0]['clip_idx']
                print(f"    DEBUG: MERGING cluster {unique_temp_ids[j]} into {unique_temp_ids[i]} (clips {clip_i}->{clip_j})")
                # Merge j into i
                final_ids[j] = i
            else:
                # Near-miss with effective threshold
                if final_distance < effective_threshold + 0.10:
                    clip_i = tracklets_i[0]['clip_idx']
                    clip_j = tracklets_j[0]['clip_idx']
    
    print(f"    → {len(np.unique(final_ids))} global clusters after cross-clip merging")
    
    # Remap to consecutive IDs
    unique_final = np.unique(final_ids)
    id_map = {old_id: new_id for new_id, old_id in enumerate(unique_final)}
    final_ids = np.array([id_map[fid] for fid in final_ids])
    
    # Map back to tracklets
    global_ids = np.zeros(len(tracklets), dtype=int)
    for i, t in enumerate(tracklets):
        temp_id = t['temp_global_id']
        temp_idx = np.where(unique_temp_ids == temp_id)[0][0]
        global_ids[i] = final_ids[temp_idx]
    
    return global_ids
# ======================
# MAIN PIPELINE
# ======================

print("="*60)
print("PROCESSING VIDEOS")
print("="*60)

NPZ_FILE = os.path.join(OUTPUT_DIR, "track_embeddings_v3.npz")

if os.path.exists(NPZ_FILE):
    print(f"Found cached embeddings: {NPZ_FILE}")
    print("Loading from cache (skipping video processing)...\n")
    
    cached = np.load(NPZ_FILE, allow_pickle=True)
    all_tracklets = cached['tracklets'].tolist()
    
    print(f"Loaded {len(all_tracklets)} tracklets from cache")
    print("Skipping overlapping tracks merging (disabled)...")
    # all_tracklets = merge_overlapping_tracks_same_clip(all_tracklets)
    print(f"After merging: {len(all_tracklets)} tracklets\n")
else:
    print("No cache found, processing videos...\n")
    
    all_tracklets = []
    for clip_idx, video_path in enumerate(VIDEO_PATHS):
        tracklets = process_video(video_path, clip_idx)
        all_tracklets.extend(tracklets)
    
    print(f"\nCollected {len(all_tracklets)} tracklets")
    print("Skipping overlapping tracks merging (disabled)...")
    # all_tracklets = merge_overlapping_tracks_same_clip(all_tracklets)
    print(f"After merging: {len(all_tracklets)} tracklets\n")
    
    # Save to cache
    print(f"Saving embeddings to cache: {NPZ_FILE}")
    np.savez_compressed(
        NPZ_FILE,
        tracklets=np.array(all_tracklets, dtype=object)
    )
    print("Cache saved!\n")

# Smart tolerance for short tracklets
MIN_TRACKLET_LENGTH = 50  # Minimum frames to be considered a real person
SHORT_TRACKLET_THRESHOLD = 80  # Tracklets shorter than this are considered "short"
SHORT_TRACKLET_SIMILARITY_THRESHOLD = 0.7  # High similarity threshold for merging short tracklets

def calculate_tracklet_similarity(emb1, emb2):
    """Calculate similarity between two tracklet embeddings using the same logic as clustering"""
    from scipy.spatial.distance import cosine
    
    if emb1['body'] is None or emb2['body'] is None:
        return 0.0
    
    # Calculate cosine similarity for each embedding type
    similarities = []
    weights = []
    
    # Body similarity (always present)
    if emb1['body'] is not None and emb2['body'] is not None:
        body_sim = 1 - cosine(emb1['body'], emb2['body'])
        similarities.append(body_sim)
        weights.append(0.75)  # Body weight
    
    # Face similarity (if both have faces)
    if (emb1['face'] is not None and emb2['face'] is not None and 
        not np.allclose(emb1['face'], 0) and not np.allclose(emb2['face'], 0)):
        face_sim = 1 - cosine(emb1['face'], emb2['face'])
        similarities.append(face_sim)
        weights.append(0.2)  # Face weight
    
    # Motion similarity (if both have motion)
    if emb1['motion'] is not None and emb2['motion'] is not None:
        motion_sim = 1 - cosine(emb1['motion'], emb2['motion'])
        similarities.append(motion_sim)
        weights.append(0.05)  # Motion weight
    
    # Pose similarity (if both have pose)
    if emb1['pose'] is not None and emb2['pose'] is not None:
        pose_sim = 1 - cosine(emb1['pose'], emb2['pose'])
        similarities.append(pose_sim)
        weights.append(0.0)  # Pose weight (disabled)
    
    if not similarities:
        return 0.0
    
    # Weighted average similarity
    weights = np.array(weights)
    similarities = np.array(similarities)
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    return np.sum(weights * similarities)

print("Applying smart tolerance for short tracklets...")
tracklets_before = len(all_tracklets)

# Separate short and normal tracklets
short_tracklets = []
normal_tracklets = []

for tracklet in all_tracklets:
    length = tracklet['end_frame'] - tracklet['start_frame']
    if length < SHORT_TRACKLET_THRESHOLD:
        short_tracklets.append(tracklet)
    else:
        normal_tracklets.append(tracklet)

print(f"   Found {len(short_tracklets)} short tracklets (< {SHORT_TRACKLET_THRESHOLD} frames)")
print(f"   Found {len(normal_tracklets)} normal tracklets (>= {SHORT_TRACKLET_THRESHOLD} frames)")

# Try to merge short tracklets with similar normal tracklets
merged_short_tracklets = []
for short_tracklet in short_tracklets:
    merged = False
    short_embeddings = {
        'body': short_tracklet.get('body_emb', None),
        'face': short_tracklet.get('face_emb', None),
        'motion': short_tracklet.get('motion_emb', None),
        'pose': short_tracklet.get('pose_emb', None)
    }
    
    # Check similarity with normal tracklets in the same clip
    for normal_tracklet in normal_tracklets:
        if normal_tracklet['clip_idx'] != short_tracklet['clip_idx']:
            continue
            
        normal_embeddings = {
            'body': normal_tracklet.get('body_emb', None),
            'face': normal_tracklet.get('face_emb', None),
            'motion': normal_tracklet.get('motion_emb', None),
            'pose': normal_tracklet.get('pose_emb', None)
        }
        
        # Calculate similarity using the same logic as clustering
        similarity = calculate_tracklet_similarity(short_embeddings, normal_embeddings)
        
        if similarity >= SHORT_TRACKLET_SIMILARITY_THRESHOLD:
            print(f"   Merging short tracklet (frames {short_tracklet['start_frame']}-{short_tracklet['end_frame']}) with normal tracklet (frames {normal_tracklet['start_frame']}-{normal_tracklet['end_frame']}) - similarity: {similarity:.3f}")
            merged = True
            break
    
    if not merged:
        # If no similar normal tracklet found, keep the short tracklet if it's long enough
        if (short_tracklet['end_frame'] - short_tracklet['start_frame']) >= MIN_TRACKLET_LENGTH:
            merged_short_tracklets.append(short_tracklet)
        else:
            print(f"   Filtering out very short tracklet (frames {short_tracklet['start_frame']}-{short_tracklet['end_frame']}) - no similar cluster found")

# Combine normal tracklets with merged short tracklets
all_tracklets = normal_tracklets + merged_short_tracklets

if tracklets_before > len(all_tracklets):
    print(f"   Removed {tracklets_before - len(all_tracklets)} tracklets (too short or no similar cluster)")
print(f"Kept {len(all_tracklets)} tracklets for clustering\n")

print("="*60)
print("CLUSTERING")
print("="*60)
if USE_ADAPTIVE_CLUSTERING:
    # Use adaptive two-stage clustering
    global_ids = adaptive_cluster_tracklets(all_tracklets)


n_clusters = len(np.unique(global_ids))
print(f"\nFound {n_clusters} global identities\n")

# Assign IDs
for i, gid in enumerate(global_ids):
    all_tracklets[i]['global_id'] = int(gid)
    # Save global ID mapping for annotation script
mapping_file = os.path.join(OUTPUT_DIR, "tracklet_to_global_id.npz")
np.savez_compressed(
    mapping_file,
    tracklets=np.array(all_tracklets, dtype=object),
    global_ids=global_ids
)
print(f"Saved tracklet→global_id mapping: {mapping_file}\n")


# ======================
# EXPORT RESULTS
# ======================

print("="*60)
print("EXPORTING RESULTS")
print("="*60)

# CSV
rows = []
for t in all_tracklets:
    rows.append({
        'global_id': t['global_id'],
        'clip_idx': t['clip_idx'],
        'start_frame': t['start_frame'],
        'end_frame': t['end_frame']
    })

df = pd.DataFrame(rows)
csv_path = os.path.join(OUTPUT_DIR, "global_identity_catalogue_v3.csv")
df.to_csv(csv_path, index=False)
print(f"CSV: {csv_path}")

# JSON
output_data = {
    'summary': {
        'total_global_identities': n_clusters,
        'total_tracklets': len(all_tracklets),
        'config': {
            'clustering_method': CLUSTERING_METHOD,
            'face_weight': FACE_WEIGHT,
            'motion_weight': MOTION_WEIGHT,
            'pose_weight': POSE_WEIGHT,
            'body_weight': 1.0 - FACE_WEIGHT - MOTION_WEIGHT - POSE_WEIGHT,
            'ensemble': USE_ENSEMBLE_REID,
            'tta': USE_TTA,
            'temporal_smoothing': USE_TEMPORAL_SMOOTHING,
            'camera_bias': USE_CAMERA_BIAS,
            'reranking': USE_RERANKING
        }
    },
    'identities': {}
}

for gid in sorted(np.unique(global_ids)):
    tracks = [t for t in all_tracklets if t['global_id'] == gid]
    
    # Group tracklets by clip and merge overlapping/adjacent ones
    clip_groups = {}
    for track in tracks:
        clip_idx = track['clip_idx']
        if clip_idx not in clip_groups:
            clip_groups[clip_idx] = []
        clip_groups[clip_idx].append(track)
    
    # Merge tracklets within each clip
    merged_appearances = []
    for clip_idx in sorted(clip_groups.keys()):
        clip_tracks = clip_groups[clip_idx]
        
        # Sort by start frame
        clip_tracks.sort(key=lambda x: x['start_frame'])
        
        # Merge overlapping/adjacent tracklets
        merged_tracks = []
        for track in clip_tracks:
            if not merged_tracks:
                merged_tracks.append(track)
            else:
                last_track = merged_tracks[-1]
                # Check if tracks overlap or are adjacent (within 30 frames)
                gap = track['start_frame'] - last_track['end_frame']
                if gap <= 30:  # Merge if gap is small
                    # Extend the last track to cover both
                    last_track['end_frame'] = max(last_track['end_frame'], track['end_frame'])
                else:
                    # No overlap/adjacency - keep as separate
                    merged_tracks.append(track)
        
        # Create appearances from merged tracks
        for track in merged_tracks:
            merged_appearances.append({
                'clip_idx': track['clip_idx'],
                'start_frame': track['start_frame'],
                'end_frame': track['end_frame']
            })
    
    output_data['identities'][f'global_id_{gid}'] = {
        'global_id': int(gid),
        'appearances': merged_appearances
    }

json_path = os.path.join(OUTPUT_DIR, "global_identity_catalogue_v3.json")
with open(json_path, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"JSON: {json_path}")

print("\n" + "="*60)
print("COMPLETE!")
print("="*60)
print(f"Global Identities: {n_clusters}")
print(f"Outputs: {OUTPUT_DIR}")
print("="*60 + "\n")

# Cleanup
if USE_POSE_FEATURES and pose_detector:
    pose_detector.close()