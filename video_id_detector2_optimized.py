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
from config.defaults import _C as cfg_default  # This is correct
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

"""
Video ID Detector 2 - Advanced Person Re-identification System

OVERVIEW:
This system performs person re-identification across multiple video clips using advanced
computer vision and machine learning techniques. It identifies the same person appearing
in different video clips, even when viewed from different angles, lighting conditions,
or with occlusions.

ALGORITHM OVERVIEW:
The system uses a two-stage adaptive clustering approach:
1. Stage 1 (Within-Clip Clustering): Strict separation of different people within the same video clip
   - Uses high face weight (0.4) since faces are reliable within same lighting/angle
   - Applies temporal overlap analysis to prevent merging simultaneous people
   - Adaptive thresholds based on clip characteristics (crowding, tracking quality, face visibility)
   - Hard block for overlap > 50% (different people appearing simultaneously)

2. Stage 2 (Cross-Clip Merging): Lenient matching of the same person across different video clips
   - Uses face weight 0.45 (45%) for cross-clip matching (slightly higher than within-clip 0.4)
   - Image-only embeddings (body + face when both available)
   - Best-of-cluster matching: compares best tracklet pairs to prevent single bad tracklet from dominating
   - Two-pass merging: face/no-face pairs first, then face-to-face pairs
   - Adaptive thresholds based on tracklet length and face availability

KEY FEATURES:
1. Multi-Model Ensemble: Combines OSNet (IBN-Net) and TransReID for robust body feature extraction
2. Face Recognition: InsightFace for face embeddings when faces are detected
3. Pose Features: MediaPipe for pose keypoints (optional, currently disabled)
4. View Invariance: Test-Time Augmentation (horizontal flip) and temporal smoothing
5. Quality-Weighted Embeddings: Higher weight for frames with faces and good pose detection
6. Representative Frame Selection: Chooses diverse high-quality frames for multi-frame matching
7. Caching System: NPZ file caching for fast parameter tuning and development

EMBEDDING EXTRACTION:
- Body: OSNet (512D) + TransReID (768D) ensemble → 1280D (weighted 70%/30%)
- Face: InsightFace → 512D (when face detected)
- Pose: MediaPipe → 66D (optional, currently disabled)
- Motion: Temporal features → 5D (currently disabled)

DISTANCE CALCULATION:
- Within-clip: Body (60%) + Face (40%) when both have faces (WITHIN_CLIP_FACE_WEIGHT = 0.4)
- Cross-clip: Body (55%) + Face (45%) when both have faces, body-only otherwise (CROSS_CLIP_FACE_WEIGHT = 0.45)
- Best-of-cluster: Uses minimum distance among all tracklet pairs for merged clusters

INPUT:
- Multiple video files (MP4 format) in ./videos/ directory
- Each video represents a different scene/time period
- Videos are processed sequentially with unique clip indices

OUTPUT:
- Global identity catalogue (JSON): Summary and detailed appearance information
- Tracklet-to-global-ID mappings (NPZ): For caching and analysis
- Embeddings cache (NPZ): Pre-computed embeddings for fast iteration

USAGE:
    python video_id_detector2_optimized.py

CONFIGURATION:
- Adjust clustering thresholds in CONFIGURATION section
- Enable/disable features via flags (USE_TTA, USE_POSE_FEATURES, etc.)
- Per-clip thresholds can be manually overridden or auto-calculated

"""

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
MOTION_WEIGHT = 0.1            # Motion/temporal features (10%)
POSE_WEIGHT = 0.1              # Pose estimation features (10%)
# Body features get remaining weight: 1 - 0.2 - 0.1 - 0.1 = 0.6 (60%)

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


# ======================
# CLUSTERING CONFIGURATION
# ======================

USE_ADVANCED_CLUSTERING_LOGIC = True


# View Invariance Features
# Test-time augmentation for improved robustness across different viewing angles
USE_TTA = True                        # Horizontal flip augmentation
USE_TEMPORAL_SMOOTHING = True         # Smooth embeddings within tracks for stability
SMOOTHING_WINDOW = 5                  # Number of frames to smooth over

# ======================
# CLUSTERING ALGORITHM CONFIGURATION
# ======================

# Distance Metrics
DISTANCE_METRIC = "cosine"            # Primary distance metric
USE_CHAMFER_DISTANCE = False          # Alternative distance metric for cross-clip matching

# Adaptive Clustering System
# Two-stage approach: strict within-clip, lenient cross-clip
USE_ADAPTIVE_CLUSTERING = True

# Stage 1: Within-clip clustering (strict - separate people in same scene)
WITHIN_CLIP_THRESHOLD = 0.15 # More reasonable for Clip 2 similarities
WITHIN_CLIP_FACE_WEIGHT = 0.4  # High - faces reliable in same lighting/angle

# Stage 2: Cross-clip merging (lenient - match same person across scenes)
CROSS_CLIP_THRESHOLD = 0.46  # Lenient for matching
CROSS_CLIP_FACE_WEIGHT = 0.45  # 45% - slightly higher than within-clip (0.4) for cross-clip matching

# Per-clip adaptive thresholds (AUTOMATIC ANALYSIS)
USE_PER_CLIP_THRESHOLDS = True  # Enable automatic per-clip analysis
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
    """
    Extract OSNet body embedding with Test-Time Augmentation (TTA).
    
    This function extracts body embeddings from a person crop using OSNet (IBN-Net)
    with optional horizontal flip augmentation for improved view invariance.
    
    Args:
        crop (numpy.ndarray): Person crop image in BGR format
        
    Returns:
        numpy.ndarray: 512-dimensional normalized OSNet embedding vector
        
    Processing:
        1. Resizes crop to 128x256 (HxW) for OSNet input
        2. Normalizes using ImageNet statistics
        3. Extracts embedding from original image
        4. If TTA enabled: extracts embedding from horizontally flipped image
        5. Averages original and flipped embeddings
        6. Normalizes to unit length
        
    Features:
        - Test-Time Augmentation improves robustness to view changes
        - Returns zero vector on extraction failure
    """
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


def extract_body_batch(crops):
    """
    Extract body embeddings for a batch of person crops using ensemble of OSNet and TransReID.
    
    This function processes multiple person crops and extracts combined body embeddings
    using an ensemble of two models: OSNet (IBN-Net) and TransReID. The ensemble
    provides more robust and discriminative features than a single model.
    
    Args:
        crops (list): List of person crop images (numpy arrays in BGR format)
        
    Returns:
        list: List of body embedding vectors, each 1280-dimensional (if ensemble enabled)
              or 512-dimensional (OSNet only)
              
    Ensemble Configuration:
        - OSNet weight: 0.7 (70% of 512D = 358.4D effective)
        - TransReID weight: 0.3 (30% of 768D = 230.4D effective)
        - Final concatenated: 512D + 768D = 1280D total
        
    Processing:
        1. Extracts OSNet embedding with TTA for each crop
        2. If ensemble enabled: extracts TransReID embedding
        3. Combines embeddings with weighted concatenation
        4. Returns list of combined embeddings
        
    Note:
        - OSNet embedding already includes TTA (horizontal flip averaging)
        - TransReID embedding is extracted separately without TTA
        - Both embeddings are normalized individually before concatenation
    """
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
    Compute quality score for a frame based on detection confidence and feature availability.
    
    Higher quality scores indicate more reliable frames for ReID embedding extraction.
    Quality is used to weight frame embeddings when aggregating tracklet representations.
    
    Args:
        bbox_conf (float): Bounding box detection confidence (0-1)
        face_detected (bool): Whether a face was detected in this frame
        pose_conf (float): Pose detection confidence (0-1)
        
    Returns:
        float: Quality score in range [0, 1], where 1.0 is highest quality
        
    Weighting:
        - Face presence: 60% weight (binary: 1.0 if detected, 0.0 if not)
        - Pose confidence: 30% weight (clipped to [0, 1])
        - Bbox confidence: 10% weight (clipped to [0, 1], default 0.5 if missing)
        
    Rationale:
        Face presence is most important for reliable person identification.
        Pose features provide additional discriminative information.
        Bbox confidence indicates detection quality.
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
    
    This function aggregates multiple frame embeddings into a single tracklet embedding
    by weighting each frame according to its quality score. High-quality frames (with
    faces and good pose detection) contribute more to the final representation.
    
    Args:
        embeddings_list (list): List of embedding vectors from consecutive frames
        qualities (list): List of quality scores (0-1) corresponding to each frame
        
    Returns:
        numpy.ndarray: Single weighted average embedding vector (normalized to unit length)
        None: If input list is empty
        
    Algorithm:
        1. Converts embeddings and qualities to numpy arrays
        2. Applies softmax weighting with temperature=3.0 to sharpen distribution
        3. Computes weighted average: sum(embedding * weight) for each dimension
        4. Normalizes result to unit length
        
    Benefits:
        - High-quality frames dominate the final representation
        - Smooth transition between frames
        - Normalized output ensures consistent distance calculations
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

def chamfer_distance(frames1, frames2, metric='cosine'):
    """
    Compute Chamfer distance between two sets of frame embeddings.
    
    Chamfer distance measures the average distance from each point in one set to its
    nearest neighbor in the other set, computed bidirectionally. This is useful for
    comparing tracklets with multiple representative frames.
    
    Args:
        frames1 (numpy.ndarray or list): First set of frame embeddings [N, D]
        frames2 (numpy.ndarray or list): Second set of frame embeddings [M, D]
        metric (str): Distance metric ('cosine' or 'euclidean')
    
    Returns:
        float: Chamfer distance value (0 = identical, larger = more different)
        1.0: If either set is empty (maximum distance)
        
    Algorithm:
        1. Computes pairwise distance matrix [N x M]
        2. For each frame in set1, finds minimum distance to any frame in set2
        3. Averages these minimum distances → avg_1to2
        4. For each frame in set2, finds minimum distance to any frame in set1
        5. Averages these minimum distances → avg_2to1
        6. Chamfer distance = (avg_1to2 + avg_2to1) / 2
        
    Note:
        Currently disabled (USE_CHAMFER_DISTANCE = False). Standard cosine distance
        with best-of-cluster matching is used instead.
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
    
    return chamfer_dist

def select_representative_frames(frame_embeddings, qualities, max_frames=15):
    """
    Select most representative frames from a tracklet for multi-frame matching.
    
    This function uses a two-stage strategy to select frames that are both high-quality
    and diverse, ensuring good coverage of the person's appearance across the tracklet.
    
    Args:
        frame_embeddings (list): List of embedding vectors [D] from consecutive frames
        qualities (list): List of quality scores (0-1) corresponding to each frame
        max_frames (int): Maximum number of frames to select (default: 15)
    
    Returns:
        numpy.ndarray: Selected frame embeddings as [K, D] array where K <= max_frames
        
    Selection Strategy:
        1. Quality-based selection: Always includes top-quality frames
           - Selects at least 5 frames or 30% of total frames (whichever is larger)
           - Prioritizes frames with faces and good pose detection
           
        2. Diversity-based selection: Adds diverse frames to cover different views
           - Uses farthest-point sampling (greedy approach)
           - Iteratively selects frame farthest from already-selected set
           - Ensures coverage of appearance space, not just time
           
        3. Budget constraint: Limits total selection to max_frames for efficiency
        
    Benefits:
        - High-quality frames ensure reliable matching
        - Diverse frames handle view changes and occlusions
        - Efficient representation reduces computation in cross-clip matching
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


def merge_overlapping_tracks_same_clip(tracklets):
    """
    Merge tracklet fragments within the same video clip that belong to the same person.
    
    This function identifies and merges tracklet fragments that were split due to
    tracking interruptions, occlusions, or detection gaps. It uses temporal overlap
    analysis and embedding similarity to safely merge fragments while preventing
    incorrect merges of different people appearing simultaneously.
    
    Args:
        tracklets (list): List of tracklet dictionaries from all video clips,
            each containing 'clip_idx', 'start_frame', 'end_frame', 'body_emb', etc.
        
    Returns:
        list: List of merged tracklets with combined temporal ranges and averaged embeddings
        
    Algorithm:
        1. Groups tracklets by video clip (clip_idx)
        2. For each clip, processes tracklets in temporal order
        3. For each tracklet, checks if it can merge with existing merged tracklets:
           a. Temporal overlap check: Must have gap <= 120 frames (3 seconds at 30fps)
           b. Hard block: If overlap > 50%, they are different people → don't merge
           c. Embedding similarity check: Must have body similarity >= 0.8
        4. Merges fragments by:
           - Combining temporal ranges (min start, max end)
           - Averaging body and face embeddings
           - Updating face detection flags (OR operation)
        
    Key Features:
        - Hard block for overlap > 50% prevents merging simultaneous different people
        - Gap tolerance allows merging fragments separated by short interruptions
        - High similarity requirement (0.8) ensures only same-person merges
        - Preserves all original tracklet metadata
        
    Note:
        This function is called before adaptive clustering to reduce fragmentation.
        It is conservative and only merges obvious fragments to avoid false merges.
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
    """
    Extract pose features for a batch of person crops.
    
    This function processes multiple person crop images and extracts pose keypoints
    for each using MediaPipe. It is a batch wrapper around extract_pose_features().
    
    Args:
        crops (list): List of person crop images (numpy arrays in BGR format)
        
    Returns:
        numpy.ndarray: Array of pose feature vectors [N, 66] where N = len(crops)
                      Each vector is 66-dimensional (33 landmarks × 2 coordinates)
                      Returns zero vectors if pose detection is disabled or fails
    """
    return np.array([extract_pose_features(c) for c in crops], dtype=np.float32)

def extract_motion_features(track_data):
    """
    Extract motion features from track data for temporal analysis.
    
    This function computes motion-related features from a person's track data,
    capturing temporal patterns in movement, position, and size that can help
    distinguish different people based on their motion characteristics.
    
    Args:
        track_data (list): List of movement dictionaries, each containing:
            - 'movement': Movement distance between consecutive frames (normalized)
            - 'bbox_size': Relative bounding box size (area / frame area)
            - 'v_pos': Vertical position (center y / frame height)
            - 'h_pos': Horizontal position (center x / frame width)
            
    Returns:
        numpy.ndarray: 5-dimensional motion feature vector:
            [avg_movement, avg_bbox_size, avg_v_pos, avg_h_pos, std_movement]
        Zero vector if track_data is empty
        
    Features:
        1. Average movement: Mean movement distance across all frames
        2. Average bbox size: Mean relative bounding box area
        3. Average vertical position: Mean vertical position in frame
        4. Average horizontal position: Mean horizontal position in frame
        5. Movement std: Standard deviation of movement (captures motion variability)
        
    Note:
        Currently disabled (MOTION_WEIGHT = 0.0) as motion features were found
        to be non-discriminative (high average similarity between different people).
        Motion features are still extracted but not used in distance calculations.
    """
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
    
    This is the main video processing function that handles the complete pipeline from
    video input to tracklet extraction with all necessary embeddings. It performs
    person detection, tracking, embedding extraction, and quality assessment.
    
    Args:
        video_path (str): Path to the input video file (MP4 format)
        clip_idx (int): Index of the video clip (0, 1, 2, ...) for identification
        
    Returns:
        list: List of tracklet dictionaries, each containing:
            - 'clip_idx': Video clip index
            - 'track_id': YOLO tracking ID (unique within clip)
            - 'start_frame', 'end_frame': Temporal boundaries (inclusive)
            - 'num_detections': Number of frames in tracklet
            - 'body_emb': Averaged body embedding vector (1280D if ensemble, 512D otherwise)
            - 'face_emb': Averaged face embedding vector (512D, zero if no face)
            - 'pose_emb': Averaged pose embedding vector (66D, zero if disabled)
            - 'motion_emb': Motion feature vector (5D)
            - 'has_face': Boolean (True if >=15 consecutive face frames detected)
            - 'face_ratio': Proportion of frames with face detection
            - 'num_face_frames': Number of frames with detected faces
            - 'max_consec_face_frames': Maximum consecutive face frames
            - 'num_frames': Total number of frames
            - 'frames': List of frame indices where person was detected
            - 'bboxes': List of bounding boxes [x1, y1, x2, y2] for each frame
            - 'body_frames': Representative body frame embeddings [K, 1280D] for multi-frame matching
            - 'face_frames': Representative face frame embeddings [K, 512D] if faces available
            
    Processing Pipeline:
        1. Video metadata extraction: Reads FPS and total frame count
        2. YOLO person detection and tracking:
           - Uses YOLOv8 with ByteTrack tracker
           - Detects only class 0 (person) with conf=0.15, iou=0.35
           - Tracks persons across frames with persistent IDs
        3. Tracklet extraction:
           - Groups detections by track_id
           - Filters out tracks with <3 detections
        4. Multi-frame embedding extraction (batched for efficiency):
           - Body: OSNet + TransReID ensemble with TTA
           - Face: InsightFace (when detected)
           - Pose: MediaPipe (optional)
           - Motion: Temporal features (position, size, movement)
        5. Temporal smoothing: Applies EMA smoothing to reduce noise
        6. Quality-weighted aggregation:
           - Body: Weighted average based on frame quality (face > pose > bbox)
           - Face: Weighted average of frames with faces only
           - Pose: Simple average
        7. Representative frame selection:
           - Selects high-quality diverse frames for multi-frame matching
           - Body: Up to 15 representative frames
           - Face: Up to 10 representative frames (if faces available)
        8. Face detection validation:
           - Requires >=15 consecutive face frames for reliable face detection
           - Sets 'has_face' flag accordingly
           
    Output:
        Returns list of tracklets ready for clustering. Each tracklet represents
        a single person's appearance in the video with consolidated embeddings and
        representative frames for robust cross-clip matching.
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
        num_frames_total = len(face_embs_list)
        num_face_frames = len(face_indices)
        face_ratio = (num_face_frames / num_frames_total) if num_frames_total > 0 else 0.0
        # Consecutive face frames gate (more robust than pure ratio)
        def longest_consecutive_true(bools):
            max_run = 0
            cur = 0
            for v in bools:
                if v:
                    cur += 1
                    if cur > max_run:
                        max_run = cur
                else:
                    cur = 0
            return max_run
        FACE_MIN_CONSEC_FRAMES = 15
        max_consec_face = longest_consecutive_true(has_faces_list)
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
            # Robust face flag: require at least N consecutive face frames
            'has_face': max_consec_face >= FACE_MIN_CONSEC_FRAMES,
            'face_ratio': face_ratio,
            'num_face_frames': int(num_face_frames),
            'max_consec_face_frames': int(max_consec_face),
            'num_frames': int(num_frames_total),
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
    """
    Compute pairwise distance matrix between all tracklets.
    
    This function calculates the distance between every pair of tracklets using
    a weighted combination of body, face, pose, and motion embeddings. The weighting
    adapts based on whether faces are detected.
    
    Args:
        tracklets (list): List of tracklet dictionaries, each containing:
            - 'body_emb': Body embedding vector
            - 'face_emb': Face embedding vector
            - 'pose_emb': Pose embedding vector
            - 'motion_emb': Motion embedding vector
            - 'has_face': Boolean indicating face detection
            
    Returns:
        numpy.ndarray: Symmetric distance matrix [N x N] where N = len(tracklets)
        
    Weighting:
        - If both tracklets have faces:
          Body(60%) + Face(20%) + Pose(10%) + Motion(10%)
        - If either lacks face:
          Body(80%) + Pose(10%) + Motion(10%)  (face weight redistributed to body)
          
    Note:
        Currently only used for legacy clustering methods (SIMILARITY, DBSCAN).
        The main pipeline uses adaptive_cluster_tracklets() which computes distances
        differently for within-clip vs cross-clip stages.
    """
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
                # No face: reweight to body only
                dist_matrix[i, j] = (
                    (body_weight + FACE_WEIGHT) * body_dist[i, j] +
                    POSE_WEIGHT * pose_dist[i, j] +
                    MOTION_WEIGHT * motion_dist[i, j]
                ) / (body_weight + FACE_WEIGHT + POSE_WEIGHT + MOTION_WEIGHT)
    
    return dist_matrix

def k_reciprocal_rerank(dist_matrix):
    """
    Apply k-reciprocal re-ranking to improve distance matrix quality.
    
    k-reciprocal re-ranking is a technique that refines distance matrices by considering
    mutual nearest neighbors. If A is in B's k nearest neighbors AND B is in A's k
    nearest neighbors, they are considered reciprocal neighbors and their distance
    is adjusted using Jaccard similarity.
    
    Args:
        dist_matrix (numpy.ndarray): Original pairwise distance matrix [N x N]
        
    Returns:
        numpy.ndarray: Re-ranked distance matrix [N x N]
        
    Algorithm:
        1. For each point, finds k nearest neighbors (k = K_RECIPROCAL = 25)
        2. Identifies reciprocal neighbor pairs (mutual k-NN)
        3. For reciprocal pairs, computes Jaccard distance based on neighbor overlap
        4. Blends original distance with re-ranked distance:
           final = (1 - lambda) * original + lambda * reranked
           where lambda = LAMBDA_VALUE = 0.5
           
    Benefits:
        - Reduces false matches by considering context
        - Improves ranking quality for similar people
        - More robust to outliers
        
    Note:
        Currently not used. The adaptive clustering uses its own distance calculation.
    """
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
    """
    Cluster tracklets into global IDs using specified clustering method.
    
    This function performs clustering on tracklets using either similarity-based
    greedy clustering or DBSCAN. It assigns each tracklet a global identity ID.
    
    Args:
        dist_matrix (numpy.ndarray): Pairwise distance matrix [N x N]
        
    Returns:
        numpy.ndarray: Array of global IDs, one per tracklet [N]
        
    Clustering Methods:
        1. SIMILARITY: Greedy clustering
           - For each tracklet, finds best existing cluster
           - Merges if distance < SIMILARITY_THRESHOLD (0.8)
           - Otherwise creates new cluster
           
        2. DBSCAN: Density-based clustering
           - Uses DBSCAN with eps=DBSCAN_EPS (0.35), min_samples=1
           - Remaps noise points (-1) to new cluster IDs
           
    Note:
        Currently not used in main pipeline. The adaptive_cluster_tracklets()
        function is used instead, which implements the two-stage approach.
        This function is kept for backward compatibility and alternative methods.
    """
    print(f"📌 Clustering with SIMILARITY method...")
    
    if True:  # SIMILARITY method
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
    
def analyze_clip_characteristics(clip_tracks):
    """
    Automatically analyze video clip characteristics to determine optimal clustering threshold.
    
    This function analyzes various clip characteristics and computes an adaptive threshold
    for within-clip clustering. The threshold is adjusted based on scene complexity,
    tracking quality, face visibility, and person diversity.
    
    Args:
        clip_tracks (list): List of tracklet dictionaries from a single video clip,
            each containing 'start_frame', 'end_frame', 'body_emb', 'has_face', etc.
            
    Returns:
        tuple: (adaptive_threshold, analysis_dict)
            - adaptive_threshold (float): Computed threshold in range [0.01, 0.25]
            - analysis_dict (dict): Detailed analysis breakdown with:
                - 'max_people': Maximum simultaneous people detected
                - 'num_tracklets': Total number of tracklets
                - 'crowding': Crowding adjustment factor
                - 'tracking': Tracking quality adjustment factor
                - 'face': Face detection rate adjustment factor
                - 'diversity': Embedding diversity adjustment factor
                - 'final': Final computed threshold
                
    Analysis Factors:
        1. Crowding Score: More tracklets → more lenient threshold
           - 0-3 tracklets: +0.00 (simple scene)
           - 4-10 tracklets: +0.02
           - 11+ tracklets: +0.04
           
        2. Tracking Quality: Longer tracks → stricter threshold (better tracking)
           - >300 frames: +0.00 (good tracking)
           - 100-300 frames: +0.01
           - <100 frames: +0.03 (fragmented)
           
        3. Face Detection Rate: More faces → stricter threshold (better features)
           - >70%: +0.00 (good faces)
           - 40-70%: +0.01
           - <40%: +0.02 (poor faces)
           
        4. Embedding Diversity: More diverse people → more lenient threshold
           - Computes pairwise cosine distances between all tracklets
           - High diversity (avg distance > 0.35): negative adjustment
           - Low diversity (avg distance < 0.18): positive adjustment
           
    Threshold Calculation:
        base_threshold = WITHIN_CLIP_THRESHOLD (0.15)
        adaptive_threshold = base_threshold + crowding + tracking + face + diversity
        adaptive_threshold = clip(adaptive_threshold, 0.01, 0.25)  # Clamp to valid range
        
    Usage:
        Called automatically during Stage 1 (within-clip clustering) if
        USE_PER_CLIP_THRESHOLDS = True. Manual overrides can be specified in
        PER_CLIP_THRESHOLDS dictionary.
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

def calculate_hausdorff_distance(bboxes1, bboxes2):
    """
    Calculate Hausdorff distance between two sets of bounding boxes for spatial pattern analysis.
    
    Hausdorff distance measures how far two sets of points are from each other by finding
    the maximum distance from any point in one set to its nearest neighbor in the other set.
    This is useful for comparing spatial patterns of person movement across tracklets.
    
    Args:
        bboxes1 (list): First set of bounding boxes, each as [x1, y1, x2, y2]
        bboxes2 (list): Second set of bounding boxes, each as [x1, y1, x2, y2]
        
    Returns:
        float: Hausdorff distance in pixels (0 = identical patterns, larger = more different)
        float('inf'): If either set is empty
        
    Algorithm:
        1. Converts bounding boxes to center points (cx, cy)
        2. Computes pairwise Euclidean distances between all center points
        3. For each point in set1, finds minimum distance to any point in set2
        4. Takes maximum of these minimum distances → H(A, B)
        5. Repeats from set2 to set1 → H(B, A)
        6. Hausdorff distance = max(H(A, B), H(B, A))
        
    Usage:
        Used in Stage 2 (cross-clip merging) to override transitive blocking when
        two clusters have very similar spatial patterns (hausdorff_dist < 50 pixels).
        This helps merge tracklets that appear in different clips but have similar
        movement patterns, indicating they are the same person.
        
    Note:
        Lower values indicate more similar spatial patterns. A threshold of 50 pixels
        is used to determine if spatial patterns are similar enough to allow merging.
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
        - Image-only embeddings (body + face when both available)
    
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
        - Temporal overlap handling to prevent false merges
        - Best-of-cluster matching for merged clusters
        - Two-pass merging (face/no-face first, then face-to-face)
    """
    print("ADAPTIVE CLUSTERING (2-stage)")
    
    # Stage 1: Within-clip clustering (strict)
    print("  Stage 1: Within-clip clustering (strict)...")
    
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
        
        # Determine threshold (automatic analysis or manual override)
        if USE_PER_CLIP_THRESHOLDS:
            if clip_idx in PER_CLIP_THRESHOLDS:
                # Use manual override if specified
                clip_threshold = PER_CLIP_THRESHOLDS[clip_idx]
                print(f"    Clip {clip_idx}: {n} tracklets (manual threshold: {clip_threshold:.3f})")
            else:
                # Automatic analysis
                clip_threshold, analysis = analyze_clip_characteristics(clip_tracks)
                print(f"    Clip {clip_idx}: {n} tracklets (auto threshold: {clip_threshold:.3f})")
                print(f"      Analysis: crowd={analysis['crowding']:+.3f}, track={analysis['tracking']:+.3f}, face={analysis['face']:+.3f}, div={analysis['diversity']:+.3f}")
        else:
            clip_threshold = WITHIN_CLIP_THRESHOLD
            print(f"    Clip {clip_idx}: {n} tracklets")
        
        # Compute within-clip distance matrix (HIGH face weight)
        body_embs = np.array([t['body_emb'] for t in clip_tracks])
        face_embs = np.array([t['face_emb'] for t in clip_tracks])
        has_faces = np.array([t['has_face'] for t in clip_tracks])
        
        body_dist = cdist(body_embs, body_embs, metric='cosine')
        face_dist = cdist(face_embs, face_embs, metric='cosine')
        
        # High face weight within clip (same lighting/angle)
                    body_w = 1.0 - WITHIN_CLIP_FACE_WEIGHT
        dist_matrix = body_w * body_dist + WITHIN_CLIP_FACE_WEIGHT * face_dist
        
        # Overlap + Similarity logic - Replace penalty with proper threshold logic
        # This will be handled in the clustering loop below with adaptive thresholds
        
        # Cluster within clip (strict threshold)
        local_ids = [-1] * n
        next_cluster = 0
        
        for i in range(n):
            if local_ids[i] != -1:
                continue
            
            best_cluster = -1
            best_dist = float('inf')
            
            # Check ALL clusters and find the best one that doesn't have >50% overlap
            for cluster_id in range(next_cluster):
                members = [j for j in range(i) if local_ids[j] == cluster_id]
                if members:
                    # Check overlap with this cluster first (use first member as representative)
                    j = members[0]
                    track_i = clip_tracks[i]
                    track_j = clip_tracks[j]
                    
                    # Calculate temporal overlap
                    overlap_start = max(track_i['start_frame'], track_j['start_frame'])
                    overlap_end = min(track_i['end_frame'], track_j['end_frame'])
                    overlap_frames = max(0, overlap_end - overlap_start)
                    min_length = min(track_i['end_frame'] - track_i['start_frame'],
                                   track_j['end_frame'] - track_j['start_frame'])
                    overlap_ratio = overlap_frames / min_length if min_length > 0 else 0
                    
                    # If overlap > 50%, skip this cluster (different people)
                    if overlap_ratio > 0.5:
                        track_i_id = track_i.get('temp_id', track_i.get('track_id', i))
                        track_j_id = track_j.get('temp_id', track_j.get('track_id', j))
                        continue  # Skip this cluster, check next one
                    
                    # If we get here, overlap <= 50%, so check if this is the best cluster
                    avg_dist = np.mean([dist_matrix[i, j] for j in members])
                    if avg_dist < best_dist:
                        best_dist = avg_dist
                        best_cluster = cluster_id
            
            # Apply overlap + similarity logic instead of simple threshold
            should_merge = False
            if best_cluster != -1:
                # Find a representative tracklet from the best cluster to check overlap and similarity
                cluster_members = [j for j in range(i) if local_ids[j] == best_cluster]
                if cluster_members:
                    # Use the first member as representative
                    j = cluster_members[0]
                    track_i = clip_tracks[i]
                    track_j = clip_tracks[j]
                    
                    # Calculate temporal overlap (should already be <= 0.5 from above check)
                    overlap_start = max(track_i['start_frame'], track_j['start_frame'])
                    overlap_end = min(track_i['end_frame'], track_j['end_frame'])
                    overlap_frames = max(0, overlap_end - overlap_start)
                    min_length = min(track_i['end_frame'] - track_i['start_frame'],
                                   track_j['end_frame'] - track_j['start_frame'])
                    overlap_ratio = overlap_frames / min_length if min_length > 0 else 0
                    
                    # Calculate actual body similarity from body distance matrix
                    body_similarity = 1 - body_dist[i, j]  # Use body distance directly
                    # Face presence adjustment (mismatch lowers expected similarity)
                    has_face_i = track_i.get('has_face', False)
                    has_face_j = track_j.get('has_face', False)
                    
                    # Get tracklet IDs for debug output
                    track_i_id = track_i.get('temp_id', track_i.get('track_id', i))
                    track_j_id = track_j.get('temp_id', track_j.get('track_id', j))
                    
                    # Compute containment: if shorter span is fully overlapped
                    span_i = track_i['end_frame'] - track_i['start_frame']
                    span_j = track_j['end_frame'] - track_j['start_frame']
                    shorter_span = min(span_i, span_j)
                    is_contained = (overlap_frames >= shorter_span and shorter_span > 0)
                    longer_span = max(span_i, span_j)
                    duration_ratio = (longer_span / shorter_span) if shorter_span > 0 else float('inf')
                     
                    # Apply overlap + similarity logic (overlap_ratio should be <= 0.5 here)
                    if overlap_ratio > 0.4:  # Medium overlap (raised from 30% to 40%)
                        required_similarity = 0.80  # Use within-clip similarity rule
                        if (not has_face_i) or (not has_face_j):
                            required_similarity = max(required_similarity, 0.80)
                        # Face-face: slightly stricter for high overlap
                        if has_face_i and has_face_j:
                            required_similarity = max(required_similarity, 0.82)
                        if is_contained:
                            required_similarity += 0.05
                        if overlap_ratio > 0 and duration_ratio >= 2.0:
                            required_similarity += 0.03
                        required_similarity = max(0.65, required_similarity)
                        if body_similarity >= required_similarity:
                            should_merge = True
                    elif overlap_ratio > 0.1:  # Low overlap
                        required_similarity = 0.70  # More lenient
                        # No-face floor
                        if (not has_face_i) or (not has_face_j):
                            required_similarity = max(required_similarity, 0.87)
                        # Face-face: slightly stricter for medium overlap
                        if has_face_i and has_face_j:
                            required_similarity = max(required_similarity, 0.82)
                        # Containment and duration tightening
                        if is_contained:
                            required_similarity += 0.05
                        if overlap_ratio > 0 and duration_ratio >= 2.0:
                            required_similarity += 0.03
                        required_similarity = max(0.65, required_similarity)
                        if body_similarity >= required_similarity:
                            should_merge = True
                    else:  # Very low overlap
                        required_similarity = 0.70  # Very lenient threshold
                        # For zero/very-low overlap, require higher sim when weak evidence
                        if (not has_face_i) or (not has_face_j) or (duration_ratio >= 2.0):
                            required_similarity = max(required_similarity, 0.87 if ((not has_face_i) or (not has_face_j)) else 0.80)
                        # Face-face at very low overlap: be stricter
                        if has_face_i and has_face_j:
                            required_similarity = max(required_similarity, 0.85)
                        # HARD GUARD: if very-low overlap and strong duration imbalance and any no-face → require 0.90
                        if overlap_ratio <= 0.1 and duration_ratio >= 2.0 and ((not has_face_i) or (not has_face_j)):
                            required_similarity = max(required_similarity, 0.90)
                        if is_contained:
                            required_similarity += 0.05
                        required_similarity = max(0.65, required_similarity)
                        if body_similarity >= required_similarity:
                            should_merge = True
                    
            
            # Cohesion check: require similarity against ALL members in best_cluster
            if should_merge and best_cluster != -1:
                all_ok = True
                for j2 in cluster_members:
                    track_j2 = clip_tracks[j2]
                    # Recompute pairwise stats per member
                    overlap_start2 = max(track_i['start_frame'], track_j2['start_frame'])
                    overlap_end2 = min(track_i['end_frame'], track_j2['end_frame'])
                    overlap_frames2 = max(0, overlap_end2 - overlap_start2)
                    min_length2 = min(
                        track_i['end_frame'] - track_i['start_frame'],
                        track_j2['end_frame'] - track_j2['start_frame']
                    )
                    overlap_ratio2 = (overlap_frames2 / min_length2) if min_length2 > 0 else 0
                    span_i2 = track_i['end_frame'] - track_i['start_frame']
                    span_j2 = track_j2['end_frame'] - track_j2['start_frame']
                    shorter_span2 = min(span_i2, span_j2)
                    longer_span2 = max(span_i2, span_j2)
                    is_contained2 = (overlap_frames2 >= shorter_span2 and shorter_span2 > 0)
                    duration_ratio2 = (longer_span2 / shorter_span2) if shorter_span2 > 0 else float('inf')
                    has_face_j2 = track_j2.get('has_face', False)
                    body_similarity2 = 1 - body_dist[i, j2]
                    # Derive required similarity per member using same bins
                    req2 = 0.70
                    if overlap_ratio2 > 0.5:
                        req2 = 0.86 if overlap_ratio2 > 0.9 else 0.80
                    elif overlap_ratio2 > 0.4:
                        req2 = 0.80
                    elif overlap_ratio2 > 0.1:
                        req2 = 0.70
                    else:
                        req2 = 0.70
                    # No-face floors and very-low face-face strictness
                    if (not has_face_i) or (not has_face_j2) or (overlap_ratio2 == 0 and duration_ratio2 >= 2.0):
                        req2 = max(req2, 0.80)
                    if overlap_ratio2 <= 0.1 and has_face_i and has_face_j2:
                        req2 = max(req2, 0.85)
                    # Containment and duration bumps
                    if is_contained2:
                        req2 += 0.05
                    if overlap_ratio2 > 0 and duration_ratio2 >= 2.0:
                        req2 += 0.03
                    req2 = max(0.65, req2)
                    if body_similarity2 < req2:
                        all_ok = False
                        break
                if not all_ok:
                    should_merge = False

            if should_merge:
                local_ids[i] = best_cluster
            else:
                local_ids[i] = next_cluster
                next_cluster += 1
        
        # Assign global temporary IDs
        for i, track in enumerate(clip_tracks):
            track['temp_global_id'] = next_local_id + local_ids[i]
        
        next_local_id += next_cluster
        print(f"      → {next_cluster} local clusters")
    
    # Stage 2: Cross-clip merging (IMAGE-ONLY: body + face when both have faces)
    print(f"  Stage 2: Cross-clip merging (IMAGE-ONLY: body + face)...")
    print(f"    Total intermediate clusters: {next_local_id}")
    
    # Extract body (image) embeddings only
    body_embs = np.array([t['body_emb'] for t in tracklets])
    
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
        # Weighted averaging: weight by tracklet length (frames) to preserve information from longer tracklets
        # Longer tracklets have more information and should contribute more
        tracklet_indices = np.where(mask)[0]
        weights = np.array([tracklets[idx]['end_frame'] - tracklets[idx]['start_frame'] 
                           for idx in tracklet_indices], dtype=np.float32)
        
        # Normalize weights (sum to 1)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(tracklet_indices)) / len(tracklet_indices)
        
        # Weighted average (preserves information from longer tracklets)
        cluster_body_embs[i] = np.average(body_embs[mask], axis=0, weights=weights)
        cluster_face_embs[i] = np.average(face_embs[mask], axis=0, weights=weights)
        cluster_has_faces[i] = np.any(has_faces[mask])
    
    # Compute IMAGE-ONLY distance (body + face when both have faces)
    print("    Computing image-only distances (body + face)...")
    
    if USE_CHAMFER_DISTANCE:
        print("    Using Chamfer distance for cross-clip matching...")
        
        # Use Chamfer distance for cross-clip matching
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
        
        # Add face component if both clusters have faces (matches previous behavior)
        if np.any(cluster_has_faces):
            face_dist = cdist(cluster_face_embs, cluster_face_embs, metric='cosine')
            
            # Blend Chamfer distance with face distance (when both have faces)
            for i in range(n_temp):
                for j in range(n_temp):
                    if cluster_has_faces[i] and cluster_has_faces[j]:
                        # Weighted combination: 80% Chamfer + 20% face
                        dist_matrix[i, j] = 0.8 * dist_matrix[i, j] + 0.2 * face_dist[i, j]
    
            # NaN guard for face distance
            face_dist = np.nan_to_num(face_dist, nan=1.0, posinf=1.0, neginf=1.0)
        
        # NaN guard: replace any NaN/inf with 1.0 (maximum distance)
        dist_matrix = np.nan_to_num(dist_matrix, nan=1.0, posinf=1.0, neginf=1.0)
    
    else:
        # Image-only cosine distance computation (body + face when available)
        # OPTIMIZED: Pre-compute cluster tracklet mappings and vectorize distance calculations
        # NEW: Best-of-cluster matching - compare each cluster to the BEST matching tracklet in the other cluster
        # This prevents a bad tracklet from dominating the weighted embedding
        body_w = 1.0 - CROSS_CLIP_FACE_WEIGHT
        
        # OPTIMIZATION 1: Pre-compute cluster tracklet mappings (avoid repeated list comprehensions)
        cluster_tracklets = {}
        for i, tid in enumerate(unique_temp_ids):
            cluster_tracklets[i] = [t for t in tracklets if t['temp_global_id'] == tid]
        
        # OPTIMIZATION 2: Pre-extract and normalize embeddings for all clusters
        cluster_body_embs_list = {}
        cluster_face_embs_list = {}
        cluster_has_faces_list = {}
        for i in range(n_temp):
            tracklets_i = cluster_tracklets[i]
            cluster_body_embs_list[i] = np.array([np.nan_to_num(t['body_emb'], nan=0.0) for t in tracklets_i])
            cluster_face_embs_list[i] = np.array([np.nan_to_num(t['face_emb'], nan=0.0) for t in tracklets_i])
            cluster_has_faces_list[i] = np.array([t.get('has_face', False) for t in tracklets_i])
        
        dist_matrix = np.zeros((n_temp, n_temp))
        
        # OPTIMIZATION 3: Vectorize best-of-cluster distance computation
        for i in range(n_temp):
            for j in range(i + 1, n_temp):  # Only compute upper triangle
                body_embs_i = cluster_body_embs_list[i]
                body_embs_j = cluster_body_embs_list[j]
                face_embs_i = cluster_face_embs_list[i]
                face_embs_j = cluster_face_embs_list[j]
                has_faces_i = cluster_has_faces_list[i]
                has_faces_j = cluster_has_faces_list[j]
                
                # Vectorized body distance: compute all pairwise distances at once
                if len(body_embs_i) > 0 and len(body_embs_j) > 0:
                    body_dist_matrix = cdist(body_embs_i, body_embs_j, metric='cosine')
                    
                    # Vectorized face distance: compute only for face-face pairs
                    face_dist_matrix = np.ones_like(body_dist_matrix)
                    # Find all face-face pairs
                    face_pair_mask = np.outer(has_faces_i, has_faces_j)
                    if face_pair_mask.any():
                        face_indices_i = np.where(has_faces_i)[0]
                        face_indices_j = np.where(has_faces_j)[0]
                        if len(face_indices_i) > 0 and len(face_indices_j) > 0:
                            face_embs_i_filtered = face_embs_i[face_indices_i]
                            face_embs_j_filtered = face_embs_j[face_indices_j]
                            # Compute all pairwise face distances
                            face_dist_submatrix = cdist(face_embs_i_filtered, face_embs_j_filtered, metric='cosine')
                            # Check for zero norms (invalid faces)
                            face_norms_i = np.linalg.norm(face_embs_i_filtered, axis=1)
                            face_norms_j = np.linalg.norm(face_embs_j_filtered, axis=1)
                            valid_i = face_norms_i > 0
                            valid_j = face_norms_j > 0
                            # Map back to full matrix
                            for idx_i, i_idx in enumerate(face_indices_i):
                                for idx_j, j_idx in enumerate(face_indices_j):
                                    if valid_i[idx_i] and valid_j[idx_j]:
                                        face_dist_matrix[i_idx, j_idx] = face_dist_submatrix[idx_i, idx_j]
                    
                    # Compute weighted distances: vectorized
                    weighted_dist_matrix = np.where(
                        (np.outer(has_faces_i, has_faces_j)) & (face_dist_matrix < 1.0),
                        body_w * body_dist_matrix + CROSS_CLIP_FACE_WEIGHT * face_dist_matrix,
                        body_dist_matrix
                    )
                    
                    # Find best (minimum) distance
                    best_distance = np.min(weighted_dist_matrix)
                else:
                    best_distance = 1.0
                
                dist_matrix[i, j] = best_distance
                dist_matrix[j, i] = best_distance  # Symmetric
        
        # NaN guard: replace any NaN/inf with 1.0 (maximum distance)
        dist_matrix = np.nan_to_num(dist_matrix, nan=1.0, posinf=1.0, neginf=1.0)
    
    print("    Image-only distances computed")
    
    # OPTIMIZATION 4: Build same-clip overlap block matrix (pre-compute clip sets)
    same_clip_overlap_block = np.zeros((n_temp, n_temp), dtype=bool)
    
    # Pre-compute clip sets for each cluster
    cluster_clips = {}
    for i in range(n_temp):
        tracklets_i = cluster_tracklets[i]
        cluster_clips[i] = set(t['clip_idx'] for t in tracklets_i)
    
    # Check for shared clips (vectorized check)
    for i in range(n_temp):
        for j in range(i + 1, n_temp):
            if cluster_clips[i] & cluster_clips[j]:  # Shared clips
                same_clip_overlap_block[i, j] = True
                same_clip_overlap_block[j, i] = True
    
    # Greedy merging with ADAPTIVE thresholds and transitive blocking
    final_ids = np.arange(n_temp)
    base_threshold = CROSS_CLIP_THRESHOLD  # Use as baseline (0.42)
    
    # Track confirmed face-to-face matches: (global_id, clip_idx) -> True
    # When a face-to-face cross-clip merge happens, this prevents other same-clip tracklets from merging
    confirmed_face_to_face_matches = {}  # {(final_cluster_id, clip_idx): True}
    
    # Track target cluster merges: (target_cluster_id, target_clip_idx) -> source_clip_idx
    # Prevents multiple clusters from the same source clip from merging with the same target cluster
    target_cluster_to_source_clip = {}  # {(target_cluster_id, target_clip_idx): source_clip_idx}
    
    
    # NEW: Collect all merge candidates first, then sort by priority (face-to-face first, then higher similarity)
    merge_candidates = []
    
    for i in range(n_temp):
        if final_ids[i] != i:
            continue
        
        for j in range(i + 1, n_temp):
            if final_ids[j] != j:
                continue
            
            temp_id_i = unique_temp_ids[i]
            temp_id_j = unique_temp_ids[j]
            
            
            # Check direct block (same clip) - NO EXCEPTIONS
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
                # OPTIMIZATION: Use pre-computed cluster tracklets
                tracklets_i = cluster_tracklets[i]
                tracklets_j = cluster_tracklets[j]
                
                if tracklets_i and tracklets_j:
                    track_i = tracklets_i[0]
                    track_j = tracklets_j[0]
                    
                    # Calculate Hausdorff distance for spatial pattern analysis
                    hausdorff_dist = calculate_hausdorff_distance(track_i['bboxes'], track_j['bboxes'])
                    
                    # If Hausdorff distance is very good (< 50 pixels), override transitive blocking
                    if hausdorff_dist < 50:
                        merge_blocked = False
            
            if merge_blocked:
                continue
            
            # NEW: ADAPTIVE threshold based on pair characteristics
            # OPTIMIZATION: Use pre-computed cluster tracklets
            tracklets_i = cluster_tracklets[i]
            tracklets_j = cluster_tracklets[j]
            
            # Factor 1: Face availability (both have faces = more reliable)
            both_have_faces = cluster_has_faces[i] and cluster_has_faces[j]
            one_has_face = cluster_has_faces[i] or cluster_has_faces[j]
            neither_has_face = not cluster_has_faces[i] and not cluster_has_faces[j]
            
            # Factor 2: Tracklet length (longer = more reliable)
            total_frames_i = sum(t['end_frame'] - t['start_frame'] for t in tracklets_i)
            total_frames_j = sum(t['end_frame'] - t['start_frame'] for t in tracklets_j)
            avg_length = (total_frames_i + total_frames_j) / 2
            
            # IMPORTANT: If clusters have merged (final_ids changed), get ALL tracklets from merged cluster
            # This ensures merged clusters are correctly represented
            merged_cluster_i = final_ids[i]
            merged_cluster_j = final_ids[j]
            
            # Get ALL tracklets that belong to the merged clusters (if merged)
            if merged_cluster_i != i:
                # Cluster i has merged with others - get all tracklets in the merged cluster
                merged_tracklets_i = [t for t in tracklets 
                                     if t['temp_global_id'] in unique_temp_ids[merged_cluster_i == final_ids]]
            else:
                merged_tracklets_i = tracklets_i
            
            if merged_cluster_j != j:
                # Cluster j has merged with others - get all tracklets in the merged cluster
                merged_tracklets_j = [t for t in tracklets 
                                     if t['temp_global_id'] in unique_temp_ids[merged_cluster_j == final_ids]]
            else:
                merged_tracklets_j = tracklets_j
            
            # OPTIMIZATION: Vectorized body similarity calculation
            # Calculate body similarity using best-of-cluster (best distance among all tracklet pairs)
            # This ensures merged clusters use their best representation
            if len(merged_tracklets_i) > 0 and len(merged_tracklets_j) > 0:
                body_embs_i = np.array([t['body_emb'] for t in merged_tracklets_i])
                body_embs_j = np.array([t['body_emb'] for t in merged_tracklets_j])
                body_dist_matrix = cdist(body_embs_i, body_embs_j, metric='cosine')
                best_body_dist = np.min(body_dist_matrix)
            else:
                best_body_dist = 1.0
            
            body_sim_check = 1 - best_body_dist
            
            # Factor 3: Determine adaptive threshold (with face/no-face adjustment)
            if both_have_faces and avg_length > 500:
                # Very high confidence: lenient for face-to-face with long tracklets
                # Make threshold more lenient: 0.43 -> 0.44 to allow 0.4309 distance
                adaptive_threshold = base_threshold - 0.02  # 0.44 (was 0.43)
                confidence = "HIGH"
            elif both_have_faces and avg_length > 300:
                # High confidence: baseline threshold
                adaptive_threshold = base_threshold  # 0.46
                confidence = "MED"
            elif avg_length > 250:
                # Medium confidence: slightly lenient (ID 4, 5 type - partial matches)
                adaptive_threshold = base_threshold + 0.03  # 0.49
                confidence = "MED+"
            else:
                # Low confidence: very lenient (ID 0, 1 fragments)
                adaptive_threshold = base_threshold + 0.05  # 0.51
                confidence = "LOW"
            
            # NEW: Lower threshold for face/no-face cross-clip pairs (one has face, other doesn't)
            # Cross-clip merging should be easier when one has no face (helps no-face tracklets merge)
            if one_has_face and not both_have_faces:
                adaptive_threshold -= 0.08  # Make it easier to merge when one has face (cross-clip only)
                confidence += " [FACE/NO-FACE - EASIER]"
            
            # Effective threshold is the adaptive threshold
            effective_threshold = adaptive_threshold
            
            # Calculate distance (standard approach)
                    current_distance = dist_matrix[i, j]
            
            # OPTIMIZATION: Use pre-computed cluster tracklets
            # (tracklets_i and tracklets_j already defined above)
            
            final_distance = current_distance
            
            # Calculate body similarity for threshold check
            # Use weighted averaged embeddings (consistent with main distance calculation)
            # Weight by tracklet length to preserve information from longer tracklets
            weights_i = np.array([t['end_frame'] - t['start_frame'] for t in tracklets_i], dtype=np.float32)
            weights_j = np.array([t['end_frame'] - t['start_frame'] for t in tracklets_j], dtype=np.float32)
            if weights_i.sum() > 0:
                weights_i = weights_i / weights_i.sum()
            else:
                weights_i = np.ones(len(tracklets_i)) / len(tracklets_i)
            if weights_j.sum() > 0:
                weights_j = weights_j / weights_j.sum()
            else:
                weights_j = np.ones(len(tracklets_j)) / len(tracklets_j)
            
            body_embs_i = np.array([t['body_emb'] for t in tracklets_i])
            body_embs_j = np.array([t['body_emb'] for t in tracklets_j])
            avg_body_i = np.average(body_embs_i, axis=0, weights=weights_i)
            avg_body_j = np.average(body_embs_j, axis=0, weights=weights_j)
            body_dist_for_check = cdist([avg_body_i], [avg_body_j], metric='cosine')[0][0]
            body_sim_for_check = 1 - body_dist_for_check
            
            # Body similarity requirement (lowered for face/no-face pairs)
            body_sim_threshold = 0.7
            if one_has_face and not both_have_faces:
                body_sim_threshold = 0.65  # Lower threshold for face/no-face pairs
            
            # Check distance against effective threshold AND body similarity
            clip_i = tracklets_i[0]['clip_idx']
            clip_j = tracklets_j[0]['clip_idx']
            
            if final_distance < effective_threshold and body_sim_for_check >= body_sim_threshold:
                # Calculate face similarity for face-to-face candidates
                face_sim_for_candidate = 0.0
                if both_have_faces:
                    # Use best-of-cluster matching for face similarity too
                    best_face_sim = 0.0
                    for t_i in tracklets_i:
                        for t_j in tracklets_j:
                            if t_i.get('has_face', False) and t_j.get('has_face', False):
                                face_emb_i = np.nan_to_num(t_i['face_emb'], nan=0.0)
                                face_emb_j = np.nan_to_num(t_j['face_emb'], nan=0.0)
                                face_i_norm = np.linalg.norm(face_emb_i)
                                face_j_norm = np.linalg.norm(face_emb_j)
                                if face_i_norm > 0 and face_j_norm > 0:
                                    face_dist_ij = cdist([face_emb_i], [face_emb_j], metric='cosine')[0, 0]
                                    face_sim_ij = 1 - face_dist_ij
                                    best_face_sim = max(best_face_sim, face_sim_ij)
                    face_sim_for_candidate = best_face_sim
                
                # Collect candidate with metadata for sorting
                merge_candidates.append({
                    'i': i,
                    'j': j,
                    'temp_id_i': temp_id_i,
                    'temp_id_j': temp_id_j,
                    'distance': final_distance,
                    'body_sim': body_sim_for_check,
                    'face_sim': face_sim_for_candidate,
                    'both_have_faces': both_have_faces,
                    'clip_i': clip_i,
                    'clip_j': clip_j,
                    'effective_threshold': effective_threshold
                })
            else:
                # Near-miss with effective threshold
                if final_distance < effective_threshold + 0.10:
                    clip_i = tracklets_i[0]['clip_idx']
                    clip_j = tracklets_j[0]['clip_idx']
    
    # Two-pass approach: (1) Face-to-no-face first (locked after merge), (2) Face-to-face second
    # Pass 1: Face-to-no-face matches (sorted by body_sim, higher = better) - clusters locked after merge
    # Pass 2: Face-to-face matches (sorted by distance, lower = better) - only process unlocked clusters
    print(f"    Found {len(merge_candidates)} merge candidates")
    
    # Separate candidates into two groups
    face_to_no_face_candidates = [c for c in merge_candidates if not c['both_have_faces']]
    face_to_face_candidates = [c for c in merge_candidates if c['both_have_faces']]
    
    # Sort each group appropriately
    face_to_no_face_candidates.sort(key=lambda x: -x['body_sim'])  # Higher body_sim first
    face_to_face_candidates.sort(key=lambda x: x['distance'])  # Lower distance first
    
    print(f"    PASS 1: Processing {len(face_to_no_face_candidates)} face-to-no-face matches (sorted by body_sim)")
    
    # Track which clusters were merged in Pass 1 (locked)
    locked_clusters = set()
    
    # PASS 1: Process face-to-no-face matches
    for candidate in face_to_no_face_candidates:
        i = candidate['i']
        j = candidate['j']
        temp_id_i = candidate['temp_id_i']
        temp_id_j = candidate['temp_id_j']
        final_distance = candidate['distance']
        body_sim_for_check = candidate['body_sim']
        clip_i = candidate['clip_i']
        clip_j = candidate['clip_j']
        
        # Skip if already merged
        if final_ids[i] != i or final_ids[j] != j:
                    continue
                
        # CRITICAL: Check same-clip overlap block - NEVER merge clusters from same clip!
        if same_clip_overlap_block[i, j]:
            continue
                
        # Check if either cluster is already locked from Pass 1
        if i in locked_clusters or j in locked_clusters:
            continue
        
        # Check if this is a face-to-face match (should be False in Pass 1)
        is_face_to_face_match = candidate['both_have_faces']
        
        # Before merging, check if target cluster already has confirmed face-to-face match from same clip
        target_cluster_id = final_ids[i]
        
        # Check if there's already a confirmed face-to-face match for this cluster in clip_j
        if (target_cluster_id, clip_j) in confirmed_face_to_face_matches:
            continue
        
        # Check same for clip_i (in case j already has confirmed match)
        source_cluster_id = final_ids[j]
        if (source_cluster_id, clip_i) in confirmed_face_to_face_matches:
            continue
        
        # CRITICAL: Prevent multiple clusters from the same source clip from merging with the same target cluster
        if (target_cluster_id, clip_j) in target_cluster_to_source_clip:
            existing_source_clip = target_cluster_to_source_clip[(target_cluster_id, clip_j)]
            if existing_source_clip == clip_i:
                continue
        
                    # Merge j into i
        final_ids[j] = final_ids[i]
        
        # Mark clusters as locked (merged in Pass 1)
        locked_clusters.add(i)
        locked_clusters.add(j)
        # Also lock any clusters that are now part of this merged cluster
        merged_cluster_id = final_ids[j]
        for k in range(len(final_ids)):
            if final_ids[k] == merged_cluster_id:
                locked_clusters.add(k)
        
        # Record that this target cluster (in clip_j) has been merged with a cluster from clip_i
        target_cluster_to_source_clip[(target_cluster_id, clip_j)] = clip_i
        
    
    print(f"    PASS 1 complete: {len(locked_clusters)} clusters locked")
    print(f"    PASS 2: Processing {len(face_to_face_candidates)} face-to-face matches (recomputing distances with merged clusters)")
    
    # FIRST: Recompute distances for ALL face-to-face candidates using merged clusters
    # This ensures we compare the ACTUAL best distances after Pass 1 merges
    body_w = 1.0 - CROSS_CLIP_FACE_WEIGHT
    
    for candidate in face_to_face_candidates:
        i = candidate['i']
        j = candidate['j']
        
        # Skip if already merged
        if final_ids[i] != i or final_ids[j] != j:
            continue
        
        # Check if clusters have merged with others (from Pass 1)
        merged_cluster_i = final_ids[i]
        merged_cluster_j = final_ids[j]
        clusters_in_i = np.sum(final_ids == merged_cluster_i)
        clusters_in_j = np.sum(final_ids == merged_cluster_j)
        has_merged_i = (clusters_in_i > 1)
        has_merged_j = (clusters_in_j > 1)
        
        if has_merged_i or has_merged_j:
            # Recompute distance using ALL tracklets from merged clusters
            merged_temp_ids_i = unique_temp_ids[final_ids == merged_cluster_i]
            merged_temp_ids_j = unique_temp_ids[final_ids == merged_cluster_j]
            merged_tracklets_i = [t for t in tracklets if t['temp_global_id'] in merged_temp_ids_i]
            merged_tracklets_j = [t for t in tracklets if t['temp_global_id'] in merged_temp_ids_j]
            
            original_distance = candidate['distance']
            best_distance_recomputed = float('inf')
            best_body_sim = 0.0
            best_face_sim = 0.0
            best_pair_info = None
            
            # For face-to-face candidates, prioritize pairs where BOTH have faces
            # This ensures we use face information when available
            both_have_faces_candidates = []
            other_candidates = []
            
            for t_i in merged_tracklets_i:
                for t_j in merged_tracklets_j:
                    body_emb_i = np.nan_to_num(t_i['body_emb'], nan=0.0)
                    body_emb_j = np.nan_to_num(t_j['body_emb'], nan=0.0)
                    body_dist_ij = cdist([body_emb_i], [body_emb_j], metric='cosine')[0, 0]
                    body_sim_ij = 1 - body_dist_ij
                    
                    face_dist_ij = 1.0
                    face_sim_ij = 0.0
                    both_have_faces = t_i.get('has_face', False) and t_j.get('has_face', False)
                    
                    if both_have_faces:
                        face_emb_i = np.nan_to_num(t_i['face_emb'], nan=0.0)
                        face_emb_j = np.nan_to_num(t_j['face_emb'], nan=0.0)
                        face_i_norm = np.linalg.norm(face_emb_i)
                        face_j_norm = np.linalg.norm(face_emb_j)
                        if face_i_norm > 0 and face_j_norm > 0:
                            face_dist_ij = cdist([face_emb_i], [face_emb_j], metric='cosine')[0, 0]
                            face_sim_ij = 1 - face_dist_ij
            else:
                            face_dist_ij = 1.0
                            face_sim_ij = 0.0
                    else:
                        face_dist_ij = 1.0
                        face_sim_ij = 0.0
                    
                    if both_have_faces and face_dist_ij < 1.0:
                        weighted_dist = body_w * body_dist_ij + CROSS_CLIP_FACE_WEIGHT * face_dist_ij
                    else:
                        weighted_dist = body_dist_ij

            pair_info = {
                'dist': weighted_dist,
                'body_sim': body_sim_ij,
                'face_sim': face_sim_ij,
                'frames_i': (t_i.get('start_frame', -1), t_i.get('end_frame', -1)),
                'frames_j': (t_j.get('start_frame', -1), t_j.get('end_frame', -1))
            }
            if both_have_faces:
                both_have_faces_candidates.append(pair_info)
            else:
                other_candidates.append(pair_info)
            
            # First, try to find best from face-to-face pairs
            if both_have_faces_candidates:
                best_pair = min(both_have_faces_candidates, key=lambda x: x['dist'])
                best_distance_recomputed = best_pair['dist']
                best_body_sim = best_pair['body_sim']
                best_face_sim = best_pair['face_sim']
                best_pair_info = best_pair['frames_i'] + best_pair['frames_j']
            # Fall back to other pairs only if no face-to-face pairs found
            elif other_candidates:
                best_pair = min(other_candidates, key=lambda x: x['dist'])
                best_distance_recomputed = best_pair['dist']
                best_body_sim = best_pair['body_sim']
                best_face_sim = best_pair['face_sim']
                best_pair_info = best_pair['frames_i'] + best_pair['frames_j']
            
            # Update candidate distance with recomputed value
            # Always print recomputation details for debugging (even if small change)
            candidate['distance'] = best_distance_recomputed
            candidate['body_sim'] = best_body_sim
            candidate['face_sim'] = best_face_sim
    
    # PASS 2: Best-match logic - ensure each target cluster (j) gets merged with its BEST (lowest distance) source candidate
    # Now using recomputed distances that account for merged clusters
    target_cluster_j_to_best_candidate = {}  # {cluster_j: best_candidate_for_j}
    
    for candidate in face_to_face_candidates:
        i = candidate['i']
        j = candidate['j']
        
        # Skip if already merged
        if final_ids[i] != i or final_ids[j] != j:
                    continue
                
        # CRITICAL: Check same-clip overlap block - NEVER merge clusters from same clip!
        if same_clip_overlap_block[i, j]:
            continue
        
        # In Pass 2, allow locked clusters to merge with NEW unlocked clusters
        # Only skip if BOTH are locked (both already merged in Pass 1)
        both_locked = (i in locked_clusters and j in locked_clusters)
        if both_locked:
            continue
        
        # Group by target cluster j (the one from Clip 2 that will be merged into source i)
        # We want the BEST source (i) for each target (j), using recomputed distances
        if j not in target_cluster_j_to_best_candidate:
            target_cluster_j_to_best_candidate[j] = candidate
        else:
            existing_candidate = target_cluster_j_to_best_candidate[j]
            existing_i = existing_candidate['i']
            
            # Select best candidate: if both belong to same Global ID, prefer lower index; otherwise prefer lower distance
            if final_ids[i] == final_ids[existing_i]:
                if i < existing_i:
                    target_cluster_j_to_best_candidate[j] = candidate
            else:
                if candidate['distance'] < existing_candidate['distance']:
                    target_cluster_j_to_best_candidate[j] = candidate
    
    # Process only the best candidates for each target cluster j
    best_candidates = list(target_cluster_j_to_best_candidate.values())
    best_candidates.sort(key=lambda x: x['distance'])  # Sort by distance (lower = better)
    
    print(f"    PASS 2 (best-match): Processing {len(best_candidates)} best face-to-face matches (one per target cluster)")
    
    # PASS 2: Process face-to-face matches (only best matches, only unlocked clusters)
    for candidate in best_candidates:
        i = candidate['i']
        j = candidate['j']
        temp_id_i = candidate['temp_id_i']
        temp_id_j = candidate['temp_id_j']
        original_distance = candidate['distance']  # Distance before Pass 1 merges
        body_sim_for_check = candidate['body_sim']
        both_have_faces = candidate['both_have_faces']
        clip_i = candidate['clip_i']
        clip_j = candidate['clip_j']
        
        # Skip if already merged
        if final_ids[i] != i or final_ids[j] != j:
            continue
        
        # RECOMPUTE distance using merged clusters (if clusters merged in Pass 1)
        # Get ALL tracklets in merged clusters to find best match among all tracklet pairs
        merged_cluster_i = final_ids[i]
        merged_cluster_j = final_ids[j]
        
        # Check if clusters have merged with others: count how many clusters share the same final_id
        clusters_in_i = np.sum(final_ids == merged_cluster_i)
        clusters_in_j = np.sum(final_ids == merged_cluster_j)
        has_merged_i = (clusters_in_i > 1)  # More than 1 cluster merged together
        has_merged_j = (clusters_in_j > 1)
        
        if has_merged_i or has_merged_j:
            # Get all clusters that merged into the same final cluster
            merged_temp_ids_i = unique_temp_ids[final_ids == merged_cluster_i]
            merged_temp_ids_j = unique_temp_ids[final_ids == merged_cluster_j]
            
            # Get all tracklets belonging to merged clusters
            merged_tracklets_i = [t for t in tracklets if t['temp_global_id'] in merged_temp_ids_i]
            merged_tracklets_j = [t for t in tracklets if t['temp_global_id'] in merged_temp_ids_j]
            
            # Recompute best distance among ALL tracklet pairs in merged clusters
            body_w = 1.0 - CROSS_CLIP_FACE_WEIGHT
            best_distance_merged = float('inf')
            best_body_dist_merged = float('inf')
            
            for t_i in merged_tracklets_i:
                for t_j in merged_tracklets_j:
                    # Calculate body distance
                    body_emb_i = np.nan_to_num(t_i['body_emb'], nan=0.0)
                    body_emb_j = np.nan_to_num(t_j['body_emb'], nan=0.0)
                    body_dist_ij = cdist([body_emb_i], [body_emb_j], metric='cosine')[0, 0]
                    best_body_dist_merged = min(best_body_dist_merged, body_dist_ij)
                    
                    # Calculate face distance if both have faces
                    face_dist_ij = 1.0
                    if t_i.get('has_face', False) and t_j.get('has_face', False):
                        face_emb_i = np.nan_to_num(t_i['face_emb'], nan=0.0)
                        face_emb_j = np.nan_to_num(t_j['face_emb'], nan=0.0)
                        face_i_norm = np.linalg.norm(face_emb_i)
                        face_j_norm = np.linalg.norm(face_emb_j)
                        if face_i_norm > 0 and face_j_norm > 0:
                            face_dist_ij = cdist([face_emb_i], [face_emb_j], metric='cosine')[0, 0]
                else:
                            face_dist_ij = 1.0
            else:
                        face_dist_ij = 1.0
                    
                    # Weighted distance
                    if t_i.get('has_face', False) and t_j.get('has_face', False) and face_dist_ij < 1.0:
                        weighted_dist = body_w * body_dist_ij + CROSS_CLIP_FACE_WEIGHT * face_dist_ij
                    else:
                        weighted_dist = body_dist_ij
                    
                    best_distance_merged = min(best_distance_merged, weighted_dist)
            
            # Use recomputed distance and body sim
            final_distance = best_distance_merged
            body_sim_for_check = 1 - best_body_dist_merged
        else:
            # No merging - use original distance
            final_distance = original_distance
        
        # Get effective threshold for distance check
        effective_threshold = candidate.get('effective_threshold', CROSS_CLIP_THRESHOLD)
        body_sim_threshold = 0.7 if both_have_faces else 0.65
        
        # Check if distance passes threshold
        if final_distance >= effective_threshold or body_sim_for_check < body_sim_threshold:
            continue
        
        # In Pass 2, allow locked clusters (from Pass 1) to merge with NEW unlocked clusters
        # Only skip if BOTH are locked (both already merged in Pass 1 - would be a re-merge)
        # Skip if both clusters are already locked (merged in Pass 1)
        both_locked = (i in locked_clusters and j in locked_clusters)
        if both_locked:
            continue
        
        # Check if this is a face-to-face match
        is_face_to_face_match = both_have_faces
        
        # Before merging, check if target cluster already has confirmed face-to-face match from same clip
        target_cluster_id = final_ids[i]
        
        # Check if there's already a confirmed face-to-face match for this cluster in clip_j
        if (target_cluster_id, clip_j) in confirmed_face_to_face_matches:
            continue
        
        # Check same for clip_i (in case j already has confirmed match)
        source_cluster_id = final_ids[j]
        if (source_cluster_id, clip_i) in confirmed_face_to_face_matches:
            continue
        
        # CRITICAL: Prevent multiple clusters from the same source clip from merging with the same target cluster
        if (target_cluster_id, clip_j) in target_cluster_to_source_clip:
            existing_source_clip = target_cluster_to_source_clip[(target_cluster_id, clip_j)]
            if existing_source_clip == clip_i:
                continue
        
        # Merge j into i
        final_ids[j] = final_ids[i]
        
        # Record that this target cluster (in clip_j) has been merged with a cluster from clip_i
        target_cluster_to_source_clip[(target_cluster_id, clip_j)] = clip_i
        
        # Mark this as confirmed face-to-face match if applicable
        if is_face_to_face_match:
            merged_cluster_id = final_ids[j]
            confirmed_face_to_face_matches[(merged_cluster_id, clip_i)] = True
            confirmed_face_to_face_matches[(merged_cluster_id, clip_j)] = True
    
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
    
    # CRITICAL: Fix same-clip duplicates - prevent multiple tracklets from same clip in same global ID
    # This can happen when multiple same-clip tracklets merge with the same cross-clip tracklet
    print("    Validating: Checking for same-clip duplicates...")
    duplicate_fixes = 0
    
    # Group tracklets by global_id and clip_idx
    global_id_to_clips = {}
    for i, (gid, tracklet) in enumerate(zip(global_ids, tracklets)):
        clip_idx = tracklet['clip_idx']
        if gid not in global_id_to_clips:
            global_id_to_clips[gid] = {}
        if clip_idx not in global_id_to_clips[gid]:
            global_id_to_clips[gid][clip_idx] = []
        global_id_to_clips[gid][clip_idx].append(i)
    
    # Find violations: multiple tracklets from same clip in same global ID
    next_new_id = len(np.unique(global_ids))
    for gid, clips_dict in global_id_to_clips.items():
        for clip_idx, indices in clips_dict.items():
            if len(indices) > 1:
                # Violation: multiple tracklets from same clip in same global ID
                
                # Group tracklets by temp_global_id (from Stage 1 PersonCluster)
                # Tracklets from the same PersonCluster should stay together
                temp_id_groups = {}
                for idx in indices:
                    temp_id = tracklets[idx].get('temp_global_id', -1)
                    if temp_id not in temp_id_groups:
                        temp_id_groups[temp_id] = []
                    temp_id_groups[temp_id].append(idx)
                
                
                # Find the longest tracklet group (best representative cluster)
                group_lengths = []
                for temp_id, group_indices in temp_id_groups.items():
                    total_length = sum(tracklets[i]['end_frame'] - tracklets[i]['start_frame'] for i in group_indices)
                    frames_str = ', '.join([f"{tracklets[i]['start_frame']}-{tracklets[i]['end_frame']}" for i in group_indices])
                    group_lengths.append((total_length, temp_id, group_indices, frames_str))
                group_lengths.sort(reverse=True)
                
                # If there's only ONE PersonCluster but multiple tracklets, check if it's a REAL duplicate
                # or just one PersonCluster with multiple tracklets (which is valid)
                # A duplicate is when the same Global ID has tracklets from DIFFERENT PersonClusters in the same clip
                
                if len(group_lengths) == 1:
                    # Only one PersonCluster, but multiple tracklets
                    temp_id = group_lengths[0][1]
                    group_indices = group_lengths[0][2]
                    frames_str = group_lengths[0][3]
                    
                    # GENERIC FIX: Check if ALL tracklets in this clip for this Global ID are from the SAME PersonCluster
                    # If yes, it's NOT a duplicate - just one PersonCluster with multiple tracklets (valid!)
                    better_match_found = False
                    
                    gid_tracklets_in_clip = [i for i, (gid_check, t) in enumerate(zip(global_ids, tracklets)) 
                                             if gid_check == gid and t['clip_idx'] == clip_idx]
                    
                    # Check if ALL tracklets in this Global ID + Clip are from the same PersonCluster (temp_id)
                    all_same_person_cluster = all(
                        tracklets[i].get('temp_global_id', -1) == temp_id 
                        for i in gid_tracklets_in_clip
                    )
                    
                    if all_same_person_cluster and len(gid_tracklets_in_clip) == len(group_indices):
                        # This is NOT a real duplicate - just one PersonCluster with multiple tracklets
                        # Keep them in the current Global ID (don't reassign)
                        # Don't change global_ids - they're already correct
                        better_match_found = True  # Prevent reassignment
                    else:
                        # REAL duplicate - there are OTHER tracklets from different PersonClusters
                        # This shouldn't happen, but if it does, we need to reassign
                        pass
                    
                    if not better_match_found:
                        # No better match found - reassign to new Global ID (preserve integrity)
                        reassigned_gid = next_new_id
                        for idx in group_indices:
                            global_ids[idx] = reassigned_gid
                            duplicate_fixes += 1
                        next_new_id += 1
                else:
                    # Multiple PersonClusters - keep longest, reassign others
                    keep_temp_id = group_lengths[0][1]
                    keep_indices = group_lengths[0][2]
                    keep_frames = group_lengths[0][3]
                    
                    # Assign new global IDs to all other groups (preserve cluster integrity)
                    for length, temp_id, group_indices, frames_str in group_lengths[1:]:
                        # Reassign ALL tracklets in this cluster together
                        reassigned_gid = next_new_id
                        for idx in group_indices:
                            global_ids[idx] = reassigned_gid
                            duplicate_fixes += 1
                        next_new_id += 1
    
    if duplicate_fixes > 0:
        pass
    
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
    
    # Recompute effective has_face from consecutive face frames (no cache rebuild needed)
    FACE_MIN_CONSEC_FRAMES = 15
    for t in all_tracklets:
        eff_has_face = t.get('has_face', False)
        if 'max_consec_face_frames' in t:
            eff_has_face = int(t['max_consec_face_frames']) >= FACE_MIN_CONSEC_FRAMES
        elif 'num_face_frames' in t and 'num_frames' in t:
            # Approximate consecutive by requiring at least N total face frames
            eff_has_face = int(t['num_face_frames']) >= FACE_MIN_CONSEC_FRAMES
        elif 'face_ratio' in t and 'num_frames' in t:
            # Fallback approximation
            try:
                eff_has_face = float(t['face_ratio']) * int(t['num_frames']) >= FACE_MIN_CONSEC_FRAMES
            except Exception:
                eff_has_face = t.get('has_face', False)
        t['has_face'] = bool(eff_has_face)
    
    print(f"Loaded {len(all_tracklets)} tracklets from cache")
    print("Merging overlapping tracks in same clips...")
    all_tracklets = merge_overlapping_tracks_same_clip(all_tracklets)
    print(f"After merging: {len(all_tracklets)} tracklets\n")
else:
    print("No cache found, processing videos...\n")
    
    all_tracklets = []
    for clip_idx, video_path in enumerate(VIDEO_PATHS):
        tracklets = process_video(video_path, clip_idx)
        all_tracklets.extend(tracklets)
    
    print(f"\nCollected {len(all_tracklets)} tracklets")
    print("Merging overlapping tracks in same clips...")
    all_tracklets = merge_overlapping_tracks_same_clip(all_tracklets)
    print(f"After merging: {len(all_tracklets)} tracklets\n")
    
    # Save to cache
    print(f"Saving embeddings to cache: {NPZ_FILE}")
    np.savez_compressed(
        NPZ_FILE,
        tracklets=np.array(all_tracklets, dtype=object)
    )
    print("Cache saved!\n")

# Filter out tiny fragments (< 10 frames)
MIN_TRACKLET_LENGTH = 30  # Minimum frames to be considered a real person

print("Filtering out tiny tracklets...")
tracklets_before = len(all_tracklets)
all_tracklets = [t for t in all_tracklets if (t['end_frame'] - t['start_frame']) >= MIN_TRACKLET_LENGTH]
if tracklets_before > len(all_tracklets):
    print(f"   Removed {tracklets_before - len(all_tracklets)} tiny fragments (< {MIN_TRACKLET_LENGTH} frames)")
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
            'clustering_method': 'ADAPTIVE',  # Always uses adaptive two-stage clustering
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
    output_data['identities'][f'global_id_{gid}'] = {
        'global_id': int(gid),
        'appearances': [{
            'clip_idx': t['clip_idx'],
            'start_frame': t['start_frame'],
            'end_frame': t['end_frame']
        } for t in tracks]
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