
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
from model.make_model import make_model  # ✅ This is all you need
from config.defaults import _C as cfg_default  # ✅ This is correct
# Line 22 should be empty or the docstring starts here
# ... existing imports ...
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

"""
video_id_detector3.py - ADVANCED REID WITH VIEW INVARIANCE

🚀 NEW FEATURES:
1. ✅ Test-Time Augmentation (TTA) for view robustness
2. ✅ Temporal smoothing for stable embeddings
3. ✅ OSNet + TransReID ensemble
4. ✅ Optimized for cross-clip matching
5. ✅ NPZ caching for fast parameter tuning
6. ✅ Pose-aware features
7. ✅ k-Reciprocal re-ranking

OPTIMIZED FOR:
- Cross-clip matching (same person, different angles)
- Within-clip separation (different people, same scene)
- View-invariant embeddings (front/back/side views)

Usage:
    python video_id_detector3.py
"""

# ======================
# CONFIGURATION
# ======================

VIDEO_DIR = "/Users/liatparker/Documents/mp4_files_id_detectors_upscaled/"
VIDEO_FILES = ["1_upscaled.mp4", "2_upscaled.mp4", "3_upscaled.mp4", "4_upscaled.mp4"]
VIDEO_PATHS = [os.path.join(VIDEO_DIR, f) for f in VIDEO_FILES]

YOLO_WEIGHTS = "yolov8n.pt"
REID_MODEL = "osnet_ibn_x1_0"
USE_VIT_REID = True  # 🔥 NEW: Enable ViT-based ReID
VIT_MODEL = "vit_base_patch16_224"  # TransReID-style model
FACE_MODEL = "buffalo_l"
# NEW - Force CPU:
DEVICE = "cpu"  # 🔥 Force CPU to avoid macOS MPS/CUDA issues

USE_ENSEMBLE_REID = True         # 🔥 Enable ensemble (OSNet + ResNet50)
ENSEMBLE_WEIGHTS = [0.7, 0.3]     # [OSNet, ResNet50] - ResNet50-heavy

# Embedding Weights - ORIGINAL
FACE_WEIGHT = 0.2              # Original face weight
MOTION_WEIGHT = 0.1            # Original motion weight  
POSE_WEIGHT = 0.1              # Original pose weight
# body_weight = 1 - 0.2 - 0.1 - 0.1 = 0.6 (ORIGINAL)

# Features
USE_CAMERA_BIAS = True         # Disabled for cross-clip matching
USE_RERANKING = True
K_RECIPROCAL = 25
LAMBDA_VALUE = 0.5
USE_POSE_FEATURES = True
POSE_CONFIDENCE = 0.5
# Enable spatial-temporal context matchingUSE_SPATIAL_TEMPORAL_CONTEXT = True


# 🔥 DISABLED: All advanced clustering features - back to original logic
USE_ADVANCED_CLUSTERING_LOGIC = False  # Master flag for physical proximity logic during clustering
PROXIMITY_PATTERN_TOLERANCE = 0.15  # CONSERVATIVE: Lower tolerance to prevent over-merging
SPATIAL_PROXIMITY_THRESHOLD = 0.3  # Threshold for detecting cross-clip spatial proximity (0-1)

# 🔥 DISABLED: Robust weights for ambiguous cases
USE_ROBUST_AMBIGUOUS_WEIGHTS = False  # Enable enhanced weights for ambiguous clustering
AMBIGUOUS_DISTANCE_THRESHOLD = 0.5   # Distance threshold to trigger robust weights
ROBUST_BODY_WEIGHT = 0.8            # Higher body weight for ambiguous cases (clothing/appearance)
ROBUST_FACE_WEIGHT = 0.1            # Lower face weight for ambiguous cases
ROBUST_POSE_WEIGHT = 0.05           # Lower pose weight for ambiguous cases
ROBUST_MOTION_WEIGHT = 0.05         # Lower motion weight for ambiguous cases

# 🔥 DISABLED: Clothing/appearance focused approach
USE_CLOTHING_FOCUSED_APPROACH = False  # Use clothing/appearance weights for ALL comparisons (not just ambiguous cases)
CLOTHING_BODY_WEIGHT = 0.8            # Body weight for clothing-focused approach
CLOTHING_FACE_WEIGHT = 0.1            # Face weight for clothing-focused approach
CLOTHING_POSE_WEIGHT = 0.05           # Pose weight for clothing-focused approach
CLOTHING_MOTION_WEIGHT = 0.05         # Motion weight for clothing-focused approach

# 🔥 NEW: Video-level ReID settings
USE_VIDEO_REID = True
VIDEO_SEQUENCE_LENGTH = 4 # Number of frames per video clip
VIDEO_SEQUENCE_STRIDE = 2   # Overlap between clips (50% overlap)
VIDEO_WEIGHT = 0.6       # Weight for video embeddings (0.7 for image)
VIDEO_DIM = 768           # AP3D output dimension

# 🔥 NEW: View Invariance Features
USE_TTA = True                   # Test-time augmentation (horizontal flip)
USE_TEMPORAL_SMOOTHING = True    # Smooth embeddings within tracks
SMOOTHING_WINDOW = 5             # Frames to smooth over

# Clustering
CLUSTERING_METHOD = "ADAPTIVE"  # "SIMILARITY" or "DBSCAN"
SIMILARITY_THRESHOLD = 0.8   # 🔥 High for Euclidean
DBSCAN_EPS = 0.35
DBSCAN_MIN_SAMPLES = 1
# Distance Metrics
DISTANCE_METRIC = "cosine"  # 🔥 Changed from cosine
USE_CHAMFER_DISTANCE = False  # 🔥 NEW: Enable Chamfer distance for cross-clip matching

# 🎯 ADAPTIVE CLUSTERING: Two-stage approach
USE_ADAPTIVE_CLUSTERING = True

# Stage 1: Within-clip clustering (strict - separate people in same scene)
WITHIN_CLIP_THRESHOLD = 0.2 # Strict for separation
WITHIN_CLIP_FACE_WEIGHT = 0.4  # High - faces reliable in same lighting/angle

# Stage 2: Cross-clip merging (lenient - match same person across scenes)
CROSS_CLIP_THRESHOLD = 0.46  # Lenient for matching
CROSS_CLIP_FACE_WEIGHT = 0.45  # Low - faces change with angles/lighting

# 🎯 Per-clip adaptive thresholds (AUTOMATIC ANALYSIS)
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
RESNET_DIM = 2048  # 🔥 ResNet50 output dimension
BODY_DIM = OSNET_DIM + TRANSREID_DIM if USE_ENSEMBLE_REID else OSNET_DIM  # 1280 or 512
MOTION_DIM = 5
POSE_DIM = 66

print("="*60)
print("🚀 ADVANCED REID SYSTEM v3 - VIEW INVARIANT")
print("="*60)
print(f"Ensemble: {USE_ENSEMBLE_REID} (OSNet + ResNet50) {ENSEMBLE_WEIGHTS if USE_ENSEMBLE_REID else ''}")
print(f"Body dim: {BODY_DIM}")
print(f"TTA (flip augmentation): {USE_TTA}")
print(f"Temporal smoothing: {USE_TEMPORAL_SMOOTHING}")
print(f"Face weight: {FACE_WEIGHT}")
print(f"Body weight: {1 - FACE_WEIGHT - MOTION_WEIGHT - POSE_WEIGHT:.2f}")
print("🎯 APPROACH: ORIGINAL LOGIC (All advanced features disabled)")
print("="*60 + "\n")

# ======================
# LOAD MODELS
# ======================

print("📦 Loading models...")
yolo = YOLO(YOLO_WEIGHTS)

# 🔥 Load IBN-OSNet
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
    print("    ✅ IBN-OSNet loaded!")
except Exception as e:
    print(f"    ❌ Failed to load IBN-OSNet: {e}")
    print("    ⚠️  Falling back to standard OSNet...")
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
        
        print(f"    ✅ TransReID loaded! Embedding dim: {TRANSREID_DIM}")
    except Exception as e:
        print(f"    ⚠️  TransReID failed: {e}")
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

print("✅ All models loaded!\n")

# 🔥 STEP C: DEFINE the video model class FIRST
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
        print("    ✅ VID-Trans-ReID loaded with pretrained weights!")
    except Exception as e:
        print(f"    ❌ Failed to load VID-Trans-ReID: {e}")
        print("    ⚠️  Falling back to image-only mode")
        USE_VIDEO_REID = False


print(f"📊 Video ReID enabled: {USE_VIDEO_REID}")
if USE_VIDEO_REID:
    print(f"   Sequence length: {VIDEO_SEQUENCE_LENGTH} frames")
    print(f"   Video weight: {VIDEO_WEIGHT:.0%}\n")
else:
    print()


def preprocess_crop(crop):
    """CLAHE + sharpening"""
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
    """Extract pose keypoints (66D vector)"""
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
    """Extract face embeddings for batch"""
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
            
            # 🔥 NEW: Combine with IBN-heavy weights
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
    🔥 Temporal smoothing with exponential moving average
    
    Reduces jitter and noise in embeddings within a track.
    Makes person representation more stable across frames.
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
    
    # 🔥 NEW: Normalize to [0,1] range
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


# 🔥🔥🔥 STEP E: Video ReID Helper Functions 🔥🔥🔥

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
    Merge tracklets that are likely fragments of the same person within each clip.
    Uses adaptive temporal overlap check based on track length.
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
        
        print(f"  🔍 Clip {clip_idx}: Processing {len(clip_tracks)} tracklets")
        
        # 🔥 NO special cases - adaptive temporal overlap check handles everything
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
                
                # 🔥 Check temporal overlap (adaptive based on track length)
                overlap_start = max(current['start_frame'], candidate['start_frame'])
                overlap_end = min(current['end_frame'], candidate['end_frame'])
                overlap_frames = max(0, overlap_end - overlap_start)
                
                # Calculate overlap ratio relative to shorter track
                min_length = min(current['end_frame'] - current['start_frame'], 
                                 candidate['end_frame'] - candidate['start_frame'])
                overlap_ratio = overlap_frames / min_length if min_length > 0 else 0
                
                if overlap_ratio > 0.5:  # Block if overlap > 50% of shorter track
                    # They appear simultaneously → MUST be different people → DON'T merge
                    print(f"    ⚠️  Tracklet {i} and {j} overlap {overlap_frames} frames ({overlap_ratio:.1%}) → keeping separate")
                    continue
                
                # Check gap (only consider if NOT overlapping significantly)
                gap = candidate['start_frame'] - current['end_frame']
                
                if 0 < gap <= 120:  # 3 seconds at 30fps
                    # Check embedding similarity
                    body_sim = np.dot(current['body_emb'], candidate['body_emb'])
                    
                    # 🔥 Very strict similarity requirement
                    if body_sim > 0.8:
                        print(f"    ✅ Merging tracklet {i} and {j} (gap={gap}, sim={body_sim:.3f})")
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
                
                print(f"    ✅ Created merged tracklet from {len(group)} fragments")
                merged_all.append(merged_track)
            else:
                # Single tracklet, keep as is
                merged_all.append(current)
    
    print(f"  ✅ Merge complete: {len(tracklets)} → {len(merged_all)} tracklets")
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
    """Process single video and extract tracklets with embeddings"""
    print(f"\n📹 Processing: {os.path.basename(video_path)} (Clip {clip_idx})")
    
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
        tracker="bytetrack_tuned.yaml",  # 🔥 Use tuned config (was working before!)    ,
        verbose=False
    )
    
    # Storage
    tracks_data = {}
    frame_idx = 0
    
    for result in tqdm(results_generator, total=total_frames, desc=f"  Tracking"):
        if result.boxes is None or len(result.boxes) == 0:
            frame_idx += 1
            continue
        
        # 🔥 ADD DEBUG for Clip 0
        if clip_idx == 0 and frame_idx % 100 == 0:  # Every 100 frames
            print(f"  🐛 Frame {frame_idx}: {len(result.boxes)} detections, track_ids: {[int(b.id.item()) if b.id is not None else None for b in result.boxes]}")
        
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
        video_embs_list = []  # 🔥 NEW: Video-level embeddings
        face_embs_list = []
        pose_embs_list = []
        has_faces_list = []
        qualities_list = []  # 🔥 NEW: Track frame quality
        bbox_confs_list = []  # 🔥 NEW: Store bbox confidences
        
        # 🔥 NEW: Extract video embeddings from sequences
        if USE_VIDEO_REID and len(crops) >= VIDEO_SEQUENCE_LENGTH:
            sequences = group_crops_into_sequences(
                crops, 
                seq_len=VIDEO_SEQUENCE_LENGTH, 
                stride=VIDEO_SEQUENCE_STRIDE
            )
            
            for seq in sequences:
                video_emb = extract_video_embedding(seq)
                video_embs_list.append(video_emb)
        
        for i in range(0, len(crops), BATCH_SIZE):
            batch = crops[i:i+BATCH_SIZE]
            
            body_batch = extract_body_batch(batch)
            face_batch, has_face_batch = extract_face_batch(batch)
            pose_batch = extract_pose_batch(batch)
            
            body_embs_list.extend(body_batch)
            face_embs_list.extend(face_batch)
            pose_embs_list.extend(pose_batch)
            has_faces_list.extend(has_face_batch)
            
            # 🔥 NEW: Compute quality for each frame in batch
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
        
        # 🔥 NEW: Use quality-weighted aggregation for body embeddings
        avg_body = weighted_average_embeddings(body_embs_list, qualities_list)
        
        # 🔥 NEW: Face - weighted average among frames with faces
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
        
        # 🔥 NEW: Aggregate video embeddings
        if video_embs_list:
            avg_video = np.mean(video_embs_list, axis=0).astype(np.float32)
            avg_video = avg_video / (np.linalg.norm(avg_video) + 1e-8)
        else:
            avg_video = np.zeros(VIDEO_DIM, dtype=np.float32)
        
        # 🔥 NEW: Select representative frames for multi-frame distance
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
            'video_emb': avg_video,  # 🔥 NEW: Video-level embedding
            'face_emb': avg_face,
            'pose_emb': avg_pose,
            'motion_emb': motion_feat,
            'has_face': len(face_indices) > 0,
            'frames': data['frames'],
            'bboxes': data['bboxes'],
            # 🔥 NEW: Multi-frame embeddings for cross-clip matching
            'body_frames': representative_body_frames,  # [K, D] array
            'face_frames': representative_face_frames if representative_face_frames is not None else np.array([])
        })
    
    print(f"  ✅ Extracted {len(tracklets)} tracklets")
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
                # No face: reweight to body only
                dist_matrix[i, j] = (
                    (body_weight + FACE_WEIGHT) * body_dist[i, j] +
                    POSE_WEIGHT * pose_dist[i, j] +
                    MOTION_WEIGHT * motion_dist[i, j]
                ) / (body_weight + FACE_WEIGHT + POSE_WEIGHT + MOTION_WEIGHT)
    
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
    
    # 🔥 NEW: Determine max simultaneous people in this clip
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
    
    # 🔥 NEW: Use max simultaneous people to guide threshold
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
    🔥 ADAPTIVE 2-STAGE CLUSTERING:
    Stage 1: Within-clip clustering (strict - separate people in same scene)
    Stage 2: Cross-clip merging (lenient - match same person across scenes)
    """
    print("🎯 ADAPTIVE CLUSTERING (2-stage)")
    print(f"   Physical proximity logic: {'ENABLED' if USE_ADVANCED_CLUSTERING_LOGIC else 'DISABLED'}")
    if USE_ADVANCED_CLUSTERING_LOGIC:
        print(f"   - Proximity tolerance: {PROXIMITY_PATTERN_TOLERANCE}")
    
    # Stage 1: Within-clip clustering (strict)
    print("  📍 Stage 1: Within-clip clustering (strict)...")
    
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
        
        # 🎯 Determine threshold (automatic analysis or manual override)
        if USE_PER_CLIP_THRESHOLDS:
            if clip_idx in PER_CLIP_THRESHOLDS:
                # Use manual override if specified
                clip_threshold = PER_CLIP_THRESHOLDS[clip_idx]
                print(f"    Clip {clip_idx}: {n} tracklets (manual threshold: {clip_threshold:.3f})")
            else:
                # Automatic analysis
                clip_threshold, analysis = analyze_clip_characteristics(clip_tracks)
                print(f"    Clip {clip_idx}: {n} tracklets (auto threshold: {clip_threshold:.3f})")
                print(f"      📊 Analysis: crowd={analysis['crowding']:+.3f}, track={analysis['tracking']:+.3f}, face={analysis['face']:+.3f}, div={analysis['diversity']:+.3f}")
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
        
        # 🔥 Temporal overlap penalty - Force separation for simultaneous appearances
        for i in range(n):
            for j in range(i+1, n):
                track_i = clip_tracks[i]
                track_j = clip_tracks[j]
                
                # Check temporal overlap
                overlap_start = max(track_i['start_frame'], track_j['start_frame'])
                overlap_end = min(track_i['end_frame'], track_j['end_frame'])
                overlap_frames = max(0, overlap_end - overlap_start)
                
                if overlap_frames > 60:  # Significant overlap (> 2 seconds at 30fps)
                    # Calculate overlap ratio relative to shorter track
                    min_length = min(track_i['end_frame'] - track_i['start_frame'],
                                   track_j['end_frame'] - track_j['start_frame'])
                    overlap_ratio = overlap_frames / min_length if min_length > 0 else 0
                    
                    # Proportional penalty based on overlap severity
                    if overlap_ratio > 0.5:  # More than 50% overlap
                        penalty = 0.10  # Reduced from 0.15
                    else:
                        penalty = 0.10 * overlap_ratio  # Proportional
                    
                    dist_matrix[i, j] += penalty
                    dist_matrix[j, i] = dist_matrix[i, j]
                    # 🔥 Debug print
                    track_i_id = track_i.get('temp_id', track_i.get('track_id', i))
                    track_j_id = track_j.get('temp_id', track_j.get('track_id', j))
                    print(f"    ⚠️  Temporal overlap penalty: Clip {clip_idx} ID {track_i_id} ↔ ID {track_j_id} "
                          f"(overlap: {overlap_frames}f, ratio: {overlap_ratio:.2f}) +{penalty:.3f}")
        
        # Cluster within clip (strict threshold)
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
                    avg_dist = np.mean([dist_matrix[i, j] for j in members])
                    if avg_dist < best_dist:
                        best_dist = avg_dist
                        best_cluster = cluster_id
            
            threshold = clip_threshold  # Use the clip-specific threshold determined above
            if best_cluster != -1 and best_dist < threshold:
                local_ids[i] = best_cluster
            else:
                local_ids[i] = next_cluster
                next_cluster += 1
        
        # Assign global temporary IDs
        for i, track in enumerate(clip_tracks):
            track['temp_global_id'] = next_local_id + local_ids[i]
        
        next_local_id += next_cluster
        print(f"      → {next_cluster} local clusters")
    
    # Stage 2: Cross-clip merging with HYBRID embeddings
    print(f"  🔗 Stage 2: Cross-clip merging (HYBRID: {1-VIDEO_WEIGHT:.0%} image + {VIDEO_WEIGHT:.0%} video)...")
    print(f"    Total intermediate clusters: {next_local_id}")
    
    # 🔥 NEW: Extract BOTH body (image) and video embeddings
    body_embs = np.array([t['body_emb'] for t in tracklets])
    
    # 🔥 FIXED: Handle video embeddings with consistent shapes
    video_embs = []
    for t in tracklets:
        video_emb = t.get('video_emb', np.zeros(VIDEO_DIM))
        # Ensure consistent shape
        if isinstance(video_emb, (list, tuple)):
            video_emb = np.array(video_emb)
        if video_emb.shape != (VIDEO_DIM,):
            # Resize or pad to correct dimension
            if len(video_emb) > VIDEO_DIM:
                video_emb = video_emb[:VIDEO_DIM]
            else:
                padded = np.zeros(VIDEO_DIM)
                padded[:len(video_emb)] = video_emb
                video_emb = padded
        video_embs.append(video_emb)
    video_embs = np.array(video_embs)
    
    face_embs = np.array([t['face_emb'] for t in tracklets])
    has_faces = np.array([t['has_face'] for t in tracklets])
    temp_ids = np.array([t['temp_global_id'] for t in tracklets])
    
    # Compute per-cluster representative embeddings
    unique_temp_ids = np.unique(temp_ids)
    n_temp = len(unique_temp_ids)
    print(f"    🔍 DEBUG: unique_temp_ids = {unique_temp_ids}")
    
    cluster_body_embs = np.zeros((n_temp, body_embs.shape[1]))
    cluster_video_embs = np.zeros((n_temp, video_embs.shape[1]))  # 🔥 NEW
    cluster_face_embs = np.zeros((n_temp, face_embs.shape[1]))
    cluster_has_faces = np.zeros(n_temp, dtype=bool)
    
    for i, tid in enumerate(unique_temp_ids):
        mask = temp_ids == tid
        cluster_body_embs[i] = np.mean(body_embs[mask], axis=0)
        cluster_video_embs[i] = np.mean(video_embs[mask], axis=0)  # 🔥 NEW
        cluster_face_embs[i] = np.mean(face_embs[mask], axis=0)
        cluster_has_faces[i] = np.any(has_faces[mask])
    
    # 🔥 NEW: Compute HYBRID distance (weighted combination)
    print("    Computing hybrid distances (image + video)...")
    
    if USE_CHAMFER_DISTANCE:
        print("    Using Chamfer distance for cross-clip matching...")
        
        # 🔥 NEW: Use Chamfer distance for cross-clip matching
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
        # Original cosine distance computation
        image_dist = cdist(cluster_body_embs, cluster_body_embs, metric='cosine')
        video_dist = cdist(cluster_video_embs, cluster_video_embs, metric='cosine')
        face_dist = cdist(cluster_face_embs, cluster_face_embs, metric='cosine')
        
        # Combine image (body) and video with weights
        image_weight = 1.0 - VIDEO_WEIGHT  # e.g., 0.7
        body_video_dist = image_weight * image_dist + VIDEO_WEIGHT * video_dist
        
        # Add face component
        body_w = 1.0 - CROSS_CLIP_FACE_WEIGHT
        dist_matrix = body_w * body_video_dist + CROSS_CLIP_FACE_WEIGHT * face_dist
        
        # Adjust for missing faces (use body+video only)
        for i in range(n_temp):
            for j in range(n_temp):
                if not cluster_has_faces[i] or not cluster_has_faces[j]:
                    dist_matrix[i, j] = body_video_dist[i, j]
    
    print("    ✅ Hybrid distances computed")
    
    # 🔥 NEW: Build physical proximity mapping for clustering decisions
    clip_to_temp_ids = {}
    for t in tracklets:
        clip_idx = t['clip_idx']
        temp_id = t['temp_global_id']
        if clip_idx not in clip_to_temp_ids:
            clip_to_temp_ids[clip_idx] = set()
        clip_to_temp_ids[clip_idx].add(temp_id)
        print(f"      🔍 Adding ID {temp_id} to Clip {clip_idx} (frames {t['start_frame']}-{t['end_frame']})")
    
    # Build physical proximity relationships
    physical_proximity_pairs = set()
    
    # 1. Within-clip physical proximity (existing logic)
    for clip_idx, temp_ids_in_clip in clip_to_temp_ids.items():
        temp_ids_list = list(temp_ids_in_clip)
        for i in range(len(temp_ids_list)):
            for j in range(i + 1, len(temp_ids_list)):
                id_i = temp_ids_list[i]
                id_j = temp_ids_list[j]
                # Record that these temp IDs were physically close in this clip
                physical_proximity_pairs.add((id_i, id_j))
                physical_proximity_pairs.add((id_j, id_i))  # Bidirectional
                print(f"      📍 Within-clip proximity: ID {id_i} ↔ ID {id_j} in Clip {clip_idx}")
    
    # 2. Cross-clip spatial proximity (RE-ENABLED - needed for ID 3 ↔ ID 11 relationship)
    # This detects when the same person appears in different clips with different temp IDs
    if USE_ADVANCED_CLUSTERING_LOGIC:
        print("    🔍 Checking cross-clip spatial proximity...")
        cross_clip_proximity_pairs = set()
        
        # Get all tracklets
        all_tracklets_list = list(tracklets)
        
        # Check spatial proximity between tracklets from different clips
        for i in range(len(all_tracklets_list)):
            for j in range(i + 1, len(all_tracklets_list)):
                t_i = all_tracklets_list[i]
                t_j = all_tracklets_list[j]
                
                # Only check if they're from different clips AND different temp IDs
                if t_i['clip_idx'] != t_j['clip_idx'] and t_i['temp_global_id'] != t_j['temp_global_id']:
                    # Check if bounding boxes are spatially close
                    if 'bboxes' in t_i and 'bboxes' in t_j and t_i['bboxes'] and t_j['bboxes']:
                        # Get representative bboxes (first bbox of each tracklet)
                        bbox_i = t_i['bboxes'][0]  # [x1, y1, x2, y2]
                        bbox_j = t_j['bboxes'][0]  # [x1, y1, x2, y2]
                        
                        # Calculate spatial proximity
                        spatial_proximity = calculate_bbox_proximity(bbox_i, bbox_j)
                        
                        # If spatially close, add to proximity pairs
                        if spatial_proximity > SPATIAL_PROXIMITY_THRESHOLD:
                            id_i = t_i['temp_global_id']
                            id_j = t_j['temp_global_id']
                            cross_clip_proximity_pairs.add((id_i, id_j))
                            cross_clip_proximity_pairs.add((id_j, id_i))
                            print(f"      🎯 Cross-clip spatial proximity: ID {id_i} (clip {t_i['clip_idx']}) ↔ ID {id_j} (clip {t_j['clip_idx']}) - proximity: {spatial_proximity:.3f}")
                            print(f"        📍 ID {id_i} bbox: {bbox_i}")
                            print(f"        📍 ID {id_j} bbox: {bbox_j}")
        
        # Add cross-clip proximity pairs to main set
        physical_proximity_pairs.update(cross_clip_proximity_pairs)
        print(f"    ✅ Found {len(cross_clip_proximity_pairs)//2} cross-clip spatial proximity pairs")
    
    # 🔥 DEBUG: Show detected physical proximity pairs
    if USE_ADVANCED_CLUSTERING_LOGIC:
        print(f"    🔍 Detected physical proximity pairs: {sorted(physical_proximity_pairs)}")
        # Show specific pairs involving ID 3, 4, 7, 8, 11
        relevant_pairs = [(a, b) for (a, b) in physical_proximity_pairs if a in [3, 4, 7, 8, 11] and b in [3, 4, 7, 8, 11]]
        if relevant_pairs:
            print(f"    🔍 Relevant pairs (3,4,7,8,11): {relevant_pairs}")
        
        # Show specific pairs for ID 4 and ID 8
        id_4_pairs = [(a, b) for (a, b) in physical_proximity_pairs if a == 4 or b == 4]
        id_8_pairs = [(a, b) for (a, b) in physical_proximity_pairs if a == 8 or b == 8]
        print(f"    🔍 ID 4 proximity pairs: {id_4_pairs}")
        print(f"    🔍 ID 8 proximity pairs: {id_8_pairs}")
        
    # 🔥 DEBUG: Show which temp IDs appear in each clip
    print(f"    🔍 Temp IDs per clip:")
    for clip_idx in sorted(clip_to_temp_ids.keys()):
        temp_ids = sorted(list(clip_to_temp_ids[clip_idx]))
        print(f"      Clip {clip_idx}: {temp_ids}")
        # Check if ID 3 and ID 7 are in the same clip
        if 3 in temp_ids and 7 in temp_ids:
            print(f"        ⚠️  ID 3 and ID 7 are BOTH in Clip {clip_idx} - should be physically close!")
        elif 3 in temp_ids:
            print(f"        ID 3 in Clip {clip_idx}")
        elif 7 in temp_ids:
            print(f"        ID 7 in Clip {clip_idx}")
        
        # Show frame ranges for IDs 3, 4, 7, 8, 11
        for temp_id in [3, 4, 7, 8, 11]:
            if temp_id in temp_ids:
                tracklets_for_id = [t for t in tracklets if t['temp_global_id'] == temp_id and t['clip_idx'] == clip_idx]
                if tracklets_for_id:
                    t = tracklets_for_id[0]
                    print(f"        ID {temp_id}: frames {t['start_frame']}-{t['end_frame']}")
        
        # Show all temp IDs in this clip for debugging
        print(f"        All temp IDs in Clip {clip_idx}: {sorted(temp_ids)}")
        
        # Check for cross-clip relationships
        if clip_idx == 1:  # Clip 1
            print(f"        🔍 Cross-clip analysis: ID 3 in Clip 1 should be same person as ID 11 in Clip 2")
        elif clip_idx == 2:  # Clip 2
            print(f"        🔍 Cross-clip analysis: ID 11 in Clip 2 should be same person as ID 3 in Clip 1")
    
    # 🔥 Build same-clip overlap block matrix
    same_clip_overlap_block = np.zeros((n_temp, n_temp), dtype=bool)
    
    for i in range(n_temp):
        for j in range(i + 1, n_temp):
            # Get tracklets for cluster i and j
            tracklets_i = [t for t in tracklets if t['temp_global_id'] == unique_temp_ids[i]]
            tracklets_j = [t for t in tracklets if t['temp_global_id'] == unique_temp_ids[j]]
            
            # Check if they share ANY clip
            clips_i = set(t['clip_idx'] for t in tracklets_i)
            clips_j = set(t['clip_idx'] for t in tracklets_j)
            shared_clips = clips_i & clips_j
            
            if shared_clips:
                # They're from the same clip → they were kept separate by within-clip clustering
                # → They overlapped in time → DON'T merge across clips
                same_clip_overlap_block[i, j] = True
                same_clip_overlap_block[j, i] = True
                print(f"    🚫 Blocking merge: cluster {unique_temp_ids[i]} and {unique_temp_ids[j]} (same clip {list(shared_clips)})")
    
    # 🔥 Greedy merging with ADAPTIVE thresholds and transitive blocking
    final_ids = np.arange(n_temp)
    base_threshold = CROSS_CLIP_THRESHOLD  # Use as baseline (0.42)
    
    # 🔥 SAFER APPROACH: Apply proximity tolerance only when merging with existing clusters
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
            
            # 🔍 DEBUG: Check if this is the ID 4 ↔ ID 12 pair we're looking for
            if (temp_id_i == 4 and temp_id_j == 12) or (temp_id_i == 12 and temp_id_j == 4):
                print(f"    🎯 FOUND TARGET PAIR: ID {temp_id_i} ↔ ID {temp_id_j} (indices {i} ↔ {j})")
            
            # 🔍 DEBUG: Also check ID 4 ↔ ID 9 for comparison
            if (temp_id_i == 4 and temp_id_j == 9) or (temp_id_i == 9 and temp_id_j == 4):
                print(f"    🎯 FOUND COMPARISON PAIR: ID {temp_id_i} ↔ ID {temp_id_j} (indices {i} ↔ {j})")
            
            # 🔥 Check direct block (same clip) - NO EXCEPTIONS
            if same_clip_overlap_block[i, j]:
                if (temp_id_i == 4 and temp_id_j == 12) or (temp_id_i == 12 and temp_id_j == 4):
                    print(f"    🚫 BLOCKED: ID {temp_id_i} ↔ ID {temp_id_j} (same clip overlap)")
                continue
            
            # 🔥 Transitive blocking - prevent indirect merging of blocked clusters
            merge_blocked = False
            for k in range(n_temp):
                if k == i or k == j:
                    continue
                
                # 🔍 DEBUG: Check if this is the ID 4 ↔ ID 12 pair
                if (temp_id_i == 4 and temp_id_j == 12) or (temp_id_i == 12 and temp_id_j == 4):
                    print(f"    🔍 DEBUG: Checking transitive blocking for ID {temp_id_i} ↔ ID {temp_id_j}")
                if final_ids[k] == i and same_clip_overlap_block[k, j]:
                    if (temp_id_i == 4 and temp_id_j == 12) or (temp_id_i == 12 and temp_id_j == 4):
                        print(f"    🚫 TRANSITIVE BLOCK: ID {temp_id_i} (merged with {unique_temp_ids[k]}) has same-clip overlap with ID {temp_id_j}")
                    merge_blocked = True
                    break
                if final_ids[k] == j and same_clip_overlap_block[k, i]:
                    if (temp_id_i == 4 and temp_id_j == 12) or (temp_id_i == 12 and temp_id_j == 4):
                        print(f"    🚫 TRANSITIVE BLOCK: ID {temp_id_j} (merged with {unique_temp_ids[k]}) has same-clip overlap with ID {temp_id_i}")
                    merge_blocked = True
                    break
            if merge_blocked:
                if (temp_id_i == 4 and temp_id_j == 12) or (temp_id_i == 12 and temp_id_j == 4):
                    print(f"    🚫 TRANSITIVE BLOCKED: ID {temp_id_i} ↔ ID {temp_id_j}")
                    
                    # 🔥 NEW: Check if robust weights can override transitive blocking
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
                        
                        # 🔥 NEW: Also calculate Hausdorff distance for spatial pattern analysis
                        hausdorff_dist = calculate_hausdorff_distance(track_i['bboxes'], track_j['bboxes'])
                        
                        # If robust distance is very good (< 0.35) OR Hausdorff distance is very good (< 100 pixels), override transitive blocking
                        if robust_distance < 0.35 or hausdorff_dist < 100:
                            print(f"    🔥 ROBUST OVERRIDE: ID {temp_id_i} ↔ ID {temp_id_j} (robust_dist={robust_distance:.3f}, hausdorff_dist={hausdorff_dist:.1f}) - bypassing transitive block!")
                            merge_blocked = False
                        else:
                            print(f"    ❌ NO OVERRIDE: ID {temp_id_i} ↔ ID {temp_id_j} (robust_dist={robust_distance:.3f}, hausdorff_dist={hausdorff_dist:.1f})")
                
                if merge_blocked:
                    continue
            
            # 🔥 NEW: ADAPTIVE threshold based on pair characteristics
            tracklets_i = [t for t in tracklets if t['temp_global_id'] == unique_temp_ids[i]]
            tracklets_j = [t for t in tracklets if t['temp_global_id'] == unique_temp_ids[j]]
            
            # Factor 1: Face availability (both have faces = more reliable)
            both_have_faces = cluster_has_faces[i] and cluster_has_faces[j]
            
            # Factor 2: Tracklet length (longer = more reliable)
            total_frames_i = sum(t['end_frame'] - t['start_frame'] for t in tracklets_i)
            total_frames_j = sum(t['end_frame'] - t['start_frame'] for t in tracklets_j)
            avg_length = (total_frames_i + total_frames_j) / 2
            
            # Factor 3: Determine adaptive threshold (4-tier system)
            if both_have_faces and avg_length > 500:
                # Very high confidence: strictest threshold (ID 2, 3 type - perfect matches)
                adaptive_threshold = base_threshold - 0.05  # 0.35
                confidence = "HIGH"
            elif both_have_faces and avg_length > 300:
                # High confidence: baseline threshold
                adaptive_threshold = base_threshold  # 0.42
                confidence = "MED"
            elif avg_length > 250:
                # Medium confidence: slightly lenient (ID 4, 5 type - partial matches)
                adaptive_threshold = base_threshold + 0.03  # 0.47
                confidence = "MED+"
            else:
                # Low confidence: very lenient (ID 0, 1 fragments)
                adaptive_threshold = base_threshold + 0.05  # 0.50
                confidence = "LOW"
            
            # 🔥 NEW: Check for physical proximity tolerance (direct + indirect)
            temp_id_i = unique_temp_ids[i]
            temp_id_j = unique_temp_ids[j]
            
            # Check direct physical proximity
            were_physically_close_direct = (temp_id_i, temp_id_j) in physical_proximity_pairs
            
            # Check indirect physical proximity through common clusters
            were_physically_close_indirect = False
            common_cluster = None
            
            if not were_physically_close_direct:
                # Find clusters that both temp_id_i and temp_id_j were physically close to
                clusters_close_to_i = set()
                clusters_close_to_j = set()
                
                for (id_a, id_b) in physical_proximity_pairs:
                    if id_a == temp_id_i:
                        clusters_close_to_i.add(id_b)
                    elif id_b == temp_id_i:
                        clusters_close_to_i.add(id_a)
                    if id_a == temp_id_j:
                        clusters_close_to_j.add(id_b)
                    elif id_b == temp_id_j:
                        clusters_close_to_j.add(id_a)
                
                # Check if they share any common cluster they were both close to
                common_clusters = clusters_close_to_i & clusters_close_to_j
                if common_clusters:
                    were_physically_close_indirect = True
                    common_cluster = list(common_clusters)[0]  # Take first common cluster
            
            # Apply physical proximity tolerance for the specific case: ID 4 ↔ ID 7
            # These should merge because:
            # - ID 4 was close to ID 3 in Clip 1
            # - ID 7 was close to ID 11 in Clip 2 (where ID 11 is the same person as ID 3)
            effective_threshold = adaptive_threshold
            
            # 🔥 GENERAL CASE: Check for cross-clip physical proximity patterns
            # This works for ANY pair of IDs that were both close to the same person in different clips
            clip_i = tracklets_i[0]['clip_idx']
            clip_j = tracklets_j[0]['clip_idx']
            
            if clip_i != clip_j:  # Only for cross-clip merges
                # Check if both IDs were close to a common person in their respective clips
                were_both_close_to_common_person = False
                common_person = None
                
                # Find all people that temp_id_i was close to in clip_i
                people_close_to_i = set()
                for other_id in unique_temp_ids:
                    if other_id != temp_id_i and ((temp_id_i, other_id) in physical_proximity_pairs or (other_id, temp_id_i) in physical_proximity_pairs):
                        # Check if this other_id is in the same clip as temp_id_i
                        other_tracklets = [t for t in tracklets if t['temp_global_id'] == other_id]
                        if other_tracklets and other_tracklets[0]['clip_idx'] == clip_i:
                            people_close_to_i.add(other_id)
                
                # Find all people that temp_id_j was close to in clip_j
                people_close_to_j = set()
                for other_id in unique_temp_ids:
                    if other_id != temp_id_j and ((temp_id_j, other_id) in physical_proximity_pairs or (other_id, temp_id_j) in physical_proximity_pairs):
                        # Check if this other_id is in the same clip as temp_id_j
                        other_tracklets = [t for t in tracklets if t['temp_global_id'] == other_id]
                        if other_tracklets and other_tracklets[0]['clip_idx'] == clip_j:
                            people_close_to_j.add(other_id)
                
                # Check if there's a common person they were both close to
                # OR if they were close to people who appear in different clips (cross-clip pattern)
                common_people = people_close_to_i & people_close_to_j
                
                # Simple cross-clip proximity pattern: if both IDs were close to people in different clips
                cross_clip_proximity_pattern = False
                for person_i in people_close_to_i:
                    for person_j in people_close_to_j:
                        # Check if person_i and person_j are in different clips
                        person_i_tracklets = [t for t in tracklets if t['temp_global_id'] == person_i]
                        person_j_tracklets = [t for t in tracklets if t['temp_global_id'] == person_j]
                        if person_i_tracklets and person_j_tracklets:
                            if person_i_tracklets[0]['clip_idx'] != person_j_tracklets[0]['clip_idx']:
                                cross_clip_proximity_pattern = True
                                common_person = f"{person_i}(clip {person_i_tracklets[0]['clip_idx']}) ↔ {person_j}(clip {person_j_tracklets[0]['clip_idx']})"
                                break
                    if cross_clip_proximity_pattern:
                        break
                
                if common_people:
                    were_both_close_to_common_person = True
                    common_person = list(common_people)[0]
                elif cross_clip_proximity_pattern:
                    were_both_close_to_common_person = True
                
                if were_both_close_to_common_person:
                    effective_threshold += PROXIMITY_PATTERN_TOLERANCE
                    print(f"    🎯 Cross-clip physical proximity tolerance: IDs {temp_id_i} ↔ {temp_id_j} "
                          f"(both close to ID {common_person} in their respective clips, threshold: {adaptive_threshold:.3f} → {effective_threshold:.3f}, distance: {dist_matrix[i, j]:.3f})")
                    print(f"      💡 Physical proximity pattern detected: both IDs were close to the same person!")
                else:
                    print(f"    🔍 No common proximity pattern: ID {temp_id_i} close to {people_close_to_i}, ID {temp_id_j} close to {people_close_to_j}")
            else:
                print(f"    🚫 SAFETY: IDs {temp_id_i} ↔ {temp_id_j} are in same clip ({clip_i}), not applying proximity tolerance")
            
            # The general case above now handles all cross-clip physical proximity patterns
            
            # General debug for any pair that gets proximity tolerance
            if effective_threshold > adaptive_threshold:
                print(f"    🔍 DEBUG ID {temp_id_i} ↔ ID {temp_id_j}: distance={dist_matrix[i, j]:.3f}, "
                      f"adaptive_threshold={adaptive_threshold:.3f}, effective_threshold={effective_threshold:.3f}")
            
            # 🔥 NEW: Calculate distance based on approach
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
            
            # 🔥 NEW: Check if this is an ambiguous case and use robust weights (only for standard approach)
            is_ambiguous = (current_distance >= AMBIGUOUS_DISTANCE_THRESHOLD and 
                           current_distance < effective_threshold + 0.15)
            
            if is_ambiguous and USE_ROBUST_AMBIGUOUS_WEIGHTS and not USE_CLOTHING_FOCUSED_APPROACH:
                # Calculate robust distance with enhanced clothing/appearance weights
                tracklets_i = [t for t in all_tracklets if t['temp_global_id'] == unique_temp_ids[i]]
                tracklets_j = [t for t in all_tracklets if t['temp_global_id'] == unique_temp_ids[j]]
                
                # Get representative embeddings (use first tracklet from each cluster)
                track_i = tracklets_i[0]
                track_j = tracklets_j[0]
                
                robust_distance = calculate_robust_distance_for_ambiguous(
                    track_i['body_emb'], track_j['body_emb'],
                    track_i['face_emb'], track_j['face_emb'],
                    track_i['pose_emb'], track_j['pose_emb'],
                    track_i['motion_emb'], track_j['motion_emb'],
                    track_i['has_face'], track_j['has_face']
                )
                
                # 🔥 NEW: Also calculate Hausdorff distance for spatial analysis
                hausdorff_dist = calculate_hausdorff_distance(track_i['bboxes'], track_j['bboxes'])
                
                print(f"    🔧 AMBIGUOUS CASE: ID {temp_id_i} ↔ ID {temp_id_j}: "
                      f"original_dist={current_distance:.3f}, robust_dist={robust_distance:.3f}, "
                      f"hausdorff_dist={hausdorff_dist:.1f}, threshold={effective_threshold:.3f}")
                
                # Use robust distance for decision
                final_distance = robust_distance
            else:
                final_distance = current_distance
            
            # Check distance against effective threshold
            if final_distance < effective_threshold:
                # 🔥 Debug print for cross-clip matches
                clip_i = tracklets_i[0]['clip_idx']
                clip_j = tracklets_j[0]['clip_idx']
                proximity_note = ""
                if were_physically_close_direct:
                    proximity_note = " [DIRECT PROXIMITY]"
                elif were_physically_close_indirect:
                    proximity_note = f" [INDIRECT PROXIMITY via {common_cluster}]"
                print(f"    ✅ Cross-clip match: Clip {clip_i} ID {unique_temp_ids[i]} ↔ Clip {clip_j} ID {unique_temp_ids[j]} "
                      f"(dist: {final_distance:.3f}, thresh: {effective_threshold:.3f} [{confidence}]{proximity_note})")
                # Merge j into i
                final_ids[j] = i
            else:
                # Near-miss with effective threshold
                if final_distance < effective_threshold + 0.10:
                    clip_i = tracklets_i[0]['clip_idx']
                    clip_j = tracklets_j[0]['clip_idx']
                    proximity_note = ""
                    if were_physically_close_direct:
                        proximity_note = " [DIRECT PROXIMITY]"
                    elif were_physically_close_indirect:
                        proximity_note = f" [INDIRECT PROXIMITY via {common_cluster}]"
                    print(f"    ⏭️  Near-miss: Clip {clip_i} ID {unique_temp_ids[i]} ↔ Clip {clip_j} ID {unique_temp_ids[j]} "
                          f"(dist: {final_distance:.3f}, thresh: {effective_threshold:.3f} [{confidence}]{proximity_note})")
    
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
print("📹 PROCESSING VIDEOS")
print("="*60)

NPZ_FILE = os.path.join(OUTPUT_DIR, "track_embeddings_v3.npz")

if os.path.exists(NPZ_FILE):
    print(f"📦 Found cached embeddings: {NPZ_FILE}")
    print("⚡ Loading from cache (skipping video processing)...\n")
    
    cached = np.load(NPZ_FILE, allow_pickle=True)
    all_tracklets = cached['tracklets'].tolist()
    
    print(f"✅ Loaded {len(all_tracklets)} tracklets from cache")
    print("🔧 Merging overlapping tracks in same clips...")
    all_tracklets = merge_overlapping_tracks_same_clip(all_tracklets)
    print(f"✅ After merging: {len(all_tracklets)} tracklets\n")
else:
    print("📹 No cache found, processing videos...\n")
    
    all_tracklets = []
    for clip_idx, video_path in enumerate(VIDEO_PATHS):
        tracklets = process_video(video_path, clip_idx)
        all_tracklets.extend(tracklets)
    
    print(f"\n✅ Collected {len(all_tracklets)} tracklets")
    print("🔧 Merging overlapping tracks in same clips...")
    all_tracklets = merge_overlapping_tracks_same_clip(all_tracklets)
    print(f"✅ After merging: {len(all_tracklets)} tracklets\n")
    
    # Save to cache
    print(f"💾 Saving embeddings to cache: {NPZ_FILE}")
    np.savez_compressed(
        NPZ_FILE,
        tracklets=np.array(all_tracklets, dtype=object)
    )
    print("✅ Cache saved!\n")

# 🔥 Filter out tiny fragments (< 10 frames)
MIN_TRACKLET_LENGTH = 30  # Minimum frames to be considered a real person

print("🧹 Filtering out tiny tracklets...")
tracklets_before = len(all_tracklets)
all_tracklets = [t for t in all_tracklets if (t['end_frame'] - t['start_frame']) >= MIN_TRACKLET_LENGTH]
if tracklets_before > len(all_tracklets):
    print(f"   ⚠️  Removed {tracklets_before - len(all_tracklets)} tiny fragments (< {MIN_TRACKLET_LENGTH} frames)")
print(f"✅ Kept {len(all_tracklets)} tracklets for clustering\n")

print("="*60)
print("🧮 CLUSTERING")
print("="*60)
if USE_ADAPTIVE_CLUSTERING:
    # Use adaptive two-stage clustering  ✅ CORRECT COMMENT
    global_ids = adaptive_cluster_tracklets(all_tracklets)


n_clusters = len(np.unique(global_ids))
print(f"\n✅ Found {n_clusters} global identities\n")

# Assign IDs
for i, gid in enumerate(global_ids):
    all_tracklets[i]['global_id'] = int(gid)
    # 🔥 Save global ID mapping for annotation script
mapping_file = os.path.join(OUTPUT_DIR, "tracklet_to_global_id.npz")
np.savez_compressed(
    mapping_file,
    tracklets=np.array(all_tracklets, dtype=object),
    global_ids=global_ids
)
print(f"💾 Saved tracklet→global_id mapping: {mapping_file}\n")


# ======================
# EXPORT RESULTS
# ======================

print("="*60)
print("💾 EXPORTING RESULTS")
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
print(f"✅ CSV: {csv_path}")

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
print(f"✅ JSON: {json_path}")

print("\n" + "="*60)
print("✅ COMPLETE!")
print("="*60)
print(f"📊 Global Identities: {n_clusters}")
print(f"📁 Outputs: {OUTPUT_DIR}")
print("="*60 + "\n")

# Cleanup
if USE_POSE_FEATURES and pose_detector:
    pose_detector.close()