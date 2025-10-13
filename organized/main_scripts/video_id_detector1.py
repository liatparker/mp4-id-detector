"""
multi_video_reid_UPGRADED_v2.py

🚀 UPGRADED COMPONENTS:
- YOLOv8 (Ultralytics) detection
- ByteTrack (Supervision) tracking - BETTER TRACKING!
- OSNet-AIN (torchreid) body embeddings - IMPROVED MODEL!
- FaceNet-PyTorch face embeddings - BETTER THAN INSIGHTFACE!
- Hybrid fusion (face/body) -> per-track averaged embedding
- Similarity-based clustering -> global IDs
- Exports CSV and hierarchical JSON

Usage:
    python video_id_detector1.py
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import torch
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import torchreid
import supervision as sv  # ByteTrack tracking (for future upgrade)
import insightface  # Face embeddings (FaceNet has SSL issues, keeping InsightFace for now)

# -----------------------
# Config
# -----------------------
dir = "/Users/liatparker/Documents/mp4_files_id_detectors/"
dir1 = "/Users/liatparker/Documents/mp4_files_id_detectors_upscaled/"
VIDEO_FILES = [dir1+"1_upscaled.mp4", dir1+"2_upscaled.mp4", dir1+"3_upscaled.mp4", dir1+"4_upscaled.mp4"]
YOLO_WEIGHTS = "yolov8n.pt"                  # choose yolov8n/yolov8s/yolov8m
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ReID / face settings
REID_MODEL_NAME = "osnet_ain_x1_0"   # 🚀 UPGRADED: OSNet-AIN (better than osnet_x1_0!)
FACE_MODEL_NAME = "buffalo_l"        # InsightFace model (FaceNet has SSL cert issues)

# Clustering with Hybrid Scoring: distance = α·dist(face) + (1-α)·dist(body) + β·dist(motion)
FACE_WEIGHT = 0.40               # α (alpha) - Higher for appearance-based separation!
MOTION_WEIGHT = 0.0             # β (beta) - TINY weight for cross-clip matching only!
DISTANCE_METRIC = "cosine"       # "cosine" (easier to tune!) or "euclidean"

# Super-Resolution for Better Embeddings (SLOW but improves discrimination!)
USE_SUPER_RESOLUTION = True     # Set to True to enable 2x upscaling before embedding extraction
                                 # ⚠️ Warning: ~1-2s per crop! Only use if clustering fails with preprocessing alone.

# Clustering Algorithm Selection
CLUSTERING_METHOD = "DBSCAN"   # "SIMILARITY" (threshold-based), "GMM" (auto), "HAC" (manual), "DBSCAN"

# SIMILARITY-BASED CLUSTERING (Recommended for ReID) ⭐
# Compares each track to existing clusters, assigns to most similar if below threshold
SIMILARITY_THRESHOLD = 0.42 # Stricter to separate clip 0, with cross-clip support from motion
SIMILARITY_MIN_SAMPLES = 1       # Min tracks per cluster (don't filter, SR improved quality)

# GMM Parameters (if using GMM) - AUTO DETECTS OPTIMAL N_CLUSTERS
GMM_MIN_CLUSTERS = 2             # Minimum number of clusters to try
GMM_MAX_CLUSTERS = 15            # Maximum number of clusters to try
GMM_CRITERION = "AIC"            # "BIC" (stricter) or "AIC" (more clusters)
GMM_COVARIANCE = "full"          # "full", "tied", "diag", "spherical"

# HAC Manual Parameters (if using "HAC")
N_CLUSTERS = 7                   # Number of people (only for manual HAC)
LINKAGE_METHOD = "average"       # "average", "complete", "single"

# DBSCAN Parameters (if using DBSCAN)
DBSCAN_EPS = 0.35           # Clustering threshold
DBSCAN_MIN_SAMPLES = 1           # Minimum appearances to form cluster

# Outputs
CSV_OUT = "global_identity_catalogue.csv"
JSON_OUT = "global_identity_catalogue.json"



# -----------------------
# Load models
# -----------------------
print("Device:", DEVICE)
print("Loading YOLOv8...")
yolo = YOLO(YOLO_WEIGHTS)  # requires ultralytics package

print(f"🚀 Loading UPGRADED OSNet-AIN ({REID_MODEL_NAME}) body embedding model...")
# Load improved torchreid model (OSNet-AIN is better than OSNet)
body_extractor = torchreid.models.build_model(
    name=REID_MODEL_NAME,
    num_classes=10,  # dummy value, we only use features
    pretrained=True
)
body_extractor.eval()
body_extractor.to(DEVICE)

print(f"Loading InsightFace ({FACE_MODEL_NAME}) for face embeddings...")
# Note: FaceNet-PyTorch has SSL cert issues. Will upgrade in future.
face_model = insightface.app.FaceAnalysis(name=FACE_MODEL_NAME)
face_model.prepare(ctx_id=0 if DEVICE.startswith("cuda") else -1)

# -----------------------
# Helpers
# -----------------------
def crop_and_safe(frame, xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]

def apply_super_resolution_2x(crop_bgr):
    """
    Apply 2x super-resolution using cv2.dnn_superres (fast, CPU-friendly).
    Falls back to bicubic if model not available.
    """
    try:
        # Try to use EDSR 2x model (lightweight SR)
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        # Download model first time: wget https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb
        model_path = "EDSR_x2.pb"
        if os.path.exists(model_path):
            sr.readModel(model_path)
            sr.setModel("edsr", 2)  # 2x upscale
            return sr.upsample(crop_bgr)
        else:
            # Fallback to bicubic (better than nothing)
            h, w = crop_bgr.shape[:2]
            return cv2.resize(crop_bgr, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    except Exception:
        # If SR fails, return original
        return crop_bgr

def enhance_crop_quality(crop_bgr):
    """
    Preprocessing pipeline to improve feature extraction:
    1. [Optional] Super-resolution - 2x upscale for better detail
    2. CLAHE - Normalize lighting/contrast across clips
    3. Mild sharpening - Enhance edges/features
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr
    
    try:
        # 0. Super-resolution (if enabled)
        if USE_SUPER_RESOLUTION:
            crop_bgr = apply_super_resolution_2x(crop_bgr)
        
        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Normalizes lighting - critical for matching across clips with different lighting
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l,a,b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 2. Mild sharpening - enhance edges without over-sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    except Exception as e:
        # If enhancement fails, return original
        return crop_bgr

def get_body_embedding_from_crop(bgr_crop):
    # Process crop for torchreid model
    try:
        img_rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        # Resize to model expected size (256x128 for OSNet)
        img_resized = cv2.resize(img_rgb, (128, 256))
        img = img_resized.astype("float32") / 255.0
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        tensor = torch.from_numpy(img).unsqueeze(0).to(DEVICE)
        
        # Extract features
        with torch.no_grad():
            emb = body_extractor(tensor)
            emb_np = emb.cpu().numpy().reshape(-1)
        
        # Normalize
        if np.linalg.norm(emb_np) > 0:
            emb_np = emb_np / np.linalg.norm(emb_np)
        return emb_np.astype(np.float32)
    except Exception:
        return None

def get_face_embedding_from_crop(bgr_crop):
    """
    InsightFace for face embeddings.
    Note: Will upgrade to FaceNet-PyTorch once SSL cert issue is resolved.
    """
    try:
        faces = face_model.get(bgr_crop)
    except Exception:
        faces = []
    if faces and len(faces) > 0 and hasattr(faces[0], "embedding"):
        emb = np.array(faces[0].embedding, dtype=np.float32)
        if np.linalg.norm(emb) > 0:
            emb = emb / np.linalg.norm(emb)
        return emb
    return None

def fuse_embeddings_hybrid(face_emb, body_emb, bbox=None, frame_num=None):
    """
    Store face and body embeddings separately for hybrid scoring.
    Now also stores bbox position for motion analysis.
    Returns a dict with embeddings + motion metadata.
    """
    has_face = face_emb is not None
    
    # Ensure face_emb has proper shape (use zeros if None)
    if face_emb is None:
        face_emb = np.zeros(512, dtype=np.float32)  # 512-dim for InsightFace
    
    return {
        'face': face_emb.astype(np.float32),
        'body': body_emb.astype(np.float32),
        'has_face': has_face,
        'bbox': bbox if bbox is not None else None,  # [x1, y1, x2, y2]
        'frame': frame_num if frame_num is not None else None
    }

def extract_motion_features(embs_list, fps=25.0):
    """
    Extract motion/behavior features from a track's frame-by-frame data.
    Returns a 5-dimensional motion feature vector.
    """
    if len(embs_list) < 2:
        # Not enough frames for motion analysis
        return np.zeros(5, dtype=np.float32)
    
    bboxes = [e['bbox'] for e in embs_list if e['bbox'] is not None]
    frames = [e['frame'] for e in embs_list if e['frame'] is not None]
    
    if len(bboxes) < 2:
        return np.zeros(5, dtype=np.float32)
    
    # 1. Average movement speed (pixels per second)
    centers = [((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2) for bbox in bboxes]
    displacements = [np.sqrt((centers[i+1][0]-centers[i][0])**2 + (centers[i+1][1]-centers[i][1])**2) 
                     for i in range(len(centers)-1)]
    avg_speed = np.mean(displacements) * fps if displacements else 0.0
    
    # 2. Movement variance (how erratic the movement is)
    speed_variance = np.std(displacements) if len(displacements) > 1 else 0.0
    
    # 3. Average bbox size (person scale)
    sizes = [(bbox[2]-bbox[0]) * (bbox[3]-bbox[1]) for bbox in bboxes]
    avg_size = np.mean(sizes) if sizes else 0.0
    
    # 4. Average vertical position (normalized 0-1, where in frame vertically)
    avg_y = np.mean([(bbox[1] + bbox[3])/2 for bbox in bboxes])
    
    # 5. Average horizontal position (normalized 0-1, where in frame horizontally)
    avg_x = np.mean([(bbox[0] + bbox[2])/2 for bbox in bboxes])
    
    # Normalize features to similar scales
    motion_vec = np.array([
        avg_speed / 100.0,      # Normalize speed (assuming max ~100 px/s)
        speed_variance / 50.0,  # Normalize variance
        avg_size / 100000.0,    # Normalize area (e.g., 400x300 = 120k px²)
        avg_y / 720.0,          # Normalize to typical frame height
        avg_x / 1280.0          # Normalize to typical frame width
    ], dtype=np.float32)
    
    return motion_vec

def _compute_motion_dist(emb1, emb2):
    """Helper to compute motion distance between two embeddings."""
    motion1 = emb1.get('motion', np.zeros(5, dtype=np.float32))
    motion2 = emb2.get('motion', np.zeros(5, dtype=np.float32))
    motion_sim = np.dot(motion1, motion2) / (np.linalg.norm(motion1) * np.linalg.norm(motion2) + 1e-8)
    return 1 - motion_sim

def compute_hybrid_distance(emb1, emb2, alpha=FACE_WEIGHT, motion_weight=MOTION_WEIGHT, metric=DISTANCE_METRIC):
    """
    Compute hybrid distance with appearance + motion/behavior:
    
    NEW LOGIC (per-comparison face weighting + configurable metric):
    - If both have faces: α·dist(face) + (1-α)·dist(body)
    - If only one has face: α·penalty + (1-α)·dist(body)
    - If neither has face: 1.0·dist(body)
    
    Distance metric: "cosine" (direction) or "euclidean" (magnitude + direction)
    
    Lower distance = more similar
    """
    # ===== APPEARANCE DISTANCE =====
    has_face1 = emb1.get('has_face', False)
    has_face2 = emb2.get('has_face', False)
    
    # Always compute body distance (always available)
    if metric == "euclidean":
        body_dist = np.linalg.norm(emb1['body'] - emb2['body'])
    else:  # cosine
        body_sim = np.dot(emb1['body'], emb2['body']) / (
            np.linalg.norm(emb1['body']) * np.linalg.norm(emb2['body']) + 1e-8
        )
        body_dist = 1 - body_sim
    
    # Compute face distance based on availability
    if has_face1 and has_face2:
        # CASE 1: Both have faces → Compute actual face distance
        if metric == "euclidean":
            face_dist = np.linalg.norm(emb1['face'] - emb2['face'])
        else:  # cosine
            face_sim = np.dot(emb1['face'], emb2['face']) / (
                np.linalg.norm(emb1['face']) * np.linalg.norm(emb2['face']) + 1e-8
            )
            face_dist = 1 - face_sim
        
    elif has_face1 or has_face2:
        # CASE 2: Only one has face → Use penalty distance
        # Default: assume faces are moderately different (not same, not opposite)
        # This allows face info to still contribute when available!
        face_dist = 0.6  # Moderate penalty (tune: 0.5=neutral, 0.7=more penalty)
        
    else:
        # CASE 3: Neither has face → No face information
        # Use body-only (set alpha to 0 for this comparison)
        appearance_dist = body_dist
        # Skip face weighting since no face data exists
        return appearance_dist if motion_weight == 0 else (1 - motion_weight) * appearance_dist + motion_weight * _compute_motion_dist(emb1, emb2)
    
    # Weighted combination: α*face + (1-α)*body
    appearance_dist = alpha * face_dist + (1 - alpha) * body_dist
    
    # ===== MOTION/BEHAVIOR DISTANCE =====
    if motion_weight > 0:
        motion_dist = _compute_motion_dist(emb1, emb2)
        # Blend appearance and motion
        return (1 - motion_weight) * appearance_dist + motion_weight * motion_dist
    else:
        return appearance_dist

def convert_embeddings_to_feature_vectors(tracklets):
    """
    Convert hybrid dict embeddings to flat feature vectors for GMM.
    Concatenates face + body embeddings into single vector.
    """
    features = []
    for t in tracklets:
        emb = t["embedding"]
        # Concatenate face and body (both 512-dim → 1024-dim total)
        feat = np.concatenate([emb['face'], emb['body']])
        features.append(feat)
    return np.array(features, dtype=np.float32)

# -----------------------
# Check if embeddings already exist - if so, load them instead of re-processing
# -----------------------
EMBEDDINGS_FILE = "track_embeddings.npz"
SKIP_VIDEO_PROCESSING = os.path.exists(EMBEDDINGS_FILE)

if SKIP_VIDEO_PROCESSING:
    print(f"\n{'='*60}")
    print(f"🎉 Found existing embeddings: {EMBEDDINGS_FILE}")
    print(f"⚡ Skipping video processing - loading saved embeddings...")
    print(f"💡 To re-extract embeddings, delete {EMBEDDINGS_FILE} first")
    print(f"{'='*60}\n")
    
    # Load saved embeddings (hybrid: face + body + motion)
    data = np.load(EMBEDDINGS_FILE)
    face_embeddings = data['face_embeddings']
    body_embeddings = data['body_embeddings']
    motion_embeddings = data.get('motion_embeddings', np.zeros((len(face_embeddings), 5), dtype=np.float32))  # Backward compat
    has_face = data['has_face']
    clips = data['clips']
    fps_array = data['fps']
    start_frames = data['start_frames']
    end_frames = data['end_frames']
    
    # Reconstruct all_tracklets from saved data
    all_tracklets = []
    for i in range(len(face_embeddings)):
        all_tracklets.append({
            "clip": int(clips[i]),
            "fps": float(fps_array[i]),
            "start_frame": int(start_frames[i]),
            "end_frame": int(end_frames[i]),
            "embedding": {
                'face': face_embeddings[i],
                'body': body_embeddings[i],
                'has_face': bool(has_face[i]),
                'motion': motion_embeddings[i]  # 🚀 Load motion features
            }
        })
    
    print(f"✅ Loaded {len(all_tracklets)} track embeddings from file")
    print(f"   Face: {face_embeddings.shape[1]}-dim, Body: {body_embeddings.shape[1]}-dim, Motion: {motion_embeddings.shape[1]}-dim")

# -----------------------
# Process each video with YOLOv8.track (ByteTrack)
# -----------------------
# NOTE: ultralytics YOLO.track supports built-in trackers. Use tracker="bytetrack.yaml" or "bytetrack".
# We'll call model.track(..., tracker="bytetrack.yaml") to obtain per-frame tracked boxes with track ids.
if not SKIP_VIDEO_PROCESSING:
    all_tracklets = []  # each entry: dict with clip, fps, start_frame, end_frame, embeddings list

    for clip_idx, video_path in enumerate(VIDEO_FILES):
        if not os.path.exists(video_path):
            print(f"Warning: {video_path} not found — skipping.")
            continue

        print(f"\nProcessing {video_path} (clip {clip_idx}) ...")
        # Run tracking inference using YOLOv8 track API. This will run detection+tracking and return per-frame results.
        # We request tracker byteTrack config with TUNED parameters for better multi-person tracking
        # classes=[0] restricts to person class
        # Using custom bytetrack_tuned.yaml with VERY AGGRESSIVE thresholds!
        # conf=0.25 is fine - detection working! Problem is clustering threshold
        results = yolo.track(source=video_path, tracker="bytetrack_tuned.yaml", classes=[0], conf=0.25, persist=True, verbose=False)

        # results is an iterable of Results for each frame; results[i].boxes contains tracked boxes with .xyxy and .id
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        # We'll build a dictionary for each track_id (local) with frame ranges and embeddings per frame
        per_local = {}  # key = (clip_idx, local_track_id)
        frame_counter = 0

        print("Iterating frames and extracting embeddings (this may be slow)...")
        for r in tqdm(results, desc=f"Frames {os.path.basename(video_path)}"):
            # r is a single-frame result, update frame counter
            frame_counter += 1

            # r.boxes may be empty for frames with no person
            if not hasattr(r, "boxes") or r.boxes is None or len(r.boxes) == 0:
                continue

            # r.boxes is a Boxes object with fields .xyxy, .conf, .cls, and .id (tracked id)
            # Convert to numpy arrays
            try:
                xyxy_arr = r.boxes.xyxy.cpu().numpy()    # Nx4
                id_arr = r.boxes.id.cpu().numpy() if r.boxes.id is not None else np.array([-1]*len(xyxy_arr))
            except Exception:
                # Fallback: try data access via .data
                try:
                    data = r.boxes.data.cpu().numpy()  # x1,y1,x2,y2,conf,class,id?
                    # not guaranteed; skip if can't parse
                    continue
                except Exception:
                    continue

            # Need an RGB/BGR image for cropping: r.orig_img available in ultralytics Result
            frame_img = getattr(r, "orig_img", None)
            if frame_img is None:
                continue  # can't crop

            for box_xyxy, track_id in zip(xyxy_arr, id_arr):
                # crop safe
                crop = crop_and_safe(frame_img, box_xyxy)
                if crop is None or crop.size == 0:
                    continue

                # 🎨 Enhance crop quality (lighting normalization + sharpening)
                crop = enhance_crop_quality(crop)
                
                # body and face embeddings
                body_emb = get_body_embedding_from_crop(crop)
                if body_emb is None:
                    continue    # skip if body embedding fails

                face_emb = get_face_embedding_from_crop(crop)   # may be None
                fused = fuse_embeddings_hybrid(face_emb, body_emb, bbox=box_xyxy, frame_num=frame_counter)

                key = (clip_idx, int(track_id))
                if key not in per_local:
                    per_local[key] = {
                        "clip": clip_idx,
                        "fps": fps,
                        "start_frame": frame_counter,
                        "end_frame": frame_counter,
                        "embs": [fused]
                    }
                else:
                    per_local[key]["end_frame"] = frame_counter
                    per_local[key]["embs"].append(fused)

        # Aggregate per local track -> produce one averaged embedding per appearance
        for key, d in per_local.items():
            embs_list = d["embs"]  # List of dicts with 'face', 'body', 'has_face'
            
            # Average face embeddings
            face_embs = [e['face'] for e in embs_list]
            avg_face = np.mean(face_embs, axis=0)
            if np.linalg.norm(avg_face) > 0:
                avg_face = avg_face / np.linalg.norm(avg_face)
            
            # Average body embeddings
            body_embs = [e['body'] for e in embs_list]
            avg_body = np.mean(body_embs, axis=0)
            if np.linalg.norm(avg_body) > 0:
                avg_body = avg_body / np.linalg.norm(avg_body)
            
            # Check if any frame had a face
            has_face = any(e['has_face'] for e in embs_list)
            
            # 🚀 NEW: Extract motion/behavior features
            motion_features = extract_motion_features(embs_list, fps=d["fps"])
            
            all_tracklets.append({
                "clip": d["clip"],
                "fps": d["fps"],
                "start_frame": int(d["start_frame"]),
                "end_frame": int(d["end_frame"]),
                "embedding": {
                    'face': avg_face.astype(np.float32),
                    'body': avg_body.astype(np.float32),
                    'has_face': has_face,
                    'motion': motion_features  # 🚀 5-dim motion/behavior vector
                }
            })

        # free memory for results (ultralytics stores frames)
        try:
            results.close()
        except Exception:
            pass

print(f"\nCollected {len(all_tracklets)} track appearances across clips.")

if len(all_tracklets) == 0:
    raise SystemExit("No tracklets collected. Check videos / detector.")

# -----------------------
# Save embeddings to file for later re-clustering (only if we just extracted them)
# -----------------------
if not SKIP_VIDEO_PROCESSING:
    print(f"Saving embeddings to {EMBEDDINGS_FILE}...")
    # Save all data needed for clustering (face, body, and motion separate)
    face_embs = np.stack([t["embedding"]['face'] for t in all_tracklets], axis=0)
    body_embs = np.stack([t["embedding"]['body'] for t in all_tracklets], axis=0)
    motion_embs = np.stack([t["embedding"]['motion'] for t in all_tracklets], axis=0)
    has_face = np.array([t["embedding"]['has_face'] for t in all_tracklets])
    
    np.savez(
        EMBEDDINGS_FILE,
        face_embeddings=face_embs,
        body_embeddings=body_embs,
        motion_embeddings=motion_embs,  # 🚀 NEW: Save motion features
        has_face=has_face,
        clips=np.array([t["clip"] for t in all_tracklets]),
        fps=np.array([t["fps"] for t in all_tracklets]),
        start_frames=np.array([t["start_frame"] for t in all_tracklets]),
        end_frames=np.array([t["end_frame"] for t in all_tracklets])
    )
    print(f"✅ Embeddings saved with motion features! You can now tune clustering parameters without re-extracting features.")

# -----------------------
# Clustering -> global IDs using hybrid distance
# -----------------------
print(f"Clustering embeddings with {CLUSTERING_METHOD} (hybrid: α={FACE_WEIGHT} face + {1-FACE_WEIGHT-MOTION_WEIGHT:.2f} body + β={MOTION_WEIGHT} motion)...")

# Create distance matrix using hybrid scoring
n = len(all_tracklets)
distance_matrix = np.zeros((n, n), dtype=np.float32)

# Count statistics (NEW: per-comparison face contribution)
both_faces_comparisons = 0      # Both have faces → actual face distance used
one_face_comparisons = 0        # One has face → penalty distance used (NEW!)
no_face_comparisons = 0         # Neither has face → body-only
total_comparisons = (n * (n - 1)) // 2

print("Computing hybrid distance matrix...")
for i in range(n):
    for j in range(i+1, n):
        # Check face availability for statistics
        has_face_i = all_tracklets[i]["embedding"].get('has_face', False)
        has_face_j = all_tracklets[j]["embedding"].get('has_face', False)
        
        if has_face_i and has_face_j:
            both_faces_comparisons += 1
        elif has_face_i or has_face_j:
            one_face_comparisons += 1  # NEW: One face still contributes!
        else:
            no_face_comparisons += 1
        
        dist = compute_hybrid_distance(
            all_tracklets[i]["embedding"],
            all_tracklets[j]["embedding"],
            alpha=FACE_WEIGHT
        )
        distance_matrix[i, j] = dist
        distance_matrix[j, i] = dist  # symmetric

print(f"✅ Distance matrix computed (NEW: per-comparison face weighting!):")
print(f"   {both_faces_comparisons}/{total_comparisons} comparisons: both have faces (actual face distance)")
print(f"   {one_face_comparisons}/{total_comparisons} comparisons: one has face (penalty distance) 🆕")
print(f"   {no_face_comparisons}/{total_comparisons} comparisons: neither has face (body-only)")

# Run clustering based on selected method
if CLUSTERING_METHOD == "SIMILARITY":
    print(f"Running Similarity-Based Clustering (threshold={SIMILARITY_THRESHOLD})...")
    print("💡 This method compares each track to existing clusters and assigns to the most similar one.")
    
    # Initialize clusters (each cluster is a list of track indices)
    clusters = []
    labels = np.full(n, -1, dtype=int)  # -1 means unassigned
    
    # Process each track sequentially
    for i in range(n):
        best_cluster_idx = -1
        best_distance = float('inf')
        
        # Compare to all existing clusters
        for cluster_idx, cluster_members in enumerate(clusters):
            # Compute average distance to all members of this cluster
            distances_to_cluster = [distance_matrix[i, j] for j in cluster_members]
            avg_distance = np.mean(distances_to_cluster)
            
            if avg_distance < best_distance:
                best_distance = avg_distance
                best_cluster_idx = cluster_idx
        
        # Assign to best cluster if below threshold, otherwise create new cluster
        if best_distance <= SIMILARITY_THRESHOLD and best_cluster_idx != -1:
            clusters[best_cluster_idx].append(i)
            labels[i] = best_cluster_idx
        else:
            # Create new cluster
            new_cluster_idx = len(clusters)
            clusters.append([i])
            labels[i] = new_cluster_idx
    
    # Filter clusters by minimum size
    if SIMILARITY_MIN_SAMPLES > 1:
        filtered_labels = np.full(n, -1, dtype=int)  # -1 for noise/singletons
        new_cluster_id = 0
        filtered_count = 0
        
        for cluster_idx, cluster_members in enumerate(clusters):
            if len(cluster_members) >= SIMILARITY_MIN_SAMPLES:
                for member_idx in cluster_members:
                    filtered_labels[member_idx] = new_cluster_id
                new_cluster_id += 1
            else:
                filtered_count += len(cluster_members)
        
        labels = filtered_labels
        n_clusters = new_cluster_id
        print(f"✅ Similarity clustering completed: {n_clusters} clusters, {filtered_count} singletons filtered")
    else:
        n_clusters = len(clusters)
        print(f"✅ Similarity clustering completed: {n_clusters} clusters (no filtering)")
    
elif CLUSTERING_METHOD == "GMM":
    print(f"Running Gaussian Mixture Model (auto-detecting optimal clusters)...")
    print(f"Testing {GMM_MIN_CLUSTERS} to {GMM_MAX_CLUSTERS} clusters using {GMM_CRITERION}...")
    
    # Convert embeddings to feature vectors for GMM
    feature_vectors = convert_embeddings_to_feature_vectors(all_tracklets)
    
    # Try different numbers of clusters and select best based on BIC/AIC
    best_score = np.inf if GMM_CRITERION == "BIC" else np.inf
    best_n_clusters = GMM_MIN_CLUSTERS
    best_gmm = None
    scores = []
    
    for n in range(GMM_MIN_CLUSTERS, GMM_MAX_CLUSTERS + 1):
        gmm = GaussianMixture(
            n_components=n,
            covariance_type=GMM_COVARIANCE,
            random_state=42,
            max_iter=200
        )
        gmm.fit(feature_vectors)
        
        if GMM_CRITERION == "BIC":
            score = gmm.bic(feature_vectors)
        else:  # AIC
            score = gmm.aic(feature_vectors)
        
        scores.append((n, score))
        print(f"  n={n:2d}: {GMM_CRITERION}={score:.2f}")
        
        if score < best_score:
            best_score = score
            best_n_clusters = n
            best_gmm = gmm
    
    labels = best_gmm.predict(feature_vectors)
    print(f"✅ GMM completed: {best_n_clusters} clusters (auto-detected via {GMM_CRITERION})")
    print(f"   Best {GMM_CRITERION} score: {best_score:.2f}")
    
elif CLUSTERING_METHOD == "HAC":
    print(f"Running Hierarchical Agglomerative Clustering (n_clusters={N_CLUSTERS}, linkage={LINKAGE_METHOD})...")
    clusterer = AgglomerativeClustering(
        n_clusters=N_CLUSTERS,
        metric='precomputed',
        linkage=LINKAGE_METHOD
    )
    labels = clusterer.fit_predict(distance_matrix)
    print(f"✅ HAC completed: {N_CLUSTERS} clusters guaranteed")
    
elif CLUSTERING_METHOD == "DBSCAN":
    print(f"Running DBSCAN (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})...")
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="precomputed")
    labels = db.fit_predict(distance_matrix)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"✅ DBSCAN completed: {n_clusters} clusters, {n_noise} noise points")
    
else:
    raise ValueError(f"Unknown clustering method: {CLUSTERING_METHOD}")

# Map labels to consecutive global IDs
# Note: GMM and HAC don't have noise (-1), DBSCAN and SIMILARITY might have noise
label_to_gid = {}
next_gid = 0

if CLUSTERING_METHOD in ["GMM", "HAC"]:
    # GMM/HAC: All labels are valid clusters
    for lbl in sorted(set(labels)):
        label_to_gid[lbl] = next_gid
        next_gid += 1
else:
    # DBSCAN/SIMILARITY: Skip noise points (-1)
    for lbl in sorted(set(labels)):
        if lbl == -1:
            continue
        label_to_gid[lbl] = next_gid
        next_gid += 1

# Attach global id to each track
for idx, t in enumerate(all_tracklets):
    lbl = labels[idx]
    if lbl in label_to_gid:
        gid = label_to_gid[lbl]
        t["global_id"] = int(gid)
    else:
        t["global_id"] = None  # Mark noise as None (will be filtered out in DBSCAN)

# -----------------------
# Export CSV (only clustered identities, no noise)
# -----------------------
print("Exporting CSV...")
rows = []
for t in all_tracklets:
    if t["global_id"] is not None:  # Skip noise
        rows.append({
            "global_id": t["global_id"],
            "clip_id": t["clip"],
            "start_frame": t["start_frame"],
            "end_frame": t["end_frame"]
        })
df = pd.DataFrame(rows)[["global_id", "clip_id", "start_frame", "end_frame"]]
df.to_csv(CSV_OUT, index=False)
print(f"Saved CSV -> {CSV_OUT}")

# -----------------------
# Export hierarchical JSON (sorted by global_id, only clusters no noise)
# -----------------------
print("Exporting JSON...")
artifact = {}
noise_count = 0
for t in all_tracklets:
    if t["global_id"] is None:  # Skip noise
        noise_count += 1
        continue
    gid_key = f"global_id_{t['global_id']}"
    if gid_key not in artifact:
        artifact[gid_key] = {"appearances": []}
    artifact[gid_key]["appearances"].append({
        "clip_id": t["clip"],
        "frame_range": [t["start_frame"], t["end_frame"]],
        "time_span": [t["start_frame"] / t["fps"], t["end_frame"] / t["fps"]]
    })

# Sort by global_id numerically
sorted_artifact = {}
for gid_key in sorted(artifact.keys(), key=lambda x: int(x.split('_')[-1])):
    sorted_artifact[gid_key] = artifact[gid_key]
    # Sort appearances by clip_id, then by start_frame
    sorted_artifact[gid_key]["appearances"].sort(key=lambda x: (x["clip_id"], x["frame_range"][0]))

# Create final output with summary
clustering_params = {
    "algorithm": CLUSTERING_METHOD,
    "alpha": FACE_WEIGHT,
    "beta": MOTION_WEIGHT,  # 🚀 NEW: Motion/behavior weight
    "method": "hybrid_with_motion"  # Updated to reflect motion features
}
if CLUSTERING_METHOD == "SIMILARITY":
    clustering_params["threshold"] = SIMILARITY_THRESHOLD
    clustering_params["min_samples"] = SIMILARITY_MIN_SAMPLES
elif CLUSTERING_METHOD == "GMM":
    clustering_params["n_clusters_detected"] = len(set(labels))
    clustering_params["criterion"] = GMM_CRITERION
    clustering_params["covariance"] = GMM_COVARIANCE
    clustering_params["search_range"] = [GMM_MIN_CLUSTERS, GMM_MAX_CLUSTERS]
elif CLUSTERING_METHOD == "HAC":
    clustering_params["n_clusters"] = N_CLUSTERS
    clustering_params["linkage"] = LINKAGE_METHOD
else:  # DBSCAN
    clustering_params["eps"] = DBSCAN_EPS
    clustering_params["min_samples"] = DBSCAN_MIN_SAMPLES

final_output = {
    "summary": {
        "total_global_ids": len(sorted_artifact),
        "total_appearances": sum(len(v["appearances"]) for v in sorted_artifact.values()),
        "noise_filtered": noise_count,
        "clustering_parameters": clustering_params
    },
    "identities": sorted_artifact
}

with open(JSON_OUT, "w") as f:
    json.dump(final_output, f, indent=2)

print(f"Saved JSON -> {JSON_OUT}")
print(f"  → {len(sorted_artifact)} global IDs (actual clusters/people)")
print(f"  → {sum(len(v['appearances']) for v in sorted_artifact.values())} total appearances")
print(f"  → {noise_count} noise points filtered out")

print("\n" + "=" * 60)
print("✅ CLUSTERING COMPLETE!")
print("=" * 60)
print(f"Method: {CLUSTERING_METHOD} + Hybrid Scoring (α·face + body + β·motion)")
print(f"Parameters:")
print(f"  CLUSTERING_METHOD: {CLUSTERING_METHOD}")
print(f"  FACE_WEIGHT (α): {FACE_WEIGHT}")
print(f"  MOTION_WEIGHT (β): {MOTION_WEIGHT}  # 🚀 NEW: Behavior/movement features")
if CLUSTERING_METHOD == "SIMILARITY":
    print(f"  SIMILARITY_THRESHOLD: {SIMILARITY_THRESHOLD}")
    print(f"  SIMILARITY_MIN_SAMPLES: {SIMILARITY_MIN_SAMPLES}")
elif CLUSTERING_METHOD == "GMM":
    print(f"  GMM_CRITERION: {GMM_CRITERION} (auto-detection)")
    print(f"  GMM_COVARIANCE: {GMM_COVARIANCE}")
    print(f"  Search range: {GMM_MIN_CLUSTERS}-{GMM_MAX_CLUSTERS}")
elif CLUSTERING_METHOD == "HAC":
    print(f"  N_CLUSTERS: {N_CLUSTERS}")
    print(f"  LINKAGE_METHOD: {LINKAGE_METHOD}")
else:  # DBSCAN
    print(f"  DBSCAN_EPS: {DBSCAN_EPS}")
    print(f"  DBSCAN_MIN_SAMPLES: {DBSCAN_MIN_SAMPLES}")
print(f"\nResults:")
print(f"  {len(sorted_artifact)} people identified")
print(f"  {sum(len(v['appearances']) for v in sorted_artifact.values())} valid appearances")
if CLUSTERING_METHOD in ["DBSCAN", "SIMILARITY"]:
    print(f"  {noise_count} noise points filtered")
print("=" * 60)
print("\n💡 To adjust clustering, modify these parameters at the top of the script:")
print(f"   CLUSTERING_METHOD = '{CLUSTERING_METHOD}'  # 'SIMILARITY' (⭐recommended), 'GMM', 'HAC', 'DBSCAN'")
print(f"   FACE_WEIGHT = {FACE_WEIGHT}  # Higher = more face weight (0.0-1.0)")
print(f"   MOTION_WEIGHT = {MOTION_WEIGHT}  # 🚀 Higher = more behavior/movement weight (0.0-0.3)")
if CLUSTERING_METHOD == "SIMILARITY":
    print(f"   SIMILARITY_THRESHOLD = {SIMILARITY_THRESHOLD}  # Lower = stricter, Higher = looser")
    print(f"   SIMILARITY_MIN_SAMPLES = {SIMILARITY_MIN_SAMPLES}  # Min appearances per person")
elif CLUSTERING_METHOD == "GMM":
    print(f"   GMM_MIN_CLUSTERS = {GMM_MIN_CLUSTERS}  # Minimum clusters to try")
    print(f"   GMM_MAX_CLUSTERS = {GMM_MAX_CLUSTERS}  # Maximum clusters to try")
    print(f"   GMM_CRITERION = '{GMM_CRITERION}'  # 'BIC' (stricter) or 'AIC' (more clusters)")
elif CLUSTERING_METHOD == "HAC":
    print(f"   N_CLUSTERS = {N_CLUSTERS}  # Number of people")
    print(f"   LINKAGE_METHOD = '{LINKAGE_METHOD}'  # 'average', 'complete', or 'single'")
else:  # DBSCAN
    print(f"   DBSCAN_EPS = {DBSCAN_EPS}  # Lower = stricter, Higher = looser")
    print(f"   DBSCAN_MIN_SAMPLES = {DBSCAN_MIN_SAMPLES}  # Min appearances per person")
print("\n🎬 Next: Visualize results with:")
print("   python export_annotated_videos.py --clip 0 --frames 1-100")
print("Done.")
