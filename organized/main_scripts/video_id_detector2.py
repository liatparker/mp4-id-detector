"""
video_id_detector2.py - ADVANCED REID WITH RE-RANKING, CAMERA BIAS, AND POSE

🚀 FEATURES:
1. ✅ Track-level averaged embeddings (face + body + motion)
2. ✅ k-Reciprocal Re-ranking for better similarity
3. ✅ Camera ID bias correction (cross-clip penalty)
4. ✅ Pose-aware features (MediaPipe body keypoints)
5. ✅ Hybrid scoring: α·face + β·body + γ·motion + δ·pose

COMPONENTS:
- YOLOv8 detection
- ByteTrack tracking
- OSNet-AIN body embeddings
- InsightFace face embeddings
- MediaPipe pose keypoints
- Advanced clustering with re-ranking

Usage:
    python video_id_detector2.py
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
import supervision as sv
import insightface
import mediapipe as mp
from scipy.spatial.distance import cdist

# -----------------------
# Config
# -----------------------
dir = "/Users/liatparker/Documents/mp4_files_id_detectors/"
dir1 = "/Users/liatparker/Documents/mp4_files_id_detectors_upscaled/"
VIDEO_FILES = [dir1+"1_upscaled.mp4", dir1+"2_upscaled.mp4", dir1+"3_upscaled.mp4", dir1+"4_upscaled.mp4"]
YOLO_WEIGHTS = "yolov8n.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ReID / face settings
REID_MODEL_NAME = "osnet_ain_x1_0"
FACE_MODEL_NAME = "buffalo_l"

# 🚀 ADVANCED: Hybrid Scoring Weights
FACE_WEIGHT = 0.35          # α - Face appearance
BODY_WEIGHT = 0.45          # Implicit (1 - α - γ - δ)
MOTION_WEIGHT = 0.05        # γ - Movement patterns
POSE_WEIGHT = 0.15          # δ - Body structure/pose
DISTANCE_METRIC = "cosine"

# 🚀 NEW: Camera/Clip Bias Correction
USE_CAMERA_BIAS = True
CAMERA_BIAS_WEIGHT = 0.15   # Penalty for cross-clip matches (0.0 = none, 0.3 = strong)

# 🚀 NEW: k-Reciprocal Re-ranking
USE_RERANKING = True
K_RECIPROCAL = 15           # k value for k-reciprocal nearest neighbors
LAMBDA_VALUE = 0.3          # Weight: (1-λ)·original + λ·reranked

# 🚀 NEW: Pose-aware features
USE_POSE_FEATURES = True
POSE_CONFIDENCE_THRESHOLD = 0.5

# Super-Resolution
USE_SUPER_RESOLUTION = False

# Clustering
CLUSTERING_METHOD = "SIMILARITY"
SIMILARITY_THRESHOLD = 0.40
SIMILARITY_MIN_SAMPLES = 1

# DBSCAN
DBSCAN_EPS = 0.35
DBSCAN_MIN_SAMPLES = 1

# Outputs
CSV_OUT = "global_identity_catalogue_v2.csv"
JSON_OUT = "global_identity_catalogue_v2.json"
EMBEDDINGS_FILE = "track_embeddings_v2.npz"

# -----------------------
# Load models
# -----------------------
print("="*60)
print("🚀 INITIALIZING ADVANCED REID SYSTEM v2")
print("="*60)
print(f"Device: {DEVICE}")
print(f"Camera Bias Correction: {USE_CAMERA_BIAS}")
print(f"k-Reciprocal Re-ranking: {USE_RERANKING}")
print(f"Pose Features: {USE_POSE_FEATURES}")
print("="*60)

print("\n📦 Loading YOLOv8...")
yolo = YOLO(YOLO_WEIGHTS)

print(f"📦 Loading OSNet-AIN ({REID_MODEL_NAME})...")
body_extractor = torchreid.models.build_model(
    name=REID_MODEL_NAME,
    num_classes=10,
    pretrained=True
)
body_extractor.eval()
body_extractor.to(DEVICE)

print(f"📦 Loading InsightFace ({FACE_MODEL_NAME})...")
face_model = insightface.app.FaceAnalysis(name=FACE_MODEL_NAME)
face_model.prepare(ctx_id=0 if DEVICE.startswith("cuda") else -1)

# 🚀 NEW: Initialize MediaPipe Pose
if USE_POSE_FEATURES:
    print("📦 Loading MediaPipe Pose for keypoint extraction...")
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=POSE_CONFIDENCE_THRESHOLD
    )
else:
    pose_detector = None

print("✅ All models loaded!\n")

# -----------------------
# Helper Functions
# -----------------------

def preprocess_crop(crop):
    """CLAHE + sharpening"""
    if crop is None or crop.size == 0:
        return crop
    try:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l,a,b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        return enhanced
    except:
        return crop

def super_resolve_crop(crop):
    """Optional 2x upscaling"""
    if crop is None or crop.size == 0:
        return crop
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel("ESPCN_x2.pb")
        sr.setModel("espcn", 2)
        return sr.upsample(crop)
    except:
        return cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

def extract_pose_features(crop, pose_detector):
    """
    🚀 NEW: Extract pose keypoint features using MediaPipe.
    Returns 66-dimensional normalized keypoint vector (33 landmarks × 2 coords).
    
    Captures body structure:
    - Upper body: shoulders, elbows, wrists
    - Lower body: hips, knees, ankles
    - Torso proportions and posture
    """
    if pose_detector is None or not USE_POSE_FEATURES:
        return np.zeros(66, dtype=np.float32)
    
    try:
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(rgb_crop)
        
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y])
            
            pose_vec = np.array(landmarks, dtype=np.float32)
            
            # Normalize
            if np.linalg.norm(pose_vec) > 0:
                pose_vec = pose_vec / np.linalg.norm(pose_vec)
            
            return pose_vec
        else:
            return np.zeros(66, dtype=np.float32)
    
    except Exception as e:
        return np.zeros(66, dtype=np.float32)

def extract_motion_features(embs_list, fps=30.0):
    """
    Extract motion/behavior features from track.
    Returns 5-dimensional vector: [speed, direction_var, size, vertical_pos, horizontal_pos]
    """
    if len(embs_list) == 0:
        return np.zeros(5, dtype=np.float32)
    
    # Extract movement data
    movements = []
    sizes = []
    v_positions = []
    h_positions = []
    
    for e in embs_list:
        if 'movement' in e:
            movements.append(e['movement'])
        if 'bbox_size' in e:
            sizes.append(e['bbox_size'])
        if 'v_pos' in e:
            v_positions.append(e['v_pos'])
        if 'h_pos' in e:
            h_positions.append(e['h_pos'])
    
    # Compute statistics
    avg_speed = np.mean([m for m in movements]) if movements else 0.0
    direction_var = np.std([m for m in movements]) if len(movements) > 1 else 0.0
    avg_size = np.mean(sizes) if sizes else 0.5
    avg_v_pos = np.mean(v_positions) if v_positions else 0.5
    avg_h_pos = np.mean(h_positions) if h_positions else 0.5
    
    return np.array([avg_speed, direction_var, avg_size, avg_v_pos, avg_h_pos], dtype=np.float32)

def extract_embeddings_from_crop(crop, body_extractor, face_model, pose_detector, device):
    """
    Extract face + body + pose embeddings from crop.
    """
    # Preprocess
    crop = preprocess_crop(crop)
    
    if USE_SUPER_RESOLUTION:
        crop = super_resolve_crop(crop)
    
    h, w = crop.shape[:2]
    
    # Face embedding
    face_emb = np.zeros(512, dtype=np.float32)
    has_face = False
    try:
        faces = face_model.get(crop)
        if faces:
            face_emb = faces[0].embedding
            has_face = True
            if np.linalg.norm(face_emb) > 0:
                face_emb = face_emb / np.linalg.norm(face_emb)
    except:
        pass
    
    # Body embedding
    body_emb = np.zeros(512, dtype=np.float32)
    try:
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 256))
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = body_extractor(img_tensor)
            body_emb = features.cpu().numpy().flatten()
            if np.linalg.norm(body_emb) > 0:
                body_emb = body_emb / np.linalg.norm(body_emb)
    except:
        pass
    
    # 🚀 NEW: Pose embedding
    pose_emb = extract_pose_features(crop, pose_detector)
    
    return {
        'face': face_emb,
        'body': body_emb,
        'has_face': has_face,
        'pose': pose_emb
    }

def k_reciprocal_rerank(dist_matrix, k=20, lambda_value=0.3):
    """
    🚀 NEW: k-Reciprocal Re-ranking
    
    Improves distance matrix by considering k-reciprocal nearest neighbors.
    If A is in k-NN of B AND B is in k-NN of A, they are k-reciprocal neighbors.
    
    Args:
        dist_matrix: NxN distance matrix
        k: Number of nearest neighbors
        lambda_value: Weight for re-ranked distance
    
    Returns:
        Re-ranked distance matrix
    """
    N = dist_matrix.shape[0]
    reranked_dist = np.zeros_like(dist_matrix)
    
    for i in range(N):
        # Get k nearest neighbors of i
        k_neighbors_i = np.argsort(dist_matrix[i])[:k+1]  # +1 to exclude self
        
        for j in range(N):
            if i == j:
                reranked_dist[i, j] = 0.0
                continue
            
            # Get k nearest neighbors of j
            k_neighbors_j = np.argsort(dist_matrix[j])[:k+1]
            
            # Check if i and j are k-reciprocal neighbors
            is_reciprocal = (j in k_neighbors_i) and (i in k_neighbors_j)
            
            if is_reciprocal:
                # Compute Jaccard distance of k-reciprocal neighbors
                neighbors_i = set(k_neighbors_i)
                neighbors_j = set(k_neighbors_j)
                jaccard = len(neighbors_i & neighbors_j) / len(neighbors_i | neighbors_j)
                
                # Re-ranked distance (lower if more overlap)
                reranked_dist[i, j] = (1 - jaccard) * dist_matrix[i, j]
            else:
                # Not reciprocal, use original distance
                reranked_dist[i, j] = dist_matrix[i, j]
    
    # Combine original and re-ranked
    final_dist = (1 - lambda_value) * dist_matrix + lambda_value * reranked_dist
    
    return final_dist

def compute_hybrid_distance_matrix(tracklets, use_reranking=True, use_camera_bias=True):
    """
    🚀 ADVANCED: Compute hybrid distance matrix with:
    1. Face + Body + Motion + Pose fusion
    2. Camera bias correction (cross-clip penalty)
    3. k-Reciprocal re-ranking
    """
    N = len(tracklets)
    
    # Extract embeddings and metadata
    face_embs = np.array([t['embedding']['face'] for t in tracklets])
    body_embs = np.array([t['embedding']['body'] for t in tracklets])
    motion_embs = np.array([t['embedding']['motion'] for t in tracklets])
    pose_embs = np.array([t['embedding']['pose'] for t in tracklets])
    has_faces = np.array([t['embedding']['has_face'] for t in tracklets])
    clip_ids = np.array([t['clip_id'] for t in tracklets])
    
    # Compute individual distance matrices
    if DISTANCE_METRIC == "cosine":
        face_dist = 1 - np.dot(face_embs, face_embs.T)
        body_dist = 1 - np.dot(body_embs, body_embs.T)
        motion_dist = 1 - np.dot(motion_embs, motion_embs.T)
        pose_dist = 1 - np.dot(pose_embs, pose_embs.T)
    else:  # euclidean
        face_dist = cdist(face_embs, face_embs, metric='euclidean')
        body_dist = cdist(body_embs, body_embs, metric='euclidean')
        motion_dist = cdist(motion_embs, motion_embs, metric='euclidean')
        pose_dist = cdist(pose_embs, pose_embs, metric='euclidean')
    
    # Normalize distances to [0, 1]
    face_dist = (face_dist - face_dist.min()) / (face_dist.max() - face_dist.min() + 1e-8)
    body_dist = (body_dist - body_dist.min()) / (body_dist.max() - body_dist.min() + 1e-8)
    motion_dist = (motion_dist - motion_dist.min()) / (motion_dist.max() - motion_dist.min() + 1e-8)
    pose_dist = (pose_dist - pose_dist.min()) / (pose_dist.max() - pose_dist.min() + 1e-8)
    
    # Hybrid fusion
    dist_matrix = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            # Adaptive face weight
            if has_faces[i] and has_faces[j]:
                alpha = FACE_WEIGHT
            else:
                alpha = 0.0  # No face, rely on body+pose
            
            # Compute weighted distance
            dist = (alpha * face_dist[i, j] +
                   (1 - alpha - MOTION_WEIGHT - POSE_WEIGHT) * body_dist[i, j] +
                   MOTION_WEIGHT * motion_dist[i, j] +
                   POSE_WEIGHT * pose_dist[i, j])
            
            dist_matrix[i, j] = dist
    
    #  Camera bias correction
    if use_camera_bias and USE_CAMERA_BIAS:
        for i in range(N):
            for j in range(N):
                if clip_ids[i] != clip_ids[j]:
                    # Cross-clip: add penalty
                    dist_matrix[i, j] += CAMERA_BIAS_WEIGHT
    
    #  k-Reciprocal re-ranking
    if use_reranking and USE_RERANKING:
        print(f"   🔄 Applying k-reciprocal re-ranking (k={K_RECIPROCAL}, λ={LAMBDA_VALUE})...")
        dist_matrix = k_reciprocal_rerank(dist_matrix, k=K_RECIPROCAL, lambda_value=LAMBDA_VALUE)
    
    return dist_matrix

# -----------------------
# Main Processing
# -----------------------

print("="*60)
print("📹 PROCESSING VIDEOS")
print("="*60)

all_tracklets = []

for clip_idx, video_file in enumerate(VIDEO_FILES):
    print(f"\n🎬 Clip {clip_idx}: {os.path.basename(video_file)}")
    
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # ByteTrack tracker
    tracker = sv.ByteTrack()
    
    per_local = {}
    frame_counter = 0
    
    pbar = tqdm(total=total_frames, desc=f"Clip {clip_idx}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO detection
        results = yolo(frame, verbose=False, classes=[0])  # person only
        
        if len(results) > 0 and results[0].boxes is not None:
            detections = sv.Detections.from_ultralytics(results[0])
            detections = tracker.update_with_detections(detections)
            
            for i, track_id in enumerate(detections.tracker_id):
                bbox = detections.xyxy[i]
                x1, y1, x2, y2 = [int(v) for v in bbox]
                
                # Crop person
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                # Extract embeddings
                emb = extract_embeddings_from_crop(crop, body_extractor, face_model, pose_detector, DEVICE)
                
                # Add movement/bbox data
                emb['movement'] = np.linalg.norm([x2-x1, y2-y1])
                emb['bbox_size'] = (x2-x1) * (y2-y1) / (frame.shape[0] * frame.shape[1])
                emb['v_pos'] = (y1 + y2) / (2 * frame.shape[0])
                emb['h_pos'] = (x1 + x2) / (2 * frame.shape[1])
                
                key = (clip_idx, track_id)
                if key not in per_local:
                    per_local[key] = {
                        "clip": os.path.basename(video_file),
                        "clip_id": clip_idx,
                        "fps": fps,
                        "start_frame": frame_counter,
                        "end_frame": frame_counter,
                        "embs": [emb]
                    }
                else:
                    per_local[key]["end_frame"] = frame_counter
                    per_local[key]["embs"].append(emb)
        
        frame_counter += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    # Aggregate per track
    for key, d in per_local.items():
        embs_list = d["embs"]
        
        # Average face
        face_embs = [e['face'] for e in embs_list]
        avg_face = np.mean(face_embs, axis=0)
        if np.linalg.norm(avg_face) > 0:
            avg_face = avg_face / np.linalg.norm(avg_face)
        
        # Average body
        body_embs = [e['body'] for e in embs_list]
        avg_body = np.mean(body_embs, axis=0)
        if np.linalg.norm(avg_body) > 0:
            avg_body = avg_body / np.linalg.norm(avg_body)
        
        # Average pose
        pose_embs = [e['pose'] for e in embs_list]
        avg_pose = np.mean(pose_embs, axis=0)
        if np.linalg.norm(avg_pose) > 0:
            avg_pose = avg_pose / np.linalg.norm(avg_pose)
        
        # Check if any frame had face
        has_face = any(e['has_face'] for e in embs_list)
        
        # Motion features
        motion_features = extract_motion_features(embs_list, fps=fps)
        
        all_tracklets.append({
            "clip": d["clip"],
            "clip_id": d["clip_id"],
            "fps": d["fps"],
            "start_frame": int(d["start_frame"]),
            "end_frame": int(d["end_frame"]),
            "embedding": {
                'face': avg_face.astype(np.float32),
                'body': avg_body.astype(np.float32),
                'has_face': has_face,
                'motion': motion_features,
                'pose': avg_pose.astype(np.float32)
            }
        })

print(f"\n✅ Collected {len(all_tracklets)} tracklets\n")

# -----------------------
# Clustering with Advanced Features
# -----------------------

print("="*60)
print("🔗 CLUSTERING WITH ADVANCED FEATURES")
print("="*60)

# Compute hybrid distance matrix
print("📊 Computing hybrid distance matrix...")
dist_matrix = compute_hybrid_distance_matrix(all_tracklets, use_reranking=USE_RERANKING, use_camera_bias=USE_CAMERA_BIAS)

# Clustering
if CLUSTERING_METHOD == "SIMILARITY":
    print(f"📌 Similarity-based clustering (threshold={SIMILARITY_THRESHOLD})...")
    
    global_ids = [-1] * len(all_tracklets)
    next_global_id = 0
    
    for i in range(len(all_tracklets)):
        if global_ids[i] != -1:
            continue
        
        # Find most similar existing cluster
        best_cluster = -1
        best_dist = float('inf')
        
        for cluster_id in range(next_global_id):
            cluster_members = [j for j in range(i) if global_ids[j] == cluster_id]
            if not cluster_members:
                continue
            
            avg_dist = np.mean([dist_matrix[i, j] for j in cluster_members])
            if avg_dist < best_dist:
                best_dist = avg_dist
                best_cluster = cluster_id
        
        # Assign to cluster or create new
        if best_cluster != -1 and best_dist < SIMILARITY_THRESHOLD:
            global_ids[i] = best_cluster
        else:
            global_ids[i] = next_global_id
            next_global_id += 1
    
    global_ids = np.array(global_ids)

elif CLUSTERING_METHOD == "DBSCAN":
    print(f"📌 DBSCAN clustering (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})...")
    clusterer = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric='precomputed')
    global_ids = clusterer.fit_predict(dist_matrix)
    
    # Remap noise (-1) to unique IDs
    noise_mask = (global_ids == -1)
    if noise_mask.any():
        max_id = global_ids.max()
        noise_ids = np.arange(max_id + 1, max_id + 1 + noise_mask.sum())
        global_ids[noise_mask] = noise_ids

else:
    raise ValueError(f"Unknown clustering method: {CLUSTERING_METHOD}")

n_clusters = len(np.unique(global_ids))
print(f"✅ Found {n_clusters} global identities\n")

# -----------------------
# Export Results
# -----------------------

print("="*60)
print("💾 EXPORTING RESULTS")
print("="*60)

# Assign global IDs to tracklets
for i, global_id in enumerate(global_ids):
    all_tracklets[i]['global_id'] = int(global_id)

# Create CSV
rows = []
for t in all_tracklets:
    rows.append({
        'global_id': t['global_id'],
        'clip': t['clip'],
        'clip_id': t['clip_id'],
        'start_frame': t['start_frame'],
        'end_frame': t['end_frame']
    })

df = pd.DataFrame(rows)
df.to_csv(CSV_OUT, index=False)
print(f"✅ CSV saved: {CSV_OUT}")

# Create JSON
output_data = {
    'summary': {
        'total_global_identities': n_clusters,
        'total_tracklets': len(all_tracklets),
        'clustering_method': CLUSTERING_METHOD,
        'features_used': {
            'face_weight': FACE_WEIGHT,
            'motion_weight': MOTION_WEIGHT,
            'pose_weight': POSE_WEIGHT,
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
            'clip': t['clip'],
            'clip_id': t['clip_id'],
            'start_frame': t['start_frame'],
            'end_frame': t['end_frame']
        } for t in tracks]
    }

with open(JSON_OUT, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"✅ JSON saved: {JSON_OUT}")

print("\n" + "="*60)
print("✅ PROCESSING COMPLETE!")
print("="*60)
print(f"📊 Global Identities: {n_clusters}")
print(f"📁 Outputs: {CSV_OUT}, {JSON_OUT}")
print("="*60 + "\n")