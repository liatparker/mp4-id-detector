#!/usr/bin/env python3
"""
Video ID Detector - Clean & Efficient Version


OVERVIEW:
This system performs person re-identification across multiple video clips using:
- YOLO for person detection
- ByteTrack for person tracking  
- Ensemble ReID: OSNet (70%) + TransReID (30%) for feature extraction
- Two-stage clustering for identity assignment
- Physical proximity logic for ambiguous cases

INPUT: Video files in ./videos/ directory
OUTPUT: Global identity assignments, annotated videos, and analysis files

"""

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
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ======================
# CONFIGURATION
# ======================

# Paths and Files
VIDEO_DIR = "./videos/"
VIDEO_FILES = ["1_upscaled.mp4", "2_upscaled.mp4", "3_upscaled.mp4", "4_upscaled.mp4"]
OUTPUT_DIR = "./outputs_clean/"
CACHE_DIR = "./outputs_v3/"

# Model Configuration
YOLO_WEIGHTS = "yolov8n.pt"
REID_MODEL = "osnet_ibn_x1_0"
FACE_MODEL = "buffalo_l"
DEVICE = "cpu"

# Ensemble Configuration (matching video_id_detector2_optimized.py)
USE_ENSEMBLE_REID = True
ENSEMBLE_WEIGHTS = [0.7, 0.3]  # [OSNet, TransReID] - OSNet gets higher weight

# Clustering Parameters
BASE_THRESHOLD = 0.6
SAME_CLIP_THRESHOLD = 0.4
CROSS_CLIP_THRESHOLD = 0.7
PROXIMITY_PATTERN_TOLERANCE = 0.15

# Feature Weights
FACE_WEIGHT = 0.2
MOTION_WEIGHT = 0.1
POSE_WEIGHT = 0.1
BODY_WEIGHT = 0.6  # Calculated as 1 - (FACE_WEIGHT + MOTION_WEIGHT + POSE_WEIGHT)

# Cache Configuration
USE_EXISTING_CACHE = True
CACHE_VERSION = "v3"

# ======================
# UTILITY FUNCTIONS
# ======================

def create_directories():
    """Create necessary output directories."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/annotated_videos", exist_ok=True)
    print(f"✅ Created output directory: {OUTPUT_DIR}")

def load_models():
    """Load all required models including ensemble ReID."""
    print("🔧 Loading models...")
    
    # YOLO for detection
    yolo_model = YOLO(YOLO_WEIGHTS)
    print(f"✅ Loaded YOLO: {YOLO_WEIGHTS}")
    
    # OSNet for ReID
    osnet_model = torchreid.models.build_model(REID_MODEL, num_classes=1000)
    osnet_model = osnet_model.to(DEVICE)
    osnet_model.eval()
    print(f"✅ Loaded OSNet: {REID_MODEL}")
    
    # TransReID for ensemble (if enabled)
    transreid_model = None
    if USE_ENSEMBLE_REID:
        try:
            # Add TransReID path
            import sys
            sys.path.append('/Users/liatparker/vscode_agentic/mp4_id_detector/TransReID')
            from model.make_model import make_model
            from config.defaults import _C as cfg_default
            
            cfg = cfg_default.clone()
            cfg.merge_from_file('/Users/liatparker/vscode_agentic/mp4_id_detector/TransReID/configs/Market/vit_base.yml')
            cfg.MODEL.DEVICE = DEVICE
            cfg.MODEL.PRETRAIN_PATH = '/Users/liatparker/.cache/torch/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth'
            cfg.freeze()
            
            transreid_model = make_model(cfg, num_class=751, camera_num=0, view_num=0)
            transreid_model.eval()
            transreid_model.to(DEVICE)
            print(f"✅ Loaded TransReID (ensemble weight: {ENSEMBLE_WEIGHTS[1]})")
        except Exception as e:
            print(f"⚠️  TransReID failed to load: {e}")
            print("   Falling back to OSNet only")
            USE_ENSEMBLE_REID = False
    
    # Face recognition
    face_model = insightface.app.FaceAnalysis(name=FACE_MODEL, providers=['CPUExecutionProvider'])
    face_model.prepare(ctx_id=0, det_size=(640, 640))
    print(f"✅ Loaded face model: {FACE_MODEL}")
    
    # Pose estimation
    pose_model = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    print("✅ Loaded pose estimation model")
    
    print(f"📊 Ensemble ReID: {'ENABLED' if USE_ENSEMBLE_REID else 'DISABLED'}")
    if USE_ENSEMBLE_REID:
        print(f"   Weights: OSNet {ENSEMBLE_WEIGHTS[0]:.1%}, TransReID {ENSEMBLE_WEIGHTS[1]:.1%}")
    
    return yolo_model, osnet_model, transreid_model, face_model, pose_model

def extract_ensemble_embedding(crop, osnet_model, transreid_model):
    """Extract ensemble embedding using OSNet + TransReID."""
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
            osnet_emb = osnet_model(img_tensor)
            osnet_emb = osnet_emb.cpu().numpy().flatten()
        
        # Extract TransReID embedding (if available)
        if USE_ENSEMBLE_REID and transreid_model is not None:
            # TransReID preprocessing (same as OSNet)
            with torch.no_grad():
                transreid_emb = transreid_model(img_tensor)
                transreid_emb = transreid_emb.cpu().numpy().flatten()
            
            # Combine with ensemble weights
            combined_emb = np.concatenate([
                osnet_emb * ENSEMBLE_WEIGHTS[0],  # OSNet weight (0.7)
                transreid_emb * ENSEMBLE_WEIGHTS[1]  # TransReID weight (0.3)
            ])
        else:
            # Use OSNet only
            combined_emb = osnet_emb
        
        # Normalize
        combined_emb = combined_emb / (np.linalg.norm(combined_emb) + 1e-8)
        return combined_emb
        
    except Exception as e:
        print(f"Ensemble embedding extraction failed: {e}")
        return np.zeros(512, dtype=np.float32)

def load_existing_cache():
    """Load existing cache files if available, prioritizing mapping NPZ for exact alignment."""
    if not USE_EXISTING_CACHE:
        return None, None, None

    mapping_path = f"{CACHE_DIR}/tracklet_to_global_id.npz"
    catalogue_path = f"{CACHE_DIR}/global_identity_catalogue_{CACHE_VERSION}.json"
    track_emb_path = f"{CACHE_DIR}/track_embeddings_{CACHE_VERSION}.npz"

    # Require mapping and catalogue; track embeddings are optional
    if not os.path.exists(mapping_path) or not os.path.exists(catalogue_path):
        missing = [p for p in [mapping_path, catalogue_path] if not os.path.exists(p)]
        print(f"⚠️  Missing cache files: {missing}")
        return None, None, None

    print("📁 Loading existing cache files (mapping-first)...")

    # Load mapping NPZ which contains tracklets and aligned global_ids
    mapping_npz = np.load(mapping_path, allow_pickle=True)
    # Prefer tracklets from mapping to preserve exact order and content
    if 'tracklets' in mapping_npz:
        tracklets = mapping_npz['tracklets']
        print(f"✅ Loaded {len(tracklets)} tracklets from mapping cache")
    else:
        # Fallback to track embeddings NPZ if mapping doesn't carry tracklets
        if not os.path.exists(track_emb_path):
            print("⚠️  tracklets not found in mapping and track_embeddings npz missing")
            return None, None, None
        emb_npz = np.load(track_emb_path, allow_pickle=True)
        tracklets = emb_npz['tracklets']
        print(f"✅ Loaded {len(tracklets)} tracklets from embeddings cache")

    global_ids = mapping_npz['global_ids']
    print(f"✅ Loaded {len(global_ids)} global ID assignments")

    # Load global identity catalogue (for summary parity)
    with open(catalogue_path, 'r') as f:
        catalogue = json.load(f)
    print(f"✅ Loaded global identity catalogue with {catalogue['summary']['total_global_identities']} identities")

    return tracklets, global_ids, catalogue

def extract_features_from_bbox(image, bbox, reid_model, face_model, pose_model):
    """Extract features from a bounding box region."""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Ensure bbox is within image bounds
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None, None, None, None
    
    # Extract person crop
    person_crop = image[y1:y2, x1:x2]
    if person_crop.size == 0:
        return None, None, None, None
    
    # Resize for models
    person_crop_resized = cv2.resize(person_crop, (128, 256))
    
    # Extract ReID features
    try:
        person_tensor = torch.from_numpy(person_crop_resized).permute(2, 0, 1).float() / 255.0
        person_tensor = person_tensor.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            reid_features = reid_model(person_tensor)
            reid_features = reid_features.cpu().numpy().flatten()
    except:
        reid_features = None
    
    # Extract face features
    try:
        faces = face_model.get(person_crop)
        if faces:
            face_features = faces[0].embedding
        else:
            face_features = None
    except:
        face_features = None
    
    # Extract pose features
    try:
        pose_results = pose_model.process(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
        if pose_results.pose_landmarks:
            pose_features = np.array([[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark]).flatten()
        else:
            pose_features = None
    except:
        pose_features = None
    
    # Calculate motion features (placeholder - would need temporal data)
    motion_features = None
    
    return reid_features, face_features, pose_features, motion_features

def calculate_distance(features1, features2):
    """Calculate weighted distance between two feature sets."""
    if features1 is None or features2 is None:
        return float('inf')
    
    total_distance = 0.0
    total_weight = 0.0
    
    # ReID features (body/clothing)
    if features1[0] is not None and features2[0] is not None:
        body_dist = 1 - np.dot(features1[0], features2[0]) / (np.linalg.norm(features1[0]) * np.linalg.norm(features2[0]))
        total_distance += BODY_WEIGHT * body_dist
        total_weight += BODY_WEIGHT
    
    # Face features
    if features1[1] is not None and features2[1] is not None:
        face_dist = 1 - np.dot(features1[1], features2[1]) / (np.linalg.norm(features1[1]) * np.linalg.norm(features2[1]))
        total_distance += FACE_WEIGHT * face_dist
        total_weight += FACE_WEIGHT
    
    # Pose features
    if features1[2] is not None and features2[2] is not None:
        pose_dist = np.linalg.norm(features1[2] - features2[2])
        total_distance += POSE_WEIGHT * pose_dist
        total_weight += POSE_WEIGHT
    
    # Motion features
    if features1[3] is not None and features2[3] is not None:
        motion_dist = np.linalg.norm(features1[3] - features2[3])
        total_distance += MOTION_WEIGHT * motion_dist
        total_weight += MOTION_WEIGHT
    
    if total_weight == 0:
        return float('inf')
    
    return total_distance / total_weight

def detect_and_track_persons(video_path, yolo_model):
    """Detect and track persons in a video."""
    print(f"🎬 Processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize ByteTracker
    tracker = sv.ByteTrack()
    
    detections = []
    frame_idx = 0
    
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO detection
            results = yolo_model(frame, verbose=False)
            
            # Filter for person detections (class 0)
            person_detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if int(box.cls) == 0 and float(box.conf) > 0.5:  # Person class with confidence > 0.5
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            person_detections.append([x1, y1, x2, y2, float(box.conf)])
            
            if person_detections:
                # Update tracker
                detections_sv = sv.Detections(np.array(person_detections))
                tracked_detections = tracker.update_with_detections(detections_sv)
                
                # Store detections
                for detection in tracked_detections:
                    x1, y1, x2, y2, track_id, conf = detection
                    detections.append({
                        'frame_idx': frame_idx,
                        'track_id': int(track_id),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(conf)
                    })
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    print(f"✅ Detected {len(detections)} person instances")
    return detections, fps, width, height

def extract_tracklet_features(detections, video_path, reid_model, face_model, pose_model):
    """Extract features for each tracklet."""
    print("🔍 Extracting tracklet features...")
    
    # Group detections by track_id
    tracklets = {}
    for det in detections:
        track_id = det['track_id']
        if track_id not in tracklets:
            tracklets[track_id] = []
        tracklets[track_id].append(det)
    
    # Extract features for each tracklet
    cap = cv2.VideoCapture(video_path)
    tracklet_features = {}
    
    for track_id, track_detections in tqdm(tracklets.items(), desc="Extracting features"):
        # Sort by frame
        track_detections.sort(key=lambda x: x['frame_idx'])
        
        # Sample frames for feature extraction (every 10th frame)
        sampled_detections = track_detections[::10]
        
        features_list = []
        frames = []
        bboxes = []
        
        for det in sampled_detections:
            frame_idx = det['frame_idx']
            bbox = det['bbox']
            
            # Read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Extract features
            reid_feat, face_feat, pose_feat, motion_feat = extract_features_from_bbox(
                frame, bbox, reid_model, face_model, pose_model
            )
            
            if reid_feat is not None:
                features_list.append((reid_feat, face_feat, pose_feat, motion_feat))
                frames.append(frame_idx)
                bboxes.append(bbox)
        
        if features_list:
            # Average features across frames
            avg_reid = np.mean([f[0] for f in features_list], axis=0)
            avg_face = np.mean([f[1] for f in features_list if f[1] is not None], axis=0) if any(f[1] is not None for f in features_list) else None
            avg_pose = np.mean([f[2] for f in features_list if f[2] is not None], axis=0) if any(f[2] is not None for f in features_list) else None
            avg_motion = np.mean([f[3] for f in features_list if f[3] is not None], axis=0) if any(f[3] is not None for f in features_list) else None
            
            tracklet_features[track_id] = {
                'features': (avg_reid, avg_face, avg_pose, avg_motion),
                'frames': frames,
                'bboxes': bboxes,
                'detections': track_detections
            }
    
    cap.release()
    print(f"✅ Extracted features for {len(tracklet_features)} tracklets")
    return tracklet_features

def cluster_tracklets(tracklet_features, clip_idx):
    """Cluster tracklets using two-stage approach."""
    print(f"🔗 Clustering tracklets for clip {clip_idx}...")
    
    if len(tracklet_features) < 2:
        # Single tracklet - assign unique ID
        global_id = 0
        for track_id in tracklet_features:
            tracklet_features[track_id]['global_id'] = global_id
        return tracklet_features
    
    # Stage 1: Within-clip clustering (strict)
    track_ids = list(tracklet_features.keys())
    n_tracklets = len(track_ids)
    
    # Calculate distance matrix
    distance_matrix = np.zeros((n_tracklets, n_tracklets))
    for i, track_id1 in enumerate(track_ids):
        for j, track_id2 in enumerate(track_ids):
            if i == j:
                distance_matrix[i, j] = 0.0
            else:
                features1 = tracklet_features[track_id1]['features']
                features2 = tracklet_features[track_id2]['features']
                distance_matrix[i, j] = calculate_distance(features1, features2)
    
    # Apply same-clip threshold
    same_clip_mask = distance_matrix < SAME_CLIP_THRESHOLD
    
    # Create clusters using connected components
    clusters = []
    visited = set()
    
    for i in range(n_tracklets):
        if i in visited:
            continue
        
        cluster = [i]
        visited.add(i)
        
        # Find all connected tracklets
        for j in range(n_tracklets):
            if j not in visited and same_clip_mask[i, j]:
                cluster.append(j)
                visited.add(j)
        
        clusters.append(cluster)
    
    # Assign global IDs
    for cluster_idx, cluster in enumerate(clusters):
        for track_idx in cluster:
            track_id = track_ids[track_idx]
            tracklet_features[track_id]['global_id'] = cluster_idx
    
    print(f"✅ Created {len(clusters)} clusters in clip {clip_idx}")
    return tracklet_features

def merge_cross_clip_clusters(all_tracklet_features):
    """Merge clusters across clips using adaptive thresholds."""
    print("🔗 Merging clusters across clips...")
    
    # Collect all tracklets with their features
    all_tracklets = []
    for clip_idx, tracklet_features in all_tracklet_features.items():
        for track_id, data in tracklet_features.items():
            all_tracklets.append({
                'clip_idx': clip_idx,
                'track_id': track_id,
                'global_id': data['global_id'],
                'features': data['features']
            })
    
    if len(all_tracklets) < 2:
        return all_tracklet_features
    
    # Calculate cross-clip distances
    n_tracklets = len(all_tracklets)
    distance_matrix = np.zeros((n_tracklets, n_tracklets))
    
    for i in range(n_tracklets):
        for j in range(n_tracklets):
            if i == j:
                distance_matrix[i, j] = 0.0
            else:
                features1 = all_tracklets[i]['features']
                features2 = all_tracklets[j]['features']
                distance_matrix[i, j] = calculate_distance(features1, features2)
    
    # Apply cross-clip threshold with adaptive logic
    cross_clip_mask = distance_matrix < CROSS_CLIP_THRESHOLD
    
    # Create global clusters
    global_clusters = []
    visited = set()
    
    for i in range(n_tracklets):
        if i in visited:
            continue
        
        cluster = [i]
        visited.add(i)
        
        # Find all connected tracklets
        for j in range(n_tracklets):
            if j not in visited and cross_clip_mask[i, j]:
                cluster.append(j)
                visited.add(j)
        
        global_clusters.append(cluster)
    
    # Update global IDs
    for cluster_idx, cluster in enumerate(global_clusters):
        for track_idx in cluster:
            tracklet = all_tracklets[track_idx]
            clip_idx = tracklet['clip_idx']
            track_id = tracklet['track_id']
            all_tracklet_features[clip_idx][track_id]['global_id'] = cluster_idx
    
    print(f"✅ Created {len(global_clusters)} global clusters")
    return all_tracklet_features

def export_results(all_tracklet_features, fps, width, height):
    """Export results to CSV, JSON, and NPZ files."""
    print("💾 Exporting results...")
    
    # Prepare data for export
    all_tracklets = []
    tracklet_to_global_id = {}
    
    for clip_idx, tracklet_features in all_tracklet_features.items():
        for track_id, data in tracklet_features.items():
            tracklet_data = {
                'clip_idx': clip_idx,
                'track_id': track_id,
                'global_id': data['global_id'],
                'frames': data['frames'],
                'bboxes': data['bboxes'],
                'detections': data['detections']
            }
            all_tracklets.append(tracklet_data)
            
            # Create mapping
            key = (clip_idx, track_id)
            tracklet_to_global_id[key] = data['global_id']
    
    # Export to CSV
    csv_data = []
    for tracklet in all_tracklets:
        for i, (frame, bbox) in enumerate(zip(tracklet['frames'], tracklet['bboxes'])):
            csv_data.append({
                'clip_idx': tracklet['clip_idx'],
                'track_id': tracklet['track_id'],
                'global_id': tracklet['global_id'],
                'frame_idx': frame,
                'bbox_x1': bbox[0],
                'bbox_y1': bbox[1],
                'bbox_x2': bbox[2],
                'bbox_y2': bbox[3]
            })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(f"{OUTPUT_DIR}/tracklet_assignments.csv", index=False)
    print(f"✅ Exported CSV: {len(csv_data)} records")
    
    # Export to JSON (aligned summary)
    json_data = {
        'summary': {
            'total_clips': len(VIDEO_FILES),
            'total_tracklets': len(all_tracklets),
            'total_global_identities': len(set(t['global_id'] for t in all_tracklets)),
            'config': {
                'ensemble_reid': USE_ENSEMBLE_REID,
                'ensemble_weights': ENSEMBLE_WEIGHTS if USE_ENSEMBLE_REID else None,
                'face_weight': FACE_WEIGHT,
                'body_weight': BODY_WEIGHT,
                'motion_weight': MOTION_WEIGHT,
                'pose_weight': POSE_WEIGHT
            }
        },
        'identities': {}
    }
    
    # Group by global ID
    for tracklet in all_tracklets:
        global_id = int(tracklet['global_id'])  # Convert to Python int
        if global_id not in json_data['identities']:
            json_data['identities'][global_id] = {
                'global_id': global_id,
                'appearances': []
            }
        
        json_data['identities'][global_id]['appearances'].append({
            'clip_idx': int(tracklet['clip_idx']),
            'track_id': int(tracklet['track_id']),
            'frames': [int(f) for f in tracklet['frames']],
            'bboxes': [[float(x) for x in bbox] for bbox in tracklet['bboxes']]
        })
    
    with open(f"{OUTPUT_DIR}/global_identity_catalogue.json", 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✅ Exported JSON: {json_data['summary']['total_global_identities']} identities")
    
    # Export to NPZ
    np.savez(
        f"{OUTPUT_DIR}/track_embeddings.npz",
        tracklets=all_tracklets
    )
    
    np.savez(
        f"{OUTPUT_DIR}/tracklet_to_global_id.npz",
        global_ids=[t['global_id'] for t in all_tracklets]
    )
    
    print(f"✅ Exported NPZ files")
    
    return json_data

def create_annotated_videos(all_tracklet_features, json_data):
    """Create annotated videos showing global IDs."""
    print("🎨 Creating annotated videos...")
    
    # Colors for different global IDs
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
        (0, 191, 255), (50, 205, 50), (255, 20, 147), (0, 206, 209),
        (255, 140, 0), (138, 43, 226), (34, 139, 34)
    ]
    
    def get_color(global_id):
        return colors[global_id % len(colors)]
    
    # Process each video
    for clip_idx, video_file in enumerate(VIDEO_FILES):
        video_path = os.path.join(VIDEO_DIR, video_file)
        output_path = f"{OUTPUT_DIR}/annotated_videos/clip{clip_idx}_annotated.mp4"
        
        print(f"🎬 Annotating clip {clip_idx}: {video_file}")
        
        # Get tracklets for this clip
        clip_tracklets = all_tracklet_features.get(clip_idx, {})
        
        # Build frame-level detections
        frame_detections = {}
        for track_id, data in clip_tracklets.items():
            global_id = data['global_id']
            for frame, bbox in zip(data['frames'], data['bboxes']):
                if frame not in frame_detections:
                    frame_detections[frame] = []
                frame_detections[frame].append((global_id, bbox))
        
        # Get video properties
        cap_temp = cv2.VideoCapture(video_path)
        fps = cap_temp.get(cv2.CAP_PROP_FPS)
        width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_temp.release()
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        
        with tqdm(total=total_frames, desc=f"Annotating clip {clip_idx}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Draw detections for this frame
                if frame_idx in frame_detections:
                    for global_id, bbox in frame_detections[frame_idx]:
                        x1, y1, x2, y2 = map(int, bbox)
                        color = get_color(global_id)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"ID {global_id}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                out.write(frame)
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        out.release()
        print(f"✅ Created annotated video: {output_path}")

def main():
    """Main function."""
    print("=" * 80)
    print("VIDEO ID DETECTOR - CLEAN & EFFICIENT VERSION")
    print("=" * 80)
    print(f"📊 Configuration:")
    print(f"   Ensemble ReID: {'ENABLED' if USE_ENSEMBLE_REID else 'DISABLED'}")
    if USE_ENSEMBLE_REID:
        print(f"   Weights: OSNet {ENSEMBLE_WEIGHTS[0]:.1%}, TransReID {ENSEMBLE_WEIGHTS[1]:.1%}")
    print(f"   Face Weight: {FACE_WEIGHT:.1%}")
    print(f"   Body Weight: {BODY_WEIGHT:.1%}")
    print("=" * 80)
    
    # Create directories
    create_directories()
    
    # Check if we can use existing cache
    tracklets, global_ids, catalogue = load_existing_cache()
    
    if tracklets is not None and global_ids is not None and catalogue is not None:
        print("✅ Using existing cache files - skipping video processing")
        
        # Convert cache to our format
        all_tracklet_features = {}
        for i, tracklet in enumerate(tracklets):
            if i < len(global_ids):
                clip_idx = tracklet['clip_idx']
                track_id = tracklet['track_id']
                global_id = global_ids[i]
                
                if clip_idx not in all_tracklet_features:
                    all_tracklet_features[clip_idx] = {}
                
                all_tracklet_features[clip_idx][track_id] = {
                    'global_id': global_id,
                    'frames': tracklet['frames'],
                    'bboxes': tracklet['bboxes'],
                    'detections': tracklet.get('detections', [])
                }
        
        # Export results
        json_data = export_results(all_tracklet_features, 30, 1920, 1080)
        
        # Create annotated videos
        create_annotated_videos(all_tracklet_features, json_data)
        
    else:
        print("🔄 Processing videos from scratch...")
        
        # Load models
        yolo_model, osnet_model, transreid_model, face_model, pose_model = load_models()
        
        # Process each video
        all_tracklet_features = {}
        
        for clip_idx, video_file in enumerate(VIDEO_FILES):
            video_path = os.path.join(VIDEO_DIR, video_file)
            
            if not os.path.exists(video_path):
                print(f"⚠️  Video not found: {video_path}")
                continue
            
            # Detect and track persons
            detections, fps, width, height = detect_and_track_persons(video_path, yolo_model)
            
            if not detections:
                print(f"⚠️  No detections in {video_file}")
                continue
            
            # Extract features
            tracklet_features = extract_tracklet_features(detections, video_path, reid_model, face_model, pose_model)
            
            # Cluster tracklets
            tracklet_features = cluster_tracklets(tracklet_features, clip_idx)
            
            all_tracklet_features[clip_idx] = tracklet_features
        
        # Merge across clips
        all_tracklet_features = merge_cross_clip_clusters(all_tracklet_features)
        
        # Export results
        json_data = export_results(all_tracklet_features, fps, width, height)
        
        # Create annotated videos
        create_annotated_videos(all_tracklet_features, json_data)
    
    print("\n" + "=" * 80)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print(f"📊 Total global identities: {json_data['summary']['total_global_identities']}")
    print(f"📹 Annotated videos: {OUTPUT_DIR}/annotated_videos/")
    print("=" * 80)

if __name__ == "__main__":
    main()
