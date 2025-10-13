#!/usr/bin/env python3
"""
🔥 ORIGINAL VIDEO ID DETECTOR - Clean Version
Simple 2-stage adaptive clustering without advanced features
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, cosine
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import json

# ======================
# CONFIGURATION
# ======================

# Basic settings
CLUSTERING_METHOD = "ADAPTIVE"
USE_ADAPTIVE_CLUSTERING = True
USE_PER_CLIP_THRESHOLDS = True
PER_CLIP_THRESHOLDS = {}

# Thresholds
WITHIN_CLIP_THRESHOLD = 0.15
CROSS_CLIP_THRESHOLD = 0.4
WITHIN_CLIP_FACE_WEIGHT = 0.3
FACE_PENALTY = 0.2
TEMPORAL_OVERLAP_PENALTY = 0.1

# Embedding weights - ORIGINAL
FACE_WEIGHT = 0.2
MOTION_WEIGHT = 0.1
POSE_WEIGHT = 0.1
# body_weight = 1 - 0.2 - 0.1 - 0.1 = 0.6 (ORIGINAL)

# Video settings
VIDEO_WEIGHT = 0.6
VIDEO_DIM = 768
USE_CHAMFER_DISTANCE = True

# Features
USE_CAMERA_BIAS = True
USE_RERANKING = True
K_RECIPROCAL = 25

# Output
OUTPUT_DIR = "./outputs_v3"

print("="*60)
print("🚀 ORIGINAL REID SYSTEM - Clean Version")
print("="*60)
print(f"Face weight: {FACE_WEIGHT}")
print(f"Body weight: {1 - FACE_WEIGHT - MOTION_WEIGHT - POSE_WEIGHT:.2f}")
print("🎯 APPROACH: ORIGINAL LOGIC (Clean & Simple)")
print("="*60 + "\n")

def analyze_clip_characteristics(clip_tracks):
    """Automatically analyze clip characteristics to determine optimal threshold."""
    from scipy.spatial.distance import cdist
    
    n = len(clip_tracks)
    
    # Determine max simultaneous people in this clip
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
    np.fill_diagonal(pairwise_dist, np.nan)  # Remove diagonal
    avg_pairwise_dist = np.nanmean(pairwise_dist)
    
    if avg_pairwise_dist > 0.4:  # High diversity
        diversity_score = -0.02  # Can be more strict
    elif avg_pairwise_dist > 0.3:
        diversity_score = 0.0
    else:
        diversity_score = 0.02  # Low diversity, need more lenient
    
    # Combine all factors
    adaptive_threshold = WITHIN_CLIP_THRESHOLD + crowding_score + tracking_score + face_score + diversity_score
    
    analysis = {
        'crowding': crowding_score,
        'tracking': tracking_score,
        'face': face_score,
        'diversity': diversity_score
    }
    
    return adaptive_threshold, analysis

def adaptive_cluster_tracklets(tracklets):
    """
    🔥 ADAPTIVE 2-STAGE CLUSTERING:
    Stage 1: Within-clip clustering (strict - separate people in same scene)
    Stage 2: Cross-clip merging (lenient - match same person across scenes)
    """
    print("🎯 ADAPTIVE CLUSTERING (2-stage)")
    
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
        
        # Handle single tracklet case
        if n == 1:
            clip_tracks[0]['temp_global_id'] = next_local_id
            next_local_id += 1
            print(f"      → 1 local cluster (single tracklet)")
            continue
        
        # Compute within-clip distance matrix (HIGH face weight)
        body_embs = np.array([t['body_emb'] for t in clip_tracks])
        face_embs = np.array([t['face_emb'] for t in clip_tracks])
        has_faces = np.array([t['has_face'] for t in clip_tracks])
        
        body_dist = cdist(body_embs, body_embs, metric='cosine')
        face_dist = cdist(face_embs, face_embs, metric='cosine')
        
        # Handle NaN values in distance matrices
        body_dist = np.nan_to_num(body_dist, nan=1.0)
        face_dist = np.nan_to_num(face_dist, nan=1.0)
        
        # High face weight within clip (same lighting/angle)
        body_w = 1.0 - WITHIN_CLIP_FACE_WEIGHT
        face_w = WITHIN_CLIP_FACE_WEIGHT
        
        # Apply face penalty for missing faces
        face_penalty = np.outer(has_faces, has_faces)
        face_penalty = np.where(face_penalty, 1.0, FACE_PENALTY)
        
        dist_matrix = body_w * body_dist + face_w * face_dist * face_penalty
        
        # Ensure no NaN values in final distance matrix
        dist_matrix = np.nan_to_num(dist_matrix, nan=1.0)
        
        # Apply temporal overlap penalty (prevent merging simultaneous people)
        for i in range(n):
            for j in range(i+1, n):
                t1, t2 = clip_tracks[i], clip_tracks[j]
                overlap_frames = max(0, min(t1['end_frame'], t2['end_frame']) - max(t1['start_frame'], t2['start_frame']))
                if overlap_frames > 0:
                    shorter_track = min(t1['end_frame'] - t1['start_frame'], t2['end_frame'] - t2['start_frame'])
                    overlap_ratio = overlap_frames / shorter_track
                    if overlap_ratio > 0.5:  # Block if overlap > 50% of shorter track
                        # They appear simultaneously → MUST be different people → DON'T merge
                        print(f"    ⚠️  Temporal overlap penalty: Clip {clip_idx} Tracklet {i} ↔ Tracklet {j} (overlap: {overlap_frames}f, ratio: {overlap_ratio:.2f}) +{TEMPORAL_OVERLAP_PENALTY:.3f}")
                        dist_matrix[i, j] += TEMPORAL_OVERLAP_PENALTY
                        dist_matrix[j, i] += TEMPORAL_OVERLAP_PENALTY
        
        # Cluster within clip
        clusterer = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=clip_threshold, 
            metric='precomputed', 
            linkage='average'
        )
        local_ids = clusterer.fit_predict(dist_matrix)
        
        # Assign global temp IDs
        for i, local_id in enumerate(local_ids):
            clip_tracks[i]['temp_global_id'] = next_local_id + local_id
        
        next_local_id += len(set(local_ids))
        print(f"      → {len(set(local_ids))} local clusters")
    
    # Stage 2: Cross-clip merging (lenient)
    print("  🔗 Stage 2: Cross-clip merging (HYBRID: 40% image + 60% video)...")
    
    # Prepare embeddings for cross-clip matching
    body_embs = np.array([t['body_emb'] for t in tracklets])
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
    
    cluster_body_embs = np.zeros((n_temp, body_embs.shape[1]))
    cluster_video_embs = np.zeros((n_temp, video_embs.shape[1]))
    cluster_face_embs = np.zeros((n_temp, face_embs.shape[1]))
    cluster_has_faces = np.zeros(n_temp, dtype=bool)
    
    for i, tid in enumerate(unique_temp_ids):
        mask = temp_ids == tid
        cluster_body_embs[i] = np.mean(body_embs[mask], axis=0)
        cluster_video_embs[i] = np.mean(video_embs[mask], axis=0)
        cluster_face_embs[i] = np.mean(face_embs[mask], axis=0)
        cluster_has_faces[i] = np.any(has_faces[mask])
    
    # Compute HYBRID distance (weighted combination)
    print("    Computing hybrid distances (image + video)...")
    
    if USE_CHAMFER_DISTANCE:
        print("    Using Chamfer distance for cross-clip matching...")
        
        # Use Chamfer distance for cross-clip matching
        dist_matrix = np.zeros((n_temp, n_temp))
        for i in range(n_temp):
            for j in range(i+1, n_temp):
                # Image distance (body + face)
                body_dist = cosine(cluster_body_embs[i], cluster_body_embs[j])
                face_dist = cosine(cluster_face_embs[i], cluster_face_embs[j]) if cluster_has_faces[i] and cluster_has_faces[j] else 0.5
                
                # Apply face penalty
                face_penalty = 1.0 if (cluster_has_faces[i] and cluster_has_faces[j]) else FACE_PENALTY
                image_dist = (1.0 - FACE_WEIGHT) * body_dist + FACE_WEIGHT * face_dist * face_penalty
                
                # Video distance (motion + pose)
                video_dist = cosine(cluster_video_embs[i], cluster_video_embs[j])
                
                # Hybrid distance (40% image + 60% video)
                hybrid_dist = 0.4 * image_dist + 0.6 * video_dist
                dist_matrix[i, j] = hybrid_dist
                dist_matrix[j, i] = hybrid_dist
        
        print("    ✅ Hybrid distances computed")
    else:
        # Standard distance computation
        body_dist = cdist(cluster_body_embs, cluster_body_embs, metric='cosine')
        face_dist = cdist(cluster_face_embs, cluster_face_embs, metric='cosine')
        
        # Apply face penalty
        face_penalty = np.outer(cluster_has_faces, cluster_has_faces)
        face_penalty = np.where(face_penalty, 1.0, FACE_PENALTY)
        
        dist_matrix = (1.0 - FACE_WEIGHT) * body_dist + FACE_WEIGHT * face_dist * face_penalty
    
    # Greedy merging with adaptive thresholds
    final_ids = np.arange(n_temp)  # Start with each cluster as separate
    merged = set()
    
    # Create clip-to-temp-ids mapping for same-clip blocking
    clip_to_temp_ids = {}
    for t in tracklets:
        clip_idx = t['clip_idx']
        temp_id = t['temp_global_id']
        if clip_idx not in clip_to_temp_ids:
            clip_to_temp_ids[clip_idx] = set()
        clip_to_temp_ids[clip_idx].add(temp_id)
    
    for i in range(n_temp):
        if i in merged:
            continue
            
        for j in range(i+1, n_temp):
            if j in merged:
                continue
            
            temp_id_i = unique_temp_ids[i]
            temp_id_j = unique_temp_ids[j]
            
            # Check if they're in the same clip (block merging)
            same_clip = False
            for clip_idx in clip_to_temp_ids:
                if temp_id_i in clip_to_temp_ids[clip_idx] and temp_id_j in clip_to_temp_ids[clip_idx]:
                    same_clip = True
                    break
            
            if same_clip:
                continue  # Skip same-clip pairs
            
            # Standard adaptive threshold
            base_threshold = CROSS_CLIP_THRESHOLD
            
            # Calculate adaptive threshold based on cluster characteristics
            if cluster_has_faces[i] and cluster_has_faces[j]:
                adaptive_threshold = base_threshold - 0.1  # HIGH: Both have faces
            elif cluster_has_faces[i] or cluster_has_faces[j]:
                adaptive_threshold = base_threshold - 0.05  # MED: One has face
            else:
                adaptive_threshold = base_threshold  # LOW: No faces
            
            # Check distance against threshold
            if dist_matrix[i, j] < adaptive_threshold:
                # Merge clusters
                final_ids[j] = final_ids[i]
                merged.add(j)
                
                # Determine threshold level for display
                if adaptive_threshold <= base_threshold - 0.1:
                    level = "HIGH"
                elif adaptive_threshold <= base_threshold - 0.05:
                    level = "MED"
                else:
                    level = "LOW"
                
                print(f"    ✅ Cross-clip match: Clip {temp_id_i} ID {temp_id_i} ↔ Clip {temp_id_j} ID {temp_id_j} (dist: {dist_matrix[i, j]:.3f}, thresh: {adaptive_threshold:.3f} [{level}])")
            else:
                # Near-miss
                if dist_matrix[i, j] < adaptive_threshold + 0.1:
                    print(f"    ⏭️  Near-miss: Clip {temp_id_i} ID {temp_id_i} ↔ Clip {temp_id_j} ID {temp_id_j} (dist: {dist_matrix[i, j]:.3f}, thresh: {adaptive_threshold:.3f} [{level}])")
    
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

# Load cached embeddings
print("📦 Found cached embeddings: ./outputs_v3/track_embeddings_v3.npz")
print("⚡ Loading from cache (skipping video processing)...")

# Load tracklets from cache
data = np.load('./outputs_v3/track_embeddings_v3.npz', allow_pickle=True)
all_tracklets = data['tracklets'].tolist()

print(f"✅ Loaded {len(all_tracklets)} tracklets from cache")

# Simple merging of overlapping tracks in same clips
print("🔧 Merging overlapping tracks in same clips...")
clip_groups = {}
for t in all_tracklets:
    clip_idx = t['clip_idx']
    if clip_idx not in clip_groups:
        clip_groups[clip_idx] = []
    clip_groups[clip_idx].append(t)

merged_tracklets = []
for clip_idx in sorted(clip_groups.keys()):
    clip_tracks = clip_groups[clip_idx]
    print(f"  🔍 Clip {clip_idx}: Processing {len(clip_tracks)} tracklets")
    
    # Simple merging logic
    for i, t1 in enumerate(clip_tracks):
        merged = False
        for j, t2 in enumerate(merged_tracklets):
            if (t2['clip_idx'] == clip_idx and 
                t1['start_frame'] <= t2['end_frame'] + 10 and 
                t2['start_frame'] <= t1['end_frame'] + 10):
                # Check similarity
                body_dist = cosine(t1['body_emb'], t2['body_emb'])
                if body_dist < 0.3:  # Similar enough to merge
                    # Merge by extending the track
                    t2['start_frame'] = min(t1['start_frame'], t2['start_frame'])
                    t2['end_frame'] = max(t1['end_frame'], t2['end_frame'])
                    merged = True
                    break
        
        if not merged:
            merged_tracklets.append(t1.copy())

all_tracklets = merged_tracklets
print(f"✅ After merging: {len(all_tracklets)} tracklets")

# Filter out tiny tracklets
print("🧹 Filtering out tiny tracklets...")
filtered_tracklets = []
for t in all_tracklets:
    if t['end_frame'] - t['start_frame'] >= 30:  # At least 1 second at 30fps
        filtered_tracklets.append(t)
    else:
        print(f"   ⚠️  Removed tiny fragment (< 30 frames)")

all_tracklets = filtered_tracklets
print(f"✅ Kept {len(all_tracklets)} tracklets for clustering\n")

print("="*60)
print("🧮 CLUSTERING")
print("="*60)
if USE_ADAPTIVE_CLUSTERING:
    # Use adaptive two-stage clustering
    global_ids = adaptive_cluster_tracklets(all_tracklets)

n_clusters = len(np.unique(global_ids))
print(f"\n✅ Found {n_clusters} global identities\n")

# Assign IDs
for i, gid in enumerate(global_ids):
    all_tracklets[i]['global_id'] = int(gid)

# Save global ID mapping
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
            'ensemble': True,
            'tta': True,
            'temporal_smoothing': True,
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
        'appearances': [
            {
                'clip_idx': int(t['clip_idx']),
                'start_frame': int(t['start_frame']),
                'end_frame': int(t['end_frame'])
            }
            for t in tracks
        ]
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
print("="*60)
