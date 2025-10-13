import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import torchreid

from torchreid.utils import FeatureExtractor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
import json
dir = "/Users/liatparker/Documents/mp4_files_id_detectors/"
VIDEO_FILES = [dir+"1.mp4", dir+"2.mp4",dir+"3.mp4",dir+"4.mp4"]  
YOLO_WEIGHTS = "yolov8n.pt"                  # small model; swap for yolov8m.pt/yolov8l.pt if you want
REID_MODEL_NAME = "osnet_x1_0"               # torchreid model name
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_TRACK_LENGTH_FOR_REID = 3                # ignore tracks with fewer detections than this
DBSCAN_EPS = 0.12                             # clustering threshold (cosine distance). Tune for your data
DBSCAN_MIN_SAMPLES = 2
OUTPUT_CSV = "global_id_catalogue.csv"
OUTPUT_JSON = "global_id_catalogue.json"


print("Device:", DEVICE)

# ---------- Load YOLO (for detection) ----------
print("Loading YOLO model...")
yolo = YOLO(YOLO_WEIGHTS)


# ---------- Load DeepSORT ----------
print("Loading DeepSORT tracker (deep_sort_realtime)...")
tracker = DeepSort(max_age=100)  # tune max_age as needed
# ---------- Load ReID model (torchreid) ----------
print("Loading ReID model (torchreid):", REID_MODEL_NAME)
# Build a pre-trained feature extractor model (we will use features, not classifier)
reid_model = torchreid.models.build_model(REID_MODEL_NAME, num_classes=10, pretrained=True)
reid_model.eval()
reid_model.to(DEVICE)

# Preprocessing helper for torchreid style input
def preprocess_crop_for_reid(img_bgr):
    # input: cropped BGR image (H,W,3) from cv2
    # output: tensor [1,3,H,W] float on DEVICE
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # resize to model expected - torchreid often uses 256x128 (h x w)
    # Many ReID models expect (C,H,W) with H >= W. OSNet common input: 256x128
    h, w = 256, 128
    img = cv2.resize(img_rgb, (w, h))
    img = img.astype("float32") / 255.0
    # normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0).to(DEVICE)
    return tensor

@torch.no_grad()
def get_embedding_from_crop(crop_bgr):
    t = preprocess_crop_for_reid(crop_bgr)
    feat = reid_model(t)
    feat_np = feat.cpu().numpy().reshape(-1)
    norm = np.linalg.norm(feat_np)
    if norm > 0:
        feat_np = feat_np / norm
    return feat_np

# ---------- Main processing ----------
all_track_embeddings = []

for video_path in VIDEO_FILES:
    if not os.path.exists(video_path):
        print(f"Warning: {video_path} not found — skipping.")
        continue

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pbar = tqdm(total=frame_count, desc=os.path.basename(video_path), unit="fr")

    video_tracks_embs = {}
    video_tracks_counts = {}
    video_tracks_frames = {}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # YOLO detection
        results = yolo.predict(frame, imgsz=640, conf=0.35, classes=[0], verbose=False)
        res = results[0]
        detections_for_tracker = []
        if hasattr(res, "boxes") and len(res.boxes) > 0:
            for box, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = map(float, box)
                if x2 - x1 < 5 or y2 - y1 < 5:
                    continue
                detections_for_tracker.append(([x1, y1, x2, y2], float(conf), 0))

        # DeepSORT update
        tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

        for t in tracks:
            if not t.is_confirmed():
                continue
            local_id = t.track_id
            l, t_, r, b = t.to_ltrb()
            l_i, t_i, r_i, b_i = map(int, [l, t_, r, b])
            if r_i - l_i < 5 or b_i - t_i < 5:
                continue
            crop = frame[t_i:b_i, l_i:r_i]
            try:
                emb = get_embedding_from_crop(crop)
            except Exception:
                continue

            if local_id not in video_tracks_embs:
                video_tracks_embs[local_id] = []
                video_tracks_counts[local_id] = 0
                video_tracks_frames[local_id] = [frame_idx, frame_idx]
            else:
                video_tracks_frames[local_id][1] = frame_idx

            video_tracks_embs[local_id].append(emb)
            video_tracks_counts[local_id] += 1

        pbar.update(1)

    pbar.close()
    cap.release()

    # Aggregate embeddings per local track
    for local_id, emb_list in video_tracks_embs.items():
        count = video_tracks_counts[local_id]
        if count < MIN_TRACK_LENGTH_FOR_REID:
            continue
        avg_emb = np.mean(np.stack(emb_list, axis=0), axis=0)
        nrm = np.linalg.norm(avg_emb)
        if nrm > 0:
            avg_emb = avg_emb / nrm
        start_f, end_f = video_tracks_frames[local_id]
        all_track_embeddings.append({
            "video": os.path.basename(video_path),
            "local_track_id": int(local_id),
            "embedding": avg_emb.astype(np.float32),
            "start_frame": int(start_f),
            "end_frame": int(end_f),
            "fps": fps
        })

print(f"Collected {len(all_track_embeddings)} track-level embeddings.")

if not all_track_embeddings:
    raise SystemExit("No embeddings collected.")

# ---------- Clustering ----------
emb_matrix = np.stack([x["embedding"] for x in all_track_embeddings], axis=0)
emb_matrix = normalize(emb_matrix, norm="l2")

clust = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="cosine").fit(emb_matrix)
labels = clust.labels_

# Assign global IDs
label_to_global = {}
next_id = 0
for lbl in sorted(set(labels)):
    if lbl == -1:
        continue
    label_to_global[lbl] = next_id
    next_id += 1
for i, lbl in enumerate(labels):
    if lbl == -1:
        label_to_global[("noise", i)] = next_id
        next_id += 1

# ---------- Build hierarchical artifact ----------
artifact = {}

for idx, track in enumerate(all_track_embeddings):
    lbl = labels[idx]
    if lbl != -1:
        gid = label_to_global[lbl]
    else:
        gid = label_to_global[("noise", idx)]
    gid_key = f"global_id_{gid}"

    if gid_key not in artifact:
        artifact[gid_key] = {"appearances": []}

    start_frame = track["start_frame"]
    end_frame = track["end_frame"]
    fps = track["fps"]

    artifact[gid_key]["appearances"].append({
        "clip_id": track["video"],
        "frame_range": [start_frame, end_frame],
        "time_span": [start_frame / fps, end_frame / fps],
        "instance_ref": f"{track['video']}_track_{track['local_track_id']}"
    })

# Save JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(artifact, f, indent=2)

print(f"✅ Saved roster to {OUTPUT_JSON}")