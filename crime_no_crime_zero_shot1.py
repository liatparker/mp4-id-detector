#!/usr/bin/env python3
"""
🚨 Zero-Shot Crime Detection (Hybrid: YOLO + CLIP) - WITH CLIP GUN VALIDATION
NO training data required!

Detects:
- Shoplifting (behavior-based via CLIP)
- Weapons (guns via YOLO + CLIP Validation)
- Suspicious activity

  CLIP-Based Gun Validation
- Validates each YOLO gun detection using CLIP
- Asks: "Is this a gun or a smartphone?"
- Filters false positives (iPhones, hands, etc.)
- Combined YOLO + CLIP scoring for robust detection


Version: 1.0 (With CLIP Validation)
"""

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import argparse
from pathlib import Path
import json

class HybridCrimeDetector:
    """
    Hybrid detector combining:
    - YOLO: For weapon/object detection
    - CLIP: For scene/behavior understanding + gun validation
    """
    
    def __init__(self, device='cpu', use_weapon_detector=True):
        print("🚀 Initializing Hybrid Crime Detector (v1 - CLIP Validation)...")
        print("  - YOLO: Object detection (weapons, people)")
        print("  - CLIP: Scene understanding (behaviors)")
        print("  - 🎯 CLIP Validation: Gun vs Phone discrimination")
        
        self.device = device
        self.use_weapon_detector = use_weapon_detector
        
        # 1. Load YOLOv8 for people detection (UPGRADED!)
        print("\n📦 Loading YOLOv8 for people detection...")
        self.yolo = YOLO('yolov8n.pt')  # Better small object detection!
        
        # ⚠️ COCO does NOT include guns!
        # Standard COCO has: knife (class 43), but NO guns/rifles/pistols
        # For gun detection, we'll use CLIP + optional custom gun model
        
        # 2. Load weapon-specific YOLO (if available)
        if use_weapon_detector:
            try:
                print("📦 Loading Weapon Detection Model...")
                # Try to load weapon-specific model
                # IMPORTANT: COCO does NOT include weapons!
                # Options:
                # 1. Roboflow weapon model (download from roboflow.com)
                # 2. Custom trained YOLOv8: 'weapon_yolov8.pt'
                # 3. Ultralytics Hub pre-trained models
                
                # Try to load custom weapon model
                import os
                weapon_model_path = 'weapon_yolov8_gun.pt'  # YOLOv8 trained on 6k gun images!
                
                if os.path.exists(weapon_model_path):
                    # Try loading as YOLOv8 first, then fall back to YOLOv5
                    try:
                        # Try YOLOv8 format (Ultralytics)
                        self.weapon_yolo = YOLO(weapon_model_path)
                        print(f"✅ Loaded gun detection model (YOLOv8): {weapon_model_path}")
                        self.weapon_classes = self.weapon_yolo.names
                        print(f"   Gun classes: {list(self.weapon_classes.values())}")
                    except Exception as e1:
                        # Fall back to YOLOv5 (torch.hub)
                        print(f"   YOLOv8 failed, trying YOLOv5 format...")
                        try:
                            self.weapon_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=weapon_model_path, force_reload=False)
                            print(f"✅ Loaded gun detection model (YOLOv5): {weapon_model_path}")
                            self.weapon_classes = {i: name for i, name in enumerate(self.weapon_yolo.names)}
                            print(f"   Gun classes: {list(self.weapon_classes.values())}")
                        except Exception as e2:
                            print(f"   ❌ Failed to load gun model: {e2}")
                            self.weapon_yolo = None
                            self.weapon_classes = {}
                else:
                    # Try to auto-download gun model
                    print("⚠️  No weapon model found. Attempting auto-download...")
                    try:
                        from roboflow import Roboflow
                        print("   📥 Downloading gun detection model from Roboflow...")
                        
                        # Use public Roboflow workspace
                        # Note: Works without API key for public models!
                        rf = Roboflow(api_key="")  # Empty key works for public models
                        project = rf.workspace("roboflow-gw7yv").project("gun-detection-y6lgv")
                        version = project.version(1)
                        
                        # Download as YOLOv8 format
                        dataset = version.download("yolov8")
                        print(f"   ✅ Dataset downloaded to: {dataset.location}")
                        print("   ⚠️  Model weights not included. Training required.")
                        print("   💡 Using CLIP-only for now...")
                        self.weapon_yolo = None
                        self.weapon_classes = {}
                        
                    except ImportError:
                        print("   ⚠️  'roboflow' not installed. Install with: pip install roboflow")
                        print("   📥 Or download manually: https://universe.roboflow.com/roboflow-gw7yv/gun-detection-y6lgv")
                        print("   💡 Using CLIP-only for gun detection")
                        self.weapon_yolo = None
                        self.weapon_classes = {}
                    except Exception as e:
                        print(f"   ⚠️  Auto-download failed: {e}")
                        print("   💡 Using CLIP-only for gun detection")
                        self.weapon_yolo = None
                        self.weapon_classes = {}
            except Exception as e:
                print(f"⚠️  Weapon detector not available: {e}")
                self.weapon_yolo = None
                self.weapon_classes = {}
        else:
            self.weapon_yolo = None
            self.weapon_classes = {}
        
        # 2. Load CLIP for zero-shot scene classification
        print("📦 Loading CLIP...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Move to device
        self.clip_model.to(device)
        
        # 3. Define crime scenarios (NO TRAINING NEEDED!) - GUN FOCUSED + BEHAVIOR!
        self.crime_labels = [
            # Shoplifting (VERY SPECIFIC to avoid false positives!)
            "person secretly stealing items and hiding them in pockets or bags",
            "shoplifter concealing stolen merchandise under clothing",
            "person nervously looking around while taking unpaid items",
            
            # GUNS (multiple variations for better detection!)
            "person holding a handgun or pistol",
            "person with a gun or firearm",
            "person pointing a gun",
            "armed person with a weapon",
            "gun visible in person's hand",
            "person carrying a rifle or shotgun",
            "firearm in the scene",
            
            # OWNER REACTION BEHAVIOR (Key signal!)
            "person rushing or running suddenly in panic",
            "shop owner hurrying and moving quickly in alarm",
            "person reacting urgently to danger or threat",
            "anxious person moving rapidly in response to emergency",
            
            # Normal scenarios
            "normal shopping in a store with customers browsing",
            "people casually walking and talking",
            "peaceful indoor scene with people",
            "calm relaxed people in a store"
        ]
        
        # Crime category mapping
        self.label_to_category = {
            0: "SHOPLIFTING", 1: "SHOPLIFTING", 2: "SHOPLIFTING",
            3: "GUN", 4: "GUN", 5: "GUN", 6: "GUN", 7: "GUN", 8: "GUN", 9: "GUN",  
            10: "URGENT_REACTION", 11: "URGENT_REACTION", 12: "URGENT_REACTION", 13: "URGENT_REACTION",
            14: "NORMAL", 15: "NORMAL", 16: "NORMAL", 17: "NORMAL"
        }
        
        print("✅ Detector ready!")
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame to improve gun and behavior detection.
        Same preprocessing as video_id_detector1.py:
        1. CLAHE - Normalize lighting across different clips
        2. Mild sharpening - Enhance edges and small objects (like guns)
        """
        if frame is None or frame.size == 0:
            return frame
        
        try:
            # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # Critical for detecting guns in varying lighting conditions!
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l,a,b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # 2. Mild sharpening - helps YOLO/CLIP detect small guns and details
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            return enhanced
        except Exception as e:
            # If preprocessing fails, return original
            return frame
    
    def extract_frames(self, video_path, sample_rate=30):
        """
        Extract frames from video for analysis.
        sample_rate: Analyze every Nth frame (default: 30 = 1 FPS for 30fps video)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video: {Path(video_path).name}")
        print(f"   Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"   FPS: {fps:.1f}, Total frames: {total_frames}")
        print(f"   Sampling every {sample_rate} frames...")
        print(f"   🎨 Preprocessing: CLAHE + Sharpening (improves gun detection!)")
        
        frames = []
        frame_indices = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_count % sample_rate == 0:
                # 🎨 Preprocess frame before analysis
                frame_preprocessed = self.preprocess_frame(frame)
                frames.append(frame_preprocessed)
                frame_indices.append(frame_count)
            
            frame_count += 1
        
        cap.release()
        print(f"   Extracted {len(frames)} preprocessed frames for analysis")
        
        return frames, frame_indices, fps
    
    def validate_gun_with_clip(self, frame, bbox, gun_confidence):
        """
        🎯 STRICT CLIP-based Gun Validation
        
        Validates whether a YOLO-detected "gun" is actually a gun using CLIP.
        Asks CLIP: "Is this a gun or a phone?"
        
        STRICT THRESHOLDS to filter iPhone false positives!
        
        Returns: (is_valid, validation_score)
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Ensure bbox is within frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return False, 0.0
        
        # Crop the detected object
        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        
        # Ask CLIP: "Gun or Phone?" - MORE DISCRIMINATIVE LABELS!
        validation_labels = [
            # Gun labels (indices 0-3)
            "a real handgun or pistol held by a person",
            "an actual firearm weapon",
            "a gun being pointed",
            "a person threatening with a gun",
            # Phone labels (indices 4-9) - MORE SPECIFIC!
            "a smartphone or iPhone being held",
            "a person looking at their phone screen",
            "a cell phone in someone's hand",
            "a black rectangular phone",
            "a person browsing on their mobile phone",
            "someone holding a small handheld device"
        ]
        
        inputs = self.clip_processor(
            text=validation_labels,
            images=crop_pil,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0].cpu().numpy()
        
        gun_prob = np.max(probs[0:4])    # Gun labels (4 labels now)
        phone_prob = np.max(probs[4:10])  # Phone labels (6 labels now)
        
        # STRICT VALIDATION LOGIC - Much harder to pass!
        # Require STRONG gun evidence and CLEAR separation from phone
        if gun_prob > phone_prob + 0.25:
            # CLIP VERY strongly believes it's a gun (25% margin!)
            validation_score = 0.9
            is_valid = True
        elif gun_prob > phone_prob + 0.15 and gun_confidence > 0.50:
            # CLIP strongly believes + decent YOLO confidence
            validation_score = 0.7
            is_valid = True
        elif gun_prob > phone_prob + 0.05 and gun_confidence > 0.65:
            # Marginal CLIP preference but HIGH YOLO confidence
            validation_score = 0.6
            is_valid = True
        else:
            # Not enough evidence - reject it!
            validation_score = 0.2
            is_valid = False
        
        return is_valid, validation_score
    
    def detect_objects_yolo(self, frame):
        """
        Detect objects using YOLOv8.
        WITH CLIP VALIDATION! 🎯 (NEW!)
        Returns: guns detected (validated), people count
        """
        # 1. Detect people with YOLOv8 COCO
        results = self.yolo(frame, verbose=False, classes=[0])  # 0=person only
        
        guns = []
        people_count = 0
        
        # Count people (class 0 in COCO)
        if len(results) > 0 and results[0].boxes is not None:
            people_count = len(results[0].boxes)
        
        # 2. Detect guns with specialized gun model (if available)
        # ⚠️ COCO does NOT have guns! Need custom gun detection model.
        if self.weapon_yolo is not None and len(self.weapon_classes) > 0:
            # Check if it's YOLOv8 or YOLOv5
            is_yolov8 = isinstance(self.weapon_yolo, YOLO)
            
            if is_yolov8:
                # YOLOv8 detection
                weapon_results = self.weapon_yolo(frame, verbose=False)
                
                if len(weapon_results) > 0 and weapon_results[0].boxes is not None:
                    for box in weapon_results[0].boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        bbox = box.xyxy[0].cpu().numpy().tolist()
                        
                        # Check if it's a gun class (lowered threshold for validation)
                        if cls_id in self.weapon_classes and conf > 0.35:  # Lower threshold
                            gun_type = self.weapon_classes[cls_id]
                            
                            # 🎯 VALIDATE THE GUN DETECTION WITH CLIP
                            is_valid, validation_score = self.validate_gun_with_clip(
                                frame, bbox, conf
                            )
                            
                            if is_valid:
                                guns.append({
                                    'type': gun_type,
                                    'confidence': conf,
                                    'validation_score': validation_score,
                                    'combined_score': (conf + validation_score) / 2,
                                    'bbox': bbox,
                                    'source': 'YOLOv8-CLIP-Validated'
                                })
                                print(f"    ✅ Validated: {gun_type} (YOLO: {conf:.2f}, CLIP: {validation_score:.2f})")
                            else:
                                print(f"    ⚠️  Rejected: {gun_type} (YOLO: {conf:.2f}, CLIP: {validation_score:.2f}) - Likely phone/false positive")
            
            else:
                # YOLOv5 detection (pandas DataFrame format)
                weapon_results = self.weapon_yolo(frame, size=640)
                weapon_df = weapon_results.pandas().xyxy[0]
                
                for _, row in weapon_df.iterrows():
                    cls_id = int(row['class'])
                    conf = float(row['confidence'])
                    bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
                    
                    # Check if it's a gun class
                    if cls_id in self.weapon_classes and conf > 0.35:
                        gun_type = self.weapon_classes[cls_id]
                        
                        # 🎯 VALIDATE THE GUN DETECTION WITH CLIP
                        is_valid, validation_score = self.validate_gun_with_clip(
                            frame, bbox, conf
                        )
                        
                        if is_valid:
                            guns.append({
                                'type': gun_type,
                                'confidence': conf,
                                'validation_score': validation_score,
                                'combined_score': (conf + validation_score) / 2,
                                'bbox': bbox,
                                'source': 'YOLOv5-CLIP-Validated'
                            })
                            print(f"    ✅ Validated: {gun_type} (YOLO: {conf:.2f}, CLIP: {validation_score:.2f})")
                        else:
                            print(f"    ⚠️  Rejected: {gun_type} (YOLO: {conf:.2f}, CLIP: {validation_score:.2f}) - Likely phone/false positive")
        
        return {
            'people_count': people_count,
            'weapons': guns,  # Only VALIDATED guns
            'has_weapons': len(guns) > 0
        }
    
    def analyze_scene_clip(self, frame):
        """
        Analyze scene using CLIP zero-shot classification.
        NO TRAINING DATA NEEDED!
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Process with CLIP
        inputs = self.clip_processor(
            text=self.crime_labels,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1)[0].cpu().numpy()
        
        # Aggregate scores by category
        shoplifting_score = np.max(probs[0:3])        # Indices 0-2
        weapon_score = np.max(probs[3:10])            # Indices 3-9 (guns)
        urgent_reaction_score = np.max(probs[10:14])  # Indices 10-13 (rushing/panic)
        normal_score = np.max(probs[14:18])           # Indices 14-17
        
        # Get top prediction
        top_idx = np.argmax(probs)
        top_label = self.crime_labels[top_idx]
        top_category = self.label_to_category[top_idx]
        top_confidence = probs[top_idx]
        
        return {
            'shoplifting': float(shoplifting_score),
            'weapon': float(weapon_score),
            'urgent_reaction': float(urgent_reaction_score),
            'normal': float(normal_score),
            'top_category': top_category,
            'top_label': top_label,
            'top_confidence': float(top_confidence),
            'all_probs': probs.tolist()
        }
    
    def detect_crime(self, video_path, sample_rate=30, threshold=0.5):
        """
        Main detection pipeline: Hybrid YOLO + CLIP + Validation.
        
        Args:
            video_path: Path to video file
            sample_rate: Analyze every Nth frame (30 = ~1 FPS for 30fps video)
            threshold: Crime detection threshold (0.0-1.0)
        
        Returns:
            dict with crime type, confidence, and detailed analysis
        """
        print(f"\n{'='*60}")
        print(f"🔍 ANALYZING VIDEO FOR CRIME (v1 - CLIP Validation)")
        print(f"{'='*60}")
        
        # Extract frames
        frames, frame_indices, fps = self.extract_frames(video_path, sample_rate)
        
        if len(frames) == 0:
            return {
                'crime_detected': False,
                'crime_type': 'NO_CRIME',
                'confidence': 0.0,
                'message': 'No frames extracted from video'
            }
        
        # Analyze each frame
        print(f"\n🔎 Analyzing {len(frames)} frames...")
        
        shoplifting_scores = []
        weapon_scores = []
        urgent_reaction_scores = []
        normal_scores = []
        people_counts = []
        weapon_detections = []
        detections_per_frame = []
        
        gun_frames_indices = []  # Track which frames have gun detections
        
        for i, frame in enumerate(frames):
            # 1. YOLO object detection (WITH VALIDATION!)
            yolo_result = self.detect_objects_yolo(frame)
            people_counts.append(yolo_result['people_count'])
            
            # Track weapon detections from YOLO (VALIDATED ONLY!)
            if yolo_result['has_weapons']:
                weapon_detections.extend(yolo_result['weapons'])
                gun_frames_indices.append(i)  # Record frame index with gun
            
            # 2. CLIP scene analysis
            clip_result = self.analyze_scene_clip(frame)
            
            shoplifting_scores.append(clip_result['shoplifting'])
            weapon_scores.append(clip_result['weapon'])
            urgent_reaction_scores.append(clip_result['urgent_reaction'])
            normal_scores.append(clip_result['normal'])
            
            detections_per_frame.append({
                'frame_idx': frame_indices[i],
                'time': frame_indices[i] / fps,
                'people_count': yolo_result['people_count'],
                'yolo_guns': len(yolo_result['weapons']),
                'clip_scores': {
                    'shoplifting': clip_result['shoplifting'],
                    'weapon': clip_result['weapon'],
                    'urgent_reaction': clip_result['urgent_reaction'],
                    'normal': clip_result['normal']
                },
                'top_prediction': {
                    'category': clip_result['top_category'],
                    'label': clip_result['top_label'],
                    'confidence': clip_result['top_confidence']
                }
            })
        
        # ===== TEMPORAL CONSISTENCY CHECK FOR GUNS =====
        # Require guns to appear in MULTIPLE frames to avoid false positives
        temporally_valid_guns = False
        gun_cluster_info = ""
        
        if len(gun_frames_indices) >= 2:
            # Check if gun detections are temporally close (within 3-frame window)
            consecutive_pairs = 0
            for i in range(len(gun_frames_indices) - 1):
                frame_gap = gun_frames_indices[i+1] - gun_frames_indices[i]
                if frame_gap <= 3:  # Within 3 frames (~3 seconds at 1fps sampling)
                    consecutive_pairs += 1
            
            if consecutive_pairs >= 1:
                temporally_valid_guns = True
                gun_cluster_info = f"{len(gun_frames_indices)} frames with guns, {consecutive_pairs} consecutive pairs"
            else:
                gun_cluster_info = f"{len(gun_frames_indices)} frames with guns, but isolated (no temporal consistency)"
        elif len(gun_frames_indices) == 1:
            gun_cluster_info = "Only 1 frame with gun (isolated detection, likely false positive)"
        else:
            gun_cluster_info = "No gun detections"
        
        # Aggregate results
        avg_shoplifting = np.mean(shoplifting_scores)
        avg_weapon = np.mean(weapon_scores)
        avg_urgent_reaction = np.mean(urgent_reaction_scores)
        avg_normal = np.mean(normal_scores)
        max_shoplifting = np.max(shoplifting_scores)
        max_weapon = np.max(weapon_scores)
        max_urgent_reaction = np.max(urgent_reaction_scores)
        avg_people = np.mean(people_counts)
        
        # Apply temporal consistency filter
        yolo_weapon_count = len(weapon_detections) if temporally_valid_guns else 0
        if not temporally_valid_guns and len(weapon_detections) > 0:
            print(f"\n⚠️  Temporal Filter: Ignoring {len(weapon_detections)} gun detections ({gun_cluster_info})")
        
        print(f"\n📊 RESULTS:")
        print(f"{'─'*60}")
        print(f"YOLO Detections (CLIP-VALIDATED):")
        print(f"  🔫 Guns Found: {yolo_weapon_count}")
        if yolo_weapon_count > 0:
            gun_types = {}
            for w in weapon_detections:
                wtype = w['type']
                gun_types[wtype] = gun_types.get(wtype, 0) + 1
            for wtype, count in gun_types.items():
                print(f"     - {wtype}: {count}x")
            
            # Show validation stats
            avg_validation = np.mean([w['validation_score'] for w in weapon_detections])
            avg_combined = np.mean([w['combined_score'] for w in weapon_detections])
            print(f"  🎯 Avg CLIP Validation Score: {avg_validation:.2f}")
            print(f"  🎯 Avg Combined Score: {avg_combined:.2f}")
        
        print(f"\nCLIP Scene Scores:")
        print(f"  🛒 Shoplifting: {avg_shoplifting:.2%} (max: {max_shoplifting:.2%})")
        print(f"  🔫 Gun Scene:   {avg_weapon:.2%} (max: {max_weapon:.2%})")
        print(f"  🏃 Urgent Rush: {avg_urgent_reaction:.2%} (max: {max_urgent_reaction:.2%})")
        print(f"  ✅ Normal:      {avg_normal:.2%}")
        print(f"\nOther Stats:")
        print(f"  👥 Avg People:  {avg_people:.1f}")
        print(f"{'─'*60}")
        
        # Determine crime type and confidence
        crime_detected = False
        crime_type = "NO_CRIME"
        confidence = avg_normal
        
        # ===== SHOPLIFTING TAKES PRIORITY (if strong signal) =====
        if avg_shoplifting > 0.50 and max_shoplifting > 0.70:
            # Strong shoplifting signal - trust this over gun false positives
            crime_detected = True
            crime_type = "SHOPLIFTING"
            confidence = avg_shoplifting
            print(f"🚨 CRIME DETECTED: SHOPLIFTING (confidence: {confidence:.2%})")
        
        # ===== GUN DETECTION WITH CLIP VALIDATION + HYBRID CONFIDENCE =====
        # Now using COMBINED scores (YOLO + CLIP Validation)!
        elif yolo_weapon_count > 0:
            # Calculate average gun detection confidence (COMBINED SCORE!)
            avg_gun_confidence = np.mean([w['combined_score'] for w in weapon_detections])
            max_gun_confidence = np.max([w['combined_score'] for w in weapon_detections])
            avg_validation_score = np.mean([w['validation_score'] for w in weapon_detections])
            
            print(f"\n  🎯 Combined Scores: avg={avg_gun_confidence:.2f}, max={max_gun_confidence:.2f}")
            print(f"  🎯 CLIP Validation: avg={avg_validation_score:.2f}")
            
            # HYBRID THRESHOLDS based on COMBINED confidence:
            # High confidence (>0.7) → Trust it!
            # Medium confidence (0.5-0.7) → Reduced filtering (already validated!)
            # Low confidence (<0.5) → Temporal support needed
            
            if avg_gun_confidence > 0.7 or max_gun_confidence > 0.8:
                # HIGH CONFIDENCE GUNS - Trust the validation!
                crime_detected = True
                crime_type = "GUN"
                confidence = 0.95
                print(f"🚨 CRIME DETECTED: GUN (Combined score: {avg_gun_confidence:.2f}, {yolo_weapon_count} validated guns)")
            
            elif avg_gun_confidence > 0.5 or max_gun_confidence > 0.6:
                # MEDIUM CONFIDENCE - Reduced filtering (validation helps!)
                if avg_people > 3.0 and yolo_weapon_count < 5:
                    print(f"⚠️  Scene Filter: Ignoring {yolo_weapon_count} guns (very crowded scene with {avg_people:.1f} people)")
                else:
                    crime_detected = True
                    crime_type = "GUN"
                    confidence = 0.90
                    print(f"🚨 CRIME DETECTED: GUN (Combined score: {avg_gun_confidence:.2f}, {yolo_weapon_count} validated guns)")
            
            else:
                # LOW CONFIDENCE - Need temporal support
                if yolo_weapon_count >= 3:  # Lowered from 4 (validation is strong!)
                    crime_detected = True
                    crime_type = "GUN"
                    confidence = 0.85
                    print(f"🚨 CRIME DETECTED: GUN ({yolo_weapon_count} validated guns across frames, combined: {avg_gun_confidence:.2f})")
                else:
                    print(f"⚠️  Insufficient temporal support: {yolo_weapon_count} validated guns (need 3+)")
        
        # URGENT REACTION without YOLO guns - likely shoplifting or other crime
        elif avg_urgent_reaction > 0.40 or max_urgent_reaction > 0.65:
            # Strong rushing/panic signal
            if avg_shoplifting > 0.30:
                crime_detected = True
                crime_type = "SHOPLIFTING"
                confidence = (avg_urgent_reaction + avg_shoplifting) / 2
                print(f"🚨 CRIME DETECTED: SHOPLIFTING (urgent reaction: {avg_urgent_reaction:.2%} + shoplifting: {avg_shoplifting:.2%})")
            elif avg_weapon > 0.20:
                crime_detected = True
                crime_type = "GUN"
                confidence = (avg_urgent_reaction + avg_weapon) / 2
                print(f"🚨 CRIME DETECTED: GUN (urgent reaction: {avg_urgent_reaction:.2%} + weapon scene: {avg_weapon:.2%})")
        
        # CLIP gun detection (if no YOLO temporal guns, but very strong CLIP signal)
        elif avg_weapon > 0.60 or max_weapon > 0.80:
            # Very high CLIP weapon scores without YOLO confirmation
            crime_detected = True
            crime_type = "GUN"
            confidence = avg_weapon
            print(f"🚨 CRIME DETECTED: GUN via CLIP only (confidence: {confidence:.2%})")
        
        # Shoplifting detection - STRICT but can work alone
        elif avg_shoplifting > 0.50 and max_shoplifting > 0.70:
            crime_detected = True
            crime_type = "SHOPLIFTING"
            confidence = avg_shoplifting
            print(f"🚨 CRIME DETECTED: SHOPLIFTING (confidence: {confidence:.2%})")
        
        else:
            print(f"✅ NO CRIME DETECTED (confidence: {avg_normal:.2%})")
        
        return {
            'crime_detected': crime_detected,
            'crime_type': crime_type,
            'confidence': float(confidence),
            'detailed_scores': {
                'shoplifting': {
                    'avg': float(avg_shoplifting),
                    'max': float(max_shoplifting)
                },
                'weapon': {
                    'avg': float(avg_weapon),
                    'max': float(max_weapon)
                },
                'normal': {
                    'avg': float(avg_normal)
                }
            },
            'people_stats': {
                'avg': float(avg_people),
                'max': int(np.max(people_counts)) if people_counts else 0
            },
            'frames_analyzed': len(frames),
            'detections_per_frame': detections_per_frame,
            'validated_weapons': yolo_weapon_count
        }


def batch_process(video_dir, sample_rate=30, threshold=0.5, device='cpu'):
    """
    Process all clips in batch mode and generate dataset summary.
    """
    import os
    
    # Video files
    video_files = [
        f"{video_dir}/1_upscaled.mp4",
        f"{video_dir}/2_upscaled.mp4",
        f"{video_dir}/3_upscaled.mp4",
        f"{video_dir}/4_upscaled.mp4"
    ]
    
    print("="*60)
    print("🚨 BATCH CRIME DETECTION WITH CLIP VALIDATION")
    print("="*60)
    print(f"\nProcessing {len(video_files)} clips...")
    print(f"Sample rate: {sample_rate} frames")
    print(f"Threshold: {threshold}")
    print(f"Device: {device}")
    print("="*60)
    
    # Initialize detector once (reuse for all clips)
    print("\n🚀 Initializing detector...")
    detector = HybridCrimeDetector(device=device)
    
    # Store results
    scene_labels = []
    
    # Process each clip
    for clip_id, video_path in enumerate(video_files):
        print(f"\n{'='*60}")
        print(f"📹 CLIP {clip_id}: {os.path.basename(video_path)}")
        print(f"{'='*60}")
        
        try:
            # Detect crime
            results = detector.detect_crime(
                video_path,
                sample_rate=sample_rate,
                threshold=threshold
            )
            
            # Save individual clip result
            output_file = f"clip{clip_id}_crime_validated.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Saved: {output_file}")
            
            # Add to summary
            scene_labels.append({
                'clip_id': clip_id,
                'label': 'crime' if results['crime_detected'] else 'normal',
                'category': results['crime_type'],
                'confidence': round(results['confidence'], 3)
            })
            
        except Exception as e:
            print(f"❌ Error processing clip {clip_id}: {e}")
            scene_labels.append({
                'clip_id': clip_id,
                'label': 'error',
                'category': 'ERROR',
                'confidence': 0.0
            })
    
    # Generate dataset-level summary
    print(f"\n{'='*60}")
    print("📊 GENERATING DATASET SUMMARY")
    print(f"{'='*60}")
    
    dataset = {
        'dataset': 'mp4_files_id_detectors_upscaled_validated',
        'scene_labels': scene_labels
    }
    
    # Save summary
    summary_file = 'scene_labels_validated.json'
    with open(summary_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✅ Dataset summary saved: {summary_file}")
    
    # Print summary table
    print(f"\n{'='*60}")
    print("📋 RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Clip':<6} {'Label':<10} {'Category':<15} {'Confidence':<12}")
    print("-"*60)
    
    for entry in scene_labels:
        print(f"{entry['clip_id']:<6} {entry['label']:<10} {entry['category']:<15} {entry['confidence']:<12.3f}")
    
    # Statistics
    crime_count = sum(1 for e in scene_labels if e['label'] == 'crime')
    normal_count = sum(1 for e in scene_labels if e['label'] == 'normal')
    gun_count = sum(1 for e in scene_labels if e['category'] == 'GUN')
    shoplifting_count = sum(1 for e in scene_labels if e['category'] == 'SHOPLIFTING')
    
    print(f"\n{'='*60}")
    print("📈 STATISTICS")
    print(f"{'='*60}")
    print(f"Total clips: {len(scene_labels)}")
    print(f"  - Crime: {crime_count}")
    print(f"    • GUN: {gun_count}")
    print(f"    • SHOPLIFTING: {shoplifting_count}")
    print(f"  - Normal: {normal_count}")
    
    # Average confidence
    crime_clips = [e for e in scene_labels if e['label'] == 'crime']
    normal_clips = [e for e in scene_labels if e['label'] == 'normal']
    
    if crime_clips:
        avg_crime_conf = sum(e['confidence'] for e in crime_clips) / len(crime_clips)
        print(f"\nAverage Crime Confidence: {avg_crime_conf:.3f}")
    if normal_clips:
        avg_normal_conf = sum(e['confidence'] for e in normal_clips) / len(normal_clips)
        print(f"Average Normal Confidence: {avg_normal_conf:.3f}")
    
    print(f"\n{'='*60}")
    print("✅ BATCH PROCESSING COMPLETE")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='🚨 Zero-Shot Crime Detection with CLIP Gun Validation (v1)'
    )
    parser.add_argument('--video', type=str, required=False,
                        help='Path to single video file')
    parser.add_argument('--batch', type=str, default=None,
                        help='Batch mode: process all clips in directory')
    parser.add_argument('--sample-rate', type=int, default=30,
                        help='Analyze every Nth frame (default: 30)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Crime detection threshold 0.0-1.0 (default: 0.5)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON file (single video mode only)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu or cuda (default: cpu)')
    
    args = parser.parse_args()
    
    # BATCH MODE
    if args.batch:
        batch_process(
            video_dir=args.batch,
            sample_rate=args.sample_rate,
            threshold=args.threshold,
            device=args.device
        )
        return
    
    # SINGLE VIDEO MODE
    if not args.video:
        parser.error("Either --video or --batch is required")
    
    # Initialize detector
    detector = HybridCrimeDetector(device=args.device)
    
    # Detect crime
    results = detector.detect_crime(
        args.video,
        sample_rate=args.sample_rate,
        threshold=args.threshold
    )
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")
    
    print(f"\n{'='*60}")
    print(f"✅ ANALYSIS COMPLETE")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    main()