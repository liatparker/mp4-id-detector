#!/usr/bin/env python3
"""
Generate Scene Labels from Crime Detection Results
Automatically produces dataset-level scene labels by testing all videos
"""

import json
import os
import sys

def generate_scene_labels():
    """Generate scene labels from existing crime detection results"""
    
    print("📋 Generating Scene Labels from Crime Detection Results")
    print("="*70)
    
    # Check if crime detection results exist
    crime_results = {}
    missing_files = []
    
    for clip_id in range(4):
        filename = f'clip{clip_id}_crime.json'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                crime_results[clip_id] = json.load(f)
            print(f"✅ Loaded: {filename}")
        else:
            missing_files.append(filename)
            print(f"⚠️  Missing: {filename}")
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} crime detection result(s)")
        print("   Run crime detection first:")
        print("   ./run_crime_detection.sh")
        print("   OR")
        print("   python3 crime_no_crime_zero_shot.py --video <video_path>")
        return None
    
    # Generate scene labels
    scene_labels = []
    
    for clip_id in sorted(crime_results.keys()):
        result = crime_results[clip_id]
        
        # Determine label from crime detection
        crime_detected = result.get('crime_detected', False)
        label = "crime" if crime_detected else "normal"
        
        scene_labels.append({
            "clip_id": clip_id,
            "label": label
        })
        
        # Print result
        crime_type = result.get('crime_type', 'NO_CRIME')
        confidence = result.get('confidence', 0.0)
        status = "🚨" if label == "crime" else "✅"
        
        print(f"{status} Clip {clip_id}: {label.upper():6s} - {crime_type:15s} (confidence: {confidence:.1%})")
    
    # Create output
    output = {
        "dataset": "mp4_files_id_detectors_upscaled",
        "scene_labels": scene_labels,
        "metadata": {
            "total_clips": len(scene_labels),
            "crime_clips": sum(1 for s in scene_labels if s['label'] == 'crime'),
            "normal_clips": sum(1 for s in scene_labels if s['label'] == 'normal'),
            "detection_method": "Automated crime detection (CLIP + YOLO)",
            "note": "Labels generated from crime detection analysis"
        }
    }
    
    # Save to file
    output_file = "scene_labels.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "="*70)
    print(f"✅ Scene labels saved to: {output_file}")
    print("="*70)
    
    # Print summary
    print("\n📊 Summary:")
    print(f"   Total clips: {output['metadata']['total_clips']}")
    print(f"   Crime clips: {output['metadata']['crime_clips']}")
    print(f"   Normal clips: {output['metadata']['normal_clips']}")
    
    # Print the JSON
    print("\n📄 Generated scene_labels.json:")
    print(json.dumps(output, indent=2))
    
    return output

if __name__ == "__main__":
    result = generate_scene_labels()
    if result is None:
        sys.exit(1)

