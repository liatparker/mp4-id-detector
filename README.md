# Video Analysis System

A comprehensive video analysis system with two main components:
1. **Video ID Detection System** - Person re-identification and tracking across multiple video clips
2. **Crime Detection System** - Zero-shot crime detection using YOLO + CLIP

##  Current Status

**Baseline Configuration**: 11 Global Identities (Physical Proximity Chain DISABLED)


##  Quick Start

### Initial Setup

**Option 1: Automated Installation (Recommended)**
```bash
# 1. Activate virtual environment
source cvenv/bin/activate

# 2. Run the installation script
python install_dependencies.py
```

**Option 2: Manual Installation**
```bash
# 1. Activate virtual environment
source cvenv/bin/activate

# 2. Install dependencies (choose one)
pip install -r requirements_minimal.txt  # Recommended
# OR
pip install -r requirements.txt          # Full installation
# OR  
pip install -r requirements_exact.txt     # Exact versions

# 3. Verify installation
python -c "import torch, cv2, ultralytics; print('✅ Dependencies installed successfully!')"
```

### Video ID Detection System

**Automatic Run (Recommended)**
```bash
# Option 1: Bash Script
./run_detector.sh

# Option 2: Python Script
python run_detector.py
```

**Manual Run**
```bash
# Activate virtual environment
source cvenv/bin/activate

# Run the ID detector
python video_id_detector2_optimized.py
```

### Crime Detection System

**Automatic Run (Recommended)**
```bash
# Option 1: Bash Script
./run_crime_detection.sh

# Option 2: Python Script
python run_crime_detection.py
```

**Manual Run**
```bash
# Activate virtual environment
source cvenv/bin/activate

# Run crime detection on all videos
python crime_no_crime_zero_shot1.py --batch ./videos --sample-rate 30 --threshold 0.5

# Run on single video
python crime_no_crime_zero_shot1.py --video ./videos/1_upscaled.mp4



### Video Annotation System
# for json and csv 
python video_id_detector2_optimized.py 
# for annotations 
python export_annotated_videoes_v3_from_cache.py

# relevant outputs at outputs_v3

# Activate virtual environment
source cvenv/bin/activate



## 📁 Project Structure

```
mp4_id_detector/
├── videos/                          # Input video files
├── outputs_v3/                      # ID Detection results
├── video_id_detector2_optimized.py  # ID Detection script
├── crime_no_crime_zero_shot1.py     # Crime Detection script
├── export_annotated_videoes_v3_from_cache.py  # Video annotation script
├── requirements.txt                 # Full dependencies
├── requirements_minimal.txt         # Minimal dependencies (recommended)
├── requirements_exact.txt          # Exact working versions
├── install_dependencies.py       # Automated installer
└── README.md                        # This file
```



## 📊 Results

### Video ID Detection System
- **Global Identity Catalogue** (JSON/CSV): Final person assignments
- **Track Embeddings**: Feature vectors for each tracklet
- **Annotated Videos**: Visual results with bounding boxes and IDs

### Crime Detection System
- **Crime Analysis JSON**: Detailed crime detection results per video
- **Weapon Detection**: YOLO + CLIP validated gun detection
- **Behavior Analysis**: CLIP-based suspicious activity detection
- **Scene Labels**: Crime/no-crime classification per frame

### Video Annotation System
- **Annotated Videos**: Visual results with identity tracking
- **Bounding Boxes**: Person detection with color-coded identities
- **Identity Labels**: Global ID, Track ID, and face detection status
- **Frame Information**: Frame counter and detection statistics

## ⚙️ Configuration

Current settings in `video_id_detector2_optimized.py`:
- **Clustering Method**: Adaptive (2-stage)
- **Face Weight**: 0.2
- **Body Weight**: 0.6
- **Motion Weight**: 0.1
- **Pose Weight**: 0.1
- **Physical Proximity Chain**: DISABLED (baseline mode)

## 🔧 Advanced Features (Available but Disabled)

- Physical Proximity Chain Logic
- Robust Appearance Weights
- Cross-clip Spatial Proximity
- Clothing-focused Distance Calculation

## 📈 Performance

- **Input**: 4 video clips
- **Output**: 11 global identities
- **Processing**: Hybrid embeddings (40% image + 60% video)
- **Models**: OSNet + TransReID ensemble

## 🛠️ Requirements

### System Requirements
- **Python**: 3.12 (recommended) or 3.9+
- **Virtual environment**: `cvenv/` (already created)
- **GPU**: CUDA-compatible GPU recommended (optional)
- **RAM**: 8GB+ recommended
- **Storage**: 10GB+ for models and cache

### Python Dependencies

**Option 1: Minimal Installation (Recommended)**
```bash
pip install -r requirements_minimal.txt
```

**Option 2: Full Installation**
```bash
pip install -r requirements.txt
```

**Option 3: Exact Versions (Most Stable)**
```bash
pip install -r requirements_exact.txt
```

### Required Models
- **OSNet**: Person re-identification
- **TransReID**: Enhanced re-identification
- **YOLO**: Object detection
- **InsightFace**: Face recognition
- **CLIP**: Vision-language understanding

### System Dependencies (Install Separately)
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg libgl1-mesa-glx libglib2.0-0

# macOS
brew install ffmpeg

# Windows
# Download ffmpeg from https://ffmpeg.org/download.html
```

## 📝 Notes

- The system uses cached embeddings for faster processing
- Results are saved in `outputs_v3/` directory
- The baseline configuration provides stable, reliable results
- Advanced features can be enabled by modifying configuration flags
# mp4-id-detector
