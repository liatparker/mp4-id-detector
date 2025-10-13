# Changelog

All notable changes to the MP4 ID Detector project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive requirements files (requirements.txt, requirements_minimal.txt, requirements_exact.txt)
- Automated dependency installer (install_dependencies.py)
- Video annotation system with bounding box visualization
- Crime detection system with YOLO + CLIP
- Clean functional version of the ID detector
- Extensive documentation and setup instructions

### Changed
- Improved README with detailed setup instructions
- Enhanced project structure documentation
- Updated troubleshooting section

### Fixed
- Annotation script data structure issues
- NPZ file reading in clean version
- Bounding box visualization problems

## [1.0.0] - 2024-10-13

### Added
- Initial release of MP4 ID Detector
- Person re-identification system with OSNet and TransReID
- Adaptive clustering with 2-stage approach
- Hybrid embeddings (40% image + 60% video)
- Global identity tracking across multiple video clips
- Crime detection with zero-shot approach
- Video annotation with color-coded identities
- Comprehensive documentation
- Multiple installation options
- Automated setup scripts

### Features
- **Video ID Detection**: Person re-identification across video clips
- **Crime Detection**: Zero-shot crime detection using YOLO + CLIP
- **Video Annotation**: Visual results with bounding boxes and identity labels
- **Adaptive Clustering**: 2-stage clustering with physical proximity logic
- **Hybrid Embeddings**: Combines image and video features
- **Caching System**: NPZ file caching for faster processing
- **Multiple Output Formats**: JSON, CSV, and annotated videos

### Technical Details
- **Models**: OSNet, TransReID, YOLO, InsightFace, CLIP, MediaPipe
- **Frameworks**: PyTorch, OpenCV, Transformers
- **Languages**: Python 3.9+
- **Dependencies**: Comprehensive requirements management
- **Documentation**: Extensive README and setup guides

### Performance
- **Input**: 4 video clips
- **Output**: 11 global identities (baseline configuration)
- **Processing**: Hybrid embeddings with temporal smoothing
- **Caching**: NPZ file system for faster subsequent runs

### Installation
- Virtual environment support
- Multiple dependency options (minimal, full, exact)
- Automated installation script
- Cross-platform compatibility

### Documentation
- Comprehensive README with setup instructions
- Troubleshooting guide
- Project structure documentation
- Contributing guidelines
- Changelog tracking

---

## Version History

- **v1.0.0**: Initial release with core functionality
- **v0.9.0**: Beta version with basic features
- **v0.8.0**: Alpha version for testing

## Future Roadmap

### Planned Features
- [ ] Web interface for video analysis
- [ ] Real-time processing capabilities
- [ ] Additional model support
- [ ] Cloud deployment options
- [ ] API endpoints for integration
- [ ] Mobile app support
- [ ] Advanced analytics dashboard

### Performance Improvements
- [ ] GPU optimization
- [ ] Memory usage optimization
- [ ] Parallel processing
- [ ] Model quantization
- [ ] Edge deployment support

### Documentation
- [ ] API documentation
- [ ] Tutorial videos
- [ ] Advanced usage examples
- [ ] Performance benchmarks
- [ ] Deployment guides
