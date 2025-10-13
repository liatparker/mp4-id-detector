#!/usr/bin/env python3
"""
Setup script for MP4 ID Detector
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements_minimal.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mp4-id-detector",
    version="1.0.0",
    author="MP4 ID Detector Team",
    author_email="your-email@example.com",
    description="A comprehensive video analysis system for person re-identification and crime detection",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mp4-id-detector",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
        "full": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "ultralytics>=8.0.0",
            "insightface>=0.7.0",
            "transformers>=4.30.0",
            "mediapipe>=0.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mp4-id-detector=video_id_detector2_optimized:main",
            "mp4-crime-detector=crime_no_crime_zero_shot1:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="computer-vision, person-reidentification, object-detection, video-analysis, crime-detection, yolo, clip, deep-learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/mp4-id-detector/issues",
        "Source": "https://github.com/yourusername/mp4-id-detector",
        "Documentation": "https://github.com/yourusername/mp4-id-detector#readme",
    },
)
