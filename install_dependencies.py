#!/usr/bin/env python3
"""
Dependency Installation Script for mp4_id_detector

This script helps install the required dependencies for the video analysis system.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e}")
        if e.stderr:
            print(f"   Details: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9+ is required. Current version:", f"{version.major}.{version.minor}")
        return False
    print(f"✅ Python version {version.major}.{version.minor} is compatible")
    return True

def check_virtual_env():
    """Check if we're in a virtual environment."""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
        return True
    else:
        print("⚠️  No virtual environment detected. Consider using 'source cvenv/bin/activate'")
        return False

def install_requirements(requirements_file):
    """Install requirements from a file."""
    if not Path(requirements_file).exists():
        print(f"❌ Requirements file not found: {requirements_file}")
        return False
    
    return run_command(f"pip install -r {requirements_file}", f"Installing {requirements_file}")

def verify_installation():
    """Verify that key packages are installed."""
    print("🔍 Verifying installation...")
    
    packages_to_check = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("ultralytics", "YOLO"),
        ("insightface", "InsightFace"),
        ("transformers", "Transformers"),
        ("mediapipe", "MediaPipe"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy")
    ]
    
    failed_packages = []
    
    for package, name in packages_to_check:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} not found")
            failed_packages.append(name)
    
    if failed_packages:
        print(f"\n❌ Installation incomplete. Missing packages: {', '.join(failed_packages)}")
        return False
    else:
        print("\n🎉 All packages installed successfully!")
        return True

def main():
    """Main installation process."""
    print("=" * 60)
    print("MP4_ID_DETECTOR DEPENDENCY INSTALLER")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check virtual environment
    check_virtual_env()
    
    print("\n📦 Available installation options:")
    print("1. Minimal (recommended) - requirements_minimal.txt")
    print("2. Full - requirements.txt") 
    print("3. Exact versions - requirements_exact.txt")
    print("4. Custom file")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    requirements_files = {
        "1": "requirements_minimal.txt",
        "2": "requirements.txt", 
        "3": "requirements_exact.txt"
    }
    
    if choice in requirements_files:
        requirements_file = requirements_files[choice]
    elif choice == "4":
        requirements_file = input("Enter path to requirements file: ").strip()
    else:
        print("❌ Invalid choice")
        sys.exit(1)
    
    print(f"\n🚀 Installing dependencies from {requirements_file}...")
    
    # Install requirements
    if not install_requirements(requirements_file):
        print("❌ Installation failed!")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("❌ Verification failed!")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 INSTALLATION COMPLETE!")
    print("=" * 60)
    print("You can now run the video analysis system:")
    print("  python video_id_detector2_optimized.py")
    print("  python crime_no_crime_zero_shot1.py --batch ./videos")

if __name__ == "__main__":
    main()
