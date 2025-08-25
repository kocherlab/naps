#!/usr/bin/env python
"""
Custom installation script to handle SLEAP's broken numpy constraints.
This script installs SLEAP without its dependencies and then installs 
compatible versions of everything else.
"""
import subprocess
import sys

def run_pip(args):
    """Run pip with given arguments"""
    result = subprocess.run([sys.executable, '-m', 'pip'] + args, 
                          capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Warning: {result.stderr}")
    return result.returncode == 0

def install_naps_with_sleap():
    """Install NAPS with SLEAP, handling dependency conflicts"""
    print("Installing NAPS with SLEAP compatibility...")
    
    # Install SLEAP without its broken numpy constraint
    print("1. Installing SLEAP (without dependencies)...")
    if not run_pip(['install', 'sleap', '--no-deps']):
        print("Failed to install SLEAP")
        return False
    
    # Install SLEAP's dependencies manually with compatible versions
    print("2. Installing SLEAP dependencies...")
    sleap_deps = [
        'python-rapidjson', 'imgstore', 'nixio>=1.5.3', 
        'qimage2ndarray', 'segmentation-models', 'pynwb>=2.3.3'
    ]
    if not run_pip(['install'] + sleap_deps):
        print("Failed to install SLEAP dependencies")
        return False
    
    # Install core scientific stack with modern versions
    print("3. Installing core dependencies...")
    core_deps = [
        'numpy>=1.26.0', 'scipy>=1.7.0', 'h5py>=3.1.0', 
        'pandas', 'opencv-contrib-python-headless'
    ]
    if not run_pip(['install'] + core_deps):
        print("Failed to install core dependencies")
        return False
    
    # Install NAPS
    print("4. Installing NAPS...")
    if not run_pip(['install', 'git+https://github.com/kocherlab/naps.git@wolf-dev-fixing-deps']):
        print("Failed to install NAPS")
        return False
    
    # Verify installation
    print("5. Verifying installation...")
    try:
        import naps
        import sleap
        import cv2
        import numpy as np
        
        # Handle OpenCV version check (some installations don't have __version__)
        try:
            cv_version = cv2.__version__
        except AttributeError:
            cv_version = getattr(cv2, 'cv2.__version__', 'unknown')
        
        print(f"✅ Success! NAPS {naps.__version__}, SLEAP {sleap.__version__}, "
              f"OpenCV {cv_version}, NumPy {np.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == '__main__':
    success = install_naps_with_sleap()
    sys.exit(0 if success else 1)