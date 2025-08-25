# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NAPS (NAPS is ArUco Plus SLEAP) is a tool for quantifying animal behavior and tracking individuals with identity-persistence. It combines deep-learning based pose estimation (SLEAP) with fiducial markers (ArUco tags) to produce high-quality behavioral data.

## Key Commands

### Installation
- **conda**: `conda create -n naps naps-track -c kocherlab -c sleap -c nvidia -c conda-forge`
- **pip**: `pip install naps-track`
- **Development install**: `pip install -e .`

### Testing
- Run all tests: `pytest` or `./tests.sh`
- Run with coverage: `pytest --cov=naps/ --cov-report=xml tests/`
- Run specific test: `pytest tests/test_aruco.py` (or any test file)

### Linting
- Check syntax errors: `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`
- Full linting: `flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics`

### Main Entry Points
- `naps-track`: Main tracking command (implemented in naps/naps_track.py)
- `naps-plot`: Plotting utilities (implemented in naps/naps_plot.py)
- `naps-interactions`: Interaction analysis (implemented in naps/naps_interactions.py)

## Architecture Overview

The codebase is organized into several key components that work together to provide hybrid tracking capabilities:

### Core Modules
- **naps/aruco.py**: ArUcoModel class wraps cv2.aruco for fiducial marker detection
- **naps/matching.py**: Matching pipeline that coordinates marker detection with SLEAP tracking data
- **naps/cost_matrix.py**: Implements cost matrix calculations for optimal assignment between markers and tracked individuals
- **naps/sleap_utils.py**: Utilities for working with SLEAP files and updating labeled frames
- **naps/color_tag.py**: Alternative marker detection using color tags (new feature)

### Data Flow
1. Video + SLEAP predictions (.slp/.h5 files) are loaded
2. ArUco markers are detected in video frames using OpenCV
3. Cost matrix algorithm matches detected markers to SLEAP-tracked individuals
4. Track identities are propagated across frames using the marker associations
5. Updated tracking data is saved back to SLEAP format

### Key Dependencies
- **SLEAP** (1.3.0): Deep learning-based pose estimation
- **OpenCV** (4.6.0.66): Computer vision operations and ArUco detection
- **pandas**: Data manipulation and analysis
- **tqdm**: Progress bars for long-running operations

The system is designed to be modular, allowing different marker detection methods to be swapped in via the marker_detector parameter in the Matching class.