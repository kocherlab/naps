from naps.sleap_utils import load_tracks_from_slp, reconstruct_slp
import logging
import numpy as np
import pytest
import sleap
import os

logger = logging.getLogger("NAPS Testing Logger")


def test_load_tracks_from_slp():
    locations = load_tracks_from_slp("data/example.h5")
    assert isinstance(locations, np.ndarray)


def test_sleap_recreation():

    path = "data/example.slp"
    outpath = "data/example_reconstruction.slp"

    matching_array = np.random.randint(0, 50, (7826), dtype="int64")
    frames = reconstruct_slp(path, matching_array)
    new_labels = sleap.Labels(labeled_frames=frames)
    try:
        sleap.save_file(new_labels, outpath)
        labels = sleap.load_file(outpath)
    finally:
        os.remove(outpath)
    assert new_labels == labels
