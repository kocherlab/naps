from naps.sleap_utils import load_tracks_from_slp, reconstruct_slp
import logging
import numpy as np
import pytest
import sleap
import os

logger = logging.getLogger(__name__)


def test_load_tracks_from_slp():
    locations = load_tracks_from_slp("tests/data/example.slp")
    assert isinstance(locations, np.ndarray)


def test_sleap_recreation():

    path = "tests/data/example.slp"
    outpath = "tests/data/example_reconstruction.slp"

    num_tracks = 50

    matching_array = np.random.randint(0, num_tracks, (1204), dtype="int64")
    frames = reconstruct_slp(path, matching_array, 0, 100)
    new_labels = sleap.Labels(labeled_frames=frames)
    try:
        sleap.Labels.save_file(new_labels, outpath)
        labels = sleap.load_file(outpath)
    finally:
        os.remove(outpath)
    assert isinstance(labels, sleap.Labels)
