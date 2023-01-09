import os
import shutil
import tempfile

import numpy as np

import naps.naps_track as naps_track
from naps.sleap_utils import load_tracks_from_slp


def test_naps_track():

    # Create a temporary directory
    test_dir = tempfile.mkdtemp()

    # Create the temporary output file
    test_output = os.path.join(test_dir, "test_naps_track_output.h5.slp")

    # Create a list of the arguments
    argument_list = [
        "--slp-path",
        "tests/data/example.slp",
        "--h5-path",
        "tests/data/example.analysis.h5",
        "--video-path",
        "tests/data/example.mp4",
        "--tag-node-name",
        "Tag",
        "--start-frame",
        "0",
        "--end-frame",
        "29",
        "--aruco-marker-set",
        "DICT_4X4_100",
        "--output-path",
        test_output,
    ]

    # Run naps-track with the argument list
    naps_track.main(argument_list)

    # Get the contents of the file, since we cannot directly compare the files
    test_locations, test_node_names = load_tracks_from_slp(test_output)
    example_locations, example_node_names = load_tracks_from_slp(
        "tests/data/example_naps_track_output.h5.slp"
    )

    # Check if we get the expected output
    assert np.array_equal(test_locations, example_locations, equal_nan=True)
    assert test_node_names == example_node_names

    # Remove the temporary directory
    shutil.rmtree(test_dir)
