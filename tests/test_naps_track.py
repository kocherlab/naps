import os
import pytest
import shutil
import filecmp
import tempfile

import pandas as pd

import naps.naps_track as naps_track

def test_naps_track():

    # Create a temporary directory 
    test_dir = tempfile.mkdtemp()

    # Create the temporary output file
    test_output = os.path.join(test_dir, 'test_naps_track_output.h5')

    # Create a list of the arguments
    argument_list = [
        '--slp-path',
        'tests/data/example.slp',
        '--h5-path',
        'tests/data/example.analysis.h5',
        '--video-path',
        'tests/data/example.mp4',
        '--tag-node',
        '0',
        '--start-frame',
        '0',
        '--end-frame',
        '29',
        '--aruco-marker-set',
        'DICT_4X4_100',
        '--output-path',
        test_output
    ]

    # Run naps-track with the argument list
    naps_track.main(argument_list)

    # Check if we get the expected output
    assert filecmp.cmp(test_output, 'tests/data/example_naps_track_output.h5')

    # Remove the temporary directory
    shutil.rmtree(test_dir)