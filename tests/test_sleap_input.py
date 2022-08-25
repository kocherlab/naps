import logging

import numpy as np

from naps.sleap_utils import load_tracks_from_slp

logger = logging.getLogger(__name__)


def test_load_tracks_from_slp():
    locations, node_names = load_tracks_from_slp("tests/data/example.slp")
    assert isinstance(locations, np.ndarray)
