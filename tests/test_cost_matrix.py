import pytest

import numpy as np
import pandas as pd

from naps.cost_matrix import CostMatrix


def test_CostMatrix():

    # Create an example of the CostMatrix input array
    value_array = np.array([[ 2.,  2.,  3.],
                            [ 3.,  1.,  3.],
                            [ 3.,  1.,  3.],
                            [ 2.,  1., np.nan],
                            [ 2.,  1., np.nan]])



    # Create the cost matrix, the assign the track/tag pairs
    cost_matrix = CostMatrix.fromArray(value_array, 0, 4, 2)
    assignment_array = cost_matrix.assignTrackTagPairs()

    # Confirm the contents of the assignment dictionary
    assert assignment_array.shape == (1, 3)
    assert list(assignment_array[0]) == [2, 1, 3]


def test_CostMatrix_linearAssignment():

    # Create a dataframe of basic counts from a dict
    value_dict = {
        "a": {1: 4, 2: 2, 3: 3},
        "b": {1: 1, 2: 0, 3: 2},
        "c": {1: 3, 2: 5, 3: 2},
    }
    value_array = pd.DataFrame.from_dict(value_dict).fillna(0).values

    # Store the linear assignment track order
    track_indices = [
        track_index for track_index, _ in CostMatrix._linearAssignment(value_array)
    ]

    # Confirm the track order
    assert track_indices == [1, 0, 2]
