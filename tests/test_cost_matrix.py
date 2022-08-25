import pandas as pd

from naps.cost_matrix import CostMatrix


def test_CostMatrix():

    # Create an example of the CostMatrix input dict
    value_dict = {
        0: {"a": [1, 2], "b": [2], "c": [1, 3]},
        1: {"a": [3], "b": [1, 2], "c": [1, 3]},
        2: {"a": [3], "b": [1, 2, 3], "c": [3]},
        3: {"a": [2], "b": [1, 2, 3], "c": []},
        4: {"a": [2], "b": [1, 2, 3], "c": []},
        5: {"a": [1, 2], "b": [2], "c": [1, 3]},
    }

    # Create the cost matrix, the assign the track/tag pairs
    cost_matrix = CostMatrix.fromDict(value_dict, 0, 5, 2)
    assignment_dict = cost_matrix.assignTrackTagPairs()

    # Confirm the contents of the assignment dictionary
    assert 2 in assignment_dict
    assert list(assignment_dict) == [2, 3]
    assert list(assignment_dict[2]) == ["b", "a", "c"]
    assert list(assignment_dict[3]) == ["b", "a", "c"]


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
