#!/usr/bin/env python
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


class CostMatrix:
    def __init__(
        self,
        unmatched_array: dict,
        first_frame: int,
        last_frame: int,
        half_rolling_window_size: int,
        **kwargs
    ):

        # Matrix arguments
        self.unmatched_array = unmatched_array
        self.matched_array = np.full([unmatched_array.shape[0] - (2 * half_rolling_window_size), unmatched_array.shape[1]], np.nan)

        # Matching arguments
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.half_rolling_window_size = half_rolling_window_size

        # Assignment argument
        self.assignment_method = self._linearAssignment

    @classmethod
    def fromArray(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def assignTrackTagPairs(self):

        # Create the cost matrix
        cost_dict = defaultdict(lambda: defaultdict(int))

        # Loop the positions in the unmatched array
        for pos in range(0, self.unmatched_array.shape[0]):

            # Assign the track/tag combinations for this position
            for track, tag in enumerate(self.unmatched_array[pos]):
                if not np.isnan(tag): cost_dict[track][int(tag)] -= 1

            # Assign the rolling window position
            match_pos = pos - self.half_rolling_window_size

            # Check if the match position should undergo linear assignment
            if (
                match_pos < self.half_rolling_window_size
                or match_pos > self.unmatched_array.shape[0] - self.half_rolling_window_size
            ):
                continue

            # Assign the first frame of the window
            start_of_window = match_pos - self.half_rolling_window_size

            # Remove a position, if needed
            if start_of_window > 0:
                for track, tag in enumerate(self.unmatched_array[start_of_window - 1]):
                    if not np.isnan(tag):
                        cost_dict[track][int(tag)] += 1
                        if cost_dict[track][int(tag)] == 0:
                            del cost_dict[track][int(tag)]

            # Create a dataframe of the matrix
            cost_dataframe = pd.DataFrame.from_dict(cost_dict).fillna(0)

            # Store the assignments
            for track_index, tag_index in self.assignment_method(cost_dataframe.values):
                self.matched_array[start_of_window][
                    cost_dataframe.columns[track_index]
                ] = cost_dataframe.index[tag_index]

        return self.matched_array

    @staticmethod
    def _linearAssignment(value_array):

        # Solve the linear assignment problem using Jonker-Volgenant algorithm
        tag_indices, track_indices = linear_sum_assignment(value_array)

        # Store the assignments
        for track_index, tag_index in zip(track_indices, tag_indices):
            yield track_index, tag_index
