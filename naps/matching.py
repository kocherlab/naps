import sys
import cv2
import math

import numpy as np

from naps.aruco import ArUcoModel
from naps.cost_matrix import CostMatrix


class Matching:
    def __init__(
        self,
        video_filename: str,
        video_first_frame: int,
        video_last_frame: int,
        aruco_model: ArUcoModel,
        tag_node_matrix: np.ndarray,
        min_sleap_score: float = 0.1,
        half_rolling_window_size: int = 5,
        aruco_crop_size: int = 50,
        threads: int = 1,
        **kwargs
    ):

        # Video arguments
        self.video_filename = video_filename
        self.video_first_frame = video_first_frame
        self.video_last_frame = video_last_frame

        # SLEAP arguments
        self.tag_node_matrix = tag_node_matrix
        self.min_sleap_score = min_sleap_score

        # Matching arguments
        self.half_rolling_window_size = half_rolling_window_size
        self.aruco_crop_size = aruco_crop_size
        self.aruco_model = aruco_model

        # General arguments
        self.threads = threads

        # Output
        self.matching_dict = {}

    @classmethod
    def test(cls, *args, **kwargs):
        return cls(
            "20210909QR_MultiCpu_010.mp4",
            21,
            99,
            aruco_model=ArUcoModel.withTagSet("DICT_4X4_100"),
            *args,
            **kwargs
        )

    def match(self):
        def framesPerThread():
            """
            Yield equal size windows based on the number of threads
            """

            # Adjust the start and end to account for the window
            rolling_window_start = (
                self.video_first_frame + self.half_rolling_window_size
            )
            rolling_window_end = (
                self.video_last_frame - self.half_rolling_window_size + 1
            )

            # Assign the frames per thread
            frames_per_thread = math.ceil(
                (rolling_window_end - rolling_window_start) / self.threads
            )

            # Return windows of equal size for each thread
            for frame in range(
                rolling_window_start, rolling_window_end, frames_per_thread
            ):
                frame_start = max(
                    [self.video_first_frame, frame - self.half_rolling_window_size]
                )
                frame_end = min(
                    [
                        self.video_last_frame,
                        frame + frames_per_thread + self.half_rolling_window_size - 1,
                    ]
                )
                yield frame_start, frame_end

        # Assign the matching jobs
        self.matching_dict = {}
        for frame_start, frame_end in framesPerThread():
            self.matching_dict.update(self._matchJob(frame_start, frame_end))
        return self.matching_dict

    def _matchJob(self, frame_start: int, frame_end: int):

        # Create a dict to store the matches for this job
        job_match_dict = {}

        # Set the current frame of the job
        current_frame = frame_start

        # Initialize OpenCV, then set the starting frame (0-based)
        video = cv2.VideoCapture(self.video_filename)
        video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        # Read in the frame, confirm it was sucessful
        frame = MatchFrame.fromCV2(*video.read())
        while frame and current_frame < frame_end:

            # REMOVE WHILE NOT TESTING
            # import numpy as np
            # test_array = np.asarray((np.asarray((2556, 3020)), np.asarray((2557, 3021))))
            tag_locations = self.tag_node_matrix[current_frame, :, :].T
            # Just get [x,y] for each
            tag_locations = [tag_locations[i, :] for i in range(tag_locations.shape[0])]
            # Crop and return the ArUco tags
            frame.cropArUcoWithCoordsArray(tag_locations, self.aruco_crop_size)
            job_match_dict[current_frame] = frame.returnArUcoTags(self.aruco_model)

            # Advance to the next frame
            frame = MatchFrame.fromCV2(*video.read())
            current_frame += 1

            # REMOVE WHILE NOT TESTING
            # break

        # REMOVE WHILE NOT TESTING
        # frame_end = frame_start

        # Create the cost matrix and assign the track/tag pairs for each frame
        job_cost_matrix = CostMatrix.fromDict(
            job_match_dict, frame_start, frame_end, self.half_rolling_window_size
        )
        job_track_tag_pair_dict = job_cost_matrix.assignTrackTagPairs()
        return job_track_tag_pair_dict


class MatchFrame:
    def __init__(self, frame_exists: bool, frame: np.ndarray, *args, **kwargs):

        # Image arguments
        self.frame = frame
        self.frame_exists = frame_exists
        self.frame_images = {}

    def __nonzero__(self):
        return self.frame_exists

    @classmethod
    def fromCV2(cls, *args, **kwargs):

        return cls(*args, **kwargs)

    def cropArUcoWithCoordsArray(self, coords_array: np.array, crop_size: int):
        def croppedCoords(coord, crop_size, coord_max):
            return np.maximum(int(coord) - crop_size, 0), np.minimum(
                int(coord) + crop_size, coord_max - 1
            )

        # Loop the frame track coordinates
        for track, coords in enumerate(coords_array):

            # Assign the min/max coords for cropping
            y_min, y_max = croppedCoords(coords[1], crop_size, self.frame.shape[0])
            x_min, x_max = croppedCoords(coords[0], crop_size, self.frame.shape[1])

            # Assign and store the cropped track image
            self.frame_images[track] = self.frame[y_min:y_max, x_min:x_max, 0]

    def returnArUcoTags(self, aruco_model: ArUcoModel):

        from collections import defaultdict

        track_tag_dict = defaultdict(list)

        # Loop the frame track images
        for track, frame_image in self.frame_images.items():

            # Detect ArUco tags
            corners, tags, rejected = aruco_model.detect(frame_image)

            # Skip to next track if no tags were found
            if len(corners) == 0:
                continue

            # Iterate through detected tags and append results to a results list
            for marker_corners, marker_tag in zip(corners, tags):
                track_tag_dict[track].append(marker_tag[0])

        return track_tag_dict


# model = ArUcoModel.withTagSet('DICT_4X4_100')
# test = Matching(model)

# test = Matching.test()
# test.match()
