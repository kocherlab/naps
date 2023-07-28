#!/usr/bin/env python
import cv2.aruco

import numpy as np


class ArUcoModel:
    """Class providing a wrapper around the cv2.aruco library"""

    def __init__(
        self,
        tag_set: str,
        adaptiveThreshWinSizeMin: int,
        adaptiveThreshWinSizeMax: int,
        adaptiveThreshWinSizeStep: int,
        adaptiveThreshConstant: float,
        perspectiveRemoveIgnoredMarginPerCell: float,
        errorCorrectionRate: float,
        tag_subset_list: list = [],
        **kwargs,
    ):

        # Assign the aruco dict
        self.aruco_dict = self._assignArucoDict(tag_set)

        """
        ArUco parameters:
        These have been adjusted by dyknapp but are worth playing with if ArUco is too slow or not detecting enough tags.
        These thresholding parameters DRAMATICALLY improve detection rate, while DRAMATICALLY hurting performance.
        Since super fast processing isn't really necessary here they should be fine as is.
        """
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        """
        Assign the corner refinement method:

        Should we permit all options?
        """
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        # Window parameters
        self.aruco_params.adaptiveThreshWinSizeMin = adaptiveThreshWinSizeMin
        self.aruco_params.adaptiveThreshWinSizeMax = adaptiveThreshWinSizeMax
        self.aruco_params.adaptiveThreshWinSizeStep = adaptiveThreshWinSizeStep

        # If too slow, start by adjusting this one up. If we want more tags, lower it (diminishing returns)
        self.aruco_params.adaptiveThreshConstant = adaptiveThreshConstant

        # No note for this option
        self.aruco_params.perspectiveRemoveIgnoredMarginPerCell = (
            perspectiveRemoveIgnoredMarginPerCell
        )

        # If false positives are a problem, lower this parameter.
        self.aruco_params.errorCorrectionRate = errorCorrectionRate

        # Assign an empty subset dict
        self.subset_dict = {}

        # Check if a tag subset list was specified
        if tag_subset_list:

            # Update the subset dict
            self.subset_dict = {i: t for i, t in enumerate(tag_subset_list)}

            # Create the subset dict
            subset_dict = cv2.aruco.custom_dictionary(0, self.aruco_dict.markerSize, 1)
            subset_dict.bytesList = np.take(self.aruco_dict.bytesList, tag_subset_list, axis = 0)

            # Replace the aruco dict with the subset dict
            self.aruco_dict = subset_dict


    @classmethod
    def withTagSet(cls, tag_set, **kwargs):
        return cls(tag_set, **kwargs)

    def detect(self, img):

        # Detect ArUco tag(s) within the image
        corners, tags, _ = cv2.aruco.detectMarkers(
            img, self.aruco_dict, parameters=self.aruco_params
        )

        # Return None if no ArUco tag was found
        if len(corners) == 0:
            return [None]

        # Assing the tags
        marker_tags = [marker_tag[0] for _, marker_tag in zip(corners, tags)]

        # Update the tags if using a subset
        if self.subset_dict:
            marker_tags = [self.subset_dict[marker_tag] for marker_tag in marker_tags]

        # Return detected ArUco tags
        return marker_tags

    def _assignArucoDict(self, tag_set):

        # Define names of each possible ArUco tag OpenCV supports
        ARUCO_DICT = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
            "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
            "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
            "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
        }

        # Check there are no problem with the tag set
        if tag_set not in ARUCO_DICT:
            raise Exception(f"Unable to assign tag set: {tag_set}")

        # Return the OpenCV tags
        return cv2.aruco.Dictionary_get(ARUCO_DICT[tag_set])
