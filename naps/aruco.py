#!/usr/bin/env python

import cv2
from cv2.aruco import Dictionary_get, DetectorParameters_create, detectMarkers


class ArUcoModel:
    def __init__(
        self,
        tag_set: str,
        adaptiveThreshWinSizeMin: int = 3,
        adaptiveThreshWinSizeMax: int = 30,
        adaptiveThreshWinSizeStep: int = 3,
        adaptiveThreshConstant: float = 12,
        perspectiveRemoveIgnoredMarginPerCell: float = 0.13,
        errorCorrectionRate: float = 0.0,
        **kwargs,
    ):

        # Assign the aruco dictionary
        self.aruco_dict = self._assignArucoDict(tag_set)

        """
		ArUco parameters:
		These have been adjusted by dyknapp but are worth playing with if ArUco is too slow or not detecting enough tags.
		These thresholding parameters DRAMATICALLY improve detection rate, while DRAMATICALLY hurting performance.
		Since super fast processing isn't really necessary here they should be fine as is.
		"""
        self.aruco_params = DetectorParameters_create()

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

    @classmethod
    def withTagSet(cls, tag_set, **kwargs):
        return cls(tag_set, **kwargs)

    def detect(self, img):

        return detectMarkers(img, self.aruco_dict, parameters=self.aruco_params)

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
        if not tag_set:
            raise Exception("No tag set defined")
        if tag_set not in ARUCO_DICT:
            raise Exception(f"Unable to assign tag set: {tag_set}")

        # Return the OpenCV tags
        return Dictionary_get(ARUCO_DICT[tag_set])
