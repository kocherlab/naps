#!/usr/bin/env python
import cv2.aruco


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
        **kwargs,
    ):

        # Store the tag set
        self.tag_set = tag_set

        # Store the ArUco parameters
        self.adaptiveThreshWinSizeMin = adaptiveThreshWinSizeMin
        self.adaptiveThreshWinSizeMax = adaptiveThreshWinSizeMax
        self.adaptiveThreshWinSizeStep = adaptiveThreshWinSizeStep
        self.adaptiveThreshConstant = adaptiveThreshConstant
        self.errorCorrectionRate = errorCorrectionRate
        self.perspectiveRemoveIgnoredMarginPerCell = (
            perspectiveRemoveIgnoredMarginPerCell
        )

        """
        Set the ArUco dict and params to None. Create w/ buildModel
        as we cannot pickle them when using multiprocessing
        """
        self.aruco_dict = None
        self.aruco_params = None
        self.model_built = False

    @classmethod
    def withTagSet(cls, tag_set, **kwargs):
        return cls(tag_set, **kwargs)

    def build(self):

        # Assign the aruco dict
        self.aruco_dict = self._assignArucoDict(self.tag_set)

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
        self.aruco_params.adaptiveThreshWinSizeMin = self.adaptiveThreshWinSizeMin
        self.aruco_params.adaptiveThreshWinSizeMax = self.adaptiveThreshWinSizeMax
        self.aruco_params.adaptiveThreshWinSizeStep = self.adaptiveThreshWinSizeStep

        # If too slow, start by adjusting this one up. If we want more tags, lower it (diminishing returns)
        self.aruco_params.adaptiveThreshConstant = self.adaptiveThreshConstant

        # No note for this option
        self.aruco_params.perspectiveRemoveIgnoredMarginPerCell = (
            self.perspectiveRemoveIgnoredMarginPerCell
        )

        # If false positives are a problem, lower this parameter.
        self.aruco_params.errorCorrectionRate = self.errorCorrectionRate

        # Indicate the model has been built
        self.model_built = True

    def detect(self, img):
        # Check if the image is empty, normally when prediction falls outside the image
        if 0 in img.shape:
            return [None]

        # Build the model if needed
        if not self.model_built:
            self.build()

        # Detect ArUco tag(s) within the image
        corners, tags, _ = cv2.aruco.detectMarkers(
            img, self.aruco_dict, parameters=self.aruco_params
        )

        # Return None if no ArUco tag was found
        if len(corners) == 0:
            return [None]

        # Return detected ArUco tags
        return [marker_tag[0] for _, marker_tag in zip(corners, tags)]

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
