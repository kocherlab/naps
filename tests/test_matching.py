import cv2
import pytest

from naps.aruco import ArUcoModel
from naps.matching import MatchFrame

def test_imread():

    # Assign the image file
    aruco_image_file = "tests/data/example_ArUco_image.jpg"

    # Confirm the model loads without error
    try:
        aruco_image = cv2.imread(aruco_image_file, 0)
    except Exception as exc:
        assert False, f"Unable to read image: {aruco_image_file}"

def test_MatchFrame():

    # Assign the image file
    aruco_image_file = "tests/data/example_ArUco_image.jpg"

    # Confirm the model loads without error
    try:
        aruco_image = cv2.imread(aruco_image_file, 0)
        MatchFrame.fromCV2(True, aruco_image)
    except Exception as exc:
        assert False, f"MatchFrame assignment error: (True, {aruco_image_file})"

def test_MatchFrame_error():

    # Assign the image file
    aruco_image_file = "tests/data/example_ArUco_image.jpg"
    aruco_image = cv2.imread(aruco_image_file, 0)

    with pytest.raises(Exception) as e_info:
        MatchFrame.fromCV2()

    with pytest.raises(Exception) as e_info:
        MatchFrame.fromCV2(True)

    with pytest.raises(Exception) as e_info:
        MatchFrame.fromCV2(aruco_image)

def test_MatchFrame_cropArUcoWithCoordsArray():

    # Assign the image file
    aruco_image_file = "tests/data/example_ArUco_image.jpg"
    aruco_image = cv2.imread(aruco_image_file)

    # Assign the centroid coords for each tag
    coords_list = [
        (225, 225),
        (925, 225),
        (575, 575),
        (225, 925),
        (925, 925)
    ]

    match_frame = MatchFrame.fromCV2(True, aruco_image)
    match_frame.cropArUcoWithCoordsArray(coords_list, 200)

    # Confirm the cropped images were created
    assert len(match_frame.frame_images) == 5
    assert list(match_frame.frame_images) == [0, 1, 2, 3, 4]

    # Check the cropped images
    for match_image in match_frame.frame_images.values():
        assert match_image.shape == (400, 400)

def test_MatchFrame_returnArUcoTags():

    # Assign the parameters for the ArUcoModel
    param_dict = {
        "adaptiveThreshWinSizeMin": 3,
        "adaptiveThreshWinSizeMax": 10,
        "adaptiveThreshWinSizeStep": 3,
        "adaptiveThreshConstant": 10,
        "perspectiveRemoveIgnoredMarginPerCell": 0.1,
        "errorCorrectionRate": 0.1,
    }

    # Build the ArUcoModel
    test_model = ArUcoModel.withTagSet("DICT_4X4_100", **param_dict)
    test_model.buildModel()

    # Assign the image file
    aruco_image_file = "tests/data/example_ArUco_image.jpg"
    aruco_image = cv2.imread(aruco_image_file)

    # Assign the centroid coords for each tag
    coords_list = [
        (225, 225),
        (925, 225),
        (575, 575),
        (225, 925),
        (925, 925)
    ]
 
    match_frame = MatchFrame.fromCV2(True, aruco_image)
    match_frame.cropArUcoWithCoordsArray(coords_list, 200)
    match_dict = match_frame.returnArUcoTags(test_model)

    # Create the matching results
    assert len(match_dict) == 5
    assert list(match_dict) == [0, 1, 2, 3, 4]
    assert list(match_dict.values()) == [[1], [2], [3], [4], [5]]