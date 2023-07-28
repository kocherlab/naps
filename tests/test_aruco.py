import cv2
import pytest

from naps.aruco import ArUcoModel


@pytest.mark.parametrize(
    "tag_set",
    [
        "DICT_4X4_50",
        "DICT_4X4_100",
        "DICT_4X4_250",
        "DICT_4X4_1000",
        "DICT_5X5_50",
        "DICT_5X5_100",
        "DICT_5X5_250",
        "DICT_5X5_1000",
        "DICT_6X6_50",
        "DICT_6X6_100",
        "DICT_6X6_250",
        "DICT_6X6_1000",
        "DICT_7X7_50",
        "DICT_7X7_100",
        "DICT_7X7_250",
        "DICT_7X7_1000",
        "DICT_ARUCO_ORIGINAL",
        "DICT_APRILTAG_16h5",
        "DICT_APRILTAG_25h9",
        "DICT_APRILTAG_36h10",
        "DICT_APRILTAG_36h11",
    ],
)
def test_ArUcoModel_tag_sets(tag_set):

    param_dict = {
        "adaptiveThreshWinSizeMin": 10,
        "adaptiveThreshWinSizeMax": 10,
        "adaptiveThreshWinSizeStep": 10,
        "adaptiveThreshConstant": 10,
        "perspectiveRemoveIgnoredMarginPerCell": 0.1,
        "errorCorrectionRate": 0.1,
    }

    # Confirm the model loads without error for each tag set
    try:
        test_model = ArUcoModel.withTagSet(tag_set, **param_dict)
    except Exception as exc:
        assert False, f"Tag set {tag_set} raised an exception {exc}"


def test_ArUcoModel_tag_set_dict_error():

    param_dict = {
        "adaptiveThreshWinSizeMin": 10,
        "adaptiveThreshWinSizeMax": 10,
        "adaptiveThreshWinSizeStep": 10,
        "adaptiveThreshConstant": 10,
        "perspectiveRemoveIgnoredMarginPerCell": 0.1,
        "errorCorrectionRate": 0.1,
    }

    with pytest.raises(Exception) as e_info:
        test_model = ArUcoModel.withTagSet("DICT_4X4_FAIL", **param_dict)


def test_ArUcoModel_no_tag_set_error():

    param_dict = {
        "adaptiveThreshWinSizeMin": 10,
        "adaptiveThreshWinSizeMax": 10,
        "adaptiveThreshWinSizeStep": 10,
        "adaptiveThreshConstant": 10,
        "perspectiveRemoveIgnoredMarginPerCell": 0.1,
        "errorCorrectionRate": 0.1,
    }

    with pytest.raises(Exception) as e_info:
        test_model = ArUcoModel.withTagSet(**param_dict)


@pytest.mark.parametrize(
    "param, value",
    [
        ("adaptiveThreshWinSizeMin", 0.1),
        ("adaptiveThreshWinSizeMax", 0.1),
        ("adaptiveThreshWinSizeStep", 0.1),
        ("adaptiveThreshConstant", "10"),
        ("perspectiveRemoveIgnoredMarginPerCell", "10"),
        ("errorCorrectionRate", "10"),
    ],
)
def test_ArUcoModel_params_type_error(param, value):

    param_dict = {
        "adaptiveThreshWinSizeMin": 10,
        "adaptiveThreshWinSizeMax": 10,
        "adaptiveThreshWinSizeStep": 10,
        "adaptiveThreshConstant": 10,
        "perspectiveRemoveIgnoredMarginPerCell": 0.1,
        "errorCorrectionRate": 0.1,
    }

    # Replace value
    param_dict[param] = value

    with pytest.raises(Exception) as e_info:
        test_model = ArUcoModel.withTagSet("DICT_4X4_100", **param_dict)


@pytest.mark.parametrize(
    "coords, tag",
    [
        ([100, 350, 100, 350], 1),
        ([100, 350, 800, 1050], 2),
        ([450, 700, 450, 700], 3),
        ([800, 1050, 100, 350], 4),
        ([800, 1050, 800, 1050], 5),
    ],
)
def test_ArUcoModel_detect(coords, tag):

    param_dict = {
        "adaptiveThreshWinSizeMin": 3,
        "adaptiveThreshWinSizeMax": 10,
        "adaptiveThreshWinSizeStep": 3,
        "adaptiveThreshConstant": 10,
        "perspectiveRemoveIgnoredMarginPerCell": 0.1,
        "errorCorrectionRate": 0.1,
    }

    # Confirm the model loads without error
    try:
        test_model = ArUcoModel.withTagSet("DICT_4X4_100", **param_dict)
    except Exception as exc:
        assert False, f"Tag set DICT_4X4_100 raised an exception {exc}"

    # Open the aruco image
    aruco_image = cv2.imread("tests/data/example_ArUco_image.jpg", 0)
    tag_image = aruco_image[
        coords[0] - 100 : coords[1] + 100, coords[2] - 100 : coords[3] + 100
    ]

    # Detect ArUco tags
    tags = test_model.detect(tag_image)
    assert tags[0] == tag


@pytest.mark.parametrize(
    "coords, tag",
    [
        ([100, 350, 100, 350], 1),
        ([100, 350, 800, 1050], 2),
        ([450, 700, 450, 700], 3),
        ([800, 1050, 100, 350], 4),
        ([800, 1050, 800, 1050], 5),
    ],
)
def test_ArUcoModel_subset_detect(coords, tag):

    param_dict = {
        "adaptiveThreshWinSizeMin": 3,
        "adaptiveThreshWinSizeMax": 10,
        "adaptiveThreshWinSizeStep": 3,
        "adaptiveThreshConstant": 10,
        "perspectiveRemoveIgnoredMarginPerCell": 0.1,
        "errorCorrectionRate": 0.1,
    }

    # Confirm the model loads without error
    try:
        test_model = ArUcoModel.withTagSet("DICT_4X4_100", tag_subset_list = [5, 4, 3, 2, 1], **param_dict)
    except Exception as exc:
        assert False, f"Tag set DICT_4X4_100 raised an exception {exc}"

    # Open the aruco image
    aruco_image = cv2.imread("tests/data/example_ArUco_image.jpg", 0)
    tag_image = aruco_image[
        coords[0] - 100 : coords[1] + 100, coords[2] - 100 : coords[3] + 100
    ]

    # Detect ArUco tags
    tags = test_model.detect(tag_image)
    assert tags[0] == tag
