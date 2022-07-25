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

    # Confirm the model loads without error for each tag set
    try:
        ArUcoModel.withTagSet(tag_set)
    except Exception as exc:
        assert False, f"Tag set {tag_set} raised an exception {exc}"


@pytest.mark.parametrize(
    "param, value",
    [
        ("adaptiveThreshWinSizeMin", 10),
        ("adaptiveThreshWinSizeMax", 10),
        ("adaptiveThreshWinSizeStep", 10),
        ("adaptiveThreshConstant", 10),
        ("perspectiveRemoveIgnoredMarginPerCell", 0.1),
        ("errorCorrectionRate", 0.1),
    ],
)
def test_ArUcoModel_params(param, value):

    # Confirm the model loads without error for each parameter and value
    try:
        ArUcoModel.withTagSet("DICT_4X4_100", **{param: value})
    except Exception as exc:
        assert False, f"Parameter{param} with value {value} raised an exception {exc}"


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
    with pytest.raises(Exception) as e_info:
        ArUcoModel.withTagSet("DICT_4X4_100", **{param: value})
