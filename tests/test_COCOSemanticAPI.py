#!/usr/bin/python3

from pathlib import Path

from mseg.dataset_apis.COCOSemanticAPI import COCOSemanticAPI

_TEST_DIR = Path(__file__).resolve().parent


def test_constructor() -> None:
    """ """
    coco_dataroot = f"{_TEST_DIR}/test_data/COCOPanoptic_test_data"
    c_api = COCOSemanticAPI(coco_dataroot)
    assert (
        len(c_api.categoryid_to_classname_map) == 202
    )  # 201 classes + 'unlabeled' idx=255

    assert list(c_api.fname_to_annot_map_splitdict["train"].keys()) == [
        "000000000009.png"
    ]
    assert list(c_api.fname_to_annot_map_splitdict["val"].keys()) == [
        "000000000139.png"
    ]


def test_get_img_annotation() -> None:
    """ """
    coco_dataroot = f"{_TEST_DIR}/test_data/COCOPanoptic_test_data"
    c_api = COCOSemanticAPI(coco_dataroot)

    fname_stem = "000000000009"
    split = "train"
    annot = c_api.get_img_annotation(split, fname_stem)
    assert annot["file_name"] == "000000000009.png"

    present_classes = c_api.get_present_classes_in_img(split, fname_stem)
    gt_present_classes = [
        "bowl",
        "bowl",
        "bowl",
        "orange",
        "orange",
        "orange",
        "orange",
        "broccoli",
        "table-merged",
        "food-other-merged",
    ]
    assert present_classes == gt_present_classes


if __name__ == "__main__":
    # test_constructor()
    test_get_img_annotation()
