#!/usr/bin/python3

from pathlib import Path

import numpy as np

from mseg.dataset_apis.COCOInstanceAPI import COCOInstanceAPI


_TEST_DIR = Path(__file__).parent


def test_constructor() -> None:
    """ """
    coco_dataroot = f"{_TEST_DIR}/test_data/COCOPanoptic_test_data"
    c_api = COCOInstanceAPI(coco_dataroot)

    assert len(c_api.instance_img_fpaths_splitdict["train"]) == 1
    assert len(c_api.instance_img_fpaths_splitdict["val"]) == 1
    assert (
        Path(c_api.instance_img_fpaths_splitdict["train"][0]).name == "000000000009.png"
    )
    assert (
        Path(c_api.instance_img_fpaths_splitdict["val"][0]).name == "000000000139.png"
    )

    assert len(c_api.fname_to_instanceimgfpath_dict.keys()) == 2
    assert (
        Path(c_api.fname_to_instanceimgfpath_dict["000000000009"]).name
        == "000000000009.png"
    )
    assert (
        Path(c_api.fname_to_instanceimgfpath_dict["000000000139"]).name
        == "000000000139.png"
    )


def test_get_instance_img_fpaths() -> None:
    """ """
    coco_dataroot = f"{_TEST_DIR}/test_data/COCOPanoptic_test_data"
    c_api = COCOInstanceAPI(coco_dataroot)
    split = "val"
    fpaths = c_api.get_instance_img_fpaths(split)
    assert len(fpaths) == 1
    assert Path(fpaths[0]).name == "000000000139.png"


def test_get_instance_id_img() -> None:
    """ """
    coco_dataroot = f"{_TEST_DIR}/test_data/COCOPanoptic_test_data"
    c_api = COCOInstanceAPI(coco_dataroot)
    split = "train"
    fname_stem = "000000000009"
    instance_id_img = c_api.get_instance_id_img(split, fname_stem)

    assert np.amax(instance_id_img) == 8922372
    assert np.amin(instance_id_img) == 0
    assert np.sum(instance_id_img) == 1451563332418


if __name__ == "__main__":
    # test_constructor()
    # test_get_instance_img_fpaths()
    test_get_instance_id_img()
