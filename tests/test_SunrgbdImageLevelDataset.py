#!/usr/bin/env python3

from pathlib import Path
import pdb
from mseg.dataset_apis.SunrgbdImageLevelDataset import SunrgbdImageLevelDataset

_TEST_DIR = Path(__file__).resolve().parent


def test_constructor() -> None:
    """ """
    dataroot = f"{_TEST_DIR}/test_data/SUNRGBD_test_data"
    bddild = SunrgbdImageLevelDataset(dataroot)


def test_get_img_pair() -> None:
    """ """
    dataroot = f"{_TEST_DIR}/test_data/SUNRGBD_test_data"
    bddild = SunrgbdImageLevelDataset(dataroot)

    split = "train"
    fname_stem = "img-000001"
    rgb_img, label_img = bddild.get_img_pair(fname_stem, split)
    assert rgb_img.mean() - 134.806 < 1e-3
    assert label_img.mean() - 16.788 < 1e-3

    split = "test"
    fname_stem = "img-000001"
    rgb_img, label_img = bddild.get_img_pair(fname_stem, split)
    assert rgb_img.mean() - 125.300 < 1e-3
    assert label_img.mean() - 47.588 < 1e-3


def test_get_segment_mask() -> None:
    """ """
    dataroot = f"{_TEST_DIR}/test_data/SUNRGBD_test_data"
    bddild = SunrgbdImageLevelDataset(dataroot)
    seq_id = ""
    query_segmentid = 21  # ceiling
    fname_stem = "img-000001"
    split = "test"

    class_mask = bddild.get_segment_mask(seq_id, query_segmentid, fname_stem, split)
    assert class_mask.sum() == 37819
    assert class_mask.mean() - 0.098 < 1e-3
    assert class_mask.size == 386900


if __name__ == "__main__":
    # pass
    # test_constructor()
    # test_get_img_pair()
    test_get_segment_mask()
