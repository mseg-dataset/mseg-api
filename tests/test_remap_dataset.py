#!/usr/bin/python3

import imageio
import numpy as np
import os
from pathlib import Path

from mseg.label_preparation.remap_dataset import relabel_pair
from mseg.utils.mask_utils import map_semantic_img_fast, form_label_mapping_array
from mseg.utils.dir_utils import create_leading_fpath_dirs

_TEST_DIR = Path(__file__).resolve().parent


def test_relabel_pair() -> None:
    """
    grayscale -> grayscale label remapping.
    """
    old_dataroot = f"{_TEST_DIR}/test_data"
    new_dataroot = f"{_TEST_DIR}/test_data"
    orig_pair = ("rgb_old.jpg", "remapping_test_data_old/label_old.png")
    remapped_pair = ("rgb_new.jpg", "remapping_test_data_new/label_new.png")
    old_label_fpath = f"{old_dataroot}/{orig_pair[1]}"
    create_leading_fpath_dirs(old_label_fpath)
    new_label_fpath = f"{new_dataroot}/{remapped_pair[1]}"

    # fmt: off
    semantic_img = np.array(
        [
            [254, 0, 1],
            [7, 8, 9]
        ], dtype=np.uint8
    )
    # fmt: on
    imageio.imwrite(old_label_fpath, semantic_img)
    # fmt: off
    label_mapping = {
        254: 253,
        0: 255,
        1: 0,
        7: 6,
        8: 7,
        9: 8
    }
    # fmt: on
    label_mapping_arr = form_label_mapping_array(label_mapping)
    relabel_pair(
        old_dataroot,
        new_dataroot,
        orig_pair,
        remapped_pair,
        label_mapping_arr,
        dataset_colors=None,
    )

    # fmt: off
    gt_mapped_img = np.array(
        [
            [253, 255, 0],
            [6, 7, 8]
        ], dtype=np.uint8
    )
    # fmt: on
    remapped_img = imageio.imread(new_label_fpath)
    assert np.allclose(gt_mapped_img, remapped_img)
    os.remove(old_label_fpath)
    os.remove(new_label_fpath)


"""
TODO: add test when remapping from RGB->grayscale
"""


if __name__ == "__main__":
    test_relabel_pair()
