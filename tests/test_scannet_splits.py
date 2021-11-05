#!/usr/bin/python3

from collections import defaultdict
from pathlib import Path
import pdb

from mseg.utils.txt_utils import (
    read_txt_file,
    generate_all_img_label_pair_relative_fpaths,
)

"""
Ensures that train/val/test splits are correct.
"""

# Make the paths absolute, resolving any symlinks.
_TEST_DIR = Path(__file__).resolve().parent


def get_scene_splits():
    """
    Returns:
        split_dict
        num_imgs_per_split_dict
        num_scans_per_split_dict
    """
    num_scans_per_split_dict = defaultdict(int)

    split_dict = {}
    for split in ["train", "val", "test"]:
        split_txt_fpath = (
            f"{_TEST_DIR}/test_data/ScanNet_data_splits/scannetv2_{split}.txt"
        )
        scan_lines = read_txt_file(split_txt_fpath)
        num_scans_per_split_dict[split] += len(scan_lines)
        for scan_line in scan_lines:
            scan_id = scan_line.strip()
            split_dict.setdefault(split, []).append(scan_id)

    return split_dict, num_scans_per_split_dict


def test_datalists() -> None:
    """ """
    split_dict, num_scans_per_split_dict = get_scene_splits()
    assert num_scans_per_split_dict["test"] == 100
    train_scenes = num_scans_per_split_dict["train"]
    val_scenes = num_scans_per_split_dict["val"]
    assert train_scenes + val_scenes == 1513

    for dname in ["scannet-20", "scannet-41"]:
        for split in ["train", "val", "test"]:  #'trainval']:# 'val']: #
            relative_img_label_pairs = generate_all_img_label_pair_relative_fpaths(
                dname, split
            )
            for (rgb_fpath, label_fpath) in relative_img_label_pairs:
                rgb_scene_id = Path(rgb_fpath).parts[-3]
                rgb_fname_stem = Path(rgb_fpath).stem
                label_scene_id = Path(label_fpath).parts[-3]
                label_fname_stem = Path(label_fpath).stem

                assert rgb_scene_id == label_scene_id
                assert rgb_fname_stem == label_fname_stem
                assert rgb_scene_id in split_dict[split]


if __name__ == "__main__":
    test_datalists()
