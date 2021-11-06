#!/usr/bin/python3

import numpy as np
from pathlib import Path

from mseg.utils.names_utils import (
    read_str_list,
    load_class_names,
    load_dataset_colors_arr,
)

_TEST_DIR = Path(__file__).resolve().parent


def test_read_str_list() -> None:
    """Ensure we can read the 3 lines in a dummy txt file."""
    fpath = f"{_TEST_DIR}/test_data/str_test1.txt"
    str_list = read_str_list(fpath)
    gt_str_list = ["abc", "def", "ghi"]
    assert str_list == gt_str_list


def test_read_str_list_extrablankline() -> None:
    """Ensure we read only the non-blank lines, if txt file has empty line at end."""
    fpath = f"{_TEST_DIR}/test_data/str_test2.txt"
    str_list = read_str_list(fpath)
    gt_str_list = ["abc", "def", "ghi"]
    assert str_list == gt_str_list


def test_load_class_names() -> None:
    """Ensure we can read in the 11 class names from the Camvid-11 `names.txt` file."""
    dataset_name = "camvid-11"
    class_names = load_class_names(dataset_name)
    gt_class_names = [
        "Building",
        "Tree",
        "Sky",
        "Car",
        "SignSymbol",
        "Road",
        "Pedestrian",
        "Fence",
        "Column_Pole",
        "Sidewalk",
        "Bicyclist",
    ]
    assert gt_class_names == class_names


def test_load_dataset_colors_arr() -> None:
    """ """
    dataset_name = "camvid-32"
    colors_arr = load_dataset_colors_arr(dataset_name)
    assert np.allclose(colors_arr[0], np.array([64, 128, 64]))
    assert np.allclose(colors_arr[-1], np.array([64, 192, 0]))


# def test_get_classname_to_dataloaderid_map():
# 	""" """

# 	get_classname_to_dataloaderid_map(
# 		dataset_name: str,
# 		include_ignore_idx_cls: bool = False,
# 		ignore_index: int = 255
# 		) -> Mapping[str,int]


# def test_get_dataloader_id_to_classname_map():
# 	""" """

# 	get_dataloader_id_to_classname_map(
# 		dataset_name: str,
# 		class_names: List[str] = None,
# 		include_ignore_idx_cls: bool = True,
# 		ignore_index: int = 255) -> Mapping[int,str]:
