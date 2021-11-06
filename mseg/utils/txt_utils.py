#!/usr/bin/python3

import numpy as np
from pathlib import Path
from typing import List, Tuple

_ROOT = Path(__file__).resolve().parent.parent


def read_txt_file(txt_fpath: str, strip_newlines: bool = False) -> List[str]:
    """
    Args:
    -	txt_fpath: string representing path to txt file

    Returns:
    -	txt_lines: list of strings, one per line of file
    """
    with open(txt_fpath, "r") as f:
        txt_lines = f.readlines()

    if strip_newlines:
        txt_lines = [line.strip() for line in txt_lines]
    return txt_lines


def get_last_n_path_elements_as_str(fpath: str, n: int) -> str:
    """
    Args:
    -	fpath: string representing file path
    -	n: integer representing last number of filepath elements to keep

    Returns:
    -
    """
    elements = fpath.split("/")[-n:]
    return "/".join(elements)


def write_txt_lines(save_fpath: str, txt_lines: List[str]) -> None:
    """Note that this function will add a carriage return, so please be
    mindful of this. Freshly loaded lines from a file will have carriage
    returns, by default.

    Args:
    -	save_fpath: string representing file path, where file should be saved
    -	txt_lines:

    Returns:
    -	None
    """
    with open(save_fpath, "w") as f:
        for txt_line in txt_lines:
            f.write(f"{txt_line}\n")


def read_rgb_and_label_tuple_file(fpath: str) -> Tuple[List[str], List[str]]:
    """
    Args:
    -	fpath

    Returns:
    -	rgb_img_fpaths
    -	label_fpaths
    """
    tuples = list(np.genfromtxt(fpath, delimiter="\n", dtype=str))
    tuples = [line.split(" ") for line in tuples]
    rgb_img_fpaths = [elements[0] for elements in tuples]
    label_fpaths = [elements[1] for elements in tuples]

    return rgb_img_fpaths, label_fpaths


def subsample_txt_lines(txt_fpath: str, save_fpath: str, subsample_nth: str) -> None:
    """
    Args:
    -	txt_fpath:
    -	save_fpath:
    -	subsample_nth:

    Returns:
    -	None
    """
    txt_lines = read_txt_file(txt_fpath)
    txt_lines = [line.strip() for line in txt_lines]
    write_txt_lines(save_fpath, txt_lines[::subsample_nth])


def generate_all_img_label_pair_fpaths(data_root: str, split_txt_fpath: str):
    """ """
    pairs = []
    rgb_img_fpaths, label_fpaths = read_rgb_and_label_tuple_file(split_txt_fpath)
    for rel_rgb_fpath, rel_label_fpath in zip(rgb_img_fpaths, label_fpaths):
        img_fpath = f"{data_root}/{rel_rgb_fpath}"
        label_fpath = f"{data_root}/{rel_label_fpath}"
        pairs += [(img_fpath, label_fpath)]
    return pairs


def generate_all_img_label_pair_relative_fpaths(dname: str, split: str):
    """
    Args:
        dname:
        split: e.g. 'train', 'val', 'trainval', etc.

    Returns:
    -
    """
    split_txt_fpath = _ROOT / f"dataset_lists/{dname}/list/{split}.txt"
    pairs = []
    rgb_img_fpaths, label_fpaths = read_rgb_and_label_tuple_file(split_txt_fpath)
    return list(zip(rgb_img_fpaths, label_fpaths))
