#!/usr/bin/python3

import collections
from pathlib import Path
from typing import NamedTuple

from mseg.utils.txt_utils import read_txt_file

_ROOT = Path(__file__).resolve().parent.parent


class LabelImageUpdateRecord(NamedTuple):
    """Data record for a single mask in a single image.

    Stores necessary information to remap a single segment of a single image.
    """

    dataset_name: str
    img_fname: str
    segmentid: int
    split: str
    orig_class: str
    relabeled_class: str


class DatasetClassUpdateRecord:
    """
    Stores necessary information to remap a list of (segment, image) objects from
    an old class to a new class.
    """

    def __init__(self, dataset_name: str, split: str, orig_class: str, relabeled_class: str, txt_fpath: str):
        """
        Args:
            dataset_name
            split
            orig_class
            relabeled_class
            txt_fpath
        """
        self.dataset_name = dataset_name
        self.split = split
        self.orig_class = orig_class
        self.relabeled_class = relabeled_class
        self.txt_fpath = txt_fpath
        full_txt_fpath = f"{_ROOT}/relabeled_data/{self.txt_fpath}"
        if not Path(full_txt_fpath).exists():
            print(f"Not found: {full_txt_fpath}. Quitting...")
            quit()
        self.img_list = read_txt_file(full_txt_fpath, strip_newlines=True)
        num_imgs = len(self.img_list)
        if False:
            print(
                f"\tFound {num_imgs} segments to update in {dataset_name}-{split} from {orig_class}->{relabeled_class}"
            )
