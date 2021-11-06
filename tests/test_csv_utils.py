#!/usr/bin/python3

import collections
import os
from pathlib import Path

from mseg.utils.csv_utils import read_csv, write_csv

_TEST_DIR = Path(__file__).resolve().parent


def test_read_csv() -> None:
    """
    Given csv data with the following form
    -------------------------
            WorkerId,image_url_1
            1,cat.png
            2,dog.png
    -------------------------
    Ensure we can read it out correctly as OrderedDicts using DictReader.
    """
    d1 = collections.OrderedDict()
    d1["WorkerId"] = "1"
    d1["image_url_1"] = "cat.png"

    d2 = collections.OrderedDict()
    d2["WorkerId"] = "2"
    d2["image_url_1"] = "dog.png"
    dict_list = [d1, d2]

    csv_fpath = f"{_TEST_DIR}/test_data/dummy_csv_data_to_read.csv"
    rows = read_csv(csv_fpath)
    assert rows == dict_list


def test_write_csv() -> None:
    """ """
    dict_list = [
        {"WorkerId": "1", "image_url_1": "cat.png", "image_url_2": "dog.png"},
        {"WorkerId": "2", "image_url_1": "elephant.png", "image_url_2": "house.png"},
    ]
    csv_fpath = f"{_TEST_DIR}/test_data/temp_written_data.csv"
    write_csv(csv_fpath, dict_list)
    rows = read_csv(csv_fpath)

    d1 = collections.OrderedDict()
    d1["WorkerId"] = "1"
    d1["image_url_1"] = "cat.png"
    d1["image_url_2"] = "dog.png"

    d2 = collections.OrderedDict()
    d2["WorkerId"] = "2"
    d2["image_url_1"] = "elephant.png"
    d2["image_url_2"] = "house.png"
    gt_dict_list = [d1, d2]
    assert gt_dict_list == rows

    os.remove(csv_fpath)


if __name__ == "__main__":
    test_write_csv()
