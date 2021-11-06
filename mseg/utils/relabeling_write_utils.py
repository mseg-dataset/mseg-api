#!/usr/bin/python3

from pathlib import Path
from typing import List


def read_txt_file(txt_fpath: str) -> List[str]:
    """
    Args:
        txt_fpath: path to a .txt file.

    Returns:
        txt_lines: stripped lines of the file.
    """
    with open(txt_fpath, "r") as f:
        txt_lines = f.readlines()

    txt_lines = [line.strip() for line in txt_lines]
    return txt_lines
