#!/usr/bin/python3

import os
from pathlib import Path


def check_mkdir(dirpath: str) -> None:
    """ """
    if not Path(dirpath).exists():
        os.makedirs(dirpath, exist_ok=True)


def create_leading_fpath_dirs(save_fpath: str, return_dir: bool = False):
    """ """
    save_dir = "/".join(save_fpath.split("/")[:-1])
    check_mkdir(save_dir)

    if return_dir:
        return save_dir


def create_leading_fpath_dirs_exist_ok(save_fpath: str, return_dir: bool = False):
    """ """
    save_dir = "/".join(save_fpath.split("/")[:-1])
    os.makedirs(save_dir, exist_ok=True)

    if return_dir:
        return save_dir


def get_rel_path_len(data_root: str, fpath: str) -> int:
    """Determine the length of a relative path (depth in file system tree).

    Args:
        data_root
        fpath

    Returns:
        rel_path_len: integer
    """
    data_root_len = len(Path(data_root).parts)
    path_len = len(Path(fpath).parts)
    rel_path_len = path_len - data_root_len
    return rel_path_len
