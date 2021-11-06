#!/usr/bin/python3

import numpy as np
from pathlib import Path
from typing import List, Mapping

from mseg.utils.tsv_utils import read_tsv_column_vals


ROOT = Path(__file__).resolve().parent.parent


def read_str_list(fpath: str) -> List[str]:
    """Obtain carriage-return separated lines of a file as a list of strings."""
    return list(np.genfromtxt(fpath, delimiter="\n", dtype=str))


def load_class_names(dataset_name: str) -> List[str]:
    """
    Args:
        dataset_name: str

    Returns:
        list of strings, representing class names
    """
    return read_str_list(f"{ROOT}/dataset_lists/{dataset_name}/{dataset_name}_names.txt")


def get_universal_class_names() -> List[str]:
    """Get the ordered classnames included in MSeg's universal taxonomy.

    Returns:
        u_names: universal class names
    """
    tsv_fpath = f"{ROOT}/class_remapping_files/MSeg_master.tsv"
    u_names = read_tsv_column_vals(tsv_fpath, col_name="universal", convert_val_to_int=False)
    return u_names[:-1]  # exclude unlabeled (last class)


def load_dataset_colors_arr(dataset_name: str) -> np.ndarray:
    """Read lines of a file, where each line is a tab-separated 3-tuple, as uint8 array

    Returns:
        Numpy array of shape (N,3) with RGB tuple corresponding to each class.
    """
    return np.loadtxt(f"{ROOT}/dataset_lists/{dataset_name}/{dataset_name}_colors.txt").astype("uint8")


def get_classname_to_dataloaderid_map(
    dataset_name: str, include_ignore_idx_cls: bool = False, ignore_index: int = 255
) -> Mapping[str, int]:
    """Obtain a 1:1 mapping from a class name to corresponding class index.

    Returns:
        classname_to_dataloaderid_map:
    """
    class_names = load_class_names(dataset_name)
    classname_to_dataloaderid_map = {classname: dataloader_id for dataloader_id, classname in enumerate(class_names)}

    if include_ignore_idx_cls:
        classname_to_dataloaderid_map["unlabeled"] = ignore_index
    return classname_to_dataloaderid_map


def get_dataloader_id_to_classname_map(
    dataset_name: str, class_names: List[str] = None, include_ignore_idx_cls: bool = True, ignore_index: int = 255
) -> Mapping[int, str]:
    """Get the 1:1 mapping stored in our `names.txt` file that maps a class name to a
            data loader class index.

    Returns:
        dataloader_id_to_classname_map: dictionary mapping integers to strings
    """
    if class_names is None:
        class_names = load_class_names(dataset_name)

    dataloader_id_to_classname_map = {dataloader_id: classname for dataloader_id, classname in enumerate(class_names)}

    if include_ignore_idx_cls:
        dataloader_id_to_classname_map[ignore_index] = "unlabeled"
    return dataloader_id_to_classname_map
