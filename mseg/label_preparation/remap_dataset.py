#!/usr/bin/python3

import argparse
import imageio
import numpy as np
from pathlib import Path
import pdb
from shutil import copyfile
from typing import Any, List, Mapping, Optional, Tuple

from mseg.utils.cv2_utils import cv2_imread_rgb
from mseg.utils.multiprocessing_utils import send_list_to_workers
from mseg.utils.dictionary_utils import convert_dictionaries
from mseg.utils.names_utils import (
    get_classname_to_dataloaderid_map,
    get_dataloader_id_to_classname_map,
    load_dataset_colors_arr,
)
from mseg.utils.mask_utils import form_label_mapping_array, map_semantic_img_fast, rgb_img_to_obj_cls_img
from mseg.utils.tsv_utils import read_label_mapping
from mseg.utils.txt_utils import generate_all_img_label_pair_relative_fpaths
from mseg.utils.dir_utils import create_leading_fpath_dirs


_ROOT = Path(__file__).resolve().parent.parent


def remap_dataset(
    dname: str,
    remapped_dname: str,
    tsv_fpath: str,
    old_dataroot: str,
    remapped_dataroot: str,
    include_ignore_idx_cls: bool = True,
    convert_label_from_rgb: bool = False,
    num_processes: int = 4,
) -> None:
    """
    Given path to a dataset, given names of _names.txt
    Remap according to the provided tsv.
    (also account for the fact that 255 is always unlabeled)

    Args:
        dname: string representing name of taxonomy for original dataset
        remapped_dname: string representing name of taxonomy for new dataset
        tsv_fpath: string representing path to a .tsv file
        old_dataroot: string representing path to original dataset
        remapped_dataroot: string representing path at which to new dataset
        include_ignore_idx_cls: whether to include unlabeled=255 from source
        convert_label_from_rgb: labels of original dataset are stored as RGB
        num_processes: integer representing number of workers to exploit
    """
    # load colors ordered with class indices, if labels encoded as RGB
    dataset_colors = load_dataset_colors_arr(dname) if convert_label_from_rgb else None

    # load up the dictionary from the tsv
    classname_remapping_dict = read_label_mapping(
        filename=tsv_fpath, label_from=dname, label_to=remapped_dname, convert_val_to_int=False
    )
    oldid_to_oldname = get_dataloader_id_to_classname_map(dname)
    newname_tonewid_map = get_classname_to_dataloaderid_map(
        remapped_dname, include_ignore_idx_cls=include_ignore_idx_cls
    )
    # form one-way mapping between IDs
    old_name_to_newid = convert_dictionaries(classname_remapping_dict, newname_tonewid_map)
    class_idx_remapping_dict = convert_dictionaries(oldid_to_oldname, old_name_to_newid)
    label_mapping_arr = form_label_mapping_array(class_idx_remapping_dict)

    for split in ["train", "val"]:  #'trainval']:# 'val']: #
        orig_relative_img_label_pairs = generate_all_img_label_pair_relative_fpaths(dname, split)
        remapped_relative_img_label_pairs = generate_all_img_label_pair_relative_fpaths(remapped_dname, split)

        send_list_to_workers(
            num_processes=num_processes,
            list_to_split=orig_relative_img_label_pairs,
            worker_func_ptr=relabel_pair_worker,
            remapped_relative_img_label_pairs=remapped_relative_img_label_pairs,
            label_mapping_arr=label_mapping_arr,
            old_dataroot=old_dataroot,
            new_dataroot=remapped_dataroot,
            dataset_colors=dataset_colors,
        )

        # relabel_pair(
        # 	old_dataroot,
        # 	remapped_dataroot,
        # 	orig_relative_img_label_pairs[0],
        # 	remapped_relative_img_label_pairs[0],
        # 	label_mapping_arr,
        # 	dataset_colors)


def relabel_pair_worker(
    orig_pairs: List[Tuple[str, str]], start_idx: int, end_idx: int, kwargs: Mapping[str, Any]
) -> None:
    """Given a list of (rgb image, label image) pairs to remap, call relabel_pair()
    on each one of them.

    Args:
        orig_pairs: list of strings
        start_idx: integer
        end_idx: integer
        kwargs: dictionary with argument names mapped to argument values.
    """
    label_mapping_arr = kwargs["label_mapping_arr"]
    old_dataroot = kwargs["old_dataroot"]
    new_dataroot = kwargs["new_dataroot"]
    dataset_colors = kwargs["dataset_colors"]
    remapped_pairs = kwargs["remapped_relative_img_label_pairs"]

    chunk_sz = end_idx - start_idx
    # process each image between start_idx and end_idx
    for idx in range(start_idx, end_idx):
        if idx % 1000 == 0:
            pct_completed = (idx - start_idx) / chunk_sz * 100
            print(f"Completed {pct_completed:.2f}%")
        orig_pair = orig_pairs[idx]
        remapped_pair = remapped_pairs[idx]
        relabel_pair(old_dataroot, new_dataroot, orig_pair, remapped_pair, label_mapping_arr, dataset_colors)


def relabel_pair(
    old_dataroot: str,
    new_dataroot: str,
    orig_pair: Tuple[str, str],
    remapped_pair: Tuple[str, str],
    label_mapping_arr: np.ndarray,
    dataset_colors: Optional[np.ndarray] = None,
):
    """
    No need to copy the RGB files again. We just update the label file paths.

    Args:
        old_dataroot:
        new_dataroot:
        orig_pair: Tuple containing relative path to RGB image and label image
        remapped_pair: Tuple containing relative path to RGB image and label image
        label_mapping_arr:
        dataset_colors:
    """
    _, orig_rel_label_fpath = orig_pair
    _, remapped_rel_label_fpath = remapped_pair

    old_label_fpath = f"{old_dataroot}/{orig_rel_label_fpath}"

    if dataset_colors is None:
        label_img = imageio.imread(old_label_fpath)
    else:
        # remap from RGB encoded labels to 1-channel class indices
        label_img_rgb = cv2_imread_rgb(old_label_fpath)
        label_img = rgb_img_to_obj_cls_img(label_img_rgb, dataset_colors)
    remapped_img = map_semantic_img_fast(label_img, label_mapping_arr)

    new_label_fpath = f"{new_dataroot}/{remapped_rel_label_fpath}"
    create_leading_fpath_dirs(new_label_fpath)
    imageio.imwrite(new_label_fpath, remapped_img)


if __name__ == "__main__":
    """
    For PASCAL-Context-460, there is an explicit unlabeled class
    (class 0) so we don't include unlabeled=255 from source.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_dname", type=str, required=True, help="original dataset's name")
    parser.add_argument("--remapped_dname", type=str, required=True, help="name for remapped dataset")
    parser.add_argument("--orig_dataroot", type=str, required=True, help="path to original data root")
    parser.add_argument(
        "--remapped_dataroot", type=str, required=True, help="data root where remapped dataset will be saved"
    )
    parser.add_argument(
        "--include_ignore_idx_cls", action="store_false", help="data root where remapped dataset will be saved"
    )
    parser.add_argument(
        "--convert_label_from_rgb", action="store_true", help="If original dataset labels are stored as RGB images."
    )
    parser.add_argument(
        "--num_processes", type=int, default=4, help="Number of cores available on machine (more->faster remapping)"
    )

    args = parser.parse_args()

    # prepare the 'tsv_fpath' according to names.
    tsv_fpath = _ROOT / "class_remapping_files" / f"{args.orig_dname}_to_{args.remapped_dname}.tsv"

    print("Remapping Parameters: ", args)
    remap_dataset(
        args.orig_dname,
        args.remapped_dname,
        tsv_fpath,
        args.orig_dataroot,
        args.remapped_dataroot,
        args.include_ignore_idx_cls,
        args.convert_label_from_rgb,
        args.num_processes,
    )
