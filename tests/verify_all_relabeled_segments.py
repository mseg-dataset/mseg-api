#!/usr/bin/python3

import argparse
import collections
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Mapping, NamedTuple, Tuple

import imageio
import numpy as np

from mseg.dataset_apis.Ade20kMaskLevelDataset import Ade20kMaskDataset
from mseg.dataset_apis.BDDImageLevelDataset import BDDImageLevelDataset
from mseg.dataset_apis.COCOPanopticJsonMaskDataset import COCOPanopticJsonMaskDataset
from mseg.dataset_apis.JsonMaskLevelDataset import JsonMaskDataset
from mseg.dataset_apis.MapillaryMaskDataset import MapillaryMaskDataset
from mseg.dataset_apis.SunrgbdImageLevelDataset import SunrgbdImageLevelDataset

from mseg.utils.multiprocessing_utils import send_list_to_workers
from mseg.utils.txt_utils import generate_all_img_label_pair_fpaths
from mseg.utils.names_utils import get_classname_to_dataloaderid_map

from mseg.label_preparation.dataset_update_records import (
    cocop_update_records,
    ade20k_update_records,
    bdd_update_records,
    idd_update_records,
    cityscapes_update_records,
    sunrgbd_update_records,
    mapillary_update_records,
)
from mseg.utils.dataset_config import infos
from mseg.label_preparation.mseg_write_relabeled_segments import (
    get_unique_mask_identifiers,
    form_fname_to_updatelist_dict,
)

"""
Simple utilities to write a remapped version of the dataset in a universal
taxonomy, then overwrite the masks in place. Each mask is defined by a
tuple of three IDs -- (sequence_ID, image_ID, segment_ID).

We subdivide multiprocessing over the image level, rather than mask level,
to prevent race conditions.

TODO: DILATE IDD MASKS BY 1 PIXEL TO FIX WEIRD BOUNDARY DIFFERENT CLASS
OR MAP TO UNLABELED FIRST FOR ALL OF RIDER...
"""

_ROOT = Path(__file__).resolve().parent.parent


def verify_all_relabeled_dataset_segments(
    num_processes: int,
    mld: Any,
    dname: str,
    dataroot: str,
    update_records,
    require_strict_boundaries: bool,
):
    """
    By using remap.py, we already have converted label img from original taxonomy, to universal taxonomy.

    Args:
        num_processes: number of processes to launch; shouldn't exceed number of cores on machine
        mld: Mask Level Dataset
        dname: string representing name of a dataset taxonomy
        dataroot: string representing path to a file directory
        update_records
    """
    classname_to_id_map = get_classname_to_dataloaderid_map(
        dname, include_ignore_idx_cls=True
    )
    # Check for incorrect class names.
    for rec in update_records:
        valid_orig = rec.orig_class in classname_to_id_map.keys()
        valid_relabeled = rec.relabeled_class in classname_to_id_map.keys()
        if not (valid_orig and valid_relabeled):
            print(rec.__dict__)
            print(
                f"Invalid universal classname: {rec.orig_class}, {rec.relabeled_class}"
            )
            quit()

    for split in ["train", "val"]:
        # Create for each split separately, since SUNRGBD has same file names for different images in train vs. val
        parent_fname_to_updatelist_dict = form_fname_to_updatelist_dict(
            dname, update_records, split
        )
        split_txt_fpath = f"{_ROOT}/mseg/dataset_lists/{dname}/list/{split}.txt"

        # load up the data root and all absolute paths from the txt file
        img_label_pairs = generate_all_img_label_pair_fpaths(
            data_root=dataroot, split_txt_fpath=split_txt_fpath
        )

        if num_processes > 1:
            send_list_to_workers(
                num_processes=num_processes,
                list_to_split=img_label_pairs,
                worker_func_ptr=verify_mask_worker,
                parent_fname_to_updatelist_dict=parent_fname_to_updatelist_dict,
                mld=mld,
                classname_to_id_map=classname_to_id_map,
                require_strict_boundaries=require_strict_boundaries,
                split=split,
            )
        elif num_processes == 1:
            # useful for debugging in a single thread
            for (img_fpath, label_img_fpath) in img_label_pairs:
                verify_label_img_masks(
                    img_fpath,
                    label_img_fpath,
                    parent_fname_to_updatelist_dict,
                    mld,
                    classname_to_id_map,
                    require_strict_boundaries,
                    split,
                )


def verify_mask_worker(
    pairs: List[Tuple[str, str]],
    start_idx: int,
    end_idx: int,
    kwargs: Mapping[str, Any],
) -> None:
    """Given a list of (rgb image, label image) pairs to remap, call relabel_pair()
    on each one of them.

    Args:
        img_fpath_list: list of strings
        start_idx: integer
        end_idx: integer
        kwargs: dictionary with argument names mapped to argument values
    """
    parent_fname_to_updatelist_dict = kwargs["parent_fname_to_updatelist_dict"]
    mld = kwargs["mld"]
    classname_to_id_map = kwargs["classname_to_id_map"]
    require_strict_boundaries = kwargs["require_strict_boundaries"]
    split = kwargs["split"]

    chunk_sz = end_idx - start_idx
    # process each image between start_idx and end_idx
    for idx in range(start_idx, end_idx):
        if idx % 500 == 0:
            pct_completed = (idx - start_idx) / chunk_sz * 100
            print(f"Completed {pct_completed:.2f}%")
        pair = pairs[idx]
        img_fpath, label_img_fpath = pair
        verify_label_img_masks(
            img_fpath,
            label_img_fpath,
            parent_fname_to_updatelist_dict,
            mld,
            classname_to_id_map,
            require_strict_boundaries,
            split,
        )


def verify_label_img_masks(
    img_fpath: str,
    label_img_fpath: str,
    parent_fname_to_updatelist_dict,
    mld: Any,
    classname_to_id_map: Mapping[str, int],
    require_strict_boundaries: bool,
    split: str,
) -> None:
    """
    Ensure pixel values inside a label map's mask were previously swapped
    to a new value. Mask's category on disk should have previously been
    changed.

    Get fname stem from rgb image file path
    Get sequence ID/parent from label image path file system parent.

    Args:
        img_fpath:
        label_img_fpath:
        parent_fname_to_updatelist_dict:
        mld:
        classname_to_id_map:
        require_strict_boundaries:
        split:
    """
    fname_stem = Path(img_fpath).stem
    parent = Path(label_img_fpath).parts[
        -2
    ]  # get parent name, functions as sequence ID

    if fname_stem not in parent_fname_to_updatelist_dict[parent]:
        # we loop through the entire dataset, and many images won't have any
        # updated masks. they'll pass through here.
        return

    # load up each label_img
    # label image will already by in universal taxonomy
    label_img = imageio.imread(label_img_fpath)

    update_records = parent_fname_to_updatelist_dict[parent][fname_stem]
    for rec in update_records:
        # if it is, perform each update as described in the object
        segment_mask = mld.get_segment_mask(parent, rec.segmentid, fname_stem, split)

        if segment_mask is None:
            print("No such mask found, exiting...")
            exit()

        # update the label image each time
        orig_class_idx = classname_to_id_map[rec.orig_class]
        new_class_idx = classname_to_id_map[rec.relabeled_class]

        assert_mask_identical(label_img, segment_mask, new_class_idx)


def assert_mask_identical(
    label_img: np.ndarray, segment_mask: np.ndarray, new_class_idx: int
):
    """
    should be perfect agreement between label image and instance image, then
    we will enforce strict boundary agreement between the two.
    """
    y, x = np.where(segment_mask == 1)
    identical = np.allclose(
        np.unique(label_img[y, x]), np.array([new_class_idx], dtype=np.uint8)
    )
    assert identical


class DatasetRewritingTask(NamedTuple):
    """Define the parameters need to rewrite the dataset on disk.

    Requires providing a mask-level dataset API as argument.
    """
    orig_dname: str
    remapped_dname: str
    mapping_tsv_fpath: str
    orig_dataroot: str
    remapped_dataroot: str
    dataset_api: Any
    update_records: Any
    require_strict_boundaries: bool


def get_relabeling_task(dname: str) -> DatasetRewritingTask:
    """
    Args:
        dname: name of dataset to apply re-labeling to

    Returns:
        DatasetRewritingTask to complete
    """
    if dname == "ade20k-150":
        return DatasetRewritingTask(
            orig_dname="ade20k-150",
            remapped_dname="ade20k-150-relabeled",
            mapping_tsv_fpath=f"{_ROOT}/mseg/class_remapping_files/ade20k-150_to_ade20k-150-relabeled.tsv",
            orig_dataroot=infos["ade20k-150"].dataroot,
            remapped_dataroot=infos["ade20k-150-relabeled"].dataroot,
            dataset_api=Ade20kMaskDataset(
                semantic_version_dataroot=infos["ade20k-151"].dataroot,
                instance_version_dataroot=infos["ade20k-151-inst"].dataroot,
            ),
            update_records=ade20k_update_records,
            require_strict_boundaries=False,
        )

    elif dname == "bdd":
        return DatasetRewritingTask(
            orig_dname="bdd",
            remapped_dname="bdd-relabeled",
            mapping_tsv_fpath=f"{_ROOT}/mseg/class_remapping_files/bdd_to_bdd-relabeled.tsv",
            orig_dataroot=infos["bdd"].dataroot,
            remapped_dataroot=infos["bdd-relabeled"].dataroot,
            dataset_api=BDDImageLevelDataset(infos["bdd"].dataroot),
            update_records=bdd_update_records,
            require_strict_boundaries=True,
        )

    elif dname == "cityscapes-19":
        return DatasetRewritingTask(
            orig_dname="cityscapes-19",
            remapped_dname="cityscapes-19-relabeled",
            mapping_tsv_fpath=f"{_ROOT}/mseg/class_remapping_files/cityscapes-19_to_cityscapes-19-relabeled.tsv",
            orig_dataroot=infos["cityscapes-19"].dataroot,
            remapped_dataroot=infos["cityscapes-19-relabeled"].dataroot,
            dataset_api=JsonMaskDataset(dataroot=infos["cityscapes-34"].dataroot),
            update_records=cityscapes_update_records,
            require_strict_boundaries=False,
        )  # polygon->raster will be nonstrict

    elif dname == "cityscapes-34":
        return DatasetRewritingTask(
            orig_dname="cityscapes-34",
            remapped_dname="cityscapes-34-relabeled",
            mapping_tsv_fpath=f"{_ROOT}/mseg/class_remapping_files/cityscapes-34_to_cityscapes-34-relabeled.tsv",
            orig_dataroot=infos["cityscapes-34"].dataroot,
            remapped_dataroot=infos["cityscapes-34-relabeled"].dataroot,
            dataset_api=JsonMaskDataset(dataroot=infos["cityscapes-34"].dataroot),
            update_records=cityscapes_update_records,
            require_strict_boundaries=False,
        )  # polygon->raster will be nonstrict

    elif dname == "coco-panoptic-133":
        return DatasetRewritingTask(
            orig_dname="coco-panoptic-133",
            remapped_dname="coco-panoptic-133-relabeled",
            mapping_tsv_fpath=f"{_ROOT}/mseg/class_remapping_files/coco-panoptic-133_to_coco-panoptic-133-relabeled.tsv",
            orig_dataroot=infos["coco-panoptic-133"].dataroot,
            remapped_dataroot=infos["coco-panoptic-133-relabeled"].dataroot,
            dataset_api=COCOPanopticJsonMaskDataset(
                coco_dataroot=infos["coco-panoptic-133"].dataroot
            ),
            update_records=cocop_update_records,
            require_strict_boundaries=True,
        )

    elif dname == "idd-39":
        return DatasetRewritingTask(
            orig_dname="idd-39",
            remapped_dname="idd-39-relabeled",
            mapping_tsv_fpath=f"{_ROOT}/mseg/class_remapping_files/idd-39_to_idd-39-relabeled.tsv",
            orig_dataroot=infos["idd-39"].dataroot,
            remapped_dataroot=infos["idd-39-relabeled"].dataroot,
            dataset_api=JsonMaskDataset(infos["idd-39-relabeled"].dataroot),
            update_records=idd_update_records,
            require_strict_boundaries=False,
        )  # polygon->raster will be nonstrict

    elif dname == "mapillary-public65":
        return DatasetRewritingTask(
            orig_dname="mapillary-public65",
            remapped_dname="mapillary-public65-relabeled",
            mapping_tsv_fpath=f"{_ROOT}/mseg/class_remapping_files/mapillary-public65_to_mapillary-public65-relabeled.tsv",
            orig_dataroot=infos["mapillary-public65"].dataroot,
            remapped_dataroot=infos["mapillary-public65-relabeled"].dataroot,
            dataset_api=MapillaryMaskDataset(
                dataroot=infos["mapillary-public66"].dataroot
            ),
            update_records=mapillary_update_records,
            require_strict_boundaries=True,
        )

    elif dname == "sunrgbd-37":
        return DatasetRewritingTask(
            orig_dname="sunrgbd-37",
            remapped_dname="sunrgbd-37-relabeled",
            mapping_tsv_fpath=f"{_ROOT}/mseg/class_remapping_files/sunrgbd-37_to_sunrgbd-37-relabeled.tsv",
            orig_dataroot=infos["sunrgbd-37"].dataroot,
            remapped_dataroot=infos["sunrgbd-37-relabeled"].dataroot,
            dataset_api=SunrgbdImageLevelDataset(infos["sunrgbd-37"].dataroot),
            update_records=sunrgbd_update_records,
            require_strict_boundaries=True,
        )

    else:
        print(f"This dataset {dname} is not currently configured for re-labeling")
        print("Exiting...")
        exit()


def main(args):
    """
    We use the MSeg dataroot explicitly, as specified in mseg/utils/dataset_config.py
    """
    dnames_to_verify = [
        "ade20k-150",
        "bdd",
        "coco-panoptic-133",
        "mapillary-public65",
        "sunrgbd-37"
        # Can't do this for the following two since masks overlap
        # 'cityscapes-19',
        # 'idd-39',
    ]
    for dname in dnames_to_verify:
        task = get_relabeling_task(dname)
        print(f"Completing task:")
        print(f"\t{task.orig_dname}")
        print(f"\t{task.remapped_dname}")
        print(f"\t{task.mapping_tsv_fpath}")
        print(f"\t{task.orig_dataroot}")
        print(f"\t{task.remapped_dataroot}")

        # Overwrite the universal labels using mask/segment updates.
        verify_all_relabeled_dataset_segments(
            args.num_processes,
            task.dataset_api,
            task.remapped_dname,
            task.remapped_dataroot,
            task.update_records,
            task.require_strict_boundaries,
        )
    print("Verifed relabeling for: ", dnames_to_verify)


if __name__ == "__main__":
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_processes",
        type=int,
        required=True,
        help="number of processes to use (multiprocessing)",
    )
    args = parser.parse_args()
    main(args)
