#!/usr/bin/python3

import argparse
import collections
from collections import defaultdict
import imageio
import numpy as np
from pathlib import Path
import pdb

from typing import Any, List, Mapping, Tuple

from mseg.utils.multiprocessing_utils import send_list_to_workers
from mseg.utils.txt_utils import generate_all_img_label_pair_fpaths
from mseg.utils.names_utils import get_classname_to_dataloaderid_map
from mseg.utils.mask_utils import swap_px_inside_mask

from mseg.dataset_apis.Ade20kMaskLevelDataset import Ade20kMaskDataset
from mseg.dataset_apis.BDDImageLevelDataset import BDDImageLevelDataset
from mseg.dataset_apis.COCOPanopticJsonMaskDataset import COCOPanopticJsonMaskDataset
from mseg.dataset_apis.JsonMaskLevelDataset import JsonMaskDataset
from mseg.dataset_apis.MapillaryMaskDataset import MapillaryMaskDataset
from mseg.dataset_apis.SunrgbdImageLevelDataset import SunrgbdImageLevelDataset

from mseg.label_preparation.relabeled_data_containers import LabelImageUpdateRecord, DatasetClassUpdateRecord
from mseg.label_preparation.remap_dataset import remap_dataset
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


def get_unique_mask_identifiers(dname: str, annot_fname: str, data_split: str) -> Tuple[str, str, int]:
    """
    Will be used to obtain the unique:
            -	filesystem parent (i.e. sequence ID),
            -	image ID
            -	segment ID
    to overwrite this mask/segment later, by matching with label image file path.

    Unfortunately, image IDs can be repeated across sequences/splits, so we need 3 unique
    identifiers per mask. The sequence ID should always be the 2nd to last item in the
    file path so that given an absolute path, we can immediately find it (`fname_parent`)

    Args:
        dname: dataset_name
        annot_fname:
        data_split: string representing data subset, e.g. 'train' or 'val'

    Returns:
        fname_parent: label file path parent
        fname_stem: rgb file name stem
        segment_id: integer unique segment identifier
    """
    annot_fname = Path(annot_fname).stem
    fname_parent = None
    if dname in ["ade20k-150", "ade20k-150-relabeled", "coco-panoptic-133", "coco-panoptic-133-relabeled"]:
        # e.g '000000024880_7237508.png' -> ('000000024880', 7237508)
        fname_stem = "_".join(annot_fname.split("_")[:-1])
        segmentid = annot_fname.split("_")[-1]
        if "ade20k" in dname:
            fname_parent = "training" if data_split == "train" else "validation"
        elif "coco" in dname:
            fname_parent = f"{data_split}2017"

    elif dname in ["bdd", "bdd-relabeled"]:
        # e.g. 0f4e8f1e-6ba53d52_11.jpg
        fname_stem = annot_fname.split("_")[0]
        segmentid = annot_fname.split("_")[-1]
        fname_parent = data_split

    elif dname in ["cityscapes-19", "cityscapes-19-relabeled", "cityscapes-34", "cityscapes-34-relabeled"]:
        # e.g seqfrankfurt_frankfurt_000000_013942_leftImg8bit_28.jpg
        fname_stem = "_".join(annot_fname.split("_")[1:-1])
        segmentid = annot_fname.split("_")[-1]
        fname_parent = annot_fname.split("_")[0][3:]

    elif dname in ["idd-39", "idd-39-relabeled"]:
        # seq173_316862_leftImg8bit_34.jpg
        fname_stem = "_".join(annot_fname.split("_")[1:-1])
        segmentid = annot_fname.split("_")[-1]
        seq_prefix = annot_fname.split("_")[0]
        fname_parent = seq_prefix[3:]  # after 'seq'

    elif dname in ["sunrgbd-37", "sunrgbd-37-relabeled"]:
        # we refer to the `test` split (on disk) as `val` split,
        # since `val` undefined.
        fname_stem = annot_fname.split("_")[0]
        segmentid = annot_fname.split("_")[-1]
        fname_parent = "train" if data_split == "train" else "test"

    elif dname in ["mapillary-public65", "mapillary-public65-relabeled"]:
        # e.g. mapillary_czkO_9In4u30opBy5H1uLg_259.jpg
        fname_stem = "_".join(annot_fname.split("_")[1:-1])
        segmentid = annot_fname.split("_")[-1]
        fname_parent = "labels"
    else:
        print("Unknown dataset")
        quit()

    return fname_parent, fname_stem, int(segmentid)


def form_fname_to_updatelist_dict(
    dname: str, update_records: List[DatasetClassUpdateRecord], split: str
) -> Mapping[str, List[LabelImageUpdateRecord]]:
    """
    Form a large dictionary mapping (parent,filename)->(update objects).
    Later we can check to see if this image is in our big dictionary of filename->updates
    so we can know if we need to overwrite the mask.

    Args:
        update_records

    Returns:
        parent_fname_to_updatelist_dict
    """
    parent_fname_to_updatelist_dict = defaultdict(dict)
    for rec in update_records:
        for annot_fname in rec.img_list:

            if rec.split != split:
                continue
            # ADE20K has underscores in the stem, so last underscore separates fname and segment ID
            parent, fname_stem, segmentid = get_unique_mask_identifiers(dname, annot_fname, rec.split)
            img_rec = LabelImageUpdateRecord(
                rec.dataset_name, fname_stem, segmentid, rec.split, rec.orig_class, rec.relabeled_class
            )

            # Check for duplicated annotations.
            same_img_recs = parent_fname_to_updatelist_dict[parent].get(fname_stem, [])
            is_dup = any([int(segmentid) == same_img_rec.segmentid for same_img_rec in same_img_recs])
            if is_dup:
                print("Found Duplicate!")
                pdb.set_trace()

            parent_fname_to_updatelist_dict[parent].setdefault(fname_stem, []).append(img_rec)

    return parent_fname_to_updatelist_dict


def write_out_updated_dataset(
    num_processes: int, mld: Any, dname: str, dataroot: str, update_records, require_strict_boundaries: bool
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
    classname_to_id_map = get_classname_to_dataloaderid_map(dname, include_ignore_idx_cls=True)
    # Check for incorrect class names.
    for rec in update_records:
        valid_orig = rec.orig_class in classname_to_id_map.keys()
        valid_relabeled = rec.relabeled_class in classname_to_id_map.keys()
        if not (valid_orig and valid_relabeled):
            print(rec.__dict__)
            print(f"Invalid universal classname: {rec.orig_class}, {rec.relabeled_class}")
            quit()

    for split in ["train", "val"]:
        # Create for each split separately, since SUNRGBD has same file names for different images in train vs. val
        parent_fname_to_updatelist_dict = form_fname_to_updatelist_dict(dname, update_records, split)
        split_txt_fpath = f"{_ROOT}/dataset_lists/{dname}/list/{split}.txt"

        # load up the data root and all absolute paths from the txt file
        img_label_pairs = generate_all_img_label_pair_fpaths(data_root=dataroot, split_txt_fpath=split_txt_fpath)

        if num_processes > 1:
            send_list_to_workers(
                num_processes=num_processes,
                list_to_split=img_label_pairs,
                worker_func_ptr=overwrite_mask_worker,
                parent_fname_to_updatelist_dict=parent_fname_to_updatelist_dict,
                mld=mld,
                classname_to_id_map=classname_to_id_map,
                require_strict_boundaries=require_strict_boundaries,
                split=split,
            )
        elif num_processes == 1:
            # useful for debugging in a single thread
            for (img_fpath, label_img_fpath) in img_label_pairs:
                overwrite_label_img_masks(
                    img_fpath,
                    label_img_fpath,
                    parent_fname_to_updatelist_dict,
                    mld,
                    classname_to_id_map,
                    require_strict_boundaries,
                    split,
                )


def overwrite_mask_worker(
    pairs: List[Tuple[str, str]], start_idx: int, end_idx: int, kwargs: Mapping[str, Any]
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
        overwrite_label_img_masks(
            img_fpath,
            label_img_fpath,
            parent_fname_to_updatelist_dict,
            mld,
            classname_to_id_map,
            require_strict_boundaries,
            split,
        )


def overwrite_label_img_masks(
    img_fpath: str,
    label_img_fpath: str,
    parent_fname_to_updatelist_dict,
    mld: Any,
    classname_to_id_map: Mapping[str, int],
    require_strict_boundaries: bool,
    split: str,
) -> None:
    """
    Swap the pixel values inside a label map's mask to a new value. This
    effectively changes the mask's category on disk.

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
    parent = Path(label_img_fpath).parts[-2]  # get parent name, functions as sequence ID

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
        label_img = swap_px_inside_mask(
            label_img, segment_mask, orig_class_idx, new_class_idx, require_strict_boundaries
        )

        # save it to disk, at new data root, using same abs path as before
    overwrite = True  # False #
    if overwrite:
        imageio.imwrite(label_img_fpath, label_img)


#####---------------------------------------------------------------

# Define the parameters need to rewrite the dataset on disk.
# Requires providing a mask-level dataset API as argument.
DatasetRewritingTask = collections.namedtuple(
    typename="DatasetRewritingTask",
    field_names="orig_dname remapped_dname mapping_tsv_fpath "
    "orig_dataroot remapped_dataroot dataset_api update_records require_strict_boundaries",
)


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
            mapping_tsv_fpath=f"{_ROOT}/class_remapping_files/ade20k-150_to_ade20k-150-relabeled.tsv",
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
            mapping_tsv_fpath=f"{_ROOT}/class_remapping_files/bdd_to_bdd-relabeled.tsv",
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
            mapping_tsv_fpath=f"{_ROOT}/class_remapping_files/cityscapes-19_to_cityscapes-19-relabeled.tsv",
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
            mapping_tsv_fpath=f"{_ROOT}/class_remapping_files/cityscapes-34_to_cityscapes-34-relabeled.tsv",
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
            mapping_tsv_fpath=f"{_ROOT}/class_remapping_files/coco-panoptic-133_to_coco-panoptic-133-relabeled.tsv",
            orig_dataroot=infos["coco-panoptic-133"].dataroot,
            remapped_dataroot=infos["coco-panoptic-133-relabeled"].dataroot,
            dataset_api=COCOPanopticJsonMaskDataset(coco_dataroot=infos["coco-panoptic-133"].dataroot),
            update_records=cocop_update_records,
            require_strict_boundaries=True,
        )

    elif dname == "idd-39":
        return DatasetRewritingTask(
            orig_dname="idd-39",
            remapped_dname="idd-39-relabeled",
            mapping_tsv_fpath=f"{_ROOT}/class_remapping_files/idd-39_to_idd-39-relabeled.tsv",
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
            mapping_tsv_fpath=f"{_ROOT}/class_remapping_files/mapillary-public65_to_mapillary-public65-relabeled.tsv",
            orig_dataroot=infos["mapillary-public65"].dataroot,
            remapped_dataroot=infos["mapillary-public65-relabeled"].dataroot,
            dataset_api=MapillaryMaskDataset(dataroot=infos["mapillary-public66"].dataroot),
            update_records=mapillary_update_records,
            require_strict_boundaries=True,
        )

    elif dname == "sunrgbd-37":
        return DatasetRewritingTask(
            orig_dname="sunrgbd-37",
            remapped_dname="sunrgbd-37-relabeled",
            mapping_tsv_fpath=f"{_ROOT}/class_remapping_files/sunrgbd-37_to_sunrgbd-37-relabeled.tsv",
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
    task = get_relabeling_task(args.dataset_to_relabel)

    # Rewrite original dataset into universal label space.
    remap_dataset(
        task.orig_dname,
        task.remapped_dname,
        task.mapping_tsv_fpath,
        old_dataroot=task.orig_dataroot,
        remapped_dataroot=task.remapped_dataroot,
        num_processes=args.num_processes,
    )
    # Overwrite the universal labels using mask/segment updates.
    write_out_updated_dataset(
        args.num_processes,
        task.dataset_api,
        task.remapped_dname,
        task.remapped_dataroot,
        task.update_records,
        task.require_strict_boundaries,
    )


if __name__ == "__main__":
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_processes", type=int, required=True, help="number of processes to use (multiprocessing)")
    parser.add_argument("--dataset_to_relabel", type=str, required=True, help="name of dataset to apply re-labeling to")

    args = parser.parse_args()
    main(args)
