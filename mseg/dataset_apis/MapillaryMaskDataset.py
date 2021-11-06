#!/usr/bin/python3

import argparse
import glob
from pathlib import Path
from typing import Any, List, Mapping

import imageio
import numpy as np

import mseg.utils.names_utils as names_utils
from mseg.utils.mask_utils import save_binary_mask_double, rgb_img_to_obj_cls_img, form_mask_triple_embedded_classnames
from mseg.utils.cv2_utils import cv2_imread_rgb
from mseg.utils.multiprocessing_utils import send_list_to_workers



"""
Note: We do not use this dataset API at training or inference time.
It is designed purely for generating the re-labeled masks of the 
MSeg dataset (found in ground truth label maps) on disk, prior to 
training/inference.
"""


class MapillaryMaskDataset:
    def __init__(self, dataroot: str) -> None:
        """
        Args:
            dataroot: string representing path to unzipped Mapillary file
        """
        self.dataroot = dataroot
        self.dataset_ordered_colors = names_utils.load_dataset_colors_arr("mapillary-public66")
        self.id_to_classname_map = names_utils.get_dataloader_id_to_classname_map(dataset_name="mapillary-public66")

    def labelrgb_to_label(self, label_img_rgb: np.ndarray) -> np.ndarray:
        """
        Args:
            label_img_rgb:

        Returns:
            label_img
        """
        return rgb_img_to_obj_cls_img(label_img_rgb, self.dataset_ordered_colors)

    def dump_class_masks(self, highlight_classname: str, folder_prefix: str, num_processes: int = 4):
        """
        'semseg-format' vs. 'labels'
        """
        for split in ["training", "validation"]:
            split_img_dir = f"{self.dataroot}/{split}/images/*.jpg"
            rgb_img_fpaths = glob.glob(split_img_dir)

            if num_processes > 1:
                send_list_to_workers(
                    num_processes=num_processes,
                    list_to_split=rgb_img_fpaths,
                    worker_func_ptr=dump_img_masks_worker,
                    highlight_classname=highlight_classname,
                    split=split,
                    folder_prefix=folder_prefix,
                    api=self,
                )
            elif num_processes == 1:
                for i, rgb_img_fpath in enumerate(rgb_img_fpaths):
                    print(i)
                    dump_img_masks(highlight_classname, split, rgb_img_fpaths[0], folder_prefix, api=self)

    def get_segment_mask(self, seq_id: None, segmentid: int, fname_stem: str, split: str) -> np.ndarray:
        """
        Use instance images and a designated image and instance ID to obtain a specific
        instance mask.

        Args:
            seq_id:
            segmentid:
            fname_stem:
            split:

        Returns:
            segment_mask:
        """
        segmentid = int(segmentid)  # if its a string, force it to int
        split_renaming_dict = {"train": "training", "val": "validation"}
        split = split_renaming_dict[split]

        instance_img_fpath = f"{self.dataroot}/{split}/instances/{fname_stem}.png"
        instance_img = imageio.imread(instance_img_fpath)

        instance_id = segmentid
        instance_mask = (instance_img == instance_id).astype(np.uint8)
        return instance_mask


def dump_img_masks_worker(rgb_img_fpaths: List[str], start_idx: int, end_idx: int, kwargs: Mapping[str, Any]) -> None:
    """Given a list of image file paths, call dump_img_masks
    on each one of them.
    
    Args:
        img_fpath_list: list of strings
        start_idx: integer
        end_idx: integer
        kwargs: dictionary with argument names mapped to argument values
    """
    highlight_classname = kwargs["highlight_classname"]
    split = kwargs["split"]
    folder_prefix = kwargs["folder_prefix"]
    api = kwargs["api"]

    chunk_sz = end_idx - start_idx
    # process each image between start_idx and end_idx
    for idx in range(start_idx, end_idx):
        if idx % 500 == 0:
            pct_completed = (idx - start_idx) / chunk_sz * 100
            print(f"Completed {pct_completed:.2f}%")
        rgb_img_fpath = rgb_img_fpaths[idx]
        dump_img_masks(highlight_classname, split, rgb_img_fpath, folder_prefix, api)


def dump_img_masks(
    highlight_classname: str, split: str, rgb_img_fpath: str, folder_prefix: str, api: MapillaryMaskDataset
):
    """ """
    # print(f'On image {i}/{len(rgb_img_fpaths)}')
    rgb_img = cv2_imread_rgb(rgb_img_fpath)
    fname_stem = Path(rgb_img_fpath).stem
    label_rgb_fpath = f"{api.dataroot}/{split}/labels/{fname_stem}.png"
    label_img_rgb = cv2_imread_rgb(label_rgb_fpath)
    label_img = api.labelrgb_to_label(label_img_rgb)

    present_classnames = [api.id_to_classname_map[id] for id in np.unique(label_img)]
    if highlight_classname not in present_classnames:
        return

    instance_img_fpath = f"{api.dataroot}/{split}/instances/{fname_stem}.png"
    instance_img = imageio.imread(instance_img_fpath)
    instance_ids = np.unique(instance_img)

    for instance_id in instance_ids:
        instance_mask = (instance_img == instance_id).astype(np.uint8)

        is_single_class, sem_class_ids = mask_belongs_single_semantic_class(instance_mask, label_img)
        assert is_single_class
        instance_classid = int(sem_class_ids)

        instance_classname = api.id_to_classname_map[int(sem_class_ids)]
        if instance_classname != highlight_classname:  # not in required_class_names:
            continue

        save_fpath = f"temp_files/{folder_prefix}_{split}_2020_04_18/mapillary_{fname_stem}_{instance_id}.jpg"
        save_binary_mask_double(rgb_img, instance_mask, save_fpath, save_to_disk=True)


def mask_belongs_single_semantic_class(segment_mask: np.ndarray, label_img: np.ndarray):
    """
    Args:
        segment_mask
        label_img

    Returns:
        is_single_class
        sem_class_ids
    """
    y, x = np.where(segment_mask == 1)
    # verify single relevant semantic class
    is_single_class = (np.unique(label_img[y, x]).size == 1,)
    sem_class_ids = np.unique(label_img[y, x])
    return is_single_class, sem_class_ids


def main() -> None:
    """
    Example Use:
    python MapillaryMaskDataset.py -classname="Ground Animal" --folder_prefix mseg_mapillary_water_2019_04_19 --num_processes 4
    --dataroot /Users/johnlamb/Documents/mseg-api-staging/MSeg_Downloaded/mseg_dataset/MapillaryVistasPublic/Mapillary-Vistas-Dataset_Public_v1.1

    python MapillaryMaskDataset.py --classname="Ground Animal" --folder_prefix mseg_phase3_cluster_ground_animal
    --dataroot /export/share/Datasets/MSegV2/mseg_dataset/MapillaryVistasPublic --num_processes 100

    folder_prefix = 'mseg_phase3_cluster_ground_animal'

    highlight_classname = 'Water'
    folder_prefix = 'mseg_phase3_cluster_fixed2ndbug_mapillary_water'
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--classname", type=str, required=True, help="name of class to visualize")
    parser.add_argument("--dataroot", type=str, required=True, help="path to Mapillary data root")
    parser.add_argument("--num_processes", type=int, required=True, help="number of workers")
    parser.add_argument("--folder_prefix", type=str, required=True, help="name of folder to save images in")

    args = parser.parse_args()
    print(args)
    m_api = MapillaryMaskDataset(args.dataroot)
    m_api.dump_class_masks(args.classname, args.folder_prefix, args.num_processes)


if __name__ == "__main__":
    main()
