#!/usr/bin/env python3

import argparse
import glob
from pathlib import Path
from typing import List, Optional, Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np

import mseg.utils.names_utils as names_utils
from mseg.utils.mask_utils import (
    form_mask_triple_embedded_classnames,
    save_binary_mask_double,
    get_present_classes_in_img,
)


"""
Note: We do not use this dataset API at training or inference time.
It is designed purely for generating the re-labeled masks of the 
MSeg dataset (found in ground truth label maps) on disk, prior to 
training/inference.
"""


class SunrgbdImageLevelDataset:
    def __init__(self, dataset_dir):
        """ """
        self.id_to_classname_map = names_utils.get_dataloader_id_to_classname_map(dataset_name="sunrgbd-37")
        self.classname_to_id_map = names_utils.get_classname_to_dataloaderid_map(dataset_name="sunrgbd-37")

        self.img_dir = f"{dataset_dir}/image"
        self.label_dir = f"{dataset_dir}/semseg-label37"

    def get_class_masks(self, required_class_names: List[str], highlight_classname: str, condition, folder_prefix: str):
        """ """
        for split in ["train", "test"]:
            rgb_fpaths = glob.glob(f"{self.img_dir}/{split}/*.jpg")

            num_split_imgs = len(rgb_fpaths)
            for i, rgb_fpath in enumerate(rgb_fpaths):
                print(f"On image {i}/{num_split_imgs-1}")

                fname_stem = Path(rgb_fpath).stem
                rgb_img, label_img = self.get_img_pair(fname_stem, split)

                present_classnames = get_present_classes_in_img(label_img, self.id_to_classname_map)
                if not all([req_name in present_classnames for req_name in required_class_names]):
                    continue

                fname_stem = Path(rgb_fpath).stem
                for class_idx in np.unique(label_img):
                    instance_classname = self.id_to_classname_map[class_idx]
                    if instance_classname != highlight_classname:  # not in required_class_names:
                        continue

                    label_mask = (label_img == class_idx).astype(np.uint8)
                    save_fpath = f"temp_files/{folder_prefix}_{split}/{fname_stem}_{class_idx}.jpg"
                    save_binary_mask_double(rgb_img, label_mask, save_fpath, save_to_disk=True)

    def get_img_pair(self, fname_stem: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """ """
        if split == "val":
            # SUNRGB has no val, only test
            split = "test"
        fname_stem = fname_stem.replace("img-", "")
        rgb_fpath = f"{self.img_dir}/{split}/img-{fname_stem}.jpg"

        extended_fname_stem = fname_stem.zfill(8)
        label_fpath = f"{self.label_dir}/{split}/{extended_fname_stem}.png"

        label_img = imageio.imread(label_fpath)
        rgb_img = imageio.imread(rgb_fpath)
        return rgb_img, label_img

    def get_segment_mask(self, seq_id: str, query_segmentid: int, fname_stem: str, split: str) -> Optional[np.ndarray]:
        """
        seq_id is only provided so that all other datasets can share a common API.
        """
        if split == "val":
            # SUNRGB has no val, only test
            split = "test"
        rgb_img, label_img = self.get_img_pair(fname_stem, split)
        for class_idx in np.unique(label_img):
            if class_idx == query_segmentid:
                label_mask = (label_img == class_idx).astype(np.uint8)
                return label_mask
        return None


def visualize_sunrgbd_class_masks(d_api, classname):
    """
    Dump one image per mask, for all masks of a specific class.
    """
    required_class_names = [classname]
    highlight_classname = classname
    condition = "intersection"
    folder_prefix = f"sunrgbd_{classname}"
    d_api.get_class_masks(
        required_class_names=required_class_names,
        highlight_classname=highlight_classname,
        condition=condition,
        folder_prefix=folder_prefix,
    )


def main():
    """
	Visualize masks of a chosen category from SUN RGB-D.

	Usage:
	python SunrgbdImageLevelDataset.py \
	--dataroot /Users/johnlamb/Downloads/SUNRGBD-37-CLUSTER --classname lamp
	"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--classname", type=str, required=True, help="name of class to visualize")
    parser.add_argument("--dataroot", type=str, required=True, help="path to SUN RGB-D data root")
    args = parser.parse_args()

    d_api = SunrgbdImageLevelDataset(args.dataroot)
    visualize_sunrgbd_class_masks(d_api, args.classname)


if __name__ == "__main__":
    main()
