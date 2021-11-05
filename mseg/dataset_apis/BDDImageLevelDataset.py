#!/usr/bin/env python3

import argparse
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pdb
from typing import List, Optional, Tuple

import mseg.utils.names_utils as names_utils

from mseg.utils.mask_utils import (
    form_mask_triple_embedded_classnames,
    save_binary_mask_double,
    convert_instance_img_to_mask_img,
    get_present_classes_in_img,
)

ROOT = Path(__file__).resolve().parent.parent.parent

"""
BDD does not include instance masks. Rather, it includes RGB images
and then semantic label maps (and semantic label maps in RGB format).

Note: We do not use this dataset API at training or inference time.
It is designed purely for generating the re-labeled masks of the 
MSeg dataset (found in ground truth label maps) on disk, prior to 
training/inference.
"""


class BDDImageLevelDataset:
    def __init__(self, dataroot):
        """ """
        self.dataroot = dataroot
        self.img_dir = f"{self.dataroot}/seg/images"
        self.label_dir = f"{self.dataroot}/seg/labels"
        self.id_to_classname_map = names_utils.get_dataloader_id_to_classname_map(dataset_name="bdd")
        self.classname_to_id_map = names_utils.get_classname_to_dataloaderid_map(dataset_name="bdd")

    def get_class_masks(self, required_class_names: List[str], highlight_classname: str, condition, folder_prefix: str):
        """ """
        for split in ["train", "val"]:
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
                # save_fpath = f'temp_files/bdd_vis_min400px/{fname_stem}.png'
                # form_mask_triple_embedded_classnames(rgb_img, label_img, self.id_to_classname_map, save_fpath, save_to_disk=True)
                # filename = Path(label_img_fpath).name

                for class_idx in np.unique(label_img):
                    instance_classname = self.id_to_classname_map[class_idx]
                    if instance_classname != highlight_classname:  # not in required_class_names:
                        continue

                    label_mask = (label_img == class_idx).astype(np.uint8)
                    save_fpath = str(ROOT / f"temp_files/{folder_prefix}_{split}/{fname_stem}_{class_idx}.jpg")
                    save_binary_mask_double(rgb_img, label_mask, save_fpath, save_to_disk=True)

    def get_img_pair(self, fname_stem, split):
        """
        Args:
            fname_stem
            split

        Returns:
            rgb_img
            label_img
        """
        rgb_fpath = f"{self.img_dir}/{split}/{fname_stem}.jpg"
        rgb_img = imageio.imread(rgb_fpath)
        label_fpath = rgb_fpath.replace("/images/", "/labels/")
        label_fpath = label_fpath.replace(".jpg", "_train_id.png")
        label_img = imageio.imread(label_fpath)
        return rgb_img, label_img

    def get_segment_mask(self, seq_id: str, query_segmentid: int, fname_stem: str, split: str) -> Optional[np.ndarray]:
        """
        seq_id is only provided so that all other datasets can share a common API.
        """
        rgb_img, label_img = self.get_img_pair(fname_stem, split)
        for class_idx in np.unique(label_img):
            if class_idx == query_segmentid:
                label_mask = (label_img == class_idx).astype(np.uint8)
                return label_mask
        return None

    def visualize_imgs(self) -> None:
        """
        Visualize label maps superimposed on RGB images for BDD.
        """
        for split in ["train", "val"]:
            split_img_fpaths = glob.glob(f"{self.img_dir}/{split}/*.jpg")
            for img_fpath in split_img_fpaths:
                fname_stem = Path(img_fpath).stem
                img_rgb = imageio.imread(img_fpath)
                label_img_fpath = f"{self.label_dir}/{split}/{fname_stem}_train_id.png"
                label_img = imageio.imread(label_img_fpath)

                mask_img = convert_instance_img_to_mask_img(label_img, img_rgb)
                plt.imshow(mask_img)
                plt.show()
                quit()


def visualize_bdd_class_masks(d_api, classname):
    """
    Dump one image per mask, for all masks of a specific class.
    """
    required_class_names = [classname]
    highlight_classname = classname
    condition = "intersection"
    folder_prefix = f"bdd_{classname}"
    d_api.get_class_masks(
        required_class_names=required_class_names,
        highlight_classname=highlight_classname,
        condition=condition,
        folder_prefix=folder_prefix,
    )


def main() -> None:
    """
	Visualize masks of a chosen category from BDD.

	Usage:
	python BDDImageLevelDataset.py \
	--dataroot /Users/johnlamb/Downloads/bdd100k --classname person
	"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--classname", type=str, required=True, help="name of class to visualize")
    parser.add_argument("--dataroot", type=str, required=True, help="path to bdd100k data root")
    args = parser.parse_args()

    d_api = BDDImageLevelDataset(args.dataroot)
    visualize_bdd_class_masks(d_api, args.classname)


if __name__ == "__main__":
    main()
