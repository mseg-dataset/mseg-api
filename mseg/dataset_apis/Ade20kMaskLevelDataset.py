#!/usr/bin/python3

import argparse
import copy
import cv2
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from typing import List, Optional, Tuple

import mseg.utils.mask_utils as mask_utils
import mseg.utils.names_utils as names_utils
from mseg.utils.cv2_utils import grayscale_to_color


MIN_REQ_PX = 10  # each instance mask must have at least 10 px to be relabeled

"""
ADE20K data is distributed in two different forms -- one for the 
Scene Parsing Challenge, and another with the raw instance data.
We require both forms (see download_scripts/README.md).

Note: We do not use this dataset API at training or inference time.
It is designed purely for generating the re-labeled masks of the 
MSeg dataset (found in ground truth label maps) on disk, prior to 
training/inference.
"""


class Ade20kMaskDataset:
    """
    Simple API to interact with instance masks of ADE20K dataset.
    """

    def __init__(self, semantic_version_dataroot: str, instance_version_dataroot: str):
        """
        Args:
            semantic_version_dataroot: from ADEChallengeData2016
            instance_verson_dataroot: from ADE20K_2016_07_26
        """
        self.img_dir = f"{semantic_version_dataroot}/images"
        self.ade20k_instance_dataroot = instance_version_dataroot

        # following two maps are only used for relabeling original data
        self.id_to_classname_map = names_utils.get_dataloader_id_to_classname_map(dataset_name="ade20k-151")
        self.classname_to_id_map = names_utils.get_classname_to_dataloaderid_map(dataset_name="ade20k-151")

        self.ade20k_split_nickname_dict = {"train": "training", "val": "validation"}

        # make it easy to look up an RGB file path, given just the filename stem
        self.fname_to_rgbfpath_dict = {}
        for split in ["train", "val"]:
            ade20k_split_nickname = self.ade20k_split_nickname_dict[split]
            rgb_fpaths = glob.glob(f"{self.img_dir}/{ade20k_split_nickname}/*.jpg")
            for rgb_fpath in rgb_fpaths:
                fname_stem = Path(rgb_fpath).stem
                assert fname_stem not in self.fname_to_rgbfpath_dict
                self.fname_to_rgbfpath_dict[fname_stem] = rgb_fpath

        # make it easy to look up a SEG RGB file path, given just the filename stem
        self.fname_to_segrgbfpath_dict = {}
        seg_rgb_fpaths = glob.glob(f"{self.ade20k_instance_dataroot}/images/**/*seg.png", recursive=True)
        for seg_rgb_fpath in seg_rgb_fpaths:
            fname_stem = Path(seg_rgb_fpath).stem
            assert "_seg" in fname_stem[-4:]
            # remove suffix now
            fname_stem = fname_stem.replace("_seg", "")
            self.fname_to_segrgbfpath_dict[fname_stem] = seg_rgb_fpath

    def dump_class_masks(
        self, required_class_names: List[str], highlight_classname: str, condition, folder_prefix: str
    ) -> None:
        """Get for all splits, at once.

        Args:
            required_class_names
            highlight_classname: class to highlight in pink
            condition
            folder_prefix
        """
        for split in ["train", "val"]:
            ade20k_split_nickname = self.ade20k_split_nickname_dict[split]
            rgb_fpaths = glob.glob(f"{self.img_dir}/{ade20k_split_nickname}/*.jpg")

            for i, rgb_fpath in enumerate(rgb_fpaths):
                print(f"On {i}/{len(rgb_fpaths)-1}")
                fname_stem = Path(rgb_fpath).stem
                assert rgb_fpath == self.fname_to_rgbfpath_dict[fname_stem]

                rgb_img, label_img = self.get_img_pair(fname_stem)

                present_class_idxs = np.unique(label_img)
                present_classnames = [self.id_to_classname_map[idx] for idx in present_class_idxs]

                if not all([req_name in present_classnames for req_name in required_class_names]):
                    continue

                seg_rgb_fpath = self.fname_to_segrgbfpath_dict[fname_stem]
                instance_masks, instance_ids = get_ade20k_instance_label_masks(seg_rgb_fpath, rgb_img)
                for (instance_mask, instance_id) in zip(instance_masks, instance_ids):

                    # test the instance's class
                    label_votes, majority_vote = mask_utils.get_instance_mask_class_votes(instance_mask, label_img)
                    if label_votes.size < MIN_REQ_PX:
                        continue

                    instance_classname = self.id_to_classname_map[majority_vote]
                    if instance_classname != highlight_classname:  # not in required_class_names:
                        continue

                    save_fname = f"{fname_stem}_{instance_id}.png"
                    save_fpath = f"temp_files/{folder_prefix}_{split}_2019_12_16/{save_fname}"
                    mask_utils.save_binary_mask_double(rgb_img, instance_mask, save_fpath, save_to_disk=True)

    def get_img_pair(self, fname_stem: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load 2-tuple of image data from disk (RGB and label).

        Args:
            fname_stem: string representing

        Returns:
            rgb_img: array of shape (H,W,3) representing RGB image.
            label_img
        """
        # look it up in the dictionary
        rgb_fpath = self.fname_to_rgbfpath_dict[fname_stem]

        label_fpath = rgb_fpath.replace("/images/", "/annotations/")
        label_fpath = label_fpath.replace(".jpg", ".png")

        rgb_img = imageio.imread(rgb_fpath)

        if rgb_img.ndim == 2:
            # this image was grayscale
            rgb_img = grayscale_to_color(rgb_img)
        assert rgb_img.ndim == 3
        h, w, ch = rgb_img.shape
        assert ch == 3

        label_img = imageio.imread(label_fpath)
        assert label_img.ndim == 2
        assert rgb_img.shape[:2] == label_img.shape[:2]

        return rgb_img, label_img

    def get_segment_mask(self, seq_id: None, segmentid: int, fname_stem: str, split: str) -> Optional[np.ndarray]:
        """
        Args:
            segmentid: integer representing segment unique ID
            fname_stem
            split: dataset split, i.e. 'train' or 'val'

        Returns:
            segment_mask
        """
        segment_mask = None
        rgb_img, label_img = self.get_img_pair(fname_stem)

        seg_rgb_fpath = self.fname_to_segrgbfpath_dict[fname_stem]
        instance_masks, instance_ids = get_ade20k_instance_label_masks(seg_rgb_fpath, rgb_img)
        for (instance_mask, instance_id) in zip(instance_masks, instance_ids):

            if instance_id == segmentid:
                segment_mask = instance_mask
                break

        # test the instance's class
        label_votes, majority_vote = mask_utils.get_instance_mask_class_votes(copy.deepcopy(segment_mask), label_img)

        if label_votes.size < MIN_REQ_PX:
            print("Big problem here! quitting...")
            quit()

        if segment_mask is None:
            print("Specified segment ID does not exist.")
            return None
        return segment_mask


def get_ade20k_instance_label_masks(seg_rgb_fpath: str, rgb_img: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    ADE20K non-Scene-Parsing-Challenge data provides instance masks, and stores the
    instance IDs in the "B" channel of an RGB image. However, the non-challenge data
    is at a different resolution from Scene-Parsing-Challenge data.

    Therefore, we resize the instance image to be the same size as label map/rgb image.
    One could instead do the opposite, i.e. resize label image to fit the instance img,
    but that is not our approach.

    From official documentation:
            *_seg.png: object segmentation mask. This image contains information about
            the object class segmentation masks and also separates each class into
            instances. The channels R and G encode the objects class masks. The channel B
            encodes the instance object masks. The function loadAde20K.m extracts both masks.

    Args:
        seg_rgb_fpath: string representing path to a 3-channel segmentation label image
        rgb_img: Numpy array representing SceneParsingChallenge image size

    Returns:
        label_masks: List of 2d arrays
        instance_ids: List of integers for unique instance IDs present
    """
    num_rows, num_cols = rgb_img.shape[:2]
    seg_rgb = imageio.imread(seg_rgb_fpath)

    R = seg_rgb[:, :, 0]
    G = seg_rgb[:, :, 1]
    B = seg_rgb[:, :, 2]

    # Resize it to our size...
    B = cv2.resize(B, (num_cols, num_rows), interpolation=cv2.INTER_NEAREST)

    # ObjectClassMasks = (R.astype(np.uint16)/10)*256+G.astype(np.uint16)
    instance_ids = np.unique(B)

    instance_masks = []
    for instance_id in instance_ids:
        instance_masks += [(B == instance_id).astype(np.uint8)]

    return instance_masks, instance_ids


def visualize_ade20k_class_masks(d_api, classname) -> None:
    """
    Dump one image per mask, for all masks of a specific class.
    """
    required_class_names = [classname]
    highlight_classname = classname
    condition = "intersection"
    folder_prefix = f"ade20k_{classname}"
    d_api.dump_class_masks(
        required_class_names=required_class_names,
        highlight_classname=highlight_classname,
        condition=condition,
        folder_prefix=folder_prefix,
    )


def main() -> None:
    """
	Visualize masks of a chosen category from COCO Panoptic.

	Interesting choices:
		'floor','curtain','mountain','rug','hill'
		'light', 'lamp', 'chandelier', 'sconce'
		'chair', 'armchair', 'swivel chair', 'stool','seat'
		'table', 'chest of drawers', 'plate', 'food'
		'shelf','base','fountain','screen door','tower',
		'hovel','booth','building','runway','skyscraper'
		'house','booth','hovel','tent','plaything','grandstand'
		'flower','pillow', 'cushion','buffet','grandstand',
		'screen door','pier','car','van','path','awning'

	Usage:
	python Ade20kMaskLevelDataset.py \
	--semantic_dataroot /Users/johnlamb/Downloads/ADE20K/ADEChallengeData2016/images
	--instance_dataroot /Users/johnlamb/Downloads/ADE20K_2016_07_26/images
	 --classname rug
	"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--classname", type=str, required=True, help="name of class to visualize")
    parser.add_argument("--instance_dataroot", type=str, required=True, help="path to ADE20K instance data root")
    parser.add_argument("--semantic_dataroot", type=str, required=True, help="path to ADE20K challenge data root")

    args = parser.parse_args()
    d_api = Ade20kMaskDataset(args.semantic_dataroot, args.instance_dataroot)
    visualize_ade20k_class_masks(d_api, args.classname)


if __name__ == "__main__":
    main()
