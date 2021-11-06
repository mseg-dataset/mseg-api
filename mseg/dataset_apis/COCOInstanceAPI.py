#!/usr/bin/python3

import glob
from pathlib import Path
from typing import List

import imageio
import numpy as np


"""
Interface for instance labels of COCO Panoptic dataset

Note: We do not use this dataset API at training or inference time.
It is designed purely for generating the re-labeled masks of the 
MSeg dataset (found in ground truth label maps) on disk, prior to 
training/inference.
"""


class COCOInstanceAPI:
    def __init__(self, coco_dataroot: str) -> None:
        """
        Args:
            coco_dataroot: path to unzipped COCO Panoptic directory
        """
        self.annotations_root = f"{coco_dataroot}/annotations"

        # map split -> instance image fpaths
        self.instance_img_fpaths_splitdict = {}

        # fname -> instance image fpath
        self.fname_to_instanceimgfpath_dict = {}
        for split in ["train", "val"]:
            instance_img_fpaths = self.get_instance_annotations(split)
            self.instance_img_fpaths_splitdict[split] = instance_img_fpaths

            # Make it easy to find the path to the 3-channel label imgs that store instance info
            for instance_img_fpath in instance_img_fpaths:
                fname_stem = Path(instance_img_fpath).stem
                self.fname_to_instanceimgfpath_dict[fname_stem] = instance_img_fpath

    def get_instance_annotations(self, split: str) -> List[str]:
        """
        Get COCO Panoptic instance annotations from .pngs

        Return a list of filepaths to the instance ID images

        Args:
            split: string representing training, validation, or testing split of the data

        Returns:
            label_img_fpaths:
            filename_to_annot_map:
        """
        instance_img_fpaths = glob.glob(f"{self.annotations_root}/panoptic_{split}2017/*.png")
        instance_img_fpaths.sort()
        return instance_img_fpaths

    def get_instance_img_fpaths(self, split: str):
        """ """
        return self.instance_img_fpaths_splitdict[split]

    def get_instance_id_img(self, split: str, fname_stem: str) -> np.ndarray:
        """
            Encoding described here:
            https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py#L30

        "Given semantic category unique ID will be generated and its RGB encoding will
        have color close to the predefined semantic category color.
        The RGB encoding used is ID = R * 256 * G + 256 * 256 + B."

        Args:
            label_img_fpath

        Returns:
            rgb_img: color image.
            label_img: category ID image
            ids: instance ID image
        """
        # get the path to the 3-channel instance id image
        instance_img_fpath = self.fname_to_instanceimgfpath_dict[fname_stem]
        RGB_inst_img = imageio.imread(instance_img_fpath)
        R = RGB_inst_img[:, :, 0]
        G = RGB_inst_img[:, :, 1]
        B = RGB_inst_img[:, :, 2]
        instance_id_img = R + (G * 256) + (B * 256 ** 2)

        return instance_id_img
