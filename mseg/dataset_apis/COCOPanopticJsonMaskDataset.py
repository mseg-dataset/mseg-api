#!/usr/bin/python3

import argparse
import glob
import imageio
import numpy as np
from pathlib import Path
import pdb
from typing import List

import mseg.utils.names_utils as names_utils
from mseg.utils.mask_utils import save_binary_mask_double
from mseg.utils.cv2_utils import grayscale_to_color
from mseg.dataset_apis.COCOSemanticAPI import COCOSemanticAPI
from mseg.dataset_apis.COCOInstanceAPI import COCOInstanceAPI

"""
Note: We do not use this dataset API at training or inference time.
It is designed purely for generating the re-labeled masks of the 
MSeg dataset (found in ground truth label maps) on disk, prior to 
training/inference.
"""


class COCOPanopticJsonMaskDataset:
    """
    Simple API to interact with instance masks of COCO Panoptic dataset.

    We take the instance ID images from the Panoptic .png files, and we take
    the semantic class images from the Panoptic .json files
    """

    def __init__(self, coco_dataroot: str) -> None:
        """ """
        self.categoryid_to_classname_map = names_utils.get_dataloader_id_to_classname_map(dataset_name="coco-panoptic-201")
        self.coco_dataroot = coco_dataroot
        self.semantic_api = COCOSemanticAPI(coco_dataroot)
        self.instance_api = COCOInstanceAPI(coco_dataroot)

    def dump_class_masks(
        self, required_class_names: List[str], highlight_classname: str, condition: str, folder_prefix: str
    ) -> None:
        """
        Write out all requested COCO masks to disk. Get for all splits, at once.
        If person-car combinations are desired, this tuple may be specified
        as a requirement.

        Args:
            required_class_names:
            highlight_classname:
            condition:
            folder_prefix:
        """
        for split in ["train", "val"]:
            instance_img_fpaths = self.instance_api.get_instance_img_fpaths(split)
            num_imgs = len(instance_img_fpaths)

            for i, instance_img_fpath in enumerate(instance_img_fpaths):
                print(f"On image {i} of {num_imgs-1}")

                fname_stem = Path(instance_img_fpath).stem
                rgb_img = self.get_rgb_img(split, fname_stem)
                instance_id_img = self.instance_api.get_instance_id_img(split, fname_stem)
                segmentid_to_class_name_map = {0: "unlabeled"}

                present_classnames = self.semantic_api.get_present_classes_in_img(split, fname_stem)
                print(present_classnames)
                if not all([req_name in present_classnames for req_name in required_class_names]):
                    continue

                img_annot = self.semantic_api.get_img_annotation(split, fname_stem)
                for segment in img_annot["segments_info"]:
                    segmentid = segment["id"]
                    categoryid = segment["category_id"]
                    instance_classname = self.categoryid_to_classname_map[categoryid]
                    segmentid_to_class_name_map[segmentid] = instance_classname

                    if instance_classname != highlight_classname:
                        continue

                    label_mask = (instance_id_img == segmentid).astype(np.uint8)
                    save_fpath = f"temp_files/{folder_prefix}_{split}/{fname_stem}_{segmentid}.png"
                    save_binary_mask_double(rgb_img, label_mask, save_fpath, save_to_disk=True)

    def get_rgb_img(self, split: str, fname_stem: str) -> np.ndarray:
        """
        Args:
            split: string representing training, validation, or testing split of the data
            fname_stem:

        Returns:
            rgb_img: color image.
        """
        rgb_img_fpath = f"{self.coco_dataroot}/{split}2017/{fname_stem}.jpg"
        rgb_img = imageio.imread(rgb_img_fpath)
        if rgb_img.ndim == 2:
            # this image was grayscale
            rgb_img = grayscale_to_color(rgb_img)
        return rgb_img

    def get_segment_mask(self, seq_id: None, segmentid: int, fname_stem: str, split: str) -> np.ndarray:
        """
        Use semantic and instance APIs to identify

        Args:
            segmentid:
            fname_stem:
            split:

        Returns:
            segment_mask:
        """
        ids = self.instance_api.get_instance_id_img(split, fname_stem)
        img_annot = self.semantic_api.get_img_annotation(split, fname_stem)

        for segment in img_annot["segments_info"]:
            if segmentid == segment["id"]:
                categoryid = segment["category_id"]
                segment_mask = (ids == segmentid).astype(np.uint8)

        return segment_mask


def visualize_coco_class_masks(d_api, classname):
    """
    Dump one image per mask, for all masks of a specific class.
    """
    required_class_names = [classname]
    highlight_classname = classname
    condition = "intersection"
    folder_prefix = f"cocopanoptic_{classname}"
    d_api.dump_class_masks(
        required_class_names=required_class_names,
        highlight_classname=highlight_classname,
        condition=condition,
        folder_prefix=folder_prefix,
    )


def main():
    """
	Visualize masks of a chosen category from COCO Panoptic.

	Interesting choices:
		'cabinet-merged', 'rug-merged', 'road', 'bus', 'truck', 
		'cup','water-other','sea','river','car','airplane'
		'grass-merged', 'person', 'tent', 'laptop','pavement-merged'
		'playingfield', 'window-other','car','motorcycle',
		'blanket', 'water-other','keyboard','door-stuff',
		'mountain-merged','building-other-merged','teddy bear'
		'tent','playingfield','bridge','platform','house'
		'roof','car','wall-other-merged','wall-brick','wall-stone',
		'wall-tile','wall-wood','counter','tent','dining table'
		'light','chair','table-merged', 'keyboard'

	Usage:
	python COCOPanopticJsonMaskDataset.py \
	--dataroot /Users/johnlamb/Downloads/COCO-Panoptic --classname sea
	"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--classname", type=str, required=True, help="name of class to visualize")
    parser.add_argument("--dataroot", type=str, required=True, help="path to COCO Panoptic data root")
    args = parser.parse_args()

    d_api = COCOPanopticJsonMaskDataset(args.dataroot)
    visualize_coco_class_masks(d_api, args.classname)


if __name__ == "__main__":
    main()
