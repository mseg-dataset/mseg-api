#!/usr/bin/python3

import argparse
import glob
from pathlib import Path
from typing import Any, List, Optional, Tuple

import imageio
import numpy as np

from mseg.utils.json_utils import read_json_file
from mseg.utils.mask_utils import get_mask_from_polygon, save_binary_mask_double

"""
Note: We do not use this dataset API at training or inference time.
It is designed purely for generating the re-labeled masks of the 
MSeg dataset (found in ground truth label maps) on disk, prior to 
training/inference.
"""


class JsonMaskDataset:
    """
    Simple API to access masks of Cityscapes and IDD. Format is
    shared almost exactly between Cityscapes and IDD, except
    IDD sequence IDs are integers, where Cityscapes uses city
    names as sequence IDs.
    """

    def __init__(self, dataroot: str) -> None:
        """ """
        self.dataroot = dataroot
        assert Path(f"{self.dataroot}/gtFine").exists()
        assert Path(f"{self.dataroot}/leftImg8bit").exists()

    def write_class_masks(
        self, required_class_names: List[str], highlight_classname: str, condition: str, folder_prefix: str
    ):
        """
        Each path resembles:
        gt_fpath = '/export/work/johnlamb/MTURK_IDD_COPY/IDD_Segmentation/gtFine/val/119/638495_gtFine_polygons.json'

        """
        for split in ["train", "val"]:
            gt_fpaths = glob.glob(f"{self.dataroot}/gtFine/{split}/**/*.json", recursive=True)
            num_split_imgs = len(gt_fpaths)
            for i, gt_fpath in enumerate(gt_fpaths):
                print(f"On image {i} / {num_split_imgs-1}")

                img_gt_dict = read_json_file(gt_fpath)
                img_gt_objs = img_gt_dict["objects"]
                img_h = img_gt_dict["imgHeight"]
                img_w = img_gt_dict["imgWidth"]

                present_classnames = [gt_obj["label"] for gt_obj in img_gt_objs]
                if highlight_classname not in present_classnames:
                    print(f"\tSkip {i}")
                    continue

                seq_id = Path(gt_fpath).parts[-2]
                fname_stem = Path(gt_fpath).stem
                rgb_img = self.get_rgb_img(seq_id, fname_stem, split)

                segment_ids = set()
                for obj_idx, gt_obj in enumerate(img_gt_objs):

                    if "id" in gt_obj.keys():
                        segment_id = gt_obj["id"]
                    else:
                        segment_id = obj_idx

                    # each segment ID must be unique for this image.
                    assert segment_id not in segment_ids
                    segment_ids.add(segment_id)
                    obj_label = gt_obj["label"]
                    if obj_label == highlight_classname:
                        obj_poly = gt_obj["polygon"]
                        # first arg is tuple_verts: Iteratable with elements of size 2 (list of 2-tuples, or Nx2 numpy array)
                        obj_mask = get_mask_from_polygon(obj_poly, img_h, img_w)
                        obj_mask = obj_mask.astype(np.uint8)
                        # not adding {obj_idx} to filename path now.
                        save_fpath = (
                            f"temp_files/{folder_prefix}_{split}_2020_03_10/seq{seq_id}_{fname_stem}_{segment_id}.jpg"
                        )
                        save_binary_mask_double(rgb_img, obj_mask, save_fpath, save_to_disk=True)

    def is_cityscapes(self, seq_id) -> bool:
        """ """
        try:
            int(seq_id)
            return False  # is IDD sequence
        except:
            return True

    def get_rgb_img(self, seq_id: int, fname_stem: str, split: str) -> np.ndarray:
        """
        e.g. leftImg8bit/train/0/022141_leftImg8bit.png

        e.g. leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png

        Args:
            seq_id:
            fname_stem:
            split:

        Returns:
            rgb_img: array of shape (H,W,3) representing RGB image.
        """
        fname_stem = fname_stem.replace("_gtFine_polygons", "")
        rgb_fpath = f"{self.dataroot}/leftImg8bit/{split}/{seq_id}/{fname_stem}_leftImg8bit.png"
        rgb_img = imageio.imread(rgb_fpath)
        return rgb_img

    def get_segment_mask(self, seq_id: str, query_segmentid: int, fname_stem: str, split: str) -> Optional[np.ndarray]:
        """
        Uses the raw version of the dataset (as originally distributed) to get a mask.

        Args:
            seq_id: sequence ID, parent of file name in file system tree.
            query_segmentid:
            fname_stem: RGB image file name stem, e.g. 'aachen_000015_000019_leftImg8bit'
            split: 'train' or 'val'

        Returns:
            obj_mask: Numpy array of shape (H,W) of type uint8, should consist of 0s and 1s
        """
        label_fname_stem = fname_stem.replace("_leftImg8bit", "")
        gt_fpath = f"{self.dataroot}/gtFine/{split}/{seq_id}/{label_fname_stem}_gtFine_polygons.json"

        img_gt_dict = read_json_file(gt_fpath)
        img_gt_objs = img_gt_dict["objects"]
        img_h = img_gt_dict["imgHeight"]
        img_w = img_gt_dict["imgWidth"]
        for obj_idx, gt_obj in enumerate(img_gt_objs):

            if "id" in gt_obj.keys():
                segment_id = gt_obj["id"]
            else:
                segment_id = obj_idx
            if segment_id == query_segmentid:
                obj_poly = gt_obj["polygon"]
                # first arg is tuple_verts: Iterable with elements of size 2 (list of 2-tuples, or Nx2 numpy array)
                obj_mask = get_mask_from_polygon(obj_poly, img_h, img_w)
                obj_mask = obj_mask.astype(np.uint8)
                return obj_mask

        return None


if __name__ == "__main__":
    """
    e.g.
    python JsonMaskLevelDataset.py --dataroot /export/share/Datasets/MSegV4/mseg_dataset/Cityscapes
    python JsonMaskLevelDataset.py --dataroot /export/share/Datasets/MSegV4/mseg_dataset/IDD/IDD_Segmentation

    --folder_prefix cityscapes_rider_shatter --classname rider
    --folder_prefix cityscapes_person_shatter --classname person
    --folder_prefix idd_rider_shatter --classname rider
    --folder_prefix idd_person_shatter --classname person
    --folder_prefix idd_motorcycle --classname motorcycle
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--classname", type=str, required=True, help="name of class to visualize/highlight")
    parser.add_argument("--dataroot", type=str, required=True, help="path to IDD or Cityscapes data root")

    parser.add_argument("--folder_prefix", type=str, required=True, help="name of folder to save images in")

    args = parser.parse_args()
    print(args)
    api = JsonMaskDataset(args.dataroot)

    condition = "intersection"
    highlight_classname = args.classname
    required_class_names = [highlight_classname]
    api.write_class_masks(required_class_names, highlight_classname, condition, args.folder_prefix)
