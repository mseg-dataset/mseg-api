#!/usr/bin/python3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from mseg.utils.names_utils import load_class_names

# -------------------------------------------------------------
# You must specify the parent directory of `mseg_dataset` below:
MSEG_DST_DIR = ""
# -------------------------------------------------------------
if MSEG_DST_DIR == "":
    raise Exception("You must populate the `MSEG_DST_DIR` variable in mseg/utils/dataset_config.py")


ROOT = Path(__file__).resolve().parent.parent  # ../..

@dataclass
class DatasetInfo:
    """Contains information about a particular dataset.

    Args:
        name: nickname for the dataset.
        dataroot: str
        trainlist: path to .txt file containing file paths for the train split.
        vallist: path to .txt file containing file paths for the validation split.
        vallist_small: path to .txt file containing file paths for a subsampled version of the validation split.
        names_path: path to .txt file containing ordered class names for the dataset.
        shortname: 
        num_classes: number of classes in the dataset.
    """
    name: str
    dataroot: str
    trainlist: Optional[str] = None
    vallist: Optional[str] = None
    vallist_small: Optional[str] = None
    names_path: Optional[str] = None
    shortname: Optional[str] = None
    num_classes: Optional[str] = None


infos = [
    DatasetInfo("ade20k-151-inst", f"{MSEG_DST_DIR}/mseg_dataset/ADE20K/ADE20K_2016_07_26"),
    DatasetInfo("ade20k-151", f"{MSEG_DST_DIR}/mseg_dataset/ADE20K/ADEChallengeData2016"),
    DatasetInfo("ade20k-150", f"{MSEG_DST_DIR}/mseg_dataset/ADE20K/ADEChallengeData2016"),
    DatasetInfo("ade20k-150-relabeled", f"{MSEG_DST_DIR}/mseg_dataset/ADE20K/ADEChallengeData2016"),

    DatasetInfo("bdd", f"{MSEG_DST_DIR}/mseg_dataset/BDD/bdd100k"),
    DatasetInfo("bdd-relabeled", f"{MSEG_DST_DIR}/mseg_dataset/BDD/bdd100k"),

    DatasetInfo("camvid-11", f"{MSEG_DST_DIR}/mseg_dataset/Camvid"),

    DatasetInfo("cityscapes-34", f"{MSEG_DST_DIR}/mseg_dataset/Cityscapes"),
    DatasetInfo("cityscapes-19", f"{MSEG_DST_DIR}/mseg_dataset/Cityscapes"),
    DatasetInfo("cityscapes-19-relabeled", f"{MSEG_DST_DIR}/mseg_dataset/Cityscapes"),
    DatasetInfo("cityscapes-34-relabeled", f"{MSEG_DST_DIR}/mseg_dataset/Cityscapes"),

    DatasetInfo("coco-panoptic-inst-201", f"{MSEG_DST_DIR}/mseg_dataset/COCOPanoptic"),
    DatasetInfo("coco-panoptic-201", f"{MSEG_DST_DIR}/mseg_dataset/COCOPanoptic"),
    DatasetInfo("coco-panoptic-133", f"{MSEG_DST_DIR}/mseg_dataset/COCOPanoptic"),
    DatasetInfo("coco-panoptic-133-relabeled", f"{MSEG_DST_DIR}/mseg_dataset/COCOPanoptic"),

    DatasetInfo("idd-40", f"{MSEG_DST_DIR}/mseg_dataset/IDD/IDD_Segmentation"),
    DatasetInfo("idd-39", f"{MSEG_DST_DIR}/mseg_dataset/IDD/IDD_Segmentation"),
    DatasetInfo("idd-39-relabeled", f"{MSEG_DST_DIR}/mseg_dataset/IDD/IDD_Segmentation"),

    DatasetInfo("kitti-34", f"{MSEG_DST_DIR}/mseg_dataset/KITTI/"),
    DatasetInfo("kitti-19", f"{MSEG_DST_DIR}/mseg_dataset/KITTI/"),

    #  mapillary-public66 -> labels are not in semseg format (RGB labels)
    DatasetInfo("mapillary-public66", f"{MSEG_DST_DIR}/mseg_dataset/MapillaryVistasPublic"),
    DatasetInfo("mapillary-public65", f"{MSEG_DST_DIR}/mseg_dataset/MapillaryVistasPublic"),
    DatasetInfo("mapillary-public65-relabeled", f"{MSEG_DST_DIR}/mseg_dataset/MapillaryVistasPublic"),

    DatasetInfo("pascal-context-460", f"{MSEG_DST_DIR}/mseg_dataset/PASCAL_Context"),
    DatasetInfo("pascal-context-60", f"{MSEG_DST_DIR}/mseg_dataset/PASCAL_Context"),

    DatasetInfo("scannet-41", f"{MSEG_DST_DIR}/mseg_dataset/ScanNet/scannet_frames_25k"),
    DatasetInfo("scannet-20", f"{MSEG_DST_DIR}/mseg_dataset/ScanNet/scannet_frames_25k"),

    DatasetInfo("sunrgbd-38", f"{MSEG_DST_DIR}/mseg_dataset/SUNRGBD"),
    DatasetInfo("sunrgbd-37", f"{MSEG_DST_DIR}/mseg_dataset/SUNRGBD"),
    DatasetInfo("sunrgbd-37-relabeled", f"{MSEG_DST_DIR}/mseg_dataset/SUNRGBD"),

    DatasetInfo("voc2012", f"{MSEG_DST_DIR}/mseg_dataset/PASCAL_VOC_2012"),

    DatasetInfo("wilddash-19", f"{MSEG_DST_DIR}/mseg_dataset/WildDash"),
]

# Dictionary with concise metadata object for each dataset
infos = {info.name: info for info in infos}

for name, info in infos.items():

    if name.endswith("-sr"):
        folder_name = name.replace("-sr", "")
    else:
        folder_name = name

    info.trainlist = f"{ROOT}/dataset_lists/{folder_name}/list/train.txt"
    info.vallist = f"{ROOT}/dataset_lists/{folder_name}/list/val.txt"
    info.names_path = f"{ROOT}/dataset_lists/{folder_name}/{info.name}_names.txt"
    info.vallist_small = f"{ROOT}/dataset_lists/{folder_name}/list/val_small.txt"
    info.shortname = info.name
    if "inst" not in name:
        info.num_classes = len(load_class_names(folder_name))


v2gids = {4.0: "886850936"}
