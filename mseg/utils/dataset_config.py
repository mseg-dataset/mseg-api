#!/usr/bin/python3

import numpy as np
from pathlib import Path
import pdb
from recordclass import recordclass

from mseg.utils.names_utils import load_class_names

# -------------------------------------------------------------
# You must specify the parent directory of `mseg_dataset` below:
MSEG_DST_DIR = ""
# -------------------------------------------------------------
if MSEG_DST_DIR == '':
	raise Exception(
		"You must populate the `MSEG_DST_DIR` variable in mseg/utils/dataset_config.py"
	)


ROOT = Path(__file__).resolve().parent.parent  # ../..

fields = ('name', 'dataroot', 'trainlist', 'vallist', 'vallist_small', 'names_path', 'shortname', 'num_classes', 'trainlen')
# recordclass is the mutable analog of collections.namedtuple
info = recordclass('info', fields, defaults=(None,) * len(fields))

infos = [
	info('ade20k-151-inst', f'{MSEG_DST_DIR}/mseg_dataset/ADE20K/ADE20K_2016_07_26'),
	info('ade20k-151', f'{MSEG_DST_DIR}/mseg_dataset/ADE20K/ADEChallengeData2016'),
	info('ade20k-150', f'{MSEG_DST_DIR}/mseg_dataset/ADE20K/ADEChallengeData2016'),
	info('ade20k-150-relabeled', f'{MSEG_DST_DIR}/mseg_dataset/ADE20K/ADEChallengeData2016'),

	info('bdd', f'{MSEG_DST_DIR}/mseg_dataset/BDD/bdd100k'),
	info('bdd-relabeled', f'{MSEG_DST_DIR}/mseg_dataset/BDD/bdd100k'),
	
	info('camvid-11', f'{MSEG_DST_DIR}/mseg_dataset/Camvid'),

	info('cityscapes-34', f'{MSEG_DST_DIR}/mseg_dataset/Cityscapes'),
	info('cityscapes-19', f'{MSEG_DST_DIR}/mseg_dataset/Cityscapes'),
	info('cityscapes-19-relabeled', f'{MSEG_DST_DIR}/mseg_dataset/Cityscapes'),
	info('cityscapes-34-relabeled', f'{MSEG_DST_DIR}/mseg_dataset/Cityscapes'),

	info('coco-panoptic-inst-201', f'{MSEG_DST_DIR}/mseg_dataset/COCOPanoptic'),
	info('coco-panoptic-201', f'{MSEG_DST_DIR}/mseg_dataset/COCOPanoptic'),
	info('coco-panoptic-133', f'{MSEG_DST_DIR}/mseg_dataset/COCOPanoptic'),
	info('coco-panoptic-133-relabeled', f'{MSEG_DST_DIR}/mseg_dataset/COCOPanoptic'),

	info('idd-40', f'{MSEG_DST_DIR}/mseg_dataset/IDD/IDD_Segmentation'),
	info('idd-39', f'{MSEG_DST_DIR}/mseg_dataset/IDD/IDD_Segmentation'),
	info('idd-39-relabeled', f'{MSEG_DST_DIR}/mseg_dataset/IDD/IDD_Segmentation'),

	info('kitti-34', f'{MSEG_DST_DIR}/mseg_dataset/KITTI/'),
	info('kitti-19', f'{MSEG_DST_DIR}/mseg_dataset/KITTI/'),

	#  mapillary-public66 -> labels are not in semseg format (RGB labels)
	info('mapillary-public66', f'{MSEG_DST_DIR}/mseg_dataset/MapillaryVistasPublic'),
	info('mapillary-public65', f'{MSEG_DST_DIR}/mseg_dataset/MapillaryVistasPublic'),
	info('mapillary-public65-relabeled', f'{MSEG_DST_DIR}/mseg_dataset/MapillaryVistasPublic'),
	
	info('pascal-context-460', f'{MSEG_DST_DIR}/mseg_dataset/PASCAL_Context'),
	info('pascal-context-60', f'{MSEG_DST_DIR}/mseg_dataset/PASCAL_Context'),

	info('scannet-41', f'{MSEG_DST_DIR}/mseg_dataset/ScanNet/scannet_frames_25k'),
	info('scannet-20', f'{MSEG_DST_DIR}/mseg_dataset/ScanNet/scannet_frames_25k'),

	info('sunrgbd-38', f'{MSEG_DST_DIR}/mseg_dataset/SUNRGBD'),
	info('sunrgbd-37', f'{MSEG_DST_DIR}/mseg_dataset/SUNRGBD'),
	info('sunrgbd-37-relabeled', f'{MSEG_DST_DIR}/mseg_dataset/SUNRGBD'),

	info('voc2012', f'{MSEG_DST_DIR}/mseg_dataset/PASCAL_VOC_2012'),

	info('wilddash-19', f'{MSEG_DST_DIR}/mseg_dataset/WildDash'),
]

# Dictionary with concise metadata object for each dataset
infos = {info.name : info for info in infos}

for name, info in infos.items():

	if name.endswith('-sr'):
		folder_name = name.replace('-sr', '')
	else:
		folder_name = name

	info.trainlist = f'{ROOT}/dataset_lists/{folder_name}/list/train.txt'
	info.vallist = f'{ROOT}/dataset_lists/{folder_name}/list/val.txt'
	info.names_path = f'{ROOT}/dataset_lists/{folder_name}/{info.name}_names.txt'
	info.vallist_small = f'{ROOT}/dataset_lists/{folder_name}/list/val_small.txt'
	info.shortname = info.name
	if 'inst' not in name:
		info.num_classes = len(load_class_names(folder_name))


v2gids = {
	4.0: '886850936'
}

