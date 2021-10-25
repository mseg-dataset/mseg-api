#!/usr/bin/python3


import argparse
from argparse import Namespace
import imageio
import numpy as np
import pdb

from typing import List, Mapping

from mseg.utils.names_utils import get_dataloader_id_to_classname_map
from mseg.utils.txt_utils import generate_all_img_label_pair_fpaths
from mseg.utils.json_utils import save_json_dict
from mseg.utils.dir_utils import check_mkdir
from mseg.taxonomy.taxonomy_converter import TaxonomyConverter

from mseg_semantic.utils import transform


def get_abs_fpaths(args, split: str) -> List[str]:
	"""
		Args:
		-	args:
		-	split: string representing dataset split, e.g. 'train' or 'val'

		Returns:
		-	label_abs_fpaths: List of strings, each representing an absolute
				path to a label map stored on disk.
	"""
	split_txt_fpath = f'/srv/scratch/jlambert30/MSeg/mseg-api/mseg/dataset_lists/{args.dataset_name}/list/{split}.txt'
	pairs = generate_all_img_label_pair_fpaths(args.data_root, split_txt_fpath)
	label_abs_fpaths = list(zip(*pairs))[1]
	return label_abs_fpaths


def print_cls_densities(
	cls_pixel_counts: np.ndarray, 
	dataloaderid_to_classname_map: Mapping[int, str], 
	num_classes: int, 
	skip_zero_entries: bool = False
) -> None:
	"""
	Dump to stdout a brief summary of which classes are most common 
	(percentages are printed).

		Args:
		-	cls_pixel_counts: 
		-	dataloaderid_to_classname_map
		-	num_classes
		-	skip_zero_entries:

		Returns:
		-	None
	"""
	# get max string length
	classnames = dataloaderid_to_classname_map.values()
	max_classname_len = max([ len(name) for name in classnames])
	
	total_num_px = cls_pixel_counts.sum()
	nonzero_indices = np.nonzero(cls_pixel_counts)[0]

	# `argsort` sorts in increasing order, but we want decreasing order
	for dataloaderid in np.argsort(-cls_pixel_counts):

		# since we go from 0-255, ignore everything without valid corresponding class
		if dataloaderid not in dataloaderid_to_classname_map:
			continue

		classname = dataloaderid_to_classname_map[dataloaderid]

		# if skip_zero_entries:
		# 	if dataloaderid not in nonzero_indices:
		# 		continue

		px_percent = cls_pixel_counts[dataloaderid] / total_num_px * 100
		print(f'\t{classname.rjust(max_classname_len)} \t {px_percent:.2f} %')


def analyze_class_pixel_distribution(args: Namespace, print_every_n_imgs: int = 400) -> None:
	"""
		Examine the densities of pixel annotations in any dataset. We loop over every
		label image and load it from disk. If wishing to count the contribution
		to the universal taxonomy statistics, we use the TaxonomyConverter class
		for remapping.

		Args:
		-	args: argparse Namespace object

		Returns:
		-	None
	"""
	if args.output_taxonomy == 'universal':
		tc = TaxonomyConverter()
		dataloaderid_to_classname_map = tc.uid2uname
		transform_list = [
			transform.ToTensor(),
			transform.ToUniversalLabel(args.dataset_name)
		]
		transform_op = transform.Compose(transform_list)

	else:
		dataloaderid_to_classname_map = get_dataloader_id_to_classname_map(args.dataset_name, include_ignore_idx_cls=True)

	num_classes = len(dataloaderid_to_classname_map.keys())
	max_class_index = max(list(dataloaderid_to_classname_map.keys()))

	for split in ['train', 'val']:

		# allow us to index into the array at any dataloaderid value
		cls_pixel_counts = np.zeros(max_class_index + 1)
		label_fpaths = get_abs_fpaths(args, split)
		num_split_imgs = len(label_fpaths)

		for j, label_fpath in enumerate(label_fpaths):
			label_img = imageio.imread(label_fpath)

			# remap to universal if needed
			if args.output_taxonomy == 'universal':
				label_img = label_img.astype(np.int64)
				h, w = label_img.shape
				rgb_img = np.zeros((h,w,3), dtype=np.uint8) # create dummy RGB image
				_, label_img = transform_op(rgb_img, label_img)

			# increments counts for each class present in this label map.
			for sem_cls in np.unique(label_img):
				cls_pixel_counts[sem_cls] += (label_img == sem_cls).sum()
			if j%print_every_n_imgs == 0:
				print(f'Processing label {j}/{num_split_imgs} in split {split}')
				print_cls_densities(cls_pixel_counts, dataloaderid_to_classname_map, num_classes)
				print()

		# save the counts for each classname
		json_dict = {}
		for id,name in dataloaderid_to_classname_map.items():
			# should all be integer-valued
			json_dict[name] = int(cls_pixel_counts[id])

		# save results to a json file
		save_dir = 'pixel_counts-universal-relabeled-2020_07_29'
		check_mkdir(save_dir)
		save_json_dict(f'{save_dir}/{args.dataset_name}_{split}.json', json_dict)


if __name__ == '__main__':
	"""
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset_name', type=str, required=True,
		help='Name of dataset.')

	parser.add_argument('--data_root', type=str, required=True,
		help='Path to dataset files (where rel paths begin).')

	parser.add_argument('--output_taxonomy', type=str, required=True,
		help='Which taxonomy to record statistics in.')

	args = parser.parse_args()
	analyze_class_pixel_distribution(args)




