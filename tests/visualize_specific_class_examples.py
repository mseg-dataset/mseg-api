#!/usr/bin/python3

import imageio
import math
import numpy as np
from pathlib import Path
import pdb

from mseg.utils.cv2_utils import (
	grayscale_to_color,
	cv2_imread_rgb
)
from mseg.utils.dir_utils import check_mkdir
from mseg.utils.dataset_config import infos
from mseg.utils.names_utils import (
	get_classname_to_dataloaderid_map,
	get_dataloader_id_to_classname_map
)
from mseg.utils.mask_utils import (
	form_mask_triple_embedded_classnames,
	save_classnames_in_image_maxcardinality,
	save_mask_triple_isolated_mask
)
from mseg.utils.mask_utils_detectron2 import Visualizer
from mseg.utils.txt_utils import generate_all_img_label_pair_fpaths
from mseg.label_preparation.mseg_write_relabeled_segments import get_unique_mask_identifiers


_ROOT = Path(__file__).resolve().parent.parent

NUM_EXAMPLES_PER_CLASS = 40

def save_class_examples(dataset_name: str):
	"""
	Save every 1000th image of each dataset, with classnames embedded.
	"""
	save_dir = f'temp_files/{dataset_name}'

	for dname, d_info in infos.items():
		print(f'Writing visual sanity checks for {dname}...')

		if dataset_name not in dname:
			continue

		id_to_classname_map = get_dataloader_id_to_classname_map(dname)
		
		classnames = id_to_classname_map.values()

		splits = ['train', 'val']
		split_lists = [d_info.trainlist, d_info.vallist]

		for split,split_list in zip(splits,split_lists):
			pairs = generate_all_img_label_pair_fpaths(d_info.dataroot, split_list)
			
			for classname in classnames:
				count = 0

				# Save 5 examples from each dataset split
				for i, (rgb_fpath, label_fpath) in enumerate(pairs):
					print(f'On {i} of {dname}')

					label_img = imageio.imread(label_fpath)

					present_classes = [id_to_classname_map[label_id] for label_id in np.unique(label_img)]
					if classname not in present_classes:
						continue

					count += 1

					rgb_img = cv2_imread_rgb(rgb_fpath)
					fname_stem = Path(rgb_fpath).stem
					save_fpath = f'{save_dir}/{dname}_{fname_stem}.jpg'
					blend_save_fpath = f'{save_dir}/{classname}/{dname}_{fname_stem}_blended.jpg'
					check_mkdir(f'{save_dir}/{classname}')

					if rgb_img.ndim == 2:
						# this image was grayscale
						rgb_img = grayscale_to_color(rgb_img)
				
					isolated_save_fpath = f'{save_dir}/{classname}/{dname}_{fname_stem}_isolated.jpg'
					save_mask_triple_isolated_mask(
						rgb_img,
						label_img,
						id_to_classname_map,
						classname,
						isolated_save_fpath
					)
					frame_visualizer = Visualizer(rgb_img, metadata=None)
					output_img = frame_visualizer.overlay_instances(
						label_map=label_img,
						id_to_class_name_map=id_to_classname_map
					)
					imageio.imwrite(blend_save_fpath, output_img)

					if count >= NUM_EXAMPLES_PER_CLASS:
						break



if __name__ == '__main__':
	dataset_name = 'ade20k-150'
	save_class_examples(dataset_name)

	
