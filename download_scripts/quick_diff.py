	

import imageio
import numpy as np
import pdb
from pathlib import Path

from mseg.utils.mask_utils import (
	form_mask_triple_embedded_classnames,
	save_binary_mask_double,
	save_classnames_in_image_maxcardinality,
	save_mask_triple_with_color_guide
)
from mseg.utils.names_utils import (
	get_classname_to_dataloaderid_map,
	get_dataloader_id_to_classname_map
)
from mseg.utils.txt_utils import generate_all_img_label_pair_fpaths

from mseg.utils.dataset_config import infos
from mseg.utils.cv2_utils import grayscale_to_color



def main():
	"""
	"""
	save_dir = 'temp_files'

	# beforehand
	# orig_dname = 'cityscapes-19'
	# orig_dataroot = '/export/share/Datasets/cityscapes'

	# rel_dname = 'cityscapes-19'
	# rel_dataroot = '/export/share/Datasets/MSeg/mseg_dataset/Cityscapes'

	orig_dname = 'coco-panoptic-133'
	orig_dname_lists = 'coco-panoptic-133-temp'
	orig_dataroot = '/export/share/Datasets/COCO2017'

	rel_dname = 'coco-panoptic-133'
	rel_dataroot = '/export/share/Datasets/MSeg/mseg_dataset/COCOPanoptic'

	orig_id_to_classname_map = get_dataloader_id_to_classname_map(orig_dname)
	rel_id_to_classname_map = get_dataloader_id_to_classname_map(rel_dname)

	for split in ['train', 'val']:
		orig_split_txt_fpath = f'../mseg/dataset_lists/{orig_dname_lists}/list/{split}.txt'
		orig_pairs = generate_all_img_label_pair_fpaths(orig_dataroot, orig_split_txt_fpath)

		rel_split_txt_fpath = f'../mseg/dataset_lists/{rel_dname}/list/{split}.txt'
		rel_pairs = generate_all_img_label_pair_fpaths(rel_dataroot, rel_split_txt_fpath)

		for i in range(len(orig_pairs))[::100]:
			orig_pair = orig_pairs[i]
			orig_rgb_fpath, orig_label_fpath = orig_pair
			orig_rgb_img = imageio.imread(orig_rgb_fpath)
			orig_label_img = imageio.imread(orig_label_fpath)

			rel_pair = rel_pairs[i]
			rel_rgb_fpath, rel_label_fpath = rel_pair
			rel_rgb_img = imageio.imread(rel_rgb_fpath)
			rel_label_img = imageio.imread(rel_label_fpath)

			if not np.allclose(orig_label_img,rel_label_img):
				pdb.set_trace()

			fname_stem = Path(orig_rgb_fpath).stem
			orig_save_fpath = f'{save_dir}/{split}_{i}_noguide_orig_{fname_stem}.png'
			orig_guide_save_fpath = f'{save_dir}/{split}_{i}_guide_orig_{fname_stem}.png'

			rel_save_fpath = f'{save_dir}/{split}_{i}_noguide_rel_{fname_stem}.png'
			rel_guide_save_fpath = f'{save_dir}/{split}_{i}_guide_rel_{fname_stem}.png'

			if orig_rgb_img.ndim == 2:
				# this image was grayscale
				orig_rgb_img = grayscale_to_color(orig_rgb_img)
			
			if rel_rgb_img.ndim == 2:
				# this image was grayscale
				rel_rgb_img = grayscale_to_color(rel_rgb_img)

			#save_classnames_in_image(img_rgb, label_img, id_to_class_name_map, save_to_disk=True, save_fpath=save_fpath)
			form_mask_triple_embedded_classnames(orig_rgb_img, orig_label_img, orig_id_to_classname_map, orig_save_fpath, save_to_disk=True)
			save_mask_triple_with_color_guide(orig_rgb_img, orig_label_img, orig_id_to_classname_map,fname_stem,save_dir, orig_guide_save_fpath)

			form_mask_triple_embedded_classnames(rel_rgb_img, rel_label_img, rel_id_to_classname_map, rel_save_fpath, save_to_disk=True)
			save_mask_triple_with_color_guide(rel_rgb_img, rel_label_img, rel_id_to_classname_map,fname_stem,save_dir, rel_guide_save_fpath)


if __name__ == '__main__':


	main()


