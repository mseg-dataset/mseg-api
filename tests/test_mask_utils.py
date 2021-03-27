#!/usr/bin/python3

import copy
import cv2
import imageio
import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pdb
from PIL import Image, ImageDraw
import time
import torch
from typing import Mapping

from mseg.utils.names_utils import get_dataloader_id_to_classname_map
from mseg.utils.colormap import colormap
from mseg.utils.mask_utils import (
	COLORMAP_OFFSET,
	convert_instance_img_to_mask_img,
	search_jittered_location_in_mask,
	find_max_cardinality_mask,
	form_label_mapping_array,
	form_contained_classes_color_guide,
	form_label_mapping_array_pytorch,
	form_mask_triple,
	get_instance_mask_class_votes,
	get_mask_from_polygon,
	get_most_populous_class,
	get_np_mode,
	get_polygons_from_binary_img,
	map_semantic_img_fast, 
	map_semantic_img_fast_pytorch,
	swap_px_inside_mask,
	rgb_img_to_obj_cls_img,
	save_classnames_in_image_maxcardinality,
	save_classnames_in_image_sufficientpx,
	save_pred_vs_label_7tuple,
	swap_px_inside_mask,
	vis_mask,
	write_six_img_grid_w_embedded_names
)

from mseg.utils.resize_util import resize_img_by_short_side
from mseg.utils.test_utils import dict_is_equal


ROOT = Path(__file__).resolve().parent
TEST_DATA_ROOT = ROOT / "test_data"



def test_swap_px_inside_mask():

	label_img = np.array(
		[
			[7,8,9,0],
			[0,7,0,7],
			[3,4,7,1]
		], dtype=np.uint8)

	segment_mask = np.array(
		[
			[1,0,0,0],
			[0,1,0,1],
			[0,0,1,0]
		], dtype=np.uint8)

	old_val = 7
	new_val = 255
	new_label_img = swap_px_inside_mask(label_img, segment_mask, old_val, new_val)
	gt_new_label_img = np.array(
		[
			[255,8,9,0],
			[0,255,0,255],
			[3,4,255,1]
		], dtype=np.uint8)

	assert np.allclose(new_label_img, gt_new_label_img)



def test_find_max_cardinality_mask():
	""" """
	masks = [
		np.array(
			[
				[0, 0, 0, 0],
				[0, 0, 1, 0],
				[0, 0, 0, 0]
			], dtype=np.uint8),
		np.array(
			[
				[0, 0, 0, 1],
				[0, 0, 0, 1],
				[0, 0, 1, 0]
			], dtype=np.uint8),
		np.array(
			[
				[0, 0, 0, 0],
				[1, 1, 0, 0],
				[0, 0, 0, 0]
			], dtype=np.uint8)
		]
	assert find_max_cardinality_mask(masks) == 1


def test_save_classnames_in_image():
	""" """
	MAX_GRAY_VAL = 128

	label_img = np.zeros((300,500), dtype=np.float32)
	label_img[:100,:300] = 0
	label_img[:100,300:] = 2
	label_img[100:200,:] = 1
	label_img[200:,:100] = 0
	label_img[200:,100:200] = 3
	label_img[200:,200:300] = 4
	label_img[200:,300:] = 0
	
	rgb_img = np.zeros((300,500,3), dtype=np.float32)
	for i in range(3):
		rgb_img[:,:,i] = MAX_GRAY_VAL * label_img / np.amax(label_img)

	rgb_img = rgb_img.astype(np.uint8)
	label_img = label_img.astype(np.uint8)

	plt.imshow(rgb_img)
	#plt.show()
	plt.close('all')

	# jittering will be random, make it deterministic
	np.random.seed(2)
	id_to_class_name_map = { i:f'Class {i}' for i in range(5) }
	rgb_img_w_text = save_classnames_in_image_sufficientpx(rgb_img, label_img, id_to_class_name_map, font_color=(255,255,255))
	
	plt.imshow(rgb_img_w_text)
	#plt.show()
	plt.close('all')
	
	print(np.sum(rgb_img_w_text))
	assert np.sum(rgb_img_w_text) == 19959786


def map_semantic_img_slow(
	semantic_img: np.ndarray, 
	label_mapping: Mapping[int,int]
	) -> np.ndarray:
	"""
	Convert grayscale image to a different grayscale image.

	Assumption is that there are less than 65,000 classes in the final mapped class for np.uint16
		for uint8, can have max 255 classes

		Args:
		-	semantic_img: Array of shape (M,N)
		-	label_mapping: dictionary from integer ID to integer ID

		Returns:
		-	mapped: Array of shape (M,N)
	"""
	mapped = np.copy(semantic_img)
	for k,v in label_mapping.items():
		mapped[semantic_img==k] = v
	assert mapped.dtype in [np.uint8, np.uint16]
	return mapped


"""
def test_PIL_Image_load():
	# Verify the widening conversion to int32 when we load uint16 with PIL.
	# PIL will convert the uint16 to int32
	# Imageio will preserve the int16
	data_dir = f'{TEST_DATA_ROOT}/scannet_v2_549/extracted_ims_scene0106_01'
	label_fpath = f'{data_dir}/0.png'
	PIL_label = Image.open(label_fpath)
	imageio_label = imageio.imread(label_fpath)

	assert np.allclose( np.array(PIL_label), imageio_label)
	assert np.array(PIL_label).dtype == np.int32
	assert imageio_label.dtype == np.uint16
"""

def test_save_pred_vs_label_7tuple_short_side_60_img():
	"""
	When the image is too low in resolution (e.g. short side 60),
	we cannot use cv2's text rendering code. Instead, we will also save
	the upsampled version.
	"""
	short_side = 60 # pixels

	data_dir = f'{TEST_DATA_ROOT}/Camvid_test_data'

	img_fpath = f'{data_dir}/images/0016E5_08159.png'
	label_fpath = f'{data_dir}/preds/0016E5_08159.png'

	img_rgb = imageio.imread(img_fpath)
	label_img = imageio.imread(label_fpath)
	
	img_rgb = resize_img_by_short_side(img_rgb, short_side, img_type='rgb')
	label_img = resize_img_by_short_side(label_img, short_side, img_type='label')

	img_h, img_w = label_img.shape

	pred_img = np.random.randint(0,200, (img_h, img_w)).astype(np.uint16)
	id_to_class_name_map = get_dataloader_id_to_classname_map('pascal-context-460')

	save_fpath = f'{TEST_DATA_ROOT}/rand_549_temp_small.png'
	save_pred_vs_label_7tuple(img_rgb, pred_img, label_img, id_to_class_name_map, save_fpath)
	os.remove(f'{TEST_DATA_ROOT}/rand_549_temp_small_upsample_pred_labels_palette.png')
	os.remove(f'{TEST_DATA_ROOT}/rand_549_temp_small_pred_labels_palette.png')



def test_save_pred_vs_label_7tuple_shifted():
	""" """
	data_dir = f'{TEST_DATA_ROOT}/Camvid_test_data'

	img_fpath = f'{data_dir}/images/0016E5_08159.png'
	label_fpath = f'{data_dir}/preds/0016E5_08159.png'

	img_rgb = imageio.imread(img_fpath)
	label_img = imageio.imread(label_fpath)

	pred_img = np.zeros_like(label_img)
	pred_img[:,150:] = label_img[:,:-150]

	id_to_class_name_map = get_dataloader_id_to_classname_map('pascal-context-460')

	save_fpath = f'{TEST_DATA_ROOT}/shifted_temp.png'
	save_pred_vs_label_7tuple(img_rgb, pred_img, label_img, id_to_class_name_map, save_fpath)
	os.remove(f'{TEST_DATA_ROOT}/shifted_temp_pred_labels_palette.png')


def test_save_pred_vs_label_7tuple_uniform_random_preds():
	"""
	"""
	data_dir = f'{TEST_DATA_ROOT}/Camvid_test_data'

	img_fpath = f'{data_dir}/images/0016E5_08159.png'
	label_fpath = f'{data_dir}/preds/0016E5_08159.png'

	img_rgb = imageio.imread(img_fpath)
	label_img = imageio.imread(label_fpath)
	img_h, img_w = label_img.shape

	pred_img = np.random.randint(0,459, (img_h, img_w)).astype(np.uint16)
	id_to_class_name_map = get_dataloader_id_to_classname_map('pascal-context-460')

	save_fpath = f'{TEST_DATA_ROOT}/rand_549_temp.png'
	save_pred_vs_label_7tuple(img_rgb, pred_img, label_img, id_to_class_name_map, save_fpath)
	os.remove(f'{TEST_DATA_ROOT}/rand_549_temp_pred_labels_palette.png')



def test_save_pred_vs_label_7tuple_100_strided_preds():
	"""
	"""
	data_dir = f'{TEST_DATA_ROOT}/Camvid_test_data'

	img_fpath = f'{data_dir}/images/0016E5_08159.png'
	label_fpath = f'{data_dir}/preds/0016E5_08159.png'

	img_rgb = imageio.imread(img_fpath)
	label_img = imageio.imread(label_fpath)
	img_h, img_w = label_img.shape

	class_ids_to_sample = np.array([0,100,200,300])
	pred_img = np.random.choice(a=class_ids_to_sample, size=(img_h, img_w)).astype(np.uint16)
	id_to_class_name_map = get_dataloader_id_to_classname_map('pascal-context-460')

	save_fpath = f'{TEST_DATA_ROOT}/100_strided_temp.png'
	save_pred_vs_label_7tuple(img_rgb, pred_img, label_img, id_to_class_name_map, save_fpath)
	os.remove(f'{TEST_DATA_ROOT}/100_strided_temp_pred_labels_palette.png')


def test_map_semantic_img_fast_pytorch():
	"""
	Test fast method on simple conversion from 2x3 grayscale -> 2x3 grayscale.
	"""
	semantic_img = np.array([[254,0,1],[7,8,9]], dtype=np.uint8)
	semantic_img = torch.from_numpy(semantic_img).type(torch.LongTensor)

	label_mapping = { 254: 253,
						0: 255,
						1: 0,
						7: 6,
						8: 7,
						9: 8	}
	label_mapping_arr = form_label_mapping_array_pytorch(label_mapping)
	mapped_img = map_semantic_img_fast_pytorch(semantic_img, label_mapping_arr)
	gt_mapped_img = np.array([	[253,255,0],
								[6,7,8]],  dtype=np.uint8)

	assert np.allclose(gt_mapped_img, mapped_img.numpy())
	assert mapped_img.dtype == torch.int64




def test_map_semantic_img_fast_pytorch_uint16_midsize_test():
	"""
	Test fast method on large example. Map 60,000 classes to a 
	different 60,000 classes with a fast array conversion.

	Should take less than 1 millisecond.
	"""
	semantic_img = np.array(range(30)).reshape(5,6).astype(np.int16)
	semantic_img = torch.from_numpy(semantic_img).type(torch.LongTensor)

	label_mapping = {}
	for i in range(30):
		label_mapping[i] = i + 1

	label_mapping_copy = copy.deepcopy(label_mapping)
	label_mapping_arr = form_label_mapping_array_pytorch(label_mapping)
	assert label_mapping == label_mapping_copy
	start = time.time()
	mapped_img = map_semantic_img_fast_pytorch(semantic_img, label_mapping_arr)
	end = time.time()

	assert mapped_img.shape == torch.Size([5,6])
	assert mapped_img.dtype == torch.int64

	print(f'Took {end-start} sec.')
	gt_mapped_img = np.array(range(30)).reshape(5,6).astype(np.uint16) + 1

	assert np.allclose(gt_mapped_img, mapped_img.numpy())




def test_map_semantic_img_fast_pytorch_uint16_stress_test():
	"""
	Test fast method on large example. Map 60,000 classes to a 
	different 60,000 classes with a fast array conversion.

	np.int16 cannot hold anything over 32767 / 32768.

	Should take less than 1 millisecond.
	"""
	semantic_img = np.array(range(60000)).reshape(3000,20).astype(np.int64)
	semantic_img = torch.from_numpy(semantic_img).type(torch.LongTensor)

	label_mapping = {}
	for i in range(60000):
		label_mapping[i] = i + 1

	label_mapping_copy = copy.deepcopy(label_mapping)
	label_mapping_arr = form_label_mapping_array_pytorch(label_mapping)
	label_mapping_arr_copy = label_mapping_arr.clone()
	assert label_mapping == label_mapping_copy
	start = time.time()
	mapped_img = map_semantic_img_fast_pytorch(semantic_img, label_mapping_arr)
	end = time.time()
	assert torch.allclose(label_mapping_arr_copy, label_mapping_arr)

	assert mapped_img.shape == torch.Size([3000,20])
	assert mapped_img.dtype == torch.int64

	print(f'Took {end-start} sec.')
	gt_mapped_img = np.array(range(60000)).reshape(3000,20).astype(np.uint16) + 1

	assert np.allclose(gt_mapped_img, mapped_img.numpy())



def test_form_label_mapping_array_pytorch_fromzero():
	"""
	Count from zero, with 8-bit unsigned int data type.
	"""
	label_mapping = { 	
		0: 1, 
		1: 2, 
		3: 4 
	}
	label_mapping_arr = form_label_mapping_array_pytorch(label_mapping)

	assert label_mapping_arr.shape == torch.Size([4])
	assert label_mapping_arr.dtype == torch.int64

	gt_label_mapping_arr = np.array([1,2,0,4], dtype=np.uint8)
	assert np.allclose(label_mapping_arr.numpy(), gt_label_mapping_arr)


def test_form_label_mapping_array_pytorch_from_nonzero():
	"""
	"""
	label_mapping = {655: 300, 654: 255, 653: 100}
	label_mapping_arr = form_label_mapping_array_pytorch(label_mapping)

	assert label_mapping_arr.shape == torch.Size([656])
	assert label_mapping_arr.dtype == torch.int64

	gt_label_mapping_arr = np.zeros((656), dtype=np.uint16)
	gt_label_mapping_arr[655] = 300
	gt_label_mapping_arr[654] = 255
	gt_label_mapping_arr[653] = 100
	print(label_mapping)
	assert np.allclose(label_mapping_arr.numpy(), gt_label_mapping_arr)


def test_map_semantic_img_uint8():
	"""
	Test slow method, from 2x3 grayscale -> 2x3 grayscale image.
	"""
	semantic_img = np.array([[254,0,1],[7,8,9]], dtype=np.uint8)

	label_mapping = { 254: 253,
						0: 255,
						1: 0,
						7: 6,
						8: 7,
						9: 8	}

	mapped_img = map_semantic_img_slow(semantic_img, label_mapping)

	gt_mapped_img = np.array([	[253,255,0],
								[6,7,8]],  dtype=np.uint8)

	assert np.allclose(gt_mapped_img, mapped_img)
	assert gt_mapped_img.dtype == mapped_img.dtype


def test_map_semantic_img_uint16():
	"""
	"""
	semantic_img = np.array([[254,0,1],
							[7,8,9]], dtype=np.uint16)

	label_mapping = { 254: 253,
						0: 255,
						1: 0,
						7: 6,
						8: 7,
						9: 8	}

	mapped_img = map_semantic_img_slow(semantic_img, label_mapping)

	gt_mapped_img = np.array([[253,255,0],[6,7,8]],  dtype=np.uint16)

	assert np.allclose(gt_mapped_img, mapped_img)
	assert gt_mapped_img.dtype == mapped_img.dtype


def test_map_semantic_img_uint16_stress_test():
	"""
	Map 60,000 classes to a different 60,000 classes in the slowest way
	possible. Should take about 1 second.
	"""
	semantic_img = np.array(range(60000)).reshape(3000,20).astype(np.uint16)

	label_mapping = {}
	for i in range(60000):
		label_mapping[i] = i + 1

	start = time.time()
	mapped_img = map_semantic_img_slow(semantic_img, label_mapping)
	end = time.time()
	print(f'Took {end-start} sec.')
	gt_mapped_img = np.array(range(60000)).reshape(3000,20).astype(np.uint16) + 1

	assert np.allclose(gt_mapped_img, mapped_img)
	assert gt_mapped_img.dtype == mapped_img.dtype


def test_map_semantic_img_fast():
	"""
	Test fast method on simple conversion from 2x3 grayscale -> 2x3 grayscale.
	"""
	semantic_img = np.array(
		[
			[254,0,1],
			[7,8,9]
		], dtype=np.uint8)

	label_mapping = { 
		254: 253,
		0: 255,
		1: 0,
		7: 6,
		8: 7,
		9: 8
	}
	label_mapping_arr = form_label_mapping_array(label_mapping)
	mapped_img = map_semantic_img_fast(semantic_img, label_mapping_arr)

	# Expect uint8 since max class index <= 255, so uint16 unnecessary
	gt_mapped_img1 = np.array(
		[
			[253,255,0],
			[6,7,8]
		],  dtype=np.uint8)
	gt_mapped_img2 = map_semantic_img_slow(semantic_img, label_mapping)

	assert np.allclose(gt_mapped_img1, mapped_img)
	assert np.allclose(gt_mapped_img2, mapped_img)
	assert gt_mapped_img1.dtype == mapped_img.dtype
	assert gt_mapped_img2.dtype == mapped_img.dtype


def test_map_semantic_img_fast_widenvalue():
	"""
	Test fast method on simple conversion from 2x3 grayscale -> 2x3 grayscale.
	"""
	semantic_img = np.array(
		[
			[255,255,255],
			[255,255,255]
		], dtype=np.uint8)

	label_mapping = {255:256}
	label_mapping_arr = form_label_mapping_array(label_mapping)
	mapped_img = map_semantic_img_fast(semantic_img, label_mapping_arr)

	# Expect uint8 since max class index <= 255, so uint16 unnecessary
	gt_mapped_img1 = np.array(
		[
			[256,256,256],
			[256,256,256]
		],  dtype=np.uint16)

	assert np.allclose(gt_mapped_img1, mapped_img)
	assert mapped_img.dtype == np.uint16


def test_map_semantic_img_fast_dontwidenvalue():
	"""
	Test fast method on simple conversion from 2x3 grayscale -> 2x3 grayscale.
	"""
	semantic_img = np.array(
		[
			[300,301,302],
			[300,301,302]
		], dtype=np.uint16)

	label_mapping = {
		300:0,
		301:1,
		302:2
	}
	label_mapping_arr = form_label_mapping_array(label_mapping)
	mapped_img = map_semantic_img_fast(semantic_img, label_mapping_arr)
	# Expect uint8 since max class index in values <= 255, so uint16 unnecessary
	gt_mapped_img1 = np.array(
		[
			[0,1,2],
			[0,1,2]
		],  dtype=np.uint8)

	assert np.allclose(gt_mapped_img1, mapped_img)
	assert mapped_img.dtype == np.uint8



def test_map_semantic_img_fast_uint16_stress_test():
	"""
	Test fast method on large example. Map 60,000 classes to a 
	different 60,000 classes with a fast array conversion.

	Should take less than 1 millisecond.
	"""
	semantic_img = np.array(range(60000)).reshape(3000,20).astype(np.uint16)
	# will expect uint16 back since 60000 < 65535, which is uint16 max
	gt_mapped_img1 = np.array(range(60000)).reshape(3000,20).astype(np.uint16) + 1

	label_mapping = {i: i+1 for i in range(60000)}

	label_mapping_copy = copy.deepcopy(label_mapping)
	label_mapping_arr = form_label_mapping_array(label_mapping)
	dict_is_equal(label_mapping, label_mapping_copy)
	assert label_mapping == label_mapping_copy
	start = time.time()
	mapped_img = map_semantic_img_fast(semantic_img, label_mapping_arr)
	end = time.time()
	print(f'Took {end-start} sec.')

	gt_mapped_img2 = map_semantic_img_slow(semantic_img, label_mapping)

	assert np.allclose(gt_mapped_img1, mapped_img)
	assert np.allclose(gt_mapped_img2, mapped_img)
	assert gt_mapped_img1.dtype == mapped_img.dtype
	assert gt_mapped_img2.dtype == mapped_img.dtype


def test_form_label_mapping_array_fromzero_uint8():
	"""
	Count from zero, with 8-bit unsigned int data type.
	"""
	label_mapping = { 	
		0: 1, 
		1: 2, 
		3: 4 
	}
	label_mapping_arr = form_label_mapping_array(label_mapping)

	# since max class index <= 255, we expect uint8
	gt_label_mapping_arr = np.array([1,2,0,4], dtype=np.uint8)
	assert np.allclose(label_mapping_arr, gt_label_mapping_arr)


def test_form_label_mapping_array_from_nonzero_uint16():
	"""
	"""
	label_mapping = {655: 300, 654: 255, 653: 100}
	label_mapping_arr = form_label_mapping_array(label_mapping)

	# since max class index > 255, we expect uint16
	gt_label_mapping_arr = np.zeros((656), dtype=np.uint16)
	gt_label_mapping_arr[655] = 300
	gt_label_mapping_arr[654] = 255
	gt_label_mapping_arr[653] = 100
	print(label_mapping)
	assert np.allclose(label_mapping_arr, gt_label_mapping_arr)


def test_form_contained_classes_color_guide_smokescreen() -> None:
	"""
	"""
	label_img = np.array([[0,255,0],[255,1,255]], dtype=np.uint8)
	id_to_class_name_map = {0: 'cat', 1: 'dog', 255: 'unlabeled'}
	fname_stem = 'temp'
	save_dir = TEST_DATA_ROOT
	_ = form_contained_classes_color_guide(label_img,
									   id_to_class_name_map,
									   fname_stem,
									   save_dir,
									   save_to_disk = True)
	palette_save_fpath = f'{TEST_DATA_ROOT}/{fname_stem}_colors.png'
	assert Path(palette_save_fpath).exists()
	os.remove(palette_save_fpath)



def test_rgb_img_to_obj_cls_img():
	"""
	"""
	label_img_rgb = np.zeros((2,3,3),dtype=np.uint8)
	label_img_rgb[0,0,:] = [0,1,2]
	label_img_rgb[0,1,:] = [3,4,5]
	label_img_rgb[0,2,:] = [255,254,253]

	label_img_rgb[1,0,:] = [254,253,252]
	label_img_rgb[1,1,:] = [255,254,253]
	label_img_rgb[1,2,:] = [0,1,2]

	dataset_ordered_colors = np.array(
		[
			[255,254,253],
			[254,253,252],
			[0,1,2],
			[3,4,5],
		], dtype=np.uint8)

	label_img_rgb_copy = label_img_rgb.copy()
	dataset_ordered_colors_copy = dataset_ordered_colors.copy()
	obj_cls_img = rgb_img_to_obj_cls_img(label_img_rgb, dataset_ordered_colors)

	# ensure unmodified label_img_rgb
	assert np.allclose(label_img_rgb_copy, label_img_rgb)
	assert label_img_rgb_copy.dtype == label_img_rgb.dtype
	assert label_img_rgb_copy.shape == label_img_rgb.shape

	# ensure unmodified  dataset_ordered_colors
	assert np.allclose(dataset_ordered_colors_copy, dataset_ordered_colors)
	assert dataset_ordered_colors_copy.dtype == dataset_ordered_colors.dtype
	assert dataset_ordered_colors_copy.shape == dataset_ordered_colors.shape

	gt_obj_cls_img = np.array([[2,3,0],[1,0,2]], dtype=np.uint8)
	assert np.allclose(obj_cls_img, gt_obj_cls_img)


def test_form_mask_triple_write_to_disk() -> None:
	"""
	"""
	rgb_img = np.zeros((3,3,3), dtype=np.uint8)
	rgb_img[0,0,:] = [255,255,255]
	rgb_img[1,1,:] = [255,255,255]
	rgb_img[2,2,:] = [255,255,255]

	label_img = np.array(
		[
			[0,1,0],
			[1,1,1],
			[0,1,0]
		]
	).astype(np.uint8)
	save_fpath = '../temp_files/temp.png'
	hstack_img = form_mask_triple(rgb_img,
								  label_img,
								  save_fpath,
								  save_to_disk=True)
	assert Path(save_fpath).exists()





def test_vis_mask_alpha_point5() -> None:
	"""
	Blend specified image with specified colors in mask region half/half.
	"""
	rgb_img = np.zeros((2,3,3), dtype=np.uint8)
	rgb_img[:,:,0] = np.array([[100,100,100],[200,200,200]], dtype=np.uint8)
	mask_img = np.array([[0,1,0],[1,1,1]], dtype=np.uint8)
	col = np.array([60,0,0], dtype=np.uint8)
	alpha = 0.5
	rgb_img_copy = rgb_img.copy()
	blended_img = vis_mask(rgb_img, mask_img, col, alpha)
	assert np.allclose(rgb_img, rgb_img_copy)
	gt_blended_img = np.zeros((2,3,3), dtype=np.uint8)
	gt_blended_img[:,:,0] = np.array([[100,80,100],[130,130,130]], dtype=np.uint8)
	assert np.allclose(gt_blended_img, blended_img)




def test_vis_mask_alpha_1() -> None:
	"""
	By setting alpha to zero, we wipe away the old RGB values completely,
	and leave only the specificed color values within the mask region.
	"""
	rgb_img = np.zeros((2,3,3), dtype=np.uint8)
	rgb_img[:,:,0] = np.array([[100,100,100],[200,200,200]], dtype=np.uint8)
	mask_img = np.array([[0,1,0],[1,1,1]], dtype=np.uint8)
	col = np.array([60,0,0], dtype=np.uint8)
	alpha = 1.0
	rgb_img_copy = rgb_img.copy()
	blended_img = vis_mask(rgb_img, mask_img, col, alpha)
	print(blended_img)
	assert np.allclose(rgb_img, rgb_img_copy)

	gt_blended_img = np.zeros((2,3,3), dtype=np.uint8)
	gt_blended_img[:,:,0] = np.array([[100,60,100],[60,60,60]], dtype=np.uint8)
	assert np.allclose(gt_blended_img, blended_img)



def test_vis_mask_alpha_0() -> None:
	"""
	By setting alpha to zero, we do not modify the RGB image.
	"""
	rgb_img = np.zeros((2,3,3), dtype=np.uint8)
	rgb_img[:,:,0] = np.array([[100,100,100],[200,200,200]], dtype=np.uint8)
	mask_img = np.array([[0,1,0],[1,1,1]], dtype=np.uint8)
	col = np.array([60,0,0], dtype=np.uint8)
	alpha = 0.0
	rgb_img_copy = rgb_img.copy()
	blended_img = vis_mask(rgb_img, mask_img, col, alpha)
	print(blended_img)
	assert np.allclose(rgb_img, rgb_img_copy)
	assert np.allclose(blended_img, rgb_img_copy)


def test_get_mask_from_polygon(visualize=False):
	""" """
	img_w = 16
	img_h = 8
	tuple_verts = [
		(2,6),
		(6,2),
		(6,4),
		(10,4),
		(10,6),
		(2,6) # may be ignored
	]
	mask = get_mask_from_polygon(tuple_verts, img_h, img_w)
	
	gt_mask = np.array([
		[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
		[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
		[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
		[False, False, False, False, False,  True,  True, False, False, False, False, False, False, False, False, False],
		[False, False, False, False,  True,  True,  True, False, False, False, False, False, False, False, False, False],
		[False, False, False,  True,  True,  True,  True,  True,  True, True,  True, False, False, False, False, False],
		[False, False,  True,  True,  True,  True,  True,  True,  True, True,  True, False, False, False, False, False],
		[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
	])
	assert np.allclose(mask, gt_mask)

	if visualize:
		plt.imshow(mask)
		plt.show()




def test_get_instance_mask_class_votes1():
	"""
	Given instance masks covering entire image, ensure get most likely
	category vote.
	"""
	instance_mask = np.array(
		[
			[1,1],
			[1,1]
		])
	label_img = np.array(
		[
			[0,0],
			[0,0]
		])
	label_votes, majority_vote = get_instance_mask_class_votes(instance_mask, label_img)
	gt_majority_vote = 0
	gt_label_votes = np.array([0,0,0,0])
	assert gt_majority_vote == majority_vote
	assert np.allclose(gt_label_votes, label_votes)


def test_get_instance_mask_class_votes2():
	"""
	Given instance masks covering partial image, ensure get most likely
	category vote. Here, our instance mask is noisy.
	"""
	instance_mask = np.array(
		[
			[0,0,1,0],
			[0,1,1,0],
			[0,1,1,0],
			[0,1,0,0]
		])
	label_img = np.array(
		[
			[8,8,8,8],
			[9,9,9,9],
			[8,9,9,9],
			[7,7,7,7]
		])
	label_votes, majority_vote = get_instance_mask_class_votes(instance_mask, label_img)
	gt_majority_vote = 9
	gt_label_votes = np.array([8, 9, 9, 9, 9, 7])
	assert gt_majority_vote == majority_vote
	assert np.allclose(gt_label_votes, label_votes)



def test_get_np_mode_1():
	""" """
	x = np.array([5,4,5,4,5])
	mode = get_np_mode(x)
	assert mode == 5

	x = np.array([5,5])
	mode = get_np_mode(x)
	assert mode == 5

	x = np.array([5])
	mode = get_np_mode(x)
	assert mode == 5



# following tests are dependent upon the choice of colormap. TODO: fix


# def test_convert_instance_img_to_mask_img_nonzero_start() -> None:
# 	"""
# 	Test when the instance IDs don't start from zero
# 	"""
# 	fb_colormap = colormap(rgb=True)
# 	fb_colormap_subset = 255 * np.array(
# 		[
# 			[0.000, 0.667, 1.000],
#             [0.000, 1.000, 1.000],
#             [0.333, 0.000, 1.000],
#             [0.333, 0.333, 1.000],
#             [0.333, 0.667, 1.000],
#             [0.333, 1.000, 1.000],
# 		]
# 	)
# 	fb_colormap_subset = fb_colormap_subset.astype(np.uint8)
# 	assert np.allclose(fb_colormap[COLORMAP_OFFSET:COLORMAP_OFFSET + 6], fb_colormap_subset)

# 	gt_mask_img = np.zeros([2,3,3], dtype=np.uint8)
# 	gt_mask_img[0,0,:] = fb_colormap[COLORMAP_OFFSET + 5]
# 	gt_mask_img[0,1,:] = fb_colormap[COLORMAP_OFFSET + 5]
# 	gt_mask_img[0,2,:] = fb_colormap[COLORMAP_OFFSET + 3]

# 	gt_mask_img[1,0,:] = fb_colormap[COLORMAP_OFFSET + 5]
# 	gt_mask_img[1,1,:] = fb_colormap[COLORMAP_OFFSET + 3]
# 	gt_mask_img[1,2,:] = fb_colormap[COLORMAP_OFFSET + 4]
# 	alpha = 0.4
# 	gt_mask_img = (gt_mask_img * alpha).astype(np.uint8)

# 	instance_img = np.array([[5,5,3],[5,3,4]], dtype=np.uint8)
# 	img_rgb = np.zeros((2,3,3), dtype=np.uint8)
	
# 	mask_img = convert_instance_img_to_mask_img(instance_img, img_rgb)
# 	assert np.allclose(mask_img, gt_mask_img)




# def test_convert_instance_img_to_mask_img_zero_start() -> None:
# 	"""
# 	Test when the instance IDs DO start from zero.
# 	"""
# 	fb_colormap = colormap(rgb=True)

# 	gt_mask_img = np.zeros([2,3,3], dtype=np.uint8)
# 	gt_mask_img[0,0,:] = fb_colormap[COLORMAP_OFFSET + 5]
# 	gt_mask_img[0,1,:] = fb_colormap[COLORMAP_OFFSET + 4]
# 	gt_mask_img[0,2,:] = fb_colormap[COLORMAP_OFFSET + 3]

# 	gt_mask_img[1,0,:] = fb_colormap[COLORMAP_OFFSET + 2]
# 	gt_mask_img[1,1,:] = fb_colormap[COLORMAP_OFFSET + 1]
# 	gt_mask_img[1,2,:] = fb_colormap[COLORMAP_OFFSET + 0]
# 	alpha = 0.4
# 	gt_mask_img = ((gt_mask_img * alpha) + 0.6).astype(np.uint8)

# 	instance_img = np.array([[5,4,3],[2,1,0]], dtype=np.uint8)
# 	img_rgb = np.ones((2,3,3), dtype=np.uint8)
	
# 	mask_img = convert_instance_img_to_mask_img(instance_img, img_rgb)
# 	assert np.allclose(mask_img, gt_mask_img)

# def test_form_mask_triple_dont_write_to_disk() -> None:
# 	"""
# 	"""
# 	rgb_img = np.zeros((3,3,3), dtype=np.uint8)
# 	rgb_img[0,0,:] = [255,255,255]
# 	rgb_img[1,1,:] = [255,255,255]
# 	rgb_img[2,2,:] = [255,255,255]

# 	label_img = np.array(
# 		[
# 			[0,1,0],
# 			[1,1,1],
# 			[0,1,0]
# 		]
# 	).astype(np.uint8)
# 	save_fpath = 'test_data/temp.png'
# 	hstack_img = form_mask_triple(rgb_img,
# 								  label_img,
# 								  save_fpath,
# 								  save_to_disk=False)
# 	assert hstack_img.shape == (3,9,3)
# 	gt_hstack_img = np.array(
# 		[
# 			[[255, 255, 255],
# 	        [  0,   0,   0],
# 	        [  0,   0,   0],
# 	        [153, 221, 255],
# 	        [  0, 102, 102],
# 	        [  0,  68, 102],
# 	        [153, 221, 255],
# 	        [153, 255, 255],
# 	        [153, 221, 255]],

# 	       [[  0,   0,   0],
# 	        [255, 255, 255],
# 	        [  0,   0,   0],
# 	        [  0, 102, 102],
# 	        [153, 255, 255],
# 	        [  0, 102, 102],
# 	        [153, 255, 255],
# 	        [153, 255, 255],
# 	        [153, 255, 255]],

# 	       [[  0,   0,   0],
# 	        [  0,   0,   0],
# 	        [255, 255, 255],
# 	        [  0,  68, 102],
# 	        [  0, 102, 102],
# 	        [153, 221, 255],
# 	        [153, 221, 255],
# 	        [153, 255, 255],
# 	        [153, 221, 255]]
#         ]
#     ).astype(np.uint8)
# 	assert np.allclose(gt_hstack_img, hstack_img )



def test_swap_px_inside_mask():
	""" """
	segment_mask = np.array(
		[
			[0,0,1,0],
			[0,1,1,0],
			[1,1,1,1]
		]
	)
	semantic_img = np.zeros((3,4), dtype=np.uint8)
	new_img = swap_px_inside_mask(
		semantic_img, 
		segment_mask,
		old_val=0,
		new_val=8,
		require_strict_boundaries=True
	)
	gt_new_img = np.array(
		[
			[0,0,8,0],
			[0,8,8,0],
			[8,8,8,8]
		]
	)
	assert np.allclose(gt_new_img, new_img)


def test_get_mask_from_polygon():
	"""
	Test PIL rasterization correctness on a simple
	2x2 square pattern.
	"""
	# define (x,y) tuples
	polygon = [ [0,0], [2,0], [2,2], [0,2] ]
	img_h = 4
	img_w = 4
	mask = get_mask_from_polygon(polygon, img_h, img_w)
	gt_mask = np.array(
		[
			[1, 1, 1, 0],
			[1, 1, 1, 0],
			[1, 1, 1, 0],
			[0, 0, 0, 0]
		], dtype=np.uint8)
	assert np.allclose(mask, gt_mask)


def test_polygon_to_mask1():
	""" """
	h = 6
	w = 6
	# define (x,y) tuples
	polygon = [ [1,1], [3,1], [3,4], [1,4] ]

	mask = get_mask_from_polygon(polygon, h, w)
	gt_mask = np.array([
		[0, 0, 0, 0, 0, 0],
		[0, 1, 1, 1, 0, 0],
		[0, 1, 1, 1, 0, 0],
		[0, 1, 1, 1, 0, 0],
		[0, 1, 1, 1, 0, 0],
		[0, 0, 0, 0, 0, 0]
	])
	assert np.allclose(mask, gt_mask)


def test_polygon_to_mask2():
	""" """
	h = 6
	w = 6
	# define (x,y) tuples
	polygon = [ [2,0], [5,3], [0,5] ]
	mask = get_mask_from_polygon(polygon, h, w)
	gt_mask = np.array(
		[
			[0, 0, 1, 0, 0, 0],
			[0, 0, 1, 1, 0, 0],
			[0, 1, 1, 1, 1, 0],
			[0, 1, 1, 1, 1, 1],
			[1, 1, 1, 0, 0, 0],
			[1, 0, 0, 0, 0, 0]
		])
	assert np.allclose(mask, gt_mask)


def test_search_jittered_location_in_mask():
	"""
	Jitter location for nonconvex object
	"""
	mask_fpath = f'{TEST_DATA_ROOT}/nonconvex_table_mask.png'
	mask = imageio.imread(mask_fpath)

	y = 475
	x = 564
	np.random.seed(1)
	x,y = search_jittered_location_in_mask(x, y, conncomp=mask)

	assert mask[y,x] == 1

	# plt.imshow(mask)
	# plt.scatter(x,y,10,color='r', marker='.')
	# plt.show()


def test_get_most_populous_class1():
	"""
	"""
	segment_mask = np.array(
		[
			[1,0,0],
			[1,0,0],
			[0,0,0]
		])
	label_map = np.array(
		[
			[9,8,8],
			[9,8,8],
			[8,8,8]
		])
	assert get_most_populous_class(segment_mask, label_map) == 9


def test_get_most_populous_class2():
	"""
	"""
	segment_mask = np.array(
		[
			[1,0,0],
			[1,1,0],
			[0,1,0]
		])
	label_map = np.array(
		[
			[9,8,8],
			[7,7,8],
			[8,7,8]
		])
	assert get_most_populous_class(segment_mask, label_map) == 7



def test_get_polygons_from_binary_img1():
	"""
	"""

	"""
	label map with four quadrants.
	|  Sky   | Road  |
	----------------
	| Person | Horse |
	"""
	H = 6
	W = 10
	img_rgb = np.ones((H,W,3), dtype=np.uint8)
	label_map = np.zeros((H,W), dtype=np.uint8)
	label_map[:H//2, :W//2] = 0
	label_map[:H//2,  W//2:] = 1
	label_map[ H//2:,:W//2] = 2
	label_map[ H//2:, W//2:] = 3

	id_to_class_name_map = { 0: 'sky', 1: 'road', 2: 'person', 3: 'horse'}

	class_idx = 0
	binary_img = class_idx == label_map
	polygons, has_holes = get_polygons_from_binary_img(binary_img)
	assert len(polygons) == 1
	assert has_holes == 0
	gt_poly = np.array(
		[
			[0, 0],
			[0, 1],
			[0, 2],
			[1, 2],
			[2, 2],
			[3, 2],
			[4, 2],
			[4, 1],
			[4, 0],
			[3, 0],
			[2, 0],
			[1, 0]
		], dtype=np.int32)
	assert np.allclose(polygons[0], gt_poly)

	class_idx = 1
	binary_img = class_idx == label_map
	polygons, has_holes = get_polygons_from_binary_img(binary_img)
	assert len(polygons) == 1
	assert has_holes == 0
	gt_poly = np.array(
		[
			[5, 0],
			[5, 1],
			[5, 2],
			[6, 2],
			[7, 2],
			[8, 2],
			[9, 2],
			[9, 1],
			[9, 0],
			[8, 0],
			[7, 0],
			[6, 0]
		], dtype=np.int32)
	assert np.allclose(polygons[0], gt_poly)

	class_idx = 2
	binary_img = class_idx == label_map
	polygons, has_holes = get_polygons_from_binary_img(binary_img)
	assert len(polygons) == 1
	assert has_holes == 0
	gt_poly = np.array(
		[
			[0, 3],
			[0, 4],
			[0, 5],
			[1, 5],
			[2, 5],
			[3, 5],
			[4, 5],
			[4, 4],
			[4, 3],
			[3, 3],
			[2, 3],
			[1, 3]
		], dtype=np.int32)
	assert np.allclose(polygons[0], gt_poly)

	class_idx = 3
	binary_img = class_idx == label_map
	polygons, has_holes = get_polygons_from_binary_img(binary_img)
	assert len(polygons) == 1
	assert has_holes == 0
	gt_poly = np.array(
		[
			[5, 3],
			[5, 4],
			[5, 5],
			[6, 5],
			[7, 5],
			[8, 5],
			[9, 5],
			[9, 4],
			[9, 3],
			[8, 3],
			[7, 3],
			[6, 3]
		], dtype=np.int32)
	assert np.allclose(polygons[0], gt_poly)
	# --- FOR VISUALIZATION ---
	# plt.imshow(label_map)
	# for polygon in polygons:
	# 	plt.scatter(polygon[:,0], polygon[:,1], 10, color='r')
	# plt.show()

def test_get_polygons_from_binary_img2():
	"""
	Create label map with two embedded circles. Each circle
	represents class 1 (the "person" class).
	"""
	H = 15
	W = 30
	img_rgb = np.ones((H,W,3), dtype=np.uint8)
	label_map = np.zeros((H,W), dtype=np.uint8)
	label_map[7,7]=1
	label_map[7,22]=1
	# only 2 pixels will have value 1
	mask_diff = np.ones_like(label_map).astype(np.uint8) - label_map

	# Calculates the distance to the closest zero pixel for each pixel of the source image.
	distance_mask = cv2.distanceTransform(mask_diff, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
	distance_mask = distance_mask.astype(np.float32)
	label_map = (distance_mask <= 5).astype(np.uint8)

	id_to_class_name_map = { 0: 'road', 1: 'person' }

	binary_img = label_map == 1
	polygons, has_holes = get_polygons_from_binary_img(binary_img)
	assert len(polygons) == 2
	assert has_holes == 0
	# right side circle
	gt_poly1 = np.array(
		[
			[ 7,  2],
			[ 6,  3],
			[ 5,  3],
			[ 4,  3],
			[ 3,  4],
			[ 3,  5],
			[ 3,  6],
			[ 2,  7],
			[ 3,  8],
			[ 3,  9],
			[ 3, 10],
			[ 4, 11],
			[ 5, 11],
			[ 6, 11],
			[ 7, 12],
			[ 8, 11],
			[ 9, 11],
			[10, 11],
			[11, 10],
			[11,  9],
			[11,  8],
			[12,  7],
			[11,  6],
			[11,  5],
			[11,  4],
			[10,  3],
			[ 9,  3],
			[ 8,  3]
		], dtype=np.int32)

	# left side circle
	gt_poly2 = np.array(
		[
			[22,  2],
			[21,  3],
			[20,  3],
			[19,  3],
			[18,  4],
			[18,  5],
			[18,  6],
			[17,  7],
			[18,  8],
			[18,  9],
			[18, 10],
			[19, 11],
			[20, 11],
			[21, 11],
			[22, 12],
			[23, 11],
			[24, 11],
			[25, 11],
			[26, 10],
			[26,  9],
			[26,  8],
			[27,  7],
			[26,  6],
			[26,  5],
			[26,  4],
			[25,  3],
			[24,  3],
			[23,  3]
		], dtype=np.int32)
	assert np.allclose(polygons[1], gt_poly1)
	assert np.allclose(polygons[0], gt_poly2)
	
	# Will get exterior rectangle, and then bounding areas on the two holes
	binary_img = label_map == 0
	polygons, has_holes = get_polygons_from_binary_img(binary_img)
	assert len(polygons) == 3
	assert has_holes == 1

	gt_poly_outer = np.array(
		[
			[ 0,  0],
			[ 0,  1],
			[ 0,  2],
			[ 0,  3],
			[ 0,  4],
			[ 0,  5],
			[ 0,  6],
			[ 0,  7],
			[ 0,  8],
			[ 0,  9],
			[ 0, 10],
			[ 0, 11],
			[ 0, 12],
			[ 0, 13],
			[ 0, 14],
			[ 1, 14],
			[ 2, 14],
			[ 3, 14],
			[ 4, 14],
			[ 5, 14],
			[ 6, 14],
			[ 7, 14],
			[ 8, 14],
			[ 9, 14],
			[10, 14],
			[11, 14],
			[12, 14],
			[13, 14],
			[14, 14],
			[15, 14],
			[16, 14],
			[17, 14],
			[18, 14],
			[19, 14],
			[20, 14],
			[21, 14],
			[22, 14],
			[23, 14],
			[24, 14],
			[25, 14],
			[26, 14],
			[27, 14],
			[28, 14],
			[29, 14],
			[29, 13],
			[29, 12],
			[29, 11],
			[29, 10],
			[29,  9],
			[29,  8],
			[29,  7],
			[29,  6],
			[29,  5],
			[29,  4],
			[29,  3],
			[29,  2],
			[29,  1],
			[29,  0],
			[28,  0],
			[27,  0],
			[26,  0],
			[25,  0],
			[24,  0],
			[23,  0],
			[22,  0],
			[21,  0],
			[20,  0],
			[19,  0],
			[18,  0],
			[17,  0],
			[16,  0],
			[15,  0],
			[14,  0],
			[13,  0],
			[12,  0],
			[11,  0],
			[10,  0],
			[ 9,  0],
			[ 8,  0],
			[ 7,  0],
			[ 6,  0],
			[ 5,  0],
			[ 4,  0],
			[ 3,  0],
			[ 2,  0],
			[ 1,  0]
		], dtype=np.int32)
	assert np.allclose(gt_poly_outer, polygons[0])

	# # # --- FOR VISUALIZATION ---
	# for polygon in polygons:
	# 	plt.imshow(label_map)
	# 	plt.scatter(polygon[:,0], polygon[:,1], 10, color='r')
	# 	plt.show()
	# 	plt.close('all')


def test_get_polygons_from_binary_img3():
	""" """
	rgb_img_fpath = f'{TEST_DATA_ROOT}/ADE20K_test_data/ADEChallengeData2016'
	rgb_img_fpath += '/images/training/ADE_train_00000001.jpg'
	label_fpath = f'{TEST_DATA_ROOT}/ADE20K_test_data/ADEChallengeData2016'
	label_fpath += '/annotations/training/ADE_train_00000001.png'
	label_img = imageio.imread(label_fpath)

	# fountain class
	class_idx = 105
	binary_img = label_img == class_idx
	polygons, has_holes = get_polygons_from_binary_img(binary_img)
	assert len(polygons) == 1
	assert np.sum(polygons) == 315790
	assert np.median(polygons[0][:,0]) == 342.0
	assert np.median(polygons[0][:,1]) == 430.0

	# # --- FOR VISUALIZATION ---
	# for polygon in polygons:
	# 	plt.imshow(label_img)
	# 	plt.scatter(polygon[:,0], polygon[:,1], 10, color='r')
	# 	plt.show()
	# 	plt.close('all')


def test_get_polygons_from_binary_img4():
	"""
	Test degenerate case (empty mask)
	"""
	rgb_img_fpath = f'{TEST_DATA_ROOT}/ADE20K_test_data/ADEChallengeData2016'
	rgb_img_fpath += '/images/training/ADE_train_00000001.jpg'
	label_fpath = f'{TEST_DATA_ROOT}/ADE20K_test_data/ADEChallengeData2016'
	label_fpath += '/annotations/training/ADE_train_00000001.png'
	label_img = imageio.imread(label_fpath)

	binary_img = np.zeros((480,640), dtype=np.uint8)
	polygons, has_holes = get_polygons_from_binary_img(binary_img)

	assert len(polygons) == 0
	assert has_holes == False

	# # --- FOR VISUALIZATION ---
	# for polygon in polygons:
	# 	plt.imshow(label_img)
	# 	plt.scatter(polygon[:,0], polygon[:,1], 10, color='r')
	# 	plt.show()
	# 	plt.close('all')


def test_polygon_to_mask_vline():
	"""
	Many thanks to jmsteitz for noting this case.
	horizontal line
	https://github.com/mseg-dataset/mseg-api/issues/7
	"""
	h, w = 8, 8
	polygon = [(2,2),(2,8)]
	mask = get_mask_from_polygon(polygon, h, w)
	gt_mask = np.array(
		[
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 1, 0, 0, 0, 0, 0],
			[0, 0, 1, 0, 0, 0, 0, 0],
			[0, 0, 1, 0, 0, 0, 0, 0],
			[0, 0, 1, 0, 0, 0, 0, 0],
			[0, 0, 1, 0, 0, 0, 0, 0],
			[0, 0, 1, 0, 0, 0, 0, 0]
		], dtype=np.uint8)
	assert np.allclose(gt_mask, mask)


def test_polygon_to_mask_hline():
	"""
	Many thanks to jmsteitz for noting this case.
	horizontal line
	https://github.com/mseg-dataset/mseg-api/issues/7
	"""
	h, w = 8, 8
	polygon = [(2,2),(8,2)]
	mask = get_mask_from_polygon(polygon, h, w)
	gt_mask = np.array(
		[
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 1, 1, 1, 1, 1, 1],
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0]
		], dtype=np.uint8)
	assert np.allclose(gt_mask, mask)

def test_write_six_img_grid_w_embedded_names() -> None:
	""" """
	id_to_class_name_map = {0: 'wall', 1: 'tv_monitor', 2: 'book'}

	rgb_fpath = f'{TEST_DATA_ROOT}/tv_image.jpg'
	rgb_img = cv2.imread(rgb_fpath)
	h,w,_ = rgb_img.shape # 260 x 454 x 3
	label_img = np.zeros((h,w), dtype=np.uint8)
	label_img[50:210,80:420] = 1

	pred = np.zeros((h,w), dtype=np.uint8)
	pred[50:210,80:420] = 2

	save_fpath = f'{TEST_DATA_ROOT}/dummy_6img_grid.jpg'
	write_six_img_grid_w_embedded_names(
		rgb_img,
		pred,
		label_img,
		id_to_class_name_map,
		save_fpath
	)
	assert Path(save_fpath).exists()
	os.remove(save_fpath)

if __name__ == '__main__':

	
	#test_convert_instance_img_to_mask_img_zero_start()
	#test_convert_instance_img_to_mask_img_nonzero_start()
	#test_form_contained_classes_color_guide_smokescreen()
	#test_form_mask_triple_dont_write_to_disk()

	# test_map_semantic_img_fast_uint16_stress_test()
	# test_map_semantic_img_fast()
	# test_map_semantic_img_fast_widenvalue()
	# test_map_semantic_img_fast_dontwidenvalue()
	#test_swap_px_inside_mask()

	# test_get_mask_from_polygon()
	# test_polygon_to_mask1()
	# test_polygon_to_mask2()

	#test_save_pred_vs_label_7tuple_100_strided_preds()
	# test_save_pred_vs_label_7tuple_short_side_60_img()
	# test_save_pred_vs_label_7tuple_shifted()
	# test_save_pred_vs_label_7tuple_uniform_random_preds()
	# # 
	# test_map_semantic_img_uint8()
	# test_map_semantic_img_uint16()
	# test_map_semantic_img_uint16_stress_test()
	# test_form_contained_classes_color_guide_smokescreen()
	#test_search_jittered_location_in_mask()
	# test_get_most_populous_class1()
	# test_get_most_populous_class2()

	# test_get_polygons_from_binary_img1()
	# test_get_polygons_from_binary_img2()
	#test_get_polygons_from_binary_img3()
	#test_get_polygons_from_binary_img4()

	#test_polygon_to_mask_vline()
	test_polygon_to_mask_hline()





