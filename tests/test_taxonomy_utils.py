#!/usr/bin/python3

""" Verify that the taxonomy functions are working correctly. """

import numpy as np
import pandas as pd
import pdb
import time
import torch

from mseg.utils.test_utils import dict_is_equal
from mseg.utils.mask_utils import form_label_mapping_array_pytorch

from mseg_semantic.taxonomy.taxonomy_converter import (
	TaxonomyConverter,
	parse_entry,
	parse_uentry
)

# from mseg_semantic.taxonomy.utils import (
# 	parse_entry,
# 	load_dataset_class_names,
# 	TaxonomyConverter
# )



# def read_tsv(tsv_fpath):
# 	return pd.read_csv(tsv_fpath, sep='\t', keep_default_na=False)

# def test_read_tsv(verbose=False):
# 	"""
# 	Parameter `keep_default_na`:
# 		Whether or not to include the default NaN values when parsing the data
# 		If keep_default_na is False, and na_values are not specified, no strings will be parsed as NaN.
# 	"""
# 	tsv_fname = 'test_data/unit_test_taxonomy.tsv'
# 	tsv_data = read_tsv(tsv_fname)

# 	gt_dataset_names = set([
# 		'mapillary_vistas_comm','coco','ade20k','scannet_37','universal',
# 		'nyudepthv2_37','idd','voc2012','cityscapes','kitti','wilddash','camvid','viper'
# 	])

# 	assert set(tsv_data.keys()) == gt_dataset_names

# 	for i, row_value in tsv_data.iterrows():
# 		if verbose:
# 			print(f'For Universal Entry:', row_value['universal'])
# 		for taxonomy_name in tsv_data.keys():
# 			if taxonomy_name == 'universal':
# 				continue
# 			if verbose:
# 				print(f'\t{taxonomy_name.rjust(20)}: {row_value[taxonomy_name]}')






# def form_tiled_semantic_img(num_classes, output_sz):
# 	"""
# 		Args:
# 		-	

# 		Returns:
# 		-	
# 	"""
# 	h, w = output_sz
# 	num_copies = int(h * w / num_classes)
# 	assert num_copies * num_classes == h * w
# 	arr_1d = np.array(range(num_classes))
# 	arr_1d = np.tile(arr_1d,(1,num_copies))
# 	arr_2d = arr_1d.reshape(h,w)

# 	label = torch.from_numpy(arr_2d.copy() )
# 	label = label.type(torch.LongTensor)

# 	return label, arr_2d


# def test_label_transform_fast_speed_test():
# 	"""
# 	Test fast method on large example. Map 100 classes to a 
# 	different 100 classes with a fast array conversion.

# 	Took 0.0034 sec, which is 3.4 milliseconds
# 	"""
# 	dataset_name = 'mapillary_vistas_comm'
# 	tc = TaxonomyConverter()

# 	id_to_id_map = {}
# 	for i in range(80):
# 		id_to_id_map[i] = i + 1

# 	for level in [1,2,3]:
# 		tc.label_mapping_arr_dict[dataset_name][level] = form_label_mapping_array_pytorch(id_to_id_map)

# 	label, arr_2d = form_tiled_semantic_img(num_classes=80, output_sz=(600,600))

# 	start = time.time()
# 	pred = tc.transform_label(label, dataset_name)
# 	end = time.time()
# 	print(f'Took {end-start} sec')

# 	gt_pred = arr_2d + 1

# 	assert np.allclose(pred[1].numpy(), gt_pred)
# 	assert np.allclose(pred[2].numpy(), gt_pred)
# 	assert np.allclose(pred[3].numpy(), gt_pred)


# def test_label_transform_slow_speed_test():
# 	"""
# 	Takes 0.30538415908813477 sec
# 	"""
# 	dataset_name = 'mapillary_vistas_comm'
# 	tc = TaxonomyConverter()

# 	id_to_id_map = {}
# 	for i in range(80):
# 		id_to_id_map[i] = i + 1

# 	for level in [1,2,3]:
# 		tc.id_to_id_maps[dataset_name][level] = id_to_id_map

# 	label, arr_2d = form_tiled_semantic_img(num_classes=80, output_sz=(600,600))

# 	start = time.time()
# 	pred = tc.transform_label_slow(label, dataset_name)
# 	end = time.time()
# 	print(f'Took {end-start} sec')

# 	gt_pred = arr_2d + 1

# 	assert np.allclose(pred[1].numpy(), gt_pred)
# 	assert np.allclose(pred[2].numpy(), gt_pred)
# 	assert np.allclose(pred[3].numpy(), gt_pred)


# def test_label_transform_slow_fast_equal():
# 	"""
# 	"""
# 	dataset_name = 'mapillary_vistas_comm'
# 	tc = TaxonomyConverter()

# 	id_to_id_map = {}
# 	for i in range(80):
# 		id_to_id_map[i] = i + 1

# 	for level in [1,2,3]:
# 		tc.id_to_id_maps[dataset_name][level] = id_to_id_map

# 	for level in [1,2,3]:
# 		tc.label_mapping_arr_dict[dataset_name][level] = form_label_mapping_array_pytorch(id_to_id_map)

# 	label, arr_2d = form_tiled_semantic_img(num_classes=80, output_sz=(600,600))

# 	pred = tc.transform_label_slow(label, dataset_name)
# 	pred2 = tc.transform_label(label, dataset_name)

# 	gt_pred = arr_2d + 1

# 	assert np.allclose(pred[1].numpy(), gt_pred)
# 	assert np.allclose(pred[2].numpy(), gt_pred)
# 	assert np.allclose(pred[3].numpy(), gt_pred)

# 	for level in [1,2,3]:
# 		assert torch.allclose(pred[level], pred2[level])
# 		assert pred[level].dtype == pred2[level].dtype
# 		assert pred[level].shape == pred2[level].shape

# 	assert set(pred.keys()) == set(pred2.keys())



# def check_taxonomytsv_vs_dataset_names(dataset_name: str) -> None:
# 	''' Check the category names with registered names on the sheet,
# 		needs to be exactly the same
# 		dataset: dataset name string

# 		Args:
# 		-	dataset_name: string representing name of dataset

# 		Returns:
# 		-	None
# 	'''
# 	print(f'==========checking {dataset_name}========')
# 	all_names = load_dataset_class_names(dataset_name)
# 	reg_names = []

# 	tc = TaxonomyConverter()
# 	for row_value in tc.tsv_data[dataset_name]:
# 		cls_names, _ = parse_entry(row_value)

# 		if dataset_name in tc.train_datasets:
# 			for cls in cls_names:
# 				# ensure no duplicate names in training set
# 				assert(cls not in reg_names)
# 		reg_names += cls_names

# 	if dataset_name == 'voc2012':
# 		reg_names += ['background']

# 	assert set(reg_names) == set(all_names)

# 	# check all registered values are in original class names
# 	for reg_name in reg_names:
# 		if reg_name not in all_names:
# 			print(f'{reg_name} is not in the original classes, please check')

# 	# # check the reverse direction
# 	for name in all_names:
# 		if name not in reg_names:
# 			print(f'{name} is not registered, please check')

# 	print('-------checking complete----------')



# def test_all_classes_present_train_datasets():
# 	"""
# 	"""
# 	tc = TaxonomyConverter()
# 	for train_dataset in ['mapillary_vistas_comm', 'scannet_37', 'coco', 'ade20k']:
# 		check_taxonomytsv_vs_dataset_names(train_dataset)



# # def test_all_classes_present_test_datasets():
# # 	"""
# # 	"""
# # 	tc = TaxonomyConverter()
# # 	for train_dataset in ['voc2012', 'cityscapes', 'idd', 'nyudepthv2_37', 'wilddash']:
# # 		pdb.set_trace()
# # 		check_taxonomytsv_vs_dataset_names(train_dataset)


# def test_transform_d2utest():
# 	""" 
# 	Smokescreen
# 	"""
# 	tc = TaxonomyConverter()
# 	dataset = 'voc2012'
# 	name2pair, id2pair = tc.transform_d2utest(dataset)

# 	for name, pair in name2pair.items():
# 		assert len(pair[0]) == 2
# 		level, id = pair[0]
# 		assert id >= 0
# 		assert id < len(tc.name2id[level])

# 	for id, pair in id2pair.items():
# 		assert len(pair[0]) == 2
# 		level, id = pair[0]
# 		assert id >= 0
# 		assert id < len(tc.name2id[level])



# def test_bubble_sum():
# 	"""
# 	"""
# 	tc = TaxonomyConverter(is_unit_test=True)
# 	level3_input = torch.ones(2,len(tc.C[3]),2,2)
# 	level2_input = tc.bubble_sum(level3_input, level=3)
# 	level1_input = tc.bubble_sum(level2_input, level=2)
# 	pdb.set_trace()

# def test_build_universal_tax():
# 	"""
# 	"""
# 	data = pd.DataFrame({
# 		'coco' : [	'backpack', 
# 					'umbrella', 
# 					'(handbag)', 
# 					'tie', 
# 					'door-stuff' ]
# 		'ade20k': [ '', 
# 					'', 
# 					'bag'
# 					''
# 					'[door, screen door]'	],
# 		'universal': [	'accessory->backpack', 
# 						'accessory->umbrella', 
# 						'accessory->bag', 
# 						'accessory->tie', 
# 						'furniture->door'	]
# 	})

# 	'.test_data/unit_test_taxonomy.tsv'


# 	build_universal_tax(data, C, name2id)




# def test_transform_d2u():
# 	"""
# 	"""

# 	name_map, id_to_id_map = transform_d2u(dataset)


# def test_transform_d2utest():
# 	"""
# 	"""
# 	name2pair, id2pair = transform_d2utest(dataset)



# def test_get_convolution_test():
# 	"""
# 	"""
# 	conv = get_convolution_test(dataset)



# def test_transform_predictions_test():
# 	"""
# 	"""
# 	output = transform_predictions_test(input, dataset, conv)


#test_taxonomy_converter()

