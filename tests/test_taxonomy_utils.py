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

TRAIN_DATASETS = [
	'mapillary-public65',
	'mapillary-public65-relabeled',
	'coco-panoptic-133',
	'coco-panoptic-133-relabeled',
	'ade20k-150',
	'ade20k-150-relabeled',
	'idd-39',
	'idd-39-relabeled',
	'cityscapes-34',
	'cityscapes-34-relabeled',
	'cityscapes-19',
	'cityscapes-19-relabeled',
	'bdd',
	'bdd-relabeled',
	'nyudepthv2-37',
	'scannet-37',
	'sunrgbd-37',
	'sunrgbd-37-relabeled'
]





def test_parse_entry_blank():
	""" """
	entry = ''
	classes, type = parse_entry(entry)

	assert classes == []
	assert type == 'none'


def test_parse_entry_brackets1():
	"""
	"""
	entry = '{computer, unlabeled}'
	classes, type = parse_entry(entry)
	assert type == 'dump'
	assert classes == ['computer', 'unlabeled']


def test_parse_entry_brackets2():
	"""
	"""
	entry = '{water, waterfall, lake, swimming pool}'
	classes, type = parse_entry(entry)
	assert type == 'dump'
	assert classes == ['water', 'waterfall', 'lake', 'swimming pool']


# def test_parse_entry_space_sep():
# 	"""
# 	Note: ADE20K class "conveyer" is typo of "conveyor"
# 	"""
# 	entry = 'conveyer belt'
# 	classes, type = parse_entry(entry)
# 	assert type == 'same'
# 	assert classes == ['conveyer belt']

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


# def test_taxonomy_converter():
# 	"""
# 	"""
# 	tc = TaxonomyConverter(is_unit_test=True, train_datasets=['mapillary_vistas_comm'], test_datasets=['coco'])
# 	assert tc.tsv_data.equals(read_tsv('test_data/unit_test_taxonomy.tsv'))

# 	depth_0_gt_dict = {}
# 	dict_is_equal(tc.name2id[0], depth_0_gt_dict)

# 	# depth_1_gt_dict = {'vehicle': 0, }
# 	# dict_is_equal(tc.name2id[1], depth_1_gt_dict)

# 	depth_2_gt_dict = {
# 		'vehicle->bicycle': 0, 'vehicle->caravan': 1, 'vehicle->car': 2, 'vehicle->van': 3, 
# 		'vehicle->motorcycle': 4, 'vehicle->airplane': 5, 'vehicle->bus': 6, 'vehicle->train': 7, 
# 		'vehicle->truck': 8, 'vehicle->trailer': 9, 'vehicle->watercraft': 10, 'vehicle->slow_wheeled_obj': 11
# 	}
# 	dict_is_equal(tc.name2id[2], depth_2_gt_dict)

# 	depth_3_gt_dict = {
# 		'vehicle->watercraft->boat': 0, 'vehicle->watercraft->ship': 1, 'vehicle->bicycle->bicycle_ext': 2, 
# 		'vehicle->caravan->caravan_ext': 3, 'vehicle->car->car_ext': 4, 'vehicle->van->van_ext': 5, 
# 		'vehicle->motorcycle->motorcycle_ext': 6, 'vehicle->airplane->airplane_ext': 7, 
# 		'vehicle->bus->bus_ext': 8, 'vehicle->train->train_ext': 9, 'vehicle->truck->truck_ext': 10, 
# 		'vehicle->trailer->trailer_ext': 11, 'vehicle->slow_wheeled_obj->slow_wheeled_obj_ext': 12
# 	}
# 	dict_is_equal(tc.name2id[3], depth_3_gt_dict)

# 	# gt_id_to_id_map = {
# 	# 	1: {80: 0, 81: 0, 82: 0, 83: 0, 84: 0, 85: 0, 86: 0, 87: 0, 88: 0, 89: 0, 90: 0, 91: 0, 255: 255}, 
# 	# 	2: {80: 0, 81: 10, 82: 6, 83: 2, 84: 1, 85: 4, 86: 7, 87: 255, 88: 9, 89: 8, 90: 255, 91: 11, 255: 255}, 
# 	# 	3: {80: 2, 81: 255, 82: 8, 83: 4, 84: 3, 85: 6, 86: 9, 87: 255, 88: 11, 89: 10, 90: 255, 91: 12, 255: 255}
# 	# }
# 	# assert dict_is_equal(tc.id_to_id_maps['mapillary_vistas_comm'][1], gt_id_to_id_map[1])
# 	# assert dict_is_equal(tc.id_to_id_maps['mapillary_vistas_comm'][2], gt_id_to_id_map[2])
# 	# assert dict_is_equal(tc.id_to_id_maps['mapillary_vistas_comm'][3], gt_id_to_id_map[3])


# def test_conv_weights_all_ones():
# 	"""
# 	"""
# 	tc = TaxonomyConverter(is_unit_test=True)
	
# 	# test pred transform, sum up to 1
# 	#rand = torch.randn(2, len(self.C[3]), 3, 3)

# 	rand = torch.ones(1,len(tc.C[3]), 2, 2)


# 	level3pred_conv = torch.ones(1,len(tc.C[3]), 2, 2)
# 	level2pred_conv = tc.conv3_to_2(level3pred_conv)

# 	assert np.allclose( level2pred_conv[0,:,0,0].numpy(), 
# 						np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 1.])
# 	)

# 	level1pred_conv = tc.conv2_to_1(level2pred_conv)


# def test_conv_weights_sumto1():
# 	"""
# 	If leaves sum to 1, then after bubble up sum, root should have probability 1.
# 	"""
# 	tc = TaxonomyConverter(is_unit_test=True)
	
# 	# test pred transform, sum up to 1
# 	rand = torch.randn(2, len(tc.C[3]), 3, 3)

# 	level3pred_conv = torch.nn.Softmax(dim=1)(rand)
# 	level2pred_conv = tc.conv3_to_2(level3pred_conv)
# 	level1pred_conv = tc.conv2_to_1(level2pred_conv)

# 	assert np.allclose(np.ones((2,1,3,3), dtype=np.int64), level1pred_conv.numpy() )


# def test_label_transform():
# 	"""
# 	85 is the motorcycle class, #6 in level 3, #4 in level 2, and #0 in level 1
# 	in our simple test taxonomy.
# 	"""
# 	tc = TaxonomyConverter(is_unit_test=True)
# 	label = torch.ones(4,4)*85
# 	label = label.type(torch.LongTensor)
# 	pred = tc.transform_label(label,'mapillary_vistas_comm')

# 	gt_level1_label = np.array(
# 		[
# 			[0, 0, 0, 0],
# 			[0, 0, 0, 0],
# 			[0, 0, 0, 0],
# 			[0, 0, 0, 0]
# 		]
# 	).astype(np.int64)
# 	assert np.allclose(pred[1].numpy(), gt_level1_label)

# 	gt_level2_label = np.array(
# 		[
# 			[4, 4, 4, 4],
# 			[4, 4, 4, 4],
# 			[4, 4, 4, 4],
# 			[4, 4, 4, 4]
# 		]
# 	).astype(np.int64)
# 	assert np.allclose(pred[2].numpy(), gt_level2_label)

# 	gt_level3_label = np.array(
# 		[
# 			[6, 6, 6, 6],
# 			[6, 6, 6, 6],
# 			[6, 6, 6, 6],
# 			[6, 6, 6, 6]
# 		]
# 	).astype(np.int64)
# 	assert np.allclose(pred[3].numpy(), gt_level3_label)


# def test_label_transform_unlabeled():
# 	"""
# 	Make sure 255 stays mapped to 255 at each level (to be ignored in cross-entropy loss).
# 	"""
# 	tc = TaxonomyConverter(is_unit_test=True)
# 	label = torch.ones(4,4)*255
# 	label = label.type(torch.LongTensor)
# 	pred = tc.transform_label(label,'mapillary_vistas_comm')

# 	gt_level1_label = 255 * np.ones((4,4), dtype=np.int64)
# 	assert np.allclose(pred[1].numpy(), gt_level1_label)

# 	gt_level2_label = 255 * np.ones((4,4), dtype=np.int64)
# 	assert np.allclose(pred[2].numpy(), gt_level2_label)

# 	gt_level3_label = 255 * np.ones((4,4), dtype=np.int64)
# 	assert np.allclose(pred[3].numpy(), gt_level3_label)


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




# # def test_taxonomy_depth():
# # 	"""
# # 	ASSERT THAT WE NEVER HAVE MORE THAN 2 ARROWS FOR ANY NODE
# # 	"""
# # 	pass



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


# def test_get_convolution_test():
# 	""" 
# 	TODO
# 	"""
# 	tc = TaxonomyConverter()
# 	dataset = 'voc2012'
# 	conv = tc.get_convolution_test(dataset)
# 	pdb.set_trace()
# 	assert isinstance(conv, torch.nn.Module)


# # test_get_convolution_test()



# def test_transform_predictions_test():
# 	"""
# 	TODO
# 	"""
# 	tc = TaxonomyConverter()
# 	# input = ''
# 	# dataset = ''
# 	# conv = tc.get_convolution_test(dataset)
# 	# output = tc.transform_predictions_test(input, dataset, conv)

# 	# output_gt = ''


# def test_conv_1x1():
# 	"""
# 	Implement simple matrix multiplication as 1x1 convolutions in PyTorch.

# 	[2]   [1 0 1 0] [1]
# 	[2] = [0 1 0 1] [1]
# 	[4]   [1 1 1 1] [1]
# 	                [1]
# 	"""
# 	conv = torch.nn.Conv2d(in_channels=4,  out_channels=3, kernel_size=1, padding=0, bias=False)
# 	conv.weight.data.fill_(0)

# 	for param in conv.parameters():
# 		param.requires_grad = False

# 	parent_child_map = [
# 		(0,0), 
# 		(0,2), 
# 		(1,1), 
# 		(1,3), 
# 		(2,0), 
# 		(2,1), 
# 		(2,2), 
# 		(2,3) 
# 	]

# 	for (parent, child) in parent_child_map:
# 		conv.weight[parent][child] = 1

# 	x = np.array([0,1,0,1]).reshape(1,4,1,1).astype(np.float32)
# 	x = torch.from_numpy(x)
# 	y = conv(x)
# 	y_gt = np.array([0,2,2]).reshape(1,3,1,1).astype(np.float32)
# 	y_gt = torch.from_numpy(y_gt)
# 	assert torch.allclose(y, y_gt)

# 	x = torch.ones(1,4,1,1).type(torch.FloatTensor)
# 	y = conv(x)
# 	y_gt = np.array([2,2,4]).reshape(1,3,1,1).astype(np.float32)
# 	y_gt = torch.from_numpy(y_gt)
# 	assert torch.allclose(y, y_gt)


# test_all_classes_present_test_datasets()


# def test_bubble_sum():
# 	"""
# 	"""
# 	tc = TaxonomyConverter(is_unit_test=True)
	
# 	# test pred transform, sum up to 1
# 	#rand = torch.randn(2, len(self.C[3]), 3, 3)

# 	rand = torch.ones(1,len(tc.C[3]), 2, 2)

# 	#level3pred_conv = torch.nn.Softmax(dim=1)(rand)
# 	level3pred_conv = torch.ones(1,len(tc.C[3]), 2, 2)
# 	# level2pred = bubble_sum(rand, level=2)
# 	# level1pred = bubble_sum(level2pred, level=1)
# 	level2pred_conv = tc.conv3_to_2(level3pred_conv)

# 	assert np.allclose( level2pred_conv[0,:,0,0].cpu(), 
# 						np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 1.])
# 	)

# 	level1pred_conv = tc.conv2_to_1(level2pred_conv)




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





# def test_extend_tree():
# 	"""
# 	"""
# 	extend_tree():


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



# def test_bubble_sum():
# 	"""
# 	"""
# 	output = bubble_sum(input, level)


#test_taxonomy_converter()

