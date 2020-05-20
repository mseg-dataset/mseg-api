#!/usr/bin/python3

import time
start = time.time()
from collections import defaultdict, namedtuple
import cv2
import numpy as np
import os
import pandas as pd
import pdb
import torch
import torch.nn as nn
from mseg.utils.names_utils import load_class_names
# import pickle

os.environ['PYTHONPATH'] = './'

from typing import List, Mapping, Tuple

end = time.time()
print(end-start)
#import vis_utils.mask_utils

"""
Node needs self.parent and self.children pointers
"""


class StupidTaxonomyConverter():
	"""
	Class that can map training data to a universal taxonomy, and universal taxonomy
	predictions at inference time to specific data taxonomies.
	"""
	def __init__(self, 
				is_unit_test: bool = False, version=1.0, finetune=None,
				train_datasets: List[str] = [
				'ade20k-150',
				'bdd',
				'cityscapes-19',
				'coco-panoptic-133',
				'idd-39',
				'mapillary-public65', 
				'sunrgbd-37',
				],
				test_datasets: List[str] = [
				'camvid-11',
				'kitti-19',
				'pascal-context-60',
				'scannet-20',
				'voc2012',
				'wilddash-19',
				],
				finetune_dataset='wilddash'): # 'camvid-qvga', CAMVID HAS PERSON/RDIER MUST BE RESOLVED IN UNIVERSAL
		"""
			TODO: make dataset an argument, and rename to dataset_names

			`C` is a map from tree depth/level (integer) to Node list
			`tsv_data` is a Pandas DataFrame
			`name2id` is a map from tree depth/level to a dictionary that 
					provides (node name)->(node ID) mappings.
		"""


		print('------initializing tc----------')
		self.is_unit_test = is_unit_test
		self.ignore_label = 255 # still 255 since in png files those are 255
		self.finetune = finetune

		self.dataset_names = {}

		# Find universal name from spreadsheet TSV index.
		self.uid2uname = {}
		# for C, finding spreadsheet TSV index from universal name.
		self.uname2uid = {}

		self.train_datasets = train_datasets
		self.test_datasets = test_datasets
		self.finetune_dataset = finetune_dataset

		for d in (self.train_datasets + self.test_datasets):
			if d.endswith('-sr'):
				folder_name = d.replace('-sr', '')
			else:
				folder_name = d
			self.dataset_names[d] = load_class_names(folder_name)

		self.build_universal_tax()
		self.classes = len(self.uid2uname) + 1 # including ignored label（id, 255), since total id > 255. (note previously it's -1)

		# assert (self.classes < 255)

		# dataset to universal id maps
		# self.dataset_names = {}
		self.dataloaderid_to_uid_maps = {}
		self.convs = {}
		self.softmax = nn.Softmax(dim=1)

		for d in self.train_datasets:
			print(f'Mapping {d} -> universal')
			dataloaderid2uid_map = self.transform_d2u(d)
			self.dataloaderid_to_uid_maps[d] = dataloaderid2uid_map

		# for d in self.test_datasets:
		# 	tax_name = self.find_tax_name(d, split = 'test')  #
		# 	self.dataset_names[d] = load_class_names(tax_name)

		for d in self.test_datasets:
		# 	tax_name = self.find_tax_name(d, split = 'test')  #
			print(f'Creating 1x1 conv for test: universal -> {d}')
			self.convs[d] = self.get_convolution_test(d)


		# assert(len(self.uname2uid) < 255)

		self.label_mapping_arr_dict = self.form_label_mapping_arrs() # whether finetune or not is handled inside function

		# pdb.set_trace()



	def build_universal_tax(self, verbose: bool = False) -> None:
		''' 
			
			Build a flat tree at C[1] by adding each `universal` taxonomy entry from the TSV here.
			Data structures to modify are passed in by reference.

			Args:
			-	verbose: boolean, whether to print info during execution

			Returns:
			-	None
		'''

		# for train_d in self.train_datasets:
			# classes = load_class_names
		id = 0
		for d in self.train_datasets:
			classes = self.dataset_names[d]
			for c in classes:
				lowercase = c.lower()
				if lowercase not in self.uname2uid.keys():
					self.uname2uid[lowercase] = id
					self.uid2uname[id] = lowercase
					id += 1

					if id == self.ignore_label: # want to reserve 255 as ignore_label
						id += 1
				else:
					# print(f'class {c} already in, lowercase is {lowercase}')
					pass
		# pdb.set_trace()


	def transform_d2u(self, dataset):
		''' Perform training data transformation.

			For only one specific dataset, map each of its class names to one universal name, and 
			id to one level of id.

			Args:
			-	dataset: string representing name of dataset

			Returns:
			-	name_map: Provides the mapping from a specific class name (node) 
					to a universal taxonomy class name (node)
			-	id_to_id_map: provide mapping to ancestor at each level of tree, up to depth 1
		'''
		dataloaderid2uid_map = {}
		# pdb.set_trace()
		classes = self.dataset_names[dataset]
		for i, c in enumerate(classes):
			lowercase = c.lower()
			dataloaderid2uid_map[i] = self.uname2uid[lowercase]
			# print(lowercase, )
			# print(lowercase)
		# pdb.set_trace()

		dataloaderid2uid_map[self.ignore_label] = self.ignore_label

		return dataloaderid2uid_map


	def transform_d2f(self, dataset):
		''' Perform training data transformation.

			For only one specific dataset, map each of its class names to one fine-tuning dataset name, and 
			id to one level of id.

			Args:
			-	dataset: string representing name of dataset

			Returns:
			-	name_map: Provides the mapping from a specific class name (node) 
					to a universal taxonomy class name (node)
			-	id_to_id_map: provide mapping to ancestor at each level of tree, up to depth 1
		'''

		dname2fname_map = {} # name to name map from (dataset -> fine-tune name)
		dname2fid_map = {}# name to id map from (dataset -> fine-tune ID)
		dataloaderid2fname_map = {}
		dataloaderid2fid_map = {}

		fclassnames = self.dataset_names[self.finetune_dataset]
	 
		for index, row in self.tsv_data.iterrows():
			classes, type = parse_entry(row[dataset])
			# finetune_tax
			f_name, f_type = parse_test_entry(row[self.finetune_dataset])

			if type == 'none': # no entry in train dataset, nothing to do
				continue

			if f_type == 'none': # has entry(ies) in train, no entry in fine-tune, map to unlabeled
				for cls in classes:
					dname2fname_map[cls] = 'unlabeled'
					dname2fid_map[cls] = self.ignore_label   #

			else: # entry in train, entry in finetune
				fid = fclassnames.index(f_name)
				for cls in classes:
					dname2fname_map[cls] = f_name
					dname2fid_map[cls] = fid

					# print(f'{dataset}: {cls} --> {f_name}')

		### special case handling
		if dataset.startswith('cityscapes') and self.finetune_dataset.startswith('wilddash'): # same taxonomy, direct mapping
			for i, name in enumerate(self.dataset_names[self.finetune_dataset]):
				dname2fname_map[name] = name
				dname2fid_map[name] = i

		if dataset.startswith(self.finetune_dataset): # if fine-tune on the dataset itself, no transform
			for i, name in enumerate(self.dataset_names[self.finetune_dataset]):
				dname2fname_map[name] = name
				dname2fid_map[name] = i


		# Map the PNG value (dataloader ID) to an ID in the name2id data structure
		all_dataset_names = self.dataset_names[dataset]
		for i, dname in enumerate(all_dataset_names):
			fid = dname2fid_map[dname]
			f_name = dname2fname_map[dname]
			dataloaderid2fid_map[i] = fid
			dataloaderid2fname_map[i] = f_name

		dataloaderid2fid_map[self.ignore_label] = self.ignore_label
		dataloaderid2fname_map[self.ignore_label] = 'unlabeled'
		dname2fname_map['unlabeled'] = 'unlabeled'
		dname2fid_map['unlabeled'] = self.ignore_label

		# print(dataset, dataloaderid2fid_map)

		return dname2fid_map, dataloaderid2fid_map


	def get_finetune_newconv(self, old_conv):
		old_conv = old_conv.cpu()
		in_dim = old_conv.weight.size(1)
		out_dim = self.finetune_classes
		new_conv = torch.nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1)
		new_conv.weight.data.fill_(0)
		new_conv.bias.data.fill_(0)

		# old_conv = old_conv.cpu()

		_, dataloaderid2pair = self.transform_d2utest(self.finetune_dataset)



		# for i, classname in enumerate(self.dataset_names[self.finetune_dataset]):
			# print('F class: ', classname)
			# uid, uname = dataloaderid2pair[i]
		for f_id, in_pair_list in dataloaderid2pair.items():
			# print('F:', self.dataset_names[self.finetune_dataset][f_id])
			for (u_id, u_name) in in_pair_list:
				# print(u_name)
				new_conv.weight[f_id].data += old_conv.weight[u_id]
				new_conv.bias[f_id].data += old_conv.bias[u_id]
			new_conv.weight[f_id].data /= len(in_pair_list)
			new_conv.bias[f_id].data /= len(in_pair_list)

		return new_conv

			# new_conv.weights[i]




	def form_label_mapping_arrs(self)->Mapping[str, Mapping[int, np.ndarray]]:
		""" Cache conversion maps so we will later be able to perform fast 
			grayscale->grayscale mapping for training data transformation.

			Args:
			-	None

			Returns:
			-	label_mapping_arr_dict: map dataset names to inner dictionaries. Inner dictionary is a map from
					level (integers) to mapping arrays (numpy arrays)
		"""
		label_mapping_arr_dict = {}
		from mseg.utils.mask_utils import form_label_mapping_array_pytorch, map_semantic_img_fast_pytorch

		for dataset in self.train_datasets:
			# set default
			label_mapping_arr_dict[dataset] = {}
			if self.finetune:
				idmap = self.dataloaderid_to_fid_maps[dataset]
			else:
				idmap = self.dataloaderid_to_uid_maps[dataset]
			label_mapping_arr_dict[dataset] = form_label_mapping_array_pytorch(idmap)

		return label_mapping_arr_dict
		


	def transform_label(self, label, dataset) -> Mapping[int, torch.Tensor]: # function to be called outside for training
		"""
			Perform fast grayscale->grayscale mapping for training data transformation.

			TODO: decide if to do it with Pytorch Tensor or with Numpy array.

			Args:
			-	label: Pytorch tensor on the cpu with dtype Torch.LongTensor, 
					representing a semantic image, according to PNG/dataloader format.
			-	dataset

			Returns:
			-	labels: vector label for each depth in the tree, compatible with softmax
		"""
		# labels = {}
		from mseg.utils.mask_utils import form_label_mapping_array_pytorch, map_semantic_img_fast_pytorch


		if self.finetune:
			label = map_semantic_img_fast_pytorch(label, self.label_mapping_arr_dict[dataset])
		else:
			label = map_semantic_img_fast_pytorch(label, self.label_mapping_arr_dict[dataset])
		return label


	def get_convolution_test(self, dataset):
		""" Explicitly for inference on our test datasets.
			Args:
			-	dataset

			Returns:
			-	conv
		"""
		dataloaderid2pair = self.transform_d2utest(dataset)
		in_channel = self.classes 
		# out_channel = len(self.dataset_names[dataset]) # including possible background class # assuming no "unlabeled" in the names now. (no scannet testing)
		out_channel = len(load_class_names(dataset))

		conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
		conv.weight.data.fill_(0)

		for param in conv.parameters():
			param.requires_grad = False

		# out_id: test dataset id, in_pair: universal (id, name)
		for out_id, in_pair_list in dataloaderid2pair.items():
			for (in_id, in_name) in in_pair_list:
				conv.weight[out_id][in_id] = 1

		return conv


	def transform_d2utest(self, dataset):
		''' Explicitly for inference on our test datasets. Transform each universal entry to a list of pairs, by id and name.

			TODO: Make a class for these dicts to make straightforward data structures.

			Args:
			-	dataset:

			Returns:
			-	name2pair:
			-	id2pair:
		'''
		# all_names = self.dataset_names[dataset]
		all_names = self.dataset_names[dataset]
		dname2pair = defaultdict(list) # test -> universal
		dataloaderid2pair = defaultdict(list) # test -> universal a list of tuples of (level, id) that needs to be summed up to get the final probability

		for i, cls in enumerate(all_names):
			# pdb.set_trace()
			lowercase = cls.lower()
			if lowercase in self.uname2uid:
				id = self.uname2uid[lowercase]
				dataloaderid2pair[i].append([id, lowercase])
				# print(id, lowercase)

		return dataloaderid2pair


	def transform_predictions_test(self, input, dataset): # function to be called outside
		""" Explicitly for inference on our test datasets.

			Args:
			-	input: after softmax probability, of universal
			-	dataset:

			Returns:
			-	output: 
		"""

		input = nn.Softmax(dim=1)(input)
		# sumb = input.sum(1)

		# self.convs[dataset].cuda()
		# print(f"Using the conv for {dataset}")
		output = self.convs[dataset](input)

		if 'background' in self.dataset_names[dataset]: # background is 1 minus all other probabilitys, not summing all other nodes
			background_id = self.dataset_names[dataset].index('background')
			all_other = torch.cat([output[:, :background_id, :, :], output[:, (background_id+1):, :, :]], dim=1)
			output[:, background_id, :, :] = 1 - torch.sum(all_other, dim=1)

		# output = 
		# sum1 = torch.sum(output, dim=1, keepdim=True) # adding this does not change much
		# output /= sum1

		return output


	def transform_predictions_universal(self, input, dataset): # function to be called outside
		""" Explicitly for inference on our test datasets.

			Args:
			-	input: after softmax probability, of universal
			-	dataset:

			Returns:
			-	output: 
		"""

		# output = nn.Softmax(dim=1).cuda()(input)
		output = self.softmax(input)
		# id_maps = self.dataloaderid_to_uid_maps[dataset] # from train to universal. do this zero out or not does not affect when training and testing on same dataset.

		# nonzero_class_ids = set(id_maps.values())
		# nonzero_class_ids = [x for x in nonzero_class_ids if x != self.ignore_label]
		# zero_class_ids = [x for x in range(self.classes) if x not in nonzero_class_ids]
		# output[:, torch.LongTensor(zero_class_ids), :, :].zero_()

		return output

	def exclude_universal_ids(self, dataset):
		id_maps = self.dataloaderid_to_uid_maps[dataset] # from train to universal. do this zero out or not does not affect when training and testing on same dataset.
		nonzero_class_ids = set(id_maps.values())
		zero_class_ids = [x for x in range(self.classes) if x not in nonzero_class_ids]


		return zero_class_ids



	def check_complete(self, dataset, values=None, split='train'):
		''' check the category names with registered names on the sheet,
			needs to be exactly the same
			dataset: dataset name string
		'''
		print(f'==========checking {dataset}========')
		all_names = self.dataset_names[dataset]

		tax_name = self.find_tax_name(dataset, split)

		reg_names = []
		if values is None:
			values = self.tsv_data[tax_name]
		for row_value in values:
			# print(row_value)
			cls_names, type = parse_entry(row_value)

			if split == 'train': # no duplicate names in training set
				for cls in cls_names:
					# print(cls)
					assert(cls not in reg_names)
			reg_names += cls_names

		# check all registered values are in original class names
		complete = True
		for reg_name in reg_names:
			if reg_name not in all_names:
				print(f'{reg_name} is not in the original classes, please check')
				complete = False

		# # check the reverse direction
		for name in all_names:
			if name not in reg_names:
				print(f'{name} is not registered, please check')
				if name != 'background': # allow background to be not registered
					complete = False

		print('-------checking complete----------')

	def check_map(self, dataset):
		all_dataset_names = load_class_names(dataset)
		for i, name in enumerate(all_dataset_names):
			uid = self.id_to_id_maps[dataset][i]
			uname = self.id2name[uid]
			print(f'{name}, uname: {uname}, id: {i}, uid: {uid}')




def check_map_test(dataset):
	""" """
	print(f'-----------checking test {dataset} ----------------')
	name2pair, id2pair = transform_d2utest(dataset)

	for key, value in id2pair.items():
		print(dataset_names[dataset][key], ' includes: ')
		# print(dataset_names[dataset][key], 'level: ', value[0], ' name: ', self.C[value[0]][value[1]].name)
		for v in value:
			print('level: ', v[0], ' name: ', self.C[v[0]][v[1]].name)
	print(f'-----------checking test {dataset} complete ----------------')



	# original_cls = [(dataset_names[dataset][x.item()] if x.item() != 255 else 'unlabeled') for x in label]
	# level1_cls = [self.C[1][x.item()].name if x.item() != 255 else 'unlabeled' for x in labels[1]]
	# level2_cls = [self.C[2][x.item()].name if x.item() != 255 else 'unlabeled' for x in labels[2]]
	# level3_cls = [self.C[3][x.item()].name if x.item() != 255 else 'unlabeled' for x in labels[3]]

	# print(f'{original_cls}\n {level1_cls}\n {level2_cls}\n {level3_cls}')
	# return labels


def run_verification_tests():
	"""
	"""
	# check_complete('ade20k')
	# check_complete('coco')
	# check_complete('voc2012')
	# check_complete('cityscapes')
	# check_complete('mapillary_vistas_comm')
	# check_complete('idd')
	
	check_complete('nyudepthv2_37')

	# check_map('mapillary_vistas_comm')
	check_map('scannet_37')

	# check_map_test('cityscapes')
	# check_map_test('voc2012')
	# check_map_test('idd')

	# test label transform
	label_test = torch.LongTensor([1,2,3,4,255])

	labels = tc.transform_label(label_test, 'coco')
	print(labels)

	# check_map('coco')
	# test label transform
	# # self.C[3]

	# # A = torch.log(level3pred)
	# B = nn.LogSoftmax(dim=1)(rand)
	# name2pair, id2pair = transform_d2utest('voc2012')
	conv = get_convolution_test('voc2012')
	# check_map('ade20k')
	# check_map_test('voc2012')
	# check_map_test('cityscapes')





# debuggings
if __name__ == '__main__':
	import torch
	# DOWNLOAD = True

	label_test = torch.LongTensor([1,2,3,4,255])

	# version = 1.6
	tc = StupidTaxonomyConverter(version=0, finetune=False, finetune_dataset='wilddash')
	print(tc.classes)

	pdb.set_trace()
	# print(tc.classes)
	# old_conv = torch.nn.Conv2d(512, 183, 1, 1).cuda()

	# pdb.set_trace()
	# new_conv = tc.get_finetune_newconv(old_conv)

	# pickle_file = f'taxonomy/taxonomy_{version}.pkl'
	# with open(pickle_file, 'wb') as f:
	# 	pickle.dump(tc, f, protocol=pickle.HIGHEST_PROTOCOL)
	# print(pickle_file)

	# pickle_file = f'taxonomy/taxonomy_{version}.pkl'
	# with open(pickle_file, 'rb') as f:
	# 	tc = pickle.load(f)

	# pdb.set_trace()



	# tc.get_subset_class_train('coco')
	# tc.get_convolution_universal('coco')

	# labels = tc.transform_label(label_test, 'coco')
	# print('classes', len(tc.name2id) - 1)

	# rand = torch.randn(2, tc.classes, 3, 3)
	# B = nn.Softmax(dim=1)(rand)
	# C = tc.transform_predictions_test(B, 'voc2012')

	# tc.check_map('mapillary_vistas_comm')

	# tc.transform_d2utest('cityscapes_18')

	# test.
	# run_verification_tests()
