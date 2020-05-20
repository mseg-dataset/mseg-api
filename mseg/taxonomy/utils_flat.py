#!/usr/bin/python3

import time
from collections import defaultdict, namedtuple
import cv2
import numpy as np
import os
import pandas as pd
import pdb
import torch
import torch.nn as nn
from mseg.utils.names_utils import load_class_names
from mseg.utils.dataset_config import v2gids


os.environ['PYTHONPATH'] = './'
from typing import List, Mapping, Tuple

TAXONOMY_SPREADSHEET_URL_template = "'https://docs.google.com/spreadsheets/d/1r9JE0UcLkFh0Ww6NIFdWjYd6uJd4MzXjREOhO6tKnMA/export?gid={}&format=tsv'"
SCANNET_SPREADSHEET_URL = "'https://docs.google.com/spreadsheets/d/1NSMd2gCmTrGW9hsMOUh6_sKLAQFzuFEwe5F3vdNR4cQ/export?gid=0&format=tsv'"
DOWNLOAD = False


def parse_entry(entry: str, verbose: bool = False) -> Tuple[List[str], str]:
	''' Parse an entry from dataset element, return list of classes.

		TODO: handle case for last column in spreadsheet, when contains carriage return

		Args:
		-	entry: string, representing cell in taxonomy spreadsheet

		Returns:
		-	classes: list of strings, representing class names in a non-universal dataset taxonomy
		-	type: string, representing the type of node, which can be `none`, `dump`, or `same`
	'''
	assert entry.strip() == entry, print(entry + 'is not stripped') # don't want entry to contain or be space/tab
	if entry == '':
		classes = []
		type = 'none'
	elif entry.startswith('{'):
		# take [1:-1] to discard `{`  `}` chars
		classes = entry[1:-1].split(',')
		classes = [c.strip() for c in classes]
		type = 'dump'
	else:
		classes = [entry]
		type = 'same'
	
	if verbose:
		print(entry, classes)
	return classes, type


def parse_uentry(uentry: str) -> Tuple[int, List[str], str]:
	"""
		Args:
		-	uentry: string, representing TSV entry from `Universal` taxonomy column

		Returns:
		-	level: integer, representing depth of node in tree
		-	names: List of strings, representing names of nodes on path from specified node to root
		-	full_name: exact duplicate of uentry string input
	"""

	full_name = uentry.strip()
	assert(full_name != '')
	return full_name


def parse_test_entry(test_entry) -> Tuple[str,str]:
	"""
		Args:
		-	test_entry

		Returns:
		-	
	"""
	assert(test_entry.strip() == test_entry) # don't want entry to contain or be space/tab
	if test_entry == '':
		cls = ''
		type = 'none'
	else:
		cls = test_entry
		type = 'sum'
	return cls, type


def extend_name(name: str) -> str:
	"""
		Args:
		-	name

		Returns:
		-	string representing `{...}->{node_name}->{node_name}_ext`, e.g.
				'accessory->backpack->backpack_ext'
	"""
	level, names, full_name = parse_uentry(name)
	return '->'.join(names[1:level+1]).strip() + '->' + names[level] + '_ext'




class TaxonomyConverter():
	"""
	Class that can map training data to a universal taxonomy, and universal taxonomy
	predictions at inference time to specific data taxonomies.
	"""
	def __init__(self, 
				is_unit_test: bool = False, version=1.0, finetune=None,
				train_datasets: List[str] = [
				'ade20k-150',
				'ade20k-150-relabeled',
				'bdd',
				'bdd-relabeled',
				'cityscapes-19',
				'cityscapes-19-relabeled',
				'coco-panoptic-133',
				'coco-panoptic-133-relabeled',
				'idd-39',
				'idd-39-relabeled',
				'mapillary-public65', 
				'mapillary-public65-relabeled',
				'sunrgbd-37',
				'sunrgbd-37-relabeled',
				],
				test_datasets: List[str] = [
				'camvid-11',
				'kitti-19',
				'pascal-context-60',
				'scannet-20',
				'voc2012',
				'wilddash-19',
				],
				finetune_dataset='wilddash'):
	

		print('------initializing tc----------')
		self.is_unit_test = is_unit_test
		self.ignore_label = 255
		self.version = version
		self.gid = v2gids[self.version]
		self.finetune = finetune
		print(f'Using Version {self.version}')

		self.url = TAXONOMY_SPREADSHEET_URL_template.format(self.gid)

		if is_unit_test:
			tsv_fpath = 'test_data/unit_test_taxonomy.tsv'
		else:
			tsv_fpath = 'taxonomy_{}.tsv'.format(self.gid)
			# if not os.path.isfile(tsv_fpath):
			if DOWNLOAD:
				os.system(f"""wget --no-check-certificate -O {tsv_fpath} {self.url}""")

		self.tsv_data = pd.read_csv(tsv_fpath, sep='\t', keep_default_na=False)

		# Find universal name from spreadsheet TSV index.
		self.uid2uname = {}
		# for C, finding spreadsheet TSV index from universal name.
		self.uname2uid = {}

		self.build_universal_tax()
		self.classes = len(self.uid2uname) - 1 # excluding ignored label（id, 255)）

		assert (self.classes < 255)

		# self.verification()

		self.train_datasets = train_datasets
		self.test_datasets = test_datasets
		self.finetune_dataset = finetune_dataset

		# for i in range(len(self.train_datasets)):
		# 	x = self.train_datasets[i]
		# 	self.train_datasets.append(x) #+"-qvga")
		# 	#self.tsv_data[x+"-qvga"] = self.tsv_data[x]

		# for i in range(len(self.test_datasets)):
		# 	x = self.test_datasets[i]
		# 	self.test_datasets.append(x) #+"-qvga")
		# 	# self.tsv_data[x+"-qvga"] = self.tsv_data[x]

		# dataset to universal id maps
		self.dataset_names = {}
		self.dataloaderid_to_uid_maps = {}
		self.convs = {}
		self.softmax = nn.Softmax(dim=1)


		# self.name_to_tax_map = {x: x for x in (self.train_datasets + self.test_datasets)}

		# for d in self.train_datasets + self.test_datasets:
			# self.dataset_names[d] = load_class_names(d)

		for d in self.train_datasets:
			tax_name = self.find_tax_name(d, split = 'train')  #
			self.dataset_names[d] = load_class_names(tax_name)
			# self.check_complete(d, split='train')

		for d in self.train_datasets:
			tax_name = self.find_tax_name(d, split = 'train')
			print(f'Mapping {d} -> universal, using taxonomy -> {tax_name}')
			dname2uid_map, dataloaderid2uid_map = self.transform_d2u(tax_name)
			self.dataloaderid_to_uid_maps[d] = dataloaderid2uid_map


		for d in self.test_datasets:
			tax_name = self.find_tax_name(d, split = 'test')  #
			self.dataset_names[d] = load_class_names(tax_name)
			# self.check_complete(d, split='test')

		for d in self.test_datasets:
			tax_name = self.find_tax_name(d, split = 'test')  #
			print(f'Creating 1x1 conv for test: universal -> {d}, using taxonomy -> {tax_name}')
			self.convs[d] = self.get_convolution_test(tax_name)


		assert(len(self.uname2uid) < 255)


		# if self.finetune:
		# 	self.dataloaderid_to_fid_maps = {}
		# 	for d in self.train_datasets:
		# 		tax_name = self.find_tax_name(d, split = 'train')  #
		# 		dname2fid_map, dataloaderid2fid_map = self.transform_d2f(tax_name)
		# 		self.dataloaderid_to_fid_maps[d] = dataloaderid2fid_map
		# 	self.finetune_classes = len(self.dataset_names[self.finetune_dataset])
			# self.label_mapping_arr_dict = self.form_label_mapping_arrs(finetune=True)
		# print(self.dataloaderid_to_fid_maps)

		self.label_mapping_arr_dict = self.form_label_mapping_arrs() # whether finetune or not is handled inside function

		# for x in self.train_datasets: # not checkint test yet, test part needs rewriting

		# for x in self.test_datasets:
		# 	self.check_complete(x)	



	def find_tax_name(self, dataset_name, split):  # return the name in the sheet, to avoid too many columns of same thing in sheet

		# special case handling
		if dataset_name.endswith('-sr'):
			dataset_name = dataset_name.replace('-sr', '')

		return dataset_name


		# if split == 'train':
		# 	if ('cityscapes' == dataset_name) or ('wilddash' == dataset_name) or ('bdd' == dataset_name):
		# 		tax_name = 'c-w-b-train'
		# 	else:
		# 		tax_name = dataset_name

		# if split == 'test':
		# 	if ('cityscapes' in dataset_name) or ('wilddash' in dataset_name) or ('bdd' in dataset_name):
		# 		tax_name = 'c-w-b-test'
		# 	else:
		# 		tax_name = dataset_name



	def build_universal_tax(self, verbose: bool = False) -> None:
		''' 
			
			Build a flat tree at C[1] by adding each `universal` taxonomy entry from the TSV here.
			Data structures to modify are passed in by reference.

			Args:
			-	verbose: boolean, whether to print info during execution

			Returns:
			-	None
		'''
		for id, row in self.tsv_data.iterrows():
			u_name = parse_uentry(row['universal'])

			assert(u_name not in self.uname2uid.keys()) # no duplicate names

			if u_name == 'unlabeled':
				id = self.ignore_label

			self.uid2uname[id] = u_name
			self.uname2uid[u_name] = id

		if self.is_unit_test:
			self.C[1].append(Node(level=1, id=len(self.C[1]), name='unlabeled', parent_id=None, parent_name=None))


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

		if dataset == 'scannet_v2_549':
			return self.transform_d2u_scannet(dataset)

		dname2uname_map = {} # name to name map from (dataset -> universal name)
		dname2uid_map = {}# name to id map from (dataset -> universal ID)
		dataloaderid2uname_map = {}
		dataloaderid2uid_map = {}
	 
		for index, row in self.tsv_data.iterrows():
			classes, type = parse_entry(row[dataset])
			u_name = parse_uentry(row['universal'])
			uid = self.uname2uid[u_name]
			if type == 'none':
				continue
			# self.form_d2u_mapping(name_map, id_map, classes, row, full_uname, level)
			for cls in classes:
				dname2uname_map[cls] = u_name
				dname2uid_map[cls] = uid

		# convert name2id map to id2id map, also mark "unlabeled"

		# Map the PNG value (dataloader ID) to an ID in the name2id data structure
		# all_dataset_names = self.dataset_names[dataset]
		all_dataset_names = load_class_names(dataset)
		for i, dname in enumerate(all_dataset_names):
			if self.is_unit_test and name not in name_map:
				continue

			uid = dname2uid_map[dname]
			u_name = dname2uname_map[dname]
			dataloaderid2uid_map[i] = uid
			dataloaderid2uname_map[i] = u_name

		dataloaderid2uid_map[self.ignore_label] = self.ignore_label
		dataloaderid2uname_map[self.ignore_label] = 'unlabeled'
		dname2uname_map['unlabeled'] = 'unlabeled'
		dname2uid_map['unlabeled'] = self.ignore_label

		# print()
		# if dataset == 'ade20k-v1':
			# print(dname2uname_map)

		return dname2uid_map, dataloaderid2uid_map


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
		_, dataloaderid2pair = self.transform_d2utest(dataset)
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
		all_names = load_class_names(dataset)
		dname2pair = defaultdict(list) # test -> universal
		dataloaderid2pair = defaultdict(list) # test -> universal a list of tuples of (level, id) that needs to be summed up to get the final probability

		for index, row in self.tsv_data.iterrows():
			dname, type = parse_test_entry(row[dataset])
			if type == 'none':
				continue
			elif type == 'sum':
				id = all_names.index(dname)

				u_name = parse_uentry(row['universal'])
				u_id = self.uname2uid[u_name]
				# if type == 'same': # the only case in pascal voc
				# id = self.name2id[name]
				dname2pair[dname].append((u_id, u_name))
				dataloaderid2pair[id].append((u_id, u_name))

		return dname2pair, dataloaderid2pair


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

		print(reg_names)

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

		# if not complete:
		assert complete, 'please check taxonomy'
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
	DOWNLOAD = True
	# DOWNLOAD = True

	label_test = torch.LongTensor([1,2,3,4,255])

	# version = 1.6
	version = 4.0
	tc = TaxonomyConverter(version=version, finetune=False, finetune_dataset='wilddash')

	# tc = TaxonomyConverter(version=version, train_datasets=['ade20k-v1'], finetune=False, finetune_dataset='wilddash')


	print(tc.classes)
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