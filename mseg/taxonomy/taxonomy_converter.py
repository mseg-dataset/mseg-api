#!/usr/bin/python3

from collections import defaultdict, namedtuple
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pdb
import torch
import torch.nn as nn
from mseg.utils.names_utils import load_class_names
from typing import List, Mapping, Tuple

"""
We train in a common, unified label space.

Inference is also performed in that common, unified label space,
and predictions are transformed using a linear mapping to any desired
evaluation taxonomy.

TODO: use underscore to show private methods.
"""

_ROOT = Path(__file__).resolve().parent

class TaxonomyConverter:
	"""
	We use 1x1 convolution for our linear mapping from universal->test_taxonomy.
	"""
	def __init__(
		self,
		train_datasets: List[str],
		test_datasets: List[str],
		tsv_fpath: str = f'{_ROOT}/class_remapping_files/MSeg_master.tsv'):
		"""
		"""
		self.train_datasets = train_datasets
		self.test_datasets = test_datasets

		self.ignore_label = 255
		self.tsv_data = pd.read_csv(tsv_fpath, sep='\t', keep_default_na=False)
		self.softmax = nn.Softmax(dim=1)

		# Find universal name from spreadsheet TSV index.
		self.uid2uname = {}
		# Inverse -- find spreadsheet TSV index from universal name.
		self.uname2uid = {}
		self.build_universal_tax()
		self.num_uclasses = len(self.uid2uname) - 1 # excluding ignored label（id, 255)）
		assert self.num_uclasses == 194

		self.dataset_classnames = {d: load_class_names(d) for d in (self.train_datasets + self.test_datasets)}

		self.id_to_uid_maps = {}

		# Map train_dataset_id -> universal_id
		for d in self.train_datasets:
			print(f'Mapping {d} -> universal')
			self.id_to_uid_maps[d] = self.transform_d2u(d)

		self.label_mapping_arr_dict = self.form_label_mapping_arrs()
		print(f'Creating 1x1 conv for test datasets')
		self.convs = {d: self.get_convolution_test(tax_name) for d in self.test_datasets}


	def build_universal_tax(self) -> None:
		''' 
			Build a flat label space by adding each `universal` taxonomy entry from the TSV here.
			We make a mapping from universal_name->universal_id, and vice versa.

			Args:
			-	None

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


	def transform_d2u(self, dataset: str) -> Mapping[int,int]:
		''' Transform a training dataset to the universal taxonomy.

			For one training dataset, map each of its class ids to one universal ids.

			Args:
			-	dataset: string representing name of dataset

			Returns:
			-	id2uid_map: provide mapping from training dataset id to universal id.
		'''
		id2uid_map = {}
	 
		for index, row in self.tsv_data.iterrows():
			classes = parse_entry(row[dataset])
			u_name = parse_uentry(row['universal'])
			uid = self.uname2uid[u_name]
			if len(classes) == 0:
				continue
			for cls in classes:
				id = self.dataset_classnames[dataset].index(cls)
				id2uid_map[id] = uid

		# unlabeled should remain unlabeled.
		id2uid_map[self.ignore_label] = self.ignore_label
		return id2uid_map


	def form_label_mapping_arrs(self)->Mapping[str, np.array]:
		""" Cache conversion maps so we will later be able to perform fast 
			grayscale->grayscale mapping for training data transformation.
			Each map is implemented as an integer array (given integer index, 
			give back integer value stored at that location).

			Args:
			-	None

			Returns:
			-	label_mapping_arr_dict: map dataset names to numpy arrays.
		"""
		label_mapping_arr_dict = {}
		from mseg.utils.mask_utils import form_label_mapping_array_pytorch

		for dataset in self.train_datasets:
			label_mapping_arr_dict[dataset] = form_label_mapping_array_pytorch(self.id_to_uid_maps[dataset])

		return label_mapping_arr_dict


	def transform_u2d(self, dataset):
		''' Explicitly for inference on our test datasets. Store correspondences
			between universal_taxonomy and test_dataset_taxonomy.

			Args:
			-	dataset: string representing dataset's name

			Returns:
			-	uid2testid: list of tuples (i,j) that form linear mapping P from
					universal -> test_dataset.
		'''
		uid2testid = []

		for index, row in self.tsv_data.iterrows():
			dname = parse_test_entry(row[dataset])
			if dname == '':
				# blank, so ignore
				continue
			test_id = self.dataset_classnames[dataset].index(dname)
			u_name = parse_uentry(row['universal'])
			u_id = self.uname2uid[u_name]
			uid2testid += [(u_id, test_id)]

		return uid2testid


	def get_convolution_test(self, dataset: str):
		""" Explicitly for inference on our test datasets.

			We implement this remapping from mu classes to mt classes 
			for the evaluation dataset as a linear mapping P.

			The matrix weights Pij are binary 0/1 values and are fixed
			before training or evaluation; the weights are determined
			manually by inspecting label maps of the test datasets. Pij
			is set to 1 if unified taxonomy class j contributes to
			evaluation dataset class i, otherwise Pij = 0.

			Args:
			-	dataset: string representing dataset's name

			Returns:
			-	conv: nn.Conv2d module representing linear mapping from universal_taxonomy
				to test_dataset_taxonomy.
		"""
		assert dataset in self.test_datasets
		uid2testid = self.transform_u2d(dataset)
		in_channel = self.num_uclasses
		out_channel = len(self.dataset_classnames(dataset))

		conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
		conv.weight.data.fill_(0)

		for param in conv.parameters():
			param.requires_grad = False

		for u_id, test_id in uid2testid:
			conv.weight[test_id][u_id] = 1

		return conv


	def transform_label(self, label: torch.Tensor, dataset: str) -> Mapping[int, torch.Tensor]:
		""" Function to be called externally for training.
			Perform fast grayscale->grayscale mapping for training data transformation.

			Args:
			-	label: Pytorch tensor on the cpu with dtype Torch.LongTensor, 
					representing a semantic image, according to PNG/dataloader format.
			-	dataset

			Returns:
			-	labels: vector label for each depth in the tree, compatible with softmax
		"""
		from mseg.utils.mask_utils import map_semantic_img_fast_pytorch
		label = map_semantic_img_fast_pytorch(label, self.label_mapping_arr_dict[dataset])
		return label


	def transform_predictions_test(self, input, dataset): # function to be called outside
		""" Explicitly for inference on our test datasets.

			Args:
			-	input: after softmax probability, of universal
			-	dataset:

			Returns:
			-	output: 
		"""
		input = self.softmax(input)
		output = self.convs[dataset](input)

		# if 'background' in self.dataset_names[dataset]: # background is 1 minus all other probabilitys, not summing all other nodes
		# 	background_id = self.dataset_names[dataset].index('background')
		# 	all_other = torch.cat([output[:, :background_id, :, :], output[:, (background_id+1):, :, :]], dim=1)
		# 	output[:, background_id, :, :] = 1 - torch.sum(all_other, dim=1)

		return output


	def transform_predictions_universal(self, input, dataset): # function to be called outside
		""" Explicitly for inference on our test datasets.

			Args:
			-	input: after softmax probability, of universal
			-	dataset:

			Returns:
			-	output: 
		"""
		output = self.softmax(input)
		return output


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


def parse_entry(entry: str) -> Tuple[List[str], str]:
	''' Parse a spreadsheet entry from dataset's column, return list of classes in this spreadsheet cell.
		TODO: handle case for last column in spreadsheet, when contains carriage return

		Args:
		-	entry: string, representing cell in taxonomy spreadsheet

		Returns:
		-	classes: list of strings, representing class names in a non-universal dataset taxonomy
	'''
	assert entry.strip() == entry, print(entry + 'is not stripped') # don't want entry to contain or be space/tab
	if entry == '':
		classes = []

	elif entry.startswith('{'):
		# take [1:-1] to discard `{`  `}` chars
		classes = entry[1:-1].split(',')
		classes = [c.strip() for c in classes]
	else:
		classes = [entry]
	
	return classes

