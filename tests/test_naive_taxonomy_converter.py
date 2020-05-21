#!/usr/bin/python3

import numpy as np
import pdb
import torch

from mseg.utils.names_utils import (
	load_class_names,
	get_universal_class_names,
	get_classname_to_dataloaderid_map
)
from mseg.taxonomy.naive_taxonomy_converter import NaiveTaxonomyConverter


def test_constructor():
	""" """
	ntc = NaiveTaxonomyConverter()


def test_number_naive_classes():
	""" """
	ntc = NaiveTaxonomyConverter()

	# exclude the `unlabeled` class
	assert len(ntc.uname2uid) - 1 == 318
	assert len(ntc.uid2uname) - 1 == 318


def test_label_transform():
	"""
	Bring label from training taxonomy (mapillary-public65)
	to the universal taxonomy.
	
	21 is the motorcyclist class in mapillary-public65
	"""
	dname = 'mapillary-public65'
	txt_classnames = load_class_names(dname)
	train_idx = txt_classnames.index('Motorcyclist')
	ntc = NaiveTaxonomyConverter()
	# training dataset label
	traind_label = torch.ones(4,4)*train_idx
	traind_label = traind_label.type(torch.LongTensor)

	# Get back the universal label
	u_label = ntc.transform_label(traind_label, dname)
	u_idx = ntc.uname2uid['motorcyclist']
	assert ntc.uid2uname[u_idx] == 'motorcyclist'

	gt_u_label = np.ones((4,4)).astype(np.int64) * u_idx
	assert np.allclose(u_label.numpy(), gt_u_label)



def test_label_transform_unlabeled():
	"""
	Make sure 255 stays mapped to 255 at each level (to be ignored in cross-entropy loss).
	"""
	IGNORE_LABEL = 255
	dname = 'mapillary-public65'
	txt_classnames = load_class_names(dname)
	name2id = get_classname_to_dataloaderid_map(dname, include_ignore_idx_cls = True)
	train_idx = name2id['unlabeled']

	ntc = NaiveTaxonomyConverter()
	# training dataset label
	traind_label = torch.ones(4,4)*train_idx
	traind_label = traind_label.type(torch.LongTensor)

	# Get back the universal label
	u_label = ntc.transform_label(traind_label, dname)
	u_idx = IGNORE_LABEL
	gt_u_label = np.ones((4,4)).astype(np.int64) * u_idx
	assert np.allclose(u_label.numpy(), gt_u_label)


def test_transform_predictions_test():
	"""
		Consider predictions made within the universal taxonomy
		over a tiny 2x3 image. We use a linear mapping to bring
		these predictions into a test dataset's taxonomy 
		(summing the probabilities where necessary).

		For Camvid, universal probabilities for `person',`bicycle'
		should both go into the 'Bicyclist' class.

		320 universal classes here, instead of 194.
	"""
	ntc = NaiveTaxonomyConverter()
	dog_uidx = ntc.uname2uid['dog'] # not in camvid
	road_uidx = ntc.uname2uid['road']
	sky_uidx = ntc.uname2uid['sky']

	input = np.zeros((320,2,3))
	input[dog_uidx, :,0] = 1.0 # far left col is dog
	input[road_uidx,:,1] = 1.0 # middle col is road
	input[sky_uidx, :,2] = 1.0 # far right col is sky

	input = torch.from_numpy(input)
	input = input.unsqueeze(0).float() # CHW -> NCHW
	assert input.shape == (1,320,2,3)

	test_dname = 'camvid-11'
	camvid_classnames = load_class_names(test_dname)

	output = ntc.transform_predictions_test(input, test_dname)
	output = output.squeeze() # NCHW -> CHW

	weight_matrix = ntc.convs[test_dname].weight.squeeze().numpy()
	row, col = np.where(weight_matrix == 1)
	for i,j in zip(row,col):
		# Show mapping from universal -> camvid
		print(f'\t Univ. {ntc.uid2uname[j]} -> Camvid {camvid_classnames[i]}')
		assert ntc.uid2uname[j] == camvid_classnames[i].lower()

	prediction = torch.argmax(output, dim=0).numpy()

	# Camvid should have predictions across 11 classes.
	prediction_gt = np.zeros((2,3))
	prediction_gt[:,1] = camvid_classnames.index('Road')
	prediction_gt[:,2] = camvid_classnames.index('Sky')

	assert np.allclose(prediction[:,1:], prediction_gt[:,1:])


def test_constructor_types():
	""" """
	ntc = NaiveTaxonomyConverter()
	for dname, conv in ntc.convs.items():
		assert isinstance(conv, torch.nn.Module)


def test_label_mapping_arrs():
	""" """
	ntc = NaiveTaxonomyConverter()
	train_idx = load_class_names('ade20k-150').index('road')
	u_idx = ntc.uname2uid['road']
	assert ntc.label_mapping_arr_dict['ade20k-150'][train_idx] == u_idx

	train_idx = load_class_names('mapillary-public65').index('Bird')
	u_idx = ntc.uname2uid['bird']
	assert ntc.label_mapping_arr_dict['mapillary-public65'][train_idx] == u_idx




if __name__ == '__main__':
	test_constructor()
	test_number_naive_classes()
	test_label_transform()
	test_label_transform_unlabeled()
	test_transform_predictions_test()
	test_constructor_types()
	test_label_mapping_arrs()




