#!/usr/bin/python3

import numpy as np
from pathlib import Path
import pdb
import torch

from mseg.utils.names_utils import (
	load_class_names,
	get_universal_class_names
)
from mseg.utils.tsv_utils import read_tsv_column_vals

from mseg.taxonomy.taxonomy_converter import (
	parse_entry,
	parse_uentry,
	parse_test_entry,
	TaxonomyConverter,
	RELABELED_TRAIN_DATASETS,
	UNRELABELED_TRAIN_DATASETS
)

_ROOT = Path(__file__).resolve().parent.parent

def strip_group(entry):
	if entry.startswith('{'):
		# take [1:-1] to discard `{`  `}` chars
		classes = entry[1:-1].split(',')
		classes = [c.strip() for c in classes]
		return classes
	else:
		return [entry]


def entries_equal(dname, tsv_fpath):
	""" """
	tsv_classnames = read_tsv_column_vals(tsv_fpath, col_name=dname, convert_val_to_int=False)
	nonempty_classnames = [name for name in tsv_classnames if name != '']
	tsv_classnames = []
	for entry in nonempty_classnames:
		tsv_classnames.extend(strip_group(entry))
	txt_classnames = load_class_names(dname)
	if set(txt_classnames) != set(tsv_classnames):
		pdb.set_trace()
	return set(txt_classnames) == set(tsv_classnames)


def test_names_complete():
	"""
	Test on dataset_config and on TaxonomyConverter

	Make sure tsv entries in a single column match EXACTLY
	to _names.txt file.
	"""
	tsv_fpath = f'{_ROOT}/mseg/class_remapping_files/MSeg_master.tsv'

	train_dnames = UNRELABELED_TRAIN_DATASETS + RELABELED_TRAIN_DATASETS
	for dname in train_dnames:
		print(f'On {dname}...')
		assert entries_equal(dname, tsv_fpath)
		print(f'{dname} passed.')
		print()

	test_dnames = [
		'camvid-11',
		'kitti-19',
		#'pascal-context-60', # {'flower', 'wood'} missing
		'scannet-20',
		'voc2012',
		'wilddash-19'
	]
	for dname in test_dnames:
		print(f'On {dname}')
		assert entries_equal(dname, tsv_fpath)


def test_parse_entry_blank():
	""" """
	entry = ''
	classes = parse_entry(entry)
	assert classes == []

def test_parse_entry_brackets1():
	"""
	"""
	entry = '{house,building,  skyscraper,  booth,  hovel,  tower, grandstand}'
	classes = parse_entry(entry)
	gt_classes = [
		'house',
		'building',
		'skyscraper',
		'booth',
		'hovel',
		'tower',
		'grandstand'
	]
	assert classes == gt_classes


def test_parse_entry_space_sep():
	"""
	Note: ADE20K class "conveyer" is typo of "conveyor"
	"""
	entry = 'conveyer belt'
	classes = parse_entry(entry)
	assert classes == ['conveyer belt']


def test_parse_uentry():
	""" """
	uentry = 'animal_other'
	fullname = parse_uentry(uentry)
	assert fullname == 'animal_other'


def test_label_transform():
	"""
	Bring label from training taxonomy (mapillary-public65)
	to the universal taxonomy.
	
	21 is the motorcyclist class in mapillary-public65
	"""
	dname = 'mapillary-public65'
	txt_classnames = load_class_names(dname)
	train_idx = txt_classnames.index('Motorcyclist')
	tc = TaxonomyConverter()
	# training dataset label
	traind_label = torch.ones(4,4)*train_idx
	traind_label = traind_label.type(torch.LongTensor)

	# Get back the universal label
	u_label = tc.transform_label(traind_label, dname)
	u_idx = get_universal_class_names().index('motorcyclist')
	gt_u_label = np.ones((4,4)).astype(np.int64) * u_idx
	assert np.allclose(u_label.numpy(), gt_u_label)


if __name__ == '__main__':
	#test_names_complete()
	#test_parse_entry_blank()
	#test_parse_entry_brackets1()
	#test_parse_entry_space_sep()
	#test_parse_uentry()
	test_label_transform()

