#!/usr/bin/python3

from pathlib import Path
import pdb

from mseg.utils.names_utils import load_class_names
from mseg.utils.tsv_utils import read_tsv_column_vals

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

	train_dnames = [
		'ade20k-150',
		'bdd',
		'cityscapes-19',
		'cityscapes-34',
		'coco-panoptic-133',
		'idd-39',
		'mapillary-public65',
		'sunrgbd-37',

		'ade20k-150-relabeled',
		'bdd-relabeled',
		'cityscapes-19-relabeled',
		'cityscapes-34-relabeled',
		'coco-panoptic-133-relabeled',
		'idd-39-relabeled',
		'mapillary-public65-relabeled',
		'sunrgbd-37-relabeled'
	]
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


if __name__ == '__main__':
	test_names_complete()