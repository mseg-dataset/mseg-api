#!/usr/bin/python3

from pathlib import Path


def read_txt_file(txt_fpath):
	"""
		Args:
		-	

		Returns:
		-	
	"""
	with open(txt_fpath, 'r') as f:
		txt_lines = f.readlines()

	txt_lines = [line.strip() for line in txt_lines]
	return txt_lines


class LabelImageUpdateRecord:
	"""
	Stores necessary information to remap a single segment of a single image.
	"""
	def __init__(self, dataset_name, img_fname, segmentid, split, orig_class, relabeled_class):
		""" """
		self.dataset_name = dataset_name
		self.img_fname = img_fname
		self.segmentid = segmentid
		self.split = split
		self.orig_class = orig_class
		self.relabeled_class = relabeled_class


class DatasetClassUpdateRecord:
	"""
	Stores necessary information to remap a list of (segment, image) objects from
	an old class to a new class.
	"""
	def __init__(self, dataset_name, split, orig_class, relabeled_class, txt_fpath):
		"""
			Args:
			-	dataset_name
			-	split
			-	orig_class
			-	relabeled_class
			-	txt_fpath

			Returns:
			-	None
		"""
		self.dataset_name = dataset_name
		self.split = split
		self.orig_class = orig_class
		self.relabeled_class = relabeled_class
		self.txt_fpath = txt_fpath
		full_txt_fpath = f'mturk/verified_reclassification_files/{self.txt_fpath}'
		if not Path(full_txt_fpath).exists():
			print(f'Not found: {full_txt_fpath}. Quitting...')
			quit()
		self.img_list = read_txt_file(full_txt_fpath)
		num_imgs = len(self.img_list)
		print(f'\tFound {num_imgs} segments to update in {dataset_name}-{split} from {orig_class}->{relabeled_class}')

