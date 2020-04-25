#!/usr/bin/python3

from collections import defaultdict
import copy
import matplotlib.pyplot as plt
import numpy as np
import pdb
import scipy.ndimage

from typing import List, Mapping



def scipy_conn_comp(img: np.ndarray) -> Mapping[int,List[np.ndarray]]:
	"""
		labelsndarray of dtype int
		Labeled array, where all connected regions
		are assigned the same integer value.

		numint, optional
		Number of labels, which equals the maximum label index 
		and is only returned if return_num is True.

		Args:
		-	img

		Returns:
		-	class_to_conncomps_dict
	"""
	class_to_conncomps_dict = defaultdict(list)
	present_class_idxs = np.unique(img)

	for class_idx in present_class_idxs:
		structure = np.ones((3, 3), dtype=np.int)  # this defines the connection filter
		instance_img, nr_objects = scipy.ndimage.label(img == class_idx, structure)
		
		for i in np.unique(instance_img):
			if i == 0: # 0 doesn't count as an instance ID
				continue
			bin_arr = (instance_img == i).astype(np.uint8)
			class_to_conncomps_dict[class_idx] += [bin_arr]

	return class_to_conncomps_dict



def test_scipy_conn_comp():
	"""
	"""
	img = np.array(
		[
			[1,1,2,3],
			[1,4,5,3],
			[0,0,1,1]
		])
	class_to_conncomps_dict = scipy_conn_comp(img)
	print(class_to_conncomps_dict)

	gt_class_to_conncomps_dict = {
		0: [
			np.array([	[0, 0, 0, 0],
					[0, 0, 0, 0],
					[1, 1, 0, 0]], dtype=np.uint8)
		], 
		1: [
			np.array([[1, 1, 0, 0],
					[1, 0, 0, 0],
					[0, 0, 0, 0]], dtype=np.uint8), 
			np.array([[0, 0, 0, 0],
					[0, 0, 0, 0],
					[0, 0, 1, 1]], dtype=np.uint8)
		], 
		2: [
			np.array([[0, 0, 1, 0],
					[0, 0, 0, 0],
					[0, 0, 0, 0]], dtype=np.uint8)
		], 
		3: [
			np.array([[0, 0, 0, 1],
					[0, 0, 0, 1],
					[0, 0, 0, 0]], dtype=np.uint8)
		], 
		4: [
			np.array([[0, 0, 0, 0],
					[0, 1, 0, 0],
					[0, 0, 0, 0]], dtype=np.uint8)
		], 
		5: [
			np.array([	[0, 0, 0, 0],
					[0, 0, 1, 0],
					[0, 0, 0, 0]], dtype=np.uint8)
		]
	}

	for class_idx, conncomps_list in class_to_conncomps_dict.items():
		gt_conncomps_list = gt_class_to_conncomps_dict[class_idx]
		for conncomp, gtconncomp in zip(conncomps_list, gt_conncomps_list):
			assert np.allclose(conncomp, gtconncomp)






if __name__ == '__main__':
	#main()
	#test_scipy_conn_comp()
	#test_find_max_cardinality_mask()
	test_save_classnames_in_image()


