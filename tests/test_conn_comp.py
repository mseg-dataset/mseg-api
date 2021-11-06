#!/usr/bin/python3

import numpy as np

from mseg.utils.conn_comp import scipy_conn_comp


def test_scipy_conn_comp():
	""" Make sure we can recover a dictionary of binary masks for each conn. component"""
	
	# toy semantic label map / label image
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
			np.array([[0, 0, 0, 0],
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




