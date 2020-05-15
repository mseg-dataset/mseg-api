
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pdb

from mseg.utils.mask_utils_detectron2 import Visualizer


def test_visualizer1():
	"""
	label map with four quadrants
	"""
	H = 640
	W = 480
	img_rgb = np.ones((H,W,3), dtype=np.uint8)
	label_map = np.zeros((H,W), dtype=np.uint8)
	label_map[:H//2, :W//2] = 0
	label_map[:H//2,  W//2:] = 1
	label_map[ H//2:,:W//2] = 2
	label_map[ H//2:, W//2:] = 3

	id_to_class_name_map = { 0: 'sky', 1: 'road', 2: 'person', 3: 'horse'}

	pdb.set_trace()
	vis_obj = Visualizer(img_rgb, None)
	output_img = vis_obj.overlay_instances(
		label_map,
		id_to_class_name_map
	)
	plt.imshow(output_img)
	plt.show()


def test_visualizer2():
	"""
	label map with four quadrants
	"""
	H = 640
	W = 480
	img_rgb = np.ones((H,W,3), dtype=np.uint8)
	label_map = np.zeros((H,W), dtype=np.uint8)
	label_map[100,300]=1
	label_map[100,100]=1
	# only 2 pixels will have value 1
	mask_diff = np.ones_like(label_map).astype(np.uint8) - label_map

	# Calculates the distance to the closest zero pixel for each pixel of the source image.
	distance_mask = cv2.distanceTransform(mask_diff, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
	distance_mask = distance_mask.astype(np.float32)
	label_map = (distance_mask <= 25).astype(np.uint8)

	id_to_class_name_map = { 0: 'road', 1: 'person' }

	plt.imshow(label_map)
	plt.show()

	pdb.set_trace()
	vis_obj = Visualizer(img_rgb, None)
	output_img = vis_obj.overlay_instances(
		label_map,
		id_to_class_name_map
	)
	plt.imshow(output_img)
	plt.show()




if __name__ == '__main__':
	#test_visualizer1()
	test_visualizer2()

