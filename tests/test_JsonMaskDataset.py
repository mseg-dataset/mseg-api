#!/usr/bin/python3

from pathlib import Path
import pdb
from mseg.dataset_apis.JsonMaskLevelDataset import JsonMaskDataset

_TEST_DIR = Path(__file__).resolve().parent

"""
Requires placing data locally into the test folder.
"""

"""
def test_constructor_idd():
	dataroot = f'{_TEST_DIR}/test_data/IDD_test_data/IDD_Segmentation'
	iddmd = JsonMaskDataset(dataroot)


def test_constructor_cityscapes():
	dataroot = f'{_TEST_DIR}/test_data/Cityscapes_test_data/cityscapes'
	csmd = JsonMaskDataset(dataroot)



def test_get_segment_mask_idd():
	# Form an IDD Mask Dataset (iddmd)
	dataroot = f'{_TEST_DIR}/test_data/IDD_test_data/IDD_Segmentation'
	iddmd = JsonMaskDataset(dataroot)

	# examine a 'car' mask
	query_segmentid = 39
	seq_id = 0
	fname_stem = '024793_leftImg8bit'
	split = 'train'
	segment_mask = iddmd.get_segment_mask(seq_id, query_segmentid, fname_stem, split)
	assert segment_mask.mean() - 0.138 < 1e-3
	assert segment_mask.sum() == 170329
	assert segment_mask.size == 1233920



def test_get_segment_mask_cityscapes():
	# Form a Cityscapes Mask Dataset (csmd)
	dataroot = f'{_TEST_DIR}/test_data/Cityscapes_test_data/cityscapes'
	csmd = JsonMaskDataset(dataroot)


	# examine a '' mask
	query_segmentid = 7 # large building in upper-left quadrant
	seq_id = 'aachen'
	fname_stem = 'aachen_000000_000019_leftImg8bit'
	split = 'train'

	segment_mask = csmd.get_segment_mask(seq_id, query_segmentid, fname_stem, split)
	assert segment_mask.sum() == 766842
	assert segment_mask.mean() - 0.366 < 1e-3
	assert segment_mask.size == 2097152
"""


if __name__ == "__main__":

    test_constructor_idd()
    test_constructor_cityscapes()
    # test_get_segment_mask_idd()
    # test_get_segment_mask_cityscapes()
