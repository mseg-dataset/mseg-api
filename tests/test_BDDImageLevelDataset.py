
import pdb
from mseg.dataset_apis.BDDImageLevelDataset import BDDImageLevelDataset


def test_constructor() -> None:
    """ """
    dataroot = "test_data/BDD100K_test_data/bdd100k"
    bddild = BDDImageLevelDataset(dataroot)


"""
def test_get_img_pair():
	dataroot = 'test_data/BDD100K_test_data/bdd100k'
	bddild = BDDImageLevelDataset(dataroot)
	
	split = 'train'
	fname_stem = '0a0a0b1a-7c39d841'
	rgb_img, label_img = bddild.get_img_pair(fname_stem, split)
	assert rgb_img.mean() - 122.029 < 1e-3
	assert label_img.mean() - 45.267 < 1e-3

	split = 'val'
	fname_stem = '7d2f7975-e0c1c5a7'
	rgb_img, label_img = bddild.get_img_pair(fname_stem, split)
	assert rgb_img.mean() - 115.705 < 1e-3
	assert label_img.mean() - 97.227 < 1e-3

	
def test_get_segment_mask():
	dataroot = 'test_data/BDD100K_test_data/bdd100k'
	bddild = BDDImageLevelDataset(dataroot)
	seq_id = ''
	query_segmentid = 13
	fname_stem = '0a0a0b1a-7c39d841'
	split = 'train'
	
	class_mask = bddild.get_segment_mask(seq_id, query_segmentid, fname_stem, split)

	assert class_mask.sum() == 16225
	assert class_mask.mean() - 0.018 < 1e-3
	assert class_mask.size == 921600
"""


if __name__ == "__main__":
    pass
    # test_constructor()
    # test_get_img_pair()
    # test_get_segment_mask()
