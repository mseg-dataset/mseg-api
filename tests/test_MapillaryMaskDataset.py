#!/usr/bin/python3

import numpy as np
from pathlib import Path

import mseg.utils.json_utils as json_utils
import mseg.utils.names_utils as names_utils

from mseg.utils.test_utils import dict_is_equal
from mseg.dataset_apis.MapillaryMaskDataset import MapillaryMaskDataset


_TEST_DIR = Path(__file__).resolve().parent

# One set of data is from Scene Parsing Challenge, other is not.
CONFIG_JSON_FPATH = (
    _TEST_DIR / "test_data/mapillary-vistas-dataset_public_v1.1_config.json"
)


def read_mapillary_config_helper():
    """ """
    return json_utils.read_json_file(CONFIG_JSON_FPATH)


def test_load_names() -> None:
    """ """
    tax_data = read_mapillary_config_helper()
    id_to_classname_map = names_utils.get_dataloader_id_to_classname_map(
        dataset_name="mapillary-public66", include_ignore_idx_cls=False
    )
    gt_id_to_classname = {
        i: entry["readable"] for i, entry in enumerate(tax_data["labels"])
    }
    dict_is_equal(gt_id_to_classname, id_to_classname_map)
    assert len(id_to_classname_map.keys()) == 66


def test_load_colors() -> None:
    """ """
    tax_data = read_mapillary_config_helper()
    num_classes = 66
    gt_dataset_ordered_colors = np.zeros((66, 3), dtype=np.uint8)
    for i in range(num_classes):
        gt_dataset_ordered_colors[i] = np.array(tax_data["labels"][i]["color"])

    colors = names_utils.load_dataset_colors_arr("mapillary-public66")
    assert np.allclose(colors, gt_dataset_ordered_colors)


"""
def test_get_segment_mask():
	dataroot = f'{_TEST_DIR}/test_data/Mapillary_test_data'
	m_api = MapillaryMaskDataset(dataroot)
	
	seq_id = '' # dummy value
	segmentid = 7936
	fname_stem = 'aDailxp-VC9IbQWfIp-8Rw'
	split = 'train'
	mask = m_api.get_segment_mask(seq_id, segmentid, fname_stem, split)
	
	# stream running through the bottom of an image, hill above it
	gt_mask = np.array(
		[
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0],
			[1, 1, 1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 0, 0, 0]
		], dtype=np.uint8)
	assert np.allclose(mask[::500,::500], gt_mask)
"""

if __name__ == "__main__":
    test_get_segment_mask()
