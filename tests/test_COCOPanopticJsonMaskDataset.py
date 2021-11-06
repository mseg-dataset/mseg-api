#!/usr/bin/python3

from pathlib import Path

from mseg.utils.json_utils import read_json_file
from mseg.utils.names_utils import get_dataloader_id_to_classname_map
from mseg.utils.test_utils import dict_is_equal


_ROOT = Path(__file__).resolve().parent


def test_categoryid_to_classname_map() -> None:
    """
    Make sure that mapping on disk is correct.
    """
    taxonomy_data_fpath = _ROOT / "test_data/coco_panoptic_categories.json"
    coco_category_info = read_json_file(taxonomy_data_fpath)

    gt_categoryid_to_classname_map = {}
    for category in coco_category_info:
        categoryid = category["id"]
        classname = category["name"]
        gt_categoryid_to_classname_map[categoryid] = classname

    # pad the rest of keys to map to 'unlabeled'
    for id in range(201):
        if id not in gt_categoryid_to_classname_map:
            gt_categoryid_to_classname_map[id] = "unlabeled"

    categoryid_to_classname_map = get_dataloader_id_to_classname_map(
        dataset_name="coco-panoptic-201", include_ignore_idx_cls=False
    )
    dict_is_equal(categoryid_to_classname_map, gt_categoryid_to_classname_map)

    for key in categoryid_to_classname_map.keys():
        assert isinstance(key, int)
    for value in categoryid_to_classname_map.values():
        assert isinstance(value, str)


"""
TODO: Need a unit test on get_segment
"""


if __name__ == "__main__":
    test_categoryid_to_classname_map()
