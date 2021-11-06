#!/usr/bin/python3

import pdb
from mseg.label_preparation.mseg_write_relabeled_segments import (
    get_unique_mask_identifiers,
    form_fname_to_updatelist_dict,
)
from mseg.label_preparation.relabeled_data_containers import DatasetClassUpdateRecord


def test_get_unique_mask_identifiers_ade20k() -> None:
    """ """
    # relabeled as glass -> wine_glass
    dname = "ade20k-150"
    annot_fname = "ADE_val_00000318_248.png"
    rec_split = "val"
    parent, fname_stem, segmentid = get_unique_mask_identifiers(
        dname, annot_fname, rec_split
    )
    assert parent == "validation"
    assert fname_stem == "ADE_val_00000318"
    assert segmentid == 248

    # relabeled as ()->()
    annot_fname = "ADE_train_00016125_113.png"
    rec_split = "train"
    parent, fname_stem, segmentid = get_unique_mask_identifiers(
        dname, annot_fname, rec_split
    )
    assert parent == "training"
    assert segmentid == 113
    assert fname_stem == "ADE_train_00016125"


def test_get_unique_mask_identifiers_coco() -> None:
    """ """
    dname = "coco-panoptic-133"
    annot_fname = "000000024880_7237508.png"
    rec_split = "train"
    parent, fname_stem, segmentid = get_unique_mask_identifiers(
        dname, annot_fname, rec_split
    )
    assert parent == "train2017"
    assert fname_stem == "000000024880"
    assert segmentid == 7237508


def test_get_unique_mask_identifiers_bdd() -> None:
    """ """
    dname = "bdd"

    # relabeled as person -> motorcyclist
    annot_fname = "0f4e8f1e-6ba53d52_11.jpg"
    rec_split = "train"
    parent, fname_stem, segmentid = get_unique_mask_identifiers(
        dname, annot_fname, rec_split
    )
    assert parent == "train"
    assert fname_stem == "0f4e8f1e-6ba53d52"
    assert segmentid == 11

    annot_fname = "972ab49a-6a6eeaf5_11.jpg"
    rec_split = "train"
    parent, fname_stem, segmentid = get_unique_mask_identifiers(
        dname, annot_fname, rec_split
    )
    assert parent == "train"
    assert fname_stem == "972ab49a-6a6eeaf5"
    assert segmentid == 11


def test_get_unique_mask_identifiers_cityscapes() -> None:
    """ """
    # relabeled as rider->bicyclist
    dname = "cityscapes-19"
    annot_fname = "seqfrankfurt_frankfurt_000000_013942_leftImg8bit_28.jpg"
    rec_split = "val"
    parent, fname_stem, segmentid = get_unique_mask_identifiers(
        dname, annot_fname, rec_split
    )
    assert parent == "frankfurt"
    assert fname_stem == "frankfurt_000000_013942_leftImg8bit"
    assert segmentid == 28


def test_get_unique_mask_identifiers_idd() -> None:
    """ """
    dname = "idd-39"

    # relabeled as rider->box
    annot_fname = "seq173_316862_leftImg8bit_34.jpg"
    rec_split = "train"
    parent, fname_stem, segmentid = get_unique_mask_identifiers(
        dname, annot_fname, rec_split
    )
    assert parent == "173"
    assert fname_stem == "316862_leftImg8bit"
    assert segmentid == 34

    # relabeled as ()->()
    annot_fname = "seq48_601929_leftImg8bit_79.jpg"
    rec_split = "dummy"
    parent, fname_stem, segmentid = get_unique_mask_identifiers(
        dname, annot_fname, rec_split
    )
    assert parent == "48"  # sequence ID
    assert segmentid == 79
    assert fname_stem == "601929_leftImg8bit"


def test_get_unique_mask_identifiers_sunrgbd() -> None:
    """
    relabeled as counter -> counter-other
    """
    dname = "sunrgbd-37"
    annot_fname = "img-002177_11.jpg"
    rec_split = "test"
    parent, fname_stem, segmentid = get_unique_mask_identifiers(
        dname, annot_fname, rec_split
    )
    assert parent == "test"
    assert fname_stem == "img-002177"
    assert segmentid == 11

    # segment_mask = mld.get_segment_mask(parent, rec.segmentid, fname_stem, split)


def test_get_unique_mask_identifiers_mapillary() -> None:
    """
    relabeled 'Ground Animal' -> horse
    """
    dname = "mapillary-public65"
    annot_fname = "mapillary_czkO_9In4u30opBy5H1uLg_259.jpg"
    rec_split = "train"
    parent, fname_stem, segmentid = get_unique_mask_identifiers(
        dname, annot_fname, rec_split
    )
    assert parent == "labels"
    assert fname_stem == "czkO_9In4u30opBy5H1uLg"
    assert segmentid == 259


"""
Do value tests on these masks now.
"""


# def test_form_fname_to_updatelist_dict_ade20k():
# 	"""
# 	"""
# 	test_update_records = [
# 		DatasetClassUpdateRecord('ade20k-v1', 'train', 	'table',	'bathroom-counter', 'ade20k_ade20k_table/ade20k_train_ade20k_table_to_bathroom-counter.txt'),
# 	]
# 	dictionary = form_fname_to_updatelist_dict(dname = 'ade20k-v3', update_records=test_update_records)


# def test_form_fname_to_updatelist_dict_cityscapes():
# 	"""
# 	"""
# 	test_update_records = [
# 		DatasetClassUpdateRecord(
# 			'cityscapes',
# 			'val',
# 			'rider-other',
# 			'bicycle',
# 			'cityscapes_cityscapes_rider/cityscapes_val_cityscapes_rider_to_bicycle.txt')
# 	]
# 	dictionary = form_fname_to_updatelist_dict(dname='cityscapes-v2', update_records=test_update_records)


# def test_form_fname_to_updatelist_dict_bdd():
# 	"""
# 	"""
# 	test_update_records = [
# 		DatasetClassUpdateRecord(
# 			'bdd',
# 			'val',
# 			'rider-other',
# 			'rider-other',
# 			'bdd_bdd_rider/bdd_val_bdd_rider_to_rider-other.txt')
# 	]
# 	dictionary = form_fname_to_updatelist_dict(dname='bdd-v2', update_records=test_update_records)


# def test_form_fname_to_updatelist_dict_idd():
# 	"""
# 	"""
# 	test_update_records = [
# 		DatasetClassUpdateRecord(
# 			'idd',
# 			'train',
# 			'rider-other',
# 			'pole',
# 			'idd_idd_rider/idd_train_idd_rider_to_pole.txt')
# 	]
# 	dictionary = form_fname_to_updatelist_dict(dname='idd-new-v2', update_records=test_update_records)


# def test_form_fname_to_updatelist_dict_cocopanoptic():
# 	"""
# 	"""
# 	test_update_records = [
# 		DatasetClassUpdateRecord(
# 			'cocopanoptic-v1',
# 			'val',
# 			'counter-other',
# 			'kitchen island',
# 			'cocop_cocop_counter/cocop_val_cocop_counter_to_kitchen-island.txt')
# 	]
# 	dictionary = form_fname_to_updatelist_dict(dname='coco-panoptic-v4', update_records=test_update_records)


# # def test_overwrite_label_img_masks():

# # 	overwrite_label_img_masks(
# # 		img_fpath,
# # 		label_img_fpath,
# # 		fname_to_updatelist_dict,
# # 		mld,
# # 		classname_to_id_map,
# # 		require_strict_boundaries,
# # 		split)


if __name__ == "__main__":
    """ """
    test_get_unique_mask_identifiers_ade20k()
    test_get_unique_mask_identifiers_coco()
    test_get_unique_mask_identifiers_bdd()
    test_get_unique_mask_identifiers_cityscapes()
    test_get_unique_mask_identifiers_idd()
    test_get_unique_mask_identifiers_sunrgbd()
    test_get_unique_mask_identifiers_mapillary()

# 	test_form_fname_to_updatelist_dict_ade20k()
# 	# test_form_fname_to_updatelist_dict_cityscapes()
# 	# test_form_fname_to_updatelist_dict_bdd()
# 	# test_form_fname_to_updatelist_dict_idd()
# 	# test_form_fname_to_updatelist_dict_cocopanoptic()
