#!/usr/bin/python3

import pathlib

import imageio
import numpy as np

from mseg.utils.test_utils import dict_is_equal
from mseg.dataset_apis.Ade20kMaskLevelDataset import (
    Ade20kMaskDataset,
    get_ade20k_instance_label_masks,
)

_TEST_DIR = pathlib.Path(__file__).resolve().parent

# One set of data is from Scene Parsing Challenge, other is not.
TEST_ADE20K_SPC_IMGDIR = f"{_TEST_DIR}/test_data/ADE20K_test_data/ADEChallengeData2016"
TEST_ADE20K_NON_SPC_DATAROOT = (
    f"{_TEST_DIR}/test_data/ADE20K_test_data/ADE20K_2016_07_26"
)


def test_constructor() -> None:
    """
    Use a simplfied version of the full dataset (2 images only) to
    test for functionality.
    """
    amd = Ade20kMaskDataset(TEST_ADE20K_SPC_IMGDIR, TEST_ADE20K_NON_SPC_DATAROOT)
    gt_fname_to_rgbfpath_dict = {
        "ADE_train_00000001": f"{_TEST_DIR}/test_data/ADE20K_test_data/ADEChallengeData2016/images/training/ADE_train_00000001.jpg",
        "ADE_val_00000001": f"{_TEST_DIR}/test_data/ADE20K_test_data/ADEChallengeData2016/images/validation/ADE_val_00000001.jpg",
    }
    gt_fname_to_segrgbfpath_dict = {
        "ADE_train_00000001": f"{_TEST_DIR}/test_data/ADE20K_test_data/ADE20K_2016_07_26/images/training/a/aiport_terminal/ADE_train_00000001_seg.png",
        "ADE_val_00000001": f"{_TEST_DIR}/test_data/ADE20K_test_data/ADE20K_2016_07_26/images/validation/a/abbey/ADE_val_00000001_seg.png",
    }
    dict_is_equal(amd.fname_to_rgbfpath_dict, gt_fname_to_rgbfpath_dict)
    dict_is_equal(amd.fname_to_segrgbfpath_dict, gt_fname_to_segrgbfpath_dict)


def test_get_img_pair_val() -> None:
    """ """
    amd = Ade20kMaskDataset(TEST_ADE20K_SPC_IMGDIR, TEST_ADE20K_NON_SPC_DATAROOT)
    fname_stem = "ADE_val_00000001"
    rgb_img, label_img = amd.get_img_pair(fname_stem)

    rgb_fpath = f"{_TEST_DIR}/test_data/ADE20K_test_data/ADEChallengeData2016/images/validation/ADE_val_00000001.jpg"
    label_fpath = f"{_TEST_DIR}/test_data/ADE20K_test_data/ADEChallengeData2016/annotations/validation/ADE_val_00000001.png"

    gt_rgb_img = imageio.imread(rgb_fpath)
    gt_label_img = imageio.imread(label_fpath)
    assert np.allclose(rgb_img, gt_rgb_img)
    assert np.allclose(label_img, gt_label_img)


def test_get_img_pair_train() -> None:
    """ """
    amd = Ade20kMaskDataset(TEST_ADE20K_SPC_IMGDIR, TEST_ADE20K_NON_SPC_DATAROOT)
    fname_stem = "ADE_train_00000001"
    rgb_img, label_img = amd.get_img_pair(fname_stem)

    rgb_fpath = f"{_TEST_DIR}/test_data/ADE20K_test_data/ADEChallengeData2016/images/training/ADE_train_00000001.jpg"
    label_fpath = f"{_TEST_DIR}/test_data/ADE20K_test_data/ADEChallengeData2016/annotations/training/ADE_train_00000001.png"

    gt_rgb_img = imageio.imread(rgb_fpath)
    gt_label_img = imageio.imread(label_fpath)
    assert np.allclose(rgb_img, gt_rgb_img)
    assert np.allclose(label_img, gt_label_img)


def test_get_segment_mask() -> None:
    """
    Get a specific mask out of an image, and compare a few summary statistics
    to make sure correct mask is obtained.
    """
    amd = Ade20kMaskDataset(TEST_ADE20K_SPC_IMGDIR, TEST_ADE20K_NON_SPC_DATAROOT)

    # majority class will be 4, representing airport terminal's floor
    seq_id = None
    segmentid = 30  # int
    fname_stem = "ADE_train_00000001"  # str
    split = "train"  # str
    segment_mask = amd.get_segment_mask(seq_id, segmentid, fname_stem, split)

    assert segment_mask.sum() == 40609
    assert segment_mask.size == 349696
    assert segment_mask.mean() - 0.116 < 1e-3


def test_get_ade20k_instance_label_masks() -> None:
    """
    Make a simple at instance mask in "L" shape at 2x2 resolution,
    get back two instance masks at 4x4 resolution.
    """
    seg_rgb_img = np.zeros((2, 2, 3), dtype=np.uint8)
    seg_rgb_img[:, :, 2] = np.array([[5, 4], [5, 5]], dtype=np.uint8)
    seg_rgb_fpath = f"{_TEST_DIR}/test_data/test_seg_rgb_img.png"
    imageio.imwrite(seg_rgb_fpath, seg_rgb_img)
    rgb_img = np.zeros((4, 4, 3))
    rgb_img[:2, :2, :] = 50
    rgb_img[:2, 2:4, :] = 100
    rgb_img[2:4, :2, :] = 150
    rgb_img[2:4, 2:4, :] = 200
    instance_masks, instance_ids = get_ade20k_instance_label_masks(
        seg_rgb_fpath, rgb_img
    )

    gt_instance_ids = np.array([4, 5], dtype=np.uint8)
    assert np.allclose(instance_ids, gt_instance_ids)
    assert len(instance_masks) == 2
    # for instance ID "4"
    gt_inst_mask0 = np.array(
        [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
    )
    # for instance ID "5"
    gt_inst_mask1 = np.array(
        [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=np.uint8
    )
    assert np.allclose(instance_masks[0], gt_inst_mask0)
    assert np.allclose(instance_masks[1], gt_inst_mask1)
