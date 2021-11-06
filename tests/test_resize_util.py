#!/usr/bin/python3

from pathlib import Path

import cv2
import imageio
import numpy as np

from mseg.utils.resize_util import (
    read_resize_write_label,
    read_resize_write_rgb,
    resize_img_by_short_side,
)

_TEST_DIR = Path(__file__).resolve().parent


def test_read_resize_write_label_landscape() -> None:
    """ """
    img = np.zeros((400, 800), dtype=np.uint8)
    img[np.arange(400), np.arange(400)] = 255
    old_fpath = f"{_TEST_DIR}/test_data/write_label_landscape_old.png"
    imageio.imwrite(old_fpath, img)

    new_fpath = f"{_TEST_DIR}/test_data/write_label_landscape_new.png"
    read_resize_write_label(old_fpath, new_fpath, short_side_sz=240)

    assert Path(old_fpath).exists()
    assert Path(new_fpath).exists()

    assert (240, 480) == cv2.imread(new_fpath, cv2.IMREAD_GRAYSCALE).shape
    assert (240, 480) == imageio.imread(new_fpath).shape


def test_read_resize_write_label_portrait() -> None:
    """ """
    img = np.zeros((800, 400), dtype=np.uint8)
    img[np.arange(400), np.arange(400)] = 255
    old_fpath = f"{_TEST_DIR}/test_data/write_label_portrait_old.png"
    imageio.imwrite(old_fpath, img)

    new_fpath = f"{_TEST_DIR}/test_data/write_label_portrait_new.png"
    read_resize_write_label(old_fpath, new_fpath, short_side_sz=240)

    assert Path(old_fpath).exists()
    assert Path(new_fpath).exists()

    assert (480, 240) == cv2.imread(new_fpath, cv2.IMREAD_GRAYSCALE).shape
    assert (480, 240) == imageio.imread(new_fpath).shape


def test_read_resize_write_rgb_portrait() -> None:
    """ """
    old_fpath = f"{_TEST_DIR}/test_data/portrait_rgb_old.jpg"
    new_fpath = f"{_TEST_DIR}/test_data/portrait_rgb_new.jpg"
    read_resize_write_rgb(old_fpath, new_fpath, short_side_sz=240)

    assert Path(old_fpath).exists()
    assert Path(new_fpath).exists()

    assert (279, 240, 3) == cv2.imread(new_fpath).shape
    assert (279, 240, 3) == imageio.imread(new_fpath).shape


def test_read_resize_write_rgb_landscape() -> None:
    """ """
    old_fpath = f"{_TEST_DIR}/test_data/landscape_rgb_old.png"
    new_fpath = f"{_TEST_DIR}/test_data/landscape_rgb_new.png"
    read_resize_write_rgb(old_fpath, new_fpath, short_side_sz=240)

    assert Path(old_fpath).exists()
    assert Path(new_fpath).exists()

    assert (240, 426, 3) == cv2.imread(new_fpath).shape
    assert (240, 426, 3) == imageio.imread(new_fpath).shape
    # 568


def test_resize_img_by_short_side_label1() -> None:
    """
    Downsample 100x200 image with random content to 10x20.
    Landscape Mode
    """
    label = np.random.randint(0, 255, size=(100, 200))
    label = label.astype(np.uint8)
    label_resized = resize_img_by_short_side(label, short_side_len=10, img_type="label")

    assert label_resized.shape == (10, 20)
    assert label_resized.dtype == np.uint8


def test_resize_img_by_short_side_label2() -> None:
    """
    Downsample 100x200 image with random content to 10x20.
    """
    label = np.random.randint(0, 255, size=(200, 100))
    label = label.astype(np.uint8)
    label_resized = resize_img_by_short_side(label, short_side_len=10, img_type="label")

    assert label_resized.shape == (20, 10)
    assert label_resized.dtype == np.uint8


def test_resize_img_by_short_side_rgb() -> None:
    """ """
    img_rgb = np.random.randn(800, 200, 3)
    img_rgb *= 255
    img_rgb = np.clip(img_rgb, 0, 255)
    img_rgb = img_rgb.astype(np.uint8)
    img_rgb_resized = resize_img_by_short_side(
        img_rgb, short_side_len=10, img_type="rgb"
    )

    assert img_rgb_resized.shape == (40, 10, 3)
    assert img_rgb_resized.dtype == np.uint8


if __name__ == "__main__":
    """ """
    # test_read_resize_write_label_landscape()
    # test_read_resize_write_label_portrait()
    test_read_resize_write_rgb_portrait()
    # test_read_resize_write_rgb_landscape()
    # test_resize_img_by_short_side_label1()
    # test_resize_img_by_short_side_label2()
    # test_resize_img_by_short_side_rgb()
