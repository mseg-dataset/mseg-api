#!/usr/bin/python3

import cv2
import imageio
import numpy as np

from mseg.utils.cv2_utils import cv2_imread_rgb

"""
We use OpenCV, as opposed to `PIL` or `imageio`.
This is because we want to discard an alpha channel
and always load 3-channel RGB images, not 4-channel 
RGB images. Our dataloader requires 3-channel input.
"""


def resize_img_by_short_side(img: np.ndarray, short_side_len: int, img_type: str) -> np.ndarray:
    """
    Resize an RGB image and label.

    Args:
        img_rgb: Array of (M,N)

    Returns:
        img_rgb_resized
    """
    if img_type == "label":
        interp_mode = cv2.INTER_NEAREST
    elif img_type == "rgb":
        interp_mode = cv2.INTER_LINEAR

    img_h = img.shape[0]
    img_w = img.shape[1]

    if img_h > img_w:
        # higher, this is portrait photo
        fx = short_side_len / img_w
        fy = fx

    else:
        # wider, this is landscape photo
        fy = short_side_len / img_h
        fx = fy

    new_img_w = int(np.round(fx * img_w))
    new_img_h = int(np.round(fy * img_h))

    # Resize to (rows, cols)
    img_resized = cv2.resize(img, dsize=(new_img_w, new_img_h), interpolation=interp_mode)
    return img_resized


def read_resize_write_rgb(old_fpath: str, new_fpath: str, short_side_sz) -> None:
    """
    Args:
        old_fpath
        new_fpath
    """
    img_rgb = cv2_imread_rgb(old_fpath)
    img_rgb_resized = resize_img_by_short_side(img_rgb, short_side_len=short_side_sz, img_type="rgb")
    cv2.imwrite(new_fpath, img_rgb_resized[:, :, ::-1])


def read_resize_write_label(old_fpath: str, new_fpath: str, short_side_sz) -> None:
    """
    Args:
        old_fpath
        new_fpath
    """

    label = imageio.imread(old_fpath)
    label_resized = resize_img_by_short_side(label, short_side_len=short_side_sz, img_type="label")
    imageio.imwrite(new_fpath, label_resized)
