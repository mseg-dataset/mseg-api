#!/usr/bin/python3

import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

from mseg.utils.dir_utils import create_leading_fpath_dirs


def cv2_write_rgb(save_fpath: str, img_rgb: np.ndarray) -> None:
    """
    Args:
        save_fpath: string representing absolute path where image should be saved
        img_rgb: (H,W,C) array representing Numpy image in RGB order
    """
    img_file_type = Path(save_fpath).suffix
    assert img_file_type in [".jpg", ".png"]
    create_leading_fpath_dirs(save_fpath)
    cv2.imwrite(save_fpath, img_rgb[:, :, ::-1])


def cv2_imread_rgb(fpath: str) -> np.ndarray:
    """
    Args:
        fpath:  string representing absolute path where image should be loaded from
    """
    if not Path(fpath).exists():
        print(f"{fpath} does not exist.")
        raise RuntimeError
        exit()
    return cv2.imread(fpath).copy()[:, :, ::-1]


def grayscale_to_color(gray_img: np.ndarray) -> np.ndarray:
    """Duplicate the grayscale channel 3 times.

    Args:
        gray_img: Array with shape (M,N)

    Returns:
        rgb_img: Array with shape (M,N,3)
    """
    h, w = gray_img.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(3):
        rgb_img[:, :, i] = gray_img
    return rgb_img


def form_hstacked_imgs(img_list: List[np.ndarray], hstack_save_fpath: str, save_to_disk: bool = True) -> np.ndarray:
    """
    Concatenate images along a horizontal axis and save them.
    Accept RGB images, and convert to BGR for OpenCV to save them.

    Args:
        img_list: list of Numpy arrays e.g. representing different RGB visualizations of same image,
            must all be of same height
        hstack_save_fpath: string, representing file path

    Returns:
        hstack_img: Numpy array representing RGB image, containing horizontally stacked images as tiles.
    """
    img_file_type = Path(hstack_save_fpath).suffix
    assert img_file_type in [".jpg", ".png"]
    create_leading_fpath_dirs(hstack_save_fpath)

    img_h, img_w, ch = img_list[0].shape
    assert ch == 3

    # height and number of channels must match
    assert all(img.shape[0] == img_h for img in img_list)
    assert all(img.shape[2] == ch for img in img_list)

    num_imgs = len(img_list)

    all_widths = [img.shape[1] for img in img_list]
    hstack_img = np.zeros((img_h, sum(all_widths), 3), dtype=np.uint8)

    running_w = 0
    for i, img in enumerate(img_list):
        h, w, _ = img.shape
        start = running_w
        end = start + w
        hstack_img[:, start:end, :] = img
        running_w += w

    if save_to_disk:
        cv2.imwrite(hstack_save_fpath, hstack_img[:, :, ::-1])
    return hstack_img


def form_vstacked_imgs(img_list: List[np.ndarray], vstack_save_fpath: str, save_to_disk: bool = True) -> np.ndarray:
    """
    Concatenate images along a vertical axis and save them.
    Accept RGB images, and convert to BGR for OpenCV to save them.

    Args:
        img_list: list of Numpy arrays representing different RGB visualizations of same image,
            must all be of same shape
        hstack_save_fpath: string, representing file path

    Returns:
        hstack_img: Numpy array representing RGB image, containing vertically stacked images as tiles.
    """
    img_file_type = Path(vstack_save_fpath).suffix
    assert img_file_type in [".jpg", ".png"]
    create_leading_fpath_dirs(vstack_save_fpath)

    img_h, img_w, ch = img_list[0].shape
    assert ch == 3

    # width and number of channels must match
    assert all(img.shape[1] == img_w for img in img_list)
    assert all(img.shape[2] == ch for img in img_list)

    num_imgs = len(img_list)
    all_heights = [img.shape[0] for img in img_list]
    vstack_img = np.zeros((sum(all_heights), img_w, 3), dtype=np.uint8)

    running_h = 0
    for i, img in enumerate(img_list):
        h, w, _ = img.shape
        start = running_h
        end = start + h
        vstack_img[start:end, :, :] = img
        running_h += h

    if save_to_disk:
        cv2.imwrite(vstack_save_fpath, vstack_img[:, :, ::-1])
    return vstack_img


def add_text_cv2(
    img: np.ndarray, text: str, coords_to_plot_at=None, font_color=(0, 0, 0), font_scale=1, thickness=2
) -> np.ndarray:
    """
    font_color = (0,0,0)
    x: x-coordinate from image origin to plot text at
    y: y-coordinate from image origin to plot text at
    """
    corner_offset = 5
    font = cv2.FONT_HERSHEY_TRIPLEX  # cv2.FONT_HERSHEY_SIMPLEX
    img_h, img_w, _ = img.shape

    if img_h < (corner_offset + 1) or img_w < (corner_offset + 1):
        return

    if coords_to_plot_at is None:
        coords_to_plot_at = (corner_offset, img_h - 1 - corner_offset)

    line_type = 2
    img = cv2.putText(
        img=img,
        text=text,
        org=coords_to_plot_at,
        fontFace=font,
        fontScale=font_scale,
        color=font_color,
        thickness=thickness,
        lineType=line_type,
    )
    return img
