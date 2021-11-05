#!/usr/bin/python3

import numpy as np

from mseg.utils.cv2_utils import (
    grayscale_to_color,
    form_hstacked_imgs,
    form_vstacked_imgs,
    add_text_cv2,
)


def test_add_text_cv2() -> None:
    """
    Smokescreen
    """
    img = 255 * np.ones((512, 512, 3), np.uint8)
    text = "Hello World!"
    add_text_cv2(img, text)
    import matplotlib.pyplot as plt

    plt.imshow(img)
    # plt.show()
    plt.close()


def test_form_hstacked_imgs_three() -> None:
    """
    Horizontally stack three 2x2 RGB images into a single 2x6 RGB image.
    """
    hstack_save_fpath = "htmp1.png"

    img1 = np.zeros((2, 2, 3), dtype=np.uint8)
    img1[0, 0, :] = [255, 0, 1]

    img2 = np.zeros((2, 2, 3), dtype=np.uint8)
    img2[1, 1, :] = [5, 10, 15]

    img3 = np.zeros((2, 2, 3), dtype=np.uint8)
    img3[0, 1, :] = [255, 254, 253]

    img_list = [img1, img2, img3]

    hstack_img = form_hstacked_imgs(img_list, hstack_save_fpath, save_to_disk=False)

    # now has 6 columns
    gt_hstack_img = np.zeros((2, 6, 3), dtype=np.uint8)
    gt_hstack_img[0, 0, :] = [255, 0, 1]
    gt_hstack_img[1, 3, :] = [5, 10, 15]
    gt_hstack_img[0, 5, :] = [255, 254, 253]

    assert np.allclose(hstack_img, gt_hstack_img)


def test_form_hstacked_imgs_two() -> None:
    """
    Horizontally stack two 2x2 RGB images into a single 2x4 RGB image.
    """
    hstack_save_fpath = "htmp2.png"

    img1 = np.zeros((2, 2, 3), dtype=np.uint8)
    img1[0, 0, :] = [255, 0, 1]

    img2 = np.zeros((2, 2, 3), dtype=np.uint8)
    img2[1, 1, :] = [5, 10, 15]

    img_list = [img1, img2]

    hstack_img = form_hstacked_imgs(img_list, hstack_save_fpath, save_to_disk=False)

    gt_hstack_img = np.zeros((2, 4, 3), dtype=np.uint8)
    gt_hstack_img[0, 0, :] = [255, 0, 1]
    gt_hstack_img[1, 3, :] = [5, 10, 15]

    assert np.allclose(hstack_img, gt_hstack_img)


def test_form_vstacked_imgs_two() -> None:
    """
    Vertically stack two 2x2 RGB images into a single 2x4 RGB image.
    """
    vstack_save_fpath = "vtmp.png"

    img1 = np.zeros((2, 2, 3), dtype=np.uint8)
    img1[0, 0, :] = [255, 0, 1]

    img2 = np.zeros((2, 2, 3), dtype=np.uint8)
    img2[1, 1, :] = [5, 10, 15]

    img_list = [img1, img2]

    vstack_img = form_vstacked_imgs(img_list, vstack_save_fpath, save_to_disk=False)

    gt_vstack_img = np.zeros((4, 2, 3), dtype=np.uint8)
    gt_vstack_img[0, 0, :] = [255, 0, 1]
    gt_vstack_img[3, 1, :] = [5, 10, 15]

    assert np.allclose(vstack_img, gt_vstack_img)


def test_grayscale_to_color() -> None:
    """
    Convert simple 2x2 grayscale image into RGB image.
    """
    gray_img = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    rgb_img = grayscale_to_color(gray_img)
    assert rgb_img.shape == (2, 2, 3)
    for i in range(2):
        assert np.allclose(rgb_img[:, :, i], gray_img)
