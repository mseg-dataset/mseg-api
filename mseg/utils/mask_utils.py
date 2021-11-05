#!/usr/bin/python3

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import pdb
from PIL import Image, ImageDraw
import torch

from typing import List, Mapping, Optional, Tuple

from mseg.utils.conn_comp import scipy_conn_comp
from mseg.utils.colormap import colormap
from mseg.utils.cv2_utils import form_hstacked_imgs, add_text_cv2, form_vstacked_imgs, cv2_write_rgb
from mseg.utils.resize_util import resize_img_by_short_side
from mseg.utils.dir_utils import create_leading_fpath_dirs


NUM_PX_PER_ROW = 50
NUM_PX_PER_COL = 400
COLORMAP_OFFSET = 40
MIN_DISCERNABLE_RES_FOR_TEXT = 200
LIGHT_BLUE = np.array([153, 221, 255])
LIME_GREEN = np.array([57, 255, 20])


def get_mean_mask_location(mask):
    """Given a binary mask, find the mean location for entries equal to 1.

    Args:
        mask

    Returns:
        coordinate of mean pixel location as (x,y)
    """
    coords = np.vstack(np.where(mask == 1)).T
    return np.mean(coords, axis=0).astype(np.int32)


def find_max_cardinality_mask(mask_list: List[np.ndarray]):
    """
    Return the index of this element in the list
    """
    mask_cardinalities = [np.sum(mask) for mask in mask_list]
    return np.argmax(np.array(mask_cardinalities))


def search_jittered_location_in_mask(mean_x: float, mean_y: float, conncomp: np.ndarray) -> Tuple[int, int]:
    """
    For visualizing classnames in an image.

    When we wish to place text over a mask, for nonconvex regions, we cannot
    use mask pixel mean location (may not fall within mask), so we will
    jitter the location until we find a valid location within mask.
    """
    H, W = conncomp.shape
    num_attempts = 100
    for i in range(num_attempts):
        # grow the jitter up to half width of image at end
        SCALE = ((i + 1) / num_attempts) * (W / 2)
        dx, dy = np.random.randn(2) * SCALE
        # print(f'On iter {i}, mul noise w/ {SCALE} to get dx,dy={dx},{dy}')
        x = int(mean_x + dx)
        y = int(mean_y + dy)

        # Enforce validity
        x = max(0, x)
        x = min(W - 1, x)
        y = max(0, y)
        y = min(H - 1, y)

        if conncomp[y, x] != 1:
            continue
        else:
            return x, y

    return mean_x, mean_y


def save_classnames_in_image_sufficientpx(
    rgb_img: np.ndarray,
    label_img: np.ndarray,
    id_to_class_name_map: Mapping[int, str],
    font_color=(0, 0, 0),
    save_to_disk: bool = False,
    save_fpath: str = "",
    min_conncomp_px: int = 4000,
    font_scale: int = 1,
):
    """
    Write a classname over each connected component of a label
    map as long as the connected component has a sufficiently
    large number of pixels (specified as argument).

    Args:
        rgb_img: Numpy array (H,W,3) representing RGB image
        label_img: Numpy array (H,W) representing label map
        id_to_class_name_map: mapping from class ID to classname
        font_color: 3-tuple representing RGB font color
        save_to_disk: whether to save image to disk
        save_fpath: absolute file path
        min_conncomp_px: minimum number of pixels to justify
            placing a text label over connected component
        font_scale: scale of font text

    Returns:
        rgb_img: Numpy array (H,W,3) with embedded classanmes
    """
    H, W, C = rgb_img.shape
    class_to_conncomps_dict = scipy_conn_comp(label_img)

    for class_idx, conncomps_list in class_to_conncomps_dict.items():
        for conncomp in conncomps_list:
            if conncomp.sum() < min_conncomp_px:
                continue
            text = id_to_class_name_map[class_idx]

            y, x = get_mean_mask_location(conncomp)
            x -= 55  # move the text so approx. centered over mask.
            x = max(0, x)
            x = min(W - 1, x)

            # jitter location if nonconvex object mean not within its mask
            if conncomp[y, x] != 1:
                x, y = search_jittered_location_in_mask(x, y, conncomp)

            # print(f'Class idx: {class_idx}: (x,y)=({x},{y})')
            rgb_img = add_text_cv2(
                rgb_img, text, coords_to_plot_at=(x, y), font_color=font_color, font_scale=font_scale, thickness=2
            )

    if save_to_disk:
        cv2_write_rgb(save_fpath, rgb_img)

    return rgb_img


def save_classnames_in_image_maxcardinality(
    rgb_img, label_img, id_to_class_name_map, font_color=(0, 0, 0), save_to_disk: bool = False, save_fpath: str = ""
) -> np.ndarray:
    """
    Args:
        rgb_img
        label_img
        id_to_class_name_map: Mapping[int,str]

    Returns:
        rgb_img
    """
    H, W, C = rgb_img.shape
    class_to_conncomps_dict = scipy_conn_comp(label_img)

    for class_idx, conncomps_list in class_to_conncomps_dict.items():
        mask_idx = find_max_cardinality_mask(conncomps_list)
        maxsz_conncomp = conncomps_list[mask_idx]
        text = id_to_class_name_map[class_idx]

        y, x = get_mean_mask_location(maxsz_conncomp)
        x -= 55
        x = max(0, x)
        x = min(W - 1, x)
        # print(f'Class idx: {class_idx}: (x,y)=({x},{y})')
        rgb_img = add_text_cv2(
            rgb_img, text, coords_to_plot_at=(x, y), font_color=font_color, font_scale=1, thickness=2
        )

    if save_to_disk:
        cv2_write_rgb(save_fpath, rgb_img)

    return rgb_img


def form_mask_triple_embedded_classnames(
    rgb_img: np.ndarray,
    label_img: np.ndarray,
    id_to_class_name_map: Mapping[int, str],
    save_fpath: str,
    save_to_disk: bool = False,
) -> np.ndarray:
    """
    Args:
        rgb_img:
        label_img:
        id_to_class_name_map
        save_fpath
        save_to_disk

    Returns:
        Array, representing 3 horizontally concatenated images: from left-to-right, they are
            RGB, RGB+Semantic Masks, Semantic Masks
    """
    rgb_with_mask = convert_instance_img_to_mask_img(label_img, rgb_img.copy())

    # or can do max cardinality conn comp of each class
    rgb2 = save_classnames_in_image_sufficientpx(rgb_with_mask, label_img, id_to_class_name_map)
    mask_img = convert_instance_img_to_mask_img(label_img, img_rgb=None)
    rgb3 = save_classnames_in_image_sufficientpx(mask_img, label_img, id_to_class_name_map)
    return form_hstacked_imgs([rgb_img, rgb2, rgb3], save_fpath, save_to_disk)


def write_six_img_grid_w_embedded_names(
    rgb_img: np.ndarray,
    pred: np.ndarray,
    label_img: np.ndarray,
    id_to_class_name_map: Mapping[int, str],
    save_fpath: str,
) -> None:
    """
    Create a 6-image tile grid with the following structure:
    ------------------------------------------------------------
    RGB Image | Blended RGB+GT Label Map   | GT Label Map
    ------------------------------------------------------------
    RGB Image | Blended RGB+Pred Label Map | Predicted Label Map
    ------------------------------------------------------------
    We embed classnames directly into the predicted and ground
    truth label maps, instead of using a colorbar.

    Args:
        rgb_img:
        pred: predicted label map
        label_img
        id_to_class_name_map
        save_fpath
    """
    assert label_img.ndim == 2
    assert pred.ndim == 2
    assert rgb_img.ndim == 3
    label_hgrid = form_mask_triple_embedded_classnames(
        rgb_img, label_img, id_to_class_name_map, save_fpath="dummy.jpg", save_to_disk=False
    )
    pred_hgrid = form_mask_triple_embedded_classnames(
        rgb_img, pred, id_to_class_name_map, save_fpath="dummy.jpg", save_to_disk=False
    )
    vstack_img = form_vstacked_imgs(img_list=[label_hgrid, pred_hgrid], vstack_save_fpath=save_fpath, save_to_disk=True)


def map_semantic_img_fast_pytorch(semantic_img: np.ndarray, label_mapping_arr: np.ndarray) -> np.ndarray:
    """Quickly remap a semantic labelmap (integers) to a new taxonomy.

    TODO: may need to make a copy here, if it won't make one for us.

    Args:
        semantic_img: Pytorch CPU long tensor representing (M,N) matrix,
            with Torch Tensor type Long (int64)
        label_mapping_arr:  Pytorch CPU long tensor representing (K+1,1) array, where
            K represents the number of classes.  With Torch Tensor type Long (int64)

    Returns:
        img:  with Torch Tensor type Long (int64)
    """
    return label_mapping_arr[semantic_img.squeeze()].squeeze()


def form_label_mapping_array_pytorch(label_mapping_dict: Mapping[int, int]) -> np.ndarray:
    """
    Args:
        label_mapping_dict: dictionary from int to int, from original class ID to a new class ID.
            This is NOT the id_to_class_name dictionary.

    Returns:
        label_mapping_arr, with Torch Tensor type Long (int64)
    """
    keys_max = max(list(label_mapping_dict.keys()))
    arr_len = keys_max + 1

    label_mapping_arr = torch.zeros(arr_len).type(torch.LongTensor)
    for k, v in label_mapping_dict.items():
        label_mapping_arr[k] = v
    return label_mapping_arr


def map_semantic_img_fast(semantic_img: np.ndarray, label_mapping_arr: np.ndarray) -> np.ndarray:
    """
    Args:
        semantic_img:
        label_mapping_arr:

    Returns:
        img:
    """
    return label_mapping_arr[semantic_img.squeeze()].squeeze()


def form_label_mapping_array(label_mapping_dict: Mapping[int, int]) -> np.ndarray:
    """
    Args:
        label_mapping_dict: dictionary from int to int, from original class ID to a new class ID.
            This is NOT the id_to_class_name dictionary.
        dtype: data type, either np.uint8 or np.uint16 (default)

    Returns:
        label_mapping_arr
    """
    v_max = max(list(label_mapping_dict.values()))
    keys_max = max(list(label_mapping_dict.keys()))
    arr_len = keys_max + 1

    UINT8_MAX = np.iinfo("uint8").max  # 255 is uint8 max
    UINT16_MAX = np.iinfo("uint16").max  # 65535 is uint16 max

    if v_max > UINT8_MAX and v_max <= UINT16_MAX:
        dtype = np.uint16

    elif v_max <= UINT8_MAX:
        dtype = np.uint8
    else:
        print("Type wont fit, quitting...")
        quit()

    label_mapping_arr = np.zeros(arr_len, dtype=dtype)
    for k, v in label_mapping_dict.items():
        label_mapping_arr[k] = v
    return label_mapping_arr


def rgb_img_to_obj_cls_img(label_img_rgb: np.ndarray, dataset_ordered_colors: np.ndarray) -> np.ndarray:
    """Any unmapped pixels (given no corresponding RGB values) will default to zero'th-class.

    Args:
        label_img_rgb: Numpy array of shape (M,N,3)
        dataset_ordered_colors: Numpy array of shape (K,3) with RGB values for K classes

    Returns:
        object_cls_img: grayscale image
    """
    object_cls_img = np.zeros((label_img_rgb.shape[0], label_img_rgb.shape[1]), dtype=np.uint8)
    for i, color in enumerate(dataset_ordered_colors):

        indices = np.where(np.all(label_img_rgb == color, axis=-1))
        object_cls_img[indices[0], indices[1]] = i
    return object_cls_img


def save_mask_triple_isolated_mask(
    rgb_img: np.ndarray, label_img: np.ndarray, id_to_class_name_map, class_name: str, save_fpath: str
) -> None:
    """Save a triplet of images to disk (RGB image, label map, and a blended version of the two).

    Args:
        rgb_img:
        label_img:
        id_to_class_name_map:
        class_name:
        save_fpath:
    """
    for id, proposed_class_name in id_to_class_name_map.items():
        if class_name == proposed_class_name:
            break

    isolated_rgb_mask = np.ones_like(rgb_img) * 255
    y, x = np.where(label_img == id)
    isolated_rgb_mask[y, x, :] = rgb_img.copy()[y, x, :]

    rgb_with_mask = convert_instance_img_to_mask_img(label_img, rgb_img.copy())
    mask_img = convert_instance_img_to_mask_img(label_img)

    concat_img = form_hstacked_imgs(
        [rgb_img, isolated_rgb_mask, rgb_with_mask, mask_img], save_fpath, save_to_disk=False
    )

    cv2.imwrite(save_fpath, concat_img[:, :, ::-1])


def save_img_with_blendedmaskimg(
    rgb_img: np.ndarray, label_img: np.ndarray, save_fpath: str, save_to_disk: bool = False
) -> None:
    """
    Args:
        rgb_img:
        label_img:
        save_fpath
        save_to_disk

    Returns:
        Array, representing 3 horizontally concatenated images: from left-to-right, they are
            RGB, RGB+Semantic Masks, Semantic Masks
    """
    rgb_with_mask = highlight_binary_mask(label_img, rgb_img.copy())
    return form_hstacked_imgs([rgb_img, rgb_with_mask], save_fpath, save_to_disk)


def save_binary_mask_triple(
    rgb_img: np.ndarray, label_img: np.ndarray, save_fpath: str, save_to_disk: bool = False
) -> np.ndarray:
    """Currently mask img background is light-blue. Instead, could set it to white. np.array([255,255,255])

    Args:
        rgb_img:
        label_img:
        save_fpath
        save_to_disk

    Returns:
        Array, representing 3 horizontally concatenated images: from left-to-right, they are
            RGB, RGB+Semantic Masks, Semantic Masks
    """
    img_h, img_w, _ = rgb_img.shape
    rgb_with_mask = highlight_binary_mask(label_img, rgb_img.copy())

    blank_img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
    y, x = np.where(label_img == 0)
    blank_img[y, x, :] = LIME_GREEN  # LIGHT_BLUE
    mask_img = highlight_binary_mask(label_img, blank_img)

    return form_hstacked_imgs([rgb_img, rgb_with_mask, mask_img], save_fpath, save_to_disk)


def save_binary_mask_double(
    rgb_img: np.ndarray, label_img: np.ndarray, save_fpath: str, save_to_disk: bool = False
) -> np.ndarray:
    """Currently blended mask img background is lime green.

    Args:
        rgb_img:
        label_img:
        save_fpath
        save_to_disk

    Returns:
        Array, representing 2 horizontally concatenated images: from left-to-right, they are
            RGB, RGB+Semantic Masks
    """
    img_h, img_w, _ = rgb_img.shape
    lime_green_rgb = vis_mask(rgb_img.copy(), 1 - label_img, LIME_GREEN, alpha=0.2)
    rgb_with_mask = highlight_binary_mask(label_img, lime_green_rgb)
    return form_hstacked_imgs([rgb_img, rgb_with_mask], save_fpath, save_to_disk)


def highlight_binary_mask(label_mask: np.ndarray, img_rgb: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Given a grayscale image where intensities denote instance IDs (same intensity denotes
    belonging to same instance), convert this to an RGB image where all pixels corresponding
    to the same instance get the same color. Note that two instances may not have unique colors,
    do to a finite-length colormap.

    Args:
        instance_img: Numpy array of shape (M,N), representing grayscale image, in [0,255]
        img_rgb: Numpy array representing RGB image, possibly blank, in [0,255]

    Returns:
        img_rgb:
    """
    img_h, img_w = label_mask.shape
    if img_rgb is None:
        img_rgb = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

    assert label_mask.dtype in [
        np.uint8,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
    ], "Label map is not composed of integers."
    assert img_rgb.dtype in [np.uint8, np.uint16]
    our_colormap = colormap(rgb=True)

    # np.unique will always sort the values
    both01 = np.allclose(np.unique(label_mask), np.array([0, 1]))
    all1 = np.allclose(np.unique(label_mask), np.array([1]))
    assert both01 or all1

    # 40 is blue, 50 is pink
    col = np.array([255, 0, 255], dtype=np.uint8)
    img_rgb = vis_mask(img_rgb, label_mask, col, alpha=0.4)
    return img_rgb


def save_pred_vs_label_7tuple(
    img_rgb: np.ndarray,
    pred_img: np.ndarray,
    label_img: np.ndarray,
    id_to_class_name_map: Mapping[int, str],
    save_fpath: str,
) -> None:
    """7-tuple consists of
                    (1-3) rgb mask 3-sequence for label,
                    (4-6) rgb mask 3-sequence for predictions,
                    (7) color palette

    Args:
        img_rgb
        pred_img
        label_img
        id_to_class_name_map
        save_fpath
    """
    img_h, img_w, _ = img_rgb.shape
    assert pred_img.shape == (img_h, img_w)
    assert label_img.shape == (img_h, img_w)

    if min(img_h, img_w) < MIN_DISCERNABLE_RES_FOR_TEXT:
        save_pred_vs_label_7tuple(
            img_rgb=resize_img_by_short_side(
                img_rgb.copy(), short_side_len=MIN_DISCERNABLE_RES_FOR_TEXT, img_type="rgb"
            ),
            pred_img=resize_img_by_short_side(
                pred_img.copy(), short_side_len=MIN_DISCERNABLE_RES_FOR_TEXT, img_type="label"
            ),
            label_img=resize_img_by_short_side(
                label_img.copy(), short_side_len=MIN_DISCERNABLE_RES_FOR_TEXT, img_type="label"
            ),
            id_to_class_name_map=id_to_class_name_map,
            save_fpath=save_fpath.replace(".png", "_upsample.png"),
        )

    NUM_HSTACKED_IMGS = 3
    hstack_img1 = form_mask_triple(img_rgb, label_img, save_fpath, save_to_disk=False)
    hstack_img2 = form_mask_triple(img_rgb, pred_img, save_fpath, save_to_disk=False)

    vstack_img1 = np.vstack([hstack_img1, hstack_img2])

    save_dir = "/".join(save_fpath.split("/")[:-1])

    present_color_ids = np.union1d(np.unique(label_img), np.unique(pred_img))
    num_present_colors = len(present_color_ids)
    max_colors_per_col = int(math.ceil(num_present_colors / NUM_HSTACKED_IMGS))

    palette_img = form_contained_classes_color_guide(
        present_color_ids, id_to_class_name_map, "", "", save_to_disk=False, max_colors_per_col=max_colors_per_col
    )

    vstack_img2 = vstack_img_with_palette(vstack_img1, palette_img)

    save_fpath = save_fpath.replace(".png", "_pred_labels_palette.png")
    cv2.imwrite(save_fpath, vstack_img2[:, :, ::-1])


def save_pred_vs_label_4tuple(
    img_rgb: np.ndarray, label_img: np.ndarray, id_to_class_name_map: Mapping[int, str], save_fpath: str
) -> None:
    """7-tuple consists of
                    (1-3) rgb mask 3-sequence for label or predictions
                    (4) color palette

    Args:
        img_rgb
        label_img
        id_to_class_name_map
        save_fpath
    """
    img_h, img_w, _ = img_rgb.shape
    assert label_img.shape == (img_h, img_w)

    if min(img_h, img_w) < MIN_DISCERNABLE_RES_FOR_TEXT:
        save_pred_vs_label_4tuple(
            img_rgb=resize_img_by_short_side(
                img_rgb.copy(), short_side_len=MIN_DISCERNABLE_RES_FOR_TEXT, img_type="rgb"
            ),
            label_img=resize_img_by_short_side(
                label_img.copy(), short_side_len=MIN_DISCERNABLE_RES_FOR_TEXT, img_type="label"
            ),
            id_to_class_name_map=id_to_class_name_map,
            save_fpath=save_fpath.replace(".png", "_upsample.png"),
        )

    NUM_HSTACKED_IMGS = 3
    hstack_img = form_mask_triple(img_rgb, label_img, save_fpath, save_to_disk=False)
    save_dir = "/".join(save_fpath.split("/")[:-1])

    present_color_ids = np.unique(label_img)
    num_present_colors = len(present_color_ids)
    max_colors_per_col = int(math.ceil(num_present_colors / NUM_HSTACKED_IMGS))

    palette_img = form_contained_classes_color_guide(
        present_color_ids, id_to_class_name_map, "", "", save_to_disk=False, max_colors_per_col=max_colors_per_col
    )

    vstack_img2 = vstack_img_with_palette(hstack_img, palette_img)
    save_fpath = save_fpath.replace(".png", "_pred_labels_palette.png")
    cv2.imwrite(save_fpath, vstack_img2[:, :, ::-1])


def vstack_img_with_palette(top_img: np.ndarray, palette_img: np.ndarray) -> np.ndarray:
    """Vertically stack an image and a palette image, placing the palette image below it.

    Args:
        top_img
        palette_img

    Returns:
        vstack_img
    """
    img_n_rows = top_img.shape[0]
    palette_n_rows = palette_img.shape[0]

    img_n_cols = top_img.shape[1]
    palette_n_cols = palette_img.shape[1]

    fx = img_n_cols / palette_n_cols
    fy = fx

    rsz_cols = int(np.round(fx * palette_n_cols))
    rsz_rows = int(np.round(fy * palette_n_rows))
    rsz_palette_img = cv2.resize(palette_img, dsize=(rsz_cols, rsz_rows), interpolation=cv2.INTER_NEAREST)

    concat_rows = img_n_rows + rsz_rows
    vstack_img = np.zeros((concat_rows, img_n_cols, 3), dtype=np.uint8)
    vstack_img[:img_n_rows, :, :] = top_img
    vstack_img[img_n_rows:, :, :] = rsz_palette_img
    return vstack_img


def save_mask_triple_with_color_guide(
    img_rgb: np.ndarray,
    label_img: np.ndarray,
    id_to_class_name_map: Mapping[int, str],
    fname_stem: str,
    save_dir: str,
    save_fpath: str,
) -> None:
    """
    Args:
        img_rgb: Array representing 3-channel image in RGB order
        label_img: Array representing grayscale image, where intensities correspond to semantic classses
        id_to_class_name_map: dictionary that maps a grayscale intensity to a class name
        fname_stem: string, representing unique name for image, e.g. `coco_40083bx` for `coco_40083bx.png`
        save_dir: string, dir where to save output image, e.g. /my/save/directory
        save_fpath: string, representing full absolute path to where image will be saved, e.g.
            /my/save/directory/coco_40083bx.png
    """

    # for every 10 classes, save a new image

    palette_img = form_contained_classes_color_guide(
        label_img, id_to_class_name_map, fname_stem, save_dir, save_to_disk=False
    )

    hstack_img = form_mask_triple(img_rgb, label_img, save_fpath, save_to_disk=False)
    rgb_plus_palette_img = hstack_img_with_palette(hstack_img, palette_img)

    save_fpath = save_fpath.replace(".png", "_concat.png")
    cv2.imwrite(save_fpath, rgb_plus_palette_img[:, :, ::-1])


def hstack_img_with_palette(left_img: np.array, palette_img: np.array) -> np.ndarray:
    """Horizontally stack a left image with a palette image on the right."""
    img_n_rows = left_img.shape[0]
    palette_n_rows = palette_img.shape[0]

    img_n_cols = left_img.shape[1]
    palette_n_cols = palette_img.shape[1]

    fy = img_n_rows / palette_n_rows
    fx = fy

    rsz_cols = int(np.round(fx * palette_n_cols))
    rsz_rows = int(np.round(fy * palette_n_rows))
    rsz_palette_img = cv2.resize(palette_img, dsize=(rsz_cols, rsz_rows), interpolation=cv2.INTER_NEAREST)

    concat_cols = img_n_cols + rsz_cols
    hstack_img = np.zeros((img_n_rows, concat_cols, 3), dtype=np.uint8)
    hstack_img[:, :img_n_cols, :] = left_img
    hstack_img[:, img_n_cols:, :] = rsz_palette_img
    return hstack_img


def form_contained_classes_color_guide(
    label_img: np.ndarray,
    id_to_class_name_map: Mapping[int, str],
    fname_stem: str,
    save_dir: str,
    save_to_disk: bool = True,
    max_colors_per_col: int = 10,
) -> np.ndarray:
    """
    Write out an image explaining the classes inside an image.

    Args:
        label_img
        id_to_class_name_map
        fname_stem
        save_dir

    Returns:
        palette_img: Array with cells colored with class color from palette
    """
    ids_present = np.unique(label_img)
    num_cols = math.ceil(len(ids_present) / max_colors_per_col)
    num_rows = max_colors_per_col
    palette_img = np.zeros((NUM_PX_PER_ROW * num_rows, NUM_PX_PER_COL * num_cols, 3), dtype=np.uint8)

    for i, labelid in enumerate(ids_present):
        col_idx = i // max_colors_per_col
        row_idx = i % max_colors_per_col
        class_name = id_to_class_name_map[labelid]
        id_img = np.ones((NUM_PX_PER_ROW, NUM_PX_PER_COL), dtype=np.uint16) * labelid
        blank_img = 255 * np.ones((NUM_PX_PER_ROW, NUM_PX_PER_COL, 3), dtype=np.uint8)
        blank_img = add_text_cv2(blank_img, str(class_name))
        color_img = convert_instance_img_to_mask_img(id_img, blank_img)
        # vertical pixel start
        v_start = row_idx * NUM_PX_PER_ROW
        v_end = (row_idx + 1) * NUM_PX_PER_ROW
        # horizontal pixel start
        h_start = NUM_PX_PER_COL * col_idx
        h_end = h_start + NUM_PX_PER_COL
        palette_img[v_start:v_end, h_start:h_end, :] = color_img
    palette_save_fpath = f"{save_dir}/{fname_stem}_colors.png"
    if save_to_disk:
        cv2.imwrite(palette_save_fpath, palette_img[:, :, ::-1])

    return palette_img


def form_mask_triple(
    rgb_img: np.ndarray, label_img: np.ndarray, save_fpath: str, save_to_disk: bool = False
) -> np.ndarray:
    """
    Args:
        rgb_img:
        label_img:
        save_fpath
        save_to_disk

    Returns:
        Array, representing 3 horizontally concatenated images: from left-to-right, they are
            RGB, RGB+Semantic Masks, Semantic Masks
    """
    rgb_with_mask = convert_instance_img_to_mask_img(label_img, rgb_img.copy())
    mask_img = convert_instance_img_to_mask_img(label_img, img_rgb=None)
    return form_hstacked_imgs([rgb_img, rgb_with_mask, mask_img], save_fpath, save_to_disk)


def form_mask_triple_vertical(
    rgb_img: np.ndarray, label_img: np.ndarray, save_fpath: str, save_to_disk: bool = False
) -> np.ndarray:
    """
    Args:
        rgb_img:
        label_img:
        save_fpath
        save_to_disk

    Returns:
    -	Array, representing 3 horizontally concatenated images: from left-to-right, they are
                    RGB, RGB+Semantic Masks, Semantic Masks
    """
    rgb_with_mask = convert_instance_img_to_mask_img(label_img, rgb_img.copy())
    mask_img = convert_instance_img_to_mask_img(label_img, img_rgb=None)
    return form_vstacked_imgs([rgb_img, rgb_with_mask, mask_img], save_fpath, save_to_disk)


def convert_instance_img_to_mask_img(instance_img: np.ndarray, img_rgb: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Given a grayscale image where intensities denote instance IDs (same intensity denotes
    belonging to same instance), convert this to an RGB image where all pixels corresponding
    to the same instance get the same color. Note that two instances may not have unique colors,
    do to a finite-length colormap.

    Args:
        instance_img: Numpy array of shape (M,N), representing grayscale image, in [0,255]
        img_rgb: Numpy array representing RGB image, possibly blank, in [0,255]

    Returns:
        img_rgb:
    """
    img_h, img_w = instance_img.shape
    if img_rgb is None:
        img_rgb = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

    assert instance_img.dtype in [
        np.uint8,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
    ], "Label map is not composed of integers."
    assert img_rgb.dtype in [np.uint8, np.uint16]
    our_colormap = colormap(rgb=True)
    num_unique_colors = our_colormap.shape[0]
    # np.unique will always sort the values
    if np.unique(instance_img).size > 0:
        for i, instance_id in enumerate(np.unique(instance_img)):
            col = our_colormap[(COLORMAP_OFFSET + instance_id) % num_unique_colors]
            mask = instance_img == instance_id
            img_rgb = vis_mask(img_rgb, mask, col, alpha=0.4)
    return img_rgb


def vis_mask(img: np.ndarray, mask: np.ndarray, col: Tuple[int, int, int], alpha: float = 0.4):
    """
    Visualizes a single binary mask by coloring the region inside a binary mask
    as a specific color, and then blending it with an RGB image.

    Args:
        img: Numpy array, representing RGB image with values in the [0,255] range
        mask: Numpy integer array, with values in [0,1] representing mask region
        col: color, tuple of integers in [0,255] representing RGB values
        alpha: blending coefficient (higher alpha shows more of mask,
            lower alpha preserves original image)

    Returns:
        image: Numpy array, representing an RGB image, representing a blended image
            of original RGB image and specified colors in mask region.
    """
    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * col

    return img.astype(np.uint8)


def visualize_colormap():
    """ """
    label_img = np.random.randint(0, 80, size=(100, 100))
    id_to_class_name_map = {i: str(i) for i in range(80)}
    form_contained_classes_color_guide(
        label_img=label_img,
        id_to_class_name_map=id_to_class_name_map,
        fname_stem="fb_colormap",
        save_dir=".",
        save_to_disk=True,
        max_colors_per_col=100,
    )


def swap_px_inside_mask(
    label_img: np.ndarray, segment_mask: np.ndarray, old_val: int, new_val: int, require_strict_boundaries: bool
):
    """
    Args:
        label_img: label map before any update has taken place.
        segment_mask: 0/1 binary image showing segment pixels
        old_val: old pixel value/category
        new_val: new pixel value/category
        require_strict_boundaries:

    Returns:
    -	label_img: updated label map
    """
    unique_vals = np.unique(segment_mask)
    zeros_and_ones = np.array([0, 1], dtype=np.uint8)
    all_ones = np.array([1], dtype=np.uint8)
    # ok if the mask is all 1's (fill entire image)
    assert np.allclose(zeros_and_ones, unique_vals) or np.allclose(all_ones, unique_vals)
    assert label_img.ndim == 2
    y, x = np.where(segment_mask == 1)

    # if perfect agreement between label image and instance image, then we will enforce
    # strict boundary agreement between the two.
    if require_strict_boundaries:
        assert np.allclose(np.unique(label_img[y, x]), np.array([old_val], dtype=np.uint8))
    label_img[y, x] = new_val
    return label_img


def get_instance_mask_class_votes(
    instance_mask: np.ndarray, label_img: np.ndarray, verbose: bool = False
) -> Tuple[np.ndarray, int]:
    """

    Since the class masks to instance masks don't match up exactly, and are provided
    in images with very different resolutions, we have to take the majority vote
    for which semantic class the instance mask belongs to.

    Args:
        instance_mask
        label_img

    Returns:
        label_votes: 1d array, containing all category votes from instance mask
        majority_vote: most likely category for this instance.
    """
    coords = np.vstack(np.where(instance_mask == 1)).T

    label_votes = label_img[coords[:, 0], coords[:, 1]]
    majority_vote = get_np_mode(label_votes)

    if verbose:
        unique_class_idxs = np.unique(label_votes)
        if unique_class_idxs.size > 1:
            for class_idx in unique_class_idxs:
                percent = (label_votes == class_idx).sum() / label_votes.size * 100
                print(f"\t {percent:.2f}%")
            print()

    return label_votes, majority_vote


def get_np_mode(x: np.ndarray) -> int:
    """Get mode value of a 1d or 2d integer array.

    Args:
        x: Numpy array of integers:

    Returns:
        integer representing mode of array values
    """
    assert x.dtype in [np.uint8, np.uint16, np.int16, np.int32, np.int64]
    counts = np.bincount(x)
    return np.argmax(counts)


def get_mask_from_polygon(polygon, img_h: int, img_w: int):
    """Rasterize a 2d polygon to a binary mask.

    Note: 60x faster than the Matplotlib rasterizer... well done pillow!

    Args:
        polygon: iterable e.g. [(x1,y1),(x2,y2),...]
        img_h: integer representing image height
        img_w: integer representing image width

    Returns:
        mask

    PIL.Image.new(mode, size, color=0)
    Creates a new image with the given mode and size.
    Parameters:
    mode – The mode to use for the new image. See: Modes.
    size – A 2-tuple, containing (width, height) in pixels.
    color – What color to use for the image. Default is black.
            If given, this should be a single integer or floating point
            value for single-band modes, and a tuple for multi-band modes
            (one value per band). When creating RGB images, you can also
            use color strings as supported by the ImageColor module. If
            the color is None, the image is not initialised.

    https://pillow.readthedocs.io/en/3.1.x/reference/ImageDraw.html
    The polygon outline consists of straight lines between the given coordinates,
    plus a straight line between the last and the first coordinate.

    Parameters:
    xy – Sequence of either 2-tuples like [(x, y), (x, y), ...] or numeric values like [x, y, x, y, ...].
    outline – Color to use for the outline.
    fill – Color to use for the fill.
    """
    polygon = [tuple([x, y]) for (x, y) in polygon]
    # this is the image that we want to create
    mask_img = Image.new("L", size=(img_w, img_h), color=0)

    # include outline
    # a drawer to draw into the image
    if len(set([y for (x, y) in polygon])) == 1:
        # draw only a line. because of known bug in .polygon() method
        # https://github.com/python-pillow/Pillow/issues/4674
        ImageDraw.Draw(mask_img).line(polygon, fill=1)
    else:
        ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
    mask = np.array(mask_img)
    return mask


def get_present_classes_in_img(label_img: np.ndarray, id_to_classname_map) -> List[str]:
    """
    Args:
        label_img:

    Returns:
        list of strings, representing classnames
    """
    present_class_idxs = np.unique(label_img)
    present_classnames = [id_to_classname_map[idx] for idx in present_class_idxs]
    return present_classnames


def get_most_populous_class(segment_mask: np.array, label_map: np.ndarray) -> int:
    """
    Args:
    -	segment_mask
    -	label_map

    Returns:
    -	class_mode_idx: integer representing most populous class index
    """
    class_indices = label_map[segment_mask.nonzero()]
    class_mode_idx = get_np_mode(class_indices)
    return class_mode_idx


def get_polygons_from_binary_img(binary_img: np.ndarray) -> Tuple[List[np.ndarray], Optional[bool]]:
    """
    cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    Internal contours (holes) are placed in hierarchy-2.
    cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.

    Args:
        binary_img: Numpy array with all 0s or 1s

    Returns:
        res: list of polygons, each (N,2) np.ndarray
    """
    assert all([val in [0, 1] for val in np.unique(binary_img)])
    binary_img = binary_img.astype("uint8")
    res, hierarchy = cv2.findContours(binary_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    if hierarchy is None:  # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0

    res = [x.squeeze() for x in res]
    res = [x for x in res if x.size >= 6]  # should have at least 3 vertices to be valid
    return res, has_holes
