#!/usr/bin/python3

from collections import defaultdict
import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

from typing import Dict, List


def scipy_conn_comp(img: np.ndarray) -> Dict[int, List[np.ndarray]]:
    """
            labelsndarray of dtype int
            Labeled array, where all connected regions
            are assigned the same integer value.

            numint, optional
            Number of labels, which equals the maximum label index
            and is only returned if return_num is True.

    Args:
       img:

    Returns:
       class_to_conncomps_dict:
    """
    class_to_conncomps_dict = defaultdict(list)
    present_class_idxs = np.unique(img)

    for class_idx in present_class_idxs:
        structure = np.ones((3, 3), dtype=np.int32)  # this defines the connection filter
        instance_img, nr_objects = scipy.ndimage.label(input=img == class_idx, structure=structure)

        for i in np.unique(instance_img):
            if i == 0:  # 0 doesn't count as an instance ID
                continue
            bin_arr = (instance_img == i).astype(np.uint8)
            class_to_conncomps_dict[class_idx] += [bin_arr]

    return class_to_conncomps_dict
