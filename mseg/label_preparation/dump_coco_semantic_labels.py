#!/usr/bin/python3

import argparse
import imageio
import numpy as np
from pathlib import Path
import pdb
from typing import Any, List, Mapping, Tuple

from mseg.dataset_apis.COCOSemanticAPI import COCOSemanticAPI
from mseg.dataset_apis.COCOInstanceAPI import COCOInstanceAPI
from mseg.utils.mask_utils import swap_px_inside_mask
from mseg.utils.dataset_config import infos
from mseg.utils.txt_utils import generate_all_img_label_pair_fpaths
from mseg.utils.dir_utils import check_mkdir
from mseg.utils.multiprocessing_utils import send_list_to_workers


ROOT = Path(__file__).resolve().parent.parent

"""
Script to extract semantic images from panoptic data.
"""


class COCOSemanticExtractor:
    def __init__(self) -> None:  # , coco_dataroot):
        """
        We will remap the instance images to semantic images, using
        semantic JSON data. Data will be dumped in 201-class taxonomy.
        """
        self.orig_dname = "coco-panoptic-inst-201"
        self.orig_dataroot = infos[self.orig_dname].dataroot
        self.instance_api = COCOInstanceAPI(self.orig_dataroot)
        self.semantic_api = COCOSemanticAPI(self.orig_dataroot)

    def extract_semantic_imgs(self, num_processes: int) -> None:
        """
        Args:
            num_processes: number of workers to exploit
        """
        for split in ["train", "val"]:

            split_txt_fpath = f"{ROOT}/dataset_lists/{self.orig_dname}/list/{split}.txt"
            # load up the data root and all absolute paths from the txt file
            img_label_pairs = generate_all_img_label_pair_fpaths(self.orig_dataroot, split_txt_fpath)
            send_list_to_workers(
                num_processes=num_processes,
                list_to_split=img_label_pairs,
                worker_func_ptr=semantic_extractor_worker,
                cse=self,
                split=split,
            )


def semantic_extractor_worker(
    pairs: List[Tuple[str, str]], start_idx: int, end_idx: int, kwargs: Mapping[str, Any]
) -> None:
    """Given a list of (rgb image, label image) pairs to remap, call
    write_semantic_from_panoptic() on each one of them.

    Args:
        pairs: list of strings
        start_idx: integer index in list, where worker should start
        end_idx: integer index in list, where worker should stop
        kwargs: dictionary with argument names mapped to argument values
    """
    cse = kwargs["cse"]
    split = kwargs["split"]

    chunk_sz = end_idx - start_idx
    # process each image between start_idx and end_idx
    for idx in range(start_idx, end_idx):
        if idx % 1000 == 0:
            pct_completed = (idx - start_idx) / chunk_sz * 100
            print(f"Completed {pct_completed:.2f}%")
        _, instance_img_fpath = pairs[idx]
        write_semantic_from_panoptic(cse, split, instance_img_fpath)


def write_semantic_from_panoptic(
    cse: COCOSemanticExtractor, split: str, instance_img_fpath: str, ignore_idx: int = 255
) -> None:
    """
    Args:
        cse
        split
        instance_img_fpath
        ignore_idx
    """
    fname_stem = Path(instance_img_fpath).stem
    instance_id_img = cse.instance_api.get_instance_id_img(split, fname_stem)
    img_annot = cse.semantic_api.get_img_annotation(split, fname_stem)

    # default pixel value is unlabeled
    semantic_img = np.ones_like(instance_id_img, dtype=np.uint8) * ignore_idx
    for segment in img_annot["segments_info"]:
        segmentid = segment["id"]
        categoryid = segment["category_id"]
        segment_mask = (instance_id_img == segmentid).astype(np.uint8)

        semantic_img = swap_px_inside_mask(
            semantic_img, segment_mask, old_val=ignore_idx, new_val=categoryid, require_strict_boundaries=True
        )
    semantic_fpath = instance_img_fpath.replace(
        f"annotations/panoptic_{split}2017", f"semantic_annotations201/{split}2017"  # in 201-class taxonomy
    )
    check_mkdir(Path(semantic_fpath).parent)
    imageio.imwrite(semantic_fpath, semantic_img)


if __name__ == "__main__":
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_processes", type=int, default=100, help="number of processes to subdivide work between")
    args = parser.parse_args()

    print("Dumping COCO semantic labels into PNGs...")
    print(args)
    cse = COCOSemanticExtractor()  # coco_dataroot)
    cse.extract_semantic_imgs(args.num_processes)
