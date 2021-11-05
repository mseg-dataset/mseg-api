#!/usr/bin/python3

import argparse
import glob
import imageio
import numpy as np
from pathlib import Path
import scipy.io
from typing import Mapping

from mseg.utils.dir_utils import check_mkdir
from mseg.utils.names_utils import get_dataloader_id_to_classname_map


def dump_pascalcontext_mat_files(pcontext_dst_dir: str) -> None:
    """
    Convert PASCAL Context annotations from .mat files to .png

    Args:
        pcontext_dst_dir: string represent absolute path to PASCAL Context destination directory.
    """
    dataset_name = "pascal-context-460"
    id_to_class_name_map = get_dataloader_id_to_classname_map(dataset_name, include_ignore_idx_cls=False)

    save_dirname = "Segmentation_GT_460cls"
    png_save_dir = f"{pcontext_dst_dir}/{save_dirname}"
    check_mkdir(png_save_dir)

    # annotation files, stored as .mat files
    mat_files_dir = f"{pcontext_dst_dir}/trainval"
    mat_fpaths = glob.glob(f"{mat_files_dir}/*.mat")

    for i, mat_fpath in enumerate(mat_fpaths):
        if i % 500 == 0:
            print(f"On {i}/{len(mat_fpaths)}")
        fname_stem = Path(mat_fpath).stem

        label_data = scipy.io.loadmat(mat_fpath)
        label_img = label_data["LabelMap"]

        label_save_fpath = f"{png_save_dir}/{fname_stem}.png"

        # Need uint16 to be able to exceed 256 value range
        # there are up to 460 classes present.
        label_img_uint16 = label_img.astype(np.uint16)

        imageio.imwrite(label_save_fpath, label_img_uint16)
        loaded_label_img = imageio.imread(label_save_fpath)
        assert np.allclose(loaded_label_img, label_img)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pcontext_dst_dir",
        type=str,
        required=True,
        help="Directory where PASCAL Context lives, should end in `.../mseg_dataset/PASCAL_Context`",
    )

    args = parser.parse_args()
    dump_pascalcontext_mat_files(args.pcontext_dst_dir)
