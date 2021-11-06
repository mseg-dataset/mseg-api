#!/usr/bin/python3

from pathlib import Path
import pdb

from mseg.utils.txt_utils import generate_all_img_label_pair_relative_fpaths
from mseg.utils.names_utils import read_str_list


_REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTS_DIR = Path(__file__).resolve().parent


def test_completeness_img_lists() -> None:
    """
    Ensure that our dataset_list files align with official VOC 2010 Main
    image lists.
    """
    for dname in ["pascal-context-60", "pascal-context-460"]:
        voc10_lists = {
            "train": f"{_TESTS_DIR}/test_data/pascal_context_splits/voc10_Main_train.txt",
            "val": f"{_TESTS_DIR}/test_data/pascal_context_splits/voc10_Main_val.txt",
        }
        train_pairs = generate_all_img_label_pair_relative_fpaths(dname, "train")
        train_stems = read_str_list(voc10_lists["train"])

        val_pairs = generate_all_img_label_pair_relative_fpaths(dname, "val")
        val_stems = read_str_list(voc10_lists["val"])

        trainval_pairs = generate_all_img_label_pair_relative_fpaths(dname, "trainval")

        # make sure stems match in train
        for (rgb_fpath, label_fpath) in train_pairs:
            assert Path(rgb_fpath).stem == Path(label_fpath).stem
        assert set(train_stems) == set(
            [Path(r_fpath).stem for (r_fpath, _) in train_pairs]
        )

        # make sure stems match in val
        for (rgb_fpath, label_fpath) in val_pairs:
            assert Path(rgb_fpath).stem == Path(label_fpath).stem
        assert set(val_stems) == set([Path(r_fpath).stem for (r_fpath, _) in val_pairs])

        trainval_stems = set(train_stems) | set(val_stems)
        assert trainval_stems == set(
            [Path(r_fpath).stem for (r_fpath, _) in trainval_pairs]
        )
