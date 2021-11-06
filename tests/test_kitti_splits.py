#!/usr/bin/python3

from pathlib import Path

from mseg.utils.txt_utils import generate_all_img_label_pair_relative_fpaths


def test_completeness_img_lists() -> None:
    """
    We randomly selected the KITTI train and val sets, from 200
    trainval images.

    We ensure that our combined train and val dataset_list files
    have exactly 200 consecutively numbered files, of the form '000001_10.png'
    """
    for dname in ["kitti-19", "kitti-34"]:
        train_pairs = generate_all_img_label_pair_relative_fpaths(dname, "train")
        val_pairs = generate_all_img_label_pair_relative_fpaths(dname, "val")

        # make sure stems match in train
        for (rgb_fpath, label_fpath) in train_pairs:
            assert Path(rgb_fpath).stem == Path(label_fpath).stem
        train_stems = set([Path(r_fpath).stem for (r_fpath, _) in train_pairs])

        # make sure stems match in val
        for (rgb_fpath, label_fpath) in val_pairs:
            assert Path(rgb_fpath).stem == Path(label_fpath).stem
        val_stems = set([Path(r_fpath).stem for (r_fpath, _) in val_pairs])

        trainval_stems = set(train_stems) | set(val_stems)
        assert trainval_stems == set([f"{str(i).zfill(6)}_10" for i in range(200)])
