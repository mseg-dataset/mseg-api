#!/usr/bin/python3

from mseg.utils.txt_utils import generate_all_img_label_pair_relative_fpaths


def test_camvid_path_loading() -> None:
    """ """
    dname = "camvid-11"
    train_pairs = generate_all_img_label_pair_relative_fpaths(dname, split="train")
    val_pairs = generate_all_img_label_pair_relative_fpaths(dname, split="val")
    test_pairs = generate_all_img_label_pair_relative_fpaths(dname, split="test")

    assert 701 == len(train_pairs) + len(val_pairs) + len(test_pairs)
