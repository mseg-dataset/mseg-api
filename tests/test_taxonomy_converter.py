#!/usr/bin/python3

import numpy as np
from pathlib import Path
import pdb
import torch

from mseg.utils.names_utils import (
    load_class_names,
    get_universal_class_names,
    get_classname_to_dataloaderid_map,
)
from mseg.utils.tsv_utils import read_tsv_column_vals

from mseg.taxonomy.taxonomy_converter import (
    parse_entry,
    parse_uentry,
    parse_test_entry,
    TaxonomyConverter,
    populate_linear_mapping,
    RELABELED_TRAIN_DATASETS,
    UNRELABELED_TRAIN_DATASETS,
)

_ROOT = Path(__file__).resolve().parent.parent


def entries_equal(dname: str, tsv_fpath: str, is_train_dataset: bool) -> bool:
    """
    Compare classnames in *_names.txt file against tsv column entries.
    For training datasets, these must be *exactly* the same.
    """
    tsv_classnames = read_tsv_column_vals(
        tsv_fpath, col_name=dname, convert_val_to_int=False
    )
    nonempty_classnames = [name for name in tsv_classnames if name != ""]
    tsv_classnames = []
    for entry in nonempty_classnames:
        tsv_classnames.extend(parse_entry(entry))
    txt_classnames = load_class_names(dname)
    if set(txt_classnames) != set(tsv_classnames):
        pdb.set_trace()

    if is_train_dataset:
        assert len(txt_classnames) == len(tsv_classnames)
        # ensure no duplicates among training dataset classnames
        assert len(list(tsv_classnames)) == len(set(tsv_classnames))
    return set(txt_classnames) == set(tsv_classnames)


def test_names_complete() -> None:
    """
    Test on dataset_config and on TaxonomyConverter

    Make sure tsv entries in a single column match EXACTLY
    to _names.txt file.
    """
    tsv_fpath = f"{_ROOT}/mseg/class_remapping_files/MSeg_master.tsv"

    train_dnames = UNRELABELED_TRAIN_DATASETS + RELABELED_TRAIN_DATASETS
    for dname in train_dnames:
        print(f"On {dname}...")
        assert entries_equal(dname, tsv_fpath, is_train_dataset=True)
        print(f"{dname} passed.")
        print()

    test_dnames = [
        "camvid-11",
        "kitti-19",
        #'pascal-context-60', # {'flower', 'wood'} missing
        "scannet-20",
        "voc2012",
        "wilddash-19",
    ]
    for dname in test_dnames:
        print(f"On {dname}")
        assert entries_equal(dname, tsv_fpath, is_train_dataset=False)


def test_parse_entry_blank() -> None:
    """ """
    entry = ""
    classes = parse_entry(entry)
    assert classes == []


def test_parse_entry_brackets1() -> None:
    """ """
    entry = "{house,building,  skyscraper,  booth,  hovel,  tower, grandstand}"
    classes = parse_entry(entry)
    gt_classes = [
        "house",
        "building",
        "skyscraper",
        "booth",
        "hovel",
        "tower",
        "grandstand",
    ]
    assert classes == gt_classes


def test_parse_entry_space_sep() -> None:
    """
    Note: ADE20K class "conveyer" is typo of "conveyor"
    """
    entry = "conveyer belt"
    classes = parse_entry(entry)
    assert classes == ["conveyer belt"]


def test_parse_uentry() -> None:
    """ """
    uentry = "animal_other"
    fullname = parse_uentry(uentry)
    assert fullname == "animal_other"


def test_label_transform() -> None:
    """
    Bring label from training taxonomy (mapillary-public65)
    to the universal taxonomy.

    21 is the motorcyclist class in mapillary-public65
    """
    dname = "mapillary-public65"
    txt_classnames = load_class_names(dname)
    train_idx = txt_classnames.index("Motorcyclist")
    tc = TaxonomyConverter()
    # training dataset label
    traind_label = torch.ones(4, 4) * train_idx
    traind_label = traind_label.type(torch.LongTensor)

    # Get back the universal label
    u_label = tc.transform_label(traind_label, dname)
    u_idx = get_universal_class_names().index("motorcyclist")
    gt_u_label = np.ones((4, 4)).astype(np.int64) * u_idx
    assert np.allclose(u_label.numpy(), gt_u_label)


def test_label_transform_unlabeled() -> None:
    """
    Make sure 255 stays mapped to 255 at each level (to be ignored in cross-entropy loss).
    """
    IGNORE_LABEL = 255
    dname = "mapillary-public65"
    txt_classnames = load_class_names(dname)
    name2id = get_classname_to_dataloaderid_map(dname, include_ignore_idx_cls=True)
    train_idx = name2id["unlabeled"]

    tc = TaxonomyConverter()
    # training dataset label
    traind_label = torch.ones(4, 4) * train_idx
    traind_label = traind_label.type(torch.LongTensor)

    # Get back the universal label
    u_label = tc.transform_label(traind_label, dname)
    u_idx = IGNORE_LABEL
    gt_u_label = np.ones((4, 4)).astype(np.int64) * u_idx
    assert np.allclose(u_label.numpy(), gt_u_label)


def test_transform_predictions_test() -> None:
    """
    Consider predictions made within the universal taxonomy
    over a tiny 2x3 image. We use a linear mapping to bring
    these predictions into a test dataset's taxonomy
    (summing the probabilities where necessary).

    For Camvid, universal probabilities for `person',`bicycle'
    should both go into the 'Bicyclist' class.
    """
    u_classnames = get_universal_class_names()
    person_uidx = u_classnames.index("person")
    bicycle_uidx = u_classnames.index("bicycle")
    sky_uidx = u_classnames.index("sky")

    tc = TaxonomyConverter()
    input = np.zeros((194, 2, 3))
    input[sky_uidx, 0, :] = 1.0  # top row is sky
    input[person_uidx, 1, :] = 0.5  # bottom row is 50/50 person or bicyclist
    input[bicycle_uidx, 1, :] = 0.5  # bottom row is 50/50 person or bicyclist
    input = torch.from_numpy(input)
    input = input.unsqueeze(0).float()  # CHW -> NCHW
    assert input.shape == (1, 194, 2, 3)

    test_dname = "camvid-11"
    output = tc.transform_predictions_test(input, test_dname)
    output = output.squeeze()  # NCHW -> CHW
    prediction = torch.argmax(output, dim=0).numpy()

    camvid_classnames = load_class_names(test_dname)
    # Camvid should have predictions across 11 classes.
    prediction_gt = np.zeros((2, 3))
    prediction_gt[0, :] = camvid_classnames.index("Sky")
    prediction_gt[1, :] = camvid_classnames.index("Bicyclist")
    assert np.allclose(prediction, prediction_gt)


def test_populate_linear_mapping1() -> None:
    """
    Implement simple matrix multiplication as 1x1 convolutions in PyTorch.

    [0]   [1 0 1 0] [0]
    [2] = [0 1 0 1] [1]
    [2]   [1 1 1 1] [0]
                    [1]
    """
    # fmt: off
    # (j,i) tuples
    inid2outid = [
    	(0,0),
    	(2,0),
    	(1,1),
    	(3,1),
    	(0,2),
    	(1,2),
    	(2,2),
    	(3,2)
    ]
    # fmt: on
    in_channel = 4
    out_channel = 3
    conv = populate_linear_mapping(in_channel, out_channel, inid2outid)

    x = np.array([0, 1, 0, 1]).reshape(1, 4, 1, 1).astype(np.float32)
    x = torch.from_numpy(x)
    y = conv(x)
    y_gt = np.array([0, 2, 2]).reshape(1, 3, 1, 1).astype(np.float32)
    y_gt = torch.from_numpy(y_gt)
    assert torch.allclose(y, y_gt)


def test_populate_linear_mapping2() -> None:
    """
    Implement simple matrix multiplication as 1x1 convolutions in PyTorch.

    [2]   [1 0 1 0] [1]
    [2] = [0 1 0 1] [1]
    [4]   [1 1 1 1] [1]
                    [1]
    """
    # fmt: off
    # (j,i) tuples
    inid2outid = [
    	(0,0),
    	(2,0),
    	(1,1),
    	(3,1),
    	(0,2),
    	(1,2),
    	(2,2),
    	(3,2)
    ]
    # fmt: on
    in_channel = 4
    out_channel = 3
    conv = populate_linear_mapping(in_channel, out_channel, inid2outid)

    x = torch.ones(1, 4, 1, 1).type(torch.FloatTensor)
    y = conv(x)
    y_gt = np.array([2, 2, 4]).reshape(1, 3, 1, 1).astype(np.float32)
    y_gt = torch.from_numpy(y_gt)
    assert torch.allclose(y, y_gt)


def test_populate_linear_mapping3() -> None:
    """
    Implement simple matrix multiplication as 1x1 convolutions in PyTorch.

    Consider the following example with universal predictions at a single px:
        armchair, swivel chair -> sum up to chair
        motorcycle -> motorcycle
        bicyclist, motorcyclist -> sum up to rider

         chair [0.3]   [1 1 0 0 0] [0.0] armchair
    motorcycle [0.1] = [0 0 1 0 0] [0.3] swivel_chair
         rider [0.6]   [0 0 0 1 1] [0.1] motorcycle
                                   [0.1] bicyclist
                                   [0.5] motorcyclist
    """
    # fmt: off
    # (j,i) tuples
    inid2outid = [
    	(0,0), # armchair -> chair
    	(1,0), # swivel_chair -> chair
    	(2,1), # motorcycle -> motorcycle
    	(3,2), # bicyclist -> rider
    	(4,2)  # motorcyclist -> rider
    ]
    # fmt: on
    in_channel = 5
    out_channel = 3
    conv = populate_linear_mapping(in_channel, out_channel, inid2outid)

    x = np.array([0.0, 0.3, 0.1, 0.1, 0.5])
    x = torch.from_numpy(x)
    x = x.reshape(1, 5, 1, 1).type(torch.FloatTensor)
    y = conv(x)
    y_gt = np.array([0.3, 0.1, 0.6]).reshape(1, 3, 1, 1).astype(np.float32)
    y_gt = torch.from_numpy(y_gt)
    assert torch.allclose(y, y_gt)


def test_constructor_types() -> None:
    """ """
    tc = TaxonomyConverter()
    for dname, conv in tc.convs.items():
        assert isinstance(conv, torch.nn.Module)


def test_label_mapping_arrs() -> None:
    """ """
    tc = TaxonomyConverter()
    train_idx = load_class_names("ade20k-150").index("minibike")
    u_idx = get_universal_class_names().index("motorcycle")
    assert tc.label_mapping_arr_dict["ade20k-150"][train_idx] == u_idx

    train_idx = load_class_names("mapillary-public65").index("Bird")
    u_idx = get_universal_class_names().index("bird")
    assert tc.label_mapping_arr_dict["mapillary-public65"][train_idx] == u_idx


if __name__ == "__main__":
    test_names_complete()
    test_parse_entry_blank()
    test_parse_entry_brackets1()
    test_parse_entry_space_sep()
    test_parse_uentry()
    test_label_transform()
    test_label_transform_unlabeled()
    test_label_transform_unlabeled()
    test_transform_predictions_test()
    test_populate_linear_mapping1()
    test_populate_linear_mapping2()
    test_populate_linear_mapping3()
    test_constructor_types()
    test_label_mapping_arrs()
