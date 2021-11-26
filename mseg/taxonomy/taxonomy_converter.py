#!/usr/bin/python3

from collections import defaultdict, namedtuple
from pathlib import Path
from typing import List, Mapping, Tuple

import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn

from mseg.utils.names_utils import load_class_names

"""
We train in a common, unified label space.

Inference is also performed in that common, unified label space,
and predictions are transformed using a linear mapping to any desired
evaluation taxonomy.
"""

UNRELABELED_TRAIN_DATASETS = [
    "ade20k-150",
    "bdd",
    "cityscapes-19",
    "coco-panoptic-133",
    "idd-39",
    "mapillary-public65",
    "sunrgbd-37",
]
# unrelabeled "cityscapes-34" is not used for training, but it could be, if desired.

RELABELED_TRAIN_DATASETS = [
    "ade20k-150-relabeled",
    "bdd-relabeled",
    "cityscapes-19-relabeled",
    "cityscapes-34-relabeled",
    "coco-panoptic-133-relabeled",
    "idd-39-relabeled",
    "mapillary-public65-relabeled",
    "sunrgbd-37-relabeled",
]
DEFAULT_TRAIN_DATASETS = UNRELABELED_TRAIN_DATASETS + RELABELED_TRAIN_DATASETS
# fmt: off
TEST_DATASETS = [
    "camvid-11",
    "kitti-19",
    "pascal-context-60",
    "scannet-20",
    "voc2012",
    "wilddash-19"
]
# fmt: on

_ROOT = Path(__file__).resolve().parent.parent


class TaxonomyConverter:
    """
    We use 1x1 convolution for our linear mapping from universal->test_taxonomy.
    Private methods marked with leading underscore.
    """

    def __init__(
        self,
        train_datasets: List[str] = DEFAULT_TRAIN_DATASETS,
        test_datasets: List[str] = TEST_DATASETS,
        tsv_fpath: str = f"{_ROOT}/class_remapping_files/MSeg_master.tsv",
    ) -> None:
        """

        Note about pandas `read_csv` parameter `keep_default_na`:
            Whether or not to include the default NaN values when parsing the data
            If keep_default_na is False, and na_values are not specified, no strings
            will be parsed as NaN.

        Args:
            train_datasets: list of training datasets
            test_datasets: list of test datasets
            tsv_fpath: absolute path to a tsv file showing the mapping between
                training datasets to universal taxonomy, and universal taxonomy
                to testing datasets.
        """
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets

        self.ignore_label = 255
        self.tsv_data = pd.read_csv(tsv_fpath, sep="\t", keep_default_na=False)
        self.softmax = nn.Softmax(dim=1)

        # Find universal name from spreadsheet TSV index.
        self.uid2uname = {}
        # Inverse -- find spreadsheet TSV index from universal name.
        self.uname2uid = {}
        self._build_universal_tax()
        self.num_uclasses = len(self.uid2uname) - 1  # excluding ignored label（id=255)
        # 255 is a privileged class index and must not be used elsewhere
        assert self.num_uclasses < 256

        self.dataset_classnames = {d: load_class_names(d) for d in (self.train_datasets + self.test_datasets)}

        self.id_to_uid_maps = {}
        self.convs = {}
        self.label_mapping_arr_dict = {}
        self._init_mappings()

    def _init_mappings(self) -> None:
        """Populate the train->universal, and universal->test mappings."""
        # Map train_dataset_id -> universal_id
        for d in self.train_datasets:
            print(f"\tMapping {d} -> universal")
            self.id_to_uid_maps[d] = self._transform_d2u(d)

        self.label_mapping_arr_dict = self._form_label_mapping_arrs()
        print(f"\n\tCreating 1x1 conv for test datasets...")
        self.convs = {dname: self._get_convolution_test(dname) for dname in self.test_datasets}

    def _build_universal_tax(self) -> None:
        """Build a flat label space by adding each `universal` taxonomy entry from the TSV here.

        We make a mapping from universal_name->universal_id, and vice versa.
        """
        for id, row in self.tsv_data.iterrows():
            u_name = parse_uentry(row["universal"])
            assert u_name not in self.uname2uid.keys()  # no duplicate names
            if u_name == "unlabeled":
                id = self.ignore_label

            self.uid2uname[id] = u_name
            self.uname2uid[u_name] = id

    def _transform_d2u(self, dataset: str) -> Mapping[int, int]:
        """Transform a training dataset to the universal taxonomy.

        For one training dataset, map each of its class ids to one universal ids.

        Args:
            dataset: string representing name of dataset

        Returns:
            id2uid_map: provide mapping from training dataset id to universal id.
        """
        id2uid_map = {}

        for index, row in self.tsv_data.iterrows():
            classes = parse_entry(row[dataset])
            u_name = parse_uentry(row["universal"])
            uid = self.uname2uid[u_name]
            if len(classes) == 0:
                continue
            for cls in classes:
                id = self.dataset_classnames[dataset].index(cls)
                id2uid_map[id] = uid

        # unlabeled should remain unlabeled.
        id2uid_map[self.ignore_label] = self.ignore_label
        return id2uid_map

    def _form_label_mapping_arrs(self) -> Mapping[str, np.ndarray]:
        """Cache conversion maps so we will later be able to perform fast grayscale->grayscale mapping
        for training data transformation.

        Each map is implemented as an integer array (given integer index, give back integer value stored at that location).

        Returns:
            label_mapping_arr_dict: map dataset names to numpy arrays.
        """
        label_mapping_arr_dict = {}
        from mseg.utils.mask_utils import form_label_mapping_array_pytorch

        for dataset in self.train_datasets:
            label_mapping_arr_dict[dataset] = form_label_mapping_array_pytorch(self.id_to_uid_maps[dataset])

        return label_mapping_arr_dict

    def _transform_u2d(self, dataset):
        """Store correspondences between universal_taxonomy and test_dataset_taxonomy.

        Note: Explicitly for inference on our test datasets.

        Args:
            dataset: string representing dataset's name

        Returns:
            uid2testid: list of tuples (i,j) that form linear mapping P from universal -> test_dataset.
        """
        uid2testid = []

        for index, row in self.tsv_data.iterrows():
            dname = parse_test_entry(row[dataset])
            if dname == "":
                # blank, so ignore
                continue
            test_id = self.dataset_classnames[dataset].index(dname)
            u_name = parse_uentry(row["universal"])
            u_id = self.uname2uid[u_name]
            uid2testid += [(u_id, test_id)]

        return uid2testid

    def _get_convolution_test(self, dataset: str) -> nn.Module:
        """Explicitly for inference on our test datasets.

        We implement this remapping from mu classes to mt classes for the evaluation dataset as a linear mapping P.

        The matrix weights Pij are binary 0/1 values and are fixed before training or evaluation; the weights are determined
        manually by inspecting label maps of the test datasets. Pij is set to 1 if unified taxonomy class j contributes to
        evaluation dataset class i, otherwise Pij = 0.

        Args:
            dataset: string representing dataset's name

        Returns:
            conv: nn.Conv2d module representing linear mapping from universal_taxonomy to test_dataset_taxonomy.
        """
        assert dataset in self.test_datasets
        uid2testid = self._transform_u2d(dataset)
        in_channel = self.num_uclasses
        out_channel = len(self.dataset_classnames[dataset])

        # Create 1x1 convolution filters.
        conv = populate_linear_mapping(in_channel, out_channel, uid2testid)
        assert isinstance(conv, nn.Module)
        return conv

    def transform_label(self, label: torch.Tensor, dataset: str) -> Mapping[int, torch.Tensor]:
        """Perform fast grayscale->grayscale mapping for training data transformation.

        Note: Function to be called externally for training.

        Args:
            label: Pytorch tensor on the cpu of shape (H,W) with dtype Torch.LongTensor,
                representing a semantic image, according to PNG/dataloader format.
            dataset: string representing dataset's name

        Returns:
            label: tensor also of shape (H,W), representing semantic classes in new taxonomy at each pixel
        """
        from mseg.utils.mask_utils import map_semantic_img_fast_pytorch

        label = map_semantic_img_fast_pytorch(label, self.label_mapping_arr_dict[dataset])
        return label

    def transform_predictions_test(self, input: torch.Tensor, dataset: str):
        """Transform predictions from the universal taxonomy to another taxonomy, at test time.

        Note: Explicitly for inference on our test datasets. Function to be called externally.
        Suppose universal taxonomy has N_u classes, and test taxonomy has N_t classes.

        Args:
            input: Pytorch tensor of shape (N_u,C,H,W) before softmax, in universal taxonomy.
            dataset: string representing dataset's name

        Returns:
            output: Pytorch tensor of shape (N_t,C,H,W)
        """
        input = self.softmax(input)
        output = self.convs[dataset](input)

        # if 'background' in self.dataset_names[dataset]: # background is 1 minus all other probabilitys, not summing all other nodes
        # 	background_id = self.dataset_names[dataset].index('background')
        # 	all_other = torch.cat([output[:, :background_id, :, :], output[:, (background_id+1):, :, :]], dim=1)
        # 	output[:, background_id, :, :] = 1 - torch.sum(all_other, dim=1)

        return output

    def transform_predictions_universal(self, input: torch.Tensor, dataset: str):
        """Transform predictions from the universal taxonomy to the universal taxonomy, at test time.

        Note: essentially a no-op, other than the softmax. Function to be called externally.

        Args:
            input: after softmax probability, of universal
            dataset: string representing dataset's name

        Returns:
            output:
        """
        output = self.softmax(input)
        return output


def parse_uentry(uentry: str) -> Tuple[int, List[str], str]:
    """Parse a universal taxonomy entry from the Master.tsv sheet.

    Note: universal taxonomy entries cannot be a blank string.

    Args:
        uentry: string, representing TSV entry from `Universal` taxonomy column

    Returns:
        full_name: exact duplicate of uentry string input
    """
    full_name = uentry.strip()
    assert full_name != ""
    return full_name


def parse_test_entry(test_entry) -> Tuple[str, str]:
    """Parse a test taxonomy entry from the Master.tsv sheet.

    Note: Can be a blank string, e.g. ''. We ensure no additional whitespace present.

    Args:
        test_entry

    Returns:
        test_entry
    """
    assert test_entry.strip() == test_entry  # don't want entry to contain or be space/tab
    return test_entry


def parse_entry(entry: str) -> List[str]:
    """Parse a spreadsheet entry from dataset's column, return list of classes in this spreadsheet cell.

    TODO: handle case for last column in spreadsheet, when contains carriage return

    Args:
        entry: string, representing cell in taxonomy spreadsheet

    Returns:
        classes: list of strings, representing class names in a non-universal dataset taxonomy
    """
    assert entry.strip() == entry, print(entry + "is not stripped")  # don't want entry to contain or be space/tab
    if entry == "":
        classes = []

    elif entry.startswith("{"):
        # take [1:-1] to discard `{`  `}` chars
        classes = entry[1:-1].split(",")
        classes = [c.strip() for c in classes]
    else:
        classes = [entry]

    return classes


def populate_linear_mapping(in_channel: int, out_channel: int, inid2outid: List[Tuple[int, int]]) -> nn.Module:
    """Use 1x1 convolution to create linear mapping P of each pixel's probabilities to a new space.

    The matrix weights Pij are binary 0/1 values and are fixed before training or evaluation; the weights are determined
    manually by inspecting label maps of the test datasets. Pij is set to 1 if input class j contributes to
    output class i, otherwise Pij = 0.

    Args:
        in_channel: number of input channels
        out_channel: number of output channels
        inid2outid: list of (j,i) tuples defining linear mapping P

    Returns:
        conv: 1x1 convolutional kernels. padding is zero by default.
    """
    conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
    conv.weight.data.fill_(0)

    for param in conv.parameters():
        param.requires_grad = False

    for (j, i) in inid2outid:
        conv.weight[i][j] = 1
    return conv
