#!/usr/bin/python3

from typing import List, Mapping, Optional, Tuple

import torch.nn as nn

import mseg.utils.names_utils as names_utils
from mseg.taxonomy.taxonomy_converter import TaxonomyConverter, UNRELABELED_TRAIN_DATASETS, TEST_DATASETS


class NaiveTaxonomyConverter(TaxonomyConverter):
    """
    Machine-generated taxonomy, generated without domain knowledge.
    Inherits from the `TaxonomyConverter` class, and re-uses most
    functionality. Input should be unrelabeled train datasets.

    As in `TaxonomyConverter`, we use 1x1 convolution for our linear
    mapping from universal->test_taxonomy. Private methods marked with
    leading underscore.
    """

    def __init__(
        self, train_datasets: List[str] = UNRELABELED_TRAIN_DATASETS, test_datasets: List[str] = TEST_DATASETS
    ) -> None:
        """
        Args:
            train_datasets: list of training datasets
            test_datasets: list of test datasets
        """
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets

        self.ignore_label = 255
        self.softmax = nn.Softmax(dim=1)

        # Find universal name from universal index.
        self.uid2uname = {}
        # Inverse -- find universal index from universal name.
        self.uname2uid = {}

        self.dataset_classnames = {
            d: names_utils.load_class_names(d) for d in (self.train_datasets + self.test_datasets)
        }
        self._build_universal_tax()

        # including ignored label（id=255), since total id > 255. (note previously it's -1)
        self.num_uclasses = len(self.uid2uname) + 1
        # 255 is a privileged class index and must not be used elsewhere

        self.id_to_uid_maps = {}
        self.convs = {}
        self.label_mapping_arr_dict = {}
        self._init_mappings()

    def _build_universal_tax(self) -> None:
        """
        Build a flat label space by creating the `universal` taxonomy here.
        Instead of using entries from a mapping TSV, we just count the universal
        taxonomy names as lowercase versions of all training dataset classnames,
        e.g. so that `Bird` and `bird` are combined into the same class.

        We make a mapping from universal_name->universal_id, and vice versa.
        """
        id = 0
        for d in self.train_datasets:
            classes = self.dataset_classnames[d]
            for c in classes:
                lowercase = c.lower()

                if lowercase in self.uname2uid.keys():
                    # print(f'class {c} already in, lowercase is {lowercase}')
                    continue

                # otherwise, `lowercase` is not found in `self.uname2uid` yet, so we'll make a new entry.
                self.uname2uid[lowercase] = id
                self.uid2uname[id] = lowercase
                id += 1
                # if incremented id is 255, then skip it
                if id == self.ignore_label:
                    # skip 255, since want to reserve 255 as ignore_label
                    id += 1

    def get_naive_taxonomy_classnames(self) -> List[Optional[str]]:
        """Get a list of ordered classes in the naive taxonomy.

        The logits of a model's predictions at each pixel should also be ordered accordingly.

        Returns:
            classnames: order list of classnames in the taxonomy. The ignore index is filled with `None`.
        """
        num_classes = max(self.uid2uname.keys()) + 1  # can't use len, since `255' is a missing key.
        classnames = [None] * num_classes

        for uid, uname in self.uid2uname.items():
            classnames[uid] = uname

        return classnames

    def _transform_d2u(self, dataset: str) -> Mapping[int, int]:
        """Transform a training dataset to the universal taxonomy.
        For one training dataset, map each of its class ids to one universal ids.

        In the naive approach, the universal classname is simply the lowercase
        version of the original classname.

        Args:
            dataset: string representing name of dataset

        Returns:
            id2uid_map: provide mapping from training dataset id to universal id.
        """
        id2uid_map = {}
        classes = self.dataset_classnames[dataset]
        for i, c in enumerate(classes):
            lowercase = c.lower()
            id2uid_map[i] = self.uname2uid[lowercase]

        id2uid_map[self.ignore_label] = self.ignore_label
        return id2uid_map

    def _transform_u2d(self, dataset: str) -> List[Tuple[int, int]]:
        """Explicitly for inference on our test datasets. Store correspondences
        between universal_taxonomy and test_dataset_taxonomy.
        Lowercase string would be the universal taxonomy classname, if exists.

        Args:
            dataset: string representing dataset's name

        Returns:
            uid2testid: list of tuples (i,j) that form linear mapping P from universal -> test_dataset.
        """
        uid2testid = []

        for test_id, cls in enumerate(self.dataset_classnames[dataset]):
            lowercase = cls.lower()
            # some test classnames might not appear in universal taxonomy,
            # nothing that can be done about performance hit in naive case.
            if lowercase in self.uname2uid:
                u_id = self.uname2uid[lowercase]
                uid2testid += [(u_id, test_id)]

        return uid2testid
