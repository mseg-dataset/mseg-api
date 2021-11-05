#!/usr/bin/python3

import copy
import csv
import collections
from typing import Any, List, Mapping


def read_csv(csv_fpath: str, delimiter="\t") -> List[collections.OrderedDict]:
    """Copy the data out of a csv file, as a list of OrderedDicts.

    Args:
        csv_fpath: string representing path to a csv file.

    Rows:
        rows: list of OrderedDicts. key is column name, value is entry at
            (row,column) of csv file.
    """
    rows = []
    with open(csv_fpath, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=delimiter)
        for row in csv_reader:
            rows += [copy.deepcopy(row)]
    return rows


def write_csv(csv_fpath: str, dict_list: List[Mapping[Any, Any]], delimiter="\t") -> None:
    """Write rows to CSV, with column names populated from dictionary keys."""
    with open(csv_fpath, "w", newline="") as csvfile:

        fieldnames = dict_list[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for row_dict in dict_list:
            writer.writerow(row_dict)
