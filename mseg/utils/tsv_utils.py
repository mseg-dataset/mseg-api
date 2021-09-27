#!/usr/bin/python3

import csv
import os
from typing import List, Mapping, Union

"""
Set of utilities for working with tab-separated values (tsv) files.
The tsv format is a convenient way to store a mapping, e.g. a 
mapping between taxonomies.
"""


def download_tsv(tsv_fpath, spreadsheet_url) -> None:
    """Use a system call to download a spreadsheet from Google Drive as tab-separated values.

    Args:
        tsv_fpath:
        spreadsheet_url:
    """
    os.system(f"""wget --no-check-certificate -O {tsv_fpath} {spreadsheet_url}""")


def represents_int(s: str) -> bool:
    """Checks if a string s represents an int.

    Args:
        s: string

    Returns:
        boolean, whether string represents an integer value
    """
    try:
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping(
    filename: str, label_from: str = "raw_category", label_to: str = "nyu40id", convert_val_to_int: bool = False
) -> Mapping[int, str]:
    """
    Args:
        filename: string representing
        label_from: string representing "source" taxonomy name
        label_to: string representing "sink"/ "target" taxonomy name
        convert_val_to_int:

    Returns:
        mapping: dictionary with (id,class name) key-value pairs from source->sink
    """
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            source_val = row[label_from]
            sink_val = row[label_to]
            mapping[source_val] = sink_val
    # if ints convert
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}

    if convert_val_to_int:
        mapping = {k: int(v) for k, v in mapping.items()}

    return mapping


def read_tsv_column_vals(tsv_fpath: str, col_name: str, convert_val_to_int: bool = False) -> List[Union[str, int]]:
    """Read out ordered values from a specific column of a tsv file (spreadsheet).

    Args:
        tsv_fpath: path to a .tsv file
        col_name: column name in the spreadsheet
        convert_val_to_int: whether to cast all entries in the desired column from strings to integers.

    Returns:
        col_vals: list containing ordered values from desired column.
    """
    assert os.path.isfile(tsv_fpath)

    col_vals = list()
    with open(tsv_fpath) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            col_vals += [row[col_name]]

    if convert_val_to_int:
        col_vals = [int(col_val) for col_val in col_vals]

    return col_vals
