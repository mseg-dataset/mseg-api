#!/usr/bin/python3

from pathlib import Path

import mseg.utils.tsv_utils as tsv_utils

MSEG_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_read_tsv_column_vals() -> None:
    """ """
    tsv_fpath = f"{MSEG_REPO_ROOT}/mseg/class_remapping_files/MSeg_master.tsv"
    dname = "universal"
    tsv_classnames = tsv_utils.read_tsv_column_vals(
        tsv_fpath, col_name=dname, convert_val_to_int=False
    )

    assert isinstance(tsv_classnames, list)
    assert all([isinstance(classname, str) for classname in tsv_classnames])
