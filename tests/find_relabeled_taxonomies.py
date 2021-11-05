import pandas as pd
from pathlib import Path
import pdb
from typing import List, Tuple

from mseg.utils.dir_utils import check_mkdir
from mseg.utils.csv_utils import write_csv
from mseg.label_preparation.dataset_update_records import (
    cocop_update_records,
    ade20k_update_records,
    bdd_update_records,
    idd_update_records,
    cityscapes_update_records,
    sunrgbd_update_records,
    mapillary_update_records,
)
import mseg.taxonomy.taxonomy_converter as taxonomy_converter

tsv_fpath = "/Users/johnlamb/Downloads/Zhuang_John_Ozan_Vladlen_Taxonomy_4.0.tsv"

REPO_ROOT = Path(__file__).resolve().parent.parent



def get_new_classes(update_records):
    """ """
    relabeled_classes = []
    for rec in update_records:
        relabeled_classes += [rec.relabeled_class]
    return set(relabeled_classes)


def find_relabeled_taxonomy(dname: str, update_records):
    """
    For any given dataset, compare relabeled classes (in universal taxonomy) with
    universal classes that correspond with original classes.

    We populate 3 separate spreadsheets:
    (1) map from [original->relabeled] taxonomies
            `remap_rows` is only for
    (2) _names.txt file for relabeled taxonomy
            `new_tax_rows`
    (3) column of master Google spreadsheet with correspondence to all universal classes
            should contain everything from (2), and more (i.e. blank entries for complement)
            `new_googlesheet_rows`
    """
    tsv_data = pd.read_csv(tsv_fpath, sep="\t", keep_default_na=False)

    remap_rows = []
    new_googlesheet_rows = []
    all_u_classes = []
    featured_u_classes = []

    relabeled_classes = get_new_classes(update_records)

    for id, row in tsv_data.iterrows():
        u_name = taxonomy_converter.parse_uentry(row["universal"])
        all_u_classes += [u_name]
        # specific dataset's classes that map to this u_name
        d_classes = taxonomy_converter.parse_entry(row[dname])
        if len(d_classes) != 0:
            # pre-existing corresponding labels, before re-labeling
            featured_u_classes += [u_name]
        # if pre-existing correspondence, or new correspondence will exist
        if len(d_classes) != 0 or u_name in relabeled_classes:
            for d_class in d_classes:
                remap_rows += [{dname: d_class, f"{dname}-relabeled": u_name}]
            new_googlesheet_rows += [
                {f"{dname}-relabeled": u_name, "universal": u_name}
            ]
        else:
            # leave blank, will be no such u_name label
            new_googlesheet_rows += [{f"{dname}-relabeled": "", "universal": u_name}]

    # ensure no typos in update records
    assert all(
        [relabeled_class in all_u_classes for relabeled_class in relabeled_classes]
    )

    new_classes = relabeled_classes - set(featured_u_classes)
    print(f"To {dname}, we added {new_classes}")

    new_taxonomy = relabeled_classes | set(featured_u_classes)
    new_tax_rows = [
        {f"{dname}-relabeled": new_tax_class} for new_tax_class in new_taxonomy
    ]
    save_new_taxonomy_csv = f"names/{dname}-relabeled_names.tsv"
    check_mkdir("names")
    write_csv(save_new_taxonomy_csv, new_tax_rows)

    remap_rows += [{dname: "unlabeled", f"{dname}-relabeled": "unlabeled"}]
    save_remap_csv = (
        f"{REPO_ROOT}/mseg/class_remapping_files/{dname}_to_{dname}-relabeled.tsv"
    )
    write_csv(save_remap_csv, remap_rows)

    new_googlesheet_csv = f"{REPO_ROOT}/{dname}_to_relabeled_universal.tsv"
    write_csv(new_googlesheet_csv, new_googlesheet_rows)


def main() -> None:
    """
    Given the full tsv with all dataset->universal mappings, and given relabeled update
    records, we form new "relabeled taxonomies" for each dataset, and a new mapping.
    """
    dataset_dict = {
        "coco-panoptic-133": cocop_update_records,
        "ade20k-150": ade20k_update_records,
        "bdd": bdd_update_records,
        "idd-39": idd_update_records,
        "cityscapes-19": cityscapes_update_records,
        "cityscapes-34": cityscapes_update_records,  # more complete than 19-class version
        "sunrgbd-37": sunrgbd_update_records,
        "mapillary-public65": mapillary_update_records,
    }

    for dname, update_records in dataset_dict.items():
        find_relabeled_taxonomy(dname, update_records)


if __name__ == "__main__":
    main()
