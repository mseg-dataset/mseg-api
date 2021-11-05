#!/usr/bin/python3

import imageio
import math
from pathlib import Path
import pdb

from mseg.utils.cv2_utils import grayscale_to_color, cv2_imread_rgb
from mseg.utils.dataset_config import infos
from mseg.utils.names_utils import (
    get_classname_to_dataloaderid_map,
    get_dataloader_id_to_classname_map,
)
from mseg.utils.mask_utils import (
    form_mask_triple_embedded_classnames,
    save_classnames_in_image_maxcardinality,
)
from mseg.utils.mask_utils_detectron2 import Visualizer
from mseg.utils.txt_utils import generate_all_img_label_pair_fpaths
from mseg.label_preparation.mseg_write_relabeled_segments import (
    get_unique_mask_identifiers,
)


_ROOT = Path(__file__).resolve().parent.parent


def verify_all_dataset_paths_exist():
    """
    Loop through all of the datasets and ensure that the absolute paths exist
    and are valid.
    """
    for dname, d_info in infos.items():
        print(f"Verifying {dname}...")

        if dname in ["ade20k-151-inst"]:
            # no such explicit list
            continue

        for split_list in [d_info.trainlist, d_info.vallist]:
            pairs = generate_all_img_label_pair_fpaths(d_info.dataroot, split_list)
            for (rgb_fpath, label_fpath) in pairs:
                assert Path(rgb_fpath).exists()
                assert Path(label_fpath).exists()
        print(f"Verified {dname}.")


def visual_sanitychecks():
    """
    Save every 1000th image of each dataset, with classnames embedded.
    """
    save_dir = "temp_files/MSeg_verify"

    for dname, d_info in infos.items():
        print(f"Writing visual sanity checks for {dname}...")

        if dname in ["coco-panoptic-inst-201", "mapillary-public66", "ade20k-151-inst"]:
            continue  # is in RGB format and not comparable

        id_to_classname_map = get_dataloader_id_to_classname_map(dname)
        splits = ["train", "val"]
        split_lists = [d_info.trainlist, d_info.vallist]

        for split, split_list in zip(splits, split_lists):
            pairs = generate_all_img_label_pair_fpaths(d_info.dataroot, split_list)

            # Save 5 examples from each dataset split
            step_sz = math.floor(len(pairs) // 5)
            for i, (rgb_fpath, label_fpath) in enumerate(pairs[::step_sz]):
                print(f"On {i} of {dname}")

                rgb_img = cv2_imread_rgb(rgb_fpath)
                label_img = imageio.imread(label_fpath)

                fname_stem = Path(rgb_fpath).stem
                save_fpath = f"{save_dir}/{dname}_{fname_stem}.jpg"
                blend_save_fpath = f"{save_dir}/{dname}_{fname_stem}_blended.jpg"

                if rgb_img.ndim == 2:
                    # this image was grayscale
                    rgb_img = grayscale_to_color(rgb_img)

                form_mask_triple_embedded_classnames(
                    rgb_img,
                    label_img,
                    id_to_classname_map,
                    save_fpath,
                    save_to_disk=True,
                )
                frame_visualizer = Visualizer(rgb_img, metadata=None)
                output_img = frame_visualizer.overlay_instances(
                    label_map=label_img, id_to_class_name_map=id_to_classname_map
                )
                imageio.imwrite(blend_save_fpath, output_img)


class SanityCheckDataset:
    def __init__(self, dataroot: str, dname: str) -> None:
        """ """
        self.dataroot = dataroot
        self.dname = dname
        self.id_to_classname_map = get_dataloader_id_to_classname_map(dname)
        self.get_classname_to_id_map = get_classname_to_dataloaderid_map(
            dataset_name=dname
        )

    def find_matches(self, examples):
        """ """
        print(f"Finding matches for {self.dname}...")
        save_dir = f"temp_files/target_visual_check_{self.dname}"
        for split in ["train", "val"]:
            split_txt_fpath = (
                f"{_ROOT}/mseg/dataset_lists/{self.dname}/list/{split}.txt"
            )
            pairs = generate_all_img_label_pair_fpaths(self.dataroot, split_txt_fpath)

            for (title, annot_fname, data_split, coords) in examples:
                parent, fname_stem, _ = get_unique_mask_identifiers(
                    self.dname, annot_fname, data_split
                )
                for (rgb_fpath, label_fpath) in pairs:
                    if (
                        Path(rgb_fpath).stem == fname_stem
                        and Path(label_fpath).parts[-2] == parent
                    ):
                        y, x = coords
                        rgb_img = imageio.imread(rgb_fpath)
                        label_img = imageio.imread(label_fpath)
                        assert label_img[y, x] == self.get_classname_to_id_map[title]


def verify_targeted_visual_examples():
    """ """
    cityscapes_examples = [
        [
            "bicycle",
            "seqtubingen_tubingen_000060_000019_leftImg8bit_57.jpg",
            "train",
            (474, 760),
        ],
        [
            "bicyclist",
            "seqhamburg_hamburg_000000_068916_leftImg8bit_223.jpg",
            "train",
            (353, 1181),
        ],
        [
            "motorcyclist",
            "seqbremen_bremen_000052_000019_leftImg8bit_102.jpg",
            "train",
            (403, 1048),
        ],
        [
            "rider_other",
            "seqaachen_aachen_000014_000019_leftImg8bit_84.jpg",
            "train",
            (542, 531),
        ],
        [
            "person",  # this is `person_nonrider`
            "seqfrankfurt_frankfurt_000001_038844_leftImg8bit_82.jpg",
            "val",
            (341, 649),
        ],
    ]
    dname = "cityscapes-19-relabeled"

    scd = SanityCheckDataset(infos[dname].dataroot, dname)
    scd.find_matches(cityscapes_examples)

    dname = "ade20k-150-relabeled"
    ade20k_examples = [
        ["zebra", "ADE_train_00016125_113.png", "train", (504, 200)],
        ["nightstand", "ADE_train_00004036_119.png", "train", (195, 37)],
        ["wine_glass", "ADE_val_00001949_134.png", "val", (300, 478)],
        ["bicyclist", "ADE_train_00017384_120.png", "train", (398, 55)],
        ["bicyclist", "ADE_train_00017404_215.png", "train", (268, 373)],
        ["counter_other", "ADE_val_00001477_157.png", "val", (160, 234)],
        ["teddy_bear", "ADE_train_00001951_185.png", "train", (328, 508)],
    ]
    scd = SanityCheckDataset(infos[dname].dataroot, dname)
    scd.find_matches(ade20k_examples)

    dname = "bdd-relabeled"
    bdd_examples = [
        ["motorcyclist", "a91b7555-00000590_12.jpg", "val", (470, 888)],
        ["bicyclist", "3516379e-43f6a6ba_12.jpg", "train", (410, 654)],
        ["motorcyclist", "972ab49a-6a6eeaf5_11.jpg", "train", (396, 284)],
        # below is `person_nonrider`
        ["person", "8413f861-580b500d_12.jpg", "train", (416, 926)],
        # below is `person_nonrider`
        ["person", "a706da19-14468d02_11.jpg", "val", (300, 1248)],
        ["bicyclist", "1ff92f74-697a077e_11.jpg", "train", (270, 946)],
    ]
    scd = SanityCheckDataset(infos[dname].dataroot, dname)
    scd.find_matches(bdd_examples)

    dname = "coco-panoptic-133-relabeled"
    cocop_examples = [
        # title, fname, split
        # below is `wheelchair`
        ["slow_wheeled_object", "000000091349_3093585.png", "train", (245, 396)],
        ["seat", "000000016509_9079654.png", "train", (46, 133)],
        ["bathroom_counter", "000000376310_2631210.png", "val", (418, 558)],
        ["bicyclist", "000000067208_4211027.png", "train", (83, 404)],
        ["motorcyclist", "000000459634_6904408.png", "val", (299, 99)],
        ["kitchen_island", "000000438774_5403260.png", "val", (199, 341)],
    ]
    scd = SanityCheckDataset(infos[dname].dataroot, dname)
    scd.find_matches(cocop_examples)

    dname = "idd-39-relabeled"
    idd_examples = [
        ["backpack", "seq173_318463_leftImg8bit_50.jpg", "train", (912, 1191)],
        ["bicyclist", "seq158_867833_leftImg8bit_54.jpg", "train", (510, 453)],
        ["box", "seq173_316862_leftImg8bit_34.jpg", "train", (752, 761)],
        ["motorcyclist", "seq98_776293_leftImg8bit_110.jpg", "train", (502, 1101)],
        # below is `person_nonrider`
        ["person", "seq1_148425_leftImg8bit_110.jpg", "train", (458, 822)],
        # ['rider-other', 'seq48_601929_leftImg8bit_79.jpg', 'val', () ],
        ["bicyclist", "seq181_016419_leftImg8bit_41.jpg", "val", (365, 889)],
        ["rider_other", "seq148_000690_leftImg8bit_67.jpg", "val", (591, 1013)],
    ]
    scd = SanityCheckDataset(infos[dname].dataroot, dname)
    scd.find_matches(idd_examples)

    dname = "sunrgbd-37-relabeled"
    sunrgbd_examples = [
        ["swivel_chair", "img-000253_4.jpg", "train", (303, 463)],
        ["door", "img-004239_4.jpg", "train", (190, 30)],
        ["desk", "img-004930_11.jpg", "train", (429, 143)],
        ["bathroom_counter", "img-002347_11.jpg", "train", (254, 34)],
        ["bathroom_counter", "img-002164_11.jpg", "test", (350, 483)],
        ["sconce", "img-004159_34.jpg", "train", (204, 284)],
        ["chandelier", "img-001692_34.jpg", "train", (3, 459)],
        ["armchair", "img-003192_4.jpg", "test", (300, 467)],
    ]
    scd = SanityCheckDataset(infos[dname].dataroot, dname)
    scd.find_matches(sunrgbd_examples)

    dname = "mapillary-public65-relabeled"
    mapillary_examples = [
        ["sea", "mapillary_EZJTbIpLNOfHnw3DvA1NEA_7936.jpg", "val", (1921, 3000)],
        [
            "fountain",
            "mapillary_1fhI-qv6dVR-1Kc8Oafb3Q_7936.jpg",
            "train",
            (1059, 2663),
        ],
        ["horse", "mapillary_CH1TcYj0ki_kw3Yc3oOiZA_258.jpg", "train", (1724, 180)],
        ["dog", "mapillary_OkOJlU8l98KqPEE05dv1tA_256.jpg", "val", (1773, 886)],
    ]
    scd = SanityCheckDataset(infos[dname].dataroot, dname)
    scd.find_matches(mapillary_examples)


if __name__ == "__main__":
    verify_targeted_visual_examples()
    visual_sanitychecks()
    verify_all_dataset_paths_exist()
