#!/bin/sh

# Remaps the IDD dataset to a common label space.

IDD_DST_DIR=$1
NUM_CORES_TO_USE=$2

# ------- Directory Setup --------------------------------------------
# navigate to repo root
cd ..
# gets the current working dir (for repo root)
REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "Repo is stored at "$REPO_ROOT

# ------- Json to PNG conversion --------------------------------------------
# First, dump the json to .png files
python $REPO_ROOT/mseg/label_preparation/dump_idd_semantic_labels.py \
	--num-workers $NUM_CORES_TO_USE --datadir $IDD_DST_DIR/IDD_Segmentation

# # ----------- Remapping ---------------------------------------------------
now=$(date +"%Y %m %d @ %H:%M:%S")
echo "Started remapping at "$now

ORIG_DNAME="idd-40"
REMAP_DNAME="idd-39"

ORIG_DROOT=$IDD_DST_DIR/IDD_Segmentation

# Will set 'convert_label_from_rgb' to True
python -u $REPO_ROOT/mseg/label_preparation/remap_dataset.py \
	--orig_dname $ORIG_DNAME --remapped_dname $REMAP_DNAME \
	--orig_dataroot $ORIG_DROOT --remapped_dataroot $ORIG_DROOT \
	--num_processes $NUM_CORES_TO_USE

now=$(date +"%Y %m %d @ %H:%M:%S")
echo "Completed remapping at "$now
