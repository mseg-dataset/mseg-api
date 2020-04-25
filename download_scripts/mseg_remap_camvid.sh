#!/bin/sh

# Remaps the Camvid dataset to a common label space.

CAMVID_DST_DIR=$1
NUM_CORES_TO_USE=$2

# navigate to repo root
cd ..
# gets the current working dir (for repo root)
REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "Repo is stored at "$REPO_ROOT

# # ----------- Remapping ---------------------------------------------------
now=$(date +"%Y %m %d @ %H:%M:%S")
echo "Started remapping at "$now

ORIG_DNAME="camvid-32"
REMAP_DNAME="camvid-11"

ORIG_DROOT=$CAMVID_DST_DIR

# Will set 'convert_label_from_rgb' to True
python -u $REPO_ROOT/mseg/label_preparation/remap_dataset.py \
	--orig_dname $ORIG_DNAME --remapped_dname $REMAP_DNAME \
	--orig_dataroot $ORIG_DROOT --remapped_dataroot $ORIG_DROOT \
	--convert_label_from_rgb --num_processes $NUM_CORES_TO_USE

now=$(date +"%Y %m %d @ %H:%M:%S")
echo "Completed remapping at "$now



