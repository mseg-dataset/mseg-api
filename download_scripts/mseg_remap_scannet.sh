#!/bin/sh

# Remaps the ScanNet dataset labels on disk.

SCANNET_DST_DIR=$1
NUM_CORES_TO_USE=$2


# ---------- Environment Variables / Directory Setup -------------------------

# navigate to repo root
cd ..
# gets the current working dir (for repo root)
REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "Repo is stored at "$REPO_ROOT

# --------- Remapping ------------------------------------------------------
cd $REPO_ROOT

ORIG_DNAME="scannet-41"
REMAP_DNAME="scannet-20"
ORIG_DROOT=$SCANNET_DST_DIR/scannet_frames_25k

echo "Original, unzipped dataset is stored at "$ORIG_DROOT

now=$(date +"%Y %m %d @ %H:%M:%S")
echo "Start remapping at "$now

# Will set 'convert_label_from_rgb' to True
python -u $REPO_ROOT/mseg/label_preparation/remap_dataset.py \
	--orig_dname $ORIG_DNAME --remapped_dname $REMAP_DNAME \
	--orig_dataroot $ORIG_DROOT --remapped_dataroot $ORIG_DROOT \
	--num_processes $NUM_CORES_TO_USE

now=$(date +"%Y %m %d @ %H:%M:%S")
echo "Completed remapping at "$now

