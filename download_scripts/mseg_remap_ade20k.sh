#!/bin/sh

# Remaps the ADE20K dataset labels on disk.

ADE20K_DST_DIR=$1
# Select default (>200 is quite helpful).
NUM_CORES_TO_USE=$2

# ---------- Environment Variables / Directory Setup -------------------------
# navigate to repo root
cd ..
# gets the current working dir (for repo root)
REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "Repo is stored at "$REPO_ROOT
# --------- Remapping ------------------------------------------------------

# now, prepare the labels on disk
cd $REPO_ROOT

ORIG_DNAME="ade20k-151"
REMAP_DNAME="ade20k-150"
ORIG_DROOT=$ADE20K_DST_DIR/ADEChallengeData2016

echo "Original, unzipped dataset is stored at "$ORIG_DROOT
echo "Remapped dataset will be stored at "$REMAP_DROOT

now=$(date +"%Y %m %d @ %H:%M:%S")
echo "Start remapping at "$now

python -u $REPO_ROOT/mseg/label_preparation/remap_dataset.py \
	--orig_dname $ORIG_DNAME --remapped_dname $REMAP_DNAME \
	--orig_dataroot $ORIG_DROOT --remapped_dataroot $ORIG_DROOT \
	--num_processes $NUM_CORES_TO_USE

now=$(date +"%Y %m %d @ %H:%M:%S")
echo "Completed remapping at "$now

