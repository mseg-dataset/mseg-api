#!/bin/sh

# Remaps the COCO Panoptic dataset.

COCOP_DST_DIR=$1
NUM_CORES_TO_USE=$2

# ----------- Directory Setup -----------------------------
# navigate to repo root
cd ..
# gets the current working dir (for repo root)
REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "Repo is stored at "$REPO_ROOT

# ----------- Dump Semantic PNG Labels from JSON ----------

python -u $REPO_ROOT/mseg/label_preparation/dump_coco_semantic_labels.py \
	--num_processes $NUM_CORES_TO_USE

# ----------- Remapping -----------------------------------

now=$(date +"%Y %m %d @ %H:%M:%S")
echo "Started remapping at "$now

ORIG_DNAME="coco-panoptic-201"
REMAP_DNAME="coco-panoptic-133"

ORIG_DROOT=$COCOP_DST_DIR

python -u $REPO_ROOT/mseg/label_preparation/remap_dataset.py \
	--orig_dname $ORIG_DNAME --remapped_dname $REMAP_DNAME \
	--orig_dataroot $ORIG_DROOT --remapped_dataroot $ORIG_DROOT \
	--num_processes $NUM_CORES_TO_USE


now=$(date +"%Y %m %d @ %H:%M:%S")
echo "Completed remapping at "$now


