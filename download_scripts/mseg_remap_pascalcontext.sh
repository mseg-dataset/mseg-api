#!/bin/sh

# This script downloads the PASCAL Context dataset.

# By using this script, you agree to any terms or 
# conditions set forth by the PASCAL VOC creators
# or the PASCAL Context dataset creators.

PCONTEXT_DST_DIR=$1
NUM_CORES_TO_USE=$2

# ----------- Environment Variables + Directory Setup -----------------------

# navigate to repo root
cd ..
# gets the current working dir (for repo root)
REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "Repo is stored at "$REPO_ROOT


# # ----------- Remapping -----------------------

# now, prepare the labels on disk

ORIG_DNAME="pascal-context-460"
REMAP_DNAME="pascal-context-60"
ORIG_DROOT=$PCONTEXT_DST_DIR

echo "Original, unzipped dataset is stored at "$ORIG_DROOT
echo "Remapped dataset will be stored at "$REMAP_DROOT

now=$(date +"%Y %m %d @ %H:%M:%S")
echo "Start remapping at "$now

# Do not include ignore_idx_class, since this is a defined class
# in PASCAL-Context-460
python -u $REPO_ROOT/mseg/label_preparation/remap_dataset.py \
	--orig_dname $ORIG_DNAME --remapped_dname $REMAP_DNAME \
	--orig_dataroot $ORIG_DROOT --remapped_dataroot $ORIG_DROOT \
	--num_processes $NUM_CORES_TO_USE --include_ignore_idx_cls

now=$(date +"%Y %m %d @ %H:%M:%S")
echo "Completed remapping at "$now
