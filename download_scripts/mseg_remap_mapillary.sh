#!/bin/sh

# Remaps the Mapillary Academic Version dataset on disk
# to a common label space.

# Destination directory for Mapillary
MAPILLARY_DST_DIR=$1
# Select default (>200 is quite helpful).
NUM_CORES_TO_USE=$2

# --------Directory Setup ---------------------------------------
# navigate to repo root
cd ..
# gets the current working dir (for repo root)
REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "Repo is stored at "$REPO_ROOT

# -------- Remapping -------------------------------------------

# now, prepare the labels on disk
cd $REPO_ROOT

ORIG_DNAME="mapillary-public66"
REMAP_DNAME="mapillary-public65"
ORIG_DROOT=$MAPILLARY_DST_DIR

echo "Original, unzipped dataset is stored at "$ORIG_DROOT


now=$(date +"%Y %m %d @ %H:%M:%S")
echo "Start remapping at "$now

# Will set 'convert_label_from_rgb' to True
python -u $REPO_ROOT/mseg/label_preparation/remap_dataset.py \
	--orig_dname $ORIG_DNAME --remapped_dname $REMAP_DNAME \
	--orig_dataroot $ORIG_DROOT --remapped_dataroot $ORIG_DROOT \
	--convert_label_from_rgb --num_processes $NUM_CORES_TO_USE

now=$(date +"%Y %m %d @ %H:%M:%S")
echo "Completed remapping at "$now
