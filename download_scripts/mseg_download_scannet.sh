#!/bin/sh

# Downloads the ScanNet dataset. Should take 1-2 hours.

# By using this script, you agree to all terms and conditions
# of the ScanNet dataset and certify you legally
# obtained a license after submitting `ScanNet_TOS.pdf` to the
# authors.

# --------- Populate Var ----------------------------------------

# Copy 'BASE_URL' variable from `download-scannet.py` script
# A url to this script will be sent to you via email after registration.
BASE_URL=""

# --------- Directory Setup------------------------------------

SCANNET_DST_DIR=$1
mkdir -p $SCANNET_DST_DIR

# --------- Downloading -----------------------------------------
echo "Downloading ScanNet dataset..."

cd $SCANNET_DST_DIR

SCANNET_ZIP_URL=$BASE_URL"v2/tasks/scannet_frames_25k.zip"
wget $SCANNET_ZIP_URL

echo "ScanNet dataset downloaded."

# --------- Extraction ------------------------------------------
echo "Extracting ScanNet dataset..."
unzip scannet_frames_25k.zip
echo "ScanNet dataset extracted."
