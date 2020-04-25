#!/bin/sh

# Downloads the Camvid dataset.

# By using this script, you agree to all terms
# and conditions specified by the Camvid
# dataset creators.

CAMVID_DST_DIR=$1

# ----------- Environment Variables + Directory Setup -----------------------
# Destination directory for MSeg

echo "Camvid will be downloaded to "$CAMVID_DST_DIR
mkdir -p $CAMVID_DST_DIR

# ----------- Downloading ---------------------------------------------------

CAMVID_IMGS_URL="http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip"
CAMVID_LABELS_URL="http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/LabeledApproved_full.zip"

echo "Downloading Camvid dataset..."
# Images are GB, Labels are 
wget -c -O $CAMVID_DST_DIR/701_StillsRaw_full.zip $CAMVID_IMGS_URL
wget -c -O $CAMVID_DST_DIR/LabeledApproved_full.zip $CAMVID_LABELS_URL
echo "Camvid dataset downloaded."

# # ----------- Extraction ---------------------------------------------------
echo "Extracting Camvid dataset..."
cd $CAMVID_DST_DIR

unzip 701_StillsRaw_full.zip
unzip -d Labels32-RGB LabeledApproved_full.zip

rm LabeledApproved_full.zip
rm 701_StillsRaw_full.zip
echo "Camvid dataset extracted."

