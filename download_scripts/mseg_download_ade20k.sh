#!/bin/sh

# Downloads the ADE20K dataset.

# By using this script, you agree to all terms and
# conditions set forth by the creators of the ADE20K
# dataset.

# ---------- Environment Variables / Directory Setup -------------------------

ADE20K_DST_DIR=$1
echo "ADE20K will be downloaded to "$ADE20K_DST_DIR
mkdir -p $ADE20K_DST_DIR

# --------- Downloading ------------------------------------------------------

ADE20K_INSTANCE_LABELS_URL="https://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2016_07_26.zip"
ADE20K_SEMANTIC_LABELS_URL="http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"

echo "Downloading ADE20K dataset..."
# instance labels zip will be 3.8 GB
wget -c --no-check-certificate -O $ADE20K_DST_DIR/ADE20K_2016_07_26.zip $ADE20K_INSTANCE_LABELS_URL
# semantic labels zip will be 923 MB
wget -c --no-check-certificate -O $ADE20K_DST_DIR/ADEChallengeData2016.zip $ADE20K_SEMANTIC_LABELS_URL

echo "ADE20K dataset downloaded."

# --------- Extraction ------------------------------------------------------

echo "Extracting ADE20K dataset..."
cd $ADE20K_DST_DIR
unzip ADE20K_2016_07_26.zip
unzip ADEChallengeData2016.zip

rm ADE20K_2016_07_26.zip
rm ADEChallengeData2016.zip
echo "ADE20K dataset extracted."
