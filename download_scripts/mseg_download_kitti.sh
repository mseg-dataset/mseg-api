#!/bin/sh

# Downloads the KITTI segmentattion dataset

# By using this script, you agree to all terms and licenses set
# forth by the KITTI Dataset enumerated at
# http://www.cvlibs.net/datasets/kitti/

# ------- Populate Variable -------------------------------

# Fill in the value from cvlibs.net email here:
KITTI_ZIP_URL=""

# ------- Downloading -------------------------------------

KITTI_DST_DIR=$1
echo "KITTI will be downloaded to "$KITTI_DST_DIR

mkdir -p $KITTI_DST_DIR
cd $KITTI_DST_DIR
wget $KITTI_ZIP_URL

unzip data_semantics.zip
rm data_semantics.zip

echo "KITTI dataset downloaded."