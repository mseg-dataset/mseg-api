#!/bin/sh

# Downloads the COCO Panoptic dataset.

# By using this script, you agree to all terms
# and conditions set forth by the creators of the
# COCO Stuff, MS COCO, and COCO Panoptic datasets.

# ----------- Directory Setup -----------------------
# Destination directory for MSeg
COCOP_DST_DIR=$1
echo "COCO Panoptic will be downloaded to "$COCOP_DST_DIR
mkdir -p $COCOP_DST_DIR

# ----------- Downloading ---------------------------
cd $COCOP_DST_DIR
echo "Downloading COCO Panoptic dataset..."
# Get the annotations.
COCOP_ANNOT_URL="http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip"
wget $COCOP_ANNOT_URL

# train2017.zip will be 19GB.
TRAIN_IMGS_URL="http://images.cocodataset.org/zips/train2017.zip"
wget $TRAIN_IMGS_URL

VAL_IMGS_URL="http://images.cocodataset.org/zips/val2017.zip"
wget $VAL_IMGS_URL

echo "COCO Panoptic dataset downloaded."
echo "Extracting COCO Panoptic dataset..."
mkdir -p images
unzip train2017.zip -d images
rm train2017.zip

unzip val2017.zip -d images
rm val2017.zip

unzip panoptic_annotations_trainval2017.zip
rm panoptic_annotations_trainval2017.zip

cd annotations
unzip panoptic_train2017.zip
rm panoptic_train2017.zip
unzip panoptic_val2017.zip
rm panoptic_val2017.zip

echo "COCO Panoptic dataset extracted."
