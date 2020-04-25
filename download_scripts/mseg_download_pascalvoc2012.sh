#!/bin/sh

# Downloads the PASCAL VOC 2012 dataset

# By using this script, you agree to all terms and conditions
# set forth by the original authors of the PASCAL VOC 2012
# dataset.

VOC12_DST_DIR=$1

# ------------ Directory Setup ------------------------------
echo "PASCAL VOC 2012 will be downloaded to "$VOC12_DST_DIR
mkdir -p $VOC12_DST_DIR

# ------------ Downloading ----------------------------------
echo "Downloading PASCAL VOC 2012 dataset..."
cd $VOC12_DST_DIR

# Download images, will be about 1.9 GB
VOC12_TAR_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
wget $VOC12_TAR_URL

# Download trainaug labels, courtesy: https://github.com/DrSleep/tensorflow-deeplab-resnet
# train_aug originally from http://home.bharathh.info/pubs/codes/SBD/download.html
wget -c -O SegmentationClassAug.zip "https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=1"

echo "PASCAL VOC 2012 dataset downloaded."

# ------------ Extraction -----------------------------------
echo "Extracting PASCAL VOC 2012 dataset..."

# unzip labels
unzip SegmentationClassAug.zip
rm SegmentationClassAug.zip

# unzip images
tar xopf VOCtrainval_11-May-2012.tar
mv VOCdevkit/VOC2012/JPEGImages .
rm VOCtrainval_11-May-2012.tar
rm -rf VOCdevkit

echo "PASCAL VOC 2012 dataset extracted."


