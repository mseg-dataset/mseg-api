#!/bin/sh

# By using this script, you agree to the terms and conditions
# of the ADE20K, Camvid, COCO, COCO Stuff, COCO Panoptic,
# PASCAL Context, PASCAL VOC, and SUN RGB-D datasets.

# (Start this script in a screen or tmux and walk away, may take ~5 hours to run)

MSEG_DST_DIR=$1

now=$(date +"%Y %m %d @ %H:%M:%S")
echo "Start downloading registration-less datasets at "$now

./mseg_download_ade20k.sh $MSEG_DST_DIR/mseg_dataset/ADE20K
./mseg_download_camvid.sh $MSEG_DST_DIR/mseg_dataset/Camvid
./mseg_download_cocopanoptic.sh $MSEG_DST_DIR/mseg_dataset/COCOPanoptic
./mseg_download_pascalcontext.sh $MSEG_DST_DIR/mseg_dataset/PASCAL_Context
./mseg_download_pascalvoc2012.sh $MSEG_DST_DIR/mseg_dataset/PASCAL_VOC_2012
./mseg_download_sunrgbd.sh $MSEG_DST_DIR/mseg_dataset/SUNRGBD

now=$(date +"%Y %m %d @ %H:%M:%S")
echo "Finished download of registration-less datasets at "$now