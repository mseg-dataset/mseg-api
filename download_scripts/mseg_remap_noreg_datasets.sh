#!/bin/sh

MSEG_DST_DIR=$1
# number of cores to use for multiprocessing
NUM_CORES=$2

now=$(date +"%Y %m %d @ %H:%M:%S")
echo "Start remapping registration-less datasets at "$now

./mseg_remap_ade20k.sh $MSEG_DST_DIR/mseg_dataset/ADE20K $NUM_CORES
./mseg_remap_camvid.sh $MSEG_DST_DIR/mseg_dataset/Camvid $NUM_CORES
./mseg_remap_cocopanoptic.sh $MSEG_DST_DIR/mseg_dataset/COCOPanoptic $NUM_CORES
./mseg_remap_pascalcontext.sh $MSEG_DST_DIR/mseg_dataset/PASCAL_Context $NUM_CORES
./mseg_remap_sunrgbd.sh $MSEG_DST_DIR/mseg_dataset/SUNRGBD $NUM_CORES

now=$(date +"%Y %m %d @ %H:%M:%S")
echo "Finished remapping of registration-less datasets at "$now