#!/bin/sh

# Extracts the Mapillary Academic Version dataset.

# By using this script, you agree to all terms and conditions
# of the Mapillary Vistas dataset and certify you legally
# obtained a license from https://www.mapillary.com/dataset/vistas

# Mapillary file (for mapillary-vistas-dataset_public_v1.1.zip) must 
# already be download before using this script.

# Destination directory for Mapillary
MAPILLARY_DST_DIR=$1
# Select default (>200 is quite helpful).
NUM_CORES_TO_USE=$2

# -------- Extraction -------------------------------------------
echo "Extracting Mapillary dataset..."

cd $MAPILLARY_DST_DIR
unzip mapillary-vistas-dataset_public_v1.1.zip

# Default is to not delete the zip file, since it's an involved process to obtain it.
# rm mapillary-vistas-dataset_public_v1.1.zip

echo "Mapillary dataset extracted."
