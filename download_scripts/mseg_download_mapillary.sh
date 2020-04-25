#!/bin/sh

# Downloads the Mapillary Academic Version dataset.

# By using this script, you agree to all terms and conditions
# of the Mapillary Vistas dataset and certify you legally
# obtained a license from https://www.mapillary.com/dataset/vistas

# Copy Mapillary download URL here (for mapillary-vistas-dataset_public_v1.1.zip)
MAPILLARY_ZIP_URL=""

# --------------Downloading ----------------------------

# Destination directory for Mapillary
MAPILLARY_DST_DIR=$1

echo "Mapillary will be downloaded to "$MAPILLARY_DST_DIR
mkdir -p $MAPILLARY_DST_DIR

# If not already uploaded, comment out this line.
wget -c -O $MAPILLARY_DST_DIR/mapillary-vistas-dataset_public_v1.1.zip $MAPILLARY_ZIP_URL
echo "Mapillary dataset downloaded."

