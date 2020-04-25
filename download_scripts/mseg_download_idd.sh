#!/bin/sh

# Downloads the IDD dataset

# By using this script, you agree to all terms and licenses set
# forth by the Indian Driving Dataset enumerated at
# https://idd.insaan.iiit.ac.in/

# An extension in the future would be to also use parts of
# IDD part 2 (idd-20k-II.tar.gz), which we have not re-labeled

# Copy IDD part 1 download URL here (for idd-segmentation.tar.gz)
IDD_PART1_ZIP_URL=""

IDD_DST_DIR=$1
mkdir -p $IDD_DST_DIR

# -------- Downloading ------------------------------------------
echo "Downloading IDD dataset..."
cd $IDD_DST_DIR

wget -c -O idd-segmentation.tar.gz "$IDD_PART1_ZIP_URL"

# Ignore Part II for now
# wget -c -O idd-20k-II.tar.gz "$IDD_PART2_ZIP_URL"

echo "IDD dataset downloaded."

# -------- Extraction ------------------------------------------
echo "Extracting IDD dataset..."

tar -xzf idd-segmentation.tar.gz
rm idd-segmentation.tar.gz

echo "IDD dataset extracted."