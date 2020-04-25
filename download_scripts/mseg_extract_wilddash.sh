#!/bin/sh

# This script extracts the WildDash dataset.

WILDDASH_DST_DIR=$1
mkdir -p $WILDDASH_DST_DIR

# -------- Extraction ---------------------
echo "Extracting WildDash dataset..."

cd $WILDDASH_DST_DIR
unzip wd_both_01.zip

# don't need two copies of these files (found in both)
rm -rf anonymized
rm authors.txt
rm license.txt
rm readme.txt

unzip wd_val_01.zip

rm wd_both_01.zip
rm wd_val_01.zip
cd ..
echo "WildDash dataset extracted."
