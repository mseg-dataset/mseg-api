#!/bin/sh

# Extracts the Cityscapes dataset.

# By using this script, you agree to all terms and
# conditions set forth by the Cityscapes dataset
# creators and certify you have lawfully obtained
# a license from their website.
# License terms can be found here:
# https://github.com/mcordts/cityscapesScripts/blob/master/license.txt

# ------------- Directory + Environment Variable Setup  -----

# Destination directory for Cityscapes
CITYSCAPES_DST_DIR=$1

echo "Cityscapes will be extracted to "$CITYSCAPES_DST_DIR
mkdir -p $CITYSCAPES_DST_DIR

# ------------- Extraction ---------------------------------

echo "Extracting Cityscapes dataset..."
cd $CITYSCAPES_DST_DIR

unzip gtFine_trainvaltest.zip

# rm to delete prevent pop-up overwrite question
# Note: README and license are found in both files
rm README
rm license.txt

unzip leftImg8bit_trainvaltest.zip

rm gtFine_trainvaltest.zip
rm leftImg8bit_trainvaltest.zip
cd ..
echo "Cityscapes dataset extracted."


