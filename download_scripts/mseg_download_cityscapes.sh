#!/bin/sh

# Downloads the Cityscapes dataset.

# By using this script, you agree to all terms and
# conditions set forth by the Cityscapes dataset
# creators and certify you have lawfully obtained
# a license from their website.
# License terms can be found here:
# https://github.com/mcordts/cityscapesScripts/blob/master/license.txt

# If over VPN, otherwise upload from browswer.

# ------------------- Directory Setup ---------------------------
CITYSCAPES_DST_DIR=$1

echo "Cityscapes will be downloaded to "$CITYSCAPES_DST_DIR
mkdir -p $CITYSCAPES_DST_DIR
# ------------------- Downloading -------------------------------

# Cityscapes images download URL (leftImg8bit_trainvaltest.zip)
CITYSCAPES_IMGS_ZIP_URL="https://www.cityscapes-dataset.com/file-handling/?packageID=3"
# Cityscapes ground truth download URL (for gtFine_trainvaltest.zip) 
CITYSCAPES_GT_ZIP_URL="https://www.cityscapes-dataset.com/file-handling/?packageID=1"

cd $CITYSCAPES_DST_DIR

USERDATA="username=$CITYSCAPES_USERNAME&password=$CITYSCAPES_PASSWORD&submit=Login"
echo $USERDATA
wget --keep-session-cookies --save-cookies=cookies.txt --post-data $USERDATA https://www.cityscapes-dataset.com/login/
# will download "gtFine_trainvaltest.zip" (241MB)
wget --load-cookies cookies.txt --content-disposition $CITYSCAPES_GT_ZIP_URL
# will download "leftImg8bit_trainvaltest.zip (11GB)"
wget --load-cookies cookies.txt --content-disposition $CITYSCAPES_IMGS_ZIP_URL
cd ..
echo "Cityscapes dataset downloaded."
