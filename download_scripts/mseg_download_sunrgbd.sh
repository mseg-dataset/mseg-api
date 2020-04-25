#!/bin/sh

# Downloads the SUN RGB-D dataset.

# By using this script, you agree to all terms and 
# conditions set forth by the creators of the 
# SUN RGBD-D dataset.


SUNRGBD_DST_DIR=$1
mkdir -p $SUNRGBD_DST_DIR

# ------- Downloading ---------------------------

echo "Downloading SUN RGB-D dataset..."
cd $SUNRGBD_DST_DIR

# Courtesy Chris Choy, https://github.com/chrischoy/SUN_RGBD.git
wget http://cvgl.stanford.edu/data2/sun_rgbd.tgz
echo "SUN RGB-D dataset downloaded."

# ------- Extraction ----------------------------

echo "Extracting SUN RGB-D dataset..."
tar -xzf sun_rgbd.tgz

# should be renamed since actually distributed
# in 38-class taxonomy.
mv label37 label38
rm sun_rgbd.tgz
rm -rf depth

echo "SUN RGB-D dataset extracted."

