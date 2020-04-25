#!/bin/sh

# This script downloads the PASCAL Context dataset.

# By using this script, you agree to any terms or 
# conditions set forth by the PASCAL VOC creators
# or the PASCAL Context dataset creators.

# ----------- Environment Variables + Directory Setup -----------------------

PCONTEXT_DST_DIR=$1
echo "PASCAL Context will be downloaded to "$PCONTEXT_DST_DIR
mkdir -p $PCONTEXT_DST_DIR

# navigate to repo root
cd ..
# gets the current working dir (for repo root)
REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "Repo is stored at "$REPO_ROOT

# ----------- Downloading -----------------------

# Uses PASCAL VOC 2010 Images.
VOC10_IMGS_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar"
# Downloads .mat files containing the PASCAL Context labels.
PASCAL_CONTEXT_LABELS_URL="https://cs.stanford.edu/~roozbeh/pascal-context/trainval.tar.gz"

echo "Downloading PASCAL Context dataset..."
wget -c -O $PCONTEXT_DST_DIR/VOCtrainval_03-May-2010.tar $VOC10_IMGS_URL
wget -c -O $PCONTEXT_DST_DIR/trainval.tar.gz $PASCAL_CONTEXT_LABELS_URL

echo "PASCAL Context dataset downloaded."

# ----------- Extraction -----------------------

echo "Extracting PASCAL Context dataset..."
cd $PCONTEXT_DST_DIR

tar xopf VOCtrainval_03-May-2010.tar
mv VOCdevkit/VOC2010/JPEGImages .
rm -rf VOCdevkit
rm VOCtrainval_03-May-2010.tar

tar -xzf trainval.tar.gz
rm trainval.tar.gz

# Dump .mat files to .pngs
python $REPO_ROOT/mseg/label_preparation/mseg_dump_pascalcontext_pngs.py \
	--pcontext_dst_dir $PCONTEXT_DST_DIR
rm -rf trainval

echo "PASCAL Context dataset extracted."

