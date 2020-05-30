#!/bin/sh

# Apply re-labeling to each of these datasets.

NUM_CORES_TO_USE=$1

# ---------- Environment Variables / Directory Setup -------------------------

# navigate to repo root
cd ..
# gets the current working dir (for repo root)
REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "Repo is stored at "$REPO_ROOT

# --------- Relabeling ------------------------------------------------------

now=$(date +"%Y %m %d @ %H:%M:%S")
echo "Start re-labeling of training datasets at "$now
echo "Apply re-labeling to ade20k-150"

python -u $REPO_ROOT/mseg/label_preparation/mseg_write_relabeled_segments.py \
	--num_processes $NUM_CORES_TO_USE --dataset_to_relabel ade20k-150

echo "Apply re-labeling to bdd"

python -u $REPO_ROOT/mseg/label_preparation/mseg_write_relabeled_segments.py \
	--num_processes $NUM_CORES_TO_USE --dataset_to_relabel bdd

echo "Apply re-labeling to cityscapes-19-relabeled"

python -u $REPO_ROOT/mseg/label_preparation/mseg_write_relabeled_segments.py \
	--num_processes $NUM_CORES_TO_USE --dataset_to_relabel cityscapes-19

echo "Apply re-labeling to cityscapes-34-relabeled"

python -u $REPO_ROOT/mseg/label_preparation/mseg_write_relabeled_segments.py \
	--num_processes $NUM_CORES_TO_USE --dataset_to_relabel cityscapes-34

echo "Apply re-labeling to coco-panoptic-133"

python -u $REPO_ROOT/mseg/label_preparation/mseg_write_relabeled_segments.py \
	--num_processes $NUM_CORES_TO_USE --dataset_to_relabel coco-panoptic-133

echo "Apply re-labeling to idd-39"

python -u $REPO_ROOT/mseg/label_preparation/mseg_write_relabeled_segments.py \
	--num_processes $NUM_CORES_TO_USE --dataset_to_relabel idd-39

echo "Apply re-labeling to mapillary-public65"

python -u $REPO_ROOT/mseg/label_preparation/mseg_write_relabeled_segments.py \
	--num_processes $NUM_CORES_TO_USE --dataset_to_relabel mapillary-public65

echo "Apply re-labeling to sunrgbd-37"

python -u $REPO_ROOT/mseg/label_preparation/mseg_write_relabeled_segments.py \
	--num_processes $NUM_CORES_TO_USE --dataset_to_relabel sunrgbd-37

now=$(date +"%Y %m %d @ %H:%M:%S")
echo "Finished re-labeling of training datasets at "$now
