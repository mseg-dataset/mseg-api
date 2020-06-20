
## MSeg Download Overview

Downloading MSeg is nontrivial and we ask that you follow these instructions **exactly, in this order**. **Skipping steps will not be possible**. You will need at least 200 GB of free space on your hard drive. It may take several days before you are granted a license for ScanNet or Mapillary Vistas.

- If downloading and unzipping downloaded files in a single thread, and 100 workers (presumably on 100 cores) are provided for label map mask preparation, the MSeg setup may take up to 40 hours.

If you already have a copy of these datasets on your local server, we recommend that you do not use them. This is because we require very specific versioning -- ADE20K, COCO, KITTI, Mapillary Vistas, PASCAL VOC, ScanNet, etc. have several different release versions. 

We will remap all of the data on disk to align with the [semseg](https://github.com/hszhao/semseg) format: classes are given indices in the range [0,num_classes-1], and the `unlabeled` category is 255. The `unlabeled` category is never used as supervision -- rather, the loss from such pixels is set to zero.

Before downloading, we ask that you read the [MSeg paper](http://vladlen.info/papers/MSeg.pdf) and [class definition list](https://drive.google.com/file/d/1zBGSokcaKjEZU95J-hRnyim6y8zDVafs/view?usp=sharing) in their entirety.

The folder structure **must** be exactly as follows:
- {MSEG_DST_DIR} e.g. `/export/share/Datasets`
    - `mseg_dataset/`
    	- `ADE20K/`
        - `BDD/`
            - bdd100k_seg.zip
        - `Camvid/`
        - `Cityscapes/`
            - leftImg8bit_trainvaltest.zip
            - gtFine_trainvaltest.zip
        - `COCOPanoptic/`
        - `KITTI/` # note that we use the semantics split
        - `IDD/`
        - `MapillaryVistasPublic/`
        - `PASCAL_VOC_2012/` # note that we use the `train_aug` split
        - `PASCAL_Context/`
        - `ScanNet/` # note that we use the official 25k frame subset
        - `SUNRGBD/`
        - `WildDash/`
            - wd_both_01.zip
            - wd_val_01.zip


## Instructions
First we'll download the datasets that do not require registration, and then we will proceed to the registration process for those datasets that require it.

Beforehand, open `mseg/utils/dataset_config.py` and replace
```
MSEG_DST_DIR=""
```
with your value of `MSEG_DST_DIR` (the parent directory of `mseg_dataset`). Now, inside the `mseg-api` directory, install the `mseg` module via:

```
pip install -e .
```
Make sure that you can run `python -c "import mseg; print('hello world')"` in python, and you are good to go!

### Download Datasets that do not require registration

Inside the `download_scripts` directory, specify your MSeg parent directory:
```
MSEG_DST_DIR= # e.g. export/share/Datasets/MSegV2
NUM_CORES_TO_USE= # e.g. 200
```
Next, run the following commands:
```
# (execute inside the `download_scripts` directory)
mkdir -p $MSEG_DST_DIR/mseg_dataset
./mseg_download_noreg_datasets.sh $MSEG_DST_DIR 2>&1 | tee mseg_download_noreg.log
```
Now, we'll remap these registration-less datasets on disk to `semseg` label spaces:
```
./mseg_remap_noreg_datasets.sh $MSEG_DST_DIR $NUM_CORES_TO_USE 2>&1 | tee mseg_remap_noreg.log
```

## Initialize Folder Structure

In case you decide to manually upload zip files to the server instead of using our download script, initialize the directory structure first:
```
mkdir -p $MSEG_DST_DIR/mseg_dataset/BDD
mkdir -p $MSEG_DST_DIR/mseg_dataset/Cityscapes
mkdir -p $MSEG_DST_DIR/mseg_dataset/IDD
mkdir -p $MSEG_DST_DIR/mseg_dataset/KITTI
mkdir -p $MSEG_DST_DIR/mseg_dataset/MapillaryVistasPublic
mkdir -p $MSEG_DST_DIR/mseg_dataset/ScanNet
mkdir -p $MSEG_DST_DIR/mseg_dataset/WildDash
```


Now, we'll for register for BDD.

### BDD100K
The Berkeley Deep Drive dataset [[paper]](https://arxiv.org/abs/1805.04687) [[website]]() is available for public use. Register [here](https://bdd-data.berkeley.edu/login.html). You should be approved immediately and receive an email shortly. 

This dataset is not available via `wget`, so we ask that you download it in a browser, and upload it (1.3 GB) to the desired location on your server.

Log in, click the "Download" tab on the left, accept terms, then click the link for "Segmentation" (file should appear under name "bdd100k_seg.zip")

Locally upload the downloaded file to
```
scp bdd100k_seg.zip username@yourserver:/path/to/destination/mseg_dataset/BDD
```
Now, on desired server, execute inside the `mseg_dataset` directory:
```
./mseg_extract_bdd.sh $MSEG_DST_DIR/mseg_dataset/BDD 2>&1 | tee bdd_extraction.log
```

### Cityscapes
The Cityscapes dataset [[paper]](https://arxiv.org/abs/1604.01685) [[website]]()
Follow https://www.cityscapes-dataset.com/login/ to register. You will receive an email with a link to activate your account. 

If your server VPN is not too prohibitive, you can likely download the data directly over `wget`. Rename `set_credentials.sh.template` to `set_credentials.sh`, and copy your username and password into the fields CITYSCAPES_USERNAME and CITYSCAPES_PASSWORD into `set_credentials.sh`. The script will then automatically download the Cityscapes raw images (11GB) and Cityscapes labels (241MB).
```
source ./set_credentials.sh
./mseg_download_cityscapes.sh $MSEG_DST_DIR/mseg_dataset/Cityscapes 2>&1 | tee download_cityscapes.log
```
Otherwise upload the files in the correct place on your machine, per file system structure above. Now, extract the files as:
```
./mseg_extract_cityscapes.sh $MSEG_DST_DIR/mseg_dataset/Cityscapes 2>&1 | tee extract_cityscapes.log
./mseg_remap_cityscapes.sh $MSEG_DST_DIR/mseg_dataset/Cityscapes $NUM_CORES_TO_USE 2>&1 | tee remap_cityscapes.log
```

### Indian Driving Dataset (IDD)
The Indian Driving Dataset [[paper]](https://arxiv.org/abs/1811.10200) [[website]](https://idd.insaan.iiit.ac.in/) is available for public use. Register [here](https://idd.insaan.iiit.ac.in/accounts/login/?next=/dataset/download/), and your request will likely be approved immediately. Once logged in, you will find a link for "IDD - Segmentation (IDD 20k Part I) (18.5 GB)". Agree to the terms, and start the download in Chrome. In the Chrome Downloads page, pause the download (this will appear as "idd-segmentation.tar.gz"). Copy the download URL to download_mseg.sh.

```
./mseg_download_idd.sh $MSEG_DST_DIR/mseg_dataset/IDD 2>&1 | tee download_idd.log
./mseg_remap_idd.sh $MSEG_DST_DIR/mseg_dataset/IDD $NUM_CORES_TO_USE 2>&1 | tee remap_idd.log
```

### KITTI Dataset
The KITTI dataset suite [[paper]](http://www.cvlibs.net/publications/Geiger2013IJRR.pdf) [[website]](http://www.cvlibs.net/datasets/kitti/)  includes a small semantic segmentation benchmark. Register [here](http://www.cvlibs.net/download.php?file=data_semantics.zip) to download the file `data_semantics.zip`.

Unfortunately, you cannot simply wget `http://www.cvlibs.net/download.php?file=data_semantics.zip`, you will have to wait for an email from cvlibs.net. Copy the Amazon AWS URL into `download_kitti.sh` and run:

```
./mseg_download_kitti.sh $MSEG_DST_DIR/mseg_dataset/KITTI 2>&1 | tee download_kitti.log
./mseg_remap_kitti.sh $MSEG_DST_DIR/mseg_dataset/KITTI $NUM_CORES_TO_USE 2>&1 | tee remap_kitti.log
```

### Mapillary Vistas
The Mapillary Vistas dataset [[paper]](https://research.mapillary.com/img/publications/ICCV17a.pdf) [[website]](https://www.mapillary.com/dataset/vistas/) is available for public use under license. Register [here](https://www.mapillary.com/dataset/vistas?) by clicking on "Request Dataset" under *Research Addition* section of the page. Within a few days, you will be sent an AWS URL for a file "mapillary-vistas-dataset_public_v1.1.zip".

Copy this AWS URL to the variable `MAPILLARY_ZIP_URL="url-goes-here"` in `mseg_download_mapillary.sh`, and run:
```
# execute inside the mseg-api/download_scripts directory
./mseg_download_mapillary.sh $MSEG_DST_DIR/mseg_dataset/MapillaryVistasPublic 2>&1 | tee download_mapillary.log
./mseg_extract_mapillary.sh $MSEG_DST_DIR/mseg_dataset/MapillaryVistasPublic 2>&1 | tee extract_mapillary.log
./mseg_remap_mapillary.sh $MSEG_DST_DIR/mseg_dataset/MapillaryVistasPublic $NUM_CORES_TO_USE 2>&1 | tee remap_mapillary.log
```
Expect re-mapping to require around 2 hours with 100 workers.

### ScanNet
The ScanNet dataset [[paper]](https://arxiv.org/abs/1702.04405) [[website]](http://www.scan-net.org/) is available for public use under license.
fill out a [PDF agreement](http://dovahkiin.stanford.edu/scannet-public/ScanNet_TOS.pdf) to the ScanNet Terms of Use and send it to the scannet group email (scannet@googlegroups.com). While the full dataset is extremely large, we use the 25k image test set [benchmark](http://kaldir.vc.in.tum.de/scannet_benchmark/documentation) only, which is much smaller. You will be provided an email with link to a Python download script. In this download script, find the line `BASE_URL=...` and copy it into `mseg_download_scannet.sh`.

Instead of downloading all 2.5M frames of ScanNet (1.2 GB), which are quite redundant (contiguous video sequences), we use the official 25k-frame subset (subsampled every 100 frames), which is just 5.6 GB.

```
./mseg_download_scannet.sh $MSEG_DST_DIR/mseg_dataset/ScanNet 2>&1 | tee download_scannet.log
```

Since we use ScanNet only as an evaluation dataset, we remap the dataset to the 20-class evaluation subset [defined here](http://kaldir.vc.in.tum.de/scannet_benchmark/labelids.txt), in accordance with the [benchmark definition](http://kaldir.vc.in.tum.de/scannet_benchmark/documentation):

```
./mseg_remap_scannet.sh $MSEG_DST_DIR/mseg_dataset/ScanNet $NUM_CORES_TO_USE 2>&1 | tee remap_scannet.log
```

### WildDash
The WildDash dataset [[paper]](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Oliver_Zendel_WildDash_-_Creating_ECCV_2018_paper.pdf) [[website]](https://wilddash.cc/) is available for public use. Register [here](https://wilddash.cc/accounts/login?next=/download). 

If your server VPN is not too prohibitive, you can likely download WildDash directly using our script. Rename `set_credentials.sh.template` to `set_credentials.sh` if you haven't yet, copy your username and password into the fields `WILDDASH_USERNAME` and `WILDDASH_PASSWORD` in `set_credentials.sh`. The script will then automatically download the WildDash raw images (??GB) and WildDash val set labels (??MB).
```
source ./set_credentials.sh
./mseg_download_wilddash.sh $MSEG_DST_DIR/mseg_dataset/WildDash 2>&1 | tee download_wilddash.log
```

Otherwise, download  to your local machine via browser "wd_both_01.zip" and "wd_val_01.zip". Then upload these to your server as:

```
scp wd_both_01.zip username@yourserver:/path/to/destination/mseg_dataset/WildDash
scp wd_val_01.zip username@yourserver:/path/to/destination/mseg_dataset/WildDash
```
Now, extract the downloaded files and remap the labels:
```
./mseg_extract_wilddash.sh $MSEG_DST_DIR/mseg_dataset/WildDash 2>&1 | tee extract_wilddash.log
./mseg_remap_wilddash.sh $MSEG_DST_DIR/mseg_dataset/WildDash $NUM_CORES_TO_USE 2>&1 | tee remap_wilddash.log
```

### Apply Re-Labeling
We've re-labeled many masks from each of the training datasets. Run:

```
./mseg_apply_relabeling.sh $NUM_CORES_TO_USE 2>&1 | tee apply_relabeling.log
```
This will take around 45 minutes to execute on 100 cores.

### Verification Stage

At this point, you have reached the final stage -- verification. We will loop through every path to ensure it exists. Run the following command:
```
python -u ../tests/verify_all_dataset_paths_exist.py
python -u ../tests/verify_all_relabeled_segments.py --num_processes $NUM_CORES_TO_USE
```

For the sake of your privacy, clear all stored credentials using:
```
source ./remove_credentials.sh
```

## FAQ

Now, please consult [taxonomy_FAQ.md](https://github.com/mseg-dataset/mseg-api-staging/blob/master/download_scripts/taxonomy_FAQ.md) to learn what each of these dataset names means.





