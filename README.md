[![Build Status](https://travis-ci.com/mseg-dataset/mseg-api.svg?branch=master)](https://travis-ci.com/mseg-dataset/mseg-api)

MSeg: A Composite Dataset for Multi-domain Semantic Segmentation (CVPR 2020, Official Repo)

This is the code for the paper:

**MSeg: A Composite Dataset for Multi-domain Semantic Segmentation** [[PDF]](http://vladlen.info/papers/MSeg.pdf)
<br>
[John Lambert*](https://johnwlambert.github.io/),
[Zhuang Liu*](https://liuzhuang13.github.io/),
[Ozan Sener](http://ozansener.net/),
[James Hays](https://www.cc.gatech.edu/~hays/),
[Vladlen Koltun](http://vladlen.info/)
<br>
Presented at [CVPR 2020](http://cvpr2018.thecvf.com/)



This repo is the first of 4 repos that introduce our work. It provides utilities to download the MSeg dataset (which is nontrivial), and prepare the data on disk in a unified taxonomy.

In a few weeks, we will add the `TaxonomyConverter` class to this repo that supports on-the-fly mapping to a unified taxonomy during training.

Three additional repos will be introduced in April and May 2020:
- ` mseg_semantic`: provides HRNet-W48 Training (sufficient to train a winning entry on the WildDash benchmark)
- `mseg_panoptic`: provides Panoptic-FPN and Mask-RCNN training, based on Detectron2
- `mseg_mturk`: provides utilities to perform large-scale Mechanical Turk re-labeling

<div align='center'>
  <img src='https://user-images.githubusercontent.com/16724970/80264666-d663c080-8662-11ea-9805-366c246befed.jpg' height="350px">
</div>

### Install the MSeg module:

* `mseg` can be installed as a python package using

        pip install -e /path_to_root_directory_of_the_repo/

Make sure that you can run `import mseg` in python, and you are good to go!

### Download MSeg

* Navigate to [download_scripts/README.md](https://github.com/mseg-dataset/mseg-api-staging/blob/master/download_scripts/README.md) for instructions.

### The MSeg Taxonomy

We provide comprehensive class definitions and examples [here](https://drive.google.com/file/d/1zBGSokcaKjEZU95J-hRnyim6y8zDVafs/view?usp=sharing). We provide [here](https://github.com/mseg-dataset/mseg-api-staging/blob/master/mseg/class_remapping_files/MSeg_master.tsv) a master spreadsheet mapping all training datasets to the MSeg Taxonomy, and the MSeg Taxonomy to test datasets. Please consult [taxonomy_FAQ.md](https://github.com/mseg-dataset/mseg-api-staging/blob/master/download_scripts/taxonomy_FAQ.md) to learn what each of the dataset taxonomy names means.

## Citing MSeg

If you find this code useful for your research, please cite:
```
@InProceedings{MSeg_2020_CVPR,
author = {Lambert, John and Zhuang, Liu and Sener, Ozan and Hays, James and Koltun, Vladlen},
title = {{MSeg}: A Composite Dataset for Multi-domain Semantic Segmentation},
booktitle = {Computer Vision and Pattern Recognition (CVPR)},
year = {2020}
}
```
