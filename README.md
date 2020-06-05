[![Build Status](https://travis-ci.com/mseg-dataset/mseg-api.svg?branch=master)](https://travis-ci.com/mseg-dataset/mseg-api)
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a>

This is the code for the paper:

**MSeg: A Composite Dataset for Multi-domain Semantic Segmentation** (CVPR 2020, Official Repo) [[PDF]](http://vladlen.info/papers/MSeg.pdf)
<br>
[John Lambert*](https://johnwlambert.github.io/),
[Zhuang Liu*](https://liuzhuang13.github.io/),
[Ozan Sener](http://ozansener.net/),
[James Hays](https://www.cc.gatech.edu/~hays/),
[Vladlen Koltun](http://vladlen.info/)
<br>
Presented at [CVPR 2020](http://cvpr2018.thecvf.com/)

This repo is the first of 4 repos that introduce our work. It provides utilities to download the MSeg dataset (which is nontrivial), and prepare the data on disk in a unified taxonomy.

<p align="left">
  <img src="https://user-images.githubusercontent.com/62491525/83895683-094caa00-a721-11ea-8905-2183df60bc4f.gif" height="250">
  <img src="https://user-images.githubusercontent.com/62491525/83893966-aeb24e80-a71e-11ea-84cc-80e591f91ec0.gif" height="250">
</p>
<p align="left">
  <img src="https://user-images.githubusercontent.com/62491525/83895915-57fa4400-a721-11ea-8fa9-3c2ff0361080.gif" height="250">
  <img src="https://user-images.githubusercontent.com/62491525/83895972-73654f00-a721-11ea-8438-7bd43b695355.gif" height="250"> 
</p>

<p align="left">
  <img src="https://user-images.githubusercontent.com/62491525/83893958-abb75e00-a71e-11ea-978c-ab4080b4e718.gif" height="250">
  <img src="https://user-images.githubusercontent.com/62491525/83895490-c094f100-a720-11ea-9f85-cf4c6b030e73.gif" height="250">
</p>

<p align="left">
  <img src="https://user-images.githubusercontent.com/62491525/83895811-35682b00-a721-11ea-9641-38e3b2c1ad0e.gif" height="250">
  <img src="https://user-images.githubusercontent.com/62491525/83895963-6e080480-a721-11ea-98b6-49835d9e733a.gif" height="250">
</p>

Three additional repos are also provided:
- [`mseg-semantic`](https://github.com/mseg-dataset/mseg-semantic): provides HRNet-W48 Training (sufficient to train a winning entry on the [WildDash](https://wilddash.cc/benchmark/summary_tbl?hc=semantic_rob) benchmark)
- `mseg-panoptic`: provides Panoptic-FPN and Mask-RCNN training, based on Detectron2 (will be introduced in June 2020)
- `mseg-mturk`: provides utilities to perform large-scale Mechanical Turk re-labeling (will be introduced in June 2020)

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

## Repo Structure
- `download_scripts`: code and instructions to download the entire MSeg dataset
- `mseg`: Python module, including
    - `dataset_apis`
    - `dataset_lists`: ordered classnames for each dataset, and corresponding relative rgb/label file paths
    - `label_preparation`: code for remapping to `semseg` format, and for relabeling masks in place
    - `relabeled_data`: MSeg data, annotated by Mechanical Turk workers, and verified by co-authors
    - `taxonomy`: on-the-fly mapping to a unified taxonomy during training, and linear mapping to evaluation taxonomies
    - `utils`: library functions for mask and image manipulation, filesystem, tsv/csv reading, and multiprocessing
- `tests`: unit tests on all code

## License
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
