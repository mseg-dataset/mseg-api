![Linux CI](https://github.com/mseg-dataset/mseg-api/workflows/Python%20CI/badge.svg)

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a>

This is the code for the paper:

**MSeg: A Composite Dataset for Multi-domain Semantic Segmentation** (CVPR 2020, Official Repo) [[CVPR PDF]](http://vladlen.info/papers/MSeg.pdf) [[Journal PDF]](https://arxiv.org/abs/2112.13762)
<br>
[John Lambert*](https://johnwlambert.github.io/),
[Zhuang Liu*](https://liuzhuang13.github.io/),
[Ozan Sener](http://ozansener.net/),
[James Hays](https://www.cc.gatech.edu/~hays/),
[Vladlen Koltun](http://vladlen.info/)
<br>
Presented at [CVPR 2020](http://cvpr2020.thecvf.com/). Link to [MSeg Video (3min) ](https://youtu.be/PzBK6K5gyyo)

**NEWS**:
- [Dec. 2021]: An updated journal-length version of our work is now available on ArXiv [here](https://arxiv.org/abs/2112.13762).

This repo is the first of 4 repos that introduce our work. It provides utilities to download the MSeg dataset (which is nontrivial), and prepare the data on disk in a unified taxonomy.

<p align="center">
  <img src="https://user-images.githubusercontent.com/62491525/83895683-094caa00-a721-11ea-8905-2183df60bc4f.gif" height="215">
  <img src="https://user-images.githubusercontent.com/62491525/83893966-aeb24e80-a71e-11ea-84cc-80e591f91ec0.gif" height="215">
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/62491525/83895915-57fa4400-a721-11ea-8fa9-3c2ff0361080.gif" height="215">
  <img src="https://user-images.githubusercontent.com/62491525/83895972-73654f00-a721-11ea-8438-7bd43b695355.gif" height="215"> 
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/62491525/83893958-abb75e00-a71e-11ea-978c-ab4080b4e718.gif" height="215">
  <img src="https://user-images.githubusercontent.com/62491525/83895490-c094f100-a720-11ea-9f85-cf4c6b030e73.gif" height="215">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/62491525/83895811-35682b00-a721-11ea-9641-38e3b2c1ad0e.gif" height="215">
  <img src="https://user-images.githubusercontent.com/62491525/83896026-8710b580-a721-11ea-86d2-a0fff9c6e26e.gif" height="215">
</p>

Three additional repos are also provided:
- [`mseg-semantic`](https://github.com/mseg-dataset/mseg-semantic): provides HRNet-W48 Training (sufficient to train a winning entry on the [WildDash](https://wilddash.cc/benchmark/summary_tbl?hc=semantic_rob) benchmark)
- `mseg-panoptic`: provides Panoptic-FPN and Mask-RCNN training, based on Detectron2 (will be introduced in January 2021)
- [`mseg-mturk`](https://github.com/mseg-dataset/mseg-mturk): utilities to perform large-scale Mechanical Turk re-labeling

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
author = {Lambert, John and Liu, Zhuang and Sener, Ozan and Hays, James and Koltun, Vladlen},
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

## Data License
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

## Frequently Asked Questions (FAQ)
**Q**: Do the weights include the model structure or it's just the weights? If the latter, which model do these weights refer to? Under the `models` directory, there are several model implementations.

**A**: The pre-trained models follow the HRNet-W48 architecture. The model structure is defined in the code [here](https://github.com/mseg-dataset/mseg-semantic/blob/master/mseg_semantic/model/seg_hrnet.py#L274). The saved weights provide a dictionary between keys (unique IDs for each weight identifying the corresponding layer/layer type) and values (the floating point weights).

**Q**: How is testing performed on the test datasets? In the paper you talk about "zero-shot transfer" -- how this is performed? Are the test dataset labels also mapped or included in the unified taxonomy? If you remapped the test dataset labels to the unified taxonomy, are the reported results the performances on the unified label space, or on each test dataset's original label space? How did you you obtain results on the WildDash dataset - which is evaluated by the server - when the MSeg taxonomy may be different from the WildDash dataset.

**A**: Regarding "zero-shot transfer", please refer to section "Using the MSeg taxonomy on a held-out dataset" on page 6 of [our paper](http://vladlen.info/papers/MSeg.pdf). This section describes how we hand-specify mappings from the unified taxonomy to each test dataset's taxonomy as a linear mapping (implemented [here](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/taxonomy/taxonomy_converter.py#L220) in mseg-api). All results are in the test dataset's original label space (i.e. if WildDash expects class indices in the range [0,18] per our [names_list](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/dataset_lists/wilddash-19/wilddash-19_names.txt), our testing script uses the `TaxonomyConverter` [`transform_predictions_test()`](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/taxonomy/taxonomy_converter.py#L267) functionality  to produce indices in that range, remapping probabilities.

**Q**: Why don't indices in `MSeg_master.tsv` match the training indices in individual datasets? For example, for the *road* class: In [idd-39](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/dataset_lists/idd-39/idd-39_names.txt#L1), *road* has index 0, but in [idd-39-relabeled](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/dataset_lists/idd-39-relabeled/idd-39-relabeled_names.txt#L20), *road* has index 19. It is index 7 in [cityscapes-34](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/dataset_lists/cityscapes-34/cityscapes-34_names.txt#L8). The [cityscapes-19-relabeled index](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/dataset_lists/cityscapes-19-relabeled/cityscapes-19-relabeled_names.txt) *road* is 11. As far as I can tell, ultimately the 'MSeg_Master.tsv' file provides the final mapping to the MSeg label space. But here, the *road* class seems to have an index of 98, which is neither 19 nor 11.

**A**: Indeed, [unified taxonomy class index 98](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/class_remapping_files/MSeg_master.tsv#L100) represents "road". But we use the TaxonomyConverter to accomplish the mapping on the fly from *idd-39-relabeled* to the unified/universal taxonomy (we use the terms "unified" and "universal" interchangeably). This is done by adding a transform in the training loop that calls [`TaxonomyConverter.transform_label()`](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/taxonomy/taxonomy_converter.py#L250) on the fly. You can see how that transform is implemented [here](https://github.com/mseg-dataset/mseg-semantic/blob/add-dataset-eval/mseg_semantic/utils/transform.py#L52.) in `mseg-semantic`.

**Q**: When testing, but there are test classes that are not in the unified taxonomy (e.g. Parking, railtrack, bridge etc. in WildDash), how do you produce predictions for that class? I understand you map the predictions with a binary matrix. But what do you do when there's no one-to-one correspondence?

**A**: WildDash v1 uses the 19-class taxonomy for evaluation, just like Cityscapes. So we use [the following script](https://github.com/mseg-dataset/mseg-api/blob/master/download_scripts/mseg_remap_wilddash.sh) to remap the 34-class taxonomy to 19-class taxonomy for WildDash  for testing inference and submission. You can see how Cityscapes evaluates just 19 of the 34 classes here in the [evaluation script](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py#L301) and in [the taxonomy definition](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py#L73). However, [bridge](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/class_remapping_files/MSeg_master.tsv#L34) and [rail track](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/class_remapping_files/MSeg_master.tsv#L99) are actually included in our unified taxonomy, as you’ll see in MSeg_master.tsv.

**Q**: How are datasets images read in for training/inference? Should I use the `dataset_apis` from `mseg-api`?

**A**: The `dataset_apis` from `mseg-api` are not for training or inference. They are purely for generating the MSeg dataset labels on disk. We read in the datasets using [`mseg_semantic/utils/dataset.py`](https://github.com/mseg-dataset/mseg-semantic/blob/master/mseg_semantic/utils/dataset.py) and then remap them to the universal space on the fly.
