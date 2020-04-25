
## What do the taxonomy names mean?

After generating the dataset with our download scripts, you will have a large number of datasets available. We use a specific format for training and testing, as described below:

In the following table, for each dataset (represented by a row), we describe the different taxonomies used to distribute the data by authors (`raw name`), our remapped version in the semseg-label format (`semseg-format name`), and name of dataset in relabeled taxonomy (`relabeled name`):

| Dataset Type |  raw name          | semseg-format name |  relabeled name              |
| :----------: | :------------:     | :----------------: | :-------------------------:  | 
|   Training   | ade20k-151         |     ade20k-150     |  ade20k-150-relabeled        |
|   Training   | bdd                | bdd                |  bdd-relabeled               |
|   Testing    | camvid-32          | camvid-11          | N/A                          |
|   Training   | cityscapes-34      | cityscapes-19      | cityscapes-19-relabeled      |
|   Training   |coco-panoptic-201*  | coco-panoptic-133  | coco-panoptic-133-relabeled  |
|   Training   |idd-40              | idd-39             | idd-39-relabeled             |
|   Testing    |kitti-34            | kitti-19           | N/A                          |
|   Training   |mapillary-public66  | mapillary-public65 | mapillary-public65-relabeled |
|   Testing    |pascal-context-460  | pascal-context-60  | N/A                          |
|   Testing    |scannet-41          | scannet-20         | N/A                          |
|   Training   |sunrgbd-38          | sunrgbd-37         | sunrgbd-37-relabeled         |
|   Testing    |voc2012             | voc2012            | N/A                          |
|   Testing    |wilddash-34         | wilddash-19        | N/A                          |

Note that testing datasets are not relabeled. Not that training is performed on the training dataset, using their `relabeled name`.

*Note: `coco-panoptic-201` is not provided in semantic form (class ids) by the authors, but rather can be created by using a combination of their panoptic JSON data and instance ID images (paths in `coco-panoptic-inst-201`).