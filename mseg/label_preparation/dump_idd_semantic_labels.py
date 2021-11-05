#!/usr/bin/python3

import argparse

import glob
import json
import os
from collections import namedtuple
from multiprocessing import Pool


import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
from tqdm import tqdm

import mseg.utils.names_utils as names_utils


"""
AutoNUE labels and classes to store, read, and write annotations.
In their hierarchy, we use 'id-type'='id' (the most fine-grained).
Uses PIllow to raster json polygons.

Ref: 
https://github.com/AutoNUE/public-code/blob/master/preperation/createLabels.py
"""

name2labelid = names_utils.get_classname_to_dataloaderid_map("idd-40", include_ignore_idx_cls=False)
id2label = names_utils.get_dataloader_id_to_classname_map("idd-40", include_ignore_idx_cls=False)


# A point in a polygon
Point = namedtuple("Point", ["x", "y"])

# Class that contains the information of a single annotated object
class CsObject:
    # Constructor
    def __init__(self):
        # the label
        self.label = ""
        # the polygon as list of points
        self.polygon = []
        # the object ID
        self.id = -1
        # If deleted or not
        self.deleted = 0

    def __str__(self):
        polyText = ""
        if self.polygon:
            if len(self.polygon) <= 4:
                for p in self.polygon:
                    polyText += "({},{}) ".format(p.x, p.y)
            else:
                polyText += "({},{}) ({},{}) ... ({},{}) ({},{})".format(
                    self.polygon[0].x,
                    self.polygon[0].y,
                    self.polygon[1].x,
                    self.polygon[1].y,
                    self.polygon[-2].x,
                    self.polygon[-2].y,
                    self.polygon[-1].x,
                    self.polygon[-1].y,
                )
        else:
            polyText = "none"
        text = "Object: {} - {}".format(self.label, polyText)
        return text

    def fromJsonText(self, jsonText, objId):
        self.id = objId
        self.label = str(jsonText["label"])
        self.polygon = [Point(p[0], p[1]) for p in jsonText["polygon"]]
        if "deleted" in jsonText.keys():
            self.deleted = jsonText["deleted"]
        else:
            self.deleted = 0


# The annotation of a whole image
class Annotation:
    # Constructor
    def __init__(self, imageWidth=0, imageHeight=0):
        # the width of that image and thus of the label image
        self.imgWidth = imageWidth
        # the height of that image and thus of the label image
        self.imgHeight = imageHeight
        # the list of objects
        self.objects = []

    def fromJsonText(self, jsonText):
        jsonDict = json.loads(jsonText)
        self.imgWidth = int(jsonDict["imgWidth"])
        self.imgHeight = int(jsonDict["imgHeight"])
        self.objects = []
        for objId, objIn in enumerate(jsonDict["objects"]):
            obj = CsObject()
            obj.fromJsonText(objIn, objId)
            self.objects.append(obj)

    # Read a json formatted polygon file and return the annotation
    def fromJsonFile(self, jsonFile):
        if not os.path.isfile(jsonFile):
            print("Given json file not found: {}".format(jsonFile))
            return
        with open(jsonFile, "r") as f:
            jsonText = f.read()
            self.fromJsonText(jsonText)


def createLabelImage(json_fpath, annotation, outline=None):
    """
    # Convert the given annotation to a label image

        Args:
        -   json_fpath
        -   annotation

        Returns:

    """
    # the size of the image
    size = (annotation.imgWidth, annotation.imgHeight)

    # the background, using encoding = "id"
    background = name2labelid["unlabeled"]

    # this is the image that we want to create
    labelImg = Image.new("L", size, background)

    # a drawer to draw into the image
    drawer = ImageDraw.Draw(labelImg)

    # loop over all objects
    for obj in annotation.objects:
        classname = obj.label
        polygon = obj.polygon

        # If the object is deleted, skip it
        if obj.deleted or len(polygon) < 3:
            continue

        # If the label is not known, but ends with a 'group' (e.g. cargroup)
        # try to remove the s and see if that works
        if (not classname in name2labelid) and classname.endswith("group"):
            classname = classname[: -len("group")]

        if not classname in name2labelid:
            print(f"Label '{classname}' not known.")
            tqdm.write("Something wrong in: " + json_fpath)
            continue

        # If the ID is negative that polygon should not be drawn
        if name2labelid[classname] < 0:
            continue

        val = name2labelid[classname]
        try:
            if outline:
                drawer.polygon(polygon, fill=val, outline=outline)
            else:
                drawer.polygon(polygon, fill=val)
        except:
            print(f"Failed to draw polygon with label {classname}")
            raise

    return labelImg


def json2labelImg(json_fpath: str) -> None:
    """
    A method that does all the work. Worker completes this task
    repeatedly. A png label file will be written to png_fpath.

        Args:
        -   json_fpath: file path to the json file

        Returns:
        -   None
    """
    png_fpath = json_fpath.replace("_polygons.json", "_labelids.png")

    # do the conversion
    try:
        annotation = Annotation()
        annotation.fromJsonFile(json_fpath)
        labelImg = createLabelImage(json_fpath, annotation)
        labelImg.save(png_fpath)
    except:
        tqdm.write("Failed to convert: {}".format(fn))
        raise


def main(args):
    """
    Convert a json file path e.g.
        gtFine/train/1/210316_gtFine_polygons.json
    to:
        gtFine/train/1/210316_gtFine_labelids.png

    """
    # how to search for all ground truth
    wildcard = os.path.join(args.datadir, "gtFine", "*", "*", "*_gt*_polygons.json")

    # search files
    files = glob.glob(wildcard)
    files.sort()

    if not files:
        print("Did not find any files. Please consult the README.")

    tqdm.write(f"Processing {len(files)} annotation files for Semantic Segmentation")

    # iterate through files
    progress = 0
    tqdm.write("Progress: {:>3} %".format(progress * 100 / len(files)), end=" ")

    # single thread debug
    # for file in files:
    #     json2labelImg(file)

    pool = Pool(args.num_workers)
    results = list(tqdm(pool.imap(json2labelImg, files), total=len(files)))
    pool.close()
    pool.join()


if __name__ == "__main__":
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="")
    parser.add_argument("--num-workers", type=int, default=10)
    args = parser.parse_args()

    main(args)
