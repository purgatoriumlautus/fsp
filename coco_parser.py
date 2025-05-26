from collections import defaultdict
import json
import numpy as np


class COCOParser:
    def __init__(self, anns_file, imgs_dir):
        with open(anns_file, 'r') as f:
            coco = json.load(f)

        self.annIm_dict = defaultdict(list)
        self.cat_dict = {}
        self.annId_dict = {}
        self.im_dict = {}
        self.licenses_dict = {}
        for ann in coco['annotations']:
            self.annIm_dict[ann['image_id']].append(ann)
            self.annId_dict[ann['id']] = ann
        for img in coco['images']:
            self.im_dict[img['id']] = img
        for cat in coco['categories']:
            self.cat_dict[cat['id']] = cat
        for license in coco['licenses']:
            self.licenses_dict[license['id']] = license

    def get_imgIds(self):
        """ Returns a list of all image IDs stored in the self.im_dict dictionary,
        which links image IDs to their corresponding image information.
        The image IDs are easily accessible by getting the keys of self.im_dict."""

        return list(self.im_dict.keys())

    def get_annIds(self, im_ids):
        """ The method get_annIds() takes im_ids, which is a list of image IDs, as an input parameter and returns the list of annotation IDs.
        The "annotations" components holds a list of dictionaries,
        each dictionary represents the data for an object within an image in the COCO dataset."""

        im_ids = im_ids if isinstance(im_ids, list) else [im_ids]
        return [ann['id'] for im_id in im_ids for ann in self.annIm_dict[im_id]]

    def load_anns(self, ann_ids):
        """The method get_annIds() takes im_ids, which is a list of image IDs,
        as an input parameter and returns the list of annotation IDs."""

        im_ids = ann_ids if isinstance(ann_ids, list) else [ann_ids]
        return [self.annId_dict[ann_id] for ann_id in ann_ids]

    def load_cats(self, class_ids):
        """The method load_cats() accepts a parameter class_ids,
        which may be either a single class ID or a list of class IDs,
        and returns the list of categories associated with the given class_ids."""

        class_ids = class_ids if isinstance(class_ids, list) else [class_ids]
        return [self.cat_dict[class_id] for class_id in class_ids]

    def get_imgLicenses(self, im_ids):
        """The method get_imgLicenses() receives a parameter im_ids,
        which can either be a single image ID or a list of image IDs,
        and returns a list of licenses corresponding to each image ID in the list im_ids."""
        im_ids = im_ids if isinstance(im_ids, list) else [im_ids]
        lic_ids = [self.im_dict[im_id]["license"] for im_id in im_ids]
        return [self.licenses_dict[lic_id] for lic_id in lic_ids]

coco_annotations_file="Solar Panel Fault Dataset.v1i.coco/train/_annotations.coco.json"
coco_images_dir="Solar Panel Fault Dataset.v1i.coco/train"
coco= COCOParser(coco_annotations_file, coco_images_dir)


