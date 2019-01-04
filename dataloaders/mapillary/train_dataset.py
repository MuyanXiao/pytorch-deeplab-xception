from itertools import chain

import glob
import torch
from PIL import Image
from os import path
from torch.utils.data import Dataset
import numpy as np
import math


class SegmentationDataset(Dataset):
    _EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]  # , "*.tif"

    def __init__(self, in_dir, crop_h, crop_w, transform):
        """
        Find all images in in_dir and prepare them as crop list
        :param in_dir: directory of the input data
        :param crop_size: size of the crop, single int value, square crop
        :param transform: data normalization
        """
        super(SegmentationDataset, self).__init__()

        self.im_dir = in_dir + 'grey/'
        self.annot_dir = in_dir + 'labels/'
        self.transform = transform
        self.crop_h = crop_h
        self.crop_w = crop_w

        self.images = []

        # Generate crop list
        for img_path in chain(*(glob.iglob(path.join(self.im_dir, ext)) for ext in SegmentationDataset._EXTENSIONS)):
            _, name_with_ext = path.split(img_path)
            idx, _ = path.splitext(name_with_ext)

            self.images.append({
                "idx": idx,
                "path": img_path,
                "annot_path": self.annot_dir + idx + '.png'
            })

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        # Load image and annotation
        img = Image.open(self.images[item]["path"])
        annot = Image.open(self.images[item]["annot_path"])

        # scale the image
        # deepscene images are much smaller than mapillary
        scale = 1024 / float(max(img.size[0], img.size[1]))
        if scale > 1.:
            out_size = tuple(int(dim * scale) for dim in img.size)
            img = img.resize(out_size, resample=Image.BILINEAR)
            annot = annot.resize(out_size, resample=Image.NEAREST)

        # random crop
        w, h = img.size

        if w<self.crop_w or h<self.crop_h:
            scale = self.crop_w/float(min(w, h))
            out_size = tuple(int(dim * scale) for dim in img.size)
            img = img.resize(out_size, resample=Image.BILINEAR)
            w, h = img.size

        top = np.random.randint(0, h - self.crop_h)
        left = np.random.randint(0, w - self.crop_w)

        img = img.crop((left, top, left+self.crop_w, top+self.crop_h))
        img = self.transform(img)

        annot = annot.crop((left, top, left+self.crop_w, top+self.crop_h))
        annot = self.transform(annot, normalize=False).long()

        return {"img": img, "annot": annot, "meta": {"idx": self.images[item]["idx"]}}


def segmentation_collate(items):
    imgs = torch.stack([item["img"] for item in items])
    annots = torch.stack([item["annot"] for item in items])
    metas = [item["meta"] for item in items]

    return {"img": imgs, "annot": annots, "meta": metas}