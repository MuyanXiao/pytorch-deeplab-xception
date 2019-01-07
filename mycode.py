import argparse
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from tensorboardX import SummaryWriter
import os
import shutil
import time

# from test_vistas_single_gpu import load_snapshot
from segmentationModule import SegmentationModule
from dataloaders.mapillary.train_dataset import SegmentationDataset, segmentation_collate
from dataloaders.mapillary.transform import SegmentationTransform
from dataloaders.mapillary import config as config, utils as utils
from modeling.deeplab import *


nclass = 66
model = DeepLab(num_classes=nclass)

model = SegmentationModule(model, nclass)
model = model.cuda()
# print(model)

state_dict = torch.load("../deeplab-resnet.pth.tar")
model_dict = model.state_dict()

i = 0
for name in model_dict:
    print(i, name)

    if i not in [0, 678, 679]:
        model_dict[name] = state_dict["state_dict"][name[8:]]
    i+=1

model.load_state_dict(model_dict)