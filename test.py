import argparse
import os
import numpy as np
from tqdm import tqdm

from mypath import Path
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from PIL import Image, ImagePalette
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataloaders.mapillary.dataset import SegmentationDataset, segmentation_collate
from dataloaders.mapillary.transform import SegmentationTransform

from pytictoc import TicToc
import time


class Tester(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        # self.saver = Saver(args)
        # self.saver.save_experiment_config()
        # # Define Tensorboard Summary
        # self.summary = TensorboardSummary(self.saver.experiment_dir)
        # self.writer = self.summary.create_summary()

        # Define Dataloader
        transformation = SegmentationTransform(
            1024,
            (0.41738699, 0.45732192, 0.46886091),
            (0.25685097, 0.26509955, 0.29067996),
        )
        testset = SegmentationDataset(args.input, transformation)

        self.test_loader = DataLoader(
            testset,
            batch_size=1,
            pin_memory=True,
            sampler=DistributedSampler(testset, 1, 0),
            num_workers=2,
            collate_fn=segmentation_collate,
            shuffle=False
        )

        # Define network
        self.nclass = 21
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        self.model = model

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()
            print(model)

        # load model
        model_state = torch.load(self.args.premodel)
        self.model.module.load_state_dict(model_state['state_dict'])

    def testing(self):
        self.model.eval()
        t = TicToc()

        with torch.no_grad():
            for batch_i, rec in enumerate(self.test_loader):
                print("Testing batch [{:3d}/{:3d}]".format(batch_i + 1, len(self.test_loader)))

                img = rec["img"].cuda(non_blocking=True)
                img_name = rec["meta"][0]["idx"]
                t.tic()

                output = self.model(img)

                pred = output.data.cpu().numpy()
                print(pred[0, :, 1, 1])
                print(pred.shape)
                pred = np.argmax(pred, axis=1)
                pred_img = get_pred_image(pred[0])
                print(pred[0])

                pred_img.save(os.path.join(self.args.output, img_name+'.png'))
                return 0


_PALETTE = np.array([[165, 42, 42],
                     [0, 192, 0],
                     [196, 196, 196],
                     [190, 153, 153],
                     [180, 165, 180],
                     [90, 120, 150],
                     [102, 102, 156],
                     [128, 64, 255],
                     [140, 140, 200],
                     [170, 170, 170],
                     [250, 170, 160],
                     [96, 96, 96],
                     [230, 150, 140],
                     [128, 64, 128],
                     [110, 110, 110],
                     [244, 35, 232],
                     [150, 100, 100],
                     [70, 70, 70],
                     [150, 120, 90],
                     [220, 20, 60],
                     [255, 0, 0],
                     [255, 0, 100],
                     [255, 0, 200],
                     [200, 128, 128],
                     [255, 255, 255],
                     [64, 170, 64],
                     [230, 160, 50],
                     [70, 130, 180],
                     [190, 255, 255],
                     [152, 251, 152],
                     [107, 142, 35],
                     [0, 170, 30],
                     [255, 255, 128],
                     [250, 0, 30],
                     [100, 140, 180],
                     [220, 220, 220],
                     [220, 128, 128],
                     [222, 40, 40],
                     [100, 170, 30],
                     [40, 40, 40],
                     [33, 33, 33],
                     [100, 128, 160],
                     [142, 0, 0],
                     [70, 100, 150]], dtype=np.uint8)
_PALETTE = np.concatenate([_PALETTE, np.zeros((256 - _PALETTE.shape[0], 3), dtype=np.uint8)], axis=0)
_PALETTE = ImagePalette.ImagePalette(
    palette=list(_PALETTE[:, 0]) + list(_PALETTE[:, 1]) + list(_PALETTE[:, 2]), mode="RGB")


def get_pred_image(pred):
    img = Image.fromarray(pred.astype(np.uint8), mode='P')
    img.putpalette(_PALETTE)

    return img


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Testing")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')

    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # trained model
    parser.add_argument("premodel", metavar="MODEL_FILE", type=str, help="Pre-trained model file to load")

    # input path for testing images and output path for predicted labels
    parser.add_argument("input", metavar="IN_DIR", type=str, help="Path to input testing images")
    parser.add_argument("output", metavar="OUT_DIR", type=str, help="Path to output folder")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # if args.test_batch_size is None:
    args.test_batch_size = 1

    print(args)

    tester = Tester(args)

    tester.testing()


if __name__ == "__main__":
    main()
