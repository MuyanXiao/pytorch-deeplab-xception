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

parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training - Mapillary")
parser.add_argument("--scales", metavar="LIST", type=str, default="[0.7, 1, 1.2]", help="List of scales")
parser.add_argument("--flip", action="store_true", help="Use horizontal flipping")
#parser.add_argument("--fusion-mode", metavar="NAME", type=str, choices=["mean", "voting", "max"], default="mean",
#                    help="How to fuse the outputs. Options: 'mean', 'voting', 'max'")
parser.add_argument('config', metavar='CONFIG_FILE',
                    help='path to configuration file')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument("--snapshot", default='', metavar="SNAPSHOT_FILE", type=str, help="Snapshot file to load")
parser.add_argument("input", metavar="IN_DIR", type=str, help="Path to images")
parser.add_argument("output", metavar="OUT_DIR", type=str, help="Path to save the trained models")
parser.add_argument('--log-dir', type=str, default='./train_logger/', metavar='PATH',
                    help='output directory for Tensorboard log')
parser.add_argument('--log-hist', action='store_true',
                    help='log histograms of the weights')

# deeplab v3+
parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

# cuda, seed and logging
parser.add_argument('--no-cuda', action='store_true', default=
                    False, help='disables CUDA training')
parser.add_argument('--gpu-ids', type=str, default='0',
                    help='use which gpu to train, must be a \
                    comma-separated list of integers only (default=0)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = None
conf = None
logger = None


def main():
    global args, conf, logger
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

    best_prec = 0
    logger = SummaryWriter(args.log_dir)

    # Torch stuff
    torch.cuda.set_device(0)
    cudnn.benchmark = True

    # Load configuration
    conf = config.load_config(args.config)

    # Create model
    nclass = 66
    model = DeepLab(num_classes=nclass,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=args.sync_bn,
                    freeze_bn=args.freeze_bn)

    model = SegmentationModule(model, nclass)
    model = model.cuda()
    print(model)

    # Data loading code
    traindir = os.path.join(args.input, 'training/')
    valdir = os.path.join(args.input, 'validation/')

    # train_loader, val_loader
    crop_h = 336
    crop_w = 448
    transformation = SegmentationTransform(
        2048,
        (0.41738699 + 0.45732192 + 0.46886091) / 3,
        (0.26509955)
    )
    trainset = SegmentationDataset(traindir, crop_h, crop_w, transformation)

    batch_size = 4
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        pin_memory=True,
        # sampler=DistributedSampler(trainset, 1, 0),
        num_workers=2,
        collate_fn=segmentation_collate,
        shuffle=True
    )
    print(traindir, len(train_loader))

    valset = SegmentationDataset(valdir, crop_h, crop_w, transformation)
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=DistributedSampler(valset, 1, 0),
        num_workers=2,
        collate_fn=segmentation_collate,
        shuffle=False
    )
    print(valdir, len(val_loader))

    # define loss function (criterion) and optimizer
    ignore_index = 65  # ignore the unlabeled class

    # set class weight to make the data more balanced
    # class_weight = np.ones(66)
    # class_weight[65] = 10  # unpaved road
    # class_weight[29] = 3  # grass
    # class_weight[15] = 3  # sidewalk
    # class_weight[11] = 3  # pedestrian area

    # class_weight = torch.from_numpy(class_weight)
    criterion = nn.NLLLoss(ignore_index=ignore_index, reduction='elementwise_mean').cuda()
                           # weight=class_weight.float()).cuda()  # optionally resume from a checkpoint
    optimizer, scheduler = utils.create_optimizer(conf["optimizer"], model)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        # model = load_model_weights(args.snapshot)
        args.start_epoch = 0

    nb_epoch = 10

    for epoch in range(nb_epoch):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, scheduler, epoch)

        # evaluate on validation set
        val_acc = val(val_loader, model, criterion, it=epoch * len(val_loader))

        # remember best prec and save checkpoint
        is_best = val_acc > best_prec
        # best_prec1 = max(val_acc, best_prec)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.output)


def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    global args, conf, logger
    # switch to train mode
    batch_time = AverageMeter()
    data_time = AverageMeter()
    accs = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    scales = eval(args.scales)
    for i, train_data in enumerate(train_loader):
        if conf["optimizer"]["schedule"]["mode"] == "step":
            scheduler.step(i + epoch * len(train_loader))

        inputIm = train_data["img"].cuda(non_blocking=True)
        target = train_data["annot"].cuda(non_blocking=True)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        # output = model(inputIm)
        output, _, pred = model(inputIm, scales, args.flip)
        # output = nn.functional.log_softmax(output, dim=1)
        # _, cls = output.max(1)
        loss = criterion(output, target)
        acc = accuracy(pred, target)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # measure accuracy and record loss
        losses.update(loss.detach(), inputIm.size(0))
        accs.update(acc.detach(), inputIm.size(0))
        if conf["optimizer"]["clip"] != 0.:
            nn.utils.clip_grad_norm(model.parameters(), conf["optimizer"]["clip"])
        optimizer.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.4f} ({acc.avg:.4f})'.format(epoch, i, len(train_loader), batch_time=batch_time,
                                                                  data_time=data_time, loss=losses, acc=accs))

        logger.add_scalar("train/loss", losses.val, i + epoch * len(train_loader))
        logger.add_scalar("train/lr", scheduler.get_lr()[0], i + epoch * len(train_loader))
        if args.log_hist and i % 10 == 0:
            for name, param in model.named_parameters():
                if name.find("fc") != -1 or name.find("bn_out") != -1:
                    logger.add_histogram(name, param.clone().cpu().data.numpy(), i + epoch * len(train_loader))


def val(val_loader, model, criterion, it=None):
    global logger
    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to evaluate mode
    model.eval()
    scales = eval(args.scales)

    end = time.time()
    for i, val_data in enumerate(val_loader):
        with torch.no_grad():
            inputIm = val_data["img"].cuda(non_blocking=True)
            target = val_data["annot"].cuda(non_blocking=True)

            # compute output
            # output = nn.functional.log_softmax(output, dim=1)
            # _, cls = output.max(1)
            output, _, pred = model(inputIm, scales, args.flip)
            loss = criterion(output, target)
            acc = accuracy(pred, target)

            # measure accuracy and record loss
            losses.update(loss.detach(), inputIm.size(0))
            accs.update(acc.detach(), inputIm.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {acc.val:.4f} ({acc.avg:.4f})'.format(i, len(val_loader), batch_time=batch_time,
                                                                      loss=losses, acc=accs))

    print(' * Accuracy {acc.avg:.3f} '.format(acc=accs))

    if it is not None:
        logger.add_scalar("val/loss", losses.avg, it)
        logger.add_scalar("val/acc", accs.avg, it)

    return accs.avg


def accuracy(output_cls, target):
    """Computes the precision ignoring the unlabeled class
    """
    # target = target.cpu()
    # output_cls = output_cls.cpu()
    n, h, w = target.size()
    ignore_class = torch.ones((n, h, w)) * 65
    ignore_class = ignore_class.cuda(non_blocking=True)
    ignore_index = target.eq(ignore_class.long())

    correct = output_cls[~ignore_index].eq(target[~ignore_index])
    acc = correct.float().sum() / (n * h * w - ignore_index.float().sum())

    return acc


# def load_model_weights(weight_file=None):
#     """Load a training snapshot"""
#     print("--- Creating model")
#     global args
#
#     # Create network
#     norm_act = partial(InPlaceABN, activation="leaky_relu", slope=.01)
#     first = models.__dict__["net_wider_resnet_first"](in_channel=1)
#     body = models.__dict__["net_wider_resnet38_a2"](norm_act=norm_act, dilation=(1, 2, 4, 4))
#     head = DeeplabV3(4096, 256, 256, norm_act=norm_act, pooling_size=(84, 84))
#
#     # freeze body and head layers
#     # for param in body.parameters():
#     #     param.requires_grad = False
#
#     # for param in head.parameters():
#     #     param.requires_grad = False
#
#     # freeze the first few layers for deepscene finetune
#     # for param in first.parameters():
#     #     param.requires_grad = False
#
#     # i = 0
#     # for param in body.parameters():
#     #    param.requires_grad = False
#     #    i += 1
#     #    if i>5: break
#
#     model = SegmentationModule(first, body, head, 256, 66, args.fusion_mode)
#
#     if weight_file is not None:
#         # Load snapshot and recover network state
#         print('Loading weight file')
#         data = torch.load(weight_file)
#         model.load_state_dict(data['state_dict'])
#
#     return model


# def load_snapshot(snapshot_file):
#     """Load a training snapshot"""
#     print("--- Loading model from snapshot")
#
#     # Create network
#     norm_act = partial(InPlaceABN, activation="leaky_relu", slope=.01)
#     first = models.__dict__["net_wider_resnet_first"](in_channel=1)
#     body = models.__dict__["net_wider_resnet38_a2"](norm_act=norm_act, dilation=(1, 2, 4, 4))
#     head = DeeplabV3(4096, 256, 256, norm_act=norm_act, pooling_size=(84, 84))

    # freeze body and head layers
    # for param in body.parameters():
    #     param.requires_grad = False
    #
    # for param in head.parameters():
    #     param.requires_grad = False

    # Load snapshot and recover network state
    # data = torch.load(snapshot_file)
    # body_state_dict = {k: data["state_dict"]["body"][k] for k in body.state_dict()}
    # body.load_state_dict(body_state_dict)
    # head.load_state_dict(data["state_dict"]["head"])
    #
    # return first, body, head, data["state_dict"]["cls"]


def save_checkpoint(state, is_best, outdir, filename='checkpoint.pth.tar'):
    torch.save(state, outdir+filename)
    if is_best:
        shutil.copyfile(outdir+filename, outdir+'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    main()
