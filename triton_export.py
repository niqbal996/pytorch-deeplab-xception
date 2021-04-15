import torch
from transformers import *
import argparse
import os
import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.analysis import Analysis

import cv2
from torchvision import transforms
from dataloaders import custom_transforms as tr
from PIL import Image
from glob import glob
import collections

class WrappedModel(torch.nn.Module):
    def __init__(self, args):
        super(WrappedModel, self).__init__()
        # self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased').cuda()
        self.model = DeepLab(num_classes=3,
                        backbone=args.backbone,
                        in_channels=3,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        # train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
        #                 {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        checkpoint = torch.load(args.model_path)
        args.start_epoch = checkpoint['epoch']
        if args.no_cpu:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.module.load_state_dict(checkpoint['state_dict'])
            # self.model.module.load_state_dict(checkpoint['state_dict'])
        self.best_pred = checkpoint['best_pred']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.model_path, checkpoint['epoch']))
    def forward(self, data):
        return self.model(data.cuda())



def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,help='network output stride (default: 8)')
    parser.add_argument('--sync-bn', type=bool, default=None,help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (default: auto)')
    parser.add_argument('--no-cpu', action='store_true', default=True, help='CPU inferencing')
    parser.add_argument('--gpu-ids', type=str, default='0', help='use which gpu to train, must be a \
                            comma-separated list of integers only (default=0)')
    parser.add_argument('--model-path', type=str, default=None, help='put the path to trained model .pth')

    args = parser.parse_args()

    if args.no_cpu:
        example = torch.zeros((1, 3, 512, 512), dtype=torch.float).cuda()  # bsz , seqlen
        pt_model = WrappedModel(args).eval().cuda()
    else:
        example = torch.zeros((1, 3, 512, 512), dtype=torch.float)  # bsz , seqlen
        pt_model = WrappedModel(args).eval()

    traced_script_module = torch.jit.trace(pt_model, example)
    traced_script_module.save("model.pt")


if __name__ == "__main__":
   main()