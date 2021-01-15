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
from utils.background_modifier import Modbackground

import cv2
from torchvision import transforms
from dataloaders import custom_transforms as tr
from PIL import Image
from glob import glob
from sklearn.manifold import TSNE
import collections
from functools import partial
import matplotlib.pyplot as plt


class Denormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.demean = [-m / s for m, s in zip(mean, std)]
        self.std = std
        self.destd = [1 / s for s in std]
        self.inplace = inplace

    def __call__(self, tensor):
        tensor = transforms.Normalize(self.demean, self.destd, self.inplace)
        # clamp to get rid of numerical errors
        return torch.clamp(tensor, 0.0, 1.0)


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        # kwargs = {'num_workers': args.workers, 'pin_memory': True}
        kwargs = {'num_workers': 0, 'pin_memory': True}

        if args.nir:
            input_channels = 4
        else:
            input_channels = 3

        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        # Define network
        model = DeepLab(num_classes=3,
                        backbone=args.backbone,
                        in_channels=input_channels,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset + '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
                weight[1] = 4
                weight[2] = 2
                weight[0] = 1
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                # place_holder_target = target
                # place_holder_output = output
                self.summary.visualize_image(self.writer,
                                             self.args.dataset,
                                             image,
                                             target,
                                             output,
                                             global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def calculate_scores(self):
        train_loss = 0.0
        self.model.eval()
        gradient_embeddings = np.zeros((2, len(self.train_loader)))
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            # self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            score = self.model.module.decoder.conv1.weight.grad
            score = score.cpu().data.numpy()[:, :, 0, 0]
            tmp = TSNE(n_components=2).fit_transform(score)
            gradient_embeddings[:, i] = tmp
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr)

        self.writer.add_scalar('total_score', train_loss)
        print('[Epoch: %d, numImages: %5d]' % (i, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

    def pred_single_image(self, path, counter):
        self.model.eval()
        img_path = path
        lbl_path = os.path.join(os.path.split(os.path.split(path)[0])[0], 'lbl', os.path.split(path)[1])
        activations = collections.defaultdict(list)

        def save_activation(name, mod, input, output):
            activations[name].append(output.cpu())

        for name, m in self.model.named_modules():
            if type(m) == nn.ReLU:
                m.register_forward_hook(partial(save_activation, name))

        input = cv2.imread(path)
        # bkg = cv2.createBackgroundSubtractorMOG2()
        # back = bkg.apply(input)
        # cv2.imshow('back', back)
        # cv2.waitKey()
        input = cv2.resize(input, (513, 513), interpolation=cv2.INTER_CUBIC)
        image = Image.open(img_path).convert('RGB')  # width x height x 3
        # _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = np.array(Image.open(img_path), dtype=np.uint8)
        _tmp[_tmp == 255] = 1
        _tmp[_tmp == 0] = 0
        _tmp[_tmp == 128] = 2
        _tmp = Image.fromarray(_tmp)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=513),
            tr.Normalize(mean=mean, std=std),
            tr.ToTensor()])
        sample = {'image': image, 'label': _tmp}
        sample = composed_transforms(sample)

        image, target = sample['image'], sample['label']

        image = torch.unsqueeze(image, dim=0)
        if self.args.cuda:
            image, target = image.cuda(), target.cuda()
        with torch.no_grad():
            output = self.model(image)

        see = Analysis('module.decoder.last_conv.6', activations)
        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        pred = np.reshape(pred, (513, 513))
        # prediction = np.append(target, pred, axis=1)
        prediction = pred

        rgb = np.zeros((prediction.shape[0], prediction.shape[1], 3))

        r = prediction.copy()
        g = prediction.copy()
        b = prediction.copy()

        g[g != 1] = 0
        g[g == 1] = 255

        r[r != 2] = 0
        r[r == 2] = 255
        b = np.zeros(b.shape)

        rgb[:, :, 0] = b
        rgb[:, :, 1] = g
        rgb[:, :, 2] = r

        prediction = np.append(input, rgb.astype(np.uint8), axis=1)
        result = np.append(input, prediction.astype(np.uint8), axis=1)
        cv2.line(rgb, (513, 0), (513, 1020), (255, 255, 255), thickness=1)
        cv2.line(rgb, (513, 0), (513, 1020), (255, 255, 255), thickness=1)
        cv2.imwrite(
            '/home/robot/git/pytorch-deeplab-xception/run/cropweed/deeplab-resnet/experiment_41/samples/synthetic_{}.png'.format(
                counter), prediction)
        # plt.imshow(prediction)
        # cv2.waitKey()
        # plt.show()



def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='cropweed',
                        choices=['pascal', 'coco', 'cityscapes', 'cropweed'],
                        help='dataset name (default: cropweed)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')

    parser.add_argument('--active_selection', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')

    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=1296,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--uniform-scale', type=bool, default=False,
                        help='resized image to be 1:1 aspect ratio')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    parser.add_argument('--infer', action='store_true', default=False,
                        help='whether should execute inference with a trained model (default: False)')
    parser.add_argument('--nir', action='store_true', default=False,
                        help='whether should use nir channel or not (default: False)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
    False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

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

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
            'cropweed': 10,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
            'cropweed': 0.01,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-' + str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    if args.infer:
        # trainer.testing()
        src_files = glob('/home/robot/datasets/arox_samples/day1/weeds/*.jpg')  # AROX data
        # src_files = glob('/home/robot/datasets/structured_cwc/train/img/*.png')     # BonniRob data
        # files = glob('/home/robot/datasets/arox_samples/synthetic/*.png')     # Synthetic data
        print('[INFO] Found {} samples in the given dataset. '.format(len(src_files)))
        for file, index in zip(src_files, range(len(src_files))):
            print('[INFO] Processing sample number {} . . . '.format(index + 1))
            trainer.pred_single_image(file, index)
            # trainer.modify_background(src_files, target_files, None)
    else:
        trainer.calculate_scores()

    trainer.writer.close()


if __name__ == "__main__":
    main()
