import argparse
import os
import shutil
import time
import sys
import sklearn
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models 

from PIL import Image

from AlexNet import *
from voc_dataset import *
from utils import *

import wandb
USE_WANDB = True # use flags, wandb is not convenient for debugging


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=30,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=256,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--eval-freq',
    default=2,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_true',
    help='use pre-trained model')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of distributed processes')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', action='store_true')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # TODO:
    # define loss function (criterion) and optimizer
    # also use an LR scheduler to decay LR by 10 every 30 epochs
    # you can also use PlateauLR scheduler, which usually works well
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=args.momentum,patience=30)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    # Data loading code
    
    #TODO: Create Datasets and Dataloaders using VOCDataset - Ensure that the sizes are as required
    # Also ensure that data directories are correct - the ones use for testing by TAs might be different
    # Resize the images to 512x512
    train_dataset = VOCDataset('trainval', image_size=512)
    val_dataset = VOCDataset('test', image_size=512)
    
    class_id_to_label = dict(enumerate(train_dataset.CLASS_NAMES))

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        # shuffle=(train_sampler is None),
        # collate_fn = train_dataset.voc_collate,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    
    # TODO: Create loggers for wandb - ideally, use flags since wandb makes it harder to debug code.
    if USE_WANDB:
        wandb.init(project="vlr-hw2")

    for epoch in range(args.start_epoch, args.epochs):
        #adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, class_id_to_label, model, criterion, optimizer, epoch)
        # scheduler.step()
        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            m1, m2 = validate(val_loader, model, criterion, epoch)

            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)




#TODO: You can add input arguments if you wish
def train(train_loader, class_id_to_label, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    #for i, (image,target,weight) in enumerate(train_loader):
    for i, (image,target,wgt) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # TODO: Get inputs from the data dict
        #image, target, wgt, gt_boxes, gt_classes = data['image'], data['label'], data['wgt'], data['gt_boxes'], data['gt_classes']
        target = target.squeeze()
        image, target, wgt = image.to('cuda'), target.to('cuda'), wgt.to('cuda')

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output such as clamping
        # TODO: Compute loss using ``criterion``
        imoutput = model(image).to('cuda')
        temp_output = imoutput
        imoutput = torch.squeeze(F.max_pool2d(imoutput, kernel_size=imoutput.size()[2:]))
        loss = criterion(imoutput,target)
                
        if (i == 1 or i == 2) and USE_WANDB and epoch == 0 or epoch == 14 or epoch == 29:
            img = torch.permute(image[0], (1,2,0))
            img = img.cpu().detach().numpy()
            img = (img - np.min(img))/(np.max(img) - np.min(img))
            #img.show()
            img = wandb.Image(img)
            wandb.log({"Image": img})
            for i in range(20):
                if target[0][i] == 1:
                    hm = torch.unsqueeze(temp_output[0][i], dim=0)
                    hm = transforms.Resize((512,512))(hm)
                    hm = torch.squeeze(torch.permute(hm, (1,2,0))) 
                    hm = hm.cpu().detach().numpy()
                    hm = np.array(hm * 255, dtype=np.uint8)
                    #hm.show()
                    hm = wandb.Image(hm)
                    wandb.log({"Heatmap for label " + str(i): hm})

        # measure metrics and record loss
        m1 = metric1(imoutput.data, target, wgt)
        m2 = metric2(imoutput.data, target, wgt)
        losses.update(loss.item(),n=image.size(0))
        avg_m1.update(m1,n=image.size(0))
        if m2 != -1:
            avg_m2.update(m2,n=image.size(0))

        # TODO:
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        #TODO: Visualize/log things as mentioned in handout
        #TODO: Visualize at appropriate intervals
        if USE_WANDB and i % args.print_freq == 0:
            wandb.log({'epoch': epoch, 'train/loss': loss})
            wandb.log({'epoch': epoch, 'train/metric1': m1})
            wandb.log({'epoch': epoch, 'train/metric2': m2})

        # End of train()


def validate(val_loader, model, criterion, epoch = 0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (image,target,wgt) in enumerate(val_loader):

            # TODO: Get inputs from the data dict
            #image, target, wgt = data['image'], data['label'], data['wgt']
            target = target.squeeze()
            image, target, wgt = image.to('cuda'), target.to('cuda'), wgt.to('cuda')


            # TODO: Get output from model
            # TODO: Perform any necessary functions on the output
            # TODO: Compute loss using ``criterion``
            imoutput = model(image).to('cuda')
            temp_output = imoutput
            imoutput = torch.squeeze(F.max_pool2d(imoutput, kernel_size=imoutput.size()[2:]))
            loss = criterion(imoutput,target)

            if (i == 50 or i == 100 or i == 150) and USE_WANDB:
                img = torch.permute(image[0], (1,2,0))
                img = img.cpu().detach().numpy()
                img = (img - np.min(img))/(np.max(img) - np.min(img))
                #img.show()
                img = wandb.Image(img)
                wandb.log({"Image": img})
                for i in range(20):
                    if target[0][i] == 1:
                        hm = torch.unsqueeze(temp_output[0][i], dim=0)
                        hm = transforms.Resize((512,512))(hm)
                        hm = torch.squeeze(torch.permute(hm, (1,2,0))) 
                        hm = hm.cpu().detach().numpy()
                        hm = np.array(hm * 255, dtype=np.uint8)
                        #hm.show()
                        hm = wandb.Image(hm)
                        wandb.log({"Heatmap for label " + str(i): hm})

            # measure metrics and record loss
            m1 = metric1(imoutput.data, target, wgt)
            m2 = metric2(imoutput.data, target, wgt)
            losses.update(loss,n=args.batch_size)
            avg_m1.update(m1,n=args.batch_size)
            if m2 != -1:
                avg_m2.update(m2,n=args.batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                    'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        avg_m1=avg_m1,
                        avg_m2=avg_m2))

            #TODO: Visualize things as mentioned in handout
            #TODO: Visualize at appropriate intervals
            if USE_WANDB and i % args.print_freq == 0:
                wandb.log({'epoch': epoch, 'valid/loss': loss})

    if epoch % 2 == 0:
        print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
            avg_m1=avg_m1, avg_m2=avg_m2))
        if USE_WANDB:
            wandb.log({'epoch': epoch, 'valid/metric1': m1})
            wandb.log({'epoch': epoch, 'valid/metric2': m2})

    return avg_m1.avg, avg_m2.avg


# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def metric1(output, target, wgt):
    # TODO: Ignore for now - proceed till instructed
    np.seterr(divide='ignore', invalid='ignore')
    sigmoid = torch.nn.Sigmoid()
    output = sigmoid(output)
    target, output, wgt = target.cpu().detach().numpy(),output.cpu().detach().numpy(),wgt.cpu().detach().numpy()
    nclasses = target.shape[1]
    AP = []
    for cid in range(nclasses):
        gt_cls = target[:, cid][wgt[:, cid] > 0].astype('float32')
        pred_cls = output[:, cid][wgt[:, cid] > 0].astype('float32')
        # As per PhilK. code:
        if np.count_nonzero(gt_cls) != 0:
            pred_cls -= 1e-5 * gt_cls
            ap = sklearn.metrics.average_precision_score(gt_cls, pred_cls)
            AP.append(ap)
    mAP = np.mean(AP)
    return mAP

def metric2(output, target, wgt):
    #TODO: Ignore for now - proceed till instructed
    np.seterr(divide='ignore', invalid='ignore')
    target, output = target.cpu().detach().numpy(),output.cpu().detach().numpy()
    no_gt = False
    for i in range(target.shape[0]):
        if np.count_nonzero(target[i]) == 0:
            no_gt = True
    if not no_gt:
        recall = sklearn.metrics.recall_score(target.astype('float32'), output.astype('float32') > 0.5, average="macro", zero_division=0)
    else:
        recall = -1
    return recall


if __name__ == '__main__':
    main()
