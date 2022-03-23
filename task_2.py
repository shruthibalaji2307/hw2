from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime
import pickle as pkl

# imports
from wsddn import WSDDN
from voc_dataset import *
import wandb
from utils import nms, tensor_to_PIL
from PIL import Image, ImageDraw

from torchvision.ops import box_iou


# hyper-parameters
# ------------
start_step = 0
end_step = 20000
lr_decay_steps = {150000}
lr_decay = 1. / 10
rand_seed = 1024
epochs = 5

lr = 0.01
momentum = 0.9
weight_decay = 0.0005
# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load datasets and create dataloaders

train_dataset = VOCDataset('trainval', 512)
val_dataset = VOCDataset('test', 512)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,   # batchsize is one for this implementation
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    sampler=None,
    drop_last=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=True)


# Create network and initialize
net = WSDDN(classes=train_dataset.CLASS_NAMES)

if os.path.exists('pretrained_alexnet.pkl'):
    pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
else:
    pret_net = model_zoo.load_url(
        'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
    pkl.dump(pret_net, open('pretrained_alexnet.pkl', 'wb'),
             pkl.HIGHEST_PROTOCOL)
own_state = net.state_dict()

for name, param in pret_net.items():
    if name not in own_state:
        continue
    if isinstance(param, Parameter):
        param = param.data
    try:
        if name.find('classifier') == -1:
            own_state[name].copy_(param)
            print('Copied {}'.format(name))
            own_state[name].requires_grad=False
    except:
        print('Did not find {}'.format(name))
        continue

# Move model to GPU and set train mode
net.load_state_dict(own_state)
net.cuda()
net.train()

net.features[0].weight.requires_grad=False
net.features[0].bias.requires_grad=False
net.features[3].weight.requires_grad=False
net.features[3].bias.requires_grad=False
net.features[6].weight.requires_grad=False
net.features[6].bias.requires_grad=False
net.features[8].weight.requires_grad=False
net.features[8].bias.requires_grad=False
net.features[10].weight.requires_grad=False
net.features[10].bias.requires_grad=False

# TODO: Create optimizer for network parameters from conv2 onwards
# (do not optimize conv1)
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=momentum)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_steps, gamma=lr_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=momentum,patience=30)

output_dir = "./"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# training
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
val_interval = 1000
#val_interval=200

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


def test_net(model, val_loader=None, thresh=0.05):
    """
    tests the networks and visualize the detections
    thresh is the confidence threshold
    """
    with torch.no_grad():
        class_box_preds = {}
        class_score_preds = {}
        class_gt_boxes = {}
        for iter, data in enumerate(val_loader):
            if iter == 500:
                break
            print("Test image: ", iter)
            # one batch = data for one image
            image           = data['image']
            target          = data['label']
            wgt             = data['wgt']
            rois            = data['rois']
            gt_boxes        = data['gt_boxes']
            gt_class_list   = data['gt_classes']
            gt_box_class    = data['gt_box_class']
            
            image, target, wgt, rois = image.to('cuda'), target.to('cuda'), wgt.to('cuda'), rois.to('cuda')
            
            #wgt = wgt.cpu().detach().numpy()
            
            #TODO: perform forward pass, compute cls_probs
            cls_probs = model(image,rois=rois,gt_vec=target)
            print(cls_probs)
            
            # TODO: Iterate over each class (follow comments)
            for class_num in range(20):            
                # get valid rois and cls_scores based on thresh
                region_scores = cls_probs[:, class_num]
                score_mask = region_scores > thresh
                rois = rois.squeeze()
                valid_rois = rois[score_mask]
                valid_scores = region_scores[score_mask]
                # use NMS to get boxes and scores
                boxes, scores = nms(valid_rois, valid_scores, threshold=0.3)
                # boxes = boxes.cpu().numpy()
                # scores = scores.cpu().numpy()
                
                if class_num not in class_box_preds:
                    class_box_preds[class_num] = boxes
                    class_score_preds[class_num] = scores
                else:
                    torch.cat((class_box_preds[class_num],boxes),axis=0)
                    torch.cat((class_score_preds[class_num],scores),axis=0)
                    # np.append(class_box_preds[class_num],boxes,axis=0)
                    # np.append(class_score_preds[class_num],scores,axis=0)
                
                if class_num in gt_box_class and not wgt[:,class_num]:
                    gt_box = gt_box_class[class_num] 
                    #gt_box = gt_box.cpu().detach().numpy()
                    if class_num not in class_gt_boxes:
                        class_gt_boxes[class_num] = gt_box
                    else:
                        torch.cat((class_gt_boxes[class_num],gt_box),axis=0)
                        #np.append(class_gt_boxes[class_num], gt_box, axis=0)
                        
        #AP and mAP
        tp = 0
        fp = 0
        allgtboxes = 0
        for class_num in range(20):
            if class_num not in class_gt_boxes or class_gt_boxes[class_num].shape[0] == 0:
                fp += class_gt_boxes[class_num].shape[0]
            else:
                for i in range(class_gt_boxes[class_num].shape[0]):
                    allgtboxes += 1
                    for j in range(class_box_preds[class_num].shape[0]):
                        if box_iou(class_gt_boxes[class_num][i],class_box_preds[class_num][j]) > 0.5:
                            tp += 1
                            np.delete(class_gt_boxes[class_num], i, 0)
                            np.delete(class_box_preds[class_num], j, 0)
                        else:
                            fp += 1
            # precision = tp / (tp + fp)
            # recall = tp / (allgtboxes)
            # print(precision,recall)

            #TODO: visualize bounding box predictions when required
            #TODO: Calculate mAP on test set
            
# Training

disp_interval = 10
train_loss = AverageMeter()
for epoch in range(epochs):
    for iter, data in enumerate(train_loader):

        #TODO: get one batch and perform forward pass
        # one batch = data for one image
        image           = data['image']
        target          = data['label']
        wgt             = data['wgt']
        rois            = data['rois']
        roi_scores      = data['roi_scores']
        gt_boxes        = data['gt_boxes']
        gt_class_list   = data['gt_classes']
        
        image, target, wgt, rois = image.to('cuda'), target.to('cuda'), wgt.to('cuda'), rois.to('cuda')

        #TODO: perform forward pass - take care that proposal values should be in pixels for the fwd pass
        # also convert inputs to cuda if training on GPU
        cls_probs = net(image,rois=rois,gt_vec=target)
        
        # backward pass and update
        loss = net.loss  
        train_loss.update(loss.item(),n=image.size(0))
        step_cnt += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()
        
        if iter % disp_interval == 0:
            print("Iteration ", iter, ", Loss = ", train_loss.avg)

        #TODO: evaluate the model every N iterations (N defined in handout)
        
        if iter%val_interval == 0 and iter != 0:
            net.eval()
            ap = test_net(net, val_loader)
            print("AP ", ap)
            net.train()


        #TODO: Perform all visualizations here
        #The intervals for different things are defined in the handout
    scheduler.step()
