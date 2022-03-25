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
import sklearn
import sklearn.metrics
from wsddn import WSDDN
from voc_dataset import *
import wandb
from utils import *
from PIL import Image, ImageDraw

from torchvision.ops import box_iou


# hyper-parameters
# ------------
start_step = 0
end_step = 20000
lr_decay_steps = 150000
lr_decay = 1. / 10
rand_seed = 1024
epochs = 5

lr = 0.001
momentum = 0.6
weight_decay = 0.0005
USE_WANDB = True
# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)
    
if USE_WANDB:
        wandb.init(project="vlr-hw2")

# load datasets and create dataloaders

train_dataset = VOCDataset('trainval', 512, top_n=500)
val_dataset = VOCDataset('test', 512, top_n=500)

class_id_to_label = dict(enumerate(train_dataset.CLASS_NAMES))

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

# Set gradients to False and ignore these layers in optimizer
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
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=lr_decay, step_size=lr_decay_steps)

output_dir = "./"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# training
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
val_interval = 5000
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
        
def compute_map(output, target):
    # TODO: Ignore for now - proceed till instructed
    np.seterr(divide='ignore', invalid='ignore')
    #sigmoid = torch.nn.Sigmoid()
    #output = sigmoid(output)
    #target, output = target.cpu().detach().numpy(),output.cpu().detach().numpy()

    nclasses = target.shape[1]
    AP = []
    for cid in range(nclasses):
        gt_cls = target[:, cid].astype('float32')
        pred_cls = output[:, cid].astype('float32')
        # As per PhilK. code:
        if np.count_nonzero(gt_cls) != 0:
            pred_cls -= 1e-5 * gt_cls
            ap = sklearn.metrics.average_precision_score(gt_cls, pred_cls, average=None)
            AP.append(ap)
    print(AP)
    mAP = np.nanmean(AP)
    return mAP


def test_net(model, val_loader=None, thresh=0.02, epoch):
    """
    tests the networks and visualize the detections
    thresh is the confidence threshold
    """
    with torch.no_grad():
        ground_truth = np.zeros((len(val_loader.dataset),20))
        predictions = np.zeros((len(val_loader.dataset),20))
        selected_boxes = {}
        tp = 0
        fp = 0
        all_gt_boxes = 0
        for iter, data in enumerate(val_loader):
            selected_boxes[iter] = []
            if iter % 500 == 0:
                print("Validation iter: ", iter)
            current_gt = np.zeros(20)
            current_pred = np.zeros(20)
            # one batch = data for one image
            image           = data['image']
            target          = data['label']
            wgt             = data['wgt']
            rois            = data['rois']
            gt_boxes        = data['gt_boxes']
            gt_class_list   = data['gt_classes']
            gt_box_class    = data['gt_box_class']
            
            image, target, wgt, rois = image.to('cuda'), target.to('cuda'), wgt.to('cuda'), rois.to('cuda')
            
            all_gt_boxes += len(gt_boxes)
            
            # TODO: perform forward pass, compute cls_probs
            cls_probs = model(image,rois=rois,gt_vec=target)
            
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
                # print(boxes.shape)
                # print(boxes)
                if class_num in gt_class_list:
                    used_gt = set()
                    for box in boxes:
                        for ind,gt_box in enumerate(gt_boxes):
                            current_gt[class_num] = 1
                            
                            gt_box_coords = torch.zeros(4)
                            for i,coord in enumerate(gt_box):
                                gt_box_coords[i] = gt_box[i]
                                
                            if len(box.shape) == 1:
                                box = torch.unsqueeze(box,dim=0)
                            gt_box = torch.unsqueeze(gt_box_coords,dim=0).to('cuda')
                            
                            if ind not in used_gt and box_iou(box,gt_box) > 0.5:
                                current_pred[class_num] = 1
                                used_gt.add(ind)
                                box = torch.squeeze(box)
                                selected_boxes[iter].append((box,class_num)) 
            ground_truth[iter,:] = current_gt
            predictions[iter,:] = current_pred
            
            if iter % 250 == 0:
                if len(selected_boxes[iter]) > 0:
                    selected_boxes_tensor = selected_boxes[iter][0][0]
                    selected_boxes_class = torch.zeros(len(selected_boxes[iter]))
                    selected_boxes_class[0] = torch.tensor(selected_boxes[iter][0][1])
                    if len(selected_boxes[iter]) > 1:
                        for i in range(1,len(selected_boxes[iter])):
                            selected_boxes_tensor = torch.cat((selected_boxes_tensor,selected_boxes[iter][i][0]),dim=0)
                            selected_boxes_class[i] = torch.tensor(selected_boxes[iter][i][1])
                if len(selected_boxes[iter]) == 1:
                    selected_boxes_tensor = torch.unsqueeze(selected_boxes_tensor,dim=0)
                #nums = range(len(selected_boxes[iter]))
                img = wandb.Image(image, boxes={
                        "predictions": {
                            "box_data": get_box_data_q2(selected_boxes_class, selected_boxes_tensor),
                            "class_labels": class_id_to_label,
                            },
                        }, caption = "Epoch: " + str(epoch))
                wandb.log({"proposals": img})

        #TODO: visualize bounding box predictions when required
        #TODO: Calculate mAP on test set
        print(ground_truth.shape,predictions.shape)
        mAP = compute_map(ground_truth,predictions)
        print(mAP)
    return mAP
# Training

def save_checkpoint(state, filename='task_2_second.pth.tar'):
    print("Saving trained model to task_2_second.pth")
    torch.save(state, filename)
    
if os.path.exists('task_2_second.pth.tar'):
    checkpoint = torch.load('task_2_second.pth.tar')
    net.load_state_dict(checkpoint['state_dict'])
    print("Loaded trained model")
    net.eval()
    mAP = test_net(net, val_loader)
    print("mAP ", mAP)
    net.train()
else:
    disp_interval = 500
    train_loss = AverageMeter()
    for epoch in range(epochs):
        print("Epoch : ", epoch)
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
            optimizer.zero_grad()
            cls_probs = net(image,rois=rois,gt_vec=target)
            
            # backward pass and update
            loss = net.loss  
            train_loss.update(loss.item(),n=image.size(0))
            step_cnt += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if iter % disp_interval == 0:
                print("Epoch: ", epoch, " Iteration:", iter, " Loss: ", train_loss.avg)
                wandb.log({'epoch': epoch, 'train/loss': train_loss.avg})

            #TODO: evaluate the model every N iterations (N defined in handout)
            
        #if iter%val_interval == 0 and iter != 0 and epoch % 5 == 0:
        # if epoch == 4:
        net.eval()
        mAP = test_net(net, val_loader, 0.02, epoch)
        print("Epoch: ", epoch, " mAP: ", mAP)
        wandb.log({'epoch': epoch, 'valid/mAP': mAP})
        net.train()

        #TODO: Perform all visualizations here
        #The intervals for different things are defined in the handout
        scheduler.step()
        
    save_checkpoint({
    'epoch': epoch + 1,
    'state_dict': net.state_dict(),
    'optimizer': optimizer.state_dict(),
    })