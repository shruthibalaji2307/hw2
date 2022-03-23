import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

import numpy as np

from torchvision.ops import roi_pool, roi_align


class WSDDN(nn.Module):
    n_classes = 20
    classes = np.asarray([
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ])

    def __init__(self, classes=None):
        super(WSDDN, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)

        #TODO: Define the WSDDN model
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(9216,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True)
        )

        self.score_fc   = nn.Linear(4096, self.n_classes)
        self.bbox_fc    = nn.Linear(4096, self.n_classes)
        
        # loss
        self.cross_entropy = None

    @property
    def loss(self):
        return self.cross_entropy

    def forward(self,
                image,
                rois=None,
                gt_vec=None,
                ):
        
        #TODO: Use image and rois as input
        # compute cls_prob which are N_roi X 20 scores
        out = self.features(image)
        
        # rois as Lx4 to roi_pool
        rois_new = np.zeros((rois.shape[1],4))
        rois_new[:,0] = rois[0,:,0].cpu().detach().numpy()
        rois_new[:,1] = rois[0,:,1].cpu().detach().numpy()
        rois_new[:,2] = rois[0,:,2].cpu().detach().numpy()
        rois_new[:,3] = rois[0,:,3].cpu().detach().numpy()
        rois_new = torch.tensor(rois_new)
        rois_new = torch.cat((torch.zeros((rois_new.shape[0], 1)),rois_new), 1).type(torch.cuda.FloatTensor)
        
        out = roi_pool(input=out, boxes=rois_new, output_size=(6,6), spatial_scale=(31/512))
        out = out.view(-1,9216)
        
        out = self.classifier(out) 
        classification_scores = F.softmax(self.score_fc(out),dim=1)
        detection_scores = F.softmax(self.bbox_fc(out),dim=0)
        cls_prob = classification_scores * detection_scores

        if self.training:
            label_vec = gt_vec.view(-1, self.n_classes)
            self.cross_entropy = self.build_loss(cls_prob, label_vec)
        
        return cls_prob

    
    def build_loss(self, cls_prob, label_vec):
        """Computes the loss

        :cls_prob: N_roix20 output scores
        :label_vec: 1x20 one hot label vector
        :returns: loss

        """
        #TODO: Compute the appropriate loss using the cls_prob that is the
        #output of forward()
        #Checkout forward() to see how it is called
        image_level_scores = torch.sum(cls_prob, dim=0, keepdim=True)
        image_level_scores = torch.clamp(image_level_scores, min=0.0, max=1.0)
        loss = torch.nn.BCEWithLogitsLoss(reduction="sum")
        loss = loss(image_level_scores,label_vec)
        return loss
