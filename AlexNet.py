import numpy as np

import torch.utils.data as data
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.utils.model_zoo as model_zoo

import torchvision.models as models

from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
        'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class LocalizerAlexNet(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNet, self).__init__()
        #TODO: Define model
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11,11), stride=(4,4), padding=(2,2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), dilation=(1,1), ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5,5), stride=(1,1), padding=(2,2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), dilation=(1,1), ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(1,1), stride=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_classes, kernel_size=(1,1), stride=(1,1)),
        )


    def forward(self, x):
        #TODO: Define forward pass
        # Features
        x = self.features(x)
        x = self.classifier(x)
        return x


class LocalizerAlexNetRobust(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNetRobust, self).__init__()
        #TODO: Define model
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11,11), stride=(4,4), padding=(2,2)),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(3,3), stride=(2,2), ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5,5), stride=(1,1), padding=(2,2)),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(3,3), stride=(2,2), ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(1,1), stride=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_classes, kernel_size=(1,1), stride=(1,1)),
        )

    def forward(self, x):
        #TODO: Define fwd pass
        x = self.features(x)
        x = self.classifier(x)
        return x
    
def weights_init_xavier(m):
    # for every Linear layer in a model..
    nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.0)


def localizer_alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNet(**kwargs)
    #TODO: Initialize weights correctly based on whether it is pretrained or not
    if pretrained:
        own_state_dict = model.state_dict()
        state_dict = load_state_dict_from_url(model_urls["alexnet"], progress=True)
        for name, param in state_dict.items():
            if name not in own_state_dict:
                continue
            if isinstance(param, Parameter):
                param = param.data
            try:
                if name.find('classifier') == -1:
                    own_state_dict[name].copy_(param)
                    print('Copied {}'.format(name))
                    own_state_dict[name].requires_grad=False
            except:
                print('Did not find {}'.format(name))
                continue
        model.load_state_dict(own_state_dict)
    model.classifier[0].apply(weights_init_xavier)
    model.classifier[2].apply(weights_init_xavier)
    model.classifier[4].apply(weights_init_xavier)
    return model


def localizer_alexnet_robust(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNetRobust(**kwargs)
    #TODO: Ignore for now until instructed
    #TODO: Initialize weights correctly based on whether it is pretrained or not
    if pretrained:
        own_state_dict = model.state_dict()
        state_dict = load_state_dict_from_url(model_urls["alexnet"], progress=True)
        for name, param in state_dict.items():
            if name not in own_state_dict:
                continue
            if isinstance(param, Parameter):
                param = param.data
            try:
                if name.find('classifier') == -1:
                    own_state_dict[name].copy_(param)
                    print('Copied {}'.format(name))
                    own_state_dict[name].requires_grad=False
            except:
                print('Did not find {}'.format(name))
                continue
        model.load_state_dict(own_state_dict)
    model.classifier[0].apply(weights_init_xavier)
    model.classifier[2].apply(weights_init_xavier)
    model.classifier[4].apply(weights_init_xavier)
    return model