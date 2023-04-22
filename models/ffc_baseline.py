import torch.nn as nn
import torch.nn.functional as F
import torch
from scipy import ndimage
import numpy as np


class ResnetBlock_remove_IN(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=True):
        super(ResnetBlock_remove_IN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.ReLU(),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class MLPModel(nn.Module):
    """Convolutional LR stream to estimate the pixel-wise MLPs parameters"""
    def __init__(self):
        super(MLPModel, self).__init__()

        self.refine = FFCResNetGenerator(4, 3, ngf=64,
                                         n_downsampling=3, n_blocks=6, res_dilation=1, decode=False)
        self.mapping = nn.Conv2d(64 * 8, 64, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1),
        )

    def forward(self, x):
        bs, _, h, w = x.size()
        x = self.refine(x)
        x = self.mapping(x)
        x = F.interpolate(x, size=(h, w), mode='nearest')
        x = self.mlp(x)
        x = torch.tanh(x)
        return x

from .ffc import FFCResNetGenerator, FFCResnetBlock, ConcatTupleLayer, FFC_BN_ACT

class FFC(nn.Module):
    def __init__(self, args, name, mask_prediction=False):
        super(FFC, self).__init__()
        self.args = args
        self.n_channels = args.n_channels
        self.n_classes = args.n_classes
        self.out_dim = args.n_classes

        self.name = name
        self.mask_prediction = mask_prediction
        if self.name == 'ffc':
            self.refine = FFCResNetGenerator(4, 3, ngf=64, n_downsampling=3, n_blocks=6, res_dilation=1, decode=True)
        elif self.name == 'mlp':
            self.refine = MLPModel()

    def forward(self, inp):
        img, mask = inp

        hole = img * mask

        output = self.refine(torch.cat([hole, mask], dim=1))

        if self.mask_prediction:
            output = output * (1 - mask) + hole

        return output, output

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

device=torch.device('cuda')
# device=torch.device('cpu')
from models import register
from argparse import Namespace
@register('ffc')
def make_unet(n_channels=3, n_classes=3, no_upsampling=False):
    args = Namespace()

    args.n_channels = n_channels
    args.n_classes = n_classes

    args.no_upsampling = no_upsampling
    return FFC(args)
