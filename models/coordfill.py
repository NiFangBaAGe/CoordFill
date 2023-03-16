import torch.nn as nn
import torch.nn.functional as F
import torch
from scipy import ndimage
import numpy as np
from .ffc import FFCResNetGenerator
from .modules import CoordFillGenerator


from .ffc import FFCResNetGenerator, FFCResnetBlock, ConcatTupleLayer, FFC_BN_ACT
class AttFFC(nn.Module):
    """Convolutional LR stream to estimate the pixel-wise MLPs parameters"""
    def __init__(self, ngf):
        super(AttFFC, self).__init__()
        self.add = FFC_BN_ACT(ngf, ngf, kernel_size=3, stride=1, padding=1,
                           norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU,
                           **{"ratio_gin": 0.75, "ratio_gout": 0.75, "enable_lfu": False})
        self.minus = FFC_BN_ACT(ngf+1, ngf, kernel_size=3, stride=1, padding=1,
                           norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU,
                           **{"ratio_gin": 0, "ratio_gout": 0.75, "enable_lfu": False})
        self.mask = FFC_BN_ACT(ngf, 1, kernel_size=3, stride=1, padding=1,
                           norm_layer=nn.BatchNorm2d, activation_layer=nn.Sigmoid,
                           **{"ratio_gin": 0.75, "ratio_gout": 0, "enable_lfu": False})

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)

        mask, _ = self.mask((x_l, x_g))

        minus_l, minus_g = self.minus(torch.cat([x_l, x_g, mask], 1))

        add_l, add_g = self.add((x_l - minus_l, x_g - minus_g))

        x_l, x_g = x_l - minus_l + add_l, x_g - minus_g + add_g

        return x_l, x_g


class AttFFCResNetGenerator(nn.Module):
    """Convolutional LR stream to estimate the pixel-wise MLPs parameters"""
    def __init__(self, ngf):
        super(AttFFCResNetGenerator, self).__init__()

        self.dowm = nn.Sequential(
            nn.ReflectionPad2d(3),
            FFC_BN_ACT(4, 64, kernel_size=7, padding=0, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU,
                           **{"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False}),
            FFC_BN_ACT(64, 128, kernel_size=4, stride=2, padding=1,
                       norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU,
                       **{"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False}),
            FFC_BN_ACT(128, 256, kernel_size=4, stride=2, padding=1,
                       norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU,
                       **{"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False}),
            FFC_BN_ACT(256, 512, kernel_size=4, stride=2, padding=1,
                       norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU,
                       **{"ratio_gin": 0, "ratio_gout": 0.75, "enable_lfu": False}),
        )
        self.block1 = AttFFC(ngf)
        self.block2 = AttFFC(ngf)
        self.block3 = AttFFC(ngf)
        self.block4 = AttFFC(ngf)
        self.block5 = AttFFC(ngf)
        self.block6 = AttFFC(ngf)
        self.c = ConcatTupleLayer()

    def forward(self, x):
        x = self.dowm(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.c(x)

        return x



from .ffc_baseline import MLPModel
class CoordFill(nn.Module):
    def __init__(self, args, name, mask_prediction=False, attffc=False,
                 scale_injection=False):
        super(CoordFill, self).__init__()
        self.args = args
        self.n_channels = args.n_channels
        self.n_classes = args.n_classes
        self.out_dim = args.n_classes
        self.in_size = 256
        self.name = name
        self.mask_prediction = mask_prediction
        self.attffc = attffc
        self.scale_injection = scale_injection

        self.opt = self.get_opt()
        self.asap = CoordFillGenerator(self.opt)

        if self.name == 'ffc':
            self.refine = FFCResNetGenerator(4, 3, ngf=64, n_downsampling=3,
                                             n_blocks=6, res_dilation=1, decode=True)
        elif self.name == 'mlp':
            self.refine = MLPModel()
        elif self.name == 'coordfill':
            if self.attffc:
                self.refine = AttFFCResNetGenerator(512)
            else:
                self.refine = FFCResNetGenerator(4, 3, ngf=64, n_downsampling=3,
                                                 n_blocks=6, res_dilation=1, decode=False)

    def get_opt(self):
        from yacs.config import CfgNode as CN
        opt = CN()
        opt.label_nc = 0
        # opt.label_nc = 1
        opt.lr_instance = False
        opt.crop_size = 512
        opt.ds_scale = 32
        opt.aspect_ratio = 1.0
        opt.contain_dontcare_label = False
        opt.no_instance_edge = True
        opt.no_instance_dist = True
        opt.gpu_ids = 0
        opt.output_nc = 3
        opt.hr_width = 64
        opt.hr_depth = 5
        opt.scale_injection = self.scale_injection

        opt.no_one_hot = False
        opt.lr_instance = False
        opt.norm_G = 'batch'

        opt.lr_width = 256
        opt.lr_max_width = 256
        opt.lr_depth = 5
        opt.learned_ds_factor = 1
        opt.reflection_pad = False

        return opt

    def forward(self, inp):
        img, mask = inp
        hr_hole = img * mask

        lr_img = F.interpolate(img, size=(self.in_size, self.in_size), mode='bilinear')
        lr_mask = F.interpolate(mask, size=(self.in_size, self.in_size), mode='nearest')
        lr_hole = lr_img * lr_mask

        lr_features = self.asap.lowres_stream(self.refine, torch.cat([lr_hole, lr_mask], dim=1), hr_hole)

        output = self.asap.highres_stream(hr_hole, lr_features)

        if self.mask_prediction:
            output = output * (1 - mask) + hr_hole

        return output

    def mask_predict(self, inp):
        img, mask = inp
        hr_hole = img * mask

        lr_img = F.interpolate(img, size=(self.in_size, self.in_size), mode='bilinear')
        lr_mask = F.interpolate(mask, size=(self.in_size, self.in_size), mode='nearest')
        lr_hole = lr_img * lr_mask

        lr_features, temp_mask = self.asap.lowres_stream.mask_predict(self.refine, torch.cat([lr_hole, lr_mask], dim=1), hr_hole, mask)

        output = self.asap.highres_stream.mask_predict(hr_hole, lr_features, mask, temp_mask)
        output = output * (1 - mask) + hr_hole

        return output

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
@register('asap')
def make_unet(n_channels=3, n_classes=3, no_upsampling=False):
    args = Namespace()

    args.n_channels = n_channels
    args.n_classes = n_classes

    args.no_upsampling = no_upsampling
    return LPTN(args)