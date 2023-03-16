import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
import math
import numpy as np
from torch.autograd import Variable
import os
import logging
logger = logging.getLogger(__name__)
from .coordfill import CoordFill
from .ffc_baseline import FFC
from .adv_loss import AdversarialLoss
from collections import OrderedDict
from .LPIPS.models import dist_model as dm
import random

class D_Net(nn.Module):
    def __init__(self, in_channels=3, use_sigmoid=True, use_spectral_norm=True):
        super(D_Net, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4]


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module



@register('gan')
class GAN(nn.Module):
    def __init__(self, encoder_spec=None):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from argparse import Namespace
        args = Namespace()
        args.n_channels = 3
        args.n_classes = 3
        args.no_upsampling = True

        self.mode = encoder_spec['name']
        if encoder_spec['name'] == 'baseline':
            self.encoder = Baseline(args)
        elif encoder_spec['name'] == 'ffc' or encoder_spec['name'] == 'mlp':
            self.encoder = FFC(args, encoder_spec['name'], encoder_spec['mask_prediction'])
        else:
            self.encoder = CoordFill(args, encoder_spec['name'],
                                encoder_spec['mask_prediction'], encoder_spec['attffc'],
                                encoder_spec['scale_injection'])

        self.model_LPIPS = dm.DistModel()
        self.model_LPIPS.initialize(model='net-lin', net='alex', use_gpu=True)

        self.fm_loss = torch.nn.L1Loss()

        self.discriminator = D_Net(use_sigmoid=True)
        self.criterionGAN = AdversarialLoss('nsgan')

        self.lambda_D = 1
        self.lambda_perceptual = 10
        self.lambda_fm = 100

        self.multi_res_training = encoder_spec['multi_res_training']
        self.optimizer_G = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)

    def set_input(self, inp, gt, input_mask):
        self.input = inp.to(self.device)
        self.gt = gt.to(self.device)
        self.input_mask = input_mask.to(self.device)

        if self.multi_res_training:
            ratio = random.randint(0, 8)
            size = 256 + 32 * ratio
            self.input = F.interpolate(self.input, size=(size, size), mode='bilinear')
            self.gt = F.interpolate(self.gt, size=(size, size), mode='bilinear')
            self.input_mask = F.interpolate(self.input_mask, size=(size, size), mode='nearest')

    def forward(self):
        self.pred = self.encoder([self.input, self.input_mask])

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        pred_fake, _ = self.discriminator(self.pred.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False, True)

        pred_real, _ = self.discriminator(self.gt)
        self.loss_D_real = self.criterionGAN(pred_real, True, True)

        self.loss_D = self.loss_D_fake + self.loss_D_real

        # combine loss and calculate gradients
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        pred_fake, feat_fake = self.discriminator(self.pred)
        self.loss_GAN = self.criterionGAN(pred_fake, True, False) * self.lambda_D

        self.feat_match_loss = 0
        pred_real_hr, feat_real = self.discriminator(self.gt)
        for i in range(len(feat_fake)):
            self.feat_match_loss += self.fm_loss(feat_fake[i], feat_real[i].detach())
        self.feat_match_loss = self.feat_match_loss * self.lambda_fm

        self.loss_LPIPS, _ = self.model_LPIPS.forward_pair(self.pred, self.gt)
        self.loss_perceptual = torch.mean(self.loss_LPIPS) * self.lambda_perceptual

        self.loss_G = self.loss_perceptual + self.loss_GAN + self.feat_match_loss

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)

        self.set_requires_grad(self.discriminator, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

        self.set_requires_grad(self.discriminator, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
