import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register

def to_mask(mask):
    return transforms.ToTensor()(
        transforms.Grayscale(num_output_channels=1)(
            transforms.ToPILImage()(mask)))


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size)(
            transforms.ToPILImage()(img)))


def get_coord(shape):
    ranges = None
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    return ret


@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[[idx, idx]]

        size = self.inp_size

        img = resize_fn(img, (size, size))
        mask = resize_fn(mask, (size, size))
        mask = to_mask(mask)
        mask[mask > 0] = 1
        mask = 1 - mask

        return {
            'inp': img,
            'gt_rgb': img,
            'mask': mask,
        }

@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment

        self.count = 0
        self.scale = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[[idx, idx]]

        size = self.size_max

        img = resize_fn(img, (size, size))
        mask = resize_fn(mask, (size, size))
        mask = to_mask(mask)
        mask[mask > 0] = 1
        mask = 1 - mask

        if self.augment:
            if random.random() < 0.5:
                img = img.flip(-1)
                mask = mask.flip(-1)

        return {
            'inp': img,
            'gt_rgb': img,
            'mask': mask,
        }
