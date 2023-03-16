import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register


@register('image-folder')
class ImageFolder(Dataset):
    def __init__(self, path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache=False):
        self.repeat = repeat
        self.cache = False

        if split_file is None:
            filenames = sorted(os.listdir(path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filepath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if self.cache:
                    self.files.append(
                        transforms.ToTensor()(Image.open(os.path.join(filepath, filename)).convert('RGB')))
                else:
                    self.files.append(os.path.join(filepath, filename))

        if first_k is not None:
            self.files = self.files[:first_k]

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        if self.cache:
            return x
        else:
            return transforms.ToTensor()(Image.open(x).convert('RGB'))


@register('paired-image-folders')
class PairedImageFolders(Dataset):
    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        idx1, idx2 = idx
        return self.dataset_1[idx1], self.dataset_2[idx2]

