import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size)(
            transforms.ToPILImage()(img)))


def to_mask(mask):
    return transforms.ToTensor()(
        transforms.Grayscale(num_output_channels=1)(
            transforms.ToPILImage()(mask)))

import yaml
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--mask')
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--resolution')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))

    model = models.make(config['model']).cuda()
    model.encoder.load_state_dict(torch.load(args.model, map_location='cuda:0'))

    h, w = list(map(int, args.resolution.split(',')))

    mask = transforms.ToTensor()(Image.open(args.mask).convert('RGB'))

    img = resize_fn(img, (h, w))
    img = (img - 0.5) / 0.5
    mask = resize_fn(mask, (h, w))
    mask = to_mask(mask)
    mask[mask > 0] = 1
    mask = 1 - mask

    with torch.no_grad():
        pred = model.encoder.mask_predict([img.unsqueeze(0).cuda(), mask.unsqueeze(0).cuda()])

    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(3, h, w).cpu()
    transforms.ToPILImage()(pred).save(args.output)