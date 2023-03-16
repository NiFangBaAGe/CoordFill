import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

from PIL import Image
from torchvision import transforms
from torchsummary import summary
import numpy as np

def batched_predict(model, inp, coord, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred, preds


def tensor2PIL(tensor):
    # img = tensor.cpu().clone()
    # img = img.squeeze(0)
    # img = unloader(img)
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)

def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt_rgb']
    gt_rgb_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_rgb_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    # val_res = utils.Averager()
    val_psnr = utils.Averager()
    val_ssim = utils.Averager()
    val_l1 = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')


    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        gt = (batch['gt_rgb'] - gt_rgb_sub) / gt_rgb_div
        if eval_bsize is None:
            with torch.no_grad():
                # pred = model.encoder.mask_predict([inp, batch['mask']])
                pred = model.encoder([inp, batch['mask']])
        else:
            pred = batched_predict(model, inp, batch['coord'], eval_bsize)

        pred = (pred * (1 - batch['mask']) + gt * batch['mask']) * gt_rgb_div + gt_rgb_sub
        pred.clamp_(0, 1)

        if eval_type is not None: # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()

        psnr, ssim, l1 = metric_fn(model, pred, batch['gt_rgb'])

        val_psnr.add(psnr.item(), inp.shape[0])
        val_ssim.add(ssim.item(), inp.shape[0])
        val_l1.add(l1.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val psnr{:.4f}'.format(val_psnr.item()))
            pbar.set_description('val ssim{:.4f}'.format(val_ssim.item()))
            pbar.set_description('val lpips{:.4f}'.format(val_l1.item()))

    return val_psnr.item(), val_ssim.item(), val_l1.item()


from collections import OrderedDict
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)

    model = models.make(config['model']).cuda()
    model.encoder.load_state_dict(torch.load(args.model, map_location='cuda:0'))

    res = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True)

    print('result psnr: {:.6f}'.format(res[0]))
    print('result ssim: {:.6f}'.format(res[1]))
    print('result lpips: {:.6f}'.format(res[2]))
