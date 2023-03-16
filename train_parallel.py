import argparse
import os


import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torchvision import transforms

import random
import datasets
import models
import utils
from test import eval_psnr, batched_predict
import numpy as np
from collections import OrderedDict

from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    # loader = DataLoader(dataset, batch_size=spec['batch_size'],
    #     shuffle=(tag == 'train'), num_workers=8, pin_memory=True)
    sampler = DistributedSampler(dataset, shuffle=(tag == 'train'))
    loader = DataLoader(dataset, batch_size=spec['batch_size'], sampler=sampler, num_workers=8, pin_memory=True)
    return loader

def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    # lr_scheduler = LambdaLR(optimizer, lr_lambda= lambda epoch: (1-(epoch/max_epoch))**0.9)
    lr_scheduler = None
    # log('model: #params={}'.format(utils.compute_num_params(model.encoder, text=True)))
    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler

def train(train_loader, model, optimizer):
    model.train()

    train_loss_G = utils.Averager()
    train_loss_D = utils.Averager()

    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt_rgb']
    gt_rgb_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    gt_rgb_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()

    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        gt_rgb = (batch['gt_rgb'] - gt_rgb_sub) / gt_rgb_div
        model.set_input(inp, gt_rgb, batch['mask'])

        model.optimize_parameters()

        train_loss_G.add(model.loss_G.item())
        # if model.discriminator != None:
        train_loss_D.add(model.loss_D.item())

    return train_loss_G.item(), train_loss_D.item()


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()


    n_gpus = 8
    if n_gpus > 1:
        # model = nn.parallel.DataParallel(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)  # device_ids will include all GPU devices by default

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # train_loss_G, train_loss_D = train(train_loader, model, optimizer)
        if n_gpus > 1:
            train_loss_G, train_loss_D = train(train_loader, model.module, optimizer)
        else:
            train_loss_G, train_loss_D = train(train_loader, model, optimizer)

        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train G: loss={:.4f}'.format(train_loss_G))
        writer.add_scalars('loss', {'train G': train_loss_G}, epoch)
        log_info.append('train D: loss={:.4f}'.format(train_loss_D))
        writer.add_scalars('loss', {'train D': train_loss_D}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()

        torch.save(model_.encoder.state_dict(), os.path.join(save_path, 'encoder-epoch-last.pth'))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            # if n_gpus > 1 and (config.get('eval_bsize') is not None):
            if n_gpus > 1:
                model_ = model.module
            else:
                model_ = model
            val_psnr, val_ssim, val_lpips = eval_psnr(val_loader, model_,
                data_norm=config['data_norm'],
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'))

            log_info.append('val: psnr={:.4f}'.format(val_psnr))
            writer.add_scalars('psnr', {'val': val_psnr}, epoch)
            log_info.append('val: ssim={:.4f}'.format(val_ssim))
            writer.add_scalars('ssim', {'val': val_ssim}, epoch)
            log_info.append('val: lpips={:.4f}'.format(val_lpips))
            writer.add_scalars('lpips', {'val': val_lpips}, epoch)
            if val_psnr > max_val_v:
                max_val_v = val_psnr
                torch.save(model_.encoder.state_dict(), os.path.join(save_path, 'encoder-epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--local_rank', default=0, type=int,
                        help='node rank for distributed training')
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl")

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name = save_name + '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path)
