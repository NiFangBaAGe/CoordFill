import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import BaseNetwork
from .networks import get_nonspade_norm_layer
from .networks import MySeparableBilinearDownsample as BilinearDownsample
import torch.nn.utils.spectral_norm as spectral_norm
import torch as th
from math import pi
from math import log2
import time
import math


class CoordFillGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(lr_instance=True)
        parser.set_defaults(no_instance_dist=True)
        parser.set_defaults(hr_coor="cosine")
        return parser

    def __init__(self, opt, hr_stream=None, lr_stream=None, fast=False):
        super(CoordFillGenerator, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        self.num_inputs = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if (opt.no_instance_edge & opt.no_instance_dist) else 1)
        self.lr_instance = opt.lr_instance
        self.learned_ds_factor = opt.learned_ds_factor #(S2 in sec. 3.2)
        self.gpu_ids = opt.gpu_ids

        self.downsampling = opt.crop_size // opt.ds_scale

        self.highres_stream = PixelQueryNet(self.downsampling, num_inputs=self.num_inputs,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth,
                                               no_one_hot=opt.no_one_hot, lr_instance=opt.lr_instance,
                                               **hr_stream)

        num_params = self.highres_stream.num_params
        self.lowres_stream = ParaGenNet(num_params, scale_injection=opt.scale_injection)

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def get_lowres(self, im):
        """Creates a lowres version of the input."""
        device = self.use_gpu()
        if(self.learned_ds_factor != self.downsampling):
            myds = BilinearDownsample(int(self.downsampling//self.learned_ds_factor), self.num_inputs,device)
            return myds(im)
        else:
            return im

    def forward(self, highres):
        lowres = self.get_lowres(highres)
        lr_features = self.lowres_stream(lowres)
        output = self.highres_stream(highres, lr_features)
        return output, lr_features#, lowres


def _get_coords(bs, h, w, device, ds):
    """Creates the position encoding for the pixel-wise MLPs"""
    x = th.arange(0, w).float()
    y = th.arange(0, h).float()
    scale = 7 / 8
    x_cos = th.remainder(x, ds).float() / ds
    x_sin = th.remainder(x, ds).float() / ds
    y_cos = th.remainder(y, ds).float() / ds
    y_sin = th.remainder(y, ds).float() / ds
    x_cos = x_cos / (max(x_cos) / scale)
    x_sin = x_sin / (max(x_sin) / scale)
    y_cos = x_cos / (max(y_cos) / scale)
    y_sin = x_cos / (max(y_sin) / scale)
    xcos = th.cos((2 * pi * x_cos).float())
    xsin = th.sin((2 * pi * x_sin).float())
    ycos = th.cos((2 * pi * y_cos).float())
    ysin = th.sin((2 * pi * y_sin).float())
    xcos = xcos.view(1, 1, 1, w).repeat(bs, 1, h, 1)
    xsin = xsin.view(1, 1, 1, w).repeat(bs, 1, h, 1)
    ycos = ycos.view(1, 1, h, 1).repeat(bs, 1, 1, w)
    ysin = ysin.view(1, 1, h, 1).repeat(bs, 1, 1, w)
    coords = th.cat([xcos, xsin, ycos, ysin], 1).to(device)

    return coords.to(device)



def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class ParaGenNet(th.nn.Module):
    """Convolutional LR stream to estimate the pixel-wise MLPs parameters"""
    def __init__(self, num_out, scale_injection=False):
        super(ParaGenNet, self).__init__()

        self.num_out = num_out
        self.scale_injection = scale_injection

        ngf = 64
        if self.scale_injection:
            self.out_para = nn.Sequential(
                th.nn.Linear(ngf * 8 + 1, self.num_out)
            )
        else:
            self.out_para = nn.Sequential(
                th.nn.Linear(ngf * 8, self.num_out)
            )

    def forward(self, model, x, x_hr):
        structure = model(x)
        if self.scale_injection:
            scale = (torch.ones(x_hr.size(0), 1, 1, 1) * (structure.size(3) / x_hr.size(3))) \
                .to(structure.device)
            scale = scale.repeat(1, structure.size(2), structure.size(3), 1)
            structure = torch.cat([structure.permute(0, 2, 3, 1), scale], dim=-1)
            para = self.out_para(structure).permute(0, 3, 1, 2)
        else:
            para = self.out_para(structure.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return para

    def mask_predict(self, model, x, x_hr, mask):
        structure = model(x)

        if self.scale_injection:
            scale = (torch.ones(x_hr.size(0), 1, 1, 1) * (structure.size(3) / x_hr.size(3))) \
                .to(structure.device)
            scale = scale.repeat(1, structure.size(2), structure.size(3), 1)
            structure = torch.cat([structure.permute(0, 2, 3, 1), scale], dim=-1)
        else:
            structure = structure.permute(0, 2, 3, 1)

        bs, h, w, c = structure.size()
        k = mask.size(2) // h
        mask = mask.unfold(2, k, k).unfold(3, k, k)
        mask = mask.permute(0, 2, 3, 4, 5, 1).contiguous().view(
            bs, h, w, int(k * k))
        lr_mask = torch.mean(mask, dim=-1).view(h * w)
        structure = structure.view(bs, h * w, c)
        index = torch.nonzero(1 - lr_mask).squeeze(1)
        structure = structure[:, index, :]
        para = self.out_para(structure).permute(0, 2, 1)
        return para, mask


class PixelQueryNet(th.nn.Module):
    """Addaptive pixel-wise MLPs"""
    def __init__(self, downsampling,
                 num_inputs=13, num_outputs=3, width=64, depth=5, coordinates="cosine",
                 no_one_hot=False, lr_instance=False):
        super(PixelQueryNet, self).__init__()

        self.lr_instance = lr_instance
        self.downsampling = downsampling
        self.num_inputs = num_inputs - (1 if self.lr_instance else 0)
        self.num_outputs = num_outputs
        self.width = width
        self.depth = depth
        self.coordinates = coordinates
        self.xy_coords = None
        self.no_one_hot = no_one_hot
        self.channels = []
        self._set_channels()

        self.num_params = 0
        self.splits = {}
        self._set_num_params()

    @property  # for backward compatibility
    def ds(self):
        return self.downsampling

    def _set_channels(self):
        """Compute and store the hr-stream layer dimensions."""
        in_ch = self.num_inputs
        in_ch = in_ch + int(4)
        self.channels = [in_ch]
        for _ in range(self.depth - 1):  # intermediate layer -> cste size
            self.channels.append(self.width)
        # output layer
        self.channels.append(self.num_outputs)

    def _set_num_params(self):
        nparams = 0
        self.splits = {
            "biases": [],
            "weights": [],
        }

        # # go over input/output channels for each layer
        idx = 0
        for layer, nci in enumerate(self.channels[:-1]):
            nco = self.channels[layer + 1]
            nparams = nparams + nco  # FC biases
            self.splits["biases"].append((idx, idx + nco))
            idx = idx + nco

            nparams = nparams + nci * nco  # FC weights
            self.splits["weights"].append((idx, idx + nco * nci))
            idx = idx + nco * nci

        self.num_params = nparams

    def _get_weight_indices(self, idx):
        return self.splits["weights"][idx]

    def _get_bias_indices(self, idx):
        return self.splits["biases"][idx]

    def forward(self, highres, lr_params):
        assert lr_params.shape[1] == self.num_params, "incorrect input params"

        if self.lr_instance:
            highres = highres[:, :-1, :, :]

        # Fetch sizes
        bs, _, h, w = highres.shape
        bs, _, h_lr, w_lr = lr_params.shape
        k = h // h_lr

        self.xy_coords = _get_coords(1, h, w, highres.device, h // h_lr)

        highres = torch.repeat_interleave(self.xy_coords, repeats=bs, dim=0)

        # Split input in tiles of size kxk according to the NN interp factor (the total downsampling factor),
        # with channels last (for matmul)
        # all pixels within a tile of kxk are processed by the same MLPs parameters
        nci = highres.shape[1]

        tiles = highres.unfold(2, k, k).unfold(3, k, k)
        tiles = tiles.permute(0, 2, 3, 4, 5, 1).contiguous().view(
            bs, h_lr, w_lr, int(k * k), nci)
        out = tiles
        num_layers = len(self.channels) - 1

        for idx, nci in enumerate(self.channels[:-1]):
            nco = self.channels[idx + 1]

            # Select params in lowres buffer
            bstart, bstop = self._get_bias_indices(idx)
            wstart, wstop = self._get_weight_indices(idx)

            w_ = lr_params[:, wstart:wstop]
            b_ = lr_params[:, bstart:bstop]

            w_ = w_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, nci, nco)
            b_ = b_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, 1, nco)

            out = th.matmul(out, w_) + b_

            # Apply RelU non-linearity in all but the last layer, and tanh in the last
            # out = th.nn.functional.leaky_relu(out, 0.01)
            if idx < num_layers - 1:
                out = th.nn.functional.leaky_relu(out, 0.01)
            else:
                out = torch.tanh(out)
        #
        # reorder the tiles in their correct position, and put channels first
        out = out.view(bs, h_lr, w_lr, k, k, self.num_outputs).permute(
            0, 5, 1, 3, 2, 4)
        out = out.contiguous().view(bs, self.num_outputs, h, w)

        return out

    def mask_predict(self, highres, lr_params, hr_mask, lr_mask):
        assert lr_params.shape[1] == self.num_params, "incorrect input params"

        if self.lr_instance:
            highres = highres[:, :-1, :, :]

        bs, _, h, w = highres.shape
        bs, h_lr, w_lr, _ = lr_mask.shape
        k = h // h_lr

        self.xy_coords = _get_coords(1, h, w, highres.device, h // h_lr)
        pe = torch.repeat_interleave(self.xy_coords, repeats=bs, dim=0)
        # Split input in tiles of size kxk according to the NN interp factor (the total downsampling factor),
        # with channels last (for matmul)
        # all pixels within a tile of kxk are processed by the same MLPs parameters
        nci = pe.shape[1]
        # bs, 5 rgbxy, h//k=h_lr, w//k=w_lr, k, k
        tiles = pe.unfold(2, k, k).unfold(3, k, k)
        tiles = tiles.permute(0, 2, 3, 4, 5, 1).contiguous().view(
            bs, h_lr, w_lr, int(k * k), nci)

        mask = torch.mean(lr_mask, dim=-1).view(h_lr * w_lr)
        index = torch.nonzero(1 - mask).squeeze(1)
        out = tiles
        num_layers = len(self.channels) - 1

        out = out.view(bs, h_lr * w_lr, int(k * k), nci)[:, index, :, :]
        num = out.size(1)

        for idx, nci in enumerate(self.channels[:-1]):
            nco = self.channels[idx + 1]

            # Select params in lowres buffer
            bstart, bstop = self._get_bias_indices(idx)
            wstart, wstop = self._get_weight_indices(idx)

            w_ = lr_params[:, wstart:wstop]
            b_ = lr_params[:, bstart:bstop]

            w_ = w_.permute(0, 2, 1).view(bs, num, nci, nco)
            b_ = b_.permute(0, 2, 1).view(bs, num, 1, nco)

            out = th.matmul(out, w_) + b_

            # Apply RelU non-linearity in all but the last layer, and tanh in the last
            if idx < num_layers - 1:
                out = th.nn.functional.leaky_relu(out, 0.01)
            else:
                out = torch.tanh(out)

        highres = highres.unfold(2, k, k).unfold(3, k, k)
        highres = highres.permute(0, 2, 3, 4, 5, 1).contiguous().view(
            bs, h_lr, w_lr, int(k * k), 3).view(bs, h_lr * w_lr, int(k * k), 3)

        highres[:, index, :, :] = out
        out = highres.view(bs, h_lr, w_lr, k, k, self.num_outputs).permute(
            0, 5, 1, 3, 2, 4)
        out = out.contiguous().view(bs, self.num_outputs, h, w)

        return out