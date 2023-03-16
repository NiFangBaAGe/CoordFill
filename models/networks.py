import torch.nn as nn
from torch.nn import init
import torch.nn.utils.spectral_norm as spectral_norm
import torch
import torch.nn.functional as F
import functools
import numpy as np


class MySeparableBilinearDownsample(torch.nn.Module):
    def __init__(self, stride, channels, use_gpu):
        super().__init__()
        self.stride = stride
        self.channels = channels

        # create tent kernel
        kernel = np.arange(1,2*stride+1,2) # ramp up
        kernel = np.concatenate((kernel,kernel[::-1])) # reflect it and concatenate
        if use_gpu:
            kernel = torch.Tensor(kernel/np.sum(kernel)).to(device='cuda') # normalize
        else:
            kernel = torch.Tensor(kernel / np.sum(kernel))
        self.register_buffer('kernel_horz', kernel[None,None,None,:].repeat((self.channels,1,1,1)))
        self.register_buffer('kernel_vert', kernel[None,None,:,None].repeat((self.channels,1,1,1)))

        self.refl = nn.ReflectionPad2d(int(stride/2))#nn.ReflectionPad2d(int(stride/2))

    def forward(self, input):
        return F.conv2d(F.conv2d(self.refl(input), self.kernel_horz, stride=(1,self.stride), groups=self.channels),
                    self.kernel_vert, stride=(self.stride,1), groups=self.channels)


class ASAPNetsBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(), kernel_size=3, reflection_pad=False, replicate_pad=False):
        super().__init__()
        padw = 1
        if reflection_pad:
            self.conv_block = nn.Sequential(nn.ReflectionPad2d(padw),
                                            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=0)),
                                            activation
                                            )
        elif replicate_pad:
            self.conv_block = nn.Sequential(nn.ReplicationPad2d(padw),
                                            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=0)),
                                            activation
                                            )
        else:
            self.conv_block = nn.Sequential(norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padw)),
                                            activation
                                            )

    def forward(self, x):
        out = self.conv_block(x)
        return out


def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]
        else:
            subnorm_type = norm_type

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'spectral':
            norm_layer = torch.nn.utils.spectral_norm(get_out_channel(layer))
        # elif subnorm_type == 'sync_batch':
        #     norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        elif subnorm_type == 'instanceaffine':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=True)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params = num_params + param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            '''
            for name, param in m.named_parameters():
                if (name == "lowres_stream.params_pred.weight"):
                    print("%s_init" % name)
                    init.zeros_(param.data[0:13])
                    init.normal_(param.data[13:13 + 64 * 64], 0.0, 0.02)
                    for i in range(1,6):
                        init.zeros_(param.data[13+i*64*64+(i-1)*64:13+64*64+i*64])
                        init.normal_(param.data[13+i*64*64+i*64:13+i*64+(i+1)*64*64], 0.0, 0.02)
                    init.zeros_(param.data[13 + i * 64 * 64 + (i - 1) * 64:13 + 64 * 64 + i * 64 + 3])
                    init.normal_(param.data[13 + i * 64 * 64 + i * 64 + 3 :13 + i * 64 + i * 64 * 64 +64*3], 0.0, 0.02)
                if (name == "lowres_stream.params_pred.bias"):
                    print("%s_init" % name)
                    init.zeros_(param.data)
            '''


        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)