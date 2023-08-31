# Code adapted from https://github.com/yang-song/score_sde/blob/main/models/wideresnet_noise_conditional.py
# As a pytorch version of the noise-conditional classifier
#   proposed in https://arxiv.org/abs/2011.13456, Appendix I.1

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


def activation(channels, apply_relu=True):
    gn = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels, eps=1e-5)
    if apply_relu:
        return nn.Sequential(gn, nn.ReLU(inplace=True))
    return gn


def _output_add(block_x, orig_x):
    """Add two tensors, padding them with zeros or pooling them if necessary.

    Args:
        block_x: Output of a resnet block.
        orig_x: Residual branch to add to the output of the resnet block.

    Returns:
        The sum of blocks_x and orig_x. If necessary, orig_x will be average pooled
            or zero padded so that its shape matches orig_x.
    """
    stride = orig_x.shape[-2] // block_x.shape[-2]
    strides = (stride, stride)
    if block_x.shape[1] != orig_x.shape[1]:
        orig_x = F.avg_pool2d(orig_x, strides, strides)
        channels_to_add = block_x.shape[1] - orig_x.shape[1]
        orig_x = F.pad(orig_x, (0, 0, 0, 0, 0, channels_to_add))
    return block_x + orig_x


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class WideResnetBlock(nn.Module):
    """Defines a single WideResnetBlock."""

    def __init__(self, in_planes, planes, time_channels, stride=1, activate_before_residual=False):
        super().__init__()
        self.activate_before_residual = activate_before_residual

        self.init_bn = activation(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_2 = activation(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        # Linear layer for embeddings
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, planes)
        )

    def forward(self, x, temb=None):
        if self.activate_before_residual:
            x = self.init_bn(x)
            orig_x = x
        else:
            orig_x = x
        
        block_x = x
        if not self.activate_before_residual:
            block_x = self.init_bn(block_x)

        block_x = self.conv1(block_x)
        if temb is not None:
            block_x += self.time_emb(temb)[:, :, None, None]

        block_x = self.bn_2(block_x)
        block_x = self.conv2(block_x)

        return _output_add(block_x, orig_x)


class WideResnetGroup(nn.Module):
    """Defines a WideResnetGroup."""

    def __init__(self, blocks_per_group, in_planes, planes, time_channels, stride=1, activate_before_residual=False):
        super().__init__()
        self.blocks_per_group = blocks_per_group

        self.blocks = nn.ModuleList()
        for i in range(self.blocks_per_group):
            if i == 0:
                blk = WideResnetBlock(in_planes, planes, time_channels, stride, activate_before_residual)
            else:
                blk = WideResnetBlock(planes, planes, time_channels, 1, False)
            self.blocks.append(blk)

    def forward(self, x, temb=None):
        for b in self.blocks:
            x = b(x, temb)
        return x


class WideResnet(nn.Module):
    """Defines the WideResnet Model."""

    def __init__(self, blocks_per_group, channel_multiplier, in_channels=3, num_classes=10):
        super().__init__()
        time_channels = 128 * 4
        self.time_emb = GaussianFourierProjection(embedding_size=time_channels // 4, scale=16)
        self.time_emb_mlp = nn.Sequential(
            nn.Linear(time_channels // 2, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )
        self.init_conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.group1 = WideResnetGroup(blocks_per_group,
                                      16, 16 * channel_multiplier,
                                      time_channels,
                                      activate_before_residual=True)
        self.group2 = WideResnetGroup(blocks_per_group,
                                      16 * channel_multiplier, 32 * channel_multiplier,
                                      time_channels,
                                      stride=2)
        self.group3 = WideResnetGroup(blocks_per_group,
                                      32 * channel_multiplier, 64 * channel_multiplier,
                                      time_channels,
                                      stride=2)
        self.pre_pool_bn = activation(64 * channel_multiplier)
        self.final_linear = nn.Linear(64 * channel_multiplier, num_classes)
        
        self.apply(_weights_init)

    def forward(self, x, t):
        # per image standardization
        N = np.prod(x.shape[1:])
        x = (x - x.mean(dim=(1,2,3), keepdim=True)) / torch.maximum(torch.std(x, dim=(1,2,3), keepdim=True), 1. / torch.tensor(np.sqrt(N)))

        temb = self.time_emb(t)
        temb = self.time_emb_mlp(temb)

        x = self.init_conv(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.pre_pool_bn(x)
        x = F.avg_pool2d(x, x.shape[-1])
        x = x.view(x.shape[0], -1)
        x = self.final_linear(x)
        return x


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


def wide_28_10_song(in_channels=3, num_classes=10):
    net = WideResnet(blocks_per_group=4, channel_multiplier=10, in_channels=in_channels, num_classes=num_classes)
    test(net)
    return net
