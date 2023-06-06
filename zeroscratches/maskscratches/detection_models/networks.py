# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn

from zeroscratches.maskscratches import Downsample
from zeroscratches.maskscratches import DataParallelWithCallback


class UNet(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_channels=3,
            depth=5,
            conv_num=2,
            wf=6,
            padding=True,
            batch_norm=True,
            up_mode="upsample",
            with_tanh=False,
            sync_bn=True,
            antialiasing=True,
    ):
        """
            Implementation of
            U-Net: Convolutional Networks for Biomedical Image Segmentation
            (Ronneberger et al., 2015)
            https://arxiv.org/abs/1505.04597
            Using the default arguments will yield the exact version used
            in the original paper
            Args:
                in_channels (int): number of input channels
                out_channels (int): number of output channels
                depth (int): depth of the network
                wf (int): number of filters in the first layer is 2**wf
                padding (bool): if True, apply padding such that the input shape
                                is the same as the output.
                                This may introduce artifacts
                batch_norm (bool): Use BatchNorm after layers with an
                                   activation function
                up_mode (str): one of 'upconv' or 'upsample'.
                               'upconv' will use transposed convolutions for
                               learned upsampling.
                               'upsample' will use bilinear upsampling.
        """

        super().__init__()
        assert up_mode in ("upconv", "upsample")
        self.padding = padding
        self.depth = depth - 1
        prev_channels = in_channels

        self.first = nn.Sequential(
            *[nn.ReflectionPad2d(3), nn.Conv2d(in_channels, 2 ** wf, kernel_size=7), nn.LeakyReLU(0.2, True)]
        )
        prev_channels = 2 ** wf

        self.down_path = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        for i in range(depth):
            if antialiasing and depth > 0:
                self.down_sample.append(
                    nn.Sequential(
                        *[
                            nn.ReflectionPad2d(1),
                            nn.Conv2d(prev_channels, prev_channels, kernel_size=3, stride=1, padding=0),
                            nn.BatchNorm2d(prev_channels),
                            nn.LeakyReLU(0.2, True),
                            Downsample(channels=prev_channels, stride=2),
                        ]
                    )
                )
            else:
                self.down_sample.append(
                    nn.Sequential(
                        *[
                            nn.ReflectionPad2d(1),
                            nn.Conv2d(prev_channels, prev_channels, kernel_size=4, stride=2, padding=0),
                            nn.BatchNorm2d(prev_channels),
                            nn.LeakyReLU(0.2, True),
                        ]
                    )
                )
            self.down_path.append(
                UNetConvBlock(conv_num, prev_channels, 2 ** (wf + i + 1), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i + 1)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up_path.append(
                UNetUpBlock(conv_num, prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        if with_tanh:
            self.last = nn.Sequential(
                *[nn.ReflectionPad2d(1), nn.Conv2d(prev_channels, out_channels, kernel_size=3), nn.Tanh()]
            )
        else:
            self.last = nn.Sequential(
                *[nn.ReflectionPad2d(1), nn.Conv2d(prev_channels, out_channels, kernel_size=3)]
            )

        if sync_bn:
            self = DataParallelWithCallback(self)

    def forward(self, x):
        x = self.first(x)

        blocks = []
        for i, down_block in enumerate(self.down_path):
            blocks.append(x)
            x = self.down_sample[i](x)
            x = down_block(x)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, conv_num, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        for _ in range(conv_num):
            block.append(nn.ReflectionPad2d(padding=int(padding)))
            block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=0))
            if batch_norm:
                block.append(nn.BatchNorm2d(out_size))
            block.append(nn.LeakyReLU(0.2, True))
            in_size = out_size

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, conv_num, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == "upsample":
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2, align_corners=False),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_size, out_size, kernel_size=3, padding=0),
            )

        self.conv_block = UNetConvBlock(conv_num, in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


