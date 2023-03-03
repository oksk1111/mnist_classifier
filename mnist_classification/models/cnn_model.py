# 1. model

import torch
import torch.nn as nn


class ConvolutionBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels

        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),  # 2d image를 사용하므로 BatchNorm2d 사용 가능
            # ? stride를 통해 출력의 크기가 줄었을 텐데, 왜 batchnorm2d에서 같은 out_channels를 쓸까?
            # -> stride는 w, h를 줄이는 것이지, # output channels(=# kernels)를 줄이는 것은 아니다!!
            nn.Conv2d(out_channels, out_channels, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        # |x| = (batch_size, in_channels, h, w); graysacle이면 in_channel=1, rgb이면 in_channel=3

        y = self.layers(x)
        # |y| = (batch_size, out_channels, h, w)

        return y


class ConvolutionClassifier(nn.Module):

    def __init__(self, output_size):
        self.output_size = output_size

        super().__init__()

        self.blocks = nn.Sequential(  # |x| = (n, 1, 28, 28)
            # (n, 32, 14, 14); stride에 의해 h, w가 절반으로 줄어듬
            ConvolutionBlock(1, 32),
            ConvolutionBlock(32, 64),  # (n, 64, 7, 7)
            ConvolutionBlock(64, 128),  # (n, 128, 4, 4)
            ConvolutionBlock(128, 256),  # (n, 256, 2, 2)
            ConvolutionBlock(256, 512),  # (n, 512, 1, 1)
        )
        self.layers = nn.Sequential(
            nn.Linear(512, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, output_size),
            nn.LogSoftmax(dim=-1),  # NLL Loss사용하므로
        )

    def forward(self, x):
        assert x.dim() > 2

        if x.dim() == 3:
            # |x| = (batch_size, h, w)
            x = x.view(-1, 1, x.size(-2), x.size(-1))  # (자동, 1, h, w)
        # |x| = (batch_size, 1, h, w)

        z = self.blocks(x)
        # |z| = (batch_size, 512, 1, 1)

        y = self.layers(z.squeeze())
        # |y| = (batch_size, output_size)

        return y
