import torch
from torch import nn


class EddyNet(torch.Module):
    def __init__(self, in_channels=6):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=320,
                kernel_size=6,
            ),
            nn.Conv2d(
                in_channels=320,
                out_channels=320,
                kernel_size=5,
                stride=2,
                padding=1,
            ),
            nn.Conv2d(
                in_channels=320,
                out_channels=320,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Conv2d(
                in_channels=320,
                out_channels=320,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Conv2d(
                in_channels=320,
                out_channels=128,
                kernel_size=4,
            ),
        )
