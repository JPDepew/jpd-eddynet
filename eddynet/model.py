from glob import glob

import numpy as np
import torch
from torch import nn


class EddyNet(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=320,
                kernel_size=6,
            ),
            nn.BatchNorm2d(320),
            nn.Mish(),
            nn.Conv2d(
                in_channels=320,
                out_channels=320,
                kernel_size=5,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(320),
            nn.Mish(),
            nn.Conv2d(
                in_channels=320,
                out_channels=320,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(320),
            nn.Mish(),
            nn.Conv2d(
                in_channels=320,
                out_channels=320,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(320),
            nn.Mish(),
            nn.Conv2d(
                in_channels=320,
                out_channels=128,
                kernel_size=4,
            ),
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=512,
                kernel_size=3,
            ),
            nn.BatchNorm2d(512),
            nn.Mish(),
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                stride=(2, 1),
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.Mish(),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.Mish(),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=(4, 3),
                stride=(2, 1),
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.Mish(),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=20,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, x):
        enc = self.enc(x)

        feat = self.dec(enc)
        weights = torch.softmax(feat, dim=1)
        feat = (weights * feat).sum(dim=1, keepdim=True)

        return feat


if __name__ == "__main__":
    id = "000000"
    root_dir = "/Users/johnpaul.depew/Documents/learning/eddynet/data"
    resp_maps = []
    for dir in glob(f"{root_dir}/*MHz*"):
        resp_map = (
            np.load(
                f"/Users/johnpaul.depew/Documents/learning/eddynet/data/2.0MHz/{id}.npz"
            )["delta_z"],
        )
        # print(resp_map[0].shape)
        real_part = np.real(resp_map)
        imag_part = np.imag(resp_map)
        resp_maps.append(real_part)
        resp_maps.append(imag_part)
    resp_maps_ndarray = np.array(resp_maps)
    conv_input = torch.tensor(resp_maps_ndarray, dtype=torch.float32).view(1, 6, 40, 40)
    # print("respmap shape", conv_input.shape)
    # c_test = nn.Conv2d(
    #     in_channels=6,
    #     out_channels=320,
    #     kernel_size=6,
    # )

    # out = c_test(conv_input)
    # print("out", out.shape)

    # bn = nn.BatchNorm2d(320)

    # print(bn(out).shape)

    eddynet = EddyNet()
    enc = eddynet.enc(conv_input)
    print(enc.shape)

    dec = eddynet.dec(enc)
    print(dec.shape)

    weights = torch.softmax(dec, dim=1)
    print("w", weights.shape)
    feat = (weights * dec).sum(dim=1, keepdim=True)
    print("feat", feat.shape)
