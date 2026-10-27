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
        )


if __name__ == "__main__":
    id = "000000"
    root_dir = "/Users/johnpaul.depew/Documents/learning/eddynet/data"
    resp_maps = []
    for dir in glob(f"{root_dir}/*MHz*"):
        print(dir)
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

    conv_input = torch.tensor(resp_maps, dtype=torch.float32).view(1, 6, 40, 40)
    print("respmap shape", conv_input.shape)
    c_test = nn.Conv2d(
        in_channels=6,
        out_channels=320,
        kernel_size=6,
    )

    out = c_test(conv_input)
    print("out", out.shape)

    bn = nn.BatchNorm2d(320)

    print(bn(out).shape)
