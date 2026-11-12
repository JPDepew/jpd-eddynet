from glob import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ResponseMapDataset(Dataset):
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.resp_map_dirs = list(glob(f"{root_dir}/*MHz*"))

    def __len__(self):
        return len(glob(f"{self.resp_map_dirs[0]}/*.npz"))

    def __getitem__(self, idx: int):
        resp_maps = np.zeros(
            shape=(len(self.resp_map_dirs) * 2, 40, 40), dtype=np.float64
        )
        i = 0
        for dir in self.resp_map_dirs:
            resp_map = np.load(f"{dir}/{idx:06d}.npz")["delta_z"]
            resp_maps[i] = resp_map.real
            resp_maps[i + 1] = resp_map.imag

            i += 2

        crack = torch.tensor(
            np.load(f"{self.root_dir}/cracks/{idx:06d}.npz")["crack"],
            dtype=torch.float32,
        )
        crack = torch.unsqueeze(crack, dim=0)

        return crack, torch.tensor(resp_maps, dtype=torch.float32)
