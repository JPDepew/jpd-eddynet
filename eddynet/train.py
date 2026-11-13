import os

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from eddynet.dataset import ResponseMapDataset
from eddynet.model import EddyNet

EPOCHS = 100
BATCH_SIZE = 64
LR = 2e-4
DEVICE = "mps"


def train():
    try:
        os.mkdir("checkpoints")
    except Exception as e:
        print("Failed to make directory", e)
    resp_map_dset = ResponseMapDataset(
        "/Users/johnpaul.depew/Documents/learning/eddynet/data"
    )
    train_count = int(len(resp_map_dset) * 0.8)
    train_dset = Subset(resp_map_dset, range(train_count))
    test_dset = Subset(resp_map_dset, range(train_count, len(resp_map_dset)))

    train_dataloader = DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dset, batch_size=BATCH_SIZE)

    net = EddyNet().to(DEVICE)
    optimizer = optim.RAdam(net.parameters(), lr=LR)

    train_losses = []
    running_train_loss = []

    test_losses = []
    running_test_loss = []

    for epoch in range(EPOCHS):
        for y, profile in tqdm(train_dataloader, desc=f"Train epoch {epoch}"):
            y, profile = y.to(DEVICE), profile.to(DEVICE)

            optimizer.zero_grad()

            pred = net(profile)

            # mae
            loss = F.l1_loss(pred, y)
            loss.backward()

            optimizer.step()
            train_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)
        running_train_loss.append(train_loss)
        train_losses = []

        with torch.inference_mode():
            net.eval()
            for y, profile in tqdm(test_dataloader, desc=f"Val epoch {epoch}"):
                y, profile = y.to(DEVICE), profile.to(DEVICE)

                pred = net(profile)

                loss = F.l1_loss(pred, y)
                test_losses.append(loss.item())

            test_loss = sum(test_losses) / len(test_losses)
            running_test_loss.append(test_loss)
            test_losses = []

            net.train()

        print(f"epoch={epoch} | train_loss={train_loss} | test_loss={test_loss}")
        torch.save(net, f"checkpoints/eddynet_epoch={epoch}_loss={test_loss}.ckpt")

    plt.plot(running_train_loss)
    plt.plot(running_test_loss)
    plt.show()


if __name__ == "__main__":
    train()
