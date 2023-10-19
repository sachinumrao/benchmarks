import os, time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm

DEVICE_FLAG = os.environ.get("DEVICE", 0)
DEBUG = os.environ.get("DEBUG", 0)

if DEVICE_FLAG == "mps":
    DEVICE_TYPE = "mps"
elif DEVICE_FLAG == "cpu":
    DEVICE_TYPE = "cpu"
else:
    DEVICE_TYPE = "cpu"
    print(f"device {DEVICE_FLAG} not found. Reverting to cpu...")


class BaseModel(nn.Module):
    def __init__(self, n_dim):
        super(BaseModel, self).__init__()

        self.n_dim = n_dim
        self.fc1 = nn.Linear(self.n_dim, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = nn.functional.relu(out)
        out = self.fc2(out)
        out = nn.functional.relu(out)
        out = self.fc3(out)
        return out


def get_dataset():
    print("Generating Dataset...")
    n_dims = 512
    n_samples = 1_000_000

    data_x = np.random.rand(n_samples, n_dims)
    data_y = np.random.randint(1, size=(n_samples,))

    if DEBUG == "1":
        print("Train Data Shape: ", data_x.shape)
        print("Test Data Shape: ", data_y.shape)

    return data_x, data_y


class TabularDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x.astype(np.float32)
        self.data_y = data_y.astype(np.float32)

    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.data_x[idx, :])
        y = torch.tensor([self.data_y[idx]])
        return x, y


def get_model():
    n_dim = 512
    model = BaseModel(n_dim)

    if DEBUG == "1":
        print("Model: ", model)

    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, dataloader):
    device = torch.device(DEVICE_TYPE)
    criterion = nn.CrossEntropyLoss()

    param_count = count_parameters(model)

    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    N = len(dataloader)

    print("Device: ", DEVICE_TYPE)
    print("Model Parameters: \n", param_count)
    print("Start Training...")

    model.train()

    t_start = time.perf_counter()
    for x, y in tqdm(dataloader, total=N):
        optimizer.zero_grad(set_to_none=True)

        x = x.to(device)
        y = y.to(device)

        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

    t_stop = time.perf_counter()
    print("Time Taken: ", t_stop - t_start)


def run_benchmark():
    data_x, data_y = get_dataset()

    # prepare dataset for training
    train_dataset = TabularDataset(data_x, data_y)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)

    # get model
    model = get_model()

    # run training
    train_one_epoch(model, train_dataloader)


def main():
    run_benchmark()


if __name__ == "__main__":
    main()
