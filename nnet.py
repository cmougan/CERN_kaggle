import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


class ReadDataset(Dataset):
    """Students Performance dataset."""

    def __init__(self, csv_file):
        """Initializes instance of class StudentsPerformanceDataset.

        Args:
            csv_file (str): Path to the csv file with the students data.

        """
        self.df = pd.read_csv(csv_file).drop(columns="BUTTER")

        # Target
        self.target = "signal"

        # Save target and predictors
        self.X = self.df.drop(self.target, axis=1)
        self.y = self.df[self.target]

    def __len__(self):
        return len(self.df)

    def __shape__(self):
        return self.X.shape[1]

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X.iloc[idx].values, self.y[idx]]


class Net(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 2 * input_dim)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(2 * input_dim, 2 * output_dim)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(2 * input_dim, output_dim)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sig(x)

        return x.squeeze()
