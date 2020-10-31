#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from nnet import ReadDataset, Net
import os
import sys

from sklearn.metrics import roc_auc_score


def evaluate_auc(model, data, label):
    return roc_auc_score(label.detach().numpy(), model(data.float()).detach().numpy())


csv_file = os.path.join(sys.path[0], "test.csv")

# Read data
dataset = ReadDataset(csv_file)

# Split into training and test
#train_size = int(0.8 * len(dataset))
#test_size = len(dataset) - train_size
#trainset, testset = random_split(dataset, [train_size, test_size])


# Data loaders
#trainloader = DataLoader(trainset, batch_size=100, shuffle=True)
#testloader = DataLoader(testset, batch_size=5_000, shuffle=False)

test = Dataset(dataset)

# Use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Neural Network
nnet = Net(dataset.__shape__()).to(device)

nnet.load_state_dict(torch.load("output/weights.pt"))


nnet.predict(test)
