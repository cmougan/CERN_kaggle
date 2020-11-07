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
import time

from sklearn.metrics import roc_auc_score


def evaluate_auc(model, data, label):
    return roc_auc_score(label.detach().numpy(), model(data.float()).detach().numpy())


tic = time.time()

# Read data
train_file = "train_split.csv"
validation_file = "validation.csv"
trainset = ReadDataset(train_file)
testset = ReadDataset(validation_file)

# Data loaders
trainloader = DataLoader(trainset, batch_size=100, shuffle=True)
testloader = DataLoader(testset, batch_size=5_000, shuffle=False)


# Use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Neural Network
nnet = Net(trainset.__shape__()).to(device)

# Loss function
criterion = nn.BCELoss()

# Optimizer
optimizer = optim.Adam(
    nnet.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.000001
)


# Train the net
loss_per_iter = []
loss_per_batch = []

# Comparing training to test
dataiter = iter(testloader)
X_test, y_test = dataiter.next()
X_test = X_test.to(device)
y_test = y_test.to(device)


# Train the net
losses = []
auc_train = []
auc_test = []

# hyperparameteres
n_epochs = 1

for epoch in range(n_epochs):
    print(epoch)

    for i, (inputs, labels) in enumerate(trainloader):
        X = inputs.to(device)
        y = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forwarde
        outputs = nnet(X.float())

        # Compute diff
        loss = criterion(outputs, y.float())

        # Compute gradient
        loss.backward()

        # update weights
        optimizer.step()

        # Save loss to plot

        losses.append(loss.item())

        if i % 500 == 0:
            auc_train.append(evaluate_auc(nnet, X.float(), y.float()))
            auc_test.append(evaluate_auc(nnet, X_test, y_test))

            # Figure
            plt.figure()
            plt.ylim([0.6, 1])
            plt.plot(auc_test, label="test")
            plt.plot(auc_train, label="train")
            plt.legend()
            plt.savefig("output/auc_NN.png")
            plt.savefig("output/auc_NN.svg", format="svg")
            plt.close()

    if epoch % 10 == 0:
        path = "output/weights" + str(epoch) + ".pt"
        torch.save(nnet.state_dict(), path)

torch.save(nnet.state_dict(), "output/weights_final.pt")
toc = time.time()
elap_time = np.round(np.abs(tic - toc), 1)
print("Elapsed time: ", elap_time)
print("done")
