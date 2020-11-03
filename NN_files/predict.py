#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
plt.style.use("seaborn")
import pandas as pd

import torch

from nnet import ReadDataset, Net

from sklearn.metrics import roc_auc_score

def evaluate_auc(model, data, label):
    return roc_auc_score(label.detach().numpy(), model(data.float()).detach().numpy())

# Use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

csv_file = "train.csv"

# Read data
dataset = ReadDataset(csv_file, for_test=True)
# Transform to tensor
test = torch.tensor(dataset.X).to(device).float()

# Neural Network
nnet = Net(dataset.__shape__()).to(device)

nnet.load_state_dict(torch.load("output/weights0.pt"))


test_out = nnet.forward(test)

# Store the predictions
test = pd.read_csv("test.csv", index_col="Id")
test['Predicted'] = test_out
file
test[['Predicted']].to_csv('submissions/nn_1.csv')
