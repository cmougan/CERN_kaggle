#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt

plt.style.use("seaborn")
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from datetime import date
from nnet import ReadDataset, Net


def evaluate_auc(model, data, label):
    return roc_auc_score(label.detach().numpy(), model(data.float()).detach().numpy())


# Use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Neural Network
dataset = ReadDataset("train_split.csv")
nnet = Net(dataset.__shape__()).to(device)
nnet.load_state_dict(torch.load("output/weights0.pt"))
"""
# Train
dataset = ReadDataset('train_split.csv')
# Transform to tensor
X_train = torch.tensor(dataset.X.values).to(device).float()
y_train = torch.tensor(dataset.y.values).to(device).float()

# Predictions
print("Results in train: ", evaluate_auc(nnet, X_train, y_train))
train_out = nnet.forward(X_train)

# Store the predictions
train = pd.read_csv("train_split.csv", index_col="Id")
train["Predicted"] = train_out.detach().numpy()
file_name = (
    "submissions/train_split_preds.csv"
)
train[["Predicted"]].to_csv(file_name)

# Validation
dataset = ReadDataset('validation.csv')
# Transform to tensor
X_val = torch.tensor(dataset.X.values).to(device).float()
y_val = torch.tensor(dataset.y.values).to(device).float()

# Predictions
print("Results in validation: ", evaluate_auc(nnet, X_val, y_val))
val_out = nnet.forward(X_val)

# Store the predictions
validation = pd.read_csv("validation.csv", index_col="Id")
validation["Predicted"] = val_out.detach().numpy()
file_name = (
    "submissions/validation_preds.csv"
)
train[["Predicted"]].to_csv(file_name)
"""

# Test
test = ReadDataset("test.csv", for_test=True)
# Transform to tensor
test = torch.tensor(test.X.values).to(device).float()
test_out = nnet.forward(test)


# Store the predictions
test = pd.read_csv("test.csv", index_col="Id")
test["Predicted"] = test_out.detach().numpy()
file_name = (
    "submissions/nn_" + str(date.today().month) + "_" + str(date.today().day) + ".csv"
)
test[["Predicted"]].to_csv(file_name)
