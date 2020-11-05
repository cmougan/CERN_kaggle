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


# Use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

csv_file = "train.csv"

# Read data
dataset = ReadDataset(csv_file)
# Transform to tensor
X_train = torch.tensor(dataset.X.values).to(device).float()
y_train = torch.tensor(dataset.y.values).to(device).float()

# Read data
dataset = ReadDataset(csv_file, for_test=True)
# Transform to tensor
test = torch.tensor(dataset.X).to(device).float()

# Neural Network
nnet = Net(dataset.__shape__()).to(device)

nnet.load_state_dict(torch.load("output/weights0.pt"))

# Lets print the results in train
print("Results in train: ", evaluate_auc(nnet, X_train, y_train))
test_out = nnet.forward(test)

# Store the predictions
test = pd.read_csv("test.csv", index_col="Id")
test["Predicted"] = test_out.detach().numpy()

file_name = (
    "submissions/nn_" + str(date.today().month) + "/" + str(date.today().day) + ".csv"
)
test[["Predicted"]].to_csv(file_name)
