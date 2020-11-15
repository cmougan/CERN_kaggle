#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt

plt.style.use("seaborn")
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
import torch
from datetime import date
from nnet import ReadDataset, Net, ResNet

def evaluate_auc(model, data, label):

    pred = model(data.float()).detach().numpy()
    return roc_auc_score(label.detach().numpy(), pred)


def evaluate_log_loss(model, data, label, ensemble=False):
    if ensemble:
        pred = average_NN(data)
    else:
        pred = model(data.float()).detach().numpy()
    return log_loss(label.detach().numpy(), pred)


# Use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Neural Network
dataset = ReadDataset("data/train_split.csv")
nnet = ResNet(dataset.__shape__()).to(device)
weights_list = [
        "output/weights40.pt",
        "output/weights50.pt",
        "output/weights60.pt",
    ]


nnet40 = ResNet(dataset.__shape__()).to(device)
nnet40.load_state_dict(torch.load("output/weights40.pt"))

nnet50 = ResNet(dataset.__shape__()).to(device)
nnet50.load_state_dict(torch.load("output/weights50.pt"))

nnet60 = ResNet(dataset.__shape__()).to(device)
nnet60.load_state_dict(torch.load("output/weights60.pt"))

######################
####### Train ########
######################
'''
# Transform to tensor
X_train = torch.tensor(dataset.X.values).to(device).float()
y_train = torch.tensor(dataset.y.values).to(device).float()

# Predictions
print("Results in train AUC 40: ", evaluate_auc(nnet40, X_train, y_train))
print("Results in train AUC 50: ", evaluate_auc(nnet50, X_train, y_train))
print("Results in train AUC 60: ", evaluate_auc(nnet60, X_train, y_train))

# Store the predictions
train = pd.read_csv("data/train_split.csv", index_col="Id")

train["Predicted40"] = nnet40.forward(X_train).detach().numpy()
train["Predicted50"] = nnet50.forward(X_train).detach().numpy()
train["Predicted60"] = nnet60.forward(X_train).detach().numpy()

file_name = "submissions/train_split_preds_average.csv"
train[["Predicted40","Predicted50","Predicted60"]].to_csv(file_name)

########################
####### Validation #####
########################

dataset = ReadDataset("data/valid_split.csv")
# Transform to tensor
X_train = torch.tensor(dataset.X.values).to(device).float()
y_train = torch.tensor(dataset.y.values).to(device).float()

# Predictions
print("Results in validation AUC 40: ", evaluate_auc(nnet40, X_train, y_train))
print("Results in validation AUC 50: ", evaluate_auc(nnet50, X_train, y_train))
print("Results in validation AUC 60: ", evaluate_auc(nnet60, X_train, y_train))

# Store the predictions
train = pd.read_csv("data/valid_split.csv", index_col="Id")

train["Predicted40"] = nnet40.forward(X_train).detach().numpy()
train["Predicted50"] = nnet50.forward(X_train).detach().numpy()
train["Predicted60"] = nnet60.forward(X_train).detach().numpy()

file_name = "submissions/valid_split_preds_average.csv"
train[["Predicted40","Predicted50","Predicted60"]].to_csv(file_name)
'''

########################
#######   Test    #####
########################
test = ReadDataset("data/test.csv", for_test=True)
# Transform to tensor
test = torch.tensor(test.X.values).to(device).float()
test_out40 = nnet40.forward(test).detach().numpy()
test_out50 = nnet50.forward(test).detach().numpy()
test_out60 = nnet60.forward(test).detach().numpy()


# Store the predictions
test = pd.read_csv("data/test.csv", index_col="Id")
test["Predicted40"] = test_out40
test["Predicted50"] = test_out50
test["Predicted60"] = test_out60
file_name = (
    "submissions/nn_" + str(date.today().month) + "_" + str(date.today().day) + ".csv"
)
test[["Predicted40","Predicted50","Predicted60"]].to_csv(file_name)
print("done")
