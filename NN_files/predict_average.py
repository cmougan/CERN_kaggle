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


def average_NN(
    data,
    weights=[
        "output/weights70.pt",
        "output/weights80.pt",
        "output/90.pt",
        "output/weights_final.pt",
    ],
):
    predicciones = []
    for w in weights:
        NNet = ResNet(data.shape[0]).to(device)
        NNet.load_state_dict(torch.load(w))
        predicciones.append(NNet.forward(X_train).detach().numpy())
    return np.mean(predicciones, axis=1)


def evaluate_auc(model, data, label, ensemble=False):
    if ensemble:
        pred = average_NN(data)
    else:
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
dataset = ReadDataset("train_split.csv")


# Train
dataset = ReadDataset("train_split.csv")
# Transform to tensor
X_train = torch.tensor(dataset.X.values).to(device).float()
y_train = torch.tensor(dataset.y.values).to(device).float()

# Predictions
print("Results in train AUC: ", evaluate_auc(nnet, X_train, y_train))
print("Results in train BCE: ", evaluate_log_loss(nnet, X_train, y_train))
print("Results in train AUC: ", evaluate_auc(nnet, X_train, y_train))
print("Results in train BCE: ", evaluate_log_loss(nnet, X_train, y_train))
train_out = average_NN(X_train)

# Store the predictions
train = pd.read_csv("train_split.csv", index_col="Id")
train["Predicted"] = train_out.detach().numpy()
file_name = "submissions/train_split_preds.csv"
train[["Predicted"]].to_csv(file_name)

# Validation
dataset = ReadDataset("validation.csv")
# Transform to tensor
X_val = torch.tensor(dataset.X.values).to(device).float()
y_val = torch.tensor(dataset.y.values).to(device).float()

# Predictions
print("Results in validation AUC: ", evaluate_auc(nnet, X_val, y_val))
print("Results in validation BCE: ", evaluate_log_loss(nnet, X_val, y_val))

val_out = nnet.forward(X_val)

# Store the predictions
validation = pd.read_csv("validation.csv", index_col="Id")
validation["Predicted"] = val_out.detach().numpy()
file_name = "submissions/validation_preds.csv"
validation[["Predicted"]].to_csv(file_name)

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
print("done")
