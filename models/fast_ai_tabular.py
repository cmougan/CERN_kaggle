#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer

from utils.features import feature_engineering, feature_engineering_cls
from fastai.tabular.all import *
from nnet import ReadDataset

random.seed(42)
np.random.seed(42)

keep_cols = ["Id", "signal", "train"]

trainset = ReadDataset('data/train.csv', gradient_boosting_features=True)
train = trainset.X
train['signal'] = trainset.y

testset = ReadDataset('data/valid_split.csv', gradient_boosting_features=True)
test = testset.X

print(train.shape)
print(test.shape)



dls = TabularDataLoaders.from_df(
    train,
    procs=[],
    cont_names=list(test.columns),
    y_names="signal",
    valid_idx=list(X_valid.index),
    bs=4096
)

learn = tabular_learner(
    dls,
    y_range=(0, 1),
    loss_func=F.binary_cross_entropy
)

# Prep train and valid indexes

valid_dl = learn.dls.test_dl(X_valid)
valid_ids = train_raw.iloc[X_valid.index, :].Id

train_dl = learn.dls.test_dl(X_train)
train_ids = train_raw.iloc[X_train.index, :].Id

n_cycles = 1
start_cycle = 4
n_epochs = 2

valid_preds = learn.get_preds(dl=valid_dl)[0].numpy() * 0

print('Tranning starting')
for i in range(n_cycles):
    print("-" * 20)
    print(f"Cycle {i}")
    learn.fit_one_cycle(n_epochs, lr_max=1e-3)

    if i >= start_cycle:

        # Get valid predictions
        valid_preds += learn.get_preds(dl=valid_dl)[0].numpy()

        print(f"Validation AUC: {roc_auc_score(y_valid, valid_preds):.4f}")

valid_preds = valid_preds / (n_cycles - start_cycle)

valid_scores = pd.DataFrame(dict(
    Id=valid_ids.values,
    prediction=valid_preds.ravel()
))

valid_scores.to_csv('data/blend/valid_fastai_nn.csv', index=False)

# Get train predictions

train_preds = learn.get_preds(dl=train_dl)[0].numpy()

train_scores = pd.DataFrame(dict(
    Id=train_ids.values,
    prediction=train_preds.ravel()
))

train_scores.to_csv('data/blend/train_fastai_nn.csv', index=False)

print(f"Train AUC: {roc_auc_score(y_train, train_preds):.4f}")


# Final train (all data)

dls_full = TabularDataLoaders.from_df(
    train,
    procs=[],
    cont_names=list(test.columns),
    y_names="signal",
    valid_idx=[0, 1],
    bs=4096
)

learn_full = tabular_learner(
    dls_full,
    y_range=(0, 1),
    loss_func=F.binary_cross_entropy
)

test_dl = learn_full.dls.test_dl(X_test)
test_preds = learn_full.get_preds(dl=test_dl)[0].numpy() * 0

for i in range(n_cycles):
    print(f"Cycle {i}")
    learn_full.fit_one_cycle(n_epochs, lr_max=1e-3)

    if i >= start_cycle:

        # Get valid predictions
        test_preds += learn_full.get_preds(dl=test_dl)[0].numpy()

test_preds = test_preds / (n_cycles - start_cycle)

test_raw['Predicted'] = test_preds

test_raw[['Id', 'Predicted']].to_csv('submissions/fastai_nn.csv', index=False)

