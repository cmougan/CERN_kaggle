#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import QuantileTransformer

from utils.features import feature_engineering
from fastai.tabular.all import *


random.seed(42)
np.random.seed(42)

train_raw = pd.read_csv("data/train.csv").drop(columns="BUTTER")
test_raw = pd.read_csv("data/test.csv").drop(columns="BUTTER")

train_raw['train'] = 1
test_raw['train'] = 0

all_df = pd.concat([train_raw, test_raw]).reset_index(drop=True)


all_df = feature_engineering(all_df)

transformed_values = QuantileTransformer().fit_transform(all_df)
transformed_df = pd.DataFrame(transformed_values)
# transformed_df = all_df.copy()

keep_cols = ["Id", "signal", "train"]
transformed_df.columns = [col if col in keep_cols else f"{col}_q" for col in all_df.columns]

transformed_df = transformed_df.drop(columns=keep_cols)

# full_df = pd.concat([all_df, transformed_df], axis=1)
full_df = pd.concat([all_df[keep_cols], transformed_df], axis=1)

train = full_df[full_df.train == 1].drop(columns=['train', 'Id'])
test = full_df[full_df.train != 1].drop(columns=['train', 'Id', 'signal'])

X_full = train.drop(columns="signal")
X_test = test.copy()
y_full = train.signal

X_train, X_valid, y_train, y_valid = train_test_split(
    X_full,
    y_full,
    stratify=train.signal
)

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

# lr_min, lr_steep = (learn.lr_find())
# print(lr_min)
# print(lr_steep)
n_cycles = 10

for i in range(n_cycles):
    print("-" * 20)
    print(f"Cycle {i}")
    learn.fit_one_cycle(20, lr_max=1e-3)

# Get valid predictions

valid_preds = learn.get_preds(dl=valid_dl)[0].numpy()

valid_scores = pd.DataFrame(dict(
    Id=valid_ids.values,
    prediction=valid_preds.ravel()
))

valid_scores.to_csv('data/blend/valid_fastai_nn.csv', index=False)

print(f"Validation AUC: {roc_auc_score(y_valid, valid_preds):.4f}")

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

for i in range(n_cycles):
    print(f"Cycle {i}")
    learn_full.fit_one_cycle(20, lr_max=1e-3)

test_dl = learn_full.dls.test_dl(X_test)
test_preds = learn_full.get_preds(dl=test_dl)[0].numpy()

test_raw['Predicted'] = test_preds

test_raw[['Id', 'Predicted']].to_csv('submissions/fastai_nn.csv', index=False)

