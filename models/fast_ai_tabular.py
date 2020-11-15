#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
# from sklearn.preprocessing import QuantileTransformer

from utils.features import feature_engineering, feature_engineering_cls, GaussRankScaler
from fastai.tabular.all import *


random.seed(42)
np.random.seed(42)

train_raw = pd.read_csv("data/train.csv").drop(columns="BUTTER")
test_raw = pd.read_csv("data/test.csv").drop(columns="BUTTER")

train_raw['train'] = 1
test_raw['train'] = 0

all_df = pd.concat([train_raw, test_raw]).reset_index(drop=True)

all_df.columns = [col.replace(" ", "") for col in all_df.columns]
all_df = feature_engineering_cls(all_df)
all_df = feature_engineering(all_df)

keep_cols = ["Id", "signal", "train"]
keep_df = all_df[keep_cols]
all_df = all_df.drop(columns=keep_cols)

transformed_values = GaussRankScaler(n_jobs=-1).fit_transform(all_df)
transformed_df = pd.DataFrame(transformed_values)
# transformed_df = all_df.copy()
print("Here!")

transformed_df.columns = [col if col in keep_cols else f"{col}_q" for col in all_df.columns]

# full_df = pd.concat([all_df, transformed_df], axis=1)
full_df = pd.concat([keep_df, transformed_df], axis=1)

train = full_df[full_df.train == 1]
test = full_df[full_df.train != 1]

X_full = train.drop(columns=['train', 'Id', 'signal'])
X_test = test.copy().drop(columns=['train', 'Id', 'signal'])
y_full = train.signal

# Carlos split's
train_ids = pd.read_csv('data/train_split.csv')['Id'].values
valid_ids = pd.read_csv('data/validation.csv')['Id'].values

X_train = X_full[train['Id'].isin(train_ids)]
y_train = y_full[train['Id'].isin(train_ids)]

X_valid = X_full[train['Id'].isin(valid_ids)]
y_valid = y_full[train['Id'].isin(valid_ids)]

train = train.drop(columns=['train', 'Id'])
test = test.drop(columns=['train', 'Id', 'signal'])



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

n_cycles = 10
start_cycle = 4
n_epochs = 20

valid_preds = learn.get_preds(dl=valid_dl)[0].numpy() * 0

for i in range(n_cycles):
    print("-" * 20)
    print(f"Cycle {i}")
    learn.fit_one_cycle(n_epochs, lr_max=1e-3)

    if i >= start_cycle:

        # Get valid predictions
        preds_single = learn.get_preds(dl=valid_dl)[0].numpy()
        valid_preds += preds_single

        print(f"Validation AUC: {roc_auc_score(y_valid, valid_preds):.4f}")
        print(f"Validation AUC single: {roc_auc_score(y_valid, preds_single):.4f}")


valid_preds_single = preds_single
valid_preds = valid_preds / (n_cycles - start_cycle)

valid_scores = pd.DataFrame(dict(
    Id=valid_ids.values,
    prediction=valid_preds.ravel()
))

valid_scores_single = pd.DataFrame(dict(
    Id=valid_ids.values,
    prediction=valid_preds_single.ravel()
))


valid_scores.to_csv('data/blend/valid_fastai_nn_grt.csv', index=False)
valid_scores_single.to_csv('data/blend/valid_fastai_nn_single_grt.csv', index=False)

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

test_preds_single = learn_full.get_preds(dl=test_dl)[0].numpy()
test_preds = test_preds / (n_cycles - start_cycle)

test_raw['Predicted'] = test_preds

test_raw[['Id', 'Predicted']].to_csv('submissions/fastai_nn_grt.csv', index=False)

test_raw['Predicted'] = test_preds_single

test_raw[['Id', 'Predicted']].to_csv('submissions/fastai_nn_grt_single.csv', index=False)