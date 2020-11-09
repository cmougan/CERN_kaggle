#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split


random.seed(42)
np.random.seed(42)

train_raw = pd.read_csv("data/train.csv")#.drop(columns="BUTTER")

X_full = train_raw.drop(columns="signal")
y_full = train_raw.signal

X_train, X_valid, y_train, y_valid = train_test_split(
    X_full,
    y_full,
    stratify=train_raw.signal
)

X_train["signal"] = y_train
X_valid["signal"] = y_valid

X_train.to_csv("data/train_split.csv", index=False)
X_valid.to_csv("data/valid_split.csv", index=False)
