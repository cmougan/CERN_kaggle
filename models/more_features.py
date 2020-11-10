#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import Pipeline


from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier

from utils.features import feature_engineering, DistanceDepthFeaturizer

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

positive_cols = [
    "Kplus_P_y_q", "Kplus_P_x_q", "Kplus_P_q",
    "B_PT_y_q", "B_PT_x_q", "B_PT_q"
]
constraint_list = [1 if col in positive_cols else 0 for col in X_train.columns]

# lgb = CatBoostClassifier(iterations=3000)
lgb = LGBMClassifier(
    n_estimators=1500,
    # monotone_constraint=constraint_list,
    n_jobs=-1,
    # monotone_constraints_method="intermediate"
)

dd = DistanceDepthFeaturizer(
    {
        "mom_eta": ["total_momentum_x2_q", "Kplus_ETA_q"],
        "mom_mom": ["total_momentum_x1_q", "total_momentum_x2_q"],
        "mom_mom_2": ["total_momentum_x1_q", "total_momentum_y2p_q"],
        # "mom_mom_3": ["total_momentum_x2p_q", "total_momentum_x2_q"],
        # "angle_mom": ["Kst_892_0_cosThetaH_q", "total_momentum_x1_q"],
    }
)

pipe = Pipeline(
    [
        ("dd", dd),
        ("lgb", lgb),
    ]
)


pipe.fit(X_train, y_train)

pred_valid = pipe.predict_proba(X_valid)[:, 1]
pred_train = pipe.predict_proba(X_train)[:, 1]


roc_valid = roc_auc_score(y_valid, pred_valid)

# Only quantiles, 500 trees: 865 (cv 864)
# Only quantiles + distance ratio, 500 trees: 868 (cv 863)
# Only quantiles + many features, 500 trees: 871 (cv 870)
# Only quantiles + many features, 1500 trees: 872 (cv 870)
# Only quantiles + many features + special distance ratios, 500 trees: 882 (cv 877)
# Only quantiles + many features + special distance ratios + b_px + b_py, 500 trees: 8816 (cv 8778)
# Only quantiles + many features + special distance ratios + b_px + b_py + hbars, 500 trees: 883 (cv 8772)
# Only quantiles + many features + special distance ratios + b_px + b_py + hbars + hbar ratios, 500 trees: 879 (cv 8787)
# Only quantiles + many features + special distance ratios + b_px + b_py + hbars + hbar ratios + sphere radius, 500 trees: 883 (cv 877)
# Only quantiles, 1500 trees: 866 (cv 865)

roc_train = roc_auc_score(y_train, pred_train)

print(f"roc valid {roc_valid:.4f}")
print(f"roc train {roc_train:.4f}")

# Save train and valid

train_ids = train_raw.iloc[X_train.index, :].Id

train_scores = pd.DataFrame(dict(
    Id=train_ids.values,
    prediction=pred_train
))

valid_ids = train_raw.iloc[X_valid.index, :].Id

valid_scores = pd.DataFrame(dict(
    Id=valid_ids.values,
    prediction=pred_valid
))

train_scores.to_csv('data/blend/train_lgbm_dd.csv', index=False)
valid_scores.to_csv('data/blend/valid_lgbm_dd.csv', index=False)

# CV and full training

roc_cv = cross_val_score(
    lgb, 
    X_train, 
    y_train, 
    scoring='roc_auc', 
    cv=3
).mean()

print(f"roc cv {roc_cv:.4f}")

lgb.fit(X_full, y_full)

test_predictions = lgb.predict_proba(X_test)[:, 1]

test_raw['Predicted'] = test_predictions

test_raw[['Id', 'Predicted']].to_csv('submissions/more_features_lgbm_dd.csv', index=False)


"""
B0 -> K0* + gamma
K0* -> K+ + pi-
abs(gamma_P + k+_P + pi-_P - B0_P) small -> signal = 1
abs(gamma_P + k+_P + pi-_P - B0_P) big -> signal = 0
"""


