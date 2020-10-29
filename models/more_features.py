#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,auc
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import QuantileTransformer


from lightgbm import LGBMClassifier


random.seed(42)
np.random.seed(42)

train_raw = pd.read_csv("data/train.csv").drop(columns="BUTTER")
test_raw = pd.read_csv("data/test.csv").drop(columns="BUTTER")

train_raw['train'] = 1
test_raw['train'] = 0

all_df = pd.concat([train_raw, test_raw]).reset_index(drop=True)

all_df.columns = [col.replace(" ", "") for col in all_df.columns]

# cos -> sin transformation
all_df["Kst_892_0_sinThetaH"] = np.sqrt(1 - all_df["Kst_892_0_cosThetaH"]**2)
all_df["B_DIRA_OWNPV_sin"] = np.sqrt(1 - all_df["B_DIRA_OWNPV"]**2)

# x and y P components
all_df["Kplus_P_x"] = all_df["Kplus_P"] * all_df["Kst_892_0_sinThetaH"]
all_df["Kplus_P_y"] = all_df["Kplus_P"] * all_df["Kst_892_0_cosThetaH"]
all_df["Kplus_P_x0"] = all_df["Kplus_P"] * np.sin(all_df["Kplus_ETA"])
all_df["Kplus_P_y0"] = all_df["Kplus_P"] * np.cos(all_df["Kplus_ETA"])
all_df["pminus_P_x"] = all_df["piminus_P"] * all_df["Kst_892_0_sinThetaH"]
all_df["pminus_P_y"] = all_df["piminus_P"] * all_df["Kst_892_0_cosThetaH"]
all_df["pminus_P_x0"] = all_df["piminus_P"] * np.sin(all_df["piminus_ETA"])
all_df["pminus_P_y0"] = all_df["piminus_P"] * np.cos(all_df["piminus_ETA"])
all_df["B_PT_x"] = all_df["B_PT"] * all_df["B_DIRA_OWNPV"]
all_df["B_PT_y"] = all_df["B_PT"] * all_df["B_DIRA_OWNPV_sin"]

# Full p
# all_df["kp_x0_total"] = all_df["Kplus_P_x0"] + all_df["pminus_P_x0"] + all_df["B_PT_x"]
# all_df["kp_x_total"] = all_df["Kplus_P_x"] + all_df["pminus_P_x"] + all_df["B_PT_x"]
# all_df["kp_y0_total"] = all_df["Kplus_P_y0"] + all_df["pminus_P_y0"] + all_df["B_PT_y"]
# all_df["kp_y_total"] = all_df["Kplus_P_y"] + all_df["pminus_P_y"] + all_df["B_PT_y"]

# Paired p
all_df["kp_x0"] = all_df["Kplus_P_x0"] + all_df["pminus_P_x0"]
all_df["kp_x"] = all_df["Kplus_P_x"] + all_df["pminus_P_x"]
all_df["kp_y0"] = all_df["Kplus_P_y0"] + all_df["pminus_P_y0"]
all_df["kp_y"] = all_df["Kplus_P_y"] + all_df["pminus_P_y"]
all_df["kbx0"] = all_df["Kplus_P_x0"] + all_df["B_PT_x"]
all_df["kb_x"] = all_df["Kplus_P_x"] + all_df["B_PT_x"]
all_df["kby0"] = all_df["Kplus_P_y0"] + all_df["B_PT_y"]
all_df["kb_y"] = all_df["Kplus_P_y"] + all_df["B_PT_y"]
all_df["kp_x0_minus"] = all_df["Kplus_P_x0"] - all_df["pminus_P_x0"]
all_df["kp_x_minus"] = all_df["Kplus_P_x"] - all_df["pminus_P_x"]
all_df["kp_y0_minus"] = all_df["Kplus_P_y0"] - all_df["pminus_P_y0"]
all_df["kp_y_minus"] = all_df["Kplus_P_y"] - all_df["pminus_P_y"]
all_df["kbx0_minus"] = all_df["Kplus_P_x0"] - all_df["B_PT_x"]
all_df["kb_x_minus"] = all_df["Kplus_P_x"] - all_df["B_PT_x"]
all_df["kby0_minus"] = all_df["Kplus_P_y0"] - all_df["B_PT_y"]
all_df["kb_y_minus"] = all_df["Kplus_P_y"] - all_df["B_PT_y"]
all_df["kp_abs_x0"] = np.abs(all_df["Kplus_P_x0"]) + np.abs(all_df["pminus_P_x0"])
all_df["kp_abs_x"] = np.abs(all_df["Kplus_P_x"]) + np.abs(all_df["pminus_P_x"])
all_df["kp_abs_y0"] = np.abs(all_df["Kplus_P_y0"]) + np.abs(all_df["pminus_P_y0"])
all_df["kp_abs_y"] = np.abs(all_df["Kplus_P_y"]) + np.abs(all_df["pminus_P_y"])

# Full p ratio
all_df["kp_x0_ratio"] = (all_df["Kplus_P_x0"] + all_df["pminus_P_x0"]) / all_df["Kplus_P"]
all_df["kp_x_ratio"] = (all_df["Kplus_P_x"] + all_df["pminus_P_x"]) / all_df["Kplus_P"]
all_df["kp_y0_ratio"] = (all_df["Kplus_P_y0"] + all_df["pminus_P_y0"]) / all_df["Kplus_P"]
all_df["kp_y_ratio"] = (all_df["Kplus_P_y"] + all_df["pminus_P_y"]) / all_df["Kplus_P"]
all_df["kbx0_ratio"] = (all_df["Kplus_P_x0"] + all_df["B_PT_x"]) / all_df["Kplus_P"]
all_df["kb_x_ratio"] = (all_df["Kplus_P_x"] + all_df["B_PT_x"]) / all_df["Kplus_P"]
all_df["kby0_ratio"] = (all_df["Kplus_P_y0"] + all_df["B_PT_y"]) / all_df["Kplus_P"]
all_df["kb_y_ratio"] = (all_df["Kplus_P_y"] + all_df["B_PT_y"]) / all_df["Kplus_P"]
all_df["kp_x0_minus_ratio"] = (all_df["Kplus_P_x0"] - all_df["pminus_P_x0"]) / all_df["Kplus_P"]
all_df["kp_x_minus_ratio"] = (all_df["Kplus_P_x"] - all_df["pminus_P_x"]) / all_df["Kplus_P"]
all_df["kp_y0_minus_ratio"] = (all_df["Kplus_P_y0"] - all_df["pminus_P_y0"]) / all_df["Kplus_P"]
all_df["kp_y_minus_ratio"] = (all_df["Kplus_P_y"] - all_df["pminus_P_y"]) / all_df["Kplus_P"]
all_df["kbx0_minus_ratio"] = (all_df["Kplus_P_x0"] - all_df["B_PT_x"]) / all_df["Kplus_P"]
all_df["kb_x_minus_ratio"] = (all_df["Kplus_P_x"] - all_df["B_PT_x"]) / all_df["Kplus_P"]
all_df["kby0_minus_ratio"] = (all_df["Kplus_P_y0"] - all_df["B_PT_y"]) / all_df["Kplus_P"]
all_df["kb_y_minus_ratio"] = (all_df["Kplus_P_y"] - all_df["B_PT_y"]) / all_df["Kplus_P"]

# things in hbar units
all_df["B_hbar"] = all_df["B_PT"] * all_df["B_IPCHI2_OWNPV"]
all_df["B_hbar_2"] = all_df["B_PT"] * all_df["B_FDCHI2_OWNPV"]
all_df["K_hbar"] = all_df["Kplus_P"] * all_df["Kplus_IP_OWNPV"]
all_df["p_hbar"] = all_df["piminus_P"] * all_df["piminus_IP_OWNPV"]

# hbar ratios
all_df["B_hbar_ratio"] = all_df["B_hbar"] / all_df["B_hbar_2"]
all_df["K_p_hbar_ratio"] = all_df["K_hbar"] / all_df["p_hbar"]
all_df["K_B_hbar_ratio"] = all_df["K_hbar"] / all_df["B_hbar"]

# p ratios
all_df["gamma_B_PT_ratio"] = (all_df["gamma_PT"] / all_df['B_PT'])
all_df["piminus_B_P_ratio"] = (all_df["piminus_P"] / all_df['B_PT'])
all_df["kplus_B_P_ratio"] = (all_df["Kplus_P"] / all_df['B_PT'])
all_df["kplus_piminus_P_ratio"] = (all_df["Kplus_P"] / all_df['piminus_P'])

# distance ratios
all_df["b_distance_ratio"] = all_df['B_IPCHI2_OWNPV'] / all_df['B_FDCHI2_OWNPV']
all_df["k_p_distance_ratio"] = all_df['Kplus_IP_OWNPV'] / all_df['piminus_IP_OWNPV']
all_df["k_b_distance_ratio"] = all_df['Kplus_IP_OWNPV'] / all_df['B_IPCHI2_OWNPV']
all_df["p_b_distance_ratio"] = all_df['piminus_IP_OWNPV'] / all_df['B_IPCHI2_OWNPV']
all_df["k_kst_distance_ratio"] = all_df['Kplus_IP_OWNPV'] / all_df['Kst_892_0_IP_OWNPV']

# shpere radius
all_df["sphere_radius_k_b"] =  all_df['Kplus_IP_OWNPV']**2 + all_df['B_IPCHI2_OWNPV']**2
all_df["sphere_radius_p_b"] =  all_df['piminus_IP_OWNPV']**2 + all_df['B_IPCHI2_OWNPV']**2

# ANGLE ratios
# all_df["b_eta"] = np.arccos(all_df["B_DIRA_OWNPV"])
# all_df["b_K_ratio"] = all_df["b_eta"] / all_df["Kplus_ETA"]
# all_df["b_p_ratio"] = all_df["b_eta"] / all_df["piminus_ETA"]

# Conservation of momentum
# all_df["total_momentum_K"] = all_df["gamma_PT"] + all_df["Kplus_P"] - all_df["B_PT"]
# all_df["total_momentum_p"] = all_df["gamma_PT"] + all_df["piminus_P"] - all_df["B_PT"]


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

lgb = LGBMClassifier(
    n_estimators=500,
    monotone_constraint=constraint_list,
    n_jobs=-1,
    # monotone_constraints_method="intermediate"
)
lgb.fit(X_train, y_train)

pred_valid = lgb.predict_proba(X_valid)[:, 1]
pred_train = lgb.predict_proba(X_train)[:, 1]


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

test_raw[['Id', 'Predicted']].to_csv('submissions/more_features_lgbm.csv', index=False)





