import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from gauss_rank_scaler import GaussRankScaler
from gbfeatures import GradientBoostingFeatureGenerator
from sklearn.externals import joblib
import random
import os

# os.path.isfile(fname)

random.seed(0)


class ReadDataset(Dataset):
    """CERN kaagle dataset."""

    def __init__(self, csv_file, for_test=False, gradient_boosting_features=False):
        """
        Args:
            csv_file (str): Path to the csv file with the students data.

        """
        self.df = pd.read_csv(csv_file).drop(columns="BUTTER")
        self.df.columns = self.df.columns.str.replace(" ", "")

        try:
            self.df = self.df.drop(columns=["Unnamed:0"])
        except Exception:
            pass
        try:
            self.df = self.df.drop(columns=["Id"])
        except Exception:
            pass

        # Target
        self.target = "signal"
        # If we read test, not load the target
        if for_test == False:
            # Save target and predictors
            self.X = self.df.drop(self.target, axis=1)
            self.y = self.df[self.target]

        else:
            self.X = self.df

        # Create features
        self.X = self.transform(self.X)
        self.X = self.feature_engineering(self.X)

        # If the scaler does not exist create it
        if os.path.isfile("output/scaler.save") == False:
            self.scaler = GaussRankScaler()
            self.scaler = self.scaler.fit(self.X)
            ## Save scaler to be later used on the prediction
            joblib.dump(self.scaler, "output/scaler.save")
        # If the scale exist, load it
        else:
            self.scaler = joblib.load("output/scaler.save")
        ## Scale data
        self.X = pd.DataFrame(self.scaler.transform(self.X), columns=self.X.columns)

        # Gradient boosting features
        if gradient_boosting_features:
            self.X = self.gb_features()
        print("data size", self.X.shape)

    def gb_features(self, sample_size=0.1):
        ## GB features
        df = pd.read_csv("train.csv").drop(columns="BUTTER")
        df.columns = df.columns.str.replace(" ", "")

        try:
            df = df.drop(columns=["Unnamed:0"])
        except Exception:
            pass
        try:
            df = df.drop(columns=["Id"])
        except Exception:
            pass

        # Target
        target = "signal"
        X_fit = df.drop(target, axis=1)
        y_fit = df[target]

        # Create features
        X_fit = self.transform(X_fit)
        X_fit = self.feature_engineering(X_fit)

        # GB features
        sample_size = int(X_fit.shape[0] * sample_size)
        gb_feat = GradientBoostingFeatureGenerator(random_state=0)
        gb_feat.fit(X_fit.head(sample_size), y_fit.head(sample_size))

        # Transform original data
        self.X = pd.DataFrame(gb_feat.transform(self.X))
        return self.X

    def transform(self, df):
        # Did not work
        df["Kst_892_0_cosThetaH_arc"] = np.arccos(df["Kst_892_0_cosThetaH"])
        df["Kst_892_0_cosThetaH_arc_sin"] = np.sin(df.Kst_892_0_cosThetaH_arc)

        df["Kplu_pXKst_892costheta"] = df["Kplus_P"] * df["Kst_892_0_cosThetaH"]

        df["momentum_by_shortest_dist"] = df["Kplus_P"] / df["Kplus_IP_OWNPV"]

        df["momentum_sum"] = df["B_PT"] + df["Kplus_P"] + df["gamma_PT"]

        # Worked individually
        df["Kplu_p_div_Kst_892costheta"] = df["Kplus_P"] / df["Kst_892_0_cosThetaH"]

        # ETA (estimated time of arrival) inversions -- worked
        df["Kplus_ETA_inv"] = 1 / df["Kplus_ETA"]
        df["piminus_ETA_inv"] = 1 / df["piminus_ETA"]

        # Momentum

        df["total_mom_2"] = df["B_PT"] ** 2 + df["gamma_PT"] ** 2 + df["Kplus_P"] ** 2
        df["total_mom_sum"] = df["B_PT"] + df["gamma_PT"] + df["Kplus_P"]

        # All mom ratios worked together
        df["mom_rat_1"] = df["B_PT"] / df["Kplus_P"]
        df["mom_rat_2"] = df["B_PT"] / df["gamma_PT"]

        df["mom_rat_3"] = df["gamma_PT"] / df["B_PT"]
        df["mom_rat_4"] = df["gamma_PT"] / df["Kplus_P"]

        df["mom_rat_5"] = df["Kplus_P"] / df["B_PT"]
        df["mom_rat_6"] = df["Kplus_P"] / df["gamma_PT"]

        df["ThetaH"] = np.arccos(df["Kst_892_0_cosThetaH"])

        df["Kplus_P_x"] = df["Kplus_P"] * np.sin(df["ThetaH"])
        df["Kplus_P_y"] = df["Kplus_P"] * np.cos(df["ThetaH"])

        df["mom_consev1"] = df["gamma_PT"] ** 2 + df["Kplus_P"] ** 2 - df["B_PT"] ** 2
        df["mom_consev1_1"] = ((df["gamma_PT"] + df["Kplus_P"])) ** 2 - df["B_PT"] ** 2

        # B meson ratios
        df["mesB_ratio_1"] = df["B_FDCHI2_OWNPV"] / df["B_IPCHI2_OWNPV"]
        df["mesB_ratio_2"] = df["B_FDCHI2_OWNPV"] / df["B_PT"]

        df["mesB_ratio_3"] = df["B_IPCHI2_OWNPV"] / df["B_PT"]
        df["mesB_ratio_4"] = df["B_IPCHI2_OWNPV"] / df["B_PT"]

        df["mesB_ratio_5"] = df["B_PT"] / df["B_IPCHI2_OWNPV"]
        df["mesB_ratio_6"] = df["B_PT"] / df["B_FDCHI2_OWNPV"]

        # Neither Improved neither worst
        df["kst_thetaH"] = np.arccos(df.Kst_892_0_cosThetaH)  # * 180 / np.pi
        df["kst_thetaH_sin"] = np.sin(df["kst_thetaH"])
        df["kst_thetaH_cos"] = np.cos(df["kst_thetaH"])
        df["kst_thetaH_sin_cos"] = np.sin(df["kst_thetaH"]) - np.cos(df["kst_thetaH"])
        df["kst_thetaH_tan"] = np.tan(df["kst_thetaH"])
        df["kst_thetaH_exp"] = np.exp(df["kst_thetaH"])
        df["kst_thetaH_exp_1"] = np.exp(-df["kst_thetaH"])

        df["B_DIRA_OWNPV_angle"] = np.arccos(df["B_DIRA_OWNPV"])  # * 180 / np.pi
        df["B_DIRA_OWNPV__cos"] = np.cos(df["B_DIRA_OWNPV_angle"])
        df["B_DIRA_OWNPV__sin"] = np.sin(df["B_DIRA_OWNPV_angle"])
        df["B_DIRA_OWNPV__tan"] = np.tan(df["B_DIRA_OWNPV_angle"])
        df["B_DIRA_OWNPV__sin_cos"] = np.sin(df["B_DIRA_OWNPV_angle"]) - np.cos(
            df["B_DIRA_OWNPV_angle"]
        )
        df["B_DIRA_OWNPV_exp"] = np.exp(df["B_DIRA_OWNPV_angle"])
        df["B_DIRA_OWNPV_exp_1"] = np.exp(-df["B_DIRA_OWNPV_angle"])

        # Momentum multiplication
        df["mom1_exp"] = df["mom_rat_1"] * df["B_DIRA_OWNPV_exp"]
        df["mom2_exp"] = df["mom_rat_2"] * df["B_DIRA_OWNPV_exp"]
        df["mom3_exp"] = df["mom_rat_3"] * df["B_DIRA_OWNPV_exp"]
        df["mom4_exp"] = df["mom_rat_4"] * df["B_DIRA_OWNPV_exp"]
        df["mom5_exp"] = df["mom_rat_5"] * df["B_DIRA_OWNPV_exp"]
        df["mom6_exp"] = df["mom_rat_6"] * df["B_DIRA_OWNPV_exp"]
        df["mom7_exp"] = df["B_PT"] * df["B_DIRA_OWNPV_exp"]
        df["mom8_exp"] = df["Kplus_P"] * df["B_DIRA_OWNPV_exp"]
        df["mom9_exp"] = df["gamma_PT"] * df["B_DIRA_OWNPV_exp"]

        df["mom11_exp"] = df["mom_rat_1"] * df["kst_thetaH"]
        df["mom22_exp"] = df["mom_rat_2"] * df["kst_thetaH"]
        df["mom33_exp"] = df["mom_rat_3"] * df["kst_thetaH"]
        df["mom44_exp"] = df["mom_rat_4"] * df["kst_thetaH"]
        df["mom55_exp"] = df["mom_rat_5"] * df["kst_thetaH"]
        df["mom66_exp"] = df["mom_rat_6"] * df["kst_thetaH"]
        df["mom77_exp"] = df["B_PT"] * df["kst_thetaH"]
        df["mom88_exp"] = df["Kplus_P"] * df["kst_thetaH"]
        df["mom99_exp"] = df["gamma_PT"] * df["kst_thetaH"]

        df["Kplus_ETA_2"] = df["Kplus_ETA"] ** 2
        df["Kplus_ETA_exp"] = np.exp(df["Kplus_ETA"])
        df["Kplus_ETA_theta"] = df["Kplus_ETA"] * df["kst_thetaH"]
        df["Kplus_ETA_div_theta"] = df["Kplus_ETA"] / df["kst_thetaH"]
        df["Kplus_ETA_theta_sin"] = df["Kplus_ETA"] * df["kst_thetaH_sin"]
        df["Kplus_ETA_theta_tan"] = df["Kplus_ETA"] * df["kst_thetaH_tan"]

        df["piminus_ETA_2"] = df["piminus_ETA"] ** 2
        df["piminus_ETA_exp"] = np.exp(df["piminus_ETA"])
        df["piminus_ETA_theta"] = df["piminus_ETA"] * df["kst_thetaH"]
        df["piminus_ETA_div_theta"] = df["piminus_ETA"] / df["kst_thetaH"]
        df["piminus_ETA_theta_sin"] = df["piminus_ETA"] * df["kst_thetaH_sin"]
        df["piminus_ETA_theta_tan"] = df["piminus_ETA"] * df["kst_thetaH_tan"]

        # Improves
        df["Kplus_by_eta"] = df["Kplus_P"] / df["Kplus_ETA"]

        df["Kplus_piminus_by_eta"] = df["Kplus_P"] / df["piminus_ETA"]

        df["ETA_add"] = df["Kplus_ETA"] + df["piminus_ETA"]
        df["ETA_rest"] = df["Kplus_ETA"] - df["piminus_ETA"]
        df["ETA_mult"] = df["Kplus_ETA"] * df["piminus_ETA"]
        df["ETA_inv"] = df["Kplus_ETA"] / df["piminus_ETA"]
        df["ETA_inv1"] = df["piminus_ETA"] / df["Kplus_ETA"]

        df["Kplus_ETA_inv"] = df["Kplus_ETA"]
        df["piminus_ETA_inv"] = df["piminus_ETA"]

        df["ETA_add_inv"] = df["Kplus_ETA_inv"] + df["piminus_ETA_inv"]
        df["ETA_rest_inv"] = df["Kplus_ETA_inv"] - df["piminus_ETA_inv"]
        df["ETA_mult_inv"] = df["Kplus_ETA_inv"] * df["piminus_ETA_inv"]
        df["ETA_inv_inv"] = df["Kplus_ETA_inv"] / df["piminus_ETA_inv"]
        df["ETA_inv1_inv"] = df["piminus_ETA_inv"] / df["Kplus_ETA_inv"]

        self.df = df
        return self.df

    def feature_engineering(self, all_df):
        """
        Features by david 05/11
        """

        all_df.columns = [col.replace(" ", "") for col in all_df.columns]
        # cos -> sin transformation
        all_df["Kst_892_0_sinThetaH"] = np.sqrt(1 - all_df["Kst_892_0_cosThetaH"] ** 2)
        all_df["B_DIRA_OWNPV_sin"] = np.sqrt(1 - all_df["B_DIRA_OWNPV"] ** 2)
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
        all_df["kp_abs_x0"] = np.abs(all_df["Kplus_P_x0"]) + np.abs(
            all_df["pminus_P_x0"]
        )
        all_df["kp_abs_x"] = np.abs(all_df["Kplus_P_x"]) + np.abs(all_df["pminus_P_x"])
        all_df["kp_abs_y0"] = np.abs(all_df["Kplus_P_y0"]) + np.abs(
            all_df["pminus_P_y0"]
        )
        all_df["kp_abs_y"] = np.abs(all_df["Kplus_P_y"]) + np.abs(all_df["pminus_P_y"])
        # Full p ratio
        all_df["kp_x0_ratio"] = (all_df["Kplus_P_x0"] + all_df["pminus_P_x0"]) / all_df[
            "Kplus_P"
        ]
        all_df["kp_x_ratio"] = (all_df["Kplus_P_x"] + all_df["pminus_P_x"]) / all_df[
            "Kplus_P"
        ]
        all_df["kp_y0_ratio"] = (all_df["Kplus_P_y0"] + all_df["pminus_P_y0"]) / all_df[
            "Kplus_P"
        ]
        all_df["kp_y_ratio"] = (all_df["Kplus_P_y"] + all_df["pminus_P_y"]) / all_df[
            "Kplus_P"
        ]
        all_df["kbx0_ratio"] = (all_df["Kplus_P_x0"] + all_df["B_PT_x"]) / all_df[
            "Kplus_P"
        ]
        all_df["kb_x_ratio"] = (all_df["Kplus_P_x"] + all_df["B_PT_x"]) / all_df[
            "Kplus_P"
        ]
        all_df["kby0_ratio"] = (all_df["Kplus_P_y0"] + all_df["B_PT_y"]) / all_df[
            "Kplus_P"
        ]
        all_df["kb_y_ratio"] = (all_df["Kplus_P_y"] + all_df["B_PT_y"]) / all_df[
            "Kplus_P"
        ]
        all_df["kp_x0_minus_ratio"] = (
            all_df["Kplus_P_x0"] - all_df["pminus_P_x0"]
        ) / all_df["Kplus_P"]
        all_df["kp_x_minus_ratio"] = (
            all_df["Kplus_P_x"] - all_df["pminus_P_x"]
        ) / all_df["Kplus_P"]
        all_df["kp_y0_minus_ratio"] = (
            all_df["Kplus_P_y0"] - all_df["pminus_P_y0"]
        ) / all_df["Kplus_P"]
        all_df["kp_y_minus_ratio"] = (
            all_df["Kplus_P_y"] - all_df["pminus_P_y"]
        ) / all_df["Kplus_P"]
        all_df["kbx0_minus_ratio"] = (all_df["Kplus_P_x0"] - all_df["B_PT_x"]) / all_df[
            "Kplus_P"
        ]
        all_df["kb_x_minus_ratio"] = (all_df["Kplus_P_x"] - all_df["B_PT_x"]) / all_df[
            "Kplus_P"
        ]
        all_df["kby0_minus_ratio"] = (all_df["Kplus_P_y0"] - all_df["B_PT_y"]) / all_df[
            "Kplus_P"
        ]
        all_df["kb_y_minus_ratio"] = (all_df["Kplus_P_y"] - all_df["B_PT_y"]) / all_df[
            "Kplus_P"
        ]
        # # things in hbar units
        # all_df["B_hbar"] = all_df["B_PT"] * all_df["B_IPCHI2_OWNPV"]
        # all_df["B_hbar_2"] = all_df["B_PT"] * all_df["B_FDCHI2_OWNPV"]
        # all_df["K_hbar"] = all_df["Kplus_P"] * all_df["Kplus_IP_OWNPV"]
        # all_df["p_hbar"] = all_df["piminus_P"] * all_df["piminus_IP_OWNPV"]
        #
        # # hbar ratios
        # all_df["B_hbar_ratio"] = all_df["B_hbar"] / all_df["B_hbar_2"]
        # all_df["K_p_hbar_ratio"] = all_df["K_hbar"] / all_df["p_hbar"]
        # all_df["K_B_hbar_ratio"] = all_df["K_hbar"] / all_df["B_hbar"]
        # p ratios
        all_df["gamma_B_PT_ratio"] = all_df["gamma_PT"] / all_df["B_PT"]
        all_df["piminus_B_P_ratio"] = all_df["piminus_P"] / all_df["B_PT"]
        # all_df["piminus_B_P_ratio_x"] = (all_df["pminus_P_x"] / all_df['B_PT'])
        # all_df["piminus_B_P_ratio_y"] = (all_df["pminus_P_y"] / all_df['B_PT'])
        all_df["kplus_B_P_ratio"] = all_df["Kplus_P"] / all_df["B_PT"]
        # all_df["kplus_B_P_ratio_x"] = (all_df["Kplus_P_x"] / all_df['B_PT'])
        # all_df["kplus_B_P_ratio_y"] = (all_df["Kplus_P_y"] / all_df['B_PT'])
        all_df["kplus_piminus_P_ratio"] = all_df["Kplus_P"] / all_df["piminus_P"]
        # distance ratios
        all_df["b_distance_ratio"] = all_df["B_IPCHI2_OWNPV"] / all_df["B_FDCHI2_OWNPV"]
        all_df["k_p_distance_ratio"] = (
            all_df["Kplus_IP_OWNPV"] / all_df["piminus_IP_OWNPV"]
        )
        all_df["k_b_distance_ratio"] = (
            all_df["Kplus_IP_OWNPV"] / all_df["B_IPCHI2_OWNPV"]
        )
        all_df["p_b_distance_ratio"] = (
            all_df["piminus_IP_OWNPV"] / all_df["B_IPCHI2_OWNPV"]
        )
        all_df["k_kst_distance_ratio"] = (
            all_df["Kplus_IP_OWNPV"] / all_df["Kst_892_0_IP_OWNPV"]
        )
        # shpere radius
        # all_df["sphere_radius_k_b"] =  all_df['Kplus_IP_OWNPV']**2 + all_df['B_IPCHI2_OWNPV']**2
        # all_df["sphere_radius_p_b"] =  all_df['piminus_IP_OWNPV']**2 + all_df['B_IPCHI2_OWNPV']**2
        # ANGLE ratios
        # all_df["B_DIR"] = np.arccos(all_df["B_DIRA_OWNPV"])
        # all_df["theta"] = np.arccos(all_df["Kst_892_0_cosThetaH"])
        # all_df["angle_ratio"] = np.log(all_df["B_DIR"] / all_df["theta"])
        # all_df["b_eta"] = np.arccos(all_df["B_DIRA_OWNPV"])
        # all_df["b_K_ratio"] = all_df["b_eta"] / all_df["Kplus_ETA"]
        # ETA ratio
        all_df["eta_ratio"] = all_df["Kplus_ETA"] / all_df["piminus_ETA"]
        # Conservation of momentum
        all_df["total_momentum"] = (
            all_df["gamma_PT"]
            + all_df["Kplus_P"]
            + all_df["piminus_P"]
            - all_df["B_PT"]
        )
        all_df["total_momentum1"] = (
            all_df["gamma_PT"]
            - all_df["Kplus_P"]
            + all_df["piminus_P"]
            - all_df["B_PT"]
        )
        all_df["total_momentum2"] = (
            all_df["gamma_PT"]
            + all_df["Kplus_P"]
            - all_df["piminus_P"]
            - all_df["B_PT"]
        )
        all_df["total_momentum_sq"] = (
            all_df["gamma_PT"] ** 2
            + all_df["Kplus_P"] ** 2
            + all_df["piminus_P"] ** 2
            - all_df["B_PT"] ** 2
        )
        all_df["total_momentum_x"] = (
            all_df["gamma_PT"]
            + all_df["Kplus_P_x"]
            + all_df["pminus_P_x"]
            - all_df["B_PT"]
        )
        all_df["total_momentum_x2p"] = (
            all_df["gamma_PT"]
            + all_df["Kplus_P_x"]
            + 2 * all_df["pminus_P_x"]
            - all_df["B_PT"]
        )
        all_df["total_momentum_x0"] = (
            all_df["gamma_PT"]
            + all_df["Kplus_P_x"]
            + all_df["pminus_P_x"]
            - all_df["B_PT"]
        )
        all_df["total_momentum_x1"] = (
            all_df["gamma_PT"]
            - all_df["Kplus_P_x"]
            + all_df["pminus_P_x"]
            - all_df["B_PT"]
        )
        all_df["total_momentum_x2"] = (
            all_df["gamma_PT"]
            + all_df["Kplus_P_x"]
            - all_df["pminus_P_x"]
            - all_df["B_PT"]
        )
        all_df["total_momentum_y"] = (
            all_df["gamma_PT"]
            + all_df["Kplus_P_y"]
            + all_df["pminus_P_y"]
            - all_df["B_PT"]
        )
        all_df["total_momentum_y2p"] = (
            all_df["gamma_PT"]
            + all_df["Kplus_P_y"]
            + 2 * all_df["pminus_P_y"]
            - all_df["B_PT"]
        )
        all_df["total_momentum_y1"] = (
            all_df["gamma_PT"]
            - all_df["Kplus_P_y"]
            + all_df["pminus_P_y"]
            - all_df["B_PT"]
        )
        all_df["total_momentum_y2"] = (
            all_df["gamma_PT"]
            + all_df["Kplus_P_y"]
            - all_df["pminus_P_y"]
            - all_df["B_PT"]
        )
        all_df["total_momentum_abs"] = np.abs(all_df["total_momentum"])
        all_df["total_momentum_x_abs"] = np.abs(all_df["total_momentum_x"])
        all_df["total_momentum_y_abs"] = np.abs(all_df["total_momentum_y"])
        # all_df["b_pi_dff_P_y"] = 2 * all_df["pminus_P_y"] - all_df["B_PT"]
        # all_df["b_pi_dff_P_x"] = 2 * all_df["pminus_P_x"] - all_df["B_PT"]
        all_df["total_momentum_K"] = (
            all_df["gamma_PT"] + all_df["Kplus_P"] - all_df["B_PT"]
        )
        all_df["total_momentum_K_x"] = (
            all_df["gamma_PT"] + all_df["Kplus_P_x"] - all_df["B_PT_x"]
        )
        all_df["total_momentum_K_y"] = (
            all_df["gamma_PT"] + all_df["Kplus_P_y"] - all_df["B_PT_y"]
        )
        all_df["total_momentum_K_abs"] = np.abs(all_df["total_momentum_K"])
        all_df["total_momentum_K_x_abs"] = np.abs(all_df["total_momentum_K_x"])
        all_df["total_momentum_K_y_abs"] = np.abs(all_df["total_momentum_K_y"])
        all_df["total_momentum_p"] = (
            all_df["gamma_PT"] + all_df["piminus_P"] - all_df["B_PT"]
        )
        all_df["total_momentum_p_x"] = (
            all_df["gamma_PT"] + all_df["pminus_P_x"] - all_df["B_PT_x"]
        )
        all_df["total_momentum_p_y"] = (
            all_df["gamma_PT"] + all_df["pminus_P_y"] - all_df["B_PT_y"]
        )
        all_df["total_momentum_p_abs"] = np.abs(all_df["total_momentum_p"])
        all_df["total_momentum_p_x_abs"] = np.abs(all_df["total_momentum_p_x"])
        all_df["total_momentum_p_y_abs"] = np.abs(all_df["total_momentum_p_y"])

        return all_df

    def __len__(self):
        return len(self.X)

    def __shape__(self):
        return self.X.shape[1]

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X.iloc[idx].values, self.y[idx]]


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 6 * input_dim)
        self.relu1 = nn.SELU()
        self.batchnorm1 = nn.BatchNorm1d(6 * input_dim)
        self.drop1 = nn.Dropout(0.05, inplace=False)

        self.fc2 = nn.Linear(6 * input_dim, 3 * input_dim, bias=False)
        self.relu2 = nn.SELU()
        self.batchnorm2 = nn.BatchNorm1d(
            3 * input_dim,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )
        self.drop2 = nn.Dropout(0.05, inplace=False)

        self.fc3 = nn.Linear(3 * input_dim, 2 * input_dim, bias=False)
        self.relu3 = nn.SELU()
        self.batchnorm3 = nn.BatchNorm1d(
            2 * input_dim,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )
        self.drop3 = nn.Dropout(0.05, inplace=False)

        self.fc4 = nn.Linear(2 * input_dim, 1 * input_dim, bias=False)
        self.relu4 = nn.SELU()
        self.batchnorm4 = nn.BatchNorm1d(
            input_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.drop4 = nn.Dropout(0.05, inplace=False)

        self.fc5 = nn.Linear(input_dim, 1, bias=True)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        self.batchnorm1(x)
        self.drop1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        self.batchnorm2(x)
        self.drop2(x)

        x = self.fc3(x)
        x = self.relu3(x)
        self.batchnorm3(x)
        self.drop3(x)

        x = self.fc4(x)
        x = self.relu4(x)
        self.batchnorm4(x)
        self.drop4(x)

        x = self.fc5(x)
        x = self.sig(x)

        return x.squeeze()

    def step(self, inputs):
        data, label = inputs  # ignore label
        outputs = self.forward(data)
        _, preds = torch.max(outputs.data, 1)
        # preds, outputs  are cuda tensors. Right?
        return preds, outputs

    def predict_proba(self, data):
        # Pass the NN forward
        out = self.forward(data)

        # Transform to
        return out.detach().numpy().squeeze()


class ResNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 6 * input_dim)
        self.relu1 = nn.SELU()
        self.batchnorm1 = nn.BatchNorm1d(6 * input_dim)
        self.drop1 = nn.Dropout(0.05, inplace=False)

        self.fc2 = nn.Linear(6 * input_dim, 3 * input_dim, bias=False)
        self.relu2 = nn.SELU()
        self.batchnorm2 = nn.BatchNorm1d(
            3 * input_dim,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )
        self.drop2 = nn.Dropout(0.05, inplace=False)

        self.fc3 = nn.Linear(3 * input_dim, 2 * input_dim, bias=False)
        self.relu3 = nn.SELU()
        self.batchnorm3 = nn.BatchNorm1d(
            2 * input_dim + input_dim,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )
        self.drop3 = nn.Dropout(0.05, inplace=False)

        self.fc4 = nn.Linear(2 * input_dim + input_dim, 1 * input_dim, bias=False)
        self.relu4 = nn.SELU()
        self.batchnorm4 = nn.BatchNorm1d(
            input_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.drop4 = nn.Dropout(0.05, inplace=False)

        self.fc5 = nn.Linear(input_dim, 1, bias=True)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        x1 = x
        x = self.fc1(x)
        x = self.relu1(x)
        self.batchnorm1(x)
        self.drop1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        self.batchnorm2(x)
        self.drop2(x)

        x = self.fc3(x)
        x = self.relu3(torch.cat((x, x1), 1))
        self.batchnorm3(x)
        self.drop3(x)

        x = self.fc4(x)
        x = self.relu4(x)
        self.batchnorm4(x)
        self.drop4(x)

        x = self.fc5(x)
        x = self.sig(x)

        return x.squeeze()

    def step(self, inputs):
        data, label = inputs  # ignore label
        outputs = self.forward(data)
        _, preds = torch.max(outputs.data, 1)
        # preds, outputs  are cuda tensors. Right?
        return preds, outputs

    def predict_proba(self, data):
        # Pass the NN forward
        out = self.forward(data)

        # Transform to
        return out.detach().numpy().squeeze()


class BasicBlock(nn.Module):
    def __init__(self, siz3, downsample=None):
        super().__init__()
        self.fc1 = nn.Linear(size, 2 * size)

        self.bn1 = nn.BatchNorm1d(2 * size)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(2 * size, size)
        self.bn2 = nn.BatchNorm1d(size)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def _make_layer(block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)
