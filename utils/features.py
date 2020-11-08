import pandas as pd
import numpy as np


def feature_engineering(all_df: pd.DataFrame) -> pd.DataFrame:

    all_df.columns = [col.replace(" ", "") for col in all_df.columns]
    # cos -> sin transformation
    all_df["Kst_892_0_sinThetaH"] = np.sqrt(
        1 - all_df["Kst_892_0_cosThetaH"] ** 2)
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
        all_df["pminus_P_x0"])
    all_df["kp_abs_x"] = np.abs(all_df["Kplus_P_x"]) + np.abs(
        all_df["pminus_P_x"])
    all_df["kp_abs_y0"] = np.abs(all_df["Kplus_P_y0"]) + np.abs(
        all_df["pminus_P_y0"])
    all_df["kp_abs_y"] = np.abs(all_df["Kplus_P_y"]) + np.abs(
        all_df["pminus_P_y"])
    # Full p ratio
    all_df["kp_x0_ratio"] = (all_df["Kplus_P_x0"] + all_df["pminus_P_x0"]) / \
                            all_df["Kplus_P"]
    all_df["kp_x_ratio"] = (all_df["Kplus_P_x"] + all_df["pminus_P_x"]) / \
                           all_df["Kplus_P"]
    all_df["kp_y0_ratio"] = (all_df["Kplus_P_y0"] + all_df["pminus_P_y0"]) / \
                            all_df["Kplus_P"]
    all_df["kp_y_ratio"] = (all_df["Kplus_P_y"] + all_df["pminus_P_y"]) / \
                           all_df["Kplus_P"]
    all_df["kbx0_ratio"] = (all_df["Kplus_P_x0"] + all_df["B_PT_x"]) / all_df[
        "Kplus_P"]
    all_df["kb_x_ratio"] = (all_df["Kplus_P_x"] + all_df["B_PT_x"]) / all_df[
        "Kplus_P"]
    all_df["kby0_ratio"] = (all_df["Kplus_P_y0"] + all_df["B_PT_y"]) / all_df[
        "Kplus_P"]
    all_df["kb_y_ratio"] = (all_df["Kplus_P_y"] + all_df["B_PT_y"]) / all_df[
        "Kplus_P"]
    all_df["kp_x0_minus_ratio"] = (all_df["Kplus_P_x0"] - all_df[
        "pminus_P_x0"]) / all_df["Kplus_P"]
    all_df["kp_x_minus_ratio"] = (all_df["Kplus_P_x"] - all_df["pminus_P_x"]) / \
                                 all_df["Kplus_P"]
    all_df["kp_y0_minus_ratio"] = (all_df["Kplus_P_y0"] - all_df[
        "pminus_P_y0"]) / all_df["Kplus_P"]
    all_df["kp_y_minus_ratio"] = (all_df["Kplus_P_y"] - all_df["pminus_P_y"]) / \
                                 all_df["Kplus_P"]
    all_df["kbx0_minus_ratio"] = (all_df["Kplus_P_x0"] - all_df["B_PT_x"]) / \
                                 all_df["Kplus_P"]
    all_df["kb_x_minus_ratio"] = (all_df["Kplus_P_x"] - all_df["B_PT_x"]) / \
                                 all_df["Kplus_P"]
    all_df["kby0_minus_ratio"] = (all_df["Kplus_P_y0"] - all_df["B_PT_y"]) / \
                                 all_df["Kplus_P"]
    all_df["kb_y_minus_ratio"] = (all_df["Kplus_P_y"] - all_df["B_PT_y"]) / \
                                 all_df["Kplus_P"]
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
    all_df["gamma_B_PT_ratio"] = (all_df["gamma_PT"] / all_df['B_PT'])
    all_df["piminus_B_P_ratio"] = (all_df["piminus_P"] / all_df['B_PT'])
    # all_df["piminus_B_P_ratio_x"] = (all_df["pminus_P_x"] / all_df['B_PT'])
    # all_df["piminus_B_P_ratio_y"] = (all_df["pminus_P_y"] / all_df['B_PT'])
    all_df["kplus_B_P_ratio"] = (all_df["Kplus_P"] / all_df['B_PT'])
    # all_df["kplus_B_P_ratio_x"] = (all_df["Kplus_P_x"] / all_df['B_PT'])
    # all_df["kplus_B_P_ratio_y"] = (all_df["Kplus_P_y"] / all_df['B_PT'])
    all_df["kplus_piminus_P_ratio"] = (all_df["Kplus_P"] / all_df['piminus_P'])
    # distance ratios
    all_df["b_distance_ratio"] = all_df['B_IPCHI2_OWNPV'] / all_df[
        'B_FDCHI2_OWNPV']
    all_df["k_p_distance_ratio"] = all_df['Kplus_IP_OWNPV'] / all_df[
        'piminus_IP_OWNPV']
    all_df["k_b_distance_ratio"] = all_df['Kplus_IP_OWNPV'] / all_df[
        'B_IPCHI2_OWNPV']
    all_df["p_b_distance_ratio"] = all_df['piminus_IP_OWNPV'] / all_df[
        'B_IPCHI2_OWNPV']
    all_df["k_kst_distance_ratio"] = all_df['Kplus_IP_OWNPV'] / all_df[
        'Kst_892_0_IP_OWNPV']
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
    all_df["total_momentum"] = all_df["gamma_PT"] + all_df["Kplus_P"] + all_df[
        "piminus_P"] - all_df["B_PT"]
    all_df["total_momentum1"] = all_df["gamma_PT"] - all_df["Kplus_P"] + all_df[
        "piminus_P"] - all_df["B_PT"]
    all_df["total_momentum2"] = all_df["gamma_PT"] + all_df["Kplus_P"] - all_df[
        "piminus_P"] - all_df["B_PT"]
    all_df["total_momentum_sq"] = all_df["gamma_PT"] ** 2 + all_df[
        "Kplus_P"] ** 2 + all_df["piminus_P"] ** 2 - all_df["B_PT"] ** 2
    all_df["total_momentum_x"] = all_df["gamma_PT"] + all_df["Kplus_P_x"] + \
                                 all_df["pminus_P_x"] - all_df["B_PT"]
    all_df["total_momentum_x2p"] = all_df["gamma_PT"] + all_df[
        "Kplus_P_x"] + 2 * all_df["pminus_P_x"] - all_df["B_PT"]
    all_df["total_momentum_x0"] = all_df["gamma_PT"] + all_df["Kplus_P_x"] + \
                                  all_df["pminus_P_x"] - all_df["B_PT"]
    all_df["total_momentum_x1"] = all_df["gamma_PT"] - all_df["Kplus_P_x"] + \
                                  all_df["pminus_P_x"] - all_df["B_PT"]
    all_df["total_momentum_x2"] = all_df["gamma_PT"] + all_df["Kplus_P_x"] - \
                                  all_df["pminus_P_x"] - all_df["B_PT"]
    all_df["total_momentum_y"] = all_df["gamma_PT"] + all_df["Kplus_P_y"] + \
                                 all_df["pminus_P_y"] - all_df["B_PT"]
    all_df["total_momentum_y2p"] = all_df["gamma_PT"] + all_df[
        "Kplus_P_y"] + 2 * all_df["pminus_P_y"] - all_df["B_PT"]
    all_df["total_momentum_y1"] = all_df["gamma_PT"] - all_df["Kplus_P_y"] + \
                                  all_df["pminus_P_y"] - all_df["B_PT"]
    all_df["total_momentum_y2"] = all_df["gamma_PT"] + all_df["Kplus_P_y"] - \
                                  all_df["pminus_P_y"] - all_df["B_PT"]
    all_df["total_momentum_abs"] = np.abs(all_df["total_momentum"])
    all_df["total_momentum_x_abs"] = np.abs(all_df["total_momentum_x"])
    all_df["total_momentum_y_abs"] = np.abs(all_df["total_momentum_y"])
    # all_df["b_pi_dff_P_y"] = 2 * all_df["pminus_P_y"] - all_df["B_PT"]
    # all_df["b_pi_dff_P_x"] = 2 * all_df["pminus_P_x"] - all_df["B_PT"]
    all_df["total_momentum_K"] = all_df["gamma_PT"] + all_df["Kplus_P"] - \
                                 all_df["B_PT"]
    all_df["total_momentum_K_x"] = all_df["gamma_PT"] + all_df["Kplus_P_x"] - \
                                   all_df["B_PT_x"]
    all_df["total_momentum_K_y"] = all_df["gamma_PT"] + all_df["Kplus_P_y"] - \
                                   all_df["B_PT_y"]
    all_df["total_momentum_K_abs"] = np.abs(all_df["total_momentum_K"])
    all_df["total_momentum_K_x_abs"] = np.abs(all_df["total_momentum_K_x"])
    all_df["total_momentum_K_y_abs"] = np.abs(all_df["total_momentum_K_y"])
    all_df["total_momentum_p"] = all_df["gamma_PT"] + all_df["piminus_P"] - \
                                 all_df["B_PT"]
    all_df["total_momentum_p_x"] = all_df["gamma_PT"] + all_df["pminus_P_x"] - \
                                   all_df["B_PT_x"]
    all_df["total_momentum_p_y"] = all_df["gamma_PT"] + all_df["pminus_P_y"] - \
                                   all_df["B_PT_y"]
    all_df["total_momentum_p_abs"] = np.abs(all_df["total_momentum_p"])
    all_df["total_momentum_p_x_abs"] = np.abs(all_df["total_momentum_p_x"])
    all_df["total_momentum_p_y_abs"] = np.abs(all_df["total_momentum_p_y"])

    return all_df