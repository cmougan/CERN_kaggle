import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_X_y
from sklearn.utils.validation import (
    check_is_fitted,
    check_array,
    FLOAT_DTYPES
)

from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.special import erf, erfinv
from sklearn.base import BaseEstimator, TransformerMixin


class GaussRankScaler(BaseEstimator, TransformerMixin):
    """Transform features by scaling each feature to a normal distribution.
    Parameters
        ----------
        epsilon : float, optional, default 1e-4
            A small amount added to the lower bound or subtracted
            from the upper bound. This value prevents infinite number
            from occurring when applying the inverse error function.
        copy : boolean, optional, default True
            If False, try to avoid a copy and do inplace scaling instead.
            This is not guaranteed to always work inplace; e.g. if the data is
            not a NumPy array, a copy may still be returned.
        n_jobs : int or None, optional, default None
            Number of jobs to run in parallel.
            ``None`` means 1 and ``-1`` means using all processors.
        interp_kind : str or int, optional, default 'linear'
           Specifies the kind of interpolation as a string
            ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
            'previous', 'next', where 'zero', 'slinear', 'quadratic' and 'cubic'
            refer to a spline interpolation of zeroth, first, second or third
            order; 'previous' and 'next' simply return the previous or next value
            of the point) or as an integer specifying the order of the spline
            interpolator to use.
        interp_copy : bool, optional, default False
            If True, the interpolation function makes internal copies of x and y.
            If False, references to `x` and `y` are used.
        Attributes
        ----------
        interp_func_ : list
            The interpolation function for each feature in the training set.
        """

    def __init__(self, epsilon=1e-4, copy=True, n_jobs=None, interp_kind='linear', interp_copy=False):
        self.epsilon = epsilon
        self.copy = copy
        self.interp_kind = interp_kind
        self.interp_copy = interp_copy
        self.fill_value = 'extrapolate'
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit interpolation function to link rank with original data for future scaling
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data used to fit interpolation function for later scaling along the features axis.
        y
            Ignored
        """
        X = check_array(X, copy=self.copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite=True)

        self.interp_func_ = Parallel(n_jobs=self.n_jobs)(delayed(self._fit)(x) for x in X.T)
        return self

    def _fit(self, x):
        x = self.drop_duplicates(x)
        rank = np.argsort(np.argsort(x))
        bound = 1.0 - self.epsilon
        factor = np.max(rank) / 2.0 * bound
        scaled_rank = np.clip(rank / factor - bound, -bound, bound)
        return interp1d(
            x, scaled_rank, kind=self.interp_kind, copy=self.interp_copy, fill_value=self.fill_value)

    def transform(self, X, copy=None):
        """Scale the data with the Gauss Rank algorithm
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.
        """
        check_is_fitted(self, 'interp_func_')

        copy = copy if copy is not None else self.copy
        X = check_array(X, copy=copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite=True)

        X = np.array(Parallel(n_jobs=self.n_jobs)(delayed(self._transform)(i, x) for i, x in enumerate(X.T))).T
        return X

    def _transform(self, i, x):
        return erfinv(self.interp_func_[i](x))

    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.
        """
        check_is_fitted(self, 'interp_func_')

        copy = copy if copy is not None else self.copy
        X = check_array(X, copy=copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite=True)

        X = np.array(Parallel(n_jobs=self.n_jobs)(delayed(self._inverse_transform)(i, x) for i, x in enumerate(X.T))).T
        return X

    def _inverse_transform(self, i, x):
        inv_interp_func = interp1d(self.interp_func_[i].y, self.interp_func_[i].x, kind=self.interp_kind,
                                   copy=self.interp_copy, fill_value=self.fill_value)
        return inv_interp_func(erf(x))

    @staticmethod
    def drop_duplicates(x):
        is_unique = np.zeros_like(x, dtype=bool)
        is_unique[np.unique(x, return_index=True)[1]] = True
        return x[is_unique]

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

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted

class DistanceDepthFeaturizer(TransformerMixin, BaseEstimator):
    """
    """

    def __init__(
            self,
            columns_dict,
            normalize=True,
            copy=True,
    ):
        self.normalize = normalize
        self.columns_dict = columns_dict
        self.copy = copy
        self.scaler = StandardScaler()
        self.group_means_dict = {}
        self.target_groups_dict = {}

    def fit(self, X, y=None):
        """
        """
        # X = check_array(
        #     X, copy=True, force_all_finite=False, estimator=self
        # )
        X_ = X.copy()

        if self.normalize:
            X_ = self.scaler.fit_transform(X_)

        X_ = pd.DataFrame(X_)
        X_.columns = X.columns

        X_["target___"] = y

        for key in self.columns_dict:

            mean_dict = {col: 'mean' for col in self.columns_dict[key]}

            self.group_means_dict[key] = (
                X_
                    .groupby("target___")
                    .agg(mean_dict)
                    .reset_index()
            )

            self.target_groups_dict[key] = (
                self
                    .group_means_dict[key]
                    .target___
                    .to_list()
            )

        return self

    def transform(self, X):
        """
        """
        check_is_fitted(self, "group_means_dict")

        for key in self.columns_dict:

            # This, we can vectorize
            for target in self.target_groups_dict[key]:
                match = self.group_means_dict[key].target___ == target
                centroid = self.group_means_dict[key].loc[
                    match,
                    self.columns_dict[key]
                ]
                acum = 0
                for col in self.columns_dict[key]:
                    acum += (centroid[col].values[0] - X[col]) ** 2

                X[f"target__distance__{key}__{target}"] = acum

        return X


# X = pd.DataFrame(
#     dict(
#         a=[1, 2, 3],
#         b=[1, 0, 1],
#         c=[2, 0, 2]
#     )
# )
#
# y = [1, 0, 1]
#
# print(DistanceDepthFeaturizer({"ab": ['a', 'b'], "bc": ['b', 'c']}).fit_transform(X, y))


def feature_engineering_cls(df: pd.DataFrame) -> pd.DataFrame:
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

    df = df
    return df