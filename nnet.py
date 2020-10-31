import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class ReadDataset(Dataset):
    """Students Performance dataset."""

    def __init__(self, csv_file,for_test=False,test_path='test.csv'):
        """Initializes instance of class StudentsPerformanceDataset.

        Args:
            csv_file (str): Path to the csv file with the students data.

        """
        df = pd.read_csv(csv_file).drop(columns="BUTTER")
        df.columns = df.columns.str.replace(" ", "")

        self.df = df
        self.df = self.transform(df)

        # Target
        self.target = "signal"

        # Save target and predictors
        self.X = self.df.drop(self.target, axis=1)
        print(self.X.shape)

        self.y = self.df[self.target]

        # Normalize
        self.scaler = MinMaxScaler()
        self.X = pd.DataFrame(
            self.scaler.fit_transform(self.X), columns=self.X.columns
        )
        if for_test:
            self.df = pd.read_csv(test_path).drop(columns="BUTTER")
            self.df.columns = self.df.columns.str.replace(" ", "")

            self.df = self.transform(self.df)
            print(self.df.shape)

            self.df = self.scaler.transform(self.df)
            self.X = self.df


    def __len__(self):
        return len(self.df)

    def transform(self, df):
        # Did not work

        df["Kst_892_0_cosThetaH_arc"] = np.arccos(df["Kst_892_0_cosThetaH"])
        df["Kst_892_0_cosThetaH_arc_sin"] = np.sin(df.Kst_892_0_cosThetaH_arc)

        df["Kplu_pXKst_892costheta"] = df["Kplus_P"] * df["Kst_892_0_cosThetaH"]

        df["momentum_by_shortest_dist"] = df["Kplus_P"] / df["Kplus_IP_OWNPV"]

        df["momentum_sum"] = df["B_PT"] + df["Kplus_P"] + df["gamma_PT"]

        # Worked individually
        df["Kplu_p_div_Kst_892costheta"] = df["Kplus_P"] / df["Kst_892_0_cosThetaH"]

        # All mom ratios worked together
        df["mom_rat_1"] = df["B_PT"] / df["Kplus_P"]
        df["mom_rat_2"] = df["B_PT"] / df["gamma_PT"]

        df["mom_rat_3"] = df["gamma_PT"] / df["B_PT"]
        df["mom_rat_4"] = df["gamma_PT"] / df["Kplus_P"]

        df["mom_rat_5"] = df["Kplus_P"] / df["B_PT"]
        df["mom_rat_6"] = df["Kplus_P"] / df["gamma_PT"]

        # B meson ratios
        df["mesB_ratio_1"] = df["B_FDCHI2_OWNPV"] / df["B_IPCHI2_OWNPV"]
        df["mesB_ratio_2"] = df["B_FDCHI2_OWNPV"] / df["B_PT"]

        df["mesB_ratio_3"] = df["B_IPCHI2_OWNPV"] / df["B_PT"]
        df["mesB_ratio_4"] = df["B_IPCHI2_OWNPV"] / df["B_PT"]

        df["mesB_ratio_5"] = df["B_PT"] / df["B_IPCHI2_OWNPV"]
        df["mesB_ratio_6"] = df["B_PT"] / df["B_FDCHI2_OWNPV"]

        # Neither Improved neither worst
        df["kst_thetaH"] = np.arccos(df.Kst_892_0_cosThetaH) * 180 / np.pi
        df["kst_thetaH_sin"] = np.sin(df["kst_thetaH"])
        df["kst_thetaH_cos"] = np.cos(df["kst_thetaH"])
        df["kst_thetaH_sin_cos"] = np.sin(df["kst_thetaH"]) - np.cos(df["kst_thetaH"])
        df["kst_thetaH_tan"] = np.tan(df["kst_thetaH"])
        df["kst_thetaH_exp"] = np.exp(df["kst_thetaH"])
        df["kst_thetaH_exp_1"] = np.exp(-df["kst_thetaH"])

        df["B_DIRA_OWNPV_angle"] = np.arccos(df["B_DIRA_OWNPV"]) * 180 / np.pi
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

        self.df = df
        return self.df

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
        self.fc1 = nn.Linear(input_dim, 2 * input_dim)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(2 * input_dim, 3 * input_dim)
        self.relu2 = nn.LeakyReLU()

        self.fc3 = nn.Linear(3 * input_dim, 2 * input_dim)
        self.relu3 = nn.LeakyReLU()

        self.fc4 = nn.Linear(2 * input_dim, 1 * input_dim)
        self.relu4 = nn.LeakyReLU()

        self.fc5 = nn.Linear(input_dim, 1)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.relu3(x)

        x = self.fc4(x)
        x = self.relu4(x)

        x = self.fc5(x)
        x = self.sig(x)

        return x.squeeze()

    def step(self, inputs):
        data, label = inputs  # ignore label
        outputs = self.forward(data)
        _, preds = torch.max(outputs.data, 1)
        # preds, outputs  are cuda tensors. Right?
        return preds, outputs

    def predict(self, dataloader):
        '''
                # Apply softmax to output.
        pred = F.softmax(self.forward(x))
        ans = []

        # Pick the class with maximum weight
        for t in pred:
            ans.append(torch.argmax(t))

        return torch.tensor(ans)
        '''
        for i, batch in enumerate(dataloader):
            pred, output = self.step(batch)
            prediction_list.append(pred.cpu())


