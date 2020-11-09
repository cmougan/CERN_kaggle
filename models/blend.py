import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

train_raw = pd.read_csv("data/train.csv").drop(columns="BUTTER")

lgbm_df = pd.read_csv('data/blend/valid_lgbm.csv')
fastai_df = pd.read_csv('data/blend/valid_fastai_nn.csv')

lgbm_submission = pd.read_csv("submissions/more_features_lgbm.csv")
fastai_submission = pd.read_csv("submissions/fastai_nn.csv")


train_raw = (train_raw
             .set_index("Id")
             .reindex(index=lgbm_df["Id"])
             .reset_index()
             )


valid_target = train_raw.loc[:, ["Id", "signal"]]


for w_fastai in np.linspace(0, 1, 50):
    print(f"weight fastai {w_fastai:.2f}")
    fastai_contr = fastai_df["prediction"] ** (w_fastai)
    lgmb_contr = lgbm_df["prediction"] ** (1 - w_fastai)

    roc = roc_auc_score(
        valid_target.signal,
        fastai_contr * lgmb_contr
    )
    print(f"roc product {roc:.4f}")


blend_submission = lgbm_submission.copy()

blend_submission["Predicted"] = \
    (lgbm_submission["Predicted"] ** 0.2) * \
    (fastai_submission["Predicted"] ** 0.8)

blend_submission.to_csv(
    "submissions/blend_lgbm_fastai_80.csv",
    index=False
)
