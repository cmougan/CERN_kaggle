import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

train_raw = pd.read_csv("data/train.csv").drop(columns="BUTTER")

lgbm_df = pd.read_csv('data/blend/valid_lgbm.csv')
fastai_df = pd.read_csv('data/blend/valid_fastai_nn.csv')
fastai_df_single = pd.read_csv('data/blend/valid_fastai_nn_single.csv')
resnet = pd.read_csv('data/blend/res1_valid_preds.csv')

lgbm_submission = pd.read_csv("submissions/more_features_lgbm.csv")
fastai_submission = pd.read_csv("submissions/fastai_nn.csv")
fastai_submission_single = pd.read_csv("submissions/fastai_nn_single.csv")
resnet_submission = pd.read_csv('submissions/resnet.csv')


train_raw = (train_raw
             .set_index("Id")
             .reindex(index=lgbm_df["Id"])
             .reset_index()
             )


valid_target = train_raw.loc[:, ["Id", "signal"]]

max_roc = 0
optimal_w = 0

for w_nn in np.linspace(0, 1, 50):
    print(f"weight fastai {w_nn:.2f}")
    w_fastai = (2 * w_nn / 3)
    w_resnet = (1 * w_nn / 3)
    w_lgbm = 1 - w_nn
    fastai_contr = fastai_df["prediction"] ** (w_fastai / 2)
    resnet_contr = resnet["Predicted"] ** (w_resnet)
    fastai_contr_single = fastai_df_single["prediction"] ** (w_fastai / 2)
    lgmb_contr = lgbm_df["prediction"] ** (w_lgbm)

    roc = roc_auc_score(
        valid_target.signal,
        fastai_contr_single + \
        fastai_contr + \
        lgmb_contr + \
        resnet_contr
    )
    print(f"roc product {roc:.6f}")

    if roc > max_roc:
        optimal_w = w_nn
        max_roc = roc

print(f"max roc {max_roc:.6f}")
print(f"best w {optimal_w:.2f}")

blend_submission = lgbm_submission.copy()


optimal_nn_w = optimal_w
blend_submission["Predicted"] = \
    (lgbm_submission["Predicted"] ** (1 - optimal_nn_w)) + \
    (fastai_submission["Predicted"] ** (1 * optimal_nn_w / 3)) + \
    (fastai_submission_single["Predicted"] ** (1 * optimal_nn_w / 3)) + \
    (resnet_submission["Predicted"] ** (1 * optimal_nn_w / 3))

blend_submission.to_csv(
    "submissions/blend_lgbm_fastai_resnet_02_111.csv",
    index=False
)
