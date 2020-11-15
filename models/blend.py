import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

train_raw = pd.read_csv("data/train.csv").drop(columns="BUTTER")

lgbm_df = pd.read_csv('data/blend/valid_lgbm.csv')
fastai_df = pd.read_csv('data/blend/valid_fastai_nn_grt.csv')
fastai_df_q = pd.read_csv('data/blend/valid_fastai_nn.csv')
fastai_df_single = pd.read_csv('data/blend/valid_fastai_nn_single_grt.csv')
fastai_df_single_q = pd.read_csv('data/blend/valid_fastai_nn_single.csv')
resnet = pd.read_csv('data/blend/res1_valid_preds.csv')

lgbm_submission = pd.read_csv("submissions/more_features_lgbm.csv")
fastai_submission = pd.read_csv("submissions/fastai_nn_grt.csv")
fastai_submission_q = pd.read_csv("submissions/fastai_nn.csv")
fastai_submission_single = pd.read_csv("submissions/fastai_nn_grt_single.csv")
fastai_submission_single_q = pd.read_csv("submissions/fastai_nn_single.csv")
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
    w_fastai = (4 * w_nn / 5)
    w_resnet = (1 * w_nn / 5)
    w_lgbm = 1 - w_nn
    w_grt = (2 / 6) * w_fastai
    w_no_grt = ((1 - 2 * w_grt) / 2) * w_fastai
    fastai_contr = fastai_df["prediction"] ** (w_grt)
    fastai_contr_q = fastai_df_q["prediction"] ** (w_no_grt)
    resnet_contr = resnet["Predicted"] ** (w_resnet)
    fastai_contr_single = fastai_df_single["prediction"] ** (w_grt)
    fastai_contr_single_q = fastai_df_single_q["prediction"] ** (w_no_grt)
    lgmb_contr = lgbm_df["prediction"] ** (w_lgbm)

    roc = roc_auc_score(
        valid_target.signal,
        fastai_contr_single + \
        fastai_contr_single_q + \
        fastai_contr + \
        fastai_contr_q + \
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
fastai_coef = (4 / 5) * (2 / 6) * optimal_nn_w
fastai_coef_q = (4 / 5) * (1 / 6) * optimal_nn_w
resnet_coef = (1 / 5)
print(2 * fastai_coef + 2 * fastai_coef_q + resnet_coef + (1 - optimal_nn_w))

blend_submission["Predicted"] = \
    (lgbm_submission["Predicted"] ** (1 - optimal_nn_w)) + \
    (fastai_submission["Predicted"] ** fastai_coef) + \
    (fastai_submission_single["Predicted"] ** fastai_coef) + \
    (resnet_submission["Predicted"] ** resnet_coef) + \
    (fastai_submission_q["Predicted"] ** fastai_coef_q) + \
    (fastai_submission_single_q["Predicted"] ** fastai_coef_q)

blend_submission.to_csv(
    "submissions/blend_fastai_resnet_415_2211.csv",
    index=False
)
