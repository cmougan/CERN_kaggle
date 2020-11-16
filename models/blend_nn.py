import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

train_raw = pd.read_csv("data/train.csv").drop(columns="BUTTER")

fastai_df = pd.read_csv('data/blend/valid_fastai_nn_grt.csv')
fastai_df_q = pd.read_csv('data/blend/valid_fastai_nn.csv')
fastai_df_single = pd.read_csv('data/blend/valid_fastai_nn_single_grt.csv')
fastai_df_single_q = pd.read_csv('data/blend/valid_fastai_nn_single.csv')
resnet = pd.read_csv('data/blend/res1_valid_preds.csv')
resnet_gauss = pd.read_csv('data/blend/valid_gaussian_res.csv')


fastai_submission = pd.read_csv("submissions/fastai_nn_grt.csv")
fastai_submission_q = pd.read_csv("submissions/fastai_nn.csv")
fastai_submission_single = pd.read_csv("submissions/fastai_nn_grt_single.csv")
fastai_submission_single_q = pd.read_csv("submissions/fastai_nn_single.csv")
resnet_submission = pd.read_csv('submissions/resnet.csv')
resnet_gauss_submission = pd.read_csv('submissions/nn_11_15_average_nan.csv')

resnet_gauss_submission["Predicted"] = (
    resnet_gauss_submission["Predicted"] \
        .fillna(resnet_submission["Predicted"])
)

train_raw = (train_raw
             .set_index("Id")
             .reindex(index=fastai_df["Id"])
             .reset_index()
             )


valid_target = train_raw.loc[:, ["Id", "signal"]]

max_roc = 0
optimal_w = 0

w_grt = 1 / 3
w_rs_gauss = 1

w_no_grt = ((1 - 2 * w_grt) / 2)
w_rs_no_gauss = 1 - w_rs_gauss

for w_fai in np.linspace(1, 0, 50):
    print(f"weight fastai {w_fai:.2f}")
    w_resnet = 1 - w_fai
    fastai_contr = fastai_df["prediction"] ** (w_grt * w_fai)
    fastai_contr_q = fastai_df_q["prediction"] ** (w_no_grt * w_fai)
    resnet_gauss_contr = resnet_gauss["Predicted"] ** (w_rs_gauss * w_resnet)
    resnet_contr = resnet["Predicted"] ** (w_rs_no_gauss * w_resnet)
    fastai_contr_single = fastai_df_single["prediction"] ** (w_grt * w_fai)
    fastai_contr_single_q = fastai_df_single_q["prediction"] ** (w_no_grt * w_fai)

    roc = roc_auc_score(
        valid_target.signal,
        fastai_contr_single + \
        fastai_contr_single_q + \
        fastai_contr + \
        fastai_contr_q + \
        resnet_contr + \
        resnet_gauss_contr
    )
    print(f"roc product {roc:.6f}")

    if roc > max_roc:
        optimal_w = w_fai
        max_roc = roc

print(f"max roc {max_roc:.6f}")
print(f"best w {optimal_w:.2f}")

blend_submission = fastai_submission.copy()

w_fai = optimal_w
w_resnet = 1 - w_fai

is_this_1 = 2 * (w_grt * w_fai) + \
            2 * (w_no_grt * w_fai) + \
            (w_rs_no_gauss * w_resnet) + \
            (w_rs_gauss * w_resnet)

print(f"Is this one? {is_this_1}")

blend_submission["Predicted"] = \
    (fastai_submission["Predicted"] ** (w_grt * w_fai)) + \
    (fastai_submission_single["Predicted"] ** (w_grt * w_fai)) + \
    (fastai_submission_q["Predicted"] ** (w_no_grt * w_fai)) + \
    (fastai_submission_single_q["Predicted"] ** (w_no_grt * w_fai)) + \
    (resnet_gauss_submission["Predicted"] ** (w_rs_gauss * w_resnet)) + \
    (resnet_submission["Predicted"] ** (w_rs_no_gauss * w_resnet))


blend_submission.to_csv(
    "submissions/blend_fastai_resnet_09.csv",
    index=False
)
