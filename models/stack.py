import pandas as pd
import numpy as np
import random
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from lightgbm import LGBMClassifier

random.seed(42)
np.random.seed(42)

train_raw = pd.read_csv("data/train.csv").drop(columns="BUTTER")

lgbm_df = pd.read_csv('data/blend/valid_lgbm.csv')
fastai_df = pd.read_csv('data/blend/valid_fastai_nn.csv')
fastai_df_single = pd.read_csv('data/blend/valid_fastai_nn_single.csv')
resnet = pd.read_csv('data/blend/res1_valid_preds.csv')
valid_df = pd.read_csv('data/stack/X_valid.csv')

lgbm_submission = pd.read_csv("submissions/more_features_lgbm.csv")
fastai_submission = pd.read_csv("submissions/fastai_nn.csv")
fastai_submission_single = pd.read_csv("submissions/fastai_nn_single.csv")
resnet_submission = pd.read_csv('submissions/resnet.csv')
test_df = pd.read_csv('data/stack/X_test.csv')

train_raw = (train_raw
             .set_index("Id")
             .reindex(index=lgbm_df["Id"])
             .reset_index()
             )

valid_target = train_raw.loc[:, ["Id", "signal"]]


def logit(x):
    return np.log(x / (1 - x))


predictions_df_valid = pd.DataFrame(
    dict(
        lgbm=logit(lgbm_df["prediction"].values),
        fastai=logit(fastai_df["prediction"].values),
        fastai_single=logit(fastai_df_single["prediction"].values),
        resnet=logit(resnet["Predicted"].values),
    )
)

predictions_df_valid = pd.concat([predictions_df_valid, valid_df], axis=1)

y = valid_target.signal.values

print(roc_auc_score(y, predictions_df_valid.fastai))

cv = StratifiedKFold(n_splits=4)

logreg = LogisticRegression(penalty='l1', solver='liblinear')
# logreg = LGBMClassifier()

cv_preds = cross_val_predict(
    logreg,
    predictions_df_valid,
    y,
    cv=cv,
    method='predict_proba'
)


print(
    roc_auc_score(
        y,
        cv_preds[:, 1]
    )
)

# logreg.fit(predictions_df_valid, y)
# print(logreg.coef_)