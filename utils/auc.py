from functools import partial
from sklearn import metrics
from scipy.optimize import fmin
import numpy as np
from scipy import stats

class OptimizeAUC:
    """
    Class for optimizing AUC
    This class is all you need to find best weights for
    any model and for any metric and for any type of predictions
    With very small changes, this class can be used for optimization of
    weights in ensemble models for _any_ type of predictions
    """
    def __init__(self):
        self.coef_ = 0
        self.classes_ = [0, 1]

    def get_params(self, deep):
        return {}

    def _auc(self, coef, X, y):
        """
        This functions calculates and returns AUC
        :param coef: coef list, of the same length as number of models
        :param X: predictions, in this case a 2d array
        :param y: targets, in our case binary 1d array
        """

        # multiply coefficients with every column of the array
        # with predictions.
        # this means: element 1 of coef is multiplied by column 1
        # of the prediction array, element 2 of coef is multiplied
        # by column 2 of the prediction array and so on!
        x_coef = X * coef
        # create predictions by taking row wise sum
        predictions = np.sum(x_coef, axis=1)

        # calculate auc score
        auc_score = metrics.roc_auc_score(y, predictions)

        #return negative auc
        return -1.0 * auc_score

    def fit(self, X, y):
        # remember partial from hypeparamer optimization chapter?
        loss_partial = partial(self._auc, X=X, y=y)

        # dirichlet distribution. you can use any distribution you want
        # to initialize the coefficients
        # we want the coefficients to sum to 1
        initial_coef = np.random.dirichlet(np.ones(X.shape[1]), size=1)

        # use scipy fmin to minimize the loss function, in our case auc
        self.coef_ = fmin(loss_partial, initial_coef, disp=False)

    def predict(self, X):
        # this is similar to _auc function
        x_coef = X * self.coef_
        predictions = np.sum(x_coef, axis=1)
        return predictions

    def predict_proba(self, X):
        return self.predict(X)


def rank_mean(probas):

    ranked = []

    for i in range(probas.shape[1]):
        rank_data = stats.rankdata(probas[:, i])
        ranked.append(rank_data)

    ranked = np.column_stack(ranked)
    return np.mean(ranked, axis=1)