import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_feature_importance(
    columnas, model_features, columns_ploted=10, model_name="Catboost"
):
    """
    This method is yet non-tested
    
    This function receives a set of columns feeded to a model, and the importance of each of feature.
    Returns a graphical visualization
    
    Call it fot catboost pipe example:
    plot_feature_importance(pipe_best_estimator[:-1].transform(X_tr).columns,pipe_best_estimator.named_steps['cb'].get_feature_importance(),20)
    
    Call it for lasso pipe example:
    plot_feature_importance(pipe_best_estimator[:-1].transform(X_tr).columns,np.array(pipe_best_estimator.named_steps['clf'].coef_.squeeze()),20)
    """

    feature_importance = pd.Series(index=columnas, data=np.abs(model_features))
    n_selected_features = (feature_importance > 0).sum()
    print(
        "{0:d} features, reduction of {1:2.2f}%".format(
            n_selected_features,
            (1 - n_selected_features / len(feature_importance)) * 100,
        )
    )
    plt.figure()
    feature_importance.sort_values().tail(columns_ploted).plot(
        kind="bar", figsize=(18, 6)
    )
    plt.title("Feature Importance for {}".format(model_name))
    plt.show()

