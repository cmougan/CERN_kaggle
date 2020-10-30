import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.style.use("seaborn-whitegrid")
rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8


def return_impurity(classifier, sample):
    """

    :param classifier: trained decision tree
    :param sample: Instance to explain
    :return: dataframe with explained path
    """
    feature = classifier.tree_.feature
    impurity = classifier.tree_.impurity
    samples = classifier.tree_.weighted_n_node_samples

    # The path that we want
    node_indicator = classifier.decision_path(sample)
    leave_id = classifier.apply(sample)

    # Extract the visited nodes
    ## Since in theory we are only going to have one sample
    ## we can hard code 0 here
    sample_id = 0
    node_index = node_indicator.indices[
        node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
    ]

    # Iterate through nodes and get information about them
    feat = []
    imp = []
    sam = []
    for node_id in node_index:
        feat.append(feature[node_id])
        imp.append(impurity[node_id])
        sam.append(samples[node_id])

    # Return dataframe
    res = pd.DataFrame(data=[feat, imp, sam]).transpose()
    res.columns = ["feature", "impurity", "samples"]
    ## Name of the columns
    res["feature"] = sample.columns[res.feature.astype(int).values].values

    ## Calculate the impurity gain with respect to each path
    res["imp_child"] = res.impurity.shift(-1)
    res["samples_child"] = res.samples.shift(-1)

    # <center>impurity_gain = $\frac{N_t}{N}(impurity - \frac{N_c}{N_t}*impurity\_child)=$</center>
    # <center>$=\frac{1}{N}(N_t * impurity - N_c*impurity\_child)$</center>

    # - $N$ total samples
    # - $N_t$ samples at father
    # - $N_c$ samples at child
    # - impurity - Gini impurity of father
    # - impurity_child - Gini impurity of child
    N = res.iloc[0].samples
    res["impurity_gain"] = (
        1
        / N
        * (res["samples"] * res["impurity"] - res["samples_child"] * res["imp_child"])
        * 100
    )

    # Samples gain
    res["samples_gain"] = 1 / N * (res.samples - res.samples_child)

    return res


def pie_plot_filtered(data, metric, title=""):
    '''
    pie_plot_filtered(res_g, "impurity_gain", title="Impurity Gain")
    :param data:
    :param metric:
    :param title:
    :return:
    '''
    plt.figure()
    plt.title(title)
    aux = data[data[metric] > 0][[metric, "feature"]]
    aux[metric] = aux[metric] / aux[metric].sum()
    plt.pie(
        aux[metric], labels=aux.feature, autopct="%1.1f%%", shadow=True, startangle=90
    )
    plt.show()


def similar_intances(data_X, data_y, classifier, instance):
    '''
    similar_intances(X,y,dt,X.iloc[[instance]])
    :param data_X:
    :param data_y:
    :param classifier:
    :param instance:
    :return:
    '''
    data = data_X.copy()
    data['target'] = data_y
    leaf = classifier.apply(instance)[0]

    data['leaf'] = classifier.apply(data_X)
    return data[data['leaf'] == leaf]


def contrastive_explainer(classifier,sample, is_ensemble=False, ensembler=''):
    """

    :param classifier: trained decision tree
    :param sample: Instance to explain
    :return: dataframe with explained path
    """

    # Extract an array with the information that we want
    feature = classifier.tree_.feature
    impurity = classifier.tree_.impurity
    samples = classifier.tree_.weighted_n_node_samples
    threshold = classifier.tree_.threshold

    # The path that we want
    node_indicator = classifier.decision_path(sample)
    leave_id = classifier.apply(sample)

    # Extract the visited nodes
    ## Since in theory we are only going to have one sample
    ## we can hard code 0 here
    sample_id = 0
    node_index = node_indicator.indices[
                 node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]
                 ]

    # Iterate through nodes and get information about them
    feat = []
    imp = []
    sam = []
    thres = []
    for node_id in node_index:
        feat.append(feature[node_id])
        imp.append(impurity[node_id])
        sam.append(samples[node_id])
        thres.append(threshold[node_id])

    # Return dataframe
    res = pd.DataFrame(data=[feat, imp, sam, thres]).transpose()
    res.columns = ["feature", "impurity", "samples", "threshold"]
    ## Name of the columns
    res["feature"] = sample.columns[res.feature.astype(int).values].values

    # Create some list where we will be appending the information
    feat = []
    orig = []
    thres = []
    mod = []
    pred_init = []
    pred_after = []

    # Iterate through the nodes and modify variable
    for index, row in res.iterrows():

        ins = sample.copy()

        # En el caso en el que sea menor o igual, lo cambiamos a justo por encima del threshold
        if sample[row["feature"]][0] <= row["threshold"]:
            ins[row["feature"]] = row["threshold"] + 0.1

        # En caso de que sea mayor, le damos el threshold
        else:
            ins[row["feature"]] = row["threshold"]

        if is_ensemble:
            original = ensembler.predict_proba(sample)[0][0]
            modified = ensembler.predict_proba(ins)[0][0]
        else:
            original = classifier.predict_proba(sample)[0][0]
            modified = classifier.predict_proba(ins)[0][0]

        feat.append(row["feature"])
        thres.append(row["threshold"])
        orig.append(sample[row["feature"]][0])
        mod.append(ins[row["feature"]][0])
        pred_init.append(original)
        pred_after.append(modified)

    # Create data frame with the information we are looking for
    contrastive = pd.DataFrame(
        data=[feat, orig, mod, thres, pred_init, pred_after]
    ).transpose()
    contrastive.columns = [
        "feature",
        "original",
        "modified",
        "threshold",
        "pred_initial",
        "pred_after",
    ]

    # Calculate the difference in the prediction
    contrastive["pred_change"] = contrastive["pred_initial"] - contrastive["pred_after"]

    return contrastive.iloc[contrastive['pred_change'].abs().argsort()[::-1]]