import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz

from .preprocess import smote


def tenfold_cv(clf, X, y, smote_data=False, threshold=0.5):
    tn, fp, fn, tp = 0, 0, 0, 0

    folds = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    for trn_idx, tst_idx in folds.split(X, y):
        # Get the data from the split
        X_train, X_test = X[trn_idx], X[tst_idx]
        y_train, y_test = y[trn_idx].ravel(), y[tst_idx].ravel()

        if smote_data:
            X_train, y_train = smote(X_train, y_train)

        # Train and test
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)
        y_pred = [1 if x >= threshold else 0 for x in y_prob[:, 1]]

        # Show the results of this fold
        conf = confusion_matrix(y_test, y_pred)
        # print_confusion_matrix(conf, ['Settled', 'Chargeback'])

        # Sum for the total confusion matrix
        _tn, _fp, _fn, _tp = conf.ravel()
        tn += _tn
        fp += _fp
        fn += _fn
        tp += _tp

    print('TP: ' + str(tp))
    print('FP: ' + str(fp))
    print('FN: ' + str(fn))
    print('TN: ' + str(tn))
    print("\n")


def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig
