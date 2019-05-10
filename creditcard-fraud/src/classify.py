import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, train_test_split

from .preprocess import smote
from .plot import tree


def tenfold_cv(clf, X, y, smote_data=False, threshold=0.5, predict=True, df=None):
    tn, fp, fn, tp = 0, 0, 0, 0
    preds = []

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

        if df is not None:
            tree(clf, df)

        if predict:
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
        else:
            preds.append((y_test, y_prob))

    if predict:
        print('TP: ' + str(tp))
        print('FP: ' + str(fp))
        print('FN: ' + str(fn))
        print('TN: ' + str(tn))
        print("\n")
    else:
        return preds


def predict_with_threshold(preds, threshold=0.5):
    tn, fp, fn, tp = 0, 0, 0, 0

    for (y_test, y_prob) in preds:
        y_pred = [1 if x >= threshold else 0 for x in y_prob[:, 1]]
        conf = confusion_matrix(y_test, y_pred)

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


def combine_preds(preds, preds1, preds2):
    tn, fp, fn, tp = 0, 0, 0, 0

    for i in range(len(preds)):
        (y_test, y_one) = preds[i]
        (y_test, y_two) = preds1[i]
        (y_test, y_three) = preds2[i]

        y_one = [1 if x >= 0.93 else 0 for x in y_one[:, 1]]
        y_two = [1 if x >= 0.91 else 0 for x in y_two[:, 1]]
        y_three = [1 if x >= 0.94 else 0 for x in y_three[:, 1]]

        pred = []

        for i in range(len(y_one)):
            vote = [y_one[i], y_two[i], y_three[1]]
            if vote.count(1) >= 2:
                pred.append(1)
            else:
                pred.append(0)

        conf = confusion_matrix(y_test, pred)
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


def roc(clf, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Without SMOTE
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)
    y_pred = y_prob[:, 1]

    # ROC analysis
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)

    # With SMOTE
    X_train, y_train = smote(X_train, y_train)

    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)
    y_pred = y_prob[:, 1]

    # ROC analysis
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'r', label='AUC = %0.2f (With SMOTE)' % roc_auc)

    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


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
