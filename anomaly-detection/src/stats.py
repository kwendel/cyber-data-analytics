import pandas as pd
from sklearn.metrics import confusion_matrix


def confusion(df: pd.DataFrame, flagged_indices: pd.Int64Index):
    flags = pd.Series(False, index=df.index)
    flags[flagged_indices] = True

    return confusion_matrix(df['flag'], flags)


def print_confusion_matrix(cm):
    labels = ['Normal', 'Attack']
    print(pd.DataFrame(cm, labels, labels))
