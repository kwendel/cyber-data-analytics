import pandas as pd
from sklearn.metrics import confusion_matrix


def confusion(df: pd.DataFrame, flagged_indices: pd.Int64Index):
    flags = pd.Series(False, index=df.index)
    flags[flagged_indices] = True

    # if df is dataset form batadal training 2, we only take into account the
    # values from the first of september for fair comparison, since the attacks
    # begin in september and the data before is used for training in the ARMA task
    if df['datetime'].iloc[0].year == 2016:
        start = 1416  # start at 00h the first of September
        df = df[start:]
        flags = flags[start:]

    return confusion_matrix(df['flag'], flags)


def print_confusion_matrix(cm):
    labels = ['Normal', 'Attack']
    print(pd.DataFrame(cm, labels, labels))
