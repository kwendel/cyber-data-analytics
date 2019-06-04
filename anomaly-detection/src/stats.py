from datetime import datetime

import pandas as pd
from sklearn.metrics import confusion_matrix

try:
    from .data import get_attack_dates, label_data
except (ImportError, ModuleNotFoundError):
    from data import get_attack_dates, label_data


def __get_df_and_flags(df: pd.DataFrame, flagged_indices: pd.Int64Index):
    flags = pd.Series(False, index=df.index)
    flags[flagged_indices] = True

    # if df is dataset form batadal training 2, we only take into account the
    # values from the first of september for fair comparison, since the attacks
    # begin in september and the data before is used for training in the ARMA task
    if df['datetime'].iloc[0].year == 2016:
        start = 1416  # start at 00h the first of September
        df = df[start:]
        flags = flags[start:]

    return df, flags


def confusion(df: pd.DataFrame, flagged_indices: pd.Int64Index):
    df, flags = __get_df_and_flags(df, flagged_indices)
    return confusion_matrix(df['flag'], flags)


def print_precision_recall(conf_matrix):
    tp = conf_matrix[1][1]
    fp = conf_matrix[0][1]
    fn = conf_matrix[1][0]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print(f"precision: {precision}\trecall:{recall}")


def print_confusion_matrix(cm):
    labels = ['Normal', 'Attack']
    print(pd.DataFrame(cm, labels, labels))


def detection_duration(df: pd.DataFrame, flagged_indices: pd.Int64Index):
    df, flags = __get_df_and_flags(df, flagged_indices)
    dates = get_attack_dates(df)
    first = dict()
    for date_range in dates:
        f = -1
        flagged_in_range = df[flags].set_index('datetime')[date_range[0]:date_range[1]]
        if len(flagged_in_range) > 0:
            delta = flagged_in_range.index[0] - datetime.strptime(date_range[0], '%Y-%m-%d %H:%M:%S')
            hours, remainder = divmod(delta.total_seconds(), 3600)
            f = hours
            if remainder > 0:
                print("WARN: remainder > 0")
        first[date_range[0]] = f
    return first
