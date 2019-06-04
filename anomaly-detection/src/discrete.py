from collections import defaultdict
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Fix that relative imports work for both notebooks and main methods
try:
    from .data import parse_to_df, label_data
    from .stats import print_confusion_matrix, confusion
except ModuleNotFoundError:
    from data import parse_to_df, label_data
    from stats import print_confusion_matrix, confusion

path_training_1 = '../data/BATADAL_training1.csv'
path_training_2 = '../data/BATADAL_training2.csv'
path_testing = '../data/BATADAL_test.csv'


def update_diff_mean(segment):
    return (segment.values[-1] - segment.values[0]) / (len(segment) - 1)


def get_segment_end(series, anchor, max_err):
    # Next point from anchor point
    i = anchor + 1

    # Compute the first mean diff to initialize the line segment
    diff_mean = series[i] - series[anchor]

    # Add points to the segment that have equal deviation from the segment mean
    while abs((series[i] - series[i - 1]) - diff_mean) <= max_err:
        # Set next point
        i += 1

        # Point was below the threshold so we add it to the segment
        diff_mean = update_diff_mean(series[anchor: i])  # exclusive indexing

        # Stop if the end is reached
        if i >= len(series) - 1:
            return i, diff_mean

    # Return i-1 as end point as point i did not fit the segment
    if i - 1 == anchor:
        # Segment was only one part long so  diff_mean must be zero
        return i - 1, 0
    else:
        return i - 1, diff_mean
    # return i - 1, diff_mean


eps = 0.001
slow = 0.25

labels = {
    'constant': 'SC',
    'slow up': 'SU',
    'slow down': 'SD',
    'quick down': 'QD',
    'quick up': 'QU'
}


def discretize(val):
    if abs(val) < eps:
        return labels['constant']
    elif abs(val) < slow:
        if np.sign(val) == 1.0:
            return labels['slow up']
        else:
            return labels['slow down']
    else:
        if np.sign(val) == 1.0:
            return labels['quick up']
        else:
            return labels['quick down']


def swide(signalseries: pd.Series, max_err):
    # Signals are normalized as this makes computing SWIDE easier
    t = signalseries.copy()
    enc = StandardScaler()
    t = pd.Series(enc.fit_transform(t.values.reshape(-1, 1)).flatten(), index=t.index)
    diff_means = pd.Series(np.zeros_like(t.values), index=t.index)

    # Start with zero index
    anchorpoints = [0]
    # diffs = []
    anchor = 0

    # SWIDE algorithm
    while True:
        # Compute a segment and get the end point
        end, diff_mean = get_segment_end(t, anchor, max_err)

        # Adds this end point to the anchors
        diff_means[anchor:end] = diff_mean
        anchorpoints.append(anchor)

        # Make the end point the new anchor
        anchor = end

        # Stop if we reached the end of the series
        if anchor >= len(t) - 1:
            break

    plt.figure()
    sns.lineplot(t.index, t.values, label="Signal")
    plt.vlines(anchorpoints, ymin=t.min(), ymax=t.max(), colors='red', label="Anchors")
    plt.title("Discrete signal")
    plt.xlabel("Hours from T=0")
    plt.ylabel("Normalized signal")
    plt.legend()

    # plt.savefig("../plots/discrete_signal.png")
    plt.show()

    # Map to a discrete value based on the average dist from the mean/gradient
    discrete = diff_means.copy()
    discrete = discrete.map(discretize)

    return pd.concat((signalseries, diff_means.rename("m_diff"), discrete.rename("discreet")), axis=1)
    # return res


def get_ngram_freq(df, n):
    freqs = defaultdict(int)
    time_step = n

    # Compute the Ngram freq
    while time_step < len(df):
        m = df.iloc[time_step - n:time_step]['discreet']
        ngram = "".join(m.values)

        freqs[ngram] += 1
        time_step += 1

    # Laplace smoothing - add one to each count
    possibilities = product(labels.values(), repeat=n)
    for pos in possibilities:
        ngram = "".join(pos)
        freqs[ngram] += 1

    return freqs


def detect_anomaly(df, freqs, threshold, n):
    time_step = n
    detected = pd.Series(np.zeros_like(df['discreet'].values), index=df.index)

    # Label as an attack if the ngram occurence is below the threshold
    while time_step < len(df):
        m = df.iloc[time_step - n:time_step]['discreet']
        ngram = "".join(m.values)

        if freqs[ngram] <= threshold:
            detected[time_step - n:time_step] = 1

        time_step += 1

    return pd.concat((df, detected.rename('attack')), axis=1)


def test_signal(df_train, df_test, signal_name, ngram, ngram_thr, discrete_thr):
    # Compute the ngram frequencies
    signal = df_train[signal_name]

    # signal.plot()
    # plt.show()

    res = swide(signal, discrete_thr)
    freqs = get_ngram_freq(res, n=ngram)

    # Save important columns
    datetimes = None
    if 'datetime' in df_test.columns:
        datetimes = df_test['datetime']

    flags = None
    if 'att_flag' in df_test.columns:
        flags = df_test['att_flag']

    # Discretize test and check anomalies
    tst = swide(df_test[signal_name], discrete_thr)
    labeled = detect_anomaly(tst, freqs, threshold=ngram_thr, n=ngram)

    if datetimes is not None:
        labeled['datetime'] = datetimes
    if flags is not None:
        labeled['att_flag'] = flags

    attacks = labeled[labeled['attack'] == 1]
    return attacks


if __name__ == '__main__':
    sns.set(font_scale=1.25)
    df = parse_to_df(path_training_1)
    df_a = parse_to_df(path_training_2)
    df_test = parse_to_df(path_testing)

    # Level T1
    trn_attacks = test_signal(df, df_a, 'l_t1', ngram=4, ngram_thr=20, discrete_thr=0.25)
    print_confusion_matrix(confusion(label_data(df_a), trn_attacks.index))
    tst_attacks = test_signal(df, df_test, 'l_t1', ngram=5, ngram_thr=20, discrete_thr=0.25)
    print_confusion_matrix(confusion(label_data(df_test), tst_attacks.index))

    # Switch pump 4
    trn_attacks = test_signal(df, df_a, 's_pu10', ngram=3, ngram_thr=10, discrete_thr=0.25)
    print_confusion_matrix(confusion(label_data(df_a), trn_attacks.index))
    tst_attacks = test_signal(df, df_test, 'f_pu1', ngram=3, ngram_thr=5, discrete_thr=0.25)
    print_confusion_matrix(confusion(label_data(df_test), tst_attacks.index))
