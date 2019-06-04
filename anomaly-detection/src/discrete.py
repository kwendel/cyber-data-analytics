from collections import defaultdict
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Fix that relative imports work for both notebooks and main methods
try:
    from .data import parse_to_df
except ModuleNotFoundError:
    from data import parse_to_df

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
        # Point was below the threshold so we add it to the segment
        diff_mean = update_diff_mean(series[anchor: i])

        # Stop if the end is reached
        if i >= len(series) - 1:
            return i, diff_mean

        # Set next point
        i += 1

    # Return i-1 as end point as point i did not fit the segment
    if i - 1 == anchor:
        # Segment was only one part long so  diff_mean must be zero
        return i - 1, 0
    else:
        return i - 1, diff_mean
    # return i - 1, diff_mean


eps = 0.001
# slow = 0.25
slow = 10.0

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
    t = signalseries.copy()
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
    sns.lineplot(t.index, t.values)
    plt.vlines(anchorpoints, ymin=t.min(), ymax=t.max(), colors='red')
    plt.show()

    # Map to a discrete value based on the average dist from the mean/gradient
    discrete = diff_means.copy()
    discrete = discrete.map(discretize)

    return pd.concat((t, diff_means.rename("m_diff"), discrete.rename("discreet")), axis=1)
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

    signal.plot()
    plt.show()

    res = swide(signal, discrete_thr)
    freqs = get_ngram_freq(res, n=ngram)

    # Save important columns
    flags = df_test['att_flag']
    datetime = df_test['datetime']

    # Discretize test and check anomalies
    tst = swide(df_test[signal_name], discrete_thr)
    labeled = detect_anomaly(tst, freqs, threshold=ngram_thr, n=ngram)

    labeled['att_flag'] = flags
    labeled['datetime'] = datetime

    return labeled


if __name__ == '__main__':
    sns.set()
    df = parse_to_df(path_training_1)
    df_a = parse_to_df(path_training_2)

    labeled = test_signal(df, df_a, 'f_pu10', ngram=5, ngram_thr=2, discrete_thr=15)
    print(len(labeled))
    detected = labeled[labeled['attack'] == 1]
    print(len(detected))
    print(len(detected[detected['att_flag'] == -999]))
    print(len(detected[detected['att_flag'] == 1]))
