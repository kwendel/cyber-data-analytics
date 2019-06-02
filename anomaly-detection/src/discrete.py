import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
    return i - 1, diff_mean


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

    discrete = diff_means.copy()
    discrete = discrete.map(discretize)

    res = pd.concat((t, diff_means.rename("m_diff"), discrete.rename("discrete")), axis=1)
    return res

def detect_anomaly():
    # TODO:
    # - set sliding window size
    # - sliding window over data and count occurence of each possible n-grams
    # - smoothing??
    # - now do sliding window over data and check if Ngram has probability > threshold


if __name__ == '__main__':
    sns.set()
    df = parse_to_df(path_training_1)

    # Lets pick a signal and see how it looks
    signal = df.loc[0:100, 'l_t6']
    signal.plot()
    plt.show()
    res = swide(signal, 0.5 * signal.std())
