# %%
import random
import time

import matplotlib.pyplot as plt
import numpy as np

from data import infected_filter, get_most_frequent
from reservoir import Reservoir
from sketch import CountMinSketch


def plot_bar_tests(tests, title, legend):
    i = 0
    l = len(tests)
    fig, ax = plt.subplots()
    ips = real_distribution.keys()
    xs = range(len(real_distribution))
    adjusted_xs = [x - 0.3 + 0.6 / l * i for x in xs]
    ax.bar(adjusted_xs, list(real_distribution.values()), width=0.2, align='center', alpha=0.8)
    i += 1
    for distributions, means, errors in tests:
        adjusted_xs = [x - 0.3 + 0.6 / l * i for x in xs]
        if errors is not None:
            ax.bar(adjusted_xs, means, width=0.2, align='center', alpha=0.5, yerr=errors, ecolor='black', capsize=2)
        else:
            ax.bar(adjusted_xs, means, width=0.2, align='center', alpha=0.5)

        i += 1
    ax.set_xticks(xs)
    ax.set_xticklabels(ips, rotation=30, ha='right')
    ax.set_ylabel('Distribution')
    ax.set_title(f'{title}: Distribution of ip\'s')
    ax.yaxis.grid(True)
    ax.legend(legend)
    # Save the figure and show
    plt.tight_layout()
    # plt.savefig('bar_plot_with_error_bars.png')
    plt.show()
    print(real_distribution)
    for t in tests:
        print(t[0], t[1])


# %% Methods to perform the analysis of the dataset
# Count the real infected connections that were made to or from the host
def test_normal():
    start = time.time()
    filter = infected_filter(data_path)
    dist = get_most_frequent(filter)
    real_time = time.time() - start

    return dist, real_time


# CountMinSketch
def test_cm(epsilon, delta):
    start = time.time()

    filter = infected_filter(data_path)
    cm = CountMinSketch(epsilon=epsilon, delta=delta).analyse_stream(filter)
    cm_distribution = cm.get_distribution(real_distribution.keys())

    elapsed = time.time() - start

    return cm_distribution, elapsed


# Reservoir sampling
def test_reservoir(size):
    start = time.time()

    filter = infected_filter(data_path)
    res = Reservoir(filter, size)
    res_distribution = res.get_distribution()

    elapsed = time.time() - start
    return res_distribution, elapsed


# %% Methods for testing
def run_multiple(runs, func):
    distributions = {}

    for ip in real_distribution:
        distributions[ip] = []

    for _ in range(runs):
        random.seed(time.time_ns())
        dist, _ = func()
        for ip in distributions:
            d = 0
            if ip in dist:
                d = dist[ip]
            distributions[ip].append(d)

    means = []
    errors = []
    for ip, dist in distributions.items():
        mean = np.mean(dist)
        err = np.std(dist)
        distributions[ip] = (mean, err)
        means.append(mean)
        errors.append(err)

    if np.mean(errors) == 0:
        errors = None
    return distributions, means, errors


# Run a function multiple times and time it
def timing_test(fn, runs=10):
    times = list()

    for _ in range(runs):
        _, t = fn()
        times.append(t)

    return np.mean(times), np.std(times)


if __name__ == '__main__':
    # %%
    # Count the real infected connections that were made to or from the host
    data_path = "data/capture20110812.pcap.netflow.labeled"
    print('Compute Real Distribution')
    real_distribution, _ = test_normal()

    # %%
    print('Reservoir Tests')
    reservoir_tests = [
        run_multiple(10, lambda: test_reservoir(100)),
        run_multiple(10, lambda: test_reservoir(1000)),
        run_multiple(10, lambda: test_reservoir(10000)),
    ]
    # %%
    plot_bar_tests(reservoir_tests, title='Reservoir', legend=('Real', '100', '1000', '10000'))

    # %%
    print('CountMinSketch Tests')
    countmin_tests = [
        run_multiple(1, lambda: test_cm(0.25, 0.25)),
        run_multiple(1, lambda: test_cm(0.1, 0.1)),
        run_multiple(1, lambda: test_cm(0.01, 0.01)),
    ]
    # %%
    plot_bar_tests(countmin_tests, title='CountMinSketch Tests',
                   legend=('Real', '(0.25, 0.25)', '(0.1, 0.1)', '(0.01, 0.01)'))

    # %% Run a method multiple times and time it
    print('Timing Tests')
    normal_t, normal_std = timing_test(lambda: test_normal())
    cm_t_1, cm_std_1 = timing_test(lambda: test_cm(0.25, 0.25))
    cm_t_2, cm_std_2 = timing_test(lambda: test_cm(0.1, 0.1))
    cm_t_3, cm_std_3 = timing_test(lambda: test_cm(0.01, 0.01))
    res_t_1, res_std_1 = timing_test(lambda: test_reservoir(100))
    res_t_2, res_std_2 = timing_test(lambda: test_reservoir(1000))
    res_t_3, res_std_3 = timing_test(lambda: test_reservoir(10000))
