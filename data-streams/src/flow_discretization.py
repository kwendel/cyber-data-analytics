# %%
from collections import Counter, OrderedDict
from functools import wraps

import matplotlib.pyplot as plt

from data import process_file


def counter(func):
    @wraps(func)
    def wrapper_counter(*args, **kwargs):
        # Use the provided function to retrieve values and count them
        res = Counter(func(*args, **kwargs))

        # Normalize the data
        c = sum(res.values(), 0.0)
        for key in res:
            res[key] /= c

        return res

    return wrapper_counter


@counter
def count_protocol(data):
    for i in data:
        yield i.protocol


@counter
def count_duration(data):
    for i in data:
        duration = i.duration

        if duration == 0.0:
            yield "0"
        elif 0.0 < duration <= 1.0:
            yield "0-1"
        elif 1.0 < duration <= 2.0:
            yield "1-2"
        elif 2.0 < duration <= 3.0:
            yield "2-3"
        elif 3.0 < duration <= 4.0:
            yield "3-4"
        else:
            yield "4+"


@counter
def count_two(data):
    for i in data:
        duration = i.duration
        protocol = i.protocol

        if duration == 0.0:
            yield protocol + "-(0)"
        elif 0.0 < duration <= 1.0:
            yield protocol + "-(0-1)"
        elif 1.0 < duration <= 2.0:
            yield protocol + "-(1-2)"
        elif 2.0 < duration <= 3.0:
            yield protocol + "-(2-3)"
        elif 3.0 < duration <= 4.0:
            yield protocol + "-(3-4)"
        else:
            yield protocol + "-(4+)"


def plot_bar(keys, data, title, legend, rotation=0):
    l = len(data)
    fig, ax = plt.subplots()

    xs = range(len(keys))
    i = 0
    for t in data:
        adjusted_xs = [x - 0.2 + 0.75 / l * i for x in xs]

        ax.bar(adjusted_xs, t.values(), width=0.25, align='center', alpha=0.5)

        i += 1
    ax.set_xticks(xs)
    ax.set_xticklabels(keys, rotation=rotation, ha='right')
    ax.set_ylabel('Normalized counts')
    ax.set_title(f'{title}: Infected vs Normal')
    ax.yaxis.grid(True)
    ax.legend(legend)
    # Save the figure and show
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # %% Load the data and split on infected and botnet
    data_path = "data/capture20110812.pcap.netflow.labeled"
    infected = list(process_file(data_path, lambda l: "Botnet" in l))
    normal = list(process_file(data_path, lambda l: "LEGITIMATE" in l))

    # %% Visualise the protocol type

    # We assume that keys are the same for both normal and infected
    # Protocol plot
    plot_bar(["TCP", "UDP", "ICMP"],
             [count_protocol(normal), count_protocol(infected)],
             "Protocol",
             ["Normal", "Infected"])

    # %% Visualise the duration
    plot_bar(["0", "0-1", "1-2", "2-3", "3-4", "4+"],
             [count_duration(normal), count_duration(infected)],
             "Duration",
             ["Normal", "Infected"])

    # %% Now combine the features
    infected_comb = count_two(infected)
    normal_comb = count_two(normal)

    # Sort on dict keys alphabetically
    keys = sorted(list(set(list(infected_comb.keys()) + list(normal_comb.keys()))))

    # Set empty keys
    for k in keys:
        if k not in infected_comb.keys():
            infected_comb[k] = 0

        if k not in normal_comb.keys():
            normal_comb[k] = 0

    # Now sort according to the keys, which is alphabetically
    infected_comb = OrderedDict(sorted(infected_comb.items()))
    normal_comb = OrderedDict(sorted(normal_comb.items()))

    plot_bar(keys,
             [normal_comb, infected_comb],
             "Discretized",
             ["Normal", "Infected"],
             rotation=30)
