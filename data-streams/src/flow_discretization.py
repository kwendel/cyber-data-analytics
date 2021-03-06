# %%
from collections import Counter, OrderedDict
from functools import wraps

import matplotlib.pyplot as plt

from data import process_file, Flow


def discrete_duration(flow: Flow):
    duration = flow.duration

    if duration == 0.0:
        return "0"
    elif 0.0 < duration <= 1.0:
        return "0-1"
    elif 1.0 < duration <= 2.0:
        return "1-2"
    elif 2.0 < duration <= 3.0:
        return "2-3"
    elif 3.0 < duration <= 4.0:
        return "3-4"
    else:
        return "4+"


def discrete_protocol(flow: Flow):
    return flow.protocol


def discrete_flow(flow):
    return f"{discrete_protocol(flow)}-({discrete_duration(flow)})"


def counter(func):
    """
    Count the occurences of each unique element in the data of the provided function
    Function decorator pattern: https://realpython.com/primer-on-python-decorators/
    """
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
        yield discrete_protocol(i)


@counter
def count_duration(data):
    for i in data:
        yield discrete_duration(i)


@counter
def count_two(data):
    for i in data:
        yield discrete_flow(i)


# %%
def plot_bar(keys, data, title, legend, rotation=0, figsize=(6.4, 4.8)):
    l = len(data)
    fig, ax = plt.subplots(figsize=figsize)

    xs = range(len(keys))
    i = 0
    for t in data:
        adjusted_xs = [x - 0.2 + 0.75 / l * i for x in xs]
        ax.bar(adjusted_xs, t.values(), width=0.25, align='center', alpha=0.5)

        i += 1
    ax.set_xticks(xs)
    ax.set_xticklabels(keys, rotation=rotation, ha='right')
    ax.set_ylabel('Percentage of class')
    ax.set_title(f'{title}: Normal vs Infected hosts')
    ax.yaxis.grid(True)
    ax.legend(legend)
    # Save the figure and show
    plt.savefig(f"figs/{title}.png")
    plt.tight_layout()
    plt.show()


# %%

if __name__ == '__main__':
    # %% Load the data and split on infected and botnet
    data_path = "data/capture20110818.pcap.netflow.labeled"
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
             "Discrete Duration",
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
             rotation=30,
             figsize=(12.8, 4.8)
             )
