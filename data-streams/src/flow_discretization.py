from collections import Counter, OrderedDict

import matplotlib.pyplot as plt

from data import process_file


def normalize(data: dict):
    c = sum(data.values(), 0.0)

    for key in data:
        data[key] /= c

    return data


def count_protocol(data):
    res = Counter()

    for i in data:
        res.update({i.protocol: 1})

    res = normalize(res)

    return res


def count_two(data):
    res = Counter()

    for i in data:
        duration = i.duration
        protocol = i.protocol

        if duration == 0.0:
            res.update({protocol + "-0": 1})
        elif 0.0 < duration <= 1.0:
            res.update({protocol + "-[0-1]": 1})
        elif 1.0 < duration <= 2.0:
            res.update({protocol + "-[1-2]": 1})
        elif 2.0 < duration <= 3.0:
            res.update({protocol + "-[2-3]": 1})
        elif 3.0 < duration <= 4.0:
            res.update({protocol + "-[3-4]": 1})
        else:
            res.update({protocol + "-4+": 1})

    res = normalize(res)

    return res


def count_duration(data):
    res = Counter()

    for i in data:
        duration = i.duration

        if duration == 0.0:
            res.update({"0": 1})
        elif 0.0 < duration <= 1.0:
            res.update({"0-1": 1})
        elif 1.0 < duration <= 2.0:
            res.update({"1-2": 1})
        elif 2.0 < duration <= 3.0:
            res.update({"2-3": 1})
        elif 3.0 < duration <= 4.0:
            res.update({"3-4": 1})
        else:
            res.update({"4+": 1})

    res = normalize(res)

    return res


def plot_bar(keys, data, title, legend, rotation=0):
    l = len(data)
    fig, ax = plt.subplots()

    xs = range(len(keys))
    i = 0
    for t in data:
        adjusted_xs = [x - 0.2 + 0.75 / l * i for x in xs]

        # Set empty values
        for k in keys:
            if k not in t.keys():
                t[k] = 0

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

    # Sort on dict keys
    keys = sorted(list(set(list(infected_comb.keys()) + list(normal_comb.keys()))))
    infected_comb = OrderedDict(sorted(infected_comb.items()))
    normal_comb = OrderedDict(sorted(normal_comb.items()))

    plot_bar(keys,
             [normal_comb, infected_comb],
             "Discretized",
             ["Normal", "Infected"],
             rotation=30)
