# %% Init cm sketch

from data import infected_filter, get_most_frequent
from sketch import CountMinSketch

if __name__ == '__main__':
    data_path = "../data/capture20110812.pcap.netflow.labeled"

    # Count the real infected connections that were made to or from the host
    filter = infected_filter(data_path)
    real_distribution = get_most_frequent(filter)

    # Do a count min sketch, and get the estimated counts for infected connections
    filter = infected_filter(data_path)
    cm = CountMinSketch(epsilon=0.01, delta=0.01).analyse_stream(filter)
    cm_distribution = cm.get_distribution(real_distribution.keys())

    print(real_distribution)
    print(cm_distribution)
