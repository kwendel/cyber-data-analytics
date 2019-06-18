# %% Init cm sketch
import random
import time

from data import infected_filter, get_most_frequent
from reservoir import Reservoir
from sketch import CountMinSketch

if __name__ == '__main__':
    data_path = "../data/capture20110812.pcap.netflow.labeled"

    # Count the real infected connections that were made to or from the host
    filter = infected_filter(data_path)
    real_distribution = get_most_frequent(filter)

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

        random.seed(42)
        filter = infected_filter(data_path)
        res = Reservoir(filter, size)
        res_distribution = res.get_distribution()

        elapsed = time.time() - start
        return res_distribution, elapsed

    cm_dis, cm_t = test_cm(0.01, 0.01)
    res_dis, res_t = test_reservoir(1000)

    # print(real_distribution)
    # print(cm_distribution)
    # print(res_distribution)
