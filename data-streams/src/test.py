# %%
import sys

sys.path.append('./src')
import typing

from data import process_file
from reservoir import Reservoir

# gen = process_file('../data/capture.test.pcap.netflow.labeled')
gen = process_file('data/capture20110812.pcap.netflow.labeled')


def filter() -> typing.Iterator[str]:
    infected_ip = '147.32.84.165'
    for flow in gen:
        if infected_ip in flow.src:
            yield flow.dst
        elif infected_ip in flow.dst:
            yield flow.src


r = Reservoir(filter(), size=1000)
r.get_distribution()
