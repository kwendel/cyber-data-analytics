# %% Init cm sketch
import typing

from data import process_file
from sketch import CountMinSketch

gen = process_file("data/capture20110812.pcap.netflow.labeled")


def infected_filter() -> typing.Iterator[str]:
    infected_ip = '147.32.84.165'
    for flow in gen:
        if infected_ip in flow.src:
            yield flow.dst
        elif infected_ip in flow.dst:
            yield flow.src


cm = CountMinSketch(epsilon=0.01, delta=0.01).analyse_stream(infected_filter())
