# %% Load the data
from data import process_file

data = process_file("data/capture20110812.pcap.netflow.labeled")

# %% Init cm sketch
from sketch import CountMinSketch

cm = CountMinSketch(epsilon=0.01, delta=0.01)

infected = "147.32.84.165"
c = 0
for f in data:
    if infected in f.src:
        print("added")
        cm.add(f.dst)
    elif infected in f.dst:
        cm.add(f.src)
        print("added")

    c += 1

    if c == 1000:
        break

