import re
import typing
from collections import Counter
from datetime import datetime

from tqdm import tqdm

# The provided infected IP from the dataset
infected_hosts = {
    "capture20110812.pcap.netflow.labeled": [
        "147.32.84.165"
    ],
    "capture20110818.pcap.netflow.labeled": [
        "147.32.84.165",
        "147.32.84.191",
        "147.32.84.192",
        "147.32.84.193",
        "147.32.84.204",
        "147.32.84.205",
        "147.32.84.206",
        "147.32.84.207",
        "147.32.84.208",
        "147.32.84.209"
    ]
}


def get_infected(path: str) -> typing.List[str]:
    file = path.split("/")[-1]
    return infected_hosts[file]


class Flow:
    def __init__(self, split_line: typing.List[str]):
        self.start = datetime.strptime(f'{split_line[0]} {split_line[1]}', '%Y-%m-%d %H:%M:%S.%f')
        self.duration = float(split_line[2])
        self.protocol = split_line[3]

        src_ip = split_line[4].split(":")
        if len(src_ip) is 1:
            self.src = src_ip[0]
            self.src_port = None
        elif len(src_ip) is 2:
            self.src = src_ip[0]
            self.src_port = src_ip[1]

        dst_ip = split_line[6].split(":")
        if len(dst_ip) is 1:
            self.dst = dst_ip[0]
            self.dst_port = None
        elif len(dst_ip) is 2:
            self.dst = dst_ip[0]
            self.dst_port = dst_ip[1]

        self.flags = split_line[7]
        self.tos = split_line[8]
        self.packets = split_line[9]
        self.bytes = split_line[10]
        self.flows = split_line[11]
        self.label = split_line[12]

    def __str__(self):
        return f"{self.src} -> {self.dst}"


def process_file(path: str, filter_fn: typing.Callable) -> typing.Iterator[Flow]:
    """
    Processes the file and emits Flow objects
    :param path: str
    :param filter_fn: lambda function to filter
    """

    try:
        regex = re.compile('\\s+')
        with open(path, 'r') as file:
            file.readline()  # skip header
            for line in tqdm(file):
                if filter_fn(line):
                    # process line
                    yield Flow(regex.split(line))
    except (IOError, OSError) as err:
        print(err)
        print("Error opening / processing file")


def infected_filter(path: str) -> typing.Iterator[str]:
    ips = get_infected(path)
    if len(ips) > 1:
        raise ValueError("Multiple IPS not supported")

    infected_ip = ips[0]
    generator = process_file(path, lambda l: infected_ip in l)
    for flow in generator:
        if infected_ip in flow.src:
            yield flow.dst
        elif infected_ip in flow.dst:
            yield flow.src


def get_most_frequent(generator, amount=10) -> dict:
    items = list(generator)
    total = len(items)

    ips = Counter(items).most_common(n=amount)

    # Turn the counts to frequencies rounded to 4 decimals after the comma
    return {k: float("{0:4f}".format(v / total)) for k, v in ips}
