import re
import typing
from datetime import datetime

from tqdm import tqdm


class Flow:
    def __init__(self, split):
        self.start = datetime.strptime(f'{split[0]} {split[1]}', '%Y-%m-%d %H:%M:%S.%f')
        self.duration = float(split[2])
        self.protocol = split[3]
        self.src = split[4]
        self.dst = split[6]
        self.flags = split[7]
        self.tos = split[8]
        self.packets = split[9]
        self.bytes = split[10]
        self.flows = split[11]
        self.label = split[12]

    def __str__(self):
        return f"{self.src} -> {self.dst}"


def process_file(path: str) -> typing.Iterator[Flow]:
    """
    Processes the file and emits Flow objects
    :param path: str
    """
    infected_ip = '147.32.84.165'

    try:
        regex = re.compile('\\s+')
        with open(path, 'r') as file:
            file.readline()  # skip header
            for line in tqdm(file):
                if infected_ip in line:
                    # process line
                    yield Flow(regex.split(line))
    except (IOError, OSError) as err:
        print(err)
        print("Error opening / processing file")
