import re
import typing
from datetime import datetime

from tqdm import tqdm


class Flow:
    def __init__(self, split_line: typing.List[str]):
        self.start = datetime.strptime(f'{split_line[0]} {split_line[1]}', '%Y-%m-%d %H:%M:%S.%f')
        self.duration = float(split_line[2])
        self.protocol = split_line[3]
        self.src = split_line[4].split(':')[0]
        self.dst = split_line[6].split(':')[0]
        self.flags = split_line[7]
        self.tos = split_line[8]
        self.packets = split_line[9]
        self.bytes = split_line[10]
        self.flows = split_line[11]
        self.label = split_line[12]

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
