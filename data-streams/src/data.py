import re
import typing
from datetime import datetime


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


def read_large_file(file_object: typing.TextIO):
    """
    Uses a generator to read a large file lazily
    :param file_object: typing.TextIO
    """
    while True:
        data = file_object.readline()
        if not data:
            break
        yield data


def process_file(path: str):
    """
    Processes the file and emits Flow objects
    :param path: str
    """
    try:
        regex = re.compile('\\s+')
        with open(path, 'r') as file_handler:
            file_handler.readline()  # skip header
            for line in read_large_file(file_handler):
                # process line
                yield Flow(regex.split(line))

            file_handler.close()
    except (IOError, OSError) as err:
        print(err)
        print("Error opening / processing file")
