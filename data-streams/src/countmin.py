# %% Import
from collections import defaultdict
from hashlib import sha256
from struct import unpack
from math import ceil, e, log

import numpy as np


class CountMinSketch:
    def __init__(self, depth=None, width=None, epsilon=None, delta=None):
        if depth is not None and width is not None:
            self.depth = depth
            self.width = width
        elif epsilon is not None and epsilon is not None:
            self.width = int(ceil(e/epsilon))
            self.depth = int(ceil(log(1./delta)))
            print(f"CM with width={self.width} depth={self.depth}")
        else:
            raise Exception("Please supply the size or bounds of the CM sketch")

        self.depth_range = range(0, self.depth)

        # A row in the CM sketch is created the first time the key is called
        self.cm = defaultdict(lambda: np.zeros(self.width))

        # Function from Python hashlib
        self.hash_fn = sha256

    def get_hashes(self, key):
        # Encode the input as UTF-8 byte string
        if type(key) is not str:
            key = str(key)
        uni_key = key.encode('UTF-8')

        # Generate multiple hashes
        hasher = self.hash_fn()
        for d in self.depth_range:
            # Update the hash function with the key and index in the CM
            # This creates different hashes for the same key with the same hash function
            hasher.update(str(d).encode('UTF-8'))
            hasher.update(uni_key)

            # Get the hash value and turn into 64 bit number
            h = hasher.digest()
            num = unpack('Q', h[:8])[0]

            # Convert to a index in the CM row
            yield num % self.width

    def add(self, key, count=1):
        # Iterate over the CM rows and the corresponding hashes
        # Note: we iterate over the depth range as the CM contains no keys the first time
        for row, column in zip(self.depth_range, self.get_hashes(key)):
            self.cm[row][column] += count

    def get(self, key) -> int:
        vals = list()

        # Get the counts from each CM row
        # Note: we iterate over the depth range as the CM contains no keys the first time
        for row, column in zip(self.depth_range, self.get_hashes(key)):
            vals.append(self.cm[row][column])

        return min(vals)

    def __getitem__(self, key) -> int:
        return self.get(key)

    def __setitem__(self, key, value):
        return self.add(key, value)


# %%
test = CountMinSketch(epsilon=0.10, delta=0.10)
test['banaan'] = 10
test['gans'] = 1

test[4] = 4

print(test['banaan'])
print(test['gans'])
print(test[4])
