import random
import typing

# from src.data import Flow


class Reservoir:
    def __init__(self, generator: typing.Iterator[str], size: int):
        self.generator = generator
        self.size = size

    # A function to randomly select
    # k items from stream[0..n-1].
    def selectKItems(self):
        i = 0
        k = self.size
        # index for elements
        # in stream[]

        # reservoir[] is the output
        # array. Initialize it with
        # first k elements from stream[]
        reservoir = [None] * k
        for i in range(k):
            reservoir[i] = next(self.generator)

        # Iterate from the (k+1)th
        # element to nth element
        for flow in self.generator:
            # Pick a random index
            # from 0 to i.
            j = random.randrange(i + 1)

            # If the randomly picked index is smaller than k, then replace the element
            # present at the index
            # with new element from stream
            if j < k:
                reservoir[j] = flow
            i += 1

        return reservoir
