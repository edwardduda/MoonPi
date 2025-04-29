import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def _update_circular_1d(buf, head, val):
    buf[head] = val
    head = (head + 1) % buf.shape[0]
    return head

@jit(nopython=True, cache=True)
def _update_circular_2d(buf, head, row):
    # assumes row.shape[0] == buf.shape[1]
    for j in range(buf.shape[1]):
        buf[head, j] = row[j]
    head = (head + 1) % buf.shape[0]
    return head

class CircularBuffer:
    def __init__(self, size, feature_dim=None):
        self.size        = size
        self.feature_dim = feature_dim
        self.head        = 0
        self.is_full     = False

        if feature_dim is None:
            self.buffer = np.zeros(size, dtype=np.float32)
        else:
            self.buffer = np.zeros((size, feature_dim), dtype=np.float32)

    def insert(self, value):
        if self.feature_dim is None:
            # scalar path
            self.head = _update_circular_1d(self.buffer, self.head, value)
        else:
            # vector path
            self.head = _update_circular_2d(self.buffer, self.head, value)

        # mark full when we wrap around
        if self.head == 0:
            self.is_full = True

    def get_ordered(self):
        if not self.is_full:
            return self.buffer[:self.head]
        return np.concatenate(
            (self.buffer[self.head:], self.buffer[:self.head]),
            axis=0
        )

    def latest(self):
        return self.buffer[self.head - 1]
