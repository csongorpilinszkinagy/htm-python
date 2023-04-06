import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

class ScalarEncoder():
    def __init__(self, value_range: tuple[int, int], num_buckets: int, num_active_cells: int):
        self.value_range = value_range
        self.num_buckets = num_buckets
        self.num_active_cells = num_active_cells

        self.size = num_buckets + num_active_cells - 1
        diff = value_range[1] - value_range[0]
        step = diff / (self.size - 1)
        self.bins = np.arange(*value_range, step)

    def encode(self, value: float) -> csr_matrix:
        idx = np.digitize(value, self.bins)
        sdr = lil_matrix((1, self.size), dtype=int)
        sdr[0, idx:idx+self.num_active_cells] = 1
        sdr = csr_matrix(sdr)
        return sdr

    def decode(self, sdr: csr_matrix) -> float:
        idx = sdr.indices[0]
        return self.bins[max(idx-1, 0)]
