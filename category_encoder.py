import numpy as np

class CategoryEncoder:
    def __init__(self, categories, num_active_bits):
        self.categories = tuple(sorted(set(categories)))
        self.num_active_bits = num_active_bits
        self.size = (len(categories)+1) * num_active_bits

    def encode(self, category):
        if category in self.categories:
            bucket_idx = self.categories.index(category) + 1
        else:
            bucket_idx = 0

        start, end = bucket_idx * self.num_active_bits, (bucket_idx + 1) * self.num_active_bits

        sdr = np.zeros(self.size, dtype=int)
        sdr[start:end] = 1
        return sdr
    
if __name__ == "__main__":
    categories = "ABCD"
    num_active_bits = 5

    encoder = CategoryEncoder(categories, num_active_bits)
    size = (len(categories) + 1) * num_active_bits

    for index, category in enumerate("XABCD"):
        sdr = encoder.encode(category)
        assert sdr.size == size, f"SDR size {sdr.size} incorrect"
        assert sum(sdr) == num_active_bits, f"Number of active bits {sum(sdr)} incorrect"

        expected_sdr = np.zeros(size, dtype=int)
        start, end = index * num_active_bits, (index + 1) * num_active_bits
        expected_sdr[start:end] = 1
        assert np.array_equal(sdr, expected_sdr), "SDR does not match expected SDR"
            