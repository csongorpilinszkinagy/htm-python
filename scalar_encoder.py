import numpy as np

class ScalarEncoder:
    def __init__(self, range, num_buckets, num_active_bits):
        self.min_value = range[0]
        self.max_value = range[1]
        self.num_buckets = num_buckets
        self.num_active_bits = num_active_bits
        self.size = num_buckets + num_active_bits - 1
        self.bucket_width = (self.max_value - self.min_value) / self.num_buckets
    
    def encode(self, value):
        """Encode a scalar value into an SDR."""
        assert value >= self.min_value and value < self.max_value, f"Value {value} out of range"
            
        bucket_idx = int((value - self.min_value) / self.bucket_width)
        start = bucket_idx
        end = start + self.num_active_bits

        sdr = np.zeros(self.size, dtype=int)
        sdr[start:end] = 1
        return sdr

if __name__ == "__main__":
    num_buckets = 100
    num_active_bits = 5
    encoder = ScalarEncoder(range=(0, 100), num_buckets=num_buckets, num_active_bits=num_active_bits)

    size = num_buckets + num_active_bits - 1

    for value in [0, 1, 50, 99]:
        sdr = encoder.encode(value)
        assert sdr.size == size, f"SDR size {sdr.size} incorrect"
        assert sum(sdr) == num_active_bits, f"Number of active bits {sum(sdr)} incorrect"
        
        expected_sdr = np.zeros(size, dtype=int)
        expected_sdr[value:value+num_active_bits] = 1
        assert np.array_equal(sdr, expected_sdr), "SDR does not match expected SDR"

    print("Tests passed")
