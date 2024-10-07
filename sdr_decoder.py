import numpy as np
from collections import defaultdict

class SDRDecoder:
    def __init__(self, size):
        self.size = size
        self.sdr_to_value = defaultdict(lambda: defaultdict(int))

    def decode(self, sdr, value=None):
        """
        Decode the input SDR by finding the maximum overlap SDR and its mapped value.
        sdr: The input SDR (e.g., cell activations) to decode.
        returns: The associated value
        """
        assert sdr.size == self.size, f"SDR size incorrect, {sdr.size} instead of {self.size}"
        
        sdr = tuple(sdr)

        if value is not None:
            self.sdr_to_value[sdr][value] += 1
        
        if sdr in self.sdr_to_value: # Exact match found
            if not self.sdr_to_value[sdr]:
                return None
            else:
                counter = self.sdr_to_value[sdr]
                return max(counter, key=counter.get)
            
        # No exact match, find the closest match
        max_overlap = 0
        result = None

        for other_sdr, counter in self.sdr_to_value.items():
            if not counter: continue
            overlap = np.dot(sdr, np.array(other_sdr))  # Dot product as overlap measure
            if overlap > max_overlap:
                max_overlap = overlap
                result = max(counter, key=counter.get)

        return result

if __name__ == "__main__":

    from scalar_encoder import ScalarEncoder

    num_active_bits = 5
    encoder = ScalarEncoder(range=(0, 100), num_buckets=100, num_active_bits=num_active_bits)
    decoder = SDRDecoder(size=encoder.size)

    for value in [0, 50, 99]:
        sdr = encoder.encode(value)
        decoder.decode(sdr, value)
        result = decoder.decode(sdr)
        assert result == value, f"Result {result} does not equal {value}"

    print("Tests passed")