import numpy as np

class ScalarDecoder:
    def __init__(self, size):
        self.size = size
        self.sdr_to_scalar = dict()  # Store SDRs mapped to scalar values

    def decode(self, sdr, value=None, learn=True):
        """
        Decode the input SDR by finding the maximum overlap SDR and its mapped scalar value.
        sdr: The input SDR (e.g., cell activations) to decode.
        returns: The scalar value, overlap score, and confidence score.
        """
        assert sdr.size == self.size, f"SDR size incorrect, {sdr.size} instead of {self.size}"
        
        sdr_tuple = tuple(int(bit) for bit in sdr)
        

        if sdr_tuple in self.sdr_to_scalar: # Exact match found
            if learn:
                average, n = self.sdr_to_scalar[sdr_tuple]
                new_average = (average * n + value) / (n + 1) # Update average from data stream
                self.sdr_to_scalar[sdr_tuple] = (new_average, n + 1)
            return self.sdr_to_scalar[sdr_tuple][0]
        else: # No exact match, find the closest match
            max_overlap = 0
            best_scalar_value = None
            for stored_sdr_tuple, (average, n) in self.sdr_to_scalar.items():
                overlap = np.dot(sdr, np.array(stored_sdr_tuple))  # Dot product as overlap measure
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_scalar_value = average

            if learn: self.sdr_to_scalar[sdr_tuple] = (value, 1)
            return best_scalar_value

if __name__ == "__main__":

    from scalar_encoder import ScalarEncoder

    num_active_bits = 5
    encoder = ScalarEncoder(range=(0, 100), num_buckets=100, num_active_bits=num_active_bits)
    decoder = ScalarDecoder(size=encoder.size)

    for value in [0, 50, 99]:
        sdr = encoder.encode(value)
        decoder.decode(sdr, value)
        result = decoder.decode(sdr, learn=False)
        assert result == value, f"Result {result} does not equal {value}"

    print("Tests passed")