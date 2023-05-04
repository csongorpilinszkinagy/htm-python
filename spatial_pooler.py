import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

class SpatialPooler():
    def __init__(
        self, input_size: int, num_columns: int,
        potential_synapse_ratio: float = 0.5, active_column_ratio: float = 0.02,
        synapse_min: int = 0, synapse_init: int = 40, synapse_threshold: int = 50, synapse_max:int = 100,
        synapse_inc: int = 2, synapse_dec: int = 1):

        self.input_size = input_size
        assert num_columns >= 100
        self.num_columns = num_columns
        self.num_active_columns = int(num_columns * active_column_ratio)
        
        self.synapse_min = synapse_min
        self.synapse_init = synapse_init
        self.synapse_threshold = synapse_threshold
        self.synapse_max = synapse_max
        self.synapse_inc = synapse_inc
        self.synapse_dec = synapse_dec
        
        self.potential_synapses, self.synapse_strengths = self._init_synapses(input_size, num_columns, potential_synapse_ratio, synapse_init)

        self.active_columns = csr_matrix((1, num_columns))

    def _init_synapses(self, input_size: int, num_columns: int, potential_synapse_ratio: float, init_value: int) -> tuple[csr_matrix, csr_matrix]:
        num_potential_synapses = int(input_size * potential_synapse_ratio)
        potential_synapses = lil_matrix((input_size, num_columns))
        synapse_strengths = lil_matrix((input_size, num_columns))
        for column in range(num_columns):
            input_idxs = np.random.choice(range(input_size), size=num_potential_synapses, replace=False)
            potential_synapses[input_idxs, column] = 1
            synapse_strengths[input_idxs, column] = init_value

        potential_synapses = csr_matrix(potential_synapses)
        synapse_strengths = csr_matrix(synapse_strengths)
        return potential_synapses, synapse_strengths
    
    def inference(self, input_sdr: csr_matrix) -> csr_matrix:
        connected_synapses = self.synapse_strengths >= self.synapse_threshold
        column_activations = input_sdr.dot(connected_synapses)
        
        # TODO: add boosting
        top_columns = np.argsort(column_activations.toarray())[0, -self.num_active_columns:]
        values = [1] * len(top_columns)
        rows = [0] * len(top_columns)
        active_columns = csr_matrix((values, (rows, top_columns)), shape=(1, self.num_columns))
        
        return active_columns
    
    def train(self, input_sdr: csr_matrix) -> csr_matrix:
        active_columns = self.inference(input_sdr)

        synapses_to_inc = input_sdr.transpose().dot(active_columns)
        synapses_to_inc = synapses_to_inc.multiply(self.potential_synapses)

        synapses_to_dec = self.potential_synapses - synapses_to_inc
        synapses_to_dec = synapses_to_dec.multiply(active_columns)

        self.synapse_strengths += synapses_to_inc * self.synapse_inc
        self.synapse_strengths -= synapses_to_dec * self.synapse_dec
        self.synapse_strengths = self.synapse_strengths.maximum(self.synapse_min).minimum(self.synapse_max)

        return active_columns
        