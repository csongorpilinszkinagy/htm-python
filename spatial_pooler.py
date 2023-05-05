import numpy as np
from scipy.sparse import spmatrix, csr_matrix, lil_matrix


class SpatialPooler():
    def __init__(
            self,
            input_size: int,
            num_columns: int,
            potential_synapse_ratio: float = 0.2,
            active_column_ratio: float = 0.02,
            synapse_min: int = 0,
            synapse_threshold: int = 20,
            synapse_init: int = 30,
            synapse_max: int = 100,
            synapse_inc: int = 2,
            synapse_dec: int = 1):

        self.input_size = input_size
        assert num_columns >= 100
        self.num_columns = num_columns
        self.num_active_columns = int(num_columns * active_column_ratio)

        self.synapse_min = synapse_min
        self.synapse_threshold = synapse_threshold
        self.synapse_init = synapse_init
        self.synapse_max = synapse_max
        self.synapse_inc = synapse_inc
        self.synapse_dec = synapse_dec

        self.potential_synapses, self.synapse_strengths = self._init_synapses(
            input_size, num_columns, potential_synapse_ratio, synapse_init)

        self.active_columns = csr_matrix((1, num_columns))
        self.column_activity = np.ones(num_columns)

    def _init_synapses(self,
                       input_size: int,
                       num_columns: int,
                       potential_synapse_ratio: float,
                       init_value: int) -> tuple[spmatrix,
                                                 spmatrix]:
        num_potential_synapses = int(input_size * potential_synapse_ratio)
        potential_synapses = lil_matrix((input_size, num_columns))
        synapse_strengths = lil_matrix((input_size, num_columns))
        for column in range(num_columns):
            input_idxs = np.random.choice(
                range(input_size),
                size=num_potential_synapses,
                replace=False)
            potential_synapses[input_idxs, column] = 1
            synapse_strengths[input_idxs, column] = init_value

        potential_synapses = csr_matrix(potential_synapses)
        synapse_strengths = csr_matrix(synapse_strengths)
        return potential_synapses, synapse_strengths

    def compute(
            self,
            input_sdr: spmatrix,
            learn: bool = False,
            boost: bool = False) -> spmatrix:
        connected_synapses = self.synapse_strengths >= self.synapse_threshold
        column_activations = input_sdr.dot(connected_synapses)

        if boost:
            boost_factors = np.ones((1, self.num_columns))
            for i, activity in enumerate(self.column_activity):
                if activity != 0:
                    boost_factors[0, i] = 1 / activity
            column_activations.multiply(boost_factors)

        top_idxs = np.argsort(
            column_activations.data)[-self.num_active_columns:]
        top_cols = column_activations.indices[top_idxs]
        values = [1] * len(top_cols)
        rows = [0] * len(top_cols)
        active_columns = csr_matrix(
            (values, (rows, top_cols)),
            shape=(1, self.num_columns))

        if learn:
            synapses_to_inc = input_sdr.transpose().dot(active_columns)
            synapses_to_inc = synapses_to_inc.multiply(self.potential_synapses)

            synapses_to_dec = self.potential_synapses - synapses_to_inc
            synapses_to_dec = synapses_to_dec.multiply(active_columns)

            self.synapse_strengths += synapses_to_inc * self.synapse_inc
            self.synapse_strengths -= synapses_to_dec * self.synapse_dec
            self.synapse_strengths = self.synapse_strengths.maximum(
                self.synapse_min).minimum(self.synapse_max)
            
        self.column_activity = self.column_activity * 0.999 + active_columns.toarray()[0, :]

        return active_columns
