import numpy as np
from scipy.sparse import spmatrix, csr_matrix, csc_matrix, coo_matrix


class TemporalMemory:
    def __init__(
            self,
            num_columns: int,
            num_column_cells: int,
            max_segment: int,
            synapse_min=0,
            synapse_init=60,
            synapse_max=100,
            synapse_threshold=50,
            synapse_inc=2,
            synapse_dec=1,
            segment_active_threshold=10,
            segment_match_threshold=7):

        self.num_columns = num_columns
        self.num_column_cells = num_column_cells
        self.max_segment = max_segment
        self.num_cells = num_columns * num_column_cells

        self.synapse_min = synapse_min
        self.synapse_init = synapse_init
        self.synapse_max = synapse_max

        self.synapse_threshold = synapse_threshold
        self.synapse_inc = synapse_inc
        self.synapse_dec = synapse_dec

        self.segment_active_threshold = segment_active_threshold
        self.segment_match_threshold = segment_match_threshold

        self.active_columns = csr_matrix((1, num_columns))

        self.bursting_winner_cells = csr_matrix(
            (num_columns, num_column_cells))
        self.predictive_cells = csr_matrix((num_columns, num_column_cells))
        self.active_cells = csr_matrix((num_columns, num_column_cells))
        self.winner_cells = csr_matrix((num_columns, num_column_cells))

        self.potential_synapses = {}
        self.synapse_strengths = {}

    def inference(self, active_columns: spmatrix):
        # (max_segment, num_cells, num_cells)
        connected_synapses = {
            idx: matrix >= self.synapse_threshold for idx,
            matrix in self.synapse_strengths.items()}  # (segments, cells, cells)
        segment_activations = csr_matrix(
            (self.num_cells, self.max_segment))  # (cells, segments)
        for idx, synapses in connected_synapses.items():
            segment_activations[:, idx] = self.active_cells.dot(synapses)

        # (cells, segments)
        active_segments = segment_activations >= self.segment_active_threshold
        predictive_cells = self._reduce_or(
            active_segments, axis=1)  # (1, cells)

        predictive_active_cells = csr_matrix(
            (self.num_columns, self.num_column_cells))
        for i in range(self.num_column_cells):
            predictive_active_cells[:, i] = active_columns.transpose().multiply(csc_matrix(
                predictive_cells.reshape(self.num_columns, self.num_column_cells))[:, i])  # AND operation, # (1, cells)

        predictive_columns = self._reduce_or(
            predictive_cells.reshape(
                self.num_columns,
                self.num_column_cells),
            axis=1)  # (1, columns)
        bursting_columns = active_columns - \
            active_columns.multiply(predictive_columns)  # (1, columns)

        active_cells = csr_matrix((self.num_columns, self.num_column_cells))
        for i in range(self.num_column_cells):
            active_cells[:, i] = predictive_active_cells[:,
                                                         i].maximum(bursting_columns.transpose())
        return active_cells.reshape((1, self.num_cells))

    def _reduce_or(self, matrix: spmatrix, axis: int):
        matrix = coo_matrix(matrix)
        if axis == 0:
            nonzero_indices = np.unique(matrix.col)
            reduced_vector = coo_matrix(
                ([1] * len(nonzero_indices),
                 ([0] * len(nonzero_indices),
                  nonzero_indices)),
                shape=(
                    1,
                    matrix.shape[1]))
        elif axis == 1:
            nonzero_indices = np.unique(matrix.row)
            reduced_vector = coo_matrix(
                ([1] * len(nonzero_indices),
                 ([0] * len(nonzero_indices),
                  nonzero_indices)),
                shape=(
                    1,
                    matrix.shape[0]))
        return csr_matrix(reduced_vector)
