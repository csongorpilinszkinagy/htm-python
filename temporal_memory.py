from scipy.sparse import csr_matrix

class TemporalMemory:
    def __init__(self, num_columns: int, num_column_cells: int, max_segment: int,
                 synapse_min=0, synapse_init=60, synapse_max=100,
                 synapse_threshold=50, synapse_inc=2, synapse_dec=1,
                 segment_active_threshold=10, segment_match_threshold=7):
        
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

        self.segment_active_thershold = segment_active_threshold
        self.segment_match_threshold = segment_match_threshold

        self.active_columns = csr_matrix((1, num_columns))

        self.bursting_winner_cells = csr_matrix((num_columns, num_column_cells))
        self.predictive_cells = csr_matrix((num_columns, num_column_cells))
        self.active_cells = csr_matrix((num_columns, num_column_cells))
        self.winner_cells = csr_matrix((num_columns, num_column_cells))

        self.potential_synapses = []
        self.synapse_strengths = []
