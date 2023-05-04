import os

import numpy as np
import matplotlib.pyplot as plt

from scalar_encoder import ScalarEncoder
from spatial_pooler import SpatialPooler
from temporal_memory import TemporalMemory

values = range(20)
encoder = ScalarEncoder(value_range=(0, 20), num_buckets=50, num_active_cells=5)
sp = SpatialPooler(input_size=encoder.size, num_columns=100)
tm = TemporalMemory(num_columns=100, num_column_cells=5, max_segment=10)

input_cell_sequence = []
active_column_sequence = []
active_cell_sequence = []

for value in values:
    input_cells = encoder.encode(value)
    active_columns = sp.inference(input_cells)
    active_columns[0, 0] = 0
    active_cells = tm.inference(active_columns)

    input_cell_sequence.append(input_cells.toarray()[0])
    active_column_sequence.append(active_columns.toarray()[0])
    active_cell_sequence.append(active_cells.toarray()[0])

if not os.path.exists('figs'):
    os.mkdir('figs')

plt.imshow(np.transpose(input_cell_sequence))
plt.savefig('figs/input_cells.png')

plt.imshow(np.transpose(active_column_sequence))
plt.savefig('figs/active_columns')

plt.imshow(np.transpose(active_cell_sequence))
plt.savefig('figs/acgtive_cells')