import numpy as np

class SpatialPooler:
    def __init__(self, input_size, num_columns, num_winners, potential_connections=0.5, permanence_threshold=20):
        self.input_size = input_size  # The size of the input SDR
        self.num_columns = num_columns  # The number of columns (output size)
        self.num_winners = num_winners
        self.potential_connections = potential_connections  # Fraction of input bits a column is connected to
        self.permanence_threshold = permanence_threshold  # Threshold above which connections are considered "active"
        self.min_permanence = 0
        self.max_permanence = 100

        self.permanences = np.random.randint(self.min_permanence, permanence_threshold * 2, (num_columns, input_size))  # Random initial permanences around the threshold
        self.connections = self.permanences > self.permanence_threshold  # Connected synapses based on permanence values

    def process(self, input_sdr, learn=True):
        """Full spatial pooling process: compute overlap, inhibit columns, and update connections."""
        overlaps = np.dot(self.connections, input_sdr)
        active_columns = self.select_winners(overlaps)
        if learn:
            self.adapt_permanences(input_sdr, active_columns)

        return active_columns
    
    def select_winners(self, overlaps):
        """Select the top `num_winners` columns based on overlap scores."""
        winners = np.argsort(overlaps)[-self.num_winners:]

        active_columns = np.zeros(self.num_columns, dtype=int)
        active_columns[winners] = 1
        return active_columns
    
    def adapt_permanences(self, input_sdr, active_columns):
        """Adapt the permanence values of the winning columns to learn from the input SDR."""
        good_connections = np.outer(active_columns, input_sdr)
        good_connections *= self.connections # masking only the valid connections
        bad_connections = self.connections - good_connections

        self.permanences += good_connections * 2
        self.permanences -= bad_connections
        self.permanences = np.clip(self.permanences, 0, 100)
        self.connections = self.permanences > self.permanence_threshold

if __name__ == "__main__":
    input_size = 5
    num_columns = 6
    num_winners = 2
    sp = SpatialPooler(input_size, num_columns, num_winners)
    initial_permanences = np.copy(sp.permanences)
    initial_connections = np.copy(sp.connections)

    assert sp.permanences.shape == (num_columns, input_size), "Permanence shape is incorrect."
    assert sp.connections.shape == (num_columns, input_size), "Connections shape is incorrect."

    input_sdr = np.array([1, 1, 0, 0, 1])
    active_columns = sp.process(input_sdr)

    assert len(active_columns) == num_columns, "Active columns size is incorrect."
    assert np.sum(active_columns) == num_winners, "Number of active columns is incorrect."

    assert sp.permanences.shape == (num_columns, input_size), "Permanence shape is incorrect."
    assert sp.connections.shape == (num_columns, input_size), "Connections shape is incorrect."
    
    assert (sp.permanences >= 0).all() and (sp.permanences <= 100).all(), "Permanence values out of bounds."

    post_process_permanences = np.copy(sp.permanences)
    
    permanence_change = post_process_permanences - initial_permanences
    good_mask = np.outer(active_columns, input_sdr)
    good_mask = good_mask * initial_connections
    bad_mask = np.outer(active_columns, 1-input_sdr)
    bad_mask = bad_mask * initial_connections

    assert np.all((permanence_change * good_mask) >= 0), "Some good permanences decreased"
    assert np.all((permanence_change * bad_mask) <= 0), "Some bad permanences increased"

    
'''
Compute the overlap with the current input for each column

for c in columns
    overlap(c) = 0
    for s in connectedSynapses(c)
        overlap(c) = overlap(c) + input(t, s.sourceInput)
    overlap(c) = overlap(c) * boost(c)

    
Compute the winning columns after inhibition

for c in columns
    minLocalActivity = kthScore(neighbors(c), numActiveColumnsPerInhArea)
    if overlap(c) > stimulusThreshold and overlap(c) ≥ minLocalActivity then
        activeColumns(t).append(c)

        
Update synapse permanences and internal variables

for c in activeColumns(t)
    for s in potentialSynapses(c)
        if active(s) then
            s.permanence += synPermActiveInc
            s.permanence = min(1.0, s.permanence)
        else
            s.permanence -= synPermInactiveDec
            s.permanence = max(0.0, s.permanence)
for c in columns:
    activeDutyCycle(c) = updateActiveDutyCycle(c)
    activeDutyCycleNeighbors = mean(activeDutyCycle(neighbors(c))
    boost(c) = boostFunction(activeDutyCycle(c), activeDutyCycleNeighbors)
    overlapDutyCycle(c) = updateOverlapDutyCycle(c)
    if overlapDutyCycle(c) < minDutyCycle(c) then
        increasePermanences(c, 0.1*connectedPerm)
inhibitionRadius = averageReceptiveFieldSize() 
'''
