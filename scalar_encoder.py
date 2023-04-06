class ScalarEncoder():
    def __init__(self, value_range: tuple[int, int], num_buckets: int, num_active_cells: int):
        self.value_range = value_range
        self.num_buckets = num_buckets
        self.num_active_cells = num_active_cells
        self.size = num_buckets + num_active_cells - 1