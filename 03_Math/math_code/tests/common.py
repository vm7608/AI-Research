"""Common random functions for tests."""
import random

LOW = -1000
HIGH = 1000
DATA_LENGTH = 100


def generate_random_values(low=LOW, high=HIGH, length=DATA_LENGTH):
    """Generate random values for vector."""
    return [random.randrange(low, high) for _ in range(length)]


def generate_random_value(low=LOW, high=HIGH):
    """Generate random vectors."""
    return random.randrange(low, high)


def generate_random_index(length=DATA_LENGTH):
    """Generate random vectors."""
    return random.randrange(0, length)


def generate_basis(length=DATA_LENGTH):
    """Generate random basis.

    i-th vector will have random_value at index i, and 0 elsewhere
    so that all vectors is othogonal to each other
    """
    vectors = []
    random_value_list = []
    for i in range(length):
        random_value = 0
        while random_value == 0:
            random_value = generate_random_value()
        vector = [0] * length
        vector[i] = random_value
        vectors.append(vector)
        random_value_list.append(random_value)
    return vectors, random_value_list


def generate_random_matrix_size(length=DATA_LENGTH):
    """Generate random matrix size."""
    row_size = col_size = 0
    while row_size == 0 or col_size == 0:
        row_size = generate_random_index(length)
        col_size = generate_random_index(length)
    return row_size, col_size


def generate_random_matrix(row_size=None, col_size=None, low=LOW, high=HIGH):
    """Generate random matrix."""
    if row_size is None:
        row_size, _ = generate_random_matrix_size()
    if col_size is None:
        col_size, _ = generate_random_matrix_size()

    return [
        [generate_random_value(low, high) for _ in range(col_size)]
        for _ in range(row_size)
    ]
