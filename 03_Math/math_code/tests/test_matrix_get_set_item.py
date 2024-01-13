"""Test cases for the Matrix class."""
import numpy as np
import pytest
from common import (
    generate_random_index,
    generate_random_matrix,
    generate_random_value,
    generate_random_values,
)
from matrix import Matrix


def test_get_item():
    """Test the __getitem__ method."""
    random_matrix = generate_random_matrix()
    m = Matrix(random_matrix)

    random_row_index = generate_random_index(len(random_matrix))
    random_row_start = generate_random_index(len(random_matrix))
    random_row_stop = generate_random_index(len(random_matrix))
    random_row_step = 0
    while random_row_step == 0:
        random_row_step = generate_random_index(len(random_matrix))

    random_col_index = generate_random_index(len(random_matrix[0]))
    random_col_start = generate_random_index(len(random_matrix[0]))
    random_col_stop = generate_random_index(len(random_matrix[0]))
    random_col_step = 0
    while random_col_step == 0:
        random_col_step = generate_random_index(len(random_matrix[0]))

    random_row_slice = slice(random_row_start, random_row_stop, random_row_step)
    random_col_slice = slice(random_col_start, random_col_stop, random_col_step)

    numpy_matrix = np.array(random_matrix)

    # Test case 1: matrix[int]
    assert (
        m[random_row_index] == numpy_matrix[random_row_index].tolist()
    ), "Case 1 is not retrieved correctly"

    # Test case 2: matrix[slice]
    assert (
        m[random_row_slice] == numpy_matrix[random_row_slice].tolist()
    ), "Case 2 are not retrieved correctly"

    # Test case 3.1: matrix[int, int]
    assert (
        m[random_row_index, random_col_index]
        == numpy_matrix[random_row_index, random_col_index]
    ), "Case 3.1 is not retrieved correctly"

    # Test case 3.2: matrix[int, slice]
    assert (
        m[random_row_index, random_col_slice]
        == numpy_matrix[random_row_index, random_col_slice].tolist()
    ), "Case 3.2 is not retrieved correctly"

    # Test case 3.3: matrix[slice, int]
    assert (
        m[random_row_slice, random_col_index]
        == numpy_matrix[random_row_slice, random_col_index].tolist()
    ), "Case 3.3 is not retrieved correctly"

    # Test case 3.4: matrix[slice, slice]
    assert (
        m[random_row_slice, random_col_slice]
        == numpy_matrix[random_row_slice, random_col_slice].tolist()
    ), "Case 3.4 is not retrieved correctly"

    # Test invalid index
    with pytest.raises(IndexError):
        m[len(random_matrix)]

    # Test invalid index type
    with pytest.raises(TypeError):
        m["A"]


def test_set_item():
    """Test the __setitem__ method."""
    random_matrix = generate_random_matrix()

    m = Matrix(random_matrix)

    random_row_index = generate_random_index(len(random_matrix))
    random_col_index = generate_random_index(len(random_matrix[0]))

    random_value = generate_random_value()
    random_row_values = generate_random_values(length=len(random_matrix[0]))

    # Test case 1: matrix[int] = value
    m[random_row_index] = random_row_values
    assert m[random_row_index] == random_row_values, "Case 1 is not set correctly"

    # Test case 2: matrix[int, int] = value
    m[random_row_index, random_col_index] = random_value
    assert (
        m[random_row_index, random_col_index] == random_value
    ), "Case 2.1 is not set correctly"

    # Test invalid value size
    with pytest.raises(ValueError):
        m[random_row_index] = random_row_values[:-1]

    # Test invalid value type
    with pytest.raises(TypeError):
        random_row_values[0] = "A"
        m[random_row_index] = random_row_values

    with pytest.raises(TypeError):
        m[random_row_index, random_col_index] = "A"
