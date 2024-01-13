"""Test cases for the Matrix class."""
import copy

import pytest
from common import (
    generate_random_index,
    generate_random_matrix,
    generate_random_value,
)
from matrix import Matrix


def test_intialization():
    """Test the initialization of a vector."""
    # Test valid initialization
    random_matrix = generate_random_matrix()
    m = Matrix(random_matrix)

    # Test values of matrix are initialized correctly
    assert m.matrix == random_matrix, "The matrix is not initialized correctly"

    # Test that the values of matrix are a copy of initialization arguments
    assert (
        m.matrix is not random_matrix
    ), "The values of matrix should be a copy of initialization arguments"

    # Test default initialization
    default_matrix = Matrix()
    assert default_matrix.matrix == [
        [0, 0],
        [0, 0],
    ], "The default matrix is not initialized correctly"

    # Test invalid initialization (wrong type)
    with pytest.raises(TypeError):
        random_matrix[0][0] = "A"
        Matrix(random_matrix)

    # Test invalid initialization (wrong size)
    random_matrix2 = generate_random_matrix()
    with pytest.raises(ValueError):
        random_value = generate_random_value()
        random_matrix2[0].append(random_value)
        Matrix(random_matrix2)

    # Test argument change after initialization does not change the matrix
    random_matrix3 = generate_random_matrix()
    random_matrix3_temp = copy.deepcopy(random_matrix3)
    m = Matrix(random_matrix3)
    random_row = generate_random_index(len(random_matrix3))
    random_col = generate_random_index(len(random_matrix3[0]))
    random_value = generate_random_value()
    random_matrix3[random_row][random_col] = random_value
    assert (
        m.matrix == random_matrix3_temp
    ), "The matrix should not change when initialization arguments change"


def test_getter():
    """Test the values getter."""
    random_matrix = generate_random_matrix()
    m = Matrix(random_matrix)
    assert m.matrix == random_matrix, "The getter does not return the correct values"

    random_row = generate_random_index(len(random_matrix))
    random_col = generate_random_index(len(random_matrix[0]))
    random_value = generate_random_value()

    # Test that the getter returns a copy of the values
    m.matrix[random_row][random_col] = random_value
    assert m.matrix == random_matrix, "The getter should return a copy of the values"

    # Test return value change does not change the matrix
    random_matrix_temp = m.matrix
    random_matrix_temp[random_row][random_col] = random_value
    assert m.matrix == random_matrix, "The getter should return a copy of the values"


def test_setter():
    """Test that the coordinates of a matrix cannot be set directly."""
    random_matrix = generate_random_matrix()
    m = Matrix(random_matrix)
    # Test that the matrix cannot be set directly
    with pytest.raises(AttributeError):
        m.matrix = generate_random_matrix()


def test_size():
    """Test the size method."""
    random_matrix = generate_random_matrix()
    m = Matrix(random_matrix)

    # Test get both dimensions
    assert m.size() == (
        len(random_matrix),
        len(random_matrix[0]),
    ), "The size method does not return the correct values"

    # Test get row dimension
    assert m.size(0) == len(
        random_matrix
    ), "The size method does not return the correct values"

    # Test get column dimension
    assert m.size(1) == len(
        random_matrix[0]
    ), "The size method does not return the correct values"

    # Test invalid dimension
    with pytest.raises(IndexError):
        m.size(2)
