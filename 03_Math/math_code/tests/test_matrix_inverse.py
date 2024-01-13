"""Test cases for the Matrix class."""
import numpy as np
import pytest
from common import (
    generate_random_matrix,
    generate_random_matrix_size,
)
from matrix import Matrix

EPS = 1e-8
NUM_TESTS = 100


def test_inverse_matrix_1_case(size=10):
    """Perform test for `inverse` method of Matrix.

    Only test for square matrix that has size less than 20
    We use numpy to calculate the inverse matrix,
    and compare it with our implementation by calculating the error ratio
    If the error ratio is less than 1e-8, we consider it as correct
    """
    random_square_size, _ = generate_random_matrix_size(length=size)
    random_square_matrix = generate_random_matrix(
        random_square_size, random_square_size
    )
    square_m = Matrix(random_square_matrix)
    numpy_square_m = np.array(random_square_matrix)

    inverse_matrix = square_m.inverse()
    calculated_inverse = inverse_matrix.matrix
    expected_inverse = np.linalg.inv(numpy_square_m)

    error_ratio_list = []
    for i in range(random_square_size):
        for j in range(random_square_size):
            if expected_inverse[i][j] == 0:
                error_ratio = abs(calculated_inverse[i][j])
            else:
                error_ratio = abs(
                    calculated_inverse[i][j] - expected_inverse[i][j]
                ) / abs(expected_inverse[i][j])

            error_ratio_list.append(error_ratio)

    acceptable_err_ratio = EPS
    assert all(
        [error_ratio < acceptable_err_ratio for error_ratio in error_ratio_list]
    ), "Inverse matrix is not calculated correctly"

    # Test not square matrix
    with pytest.raises(ValueError):
        m = Matrix(generate_random_matrix(random_square_size, random_square_size + 1))
        m.inverse()

    # Test singular matrix
    with pytest.raises(ValueError):
        square_m = Matrix(random_square_matrix)
        square_m[0] = [0 for _ in range(random_square_size)]
        m.inverse()


def test_inverse_matrix():
    """Perform test for `inverse` method of Matrix."""
    [test_inverse_matrix_1_case(size=20) for _ in range(NUM_TESTS)]
