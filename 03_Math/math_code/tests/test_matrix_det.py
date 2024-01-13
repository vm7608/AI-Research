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


def test_det_1_case(size=10):
    """Test the det method.

    Only test for square matrix that has size less than 100
    We use numpy to calculate the expected value,
    and compare it with our implementation by calculating the error ratio
    If the error ratio is less than 1e-8, we consider it as correct
    """
    random_square_size, _ = generate_random_matrix_size(length=size)
    random_square_matrix = generate_random_matrix(
        random_square_size, random_square_size
    )

    square_m = Matrix(random_square_matrix)
    numpy_square_m = np.array(random_square_matrix)

    calculated_det = square_m.det()
    expected_det = np.linalg.det(numpy_square_m)

    if np.linalg.det(numpy_square_m) == 0:
        error_ratio = abs(calculated_det)
    else:
        error_ratio = abs(calculated_det - expected_det) / abs(expected_det)

    acceptable_err_ratio = EPS
    assert error_ratio < acceptable_err_ratio, "Determinant is not calculated correctly"

    # Test not square matrix
    with pytest.raises(ValueError):
        m = Matrix(generate_random_matrix(random_square_size, random_square_size * 2))
        m.det()


def test_det():
    """Perform test for `det` method of Matrix."""
    [test_det_1_case(size=20) for _ in range(NUM_TESTS)]
