"""Test cases for the Matrix class."""
import numpy as np
import numpy.linalg as la
import pytest
from common import (
    generate_random_matrix,
    generate_random_matrix_size,
)
from matrix import Matrix

EPS = 1e-8
NUM_TESTS = 100


def gram_schmidt_coursera(matrix):
    """Gram-Schmidt process.

    This function from Coursera's Machine Learning course.
    We use it to test our Gram-Schmidt process.
    """
    almost_zero = EPS
    copy_matrix = np.array(matrix, dtype=np.float_)

    # Loop over all vectors and subtract the overlap with previous vectors.
    for i in range(copy_matrix.shape[1]):
        for j in range(i):
            copy_matrix[:, i] = (
                copy_matrix[:, i]
                - copy_matrix[:, i] @ copy_matrix[:, j] * copy_matrix[:, j]
            )

        # Normalize the vector
        if la.norm(copy_matrix[:, i]) > almost_zero:
            copy_matrix[:, i] = copy_matrix[:, i] / la.norm(copy_matrix[:, i])
        else:
            copy_matrix[:, i] = np.zeros_like(copy_matrix[:, i])

    # Finally, we return the result:
    return copy_matrix


def test_gram_schmidt_1_case(size=10):
    """Test the Gram - Schmidt process."""
    row_size, col_size = generate_random_matrix_size(length=size)
    random_matrix = generate_random_matrix(row_size=row_size, col_size=col_size)
    # print(random_matrix)
    m = Matrix(random_matrix)
    rs = m.gram_schmidt()
    calculated_matrix = rs.matrix
    expected_matrix = gram_schmidt_coursera(random_matrix)
    error_ratio_list = []
    for i in range(m.size(0)):
        for j in range(m.size(1)):
            if expected_matrix[i][j] == 0:
                error_ratio = abs(calculated_matrix[i][j])
            else:
                error_ratio = abs(
                    calculated_matrix[i][j] - expected_matrix[i][j]
                ) / abs(expected_matrix[i][j])

            error_ratio_list.append(error_ratio)

    accepted_error_ratio = EPS
    test_rs = all(
        [error_ratio < accepted_error_ratio for error_ratio in error_ratio_list]
    )
    assert test_rs, "Gram-Schmidt process do not work well"


def test_gram_schmidt():
    """Perform the Gram - Schmidt process test NUM_TESTS times."""
    [test_gram_schmidt_1_case(size=50) for _ in range(NUM_TESTS)]
