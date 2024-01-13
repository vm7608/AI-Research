"""Test cases for the Matrix class."""
import pytest
from common import (
    generate_random_index,
    generate_random_matrix,
    generate_random_value,
    generate_random_values,
)
from matrix import Matrix
from vector import Vector


def test_matrix_mul():
    """Test the __mul__ method."""
    # Test multiplication of a matrix by a matrix
    random_matrix1 = generate_random_matrix()
    # random_matrix2 have number of rows equal to number of columns of random_matrix
    # and vice versa
    random_matrix2 = generate_random_matrix(len(random_matrix1[0]), len(random_matrix1))
    matrix_mul_matrix_rs = [
        [0 for _ in range(len(random_matrix2[0]))] for _ in range(len(random_matrix1))
    ]
    for i in range(len(random_matrix1)):
        for j in range(len(random_matrix2[0])):
            for k in range(len(random_matrix2)):
                matrix_mul_matrix_rs[i][j] += (
                    random_matrix1[i][k] * random_matrix2[k][j]
                )
    assert Matrix(random_matrix1) * Matrix(random_matrix2) == Matrix(
        matrix_mul_matrix_rs
    ), "The multiplication is not correct"

    # Test multiplication of a matrix by a scalar
    random_scalar = generate_random_value()
    matrix_mul_scalar_rs = [
        [random_scalar * random_matrix1[i][j] for j in range(len(random_matrix1[0]))]
        for i in range(len(random_matrix1))
    ]
    assert Matrix(random_matrix1) * random_scalar == Matrix(
        matrix_mul_scalar_rs
    ), "The multiplication is not correct"

    # Test multiplication of a matrix by a vector
    random_vector = generate_random_values(length=len(random_matrix1[0]))
    matrix_mul_vector_rs = [0 for _ in range(len(random_matrix1))]
    for i in range(len(random_matrix1)):
        for j in range(len(random_matrix1[0])):
            matrix_mul_vector_rs[i] += random_matrix1[i][j] * random_vector[j]

    assert Matrix(random_matrix1) * Vector(*random_vector) == Vector(
        *matrix_mul_vector_rs
    ), "The multiplication is not correct"

    # Test invalid types
    with pytest.raises(TypeError):
        Matrix(random_matrix1) * "A"

    # Test invalid matrix size
    with pytest.raises(ValueError):
        Matrix(random_matrix1) * Matrix(random_matrix2[1:])

    # Test invalid vector size
    with pytest.raises(ValueError):
        Matrix(random_matrix1) * Vector(*random_vector[1:])


def test_rmul():
    """Test the __rmul__ method."""
    # Test scalar mul matrix
    random_matrix1 = generate_random_matrix()
    random_scalar = generate_random_value()
    matrix_mul_scalar_rs = [
        [random_scalar * random_matrix1[i][j] for j in range(len(random_matrix1[0]))]
        for i in range(len(random_matrix1))
    ]
    assert random_scalar * Matrix(random_matrix1) == Matrix(
        matrix_mul_scalar_rs
    ), "The multiplication is not correct"

    # Test invalid rmul
    with pytest.raises(TypeError):
        "A" * Matrix(random_matrix1)
