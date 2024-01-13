"""Test cases for the Matrix class."""
import pytest
from common import (
    generate_random_index,
    generate_random_matrix,
    generate_random_value,
    generate_random_values,
)
from matrix import Matrix


def test_eq_ne():
    """Test the __eq__ and __ne__ method."""
    random_matrix = generate_random_matrix()
    m1 = Matrix(random_matrix)
    m2 = Matrix(random_matrix)

    # Test the same matrix
    assert m1 == m2, "The two matrices should be equal"

    random_matrix2 = generate_random_matrix()
    # Make sure the two matrices are not equal
    random_matrix2 = random_matrix2 * 2
    m2 = Matrix(random_matrix2)

    # Test different elements
    assert m1 != m2, "The two matrices should not be equal"

    # Test different dimensions
    m3 = Matrix(random_matrix2[1:])
    assert m1 != m3, "The two matrices should not be equal"

    # Test different types
    with pytest.raises(TypeError):
        m1 == 1


def test_neg():
    """Test the __neg__ method."""
    random_matrix = generate_random_matrix()
    m = Matrix(random_matrix)

    random_matrix_neg = [[-i for i in row] for row in random_matrix]
    m_neg = Matrix(random_matrix_neg)

    assert -m == m_neg, "The negation is not correct"


def test_is_square():
    """Test the is_square method."""
    random_matrix = generate_random_matrix()
    m = Matrix(random_matrix)

    assert m.is_square() == (
        len(random_matrix) == len(random_matrix[0])
    ), "The is_square method is not correct"


def test_add():
    """Test the __add__ method."""
    random_matrix1 = generate_random_matrix()
    random_matrix2 = generate_random_matrix(len(random_matrix1), len(random_matrix1[0]))

    m1 = Matrix(random_matrix1)
    m2 = Matrix(random_matrix2)

    random_matrix_add = [
        [
            random_matrix1[i][j] + random_matrix2[i][j]
            for j in range(len(random_matrix1[0]))
        ]
        for i in range(len(random_matrix1))
    ]
    m_add = Matrix(random_matrix_add)

    assert m1 + m2 == m_add, "The addition is not correct"

    # Test different dimensions
    m3 = Matrix(random_matrix2[1:])
    with pytest.raises(ValueError):
        m1 + m3

    # Test different types
    with pytest.raises(TypeError):
        m1 + 1


def test_sub():
    """Test the __sub__ method."""
    random_matrix1 = generate_random_matrix()
    random_matrix2 = generate_random_matrix(len(random_matrix1), len(random_matrix1[0]))

    m1 = Matrix(random_matrix1)
    m2 = Matrix(random_matrix2)

    random_matrix_sub = [
        [
            random_matrix1[i][j] - random_matrix2[i][j]
            for j in range(len(random_matrix1[0]))
        ]
        for i in range(len(random_matrix1))
    ]
    m_sub = Matrix(random_matrix_sub)

    assert m1 - m2 == m_sub, "The subtraction is not correct"

    # Test different dimensions
    m3 = Matrix(random_matrix2[1:])
    with pytest.raises(ValueError):
        m1 - m3

    # Test different types
    with pytest.raises(TypeError):
        m1 - 1


def test_transpose():
    """Test the transpose method."""
    random_matrix = generate_random_matrix()
    m = Matrix(random_matrix)

    random_matrix_transpose = [
        [random_matrix[j][i] for j in range(len(random_matrix))]
        for i in range(len(random_matrix[0]))
    ]

    assert m.transpose() == Matrix(
        random_matrix_transpose
    ), "The transpose is not correct"
