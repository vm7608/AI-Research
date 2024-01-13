"""Test module for vector.py.

This file test the following functions:
- change_basis()
"""

import pytest
from common import generate_basis, generate_random_values
from vector import Vector


def test_change_basis():
    """Test the change_basis method."""
    values = generate_random_values()
    v = Vector(*values)

    # Generate a random basis,
    # vectors[i] will have random_value at index i, and 0 elsewhere
    # This is a valid basis that all vectors is othogonal to each other
    basis_values, random_value_list = generate_basis()
    basic_vectors = [Vector(*x) for x in basis_values]

    # The expected result is the values divided by random_value
    expected_result = [x / y for x, y in zip(values, random_value_list)]

    assert v.change_basis(*basic_vectors) == Vector(
        *expected_result
    ), "The change_basis method is not correct"

    # test invalid type
    with pytest.raises(ValueError):
        v.change_basis(*values)

    # test basis have null vector
    with pytest.raises(ValueError):
        v.change_basis(*basic_vectors + [Vector(*[0] * len(values))])

    # test invalid length
    with pytest.raises(ValueError):
        v.change_basis(*basic_vectors[1:])

    # test invalid basis (not orthogonal)
    with pytest.raises(ValueError):
        v.change_basis(*basic_vectors[:-1])
