"""Test module for vector.py.

This file test the following functions:
- __mul__, __rmul__
"""

import pytest
from common import generate_random_value, generate_random_values
from vector import Vector


def test_mul():
    """Test the __mul__ method."""
    values = generate_random_values()
    v = Vector(*values)

    random_value = generate_random_value()
    values2 = [x * random_value for x in values]

    v2 = Vector(*values2)

    dot_product = sum(a * b for a, b in zip(values, values2))

    assert v * random_value == v2, "The __mul__ method is not correct"
    assert v * v2 == dot_product, "The __mul__ method is not correct"

    with pytest.raises(TypeError):
        v * "A"

    with pytest.raises(ValueError):
        v * Vector(*values2[1:])


def test_rmul():
    """Test the __rmul__ method."""
    values = generate_random_values()
    v = Vector(*values)

    random_value = generate_random_value()
    values2 = [x * random_value for x in values]

    v2 = Vector(*values2)

    dot_product = sum(a * b for a, b in zip(values, values2))

    assert random_value * v == v2, "The __rmul__ method is not correct"
    assert v2 * v == dot_product, "The __rmul__ method is not correct"

    with pytest.raises(TypeError):
        "A" * v

    with pytest.raises(ValueError):
        Vector(*values2[1:]) * v
