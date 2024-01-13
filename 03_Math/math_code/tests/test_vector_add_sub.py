"""Test module for vector.py.

This file test the following functions:
- __len__
- __eq__, __ne__
- __neg__
- __abs__
- __add__, __radd__
- __sub__, __rsub__
"""

import math

import pytest
from common import generate_random_values
from vector import Vector


def test_len():
    """Test __len__ method of vector."""
    values = generate_random_values()
    v = Vector(*values)
    assert len(v) == len(values), "The length of the vector is not correct"


def test_equal():
    """Test __eq__ method of vector."""
    values = generate_random_values()
    v = Vector(*values)
    v2 = Vector(*values)

    values2 = values.copy() * 2
    v3 = Vector(*values2)

    values4 = values + values2
    v4 = Vector(*values4)

    assert v == v2, "The __eq__ method is not correct"
    assert not (v == v3), "The __eq__ method is not correct"
    assert not (v == v4), "The __eq__ method is not correct"

    # Test invalid comparison
    with pytest.raises(TypeError):
        v == "A"


def test_not_equal():
    """Test __ne__ method of vector."""
    values = generate_random_values()
    v = Vector(*values)
    v2 = Vector(*values)

    values2 = values.copy() * 2
    v3 = Vector(*values2)

    values4 = values + values2
    v4 = Vector(*values4)

    assert not (v != v2), "The __ne__ method is not correct"
    assert v != v3, "The __ne__ method is not correct"
    assert v != v4, "The __ne__ method is not correct"

    # Test invalid comparison
    with pytest.raises(TypeError):
        v != "A"


def test_negetive():
    """Test __neg__ method of vector."""
    values = generate_random_values()
    v = Vector(*values)
    v2 = Vector(*[-x for x in values])

    assert -v == v2, "The __neg__ method is not correct"


def test_abs():
    """Test __abs__ method of vector."""
    values = generate_random_values()
    v = Vector(*values)
    assert abs(v) == math.sqrt(
        sum(x * x for x in values)
    ), "The __abs__ method is not correct"

    # Test abs of default vector
    v2 = Vector()
    assert abs(v2) == 0, "The __abs__ method is not correct"


def test_add():
    """Test __add__ method of vector."""
    values = generate_random_values()
    v = Vector(*values)
    values2 = generate_random_values()
    v2 = Vector(*values2)

    v3 = Vector(*[x + y for x, y in zip(values, values2)])

    v4 = Vector(*(values + values2))

    assert v + v2 == v3, "The __add__ method is not correct"

    # Test invalid addition
    with pytest.raises(TypeError):
        v + "A"

    # Test addition with different length
    with pytest.raises(ValueError):
        v + v4


def test_radd():
    """Test __radd__ method of vector."""
    values = generate_random_values()
    v = Vector(*values)
    values2 = generate_random_values()
    v2 = Vector(*values2)

    v3 = Vector(*[x + y for x, y in zip(values, values2)])

    v4 = Vector(*(values + values2))

    assert v2 + v == v3, "The __radd__ method is not correct"

    # Test invalid addition
    with pytest.raises(TypeError):
        "A" + v

    # Test addition with different length
    with pytest.raises(ValueError):
        v4 + v


def test_sub():
    """Test __sub__ method of vector."""
    values = generate_random_values()
    v = Vector(*values)
    values2 = generate_random_values()
    v2 = Vector(*values2)

    v3 = Vector(*[x - y for x, y in zip(values, values2)])

    v4 = Vector(*(values + values2))

    assert v - v2 == v3, "The __sub__ method is not correct"

    # Test invalid subtraction
    with pytest.raises(TypeError):
        v - "A"

    # Test subtraction with different length
    with pytest.raises(ValueError):
        v - v4


def test_rsub():
    """Test __rsub__ method of vector."""
    values = generate_random_values()
    v = Vector(*values)
    values2 = generate_random_values()
    v2 = Vector(*values2)

    v3 = Vector(*[y - x for x, y in zip(values, values2)])

    v4 = Vector(*(values + values2))

    assert v2 - v == v3, "The __rsub__ method is not correct"

    # Test invalid subtraction
    with pytest.raises(TypeError):
        "A" - v

    # Test subtraction with different length
    with pytest.raises(ValueError):
        v4 - v
