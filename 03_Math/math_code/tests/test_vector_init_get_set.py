"""Test module for vector.py.

This file test the following functions:
- __init__
- getter/setter
- __getitem__, __setitem__
"""

import pytest
from common import generate_random_index, generate_random_value, generate_random_values
from vector import Vector


def test_intialization():
    """Test the initialization of a vector."""
    # Test valid initialization
    values = generate_random_values()
    v = Vector(*values)

    assert v.values == values, "The vector is not initialized correctly"
    assert (
        v.values is not values
    ), "The values of vector should be a copy of initialization arguments"

    # Test default initialization
    assert Vector().values == [0, 0], "The default vector is not initialized correctly"

    # Test invalid initialization
    with pytest.raises(TypeError):
        Vector(*values, "A")

    # Test argument change after initialization does not change the vector
    values2 = generate_random_values()
    values_temp = values2.copy()
    v2 = Vector(*values2)
    random_index = generate_random_index()
    random_value = generate_random_value()
    values2[random_index] = random_value
    assert (
        v2.values == values_temp
    ), "The vector should not change when initialization arguments change"


def test_getter():
    """Test the values getter."""
    values = generate_random_values()
    v = Vector(*values)
    assert v.values == values, "The getter does not return the correct values"

    # Test that the getter returns a copy of the values
    random_index = generate_random_index()
    random_value = generate_random_value()
    v.values[random_index] = random_value
    assert v.values == values, "The getter should return a copy of the values"

    values_temp = v.values
    values_temp[random_index] = random_value
    assert v.values == values, "The getter should return a copy of the values"


def test_setter():
    """Test that the coordinates of a vector cannot be set directly."""
    values = generate_random_values()
    v = Vector(*values)
    with pytest.raises(AttributeError):
        v.values = generate_random_values()


def test_setitem():
    """Test set value of a vector via the __setitem__ method."""
    values = generate_random_values()
    v = Vector(*values)
    random_index = generate_random_index()
    random_value = generate_random_value()

    v[random_index] = random_value
    assert v[random_index] == random_value, "__setitem__ does not set the correct value"

    # Test setting non-numeric value
    with pytest.raises(TypeError):
        v[random_index] = "A"

    # Test out of range index
    with pytest.raises(IndexError):
        v[len(values)] = random_value


def test_getitem():
    """Test the getting of a coordinate of a vector via the __getitem__ method."""
    values = generate_random_values()
    v = Vector(*values)

    random_index = generate_random_index()

    assert v[:] == values, "__getitem__ does not return the correct value"
    assert (
        v[random_index] == values[random_index]
    ), "__getitem__ does not return the correct value"

    # Test out of range index
    with pytest.raises(IndexError):
        v[len(values)] = generate_random_value()
