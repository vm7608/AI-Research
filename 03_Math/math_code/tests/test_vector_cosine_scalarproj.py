"""Test module for vector.py.

This file test the following functions:
- get_cosine()
- scalar_project_on()
"""
import math

import pytest
from common import generate_random_value, generate_random_values
from vector import Vector


def test_cosine():
    """Test the get_cosine method."""
    values = generate_random_values()
    v = Vector(*values)

    values2 = generate_random_values()
    v2 = Vector(*values2)

    dot_product = sum(a * b for a, b in zip(values, values2))
    norm1 = math.sqrt(sum(x**2 for x in values))
    norm2 = math.sqrt(sum(x**2 for x in values2))

    assert v.get_cosine(v2) == dot_product / (
        norm1 * norm2
    ), "The get_cosine method is not correct"

    # Test invalid input
    with pytest.raises(TypeError):
        v.get_cosine("A")

    # Test vector with different dimensions
    with pytest.raises(ValueError):
        v.get_cosine(Vector(*values2[1:]))

    # Test zero vector
    with pytest.raises(ValueError):
        v.get_cosine(Vector())

    # Test zero vector
    with pytest.raises(ValueError):
        Vector().get_cosine(v)


def test_scalar_project():
    """Test the scalar_project_on method."""
    values = generate_random_values()
    v = Vector(*values)

    values2 = generate_random_values()
    v2 = Vector(*values2)

    dot_product = sum(a * b for a, b in zip(values, values2))
    norm2 = math.sqrt(sum(x**2 for x in values2))

    assert (
        v.scalar_project_on(v2) == dot_product / norm2
    ), "The scalar_project_on method is not correct"

    # Test invalid input
    with pytest.raises(TypeError):
        v.scalar_project_on("A")

    # Test vector with different dimensions
    with pytest.raises(ValueError):
        v.scalar_project_on(Vector(*values2[1:]))

    # Test zero vector
    with pytest.raises(ValueError):
        v.scalar_project_on(Vector())

    # Test zero vector
    with pytest.raises(ValueError):
        Vector().scalar_project_on(v)
