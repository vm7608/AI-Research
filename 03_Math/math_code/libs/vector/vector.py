"""Vector class.

This module implements a Vector class with basic operations.
"""
import itertools
import math


def check_orthogonal(vectors) -> bool:
    """Check if a list of vectors is orthogonal."""
    # loop over all pairs of vectors and check if they are orthogonal,
    # return False if not
    for vec1, vec2 in itertools.combinations(vectors, 2):
        if vec1 * vec2 != 0:
            return False
    return True


class Vector:
    """Vector class."""

    def __init__(self, *args):
        """Initialize a vector with its coordinates.

        If no coordinates are given, the vector is initialized to (0, 0).
        """
        for arg in args:
            if not isinstance(arg, int | float):
                raise TypeError("Coordinates must be numeric")

        if len(args) == 0:
            # no args => default values [0, 0]
            self._values = [0, 0]

        else:
            # args is a tuple
            # convert to list because tuples are immutable
            args = list(args)
            self._values = args

    @property
    def values(self):
        """Return the coordinates of the vector."""
        return self._values.copy()

    @values.setter
    def values(self, values):
        """Set the coordinates of the vector.

        values of the vector cannot be set directly.
        We only can set the values of the vector via the __setitem__ method.
        """
        raise AttributeError("Cannot set values directly")

    def __repr__(self) -> str:
        """Define the behavior of calling print on an instance of this class."""
        return "<" + str(self.values)[1:-1] + ">"

    def __getitem__(self, key):
        """Return keyth coordinate of the vector (v[key])."""
        return self.values[key]

    def __setitem__(self, key, value):
        """Set keyth coordinate of the vector (v[key] = value)."""
        if not isinstance(value, int | float):
            raise TypeError("Coordinates must be numeric")
        self._values[key] = value

    def __len__(self) -> int:
        """Return the dimension of the vector len()."""
        return len(self.values)

    def __eq__(self, other: "Vector") -> bool:
        """Compare coordinates of two vectors (==)."""
        if not isinstance(other, Vector):
            raise TypeError("Cannot compare with non-vector type")

        if len(self) != len(other):
            return False

        return all(a == b for a, b in zip(self.values, other.values))

    def __ne__(self, other: "Vector") -> bool:
        """Compare coordinates of two vectors (!=)."""
        if not isinstance(other, Vector):
            raise TypeError("Cannot compare with non-vector type")
        return not self == other

    def __neg__(self) -> "Vector":
        """Return the negative of the vector (-v)."""
        return Vector(*[-x for x in self.values])

    def __abs__(self) -> float:
        """Return the magnitude of the vector (abs(v))."""
        return math.sqrt(sum(x * x for x in self.values))

    def __add__(self, other) -> "Vector":
        """Return the sum of two vectors (v + w)."""
        if not isinstance(other, Vector):
            raise TypeError(f"Addition with type {type(other)} not supported")

        if len(self) != len(other):
            raise ValueError("Vector dimensions must agree")

        result = [a + b for a, b in zip(self.values, other.values)]
        return Vector(*result)

    def __radd__(self, other) -> "Vector":
        """Return reversed addition of vectors (other + v)."""
        return self.__add__(other)

    def __sub__(self, other) -> "Vector":
        """Return the subtraction of two vectors (v - w)."""
        if not isinstance(other, Vector):
            raise TypeError(f"Subtraction with type {type(other)} not supported")

        if len(self) != len(other):
            raise ValueError("Vector dimensions must agree")

        result = [a - b for a, b in zip(self.values, other.values)]
        return Vector(*result)

    def __rsub__(self, other) -> "Vector":
        """Return reversed subtraction of vectors (other + v)."""
        return self.__sub__(other)

    def _dot(self, other: "Vector") -> float:
        """Return the dot product of two vectors.

        This method is used internally by the __mul__ method.
        """
        return sum(a * b for a, b in zip(self.values, other.values))

    def __mul__(self, factor):
        """Return the product of a vector and a scalar (v * x) (result is a vector).

        Or the dot product of two vectors (v * w) (result is a scalar).
        """
        if isinstance(factor, int | float):
            result = [a * factor for a in self.values]
            return Vector(*result)
        elif isinstance(factor, Vector):
            if len(self) != len(factor):
                raise ValueError("Dimensions must agree")
            return self._dot(factor)
        else:
            raise TypeError("Factor must be int, float or Vector")

    def __rmul__(self, factor):
        """Call if 10 * self for instance."""
        return self.__mul__(factor)

    def get_cosine(self, other: "Vector") -> float:
        """Return the cosine of the angle between two vectors."""
        if not isinstance(other, Vector):
            raise TypeError("The cosine requires another vector")

        if abs(self) == 0 or abs(other) == 0:
            raise ValueError("Cannot compute the cosine with a null vector")

        if len(self) != len(other):
            raise ValueError("Vector dimensions must agree")

        return (self * other) / (abs(self) * abs(other))

    def scalar_project_on(self, other: "Vector") -> float:
        """Return the scalar projection of a vector on another vector."""
        if not isinstance(other, Vector):
            raise TypeError("The cosine requires another vector")

        if abs(self) == 0 or abs(other) == 0:
            raise ValueError("Cannot compute the cosine with a null vector")

        if len(self) != len(other):
            raise ValueError("Vector dimensions must agree")

        return (self * other) / abs(other)

    def change_basis(self, *args):
        """Return the coordinates of the vector in the basis of set of input vectors."""
        vectors = list(args)

        for vector in vectors:
            if not isinstance(vector, Vector):
                raise ValueError("The basis vectors must be vectors")
            if abs(vector) == 0:
                raise ValueError("The basis vectors must not be null")

        if len(vectors) != len(self):
            raise ValueError(
                "The number of basis vectors must be equal to dimension of the vector"
            )

        if check_orthogonal(vectors) is False:
            raise ValueError("The basis vectors must be orthogonal")

        rs = []
        for vector in vectors:
            rs.append(self.scalar_project_on(vector) / abs(vector))

        return Vector(*rs)
