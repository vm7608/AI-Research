"""Matrix class and helper functions."""

import math

from vector import Vector

EPS = 1.0e-8


class Matrix:
    """Matrix class."""

    def __init__(self, *args):
        """Initialize a matrix."""
        args = list(*args)

        # Check if the matrix is valid
        for row in args:
            if len(row) != len(args[0]):
                raise ValueError("All rows must have the same length")
            for element in row:
                if not isinstance(element, int | float):
                    raise TypeError("Matrix elements must be numeric")

        # no args => default values [[0, 0], [0, 0]]
        if len(args) == 0:
            self._matrix = [[0, 0], [0, 0]]

        else:
            temp = []
            for row in args:
                temp.append(row.copy())
            self._matrix = temp

    @property
    def matrix(self):
        """Return the matrix."""
        temp = []
        for row in self._matrix:
            temp.append(row.copy())
        return temp

    @matrix.setter
    def matrix(self, matrix):
        """Set the matrix.

        Value of the matrix cannot be set directly.
        We only can set the values of the matrix via the __setitem__ method.
        """
        raise AttributeError("Cannot set matrix directly")

    def size(self, dim=None):
        """Return the size of the matrix.

        If dim is None, return the number of rows and columns.
        If dim is 0, return the number of rows.
        If dim is 1, return the number of columns.
        """
        if dim is None:
            return len(self.matrix), len(self.matrix[0])
        elif dim == 0:
            return len(self.matrix)
        elif dim == 1:
            return len(self.matrix[0])
        else:
            raise IndexError("Dimension must be 0 or 1")

    def __repr__(self):
        """Define the behavior of calling print on an instance of this class."""
        s = ""
        for row in self.matrix:
            s += "\t".join([f"{x} " for x in row])
            s += "\n"
        return s

    def __getitem__(self, index):
        """Return the value at the given index."""
        # Case 1: matrix[int]
        # return 1 row at the given index
        if isinstance(index, int):
            return self.matrix[index].copy()

        # Case 2: matrix[slice] <=> matrix[start:stop:step]
        # return a number of rows from start to stop with step
        elif isinstance(index, slice):
            row_start, row_stop, row_step = index.indices(self.size(dim=0))
            return [self.matrix[i] for i in range(row_start, row_stop, row_step)]

        # Case 3: matrix[int, int] <=> matrix[row, col]
        elif isinstance(index, tuple):
            row, col = index

            # Case 3.1: matrix[int, int] <=> matrix[row, col]
            # -> return 1 element at the given row and col
            if isinstance(row, int) and isinstance(col, int):
                return self.matrix[row][col]

            # Case 3.2: matrix[int, slice] <=> matrix[row, start:stop:step]
            # return 1 row at the given row and slice
            if isinstance(row, int) and isinstance(col, slice):
                col_start, col_stop, col_step = col.indices(self.size(dim=1))
                return [
                    self.matrix[row][i] for i in range(col_start, col_stop, col_step)
                ]

            # Case 3.3: matrix[slice, int] <=> matrix[start:stop:step, col]
            # return 1 col at the given index
            # of a number of rows from start to stop with step
            if isinstance(row, slice) and isinstance(col, int):
                row_start, row_stop, row_step = row.indices(self.size(dim=0))
                return [
                    self.matrix[i][col] for i in range(row_start, row_stop, row_step)
                ]
            # Case 3.4: matrix[slice, slice] <=> matrix[start:stop:step,start:stop:step]
            # -> return a number of rows from start to stop with step
            # then get a number of cols from start to stop with step of each row
            if isinstance(row, slice) and isinstance(col, slice):
                row_start, row_stop, row_step = row.indices(self.size(dim=0))
                col_start, col_stop, col_step = col.indices(self.size(dim=1))
                data = [self.matrix[i] for i in range(row_start, row_stop, row_step)]
                return [
                    [row[j] for j in range(col_start, col_stop, col_step)]
                    for row in data
                ]

        else:
            raise TypeError("Invalid index type")

    def __setitem__(self, index, value):
        """Set the value of the matrix at the given index.

        Only support 2 cases:
        - Assign a list of values to a row
        - Assign a value to an element
        """
        # Case 1: matrix[int]
        # set 1 row at the given index
        if isinstance(index, int):
            if not isinstance(value, list):
                raise TypeError("Invalid value type")
            for element in value:
                if not isinstance(element, int | float):
                    raise TypeError("Matrix elements must be numeric")
            if len(value) != self.size(dim=1):
                raise ValueError("Invalid value size")
            self._matrix[index] = value.copy()

        # Case 2: matrix[int, int] <=> matrix[row, col]
        # set 1 element at the given row and col
        elif isinstance(index, tuple):
            row, col = index
            if isinstance(row, int) and isinstance(col, int):
                if not isinstance(value, int | float):
                    raise TypeError("Matrix elements must be numeric")
                self._matrix[row][col] = value
            else:
                raise TypeError("Not supported that index for setitem")
        else:
            raise TypeError("Not supported that index for setitem")

    def __eq__(self, other):
        """Compare two matrices."""
        if not isinstance(other, Matrix):
            raise TypeError("Can only compare two matrices")
        # Return false if the dimensions are different
        if self.size() != other.size():
            return False
        # Return false if any elements are different
        for i in range(self.size(0)):
            if self[i] != other[i]:
                return False
        return True

    def __ne__(self, other):
        """Compare two matrices for inequality."""
        return not self == other

    def __neg__(self):
        """Return the negative of the matrix."""
        return Matrix([[-x for x in row] for row in self.matrix])

    def is_square(self):
        """Return true if the matrix is square."""
        row, col = self.size()
        return row == col

    def __add__(self, other):
        """Add two matrices that have the same dimensions."""
        if not isinstance(other, Matrix):
            raise TypeError("Can only add two matrices")
        if self.size(dim=0) != other.size(dim=0) or self.size(dim=1) != other.size(
            dim=1
        ):
            raise ValueError("Matrices must be of the same size to add them")
        result = []
        for i in range(self.size(dim=0)):
            row = []
            for j in range(self.size(dim=1)):
                row.append(self.matrix[i][j] + other[i][j])
            result.append(row)
        return Matrix(result)

    def __sub__(self, other):
        """Subtract two matrices that have the same dimensions."""
        if not isinstance(other, Matrix):
            raise TypeError("Can only subtract two matrices")
        if self.size(dim=0) != other.size(dim=0) or self.size(dim=1) != other.size(
            dim=1
        ):
            raise ValueError("Matrices must be of the same size to subtract them")
        result = []
        for i in range(self.size(dim=0)):
            row = []
            for j in range(self.size(dim=1)):
                row.append(self.matrix[i][j] - other[i][j])
            result.append(row)
        return Matrix(result)

    def transpose(self):
        """Return the transpose of the matrix."""
        return Matrix(
            [
                [self.matrix[j][i] for j in range(self.size(dim=0))]
                for i in range(self.size(dim=1))
            ]
        )

    def __mul__(self, other):
        """Multiplie two matrices or a matrix and a number or a vector."""
        if isinstance(other, Matrix):
            if self.size(1) != other.size(0):
                raise ValueError("Matrices must be of compatible size to multiply")
            result = []
            for i in range(self.size(0)):
                row = []
                for j in range(other.size(1)):
                    total = 0
                    for k in range(self.size(1)):
                        total += self.matrix[i][k] * other[k][j]
                    row.append(total)
                result.append(row)
            return Matrix(result)

        elif isinstance(other, int | float):
            result = []
            for i in range(self.size(0)):
                row = []
                for j in range(self.size(1)):
                    row.append(self.matrix[i][j] * other)
                result.append(row)
            return Matrix(result)

        # If Matrix * Vector
        # the column size of the matrix must be equal to the len of the vector
        # and the result is a vector with len equal to the number of rows in the matrix
        elif isinstance(other, Vector):
            if self.size(1) != len(other):
                raise ValueError(
                    "Matrix and vector must be of compatible size to multiply"
                )
            result = []
            for i in range(self.size(0)):
                total = 0
                for j in range(self.size(1)):
                    total += self.matrix[i][j] * other[j]
                result.append(total)
            return Vector(*result)

        else:
            raise TypeError(
                "Matrices can only be multiplied by other matrices, numbers or vectors"
            )

    def __rmul__(self, other):
        """Return reverse multiplication of two matrices or a matrix and a number."""
        return self * other

    def det(self):
        """Calculate the determinant using the Gauss-Jordan elimination method."""
        if not self.is_square():
            raise ValueError("Cannot calculate determinant of non-square matrix")

        copy_matrix = Matrix(self.matrix)

        n = self.size(0)

        det = 1.0

        for i in range(n):
            pivot_row = i
            for j in range(i + 1, n):
                if abs(copy_matrix[j, i]) > abs(copy_matrix[pivot_row, i]):
                    pivot_row = j

            # if the first column is all zeros, the determinant is zero
            if (pivot_row == i) and copy_matrix[i, i] == 0:
                return 0.0

            if pivot_row != i:
                copy_matrix[i], copy_matrix[pivot_row] = (
                    copy_matrix[pivot_row],
                    copy_matrix[i],
                )
                det *= -1.0

            # performe row operations
            for j in range(i + 1, n):
                ratio = copy_matrix[j, i] / copy_matrix[i, i]
                for idx in range(n):
                    copy_matrix[j, idx] -= ratio * copy_matrix[i, idx]

        # calculate determinant
        for i in range(n):
            det *= copy_matrix[i, i]
        return det

    def inverse(self):
        """Compute the inverse of the matrix using Gauss-Jordan elimination method."""
        if not self.is_square():
            raise ValueError("Cannot calculate inverse of non-square matrix")

        if self.det() == 0:
            raise ValueError("Cannot calculate inverse of singular matrix")

        n = self.size(0)

        # Augment the matrix with the identity matrix
        augmented_matrix = [
            row + [1 if i == j else 0 for j in range(n)]
            for i, row in enumerate(self.matrix)
        ]

        # Perform row operations
        for i in range(n):
            pivot_row = i
            for j in range(i + 1, n):
                if abs(augmented_matrix[j][i]) > abs(augmented_matrix[pivot_row][i]):
                    pivot_row = j

            if pivot_row != i:
                augmented_matrix[i], augmented_matrix[pivot_row] = (
                    augmented_matrix[pivot_row],
                    augmented_matrix[i],
                )

            for j in range(n):
                if i != j:
                    ratio = augmented_matrix[j][i] / augmented_matrix[i][i]
                    for k in range(2 * n):
                        augmented_matrix[j][k] -= ratio * augmented_matrix[i][k]

        # Normalize the right side of
        # the augmented matrix to obtain the inverse matrix
        for i in range(n):
            divisor = augmented_matrix[i][i]
            for j in range(2 * n):
                augmented_matrix[i][j] /= divisor

        # Inverse matrix is in the right side
        # of the augmented matrix
        inverse_matrix = [row[n:] for row in augmented_matrix]
        return Matrix(inverse_matrix)

    def gram_schmidt(self):
        """Perform Gram-Schmidt orthogonalization on the matrix."""
        almost_zero = EPS
        copy_matrix = self.matrix

        # Loop over all vectors and subtract the overlap with previous vectors.
        for i in range(self.size(1)):
            for j in range(i):
                dot_product = 0
                for k in range(self.size(0)):
                    dot_product += copy_matrix[k][i] * copy_matrix[k][j]
                for k in range(self.size(0)):
                    copy_matrix[k][i] -= dot_product * copy_matrix[k][j]

            # Normalize the vector
            norm = 0
            for k in range(self.size(0)):
                norm += copy_matrix[k][i] ** 2
            norm = math.sqrt(norm)

            if norm > almost_zero:
                for k in range(self.size(0)):
                    copy_matrix[k][i] /= norm
            else:
                for k in range(self.size(0)):
                    copy_matrix[k][i] = 0

        # Finally, we return the result:
        return Matrix(copy_matrix)
